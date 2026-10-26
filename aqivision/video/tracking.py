"""Object detection + multi-object tracking with Ultralytics YOLO and ByteTrack.

Why: the AQI number is a property of the scene, but the *causes* you can see on camera
(traffic density, heavy vehicles, people exposed outdoors) are objects. Counting unique tracked
objects instead of per-frame boxes avoids counting the same truck 30 times.

Model: YOLO11n pretrained on COCO (Ultralytics, AGPL-3.0). Weights are downloaded by
Ultralytics on first use (or pass a local path); they are not committed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .frames import iter_frames, video_info

# COCO ids we care about for outdoor air-quality context.
VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
PEOPLE = {0: "person"}
HEAVY = {"bus", "truck"}
DEFAULT_CLASSES = {**PEOPLE, **VEHICLES}


@dataclass
class TrackSummary:
    track_id: int
    label: str
    first_t: float
    last_t: float
    frames: int
    mean_conf: float

    @property
    def dwell_s(self) -> float:
        return round(self.last_t - self.first_t, 3)


@dataclass
class TrackingResult:
    tracks: list[TrackSummary] = field(default_factory=list)
    per_frame_counts: list[dict[str, int]] = field(default_factory=list)
    frames_processed: int = 0
    seconds: float = 0.0

    def unique_counts(self, min_frames: int = 3) -> dict[str, int]:
        """Unique objects per class; tracks seen in fewer than `min_frames` frames are ignored (flicker)."""
        out: dict[str, int] = defaultdict(int)
        for tr in self.tracks:
            if tr.frames >= min_frames:
                out[tr.label] += 1
        return dict(out)

    def summary(self, min_frames: int = 3) -> dict:
        uniq = self.unique_counts(min_frames)
        peak = {}
        for counts in self.per_frame_counts:
            for k, v in counts.items():
                peak[k] = max(peak.get(k, 0), v)
        n_vehicles = sum(v for k, v in uniq.items() if k in VEHICLES.values())
        n_heavy = sum(v for k, v in uniq.items() if k in HEAVY)
        return {
            "unique_objects": uniq,
            "peak_in_frame": peak,
            "vehicles": n_vehicles,
            "heavy_vehicle_share": round(n_heavy / n_vehicles, 3) if n_vehicles else None,
            "frames_processed": self.frames_processed,
            "processing_fps": round(self.frames_processed / self.seconds, 2) if self.seconds else None,
            "min_track_frames": min_frames,
        }


class Tracker:
    def __init__(
        self,
        weights: str = "yolo11n.pt",
        classes: dict[int, str] | None = None,
        conf: float = 0.35,
        iou: float = 0.5,
        imgsz: int = 640,
        tracker: str = "bytetrack.yaml",
        device: str | None = None,
    ):
        from ultralytics import YOLO  # lazy

        self.model = YOLO(weights)
        self.classes = classes or DEFAULT_CLASSES
        self.conf, self.iou, self.imgsz, self.tracker, self.device = conf, iou, imgsz, tracker, device

    def run(
        self,
        video: str | Path,
        stride: int = 1,
        max_frames: int | None = None,
        annotated_out: str | Path | None = None,
    ) -> TrackingResult:
        import time

        info = video_info(video)
        writer = None
        if annotated_out:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(annotated_out), fourcc, info["fps"] / stride, (info["width"], info["height"]))

        stats: dict[int, dict] = {}
        result = TrackingResult()
        t0 = time.perf_counter()
        try:
            for fr in iter_frames(video, stride=stride, max_frames=max_frames):
                r = self.model.track(
                    fr.bgr,
                    persist=True,  # keep tracker state across calls
                    tracker=self.tracker,
                    classes=list(self.classes),
                    conf=self.conf,
                    iou=self.iou,
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False,
                )[0]
                counts: dict[str, int] = defaultdict(int)
                boxes = r.boxes
                if boxes is not None and len(boxes):
                    cls = boxes.cls.int().tolist()
                    confs = boxes.conf.tolist()
                    ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(cls)
                    for c, s, tid in zip(cls, confs, ids):
                        label = self.classes.get(c, str(c))
                        counts[label] += 1
                        if tid is None:
                            continue
                        st = stats.setdefault(tid, {"label": label, "first": fr.t, "last": fr.t, "n": 0, "conf": 0.0})
                        st["last"] = fr.t
                        st["n"] += 1
                        st["conf"] += s
                result.per_frame_counts.append(dict(counts))
                result.frames_processed += 1
                if writer is not None:
                    writer.write(r.plot())
        finally:
            result.seconds = time.perf_counter() - t0
            if writer is not None:
                writer.release()
            # reset tracker state so the next video starts fresh
            if getattr(self.model, "predictor", None) is not None:
                self.model.predictor = None

        result.tracks = [
            TrackSummary(tid, s["label"], round(s["first"], 3), round(s["last"], 3), s["n"], round(s["conf"] / s["n"], 4))
            for tid, s in sorted(stats.items())
        ]
        return result


_worker_tracker: Tracker | None = None


def _init_worker(kwargs: dict) -> None:
    global _worker_tracker
    _worker_tracker = Tracker(**kwargs)


def _run_in_worker(video: str, stride: int, max_frames: int | None, annotated_out: str | None) -> TrackingResult:
    assert _worker_tracker is not None
    return _worker_tracker.run(video, stride=stride, max_frames=max_frames, annotated_out=annotated_out)


class IsolatedTracker:
    """Same interface as Tracker, but YOLO/PyTorch live in a separate (spawned) worker process.

    Loading TensorFlow (the phase-1 AQI model) and PyTorch (Ultralytics) in one Python process
    segfaults with some wheel combinations (seen with tensorflow-cpu 2.21 + torch 2.14, in either
    import order). Use this whenever the AQI classifier runs in the same process, e.g. the API.
    The model is loaded once in the worker and reused across calls.
    """

    def __init__(self, **tracker_kwargs):
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        self._pool = ProcessPoolExecutor(
            max_workers=1,
            mp_context=mp.get_context("spawn"),  # fresh interpreter: never inherits TensorFlow
            initializer=_init_worker,
            initargs=(tracker_kwargs,),
        )

    def run(self, video, stride: int = 1, max_frames: int | None = None, annotated_out=None) -> TrackingResult:
        fut = self._pool.submit(_run_in_worker, str(video), stride, max_frames,
                                str(annotated_out) if annotated_out else None)
        return fut.result()

    def close(self) -> None:
        self._pool.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def counts_to_array(per_frame_counts: list[dict[str, int]], labels: list[str]) -> np.ndarray:
    """Per-frame counts -> (N, len(labels)) int array (for plots / time series)."""
    return np.array([[c.get(k, 0) for k in labels] for c in per_frame_counts], dtype=int)
