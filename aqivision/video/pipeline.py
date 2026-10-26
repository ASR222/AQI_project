"""Clip -> JSON: quality gate + temporally aggregated AQI class + tracked objects.

    frames (OpenCV) ──┬─ sample ~2 fps ─> quality gate ─> 3-stream CNN per frame ─> sliding-window aggregate
                      └─ every frame ───> YOLO11n + ByteTrack ─> unique objects, peak counts, heavy-vehicle share

The two branches are independent: detections never change the AQI estimate, and either branch
can be switched off (`classifier=None` / `tracker=None`).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import quality, temporal
from .frames import sample, video_info

Classifier = Callable[[list[np.ndarray]], np.ndarray]  # RGB frames -> (N, C) probabilities

MIN_GOOD_FRAMES = 3


def analyse(
    video: str | Path,
    classifier: Classifier | None = None,
    classes: list[str] | None = None,
    tracker=None,
    sample_every_s: float = 0.5,
    window: int = 8,
    stride: int = 4,
    track_stride: int = 1,
    annotated_out: str | Path | None = None,
    thresholds: quality.Thresholds = quality.Thresholds(),
) -> dict:
    t0 = time.perf_counter()
    info = video_info(video)
    out: dict = {"video": {**info, "duration_s": round(info["frames"] / info["fps"], 2) if info["fps"] else None}}

    # --- AQI branch -------------------------------------------------------------------------
    frames = sample(video, every_s=sample_every_s)
    q = [quality.check(f.bgr, thresholds) for f in frames]
    good = [f for f, fq in zip(frames, q) if fq.ok]
    reasons = sorted({r for fq in q for r in fq.reject}, key=quality.REASONS.index)
    out["quality"] = {
        "sampled": len(frames),
        "passed": len(good),
        "reject_reasons": reasons,
        "per_frame": [{"t": round(f.t, 2), "luma": fq.luma, "sharpness": fq.sharpness, "sky": fq.sky, "reject": fq.reject}
                      for f, fq in zip(frames, q)],
    }

    if classifier is None:
        out["aqi"] = None
    elif len(good) < MIN_GOOD_FRAMES:
        out["aqi"] = None
        out.setdefault("caveats", []).append(
            f"Only {len(good)} usable frames (need {MIN_GOOD_FRAMES}); point the camera at the sky outdoors in daylight."
        )
    else:
        t1 = time.perf_counter()
        probs = classifier([cv2.cvtColor(f.bgr, cv2.COLOR_BGR2RGB) for f in good])
        agg = temporal.aggregate(probs, classes or [str(i) for i in range(probs.shape[1])], window, stride)
        agg["frame_times"] = [round(f.t, 2) for f in good]
        agg["seconds"] = round(time.perf_counter() - t1, 3)
        out["aqi"] = agg
        if agg["stability"] < 0.6:
            out.setdefault("caveats", []).append("Frame-level predictions disagree across the clip; treat the AQI class as low confidence.")

    # --- Detection / tracking branch --------------------------------------------------------
    if tracker is None:
        out["objects"] = None
    else:
        res = tracker.run(video, stride=track_stride, annotated_out=annotated_out)
        out["objects"] = res.summary()
        out["objects"]["tracks"] = [
            {"id": t.track_id, "label": t.label, "first_t": t.first_t, "last_t": t.last_t,
             "dwell_s": t.dwell_s, "frames": t.frames, "mean_conf": t.mean_conf}
            for t in res.tracks
        ]

    out["seconds_total"] = round(time.perf_counter() - t0, 3)
    return out
