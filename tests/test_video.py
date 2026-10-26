import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from aqivision.video import frames, pipeline, quality, temporal

CLASSES = ["A", "B", "C"]


def make_clip(path: Path, n=60, fps=30, size=(320, 240), kind="sky"):
    w, h = size
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    rng = np.random.default_rng(0)
    for i in range(n):
        if kind == "sky":
            img = np.zeros((h, w, 3), np.uint8)
            img[: h // 2] = (235, 206, 135)  # BGR light blue sky
            img[h // 2 :] = rng.integers(0, 255, (h - h // 2, w, 3), dtype=np.uint8)  # textured ground
            cv2.rectangle(img, (10 + i, h // 2 - 40), (60 + i, h // 2), (40, 40, 40), -1)  # a moving block
        else:  # dark
            img = np.full((h, w, 3), 5, np.uint8)
        vw.write(img)
    vw.release()
    return path


def test_sample_spacing(tmp_path):
    clip = make_clip(tmp_path / "c.mp4", n=60, fps=30)
    fr = frames.sample(clip, every_s=0.5)
    assert len(fr) == 4  # t = 0, 0.5, 1.0, 1.5 in a 2 s clip
    ts = [f.t for f in fr]
    assert all(b - a >= 0.49 for a, b in zip(ts, ts[1:]))


def test_iter_frames_stride(tmp_path):
    clip = make_clip(tmp_path / "c.mp4", n=30)
    assert len(list(frames.iter_frames(clip, stride=3))) == 10
    assert len(list(frames.iter_frames(clip, max_frames=7))) == 7


def test_quality_gate_sky_vs_dark(tmp_path):
    sky = make_clip(tmp_path / "s.mp4", kind="sky")
    dark = make_clip(tmp_path / "d.mp4", kind="dark")
    assert quality.check(frames.sample(sky)[0].bgr).ok
    q = quality.check(frames.sample(dark)[0].bgr)
    assert "dark" in q.reject and not q.ok


def test_sliding_windows_cover_tail():
    p = np.eye(3)[[0] * 10]  # 10 frames
    w = temporal.sliding_windows(p, window=4, stride=4)
    assert len(w) == 3  # starts 0, 4, and tail 6


def test_aggregate_majority_and_stability():
    p = np.array([[0.9, 0.05, 0.05]] * 12 + [[0.1, 0.8, 0.1]] * 4)
    agg = temporal.aggregate(p, CLASSES, window=4, stride=4)
    assert agg["label"] == "A"
    assert 0.5 < agg["stability"] < 1.0
    assert abs(sum(agg["probabilities"].values()) - 1) < 1e-3


def test_short_clip_single_window():
    p = np.array([[0.2, 0.7, 0.1], [0.3, 0.6, 0.1]])
    agg = temporal.aggregate(p, CLASSES, window=8)
    assert agg["n_windows"] == 1 and agg["label"] == "B" and agg["stability"] == 1.0


def test_ema_is_causal():
    p = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    e = temporal.ema(p, alpha=0.5)
    assert np.allclose(e[0], p[0]) and np.allclose(e[1], [0.5, 0.5, 0])


def test_pipeline_with_fake_classifier(tmp_path):
    clip = make_clip(tmp_path / "c.mp4", n=90)
    calls = []

    def fake(rgb_frames):
        calls.append(len(rgb_frames))
        assert rgb_frames[0].shape[2] == 3
        return np.tile([0.1, 0.7, 0.2], (len(rgb_frames), 1))

    out = pipeline.analyse(clip, classifier=fake, classes=CLASSES)
    assert out["aqi"]["label"] == "B"
    assert out["quality"]["passed"] == out["quality"]["sampled"] == 6
    assert calls == [6] and out["objects"] is None


def test_pipeline_rejects_dark_clip(tmp_path):
    clip = make_clip(tmp_path / "d.mp4", kind="dark")
    out = pipeline.analyse(clip, classifier=lambda f: np.ones((len(f), 3)) / 3, classes=CLASSES)
    assert out["aqi"] is None and out["quality"]["passed"] == 0
    assert "dark" in out["quality"]["reject_reasons"] and out["caveats"]


YOLO = os.environ.get("YOLO_WEIGHTS")
DEMO = os.environ.get("DEMO_CLIP")


@pytest.mark.skipif(not (YOLO and DEMO), reason="set YOLO_WEIGHTS and DEMO_CLIP to run the tracker test")
def test_tracker_on_real_clip():
    from aqivision.video.tracking import Tracker

    res = Tracker(weights=YOLO).run(DEMO, max_frames=30)
    s = res.summary()
    assert res.frames_processed == 30
    assert sum(s["unique_objects"].values()) > 0
    ids = [t.track_id for t in res.tracks]
    assert len(ids) == len(set(ids))
