"""Read a video with OpenCV.

Two views of the same clip:
- `iter_frames`: every `stride`-th frame, for detection + tracking (trackers need consecutive frames).
- `sample`: about `every_s` seconds apart, for the slower AQI classifier and the quality gate.
Timestamps come from the container when it has them (phones record variable frame rate).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class Frame:
    index: int
    t: float  # seconds
    bgr: np.ndarray


def video_info(path: str | Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        return {
            "fps": fps if 0 < fps <= 240 else 30.0,
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        cap.release()


def iter_frames(path: str | Path, stride: int = 1, max_frames: int | None = None) -> Iterator[Frame]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if 0 < fps <= 240 else 30.0
    idx, yielded = 0, 0
    try:
        while cap.grab():
            if idx % stride == 0:
                ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                ok, bgr = cap.retrieve()
                if ok:
                    t = ms / 1000 if (ms > 0 or idx == 0) else idx / fps
                    yield Frame(idx, t, bgr)
                    yielded += 1
                    if max_frames is not None and yielded >= max_frames:
                        return
            idx += 1
    finally:
        cap.release()


def sample(path: str | Path, every_s: float = 0.5, max_frames: int = 120) -> list[Frame]:
    """Frames about `every_s` seconds apart (first frame always included)."""
    out: list[Frame] = []
    next_t = 0.0
    for f in iter_frames(path):
        if f.t + 1e-6 >= next_t:
            out.append(f)
            next_t += every_s
            if len(out) >= max_frames:
                break
    if not out:
        raise ValueError(f"no frames decoded from {path}")
    return out
