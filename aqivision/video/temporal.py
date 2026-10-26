"""Temporal aggregation of per-frame class probabilities into one clip-level AQI estimate.

Single-frame predictions flicker (a car passes, the camera tilts). We smooth with a sliding
window: window probability = mean of its frames' probabilities; clip probability = mean of the
window probabilities. `stability` = share of windows whose top class equals the clip's top
class, so a clip whose windows disagree is reported as unstable instead of confidently.
`ema` gives a causal running estimate for live streams.
"""

from __future__ import annotations

import numpy as np


def sliding_windows(probs: np.ndarray, window: int = 8, stride: int = 4) -> np.ndarray:
    """(N, C) -> (W, C) mean probability per window. With N < window, one window over all frames."""
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or len(probs) == 0:
        raise ValueError("probs must be a non-empty (N, C) array")
    n = len(probs)
    if n <= window:
        return probs.mean(axis=0, keepdims=True)
    starts = list(range(0, n - window + 1, stride))
    if starts[-1] != n - window:  # cover the tail
        starts.append(n - window)
    return np.stack([probs[s : s + window].mean(axis=0) for s in starts])


def ema(probs: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Causal exponential moving average over frames, (N, C) -> (N, C)."""
    probs = np.asarray(probs, dtype=np.float64)
    out = np.empty_like(probs)
    out[0] = probs[0]
    for i in range(1, len(probs)):
        out[i] = alpha * probs[i] + (1 - alpha) * out[i - 1]
    return out


def aggregate(probs: np.ndarray, classes: list[str], window: int = 8, stride: int = 4) -> dict:
    windows = sliding_windows(probs, window, stride)
    clip = windows.mean(axis=0)
    top = int(clip.argmax())
    win_top = windows.argmax(axis=1)
    frame_top = np.asarray(probs).argmax(axis=1)
    return {
        "label": classes[top],
        "confidence": round(float(clip[top]), 4),
        "probabilities": {c: round(float(p), 4) for c, p in zip(classes, clip)},
        "stability": round(float((win_top == top).mean()), 4),
        "n_frames": int(len(probs)),
        "n_windows": int(len(windows)),
        "frame_labels": [classes[i] for i in frame_top],
    }
