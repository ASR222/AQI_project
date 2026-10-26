"""Rule-based frame quality gate, run before the AQI classifier.

The phase-1 model was trained on daytime outdoor photos with visible sky, so frames that are
dark, blurred or show no sky are out of distribution and are dropped instead of classified.

Per frame (downscaled to WIDTH px so thresholds don't depend on resolution):
- luma: mean of Y = 0.299R + 0.587G + 0.114B. Night / covered lens -> "dark".
- sharpness: variance of the Laplacian per tile on a TILES x TILES grid, PCT-th percentile over
  tiles. Plain sky has little texture, so whole-frame variance would call sharp sky shots blurred;
  a really blurred frame is soft in every tile. -> "blur".
- sky: share of pixels in the top SKY_TOP of the frame that look like sky (bright and
  low-saturation, i.e. overcast/haze, or blue hue). Indoor / pointed at the ground -> "no_sky".

Thresholds are starting values, not calibrated on a labelled set yet (see README roadmap).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

WIDTH = 320
TILES, PCT = 4, 80
SKY_TOP = 0.40
REASONS = ("dark", "blur", "no_sky")


@dataclass
class Thresholds:
    min_luma: float = 40.0
    min_sharpness: float = 10.0
    min_sky: float = 0.08


@dataclass
class FrameQuality:
    luma: float
    sharpness: float
    sky: float
    reject: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.reject


def features(bgr: np.ndarray) -> tuple[float, float, float]:
    h, w = bgr.shape[:2]
    small = cv2.resize(bgr, (WIDTH, max(1, round(h * WIDTH / w))), interpolation=cv2.INTER_AREA)
    grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    b, g, r = (small[..., i].astype(np.float32) for i in range(3))
    luma = float((0.299 * r + 0.587 * g + 0.114 * b).mean())

    lap = cv2.Laplacian(grey, cv2.CV_64F)
    th, tw = max(1, lap.shape[0] // TILES), max(1, lap.shape[1] // TILES)
    tiles = [lap[i * th : (i + 1) * th, j * tw : (j + 1) * tw].var() for i in range(TILES) for j in range(TILES)]
    sharpness = float(np.percentile(tiles, PCT))

    top = cv2.cvtColor(small[: max(1, round(small.shape[0] * SKY_TOP))], cv2.COLOR_BGR2HSV)
    hue, sat, val = top[..., 0], top[..., 1], top[..., 2]
    bright = (val >= 150) & (sat <= 80)
    blue = (hue >= 90) & (hue <= 130) & (sat >= 40) & (val >= 80)
    sky = float((bright | blue).mean())
    return round(luma, 1), round(sharpness, 1), round(sky, 3)


def check(bgr: np.ndarray, t: Thresholds = Thresholds()) -> FrameQuality:
    luma, sharpness, sky = features(bgr)
    reject = []
    if luma < t.min_luma:
        reject.append("dark")
    if sharpness < t.min_sharpness:
        reject.append("blur")
    if sky < t.min_sky:
        reject.append("no_sky")
    return FrameQuality(luma, sharpness, sky, reject)
