"""Phase-1 image model: three-stream (original / saturation / superpixel) Keras classifier.

Moved out of app.py unchanged so the image endpoint and the video pipeline share one
preprocessing path. The model file (aqi_model.keras) is not in git; put it in the repo
root or point AQI_MODEL_PATH at it. TensorFlow is imported lazily, so the video tools
(detection/tracking) work without it.
"""

from __future__ import annotations

import os
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image
from skimage.color import label2rgb
from skimage.segmentation import slic

CLASSES = [
    "GOOD",
    "MODERATE",
    "UNHEALTHY FOR SENSITIVE PEOPLE",
    "UNHEALTHY",
    "VERY UNHEALTHY",
    "SEVERE",
]

MODEL_PATH = os.environ.get("AQI_MODEL_PATH", "aqi_model.keras")
SIZE = (224, 224)


def create_saturation(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = cv2.equalizeHist(hsv[:, :, 1])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def create_superpixel(img: np.ndarray) -> np.ndarray:
    segments = slic(img, n_segments=30, compactness=5)
    sp_img = label2rgb(segments, img, kind="avg")
    return (sp_img * 255).astype(np.uint8)


def _preprocess_input(x: np.ndarray) -> np.ndarray:
    # Same as tf.keras.applications.mobilenet_v2.preprocess_input (scale to [-1, 1]),
    # which is what the deployed model was served with.
    return x / 127.5 - 1.0


def preprocess_rgb(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """uint8 HxWx3 RGB -> three (224, 224, 3) float32 arrays (orig, sat, superpixel)."""
    image = np.array(Image.fromarray(rgb).resize(SIZE)).astype("uint8")
    sat = create_saturation(image)
    sp = create_superpixel(image)
    return (
        _preprocess_input(image.astype("float32")),
        _preprocess_input(sat.astype("float32")),
        _preprocess_input(sp.astype("float32")),
    )


def preprocess_all(image: Image.Image):
    """PIL image -> batched (orig, sat, sp), as the original /predict endpoint did."""
    orig, sat, sp = preprocess_rgb(np.array(image.convert("RGB")))
    return orig[None], sat[None], sp[None]


@lru_cache(maxsize=1)
def load_model(path: str = MODEL_PATH):
    import tensorflow as tf  # lazy: heavy, and not needed for detection-only runs

    return tf.keras.models.load_model(path)


def predict_proba(rgb_frames: list[np.ndarray], batch_size: int = 16) -> np.ndarray:
    """List of uint8 RGB frames -> (N, 6) class probabilities."""
    if not rgb_frames:
        return np.zeros((0, len(CLASSES)), dtype=np.float32)
    model = load_model()
    out = []
    for i in range(0, len(rgb_frames), batch_size):
        streams = [preprocess_rgb(f) for f in rgb_frames[i : i + batch_size]]
        batch = [np.stack([s[k] for s in streams]) for k in range(3)]
        out.append(np.asarray(model.predict(batch, verbose=0), dtype=np.float32))
    return np.concatenate(out)
