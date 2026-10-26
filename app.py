import io
import os
import tempfile

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from aqivision import image_model
from aqivision.video import pipeline

app = FastAPI(title="AQI Vision", version="0.2.0")

# Add CORS middleware to allow communication with the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classes = image_model.CLASSES
MAX_VIDEO_MB = int(os.environ.get("MAX_VIDEO_MB", "50"))
YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "yolo11n.pt")

_tracker = None


def get_tracker():
    global _tracker
    if _tracker is None:
        from aqivision.video.tracking import Tracker

        _tracker = Tracker(weights=YOLO_WEIGHTS)
    return _tracker


@app.on_event("startup")
def _load():
    image_model.load_model()  # fail fast if aqi_model.keras is missing


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    orig, sat, sp = image_model.preprocess_all(image)
    prediction = image_model.load_model().predict([orig, sat, sp], verbose=0)

    class_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return {
        "label": classes[class_index],
        "confidence": confidence
    }


@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...), track: bool = True):
    """Short clip (~10 s) -> temporally aggregated AQI class + tracked objects (see aqivision/video)."""
    contents = await file.read()
    if len(contents) > MAX_VIDEO_MB * 1024 * 1024:
        raise HTTPException(413, f"video larger than {MAX_VIDEO_MB} MB")
    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        path = tmp.name
    try:
        result = pipeline.analyse(
            path,
            classifier=image_model.predict_proba,
            classes=classes,
            tracker=get_tracker() if track else None,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    finally:
        os.unlink(path)
    if result.get("objects"):
        result["objects"].pop("tracks", None)
    return result


@app.get("/")
async def root():
    return {"status": "AQI Vision Local Model is running", "endpoints": ["/predict", "/predict-video"]}
