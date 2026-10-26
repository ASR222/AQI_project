# AQI Vision

Estimating air quality from what a camera sees. This is a BIT Mesra minor project (ECE), supervised by Dr. Afaque Azam. Team: Rudraksh Soni, Abhay Singh, Arka Prabha Bauri.

- **Live demo (image model):** https://aqi-vision-classifier.vercel.app/predict
- **Sequel / deployed system:** [HazeBusters](https://github.com/ASR222/hazebusters), [live](https://main.dxwuap9t49si3.amplifyapp.com/). It is a serverless AWS app that grew out of this project's phase-2 experiments.

| Phase | What | Where |
|---|---|---|
| 1 | Image → 6-class AQI with a three-stream CNN (RGB + saturation + superpixel, EfficientNet-B0 backbones, late fusion), FastAPI + Next.js app | `aqivision/image_model.py`, `app.py` (`/predict`), `frontend/` |
| 2 | Video: frame sampling, quality gate, temporal aggregation of AQI predictions, YOLO11 detection + ByteTrack tracking of vehicles/people, leak-free (grouped) evaluation | `aqivision/video/`, `aqivision/evaluation.py`, `app.py` (`/predict-video`) |

## Phase 2: video pipeline

```
clip ──┬─ sample ~2 fps ─> quality gate (dark / blur / no sky) ─> 3-stream CNN per frame ─> sliding-window aggregate
       └─ every frame ───> YOLO11n + ByteTrack ─> unique vehicles / people, peak count, heavy-vehicle share
```

- **Frame sampling** (`video/frames.py`). OpenCV, with container timestamps because phones record variable frame rate. The tracker gets consecutive frames. The classifier gets frames about 0.5 s apart.
- **Quality gate** (`video/quality.py`). Each frame is rejected if it is dark (mean luma), blurred (Laplacian variance per tile, so a plain sky isn't called blurred) or shows no sky in its top 40%. The image model was trained on daytime outdoor photos, so these frames are out of distribution and are dropped rather than classified. Fewer than 3 usable frames gives no AQI estimate and a caveat.
- **Temporal aggregation** (`video/temporal.py`). Each sliding window (8 frames, stride 4) averages its frame probabilities, and the clip probability is the mean over windows. **Stability** is the share of windows that agree with the clip's class, and a clip below 0.6 is flagged low-confidence. An EMA gives a causal running estimate for live streams.
- **Detection + tracking** (`video/tracking.py`). YOLO11n (COCO) with ByteTrack. It counts *unique* tracks per class instead of boxes per frame, ignoring tracks shorter than 3 frames. Outputs include per-track dwell time, peak objects in a frame and heavy-vehicle (bus + truck) share. Detections are context only and never change the AQI estimate.
- **Grouped evaluation** (`aqivision/evaluation.py`). Group splits, leave-one-group-out, a leakage report and the majority-class baseline. See *Findings*.

### Run it

```bash
uv sync --extra video          # TensorFlow for the image model, Ultralytics for tracking

# CLI
uv run python -m aqivision.video clip.mp4 --annotated out.mp4 --json result.json
uv run python -m aqivision.video clip.mp4 --no-aqi       # detection/tracking only (no model file needed)

# API
uv run uvicorn app:app --reload
curl -F file=@clip.mp4 localhost:8000/predict-video
curl -F file=@photo.jpg localhost:8000/predict
```

`aqi_model.keras` (phase-1 weights) is not in git. Put it in the repo root or set `AQI_MODEL_PATH`. YOLO weights download on first use, or you can set `YOLO_WEIGHTS`.

### Tests

```bash
uv run pytest -q                                     # synthetic clips, no model files needed
YOLO_WEIGHTS=yolo11n.pt DEMO_CLIP=street.mp4 uv run pytest -q   # also runs the real tracker test
```

## Findings so far (phase 2)

Phase 1 was trained and evaluated on the India/Nepal image dataset with a random 80/20 image-level split (98.67% validation accuracy, see the report). In phase 2 we moved to **TRAQID** (IIIT Hyderabad dashcam frames with co-located PM2.5/PM10, Kathalkar et al., ICVGIP 2024) and evaluated by **calendar date**, so no day appears in both train and test. The experiments were run in the HazeBusters repo (`ml/RESULTS.md` there):

- A fine-tuned EfficientNet-B0 (3 CPCB bands, 13,789 daytime frames from 17 dates) reached **89.0% on its training dates but 32.0% on 5 held-out dates**, below the 47.5% majority baseline. It learned the day (light, route, weather), not the haze.
- With the same frozen features, a random frame split scored **63.3%** and leave-one-date-out scored **33.5%**. The split alone moved accuracy by about 30 points.
- On India/Nepal with leave-one-city-out, the predictions came out inverted, because each city is nearly one AQI band.

So phase 2 treats the camera as a source of **context** (what is visibly polluting: traffic, heavy vehicles, smoke) and of temporally smoothed estimates, and evaluates everything with grouped splits.

## Roadmap

- Calibrate the quality gate on a labelled set of our own clips (thresholds are starting values).
- Collect our own paired clips (phone video + nearest station reading, Ranchi/Jharkhand) and evaluate the three-stream model with `evaluation.group_split` by date and location.
- Fine-tune YOLO for smoke/fire and dust plumes (D-Fire) so detections cover pollution sources, not only vehicles.
- Line-crossing counts (vehicles per minute) for fixed cameras.
- Edge export (ONNX / INT8) for on-device use.

## Project structure

```
AQI_project/
├── app.py                    # FastAPI: /predict (image), /predict-video (clip)
├── aqivision/
│   ├── image_model.py        # phase-1 three-stream preprocessing + model loading
│   ├── evaluation.py         # grouped splits, leakage report, baselines
│   └── video/
│       ├── frames.py         # OpenCV frame iteration / sampling
│       ├── quality.py        # dark / blur / no-sky gate
│       ├── temporal.py       # sliding-window + EMA aggregation
│       ├── tracking.py       # YOLO11 + ByteTrack
│       ├── pipeline.py       # clip -> JSON
│       └── __main__.py       # CLI
├── scripts/check_model.py    # print the Keras model's input/output shapes
├── tests/                    # pytest
└── frontend/                 # Next.js app (phase 1)
```

## Credits and licences

- TRAQID: Kathalkar et al., ICVGIP 2024 (research use, non-commercial).
- India/Nepal air-pollution image dataset: Kaggle, CC BY-NC-SA 3.0 IGO.
- Ultralytics YOLO11 / ByteTrack: AGPL-3.0. They are an optional dependency, and the weights are not redistributed here.
- EfficientNet (Tan & Le, 2019), SLIC superpixels (scikit-image).
