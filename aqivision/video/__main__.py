"""CLI: python -m aqivision.video CLIP [--no-aqi] [--no-track] [--annotated out.mp4] [--json out.json]"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m aqivision.video", description=__doc__)
    p.add_argument("video")
    p.add_argument("--no-aqi", action="store_true", help="skip the AQI classifier (no TensorFlow / model file needed)")
    p.add_argument("--no-track", action="store_true", help="skip YOLO detection + tracking")
    p.add_argument("--weights", default="yolo11n.pt", help="YOLO weights (downloaded by Ultralytics if missing)")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--track-stride", type=int, default=1, help="run the tracker on every n-th frame")
    p.add_argument("--sample-every", type=float, default=0.5, help="seconds between frames sent to the AQI model")
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--annotated", help="write a video with boxes and track ids")
    p.add_argument("--json", help="write the result here instead of stdout")
    p.add_argument("--keep-tracks", action="store_true", help="include the per-track list in the output")
    a = p.parse_args(argv)

    from .pipeline import analyse

    classifier = classes = None
    if not a.no_aqi:
        from aqivision import image_model

        classifier, classes = image_model.predict_proba, image_model.CLASSES
    tracker = None
    if not a.no_track:
        from .tracking import IsolatedTracker, Tracker

        # TensorFlow and PyTorch can't share a process in some builds: isolate YOLO if the AQI model runs too.
        tracker = Tracker(weights=a.weights, conf=a.conf) if a.no_aqi else IsolatedTracker(weights=a.weights, conf=a.conf)

    res = analyse(a.video, classifier, classes, tracker, sample_every_s=a.sample_every, window=a.window,
                  stride=a.stride, track_stride=a.track_stride, annotated_out=a.annotated)
    if hasattr(tracker, "close"):
        tracker.close()
    if res.get("objects") and not a.keep_tracks:
        res["objects"].pop("tracks", None)
    if res.get("aqi"):
        res["aqi"].pop("frame_labels", None)
    text = json.dumps(res, indent=2)
    if a.json:
        with open(a.json, "w") as f:
            f.write(text)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
