#!/usr/bin/env python3
"""Standalone efficient inference for object detection (YOLO).

Saves annotated images and a JSON summary to `results/inference/`.

Usage example:
  ./.venv/bin/python inference_object_detection.py --model yolov8n.pt --source images/ --outdir results/inference
"""
from pathlib import Path
import argparse
import json
import os
import sys
from typing import List

import numpy as np
from ultralytics import YOLO
from PIL import Image
import cv2


def extract_confidences_from_result(res) -> np.ndarray:
    """Robustly extract detection confidences from an ultralytics Result object."""
    try:
        boxes = getattr(res, "boxes", None)
        if boxes is None:
            return np.array([])
        conf = getattr(boxes, "conf", None)
        if conf is None:
            return np.array([])
        # conf may be a tensor-like sequence; coerce to floats
        conf_list = [float(c) for c in conf]
        return np.array(conf_list, dtype=float)
    except Exception:
        return np.array([])


def find_images(source: Path) -> List[Path]:
    if source.is_file():
        return [source]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    files = [p for p in source.rglob("*") if p.suffix.lower() in exts]
    return sorted(files)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Edge/standalone inference for YOLO object detection")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path or name")
    parser.add_argument("--source", default=".", help="Image file or folder (glob supported)")
    parser.add_argument("--outdir", default="results/inference", help="Output dir for annotated images and summary")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", default="cpu", help="Device to run on, e.g. cpu or 0")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--max-images", type=int, default=100, help="Max images to process (0 = all)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    src = Path(args.source)
    images = find_images(src)
    if not images:
        print(f"No images found at {src}. Provide a folder or file path.")
        sys.exit(1)

    if args.max_images > 0:
        images = images[: args.max_images]

    print(f"Loading model {args.model} on device {args.device}")
    model = YOLO(args.model)

    summary = {"images": [], "total_detections": 0}
    counts = {"high": 0, "medium": 0, "low": 0}

    for i, img_path in enumerate(images):
        print(f"[{i+1}/{len(images)}] Inferring: {img_path}")
        try:
            results = model.predict(source=str(img_path), imgsz=args.imgsz, device=args.device, conf=args.conf, verbose=False)
            if not results:
                res = None
            else:
                res = results[0]
        except Exception as e:
            print(f"  Inference error for {img_path}: {e}")
            res = None

        confidences = extract_confidences_from_result(res) if res is not None else np.array([])
        n = len(confidences)
        avg_conf = float(confidences.mean()) if n > 0 else 0.0

        # Categorize confidences
        if n > 0:
            counts["high"] += int((confidences > 0.8).sum())
            counts["medium"] += int(((confidences > 0.5) & (confidences <= 0.8)).sum())
            counts["low"] += int((confidences <= 0.5).sum())

        summary["total_detections"] += n

        # Save annotated image (use result.plot() when available)
        out_img_path = outdir / f"{img_path.stem}_annotated{img_path.suffix}"
        saved = False
        try:
            if res is not None:
                plotted = res.plot()
                if isinstance(plotted, (list, tuple)):
                    plotted = plotted[0]
                if isinstance(plotted, np.ndarray):
                    # ultralytics plot is BGR (cv2) — convert to RGB for PIL
                    try:
                        plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
                    except Exception:
                        plotted_rgb = plotted
                    Image.fromarray(plotted_rgb).save(out_img_path)
                    saved = True
        except Exception:
            saved = False

        if not saved:
            # fallback: copy original image
            try:
                Image.open(img_path).save(out_img_path)
            except Exception:
                pass

        summary["images"].append({
            "file": str(img_path),
            "annotated": str(out_img_path),
            "detections": n,
            "avg_confidence": round(avg_conf, 4),
        })

    summary["counts_by_confidence_bucket"] = counts

    summary_path = outdir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Done — annotated images and summary written to {outdir}")
    print(f"Summary: total_detections={summary['total_detections']}, high={counts['high']}, medium={counts['medium']}, low={counts['low']}")


if __name__ == "__main__":
    main()
