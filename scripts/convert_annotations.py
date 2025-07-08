#!/usr/bin/env python3
import argparse, logging
from pathlib import Path
from blink_led_detection.annotation_utils import coco_to_yolo
from blink_led_detection.utils import setup_logging

setup_logging()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to CVAT COCO JSON export")
    ap.add_argument("--format", choices=["coco"], default="coco")
    args = ap.parse_args()
    if args.format == "coco":
        coco_to_yolo(Path(args.input))