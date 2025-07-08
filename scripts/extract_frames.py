#!/usr/bin/env python3
import argparse, logging
from pathlib import Path
from blink_led_detection.data_utils import extract_frames
from blink_led_detection.utils import setup_logging

setup_logging()

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extract frames from video")
    p.add_argument("--video", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--fps", type=float, default=25)
    p.add_argument("--downscale", type=int, default=0, help="Height in px (0 = keep)")
    args = p.parse_args()
    extract_frames(args.video, args.out_dir, args.fps, args.downscale or None)
