#!/usr/bin/env python3
import argparse, logging, cv2
from pathlib import Path
from ultralytics import YOLO
from blink_led_detection.inference_utils import draw_detections, prepare_writer, log_event
from blink_led_detection.utils import setup_logging

setup_logging()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    model = YOLO(args.weights)
    src = Path(args.source)
    out_vid = Path(args.out) if args.out else src.with_name(src.stem + "_out.mp4")

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        logging.error("Cannot open video")
        exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(3)), int(cap.get(4))
    writer = prepare_writer(out_vid, fps, w, h)

    log_f = open(str(out_vid) + "_LED_events.csv", "w", newline="")
    log_f.write("time_sec,state\n")
    prev_on = False
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        res = model.predict(frame, conf=args.conf, verbose=False)[0]
        led_on = False
        for b in res.boxes:
            if int(b.cls[0]) == 0 and float(b.conf[0]) >= args.conf:
                led_on = True
                frame = draw_detections(frame, b)
        if led_on != prev_on:
            log_event(log_f, idx / fps, "ON" if led_on else "OFF")
            prev_on = led_on
        writer.write(frame)
        idx += 1

    cap.release(); writer.release(); log_f.close()
    logging.info(f"Finished. Output saved to {out_vid}")