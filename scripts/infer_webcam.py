#!/usr/bin/env python3
import argparse, logging, cv2, time
from ultralytics import YOLO
from blink_led_detection.inference_utils import draw_detections, prepare_writer, log_event
from blink_led_detection.utils import setup_logging

setup_logging()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="outputs/videos/webcam_out.avi")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--fps", type=float, default=25)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    model = YOLO(args.weights)
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        logging.error("Cannot open webcam")
        exit(1)
    w, h = int(cap.get(3)), int(cap.get(4))
    writer = prepare_writer(args.out, args.fps, w, h)
    log_f = open(args.out + "_LED_events.csv", "w"); log_f.write("time_sec,state\n")
    prev_on, start = False, time.time()

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
        t = time.time() - start
        if led_on != prev_on:
            log_event(log_f, t, "ON" if led_on else "OFF")
            prev_on = led_on
        writer.write(frame)
        cv2.imshow("LED Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release(); writer.release(); log_f.close(); cv2.destroyAllWindows()