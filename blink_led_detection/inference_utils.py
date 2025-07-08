import cv2, logging, csv
from pathlib import Path


def draw_detections(frame, box, label="LED", color=(0,255,0)):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = float(box.conf[0])
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame


def prepare_writer(out_path: str | Path, fps: float, width: int, height: int):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))


def log_event(log_file, time_sec: float, state: str):
    log_file.write(f"{time_sec:.3f},{state}\n")
    log_file.flush()
    logging.debug(f"LED {state} at {time_sec:.3f}s")