import cv2, logging
from pathlib import Path
from .utils import ensure_dir

def extract_frames(video_path: str | Path, output_dir: str | Path,
                   target_fps: float = 25.0, downscale_height: int | None = None):
    video_path, output_dir = Path(video_path), Path(output_dir)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.error(f"Cannot open {video_path}")
        return 0
    ensure_dir(output_dir)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    interval = max(int(round(src_fps / target_fps)), 1)
    idx, saved = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            if downscale_height:
                h, w = frame.shape[:2]
                scale = downscale_height / float(h)
                frame = cv2.resize(frame, (int(w*scale), downscale_height), cv2.INTER_AREA)
            name = output_dir / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(name), frame)
            saved += 1
        idx += 1
    cap.release()
    logging.info(f"Extracted {saved} frames to {output_dir}")
    return saved
