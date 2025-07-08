import logging, shutil
from pathlib import Path
from ultralytics.data.converter import convert_coco


def coco_to_yolo(coco_json: str | Path):
    coco_json = Path(coco_json)
    logging.info(f"Converting COCO → YOLO from {coco_json} …")
    convert_coco(labels_dir=str(coco_json.parent), use_segments=False)
    src = coco_json.parent / "coco_converted/labels"
    dst = coco_json.parent / "yolo_labels"
    shutil.move(src, dst)
    logging.info(f"YOLO labels saved to {dst}")