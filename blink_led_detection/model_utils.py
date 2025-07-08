import torch, logging, yaml
from ultralytics import YOLO
from pathlib import Path
from .utils import load_yaml


def load_model(weights: str | Path, device="cpu"):
    return YOLO(str(weights)).to(device)


def train_model(config_path: str | Path):
    cfg = load_yaml(config_path)
    device = 0 if torch.cuda.is_available() else "cpu"
    model_file = f"yolov8{cfg.get('model_variant','n')}.pt"
    model = YOLO(model_file)
    logging.info(f"Training {model_file} on {cfg['dataset_config']} …")
    model.train(
        data=cfg["dataset_config"],
        imgsz=cfg["img_size"],
        epochs=cfg["epochs"],
        batch=cfg["batch_size"],
        project="models",
        name=f"yolov8{cfg['model_variant']}_led",
        device=device,
    )
    return Path("models") / f"yolov8{cfg['model_variant']}_led" / "best.pt"


def evaluate_model(weights: str | Path, data_yaml: str | Path):
    model = YOLO(str(weights))
    res = model.val(data=str(data_yaml))
    logging.info(res.results_dict)
    return res