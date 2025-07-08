#!/usr/bin/env python3
import argparse, logging
from ultralytics import YOLO
from blink_led_detection.utils import setup_logging

setup_logging()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--format", choices=["onnx", "torchscript"], required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    model = YOLO(args.weights)
    if args.format == "onnx":
        model.export(format="onnx", imgsz=args.imgsz, opset=13, half=False)
    else:
        import torch
        dummy = torch.randn(1, 3, args.imgsz, args.imgsz)
        ts = torch.jit.trace(model.model, dummy)
        out_path = str(args.weights).rsplit('.',1)[0] + ".torchscript.pt"
        ts.save(out_path)
        logging.info(f"TorchScript model saved to {out_path}")