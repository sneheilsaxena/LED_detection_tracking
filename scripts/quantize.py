#!/usr/bin/env python3
import argparse, torch, logging
from ultralytics import YOLO
from blink_led_detection.utils import setup_logging

setup_logging()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    args = ap.parse_args()

    base = YOLO(args.weights)
    model = base.model
    logging.info("Applying dynamic quantization …")
    q_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    out = str(args.weights).rsplit('.',1)[0] + "_int8.pth"
    torch.save(q_model.state_dict(), out)
    logging.info(f"Quantized model saved to {out}")