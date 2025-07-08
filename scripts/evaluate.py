import argparse, logging
from blink_led_detection.model_utils import evaluate_model
from blink_led_detection.utils import setup_logging

setup_logging()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    evaluate_model(args.weights, args.data)