import argparse, logging
from blink_led_detection.model_utils import train_model
from blink_led_detection.utils import setup_logging

setup_logging()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config YAML")
    args = ap.parse_args()
    best = train_model(args.config)
    logging.info(f"Best model saved at {best}")