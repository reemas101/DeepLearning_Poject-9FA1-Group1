"""Annotate one road image from the command line."""
import argparse
from pathlib import Path

import cv2

from traffic_signs.common import save_json
from traffic_signs.inference import TrafficSignSystem


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image", help="Road image path")
    p.add_argument("--yolo", default="outputs/yolo/train/weights/best.pt")
    p.add_argument("--cnn", default="outputs/cnn/best.pt")
    p.add_argument("--output", default="outputs/predictions")
    p.add_argument("--confidence", type=float, default=0.25)
    a = p.parse_args()
    image = cv2.imread(a.image)
    if image is None:
        raise FileNotFoundError(a.image)
    system = TrafficSignSystem(a.yolo, a.cnn, confidence=a.confidence)
    annotated, predictions = system.predict(image)
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    image_path = output / Path(a.image).name
    if not cv2.imwrite(str(image_path), annotated):
        raise OSError(f"Cannot write {image_path}")
    save_json(output / f"{Path(a.image).stem}.json", {"predictions": predictions})
    print(image_path)


if __name__ == "__main__":
    main()
