import argparse
from traffic_signs.yolo import evaluate_yolo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/gtsdb_yolo/gtsdb.yaml")
    p.add_argument("--checkpoint", default="outputs/yolo/train/weights/best.pt")
    p.add_argument("--output", default="outputs/yolo/evaluation")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None)
    a = p.parse_args()
    print(evaluate_yolo(a.data, a.checkpoint, a.output, a.split, a.size,
                        a.batch, a.device))


if __name__ == "__main__":
    main()

