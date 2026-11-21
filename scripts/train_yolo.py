import argparse
from traffic_signs.yolo import train_yolo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/gtsdb_yolo/gtsdb.yaml")
    p.add_argument("--output", default="outputs/yolo")
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    a = p.parse_args()
    print(train_yolo(a.data, a.output, a.model, a.epochs, a.size, a.batch,
                     a.lr, a.seed, a.device))


if __name__ == "__main__":
    main()

