import argparse
from traffic_signs.inference import TrafficSignSystem, evaluate_integrated


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", default="data/processed/gtsdb_yolo/gtsdb_annotations.csv")
    p.add_argument("--yolo", default="outputs/yolo/train/weights/best.pt")
    p.add_argument("--cnn", default="outputs/cnn/best.pt")
    p.add_argument("--output", default="outputs/integrated")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--confidence", type=float, default=0.25)
    a = p.parse_args()
    system = TrafficSignSystem(a.yolo, a.cnn, confidence=a.confidence)
    print(evaluate_integrated(system, a.annotations, a.output, split=a.split))


if __name__ == "__main__":
    main()

