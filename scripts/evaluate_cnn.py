import argparse
from traffic_signs.cnn import evaluate_cnn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/processed/gtsrb_manifest.csv")
    p.add_argument("--checkpoint", default="outputs/cnn/best.pt")
    p.add_argument("--output", default="outputs/cnn/evaluation")
    p.add_argument("--split", choices=["val", "test"], default="test")
    a = p.parse_args()
    print(evaluate_cnn(a.manifest, a.checkpoint, a.output, a.split))


if __name__ == "__main__":
    main()

