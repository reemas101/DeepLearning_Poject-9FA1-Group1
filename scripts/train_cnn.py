import argparse
from traffic_signs.cnn import train_cnn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/processed/gtsrb_manifest.csv")
    p.add_argument("--output", default="outputs/cnn")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-augmentation", action="store_true")
    a = p.parse_args()
    print(train_cnn(a.manifest, a.output, a.epochs, a.batch, a.size, a.lr,
                    not a.no_augmentation, a.seed, a.workers))


if __name__ == "__main__":
    main()

