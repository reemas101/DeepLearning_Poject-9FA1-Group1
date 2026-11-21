import argparse
from traffic_signs.experiments import (augmentation_ablation, feature_selection_experiment,
                                       tune_cnn, tune_yolo)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("experiment", choices=["cnn-tuning", "yolo-tuning", "augmentation", "features"])
    p.add_argument("--manifest", default="data/processed/gtsrb_manifest.csv")
    p.add_argument("--data", default="data/processed/gtsdb_yolo/gtsdb.yaml")
    p.add_argument("--cnn", default="outputs/cnn/best.pt")
    p.add_argument("--output", default="outputs/experiments")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if a.experiment == "cnn-tuning":
        result = tune_cnn(a.manifest, f"{a.output}/cnn_tuning", a.seed, a.epochs, a.workers)
    elif a.experiment == "yolo-tuning":
        result = tune_yolo(a.data, f"{a.output}/yolo_tuning", seed=a.seed, epochs=a.epochs)
    elif a.experiment == "augmentation":
        result = augmentation_ablation(a.manifest, f"{a.output}/augmentation", a.seed,
                                       a.epochs, a.workers)
    else:
        result = feature_selection_experiment(a.manifest, a.cnn, f"{a.output}/features",
                                              workers=a.workers, seed=a.seed)
    print(result)


if __name__ == "__main__":
    main()

