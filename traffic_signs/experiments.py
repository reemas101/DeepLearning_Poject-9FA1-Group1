"""Small, reproducible hyperparameter, augmentation, and feature experiments."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .cnn import load_cnn, make_loader, train_cnn
from .common import save_json, set_seed
from .yolo import evaluate_yolo, train_yolo


def _plot_comparison(table: pd.DataFrame, x: str, y: str, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(table[x].astype(str), table[y])
    ax.set(xlabel=x, ylabel=y, title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def tune_cnn(manifest: str | Path, output: str | Path, seed: int = 42,
             epochs: int = 4, workers: int = 2) -> dict:
    """Three practical configurations; selection uses validation only."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    configs = [("baseline", 1e-3, 128, 64), ("lower_lr", 3e-4, 128, 64),
               ("larger_image_smaller_batch", 1e-3, 64, 80)]
    rows = []
    for name, lr, batch, size in configs:
        result = train_cnn(manifest, output / name, epochs=epochs, batch=batch,
                           size=size, lr=lr, seed=seed, workers=workers)
        rows.append({"experiment": name, **result})
        pd.DataFrame(rows).to_csv(output / "cnn_tuning.csv", index=False)
    table = pd.DataFrame(rows)
    _plot_comparison(table, "experiment", "best_val_accuracy", output / "cnn_tuning.png",
                     "CNN validation accuracy by configuration")
    best = table.loc[table.best_val_accuracy.idxmax()].to_dict()
    save_json(output / "best_cnn_configuration.json", best)
    return best


def tune_yolo(data_yaml: str | Path, output: str | Path, model_name: str = "yolo11n.pt",
              seed: int = 42, epochs: int = 6, device=None) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    configs = [("baseline", 640, 16, 0.01), ("smaller_image", 512, 16, 0.01),
               ("smaller_batch_lr", 640, 8, 0.005)]
    rows = []
    for name, size, batch, lr in configs:
        run = output / name
        trained = train_yolo(data_yaml, run, model_name, epochs, size, batch, lr, seed,
                             device, patience=epochs)
        metrics = evaluate_yolo(data_yaml, trained["checkpoint"], run / "evaluation",
                                split="val", image_size=size, batch=batch, device=device,
                                examples=0)
        rows.append({"experiment": name, "image_size": size, "batch": batch,
                     "learning_rate": lr, "checkpoint": trained["checkpoint"], **{
                         k: metrics[k] for k in ("precision", "recall", "map50", "map50_95")}})
        pd.DataFrame(rows).to_csv(output / "yolo_tuning.csv", index=False)
    table = pd.DataFrame(rows)
    _plot_comparison(table, "experiment", "map50_95", output / "yolo_tuning.png",
                     "YOLO validation mAP@0.50:0.95 by configuration")
    best = table.loc[table.map50_95.idxmax()].to_dict()
    save_json(output / "best_yolo_configuration.json", best)
    return best


def augmentation_ablation(manifest: str | Path, output: str | Path,
                          seed: int = 42, epochs: int = 4, workers: int = 2) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for augment in (False, True):
        name = "augmented" if augment else "resize_only"
        result = train_cnn(manifest, output / name, epochs=epochs, seed=seed,
                           augment=augment, workers=workers)
        rows.append({"preprocessing": name, **result})
        pd.DataFrame(rows).to_csv(output / "augmentation_ablation.csv", index=False)
    table = pd.DataFrame(rows)
    _plot_comparison(table, "preprocessing", "best_val_accuracy",
                     output / "augmentation_ablation.png", "Augmentation validation ablation")
    best = table.loc[table.best_val_accuracy.idxmax()].to_dict()
    save_json(output / "best_augmentation.json", best)
    return best


def _embeddings(model, loader, device):
    features, labels = [], []
    model.eval()
    with torch.inference_mode():
        for images, targets in loader:
            features.append(model.extract_features(images.to(device)).cpu().numpy())
            labels.append(targets.numpy())
    return np.concatenate(features), np.concatenate(labels)


def feature_selection_experiment(manifest: str | Path, checkpoint: str | Path,
                                 output: str | Path, batch: int = 256,
                                 workers: int = 2, seed: int = 42) -> dict:
    """ANOVA ranks frozen CNN embeddings using train only; val selects k; test once."""
    set_seed(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model, size, device = load_cnn(checkpoint)
    arrays = {}
    for split in ("train", "val", "test"):
        loader = make_loader(manifest, split, size, batch, False, workers, seed)
        arrays[split] = _embeddings(model, loader, device)
    x_train, y_train = arrays["train"]
    x_val, y_val = arrays["val"]
    x_test, y_test = arrays["test"]
    rows = []
    for k in (32, 64, 128, 256):
        pipeline = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=k),
                                 SGDClassifier(loss="log_loss", alpha=1e-4,
                                               max_iter=1000, tol=1e-3, random_state=seed))
        pipeline.fit(x_train, y_train)
        prediction = pipeline.predict(x_val)
        rows.append({"selected_features": k, "val_accuracy": accuracy_score(y_val, prediction),
                     "val_f1_macro": f1_score(y_val, prediction, average="macro", zero_division=0)})
    table = pd.DataFrame(rows)
    table.to_csv(output / "feature_selection_validation.csv", index=False)
    _plot_comparison(table, "selected_features", "val_accuracy",
                     output / "feature_selection.png", "Validation accuracy by feature count")
    best_k = int(table.loc[table.val_accuracy.idxmax(), "selected_features"])
    # Refit the chosen configuration on train + val, then use held-out test exactly once.
    final = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=best_k),
                          SGDClassifier(loss="log_loss", alpha=1e-4,
                                        max_iter=1000, tol=1e-3, random_state=seed))
    final.fit(np.concatenate([x_train, x_val]), np.concatenate([y_train, y_val]))
    test_prediction = final.predict(x_test)
    summary = {"best_feature_count": best_k, "test_accuracy": accuracy_score(y_test, test_prediction),
               "test_f1_macro": f1_score(y_test, test_prediction, average="macro", zero_division=0),
               "method": "ANOVA SelectKBest on frozen 256-dimensional CNN embeddings, SGD classifier"}
    save_json(output / "feature_selection_test.json", summary)
    pd.DataFrame({"true_id": y_test, "predicted_id": test_prediction}).to_csv(
        output / "feature_selection_test_predictions.csv", index=False)
    return summary
