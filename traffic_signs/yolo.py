"""Fine-tune and evaluate a one-class Ultralytics YOLO detector."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from .common import save_json, set_seed


def train_yolo(data_yaml: str | Path, output: str | Path, model_name: str = "yolo11n.pt",
               epochs: int = 30, image_size: int = 640, batch: int = 16,
               lr: float = 0.01, seed: int = 42, device: str | int | None = None,
               patience: int = 8) -> dict:
    set_seed(seed)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_name)
    kwargs = {"data": str(Path(data_yaml).resolve()), "epochs": epochs, "imgsz": image_size,
              "batch": batch, "lr0": lr, "seed": seed, "patience": patience,
              "optimizer": "SGD",
              "project": str(output), "name": "train", "exist_ok": True,
              "plots": True, "save": True, "workers": 2,
              # Traffic signs can change meaning when mirrored or inverted.
              "fliplr": 0.0, "flipud": 0.0}
    if device is not None:
        kwargs["device"] = device
    model.train(**kwargs)
    checkpoint = output / "train" / "weights" / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"YOLO did not save {checkpoint}")
    summary = {"checkpoint": str(checkpoint), "pretrained_model": model_name,
               "epochs_requested": epochs, "image_size": image_size, "batch": batch,
               "learning_rate": lr, "seed": seed}
    save_json(output / "train_summary.json", summary)
    return summary


def evaluate_yolo(data_yaml: str | Path, checkpoint: str | Path, output: str | Path,
                  split: str = "test", image_size: int = 640, batch: int = 16,
                  device: str | int | None = None, examples: int = 6) -> dict:
    if split not in {"val", "test"}:
        raise ValueError("Evaluation split must be val or test")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(checkpoint))
    kwargs = {"data": str(Path(data_yaml).resolve()), "split": split, "imgsz": image_size,
              "batch": batch, "project": str(output), "name": f"{split}_validation",
              "exist_ok": True, "plots": True}
    if device is not None:
        kwargs["device"] = device
    metrics = model.val(**kwargs)
    box = metrics.box
    summary = {"split": split, "precision": float(box.mp), "recall": float(box.mr),
               "map50": float(box.map50), "map50_95": float(box.map),
               "mean_inference_ms_per_image": float(metrics.speed.get("inference", float("nan"))),
               "image_size": image_size, "checkpoint": str(checkpoint)}
    save_json(output / f"yolo_{split}_metrics.json", summary)
    pd.DataFrame([summary]).to_csv(output / f"yolo_{split}_metrics.csv", index=False)
    from yaml import safe_load
    config = safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    images = sorted((Path(config["path"]) / config[split]).glob("*.jpg"))[:examples]
    if images:
        predictions = model.predict(source=[str(p) for p in images], imgsz=image_size,
                                    conf=0.25, verbose=False)
        visualization = output / f"{split}_examples"
        visualization.mkdir(exist_ok=True)
        import cv2
        for source, result in zip(images, predictions):
            cv2.imwrite(str(visualization / source.name), result.plot())
    return summary
