"""Custom 43-class CNN and its training/evaluation routines."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T

from .common import CLASS_NAMES, NUM_CLASSES, device_name, save_json, set_seed
from .data import ManifestDataset


class TrafficSignCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        blocks = []
        channels = 3
        for width in (32, 64, 128):
            blocks += [nn.Conv2d(channels, width, 3, padding=1), nn.BatchNorm2d(width),
                       nn.ReLU(inplace=True), nn.Conv2d(width, width, 3, padding=1),
                       nn.BatchNorm2d(width), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            channels = width
        self.features = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d((2, 2)))
        self.embedding = nn.Sequential(nn.Flatten(), nn.Linear(128 * 2 * 2, 256), nn.ReLU(inplace=True))
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, num_classes))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(self.features(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


def image_transform(size: int = 64, augment: bool = False):
    steps = [T.Resize((size, size))]
    if augment:
        # No horizontal/vertical flips: they can change traffic sign meaning.
        steps += [T.RandomRotation(10), T.ColorJitter(brightness=0.2, contrast=0.2,
                                                      saturation=0.15)]
    return T.Compose(steps + [T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])


def make_loader(manifest, split, size=64, batch=128, augment=False, workers=2, seed=42):
    dataset = ManifestDataset(manifest, split, image_transform(size, augment))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch, shuffle=(split == "train"),
                      num_workers=workers, pin_memory=torch.cuda.is_available(),
                      generator=generator)


def run_epoch(model, loader, optimizer, device):
    training = optimizer is not None
    model.train(training)
    loss_fn = nn.CrossEntropyLoss()
    total_loss = total_correct = total = 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)
    return {"loss": total_loss / total, "accuracy": total_correct / total}


def plot_history(history: pd.DataFrame, destination: str | Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for metric, ax in zip(("loss", "accuracy"), axes):
        for split in ("train", "val"):
            ax.plot(history.epoch, history[f"{split}_{metric}"], label=split)
        ax.set(xlabel="Epoch", ylabel=metric.title(), title=f"CNN {metric.title()}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def train_cnn(manifest: str | Path, output: str | Path, epochs: int = 12,
              batch: int = 128, size: int = 64, lr: float = 1e-3,
              augment: bool = True, seed: int = 42, workers: int = 2,
              device: str = "auto", patience: int = 4) -> dict:
    set_seed(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = device_name(device)
    train_loader = make_loader(manifest, "train", size, batch, augment, workers, seed)
    val_loader = make_loader(manifest, "val", size, batch, False, workers, seed)
    model = TrafficSignCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_accuracy, stale, rows = -1.0, 0, []
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device)
        val_metrics = run_epoch(model, val_loader, None, device)
        row = {"epoch": epoch, "train_loss": train_metrics["loss"],
               "train_accuracy": train_metrics["accuracy"], "val_loss": val_metrics["loss"],
               "val_accuracy": val_metrics["accuracy"]}
        rows.append(row)
        pd.DataFrame(rows).to_csv(output / "history.csv", index=False)
        print(f"Epoch {epoch}: {row}")
        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy, stale = val_metrics["accuracy"], 0
            torch.save({"state_dict": copy.deepcopy(model.state_dict()), "image_size": size,
                        "class_names": CLASS_NAMES, "seed": seed, "epoch": epoch,
                        "val_accuracy": best_accuracy}, output / "best.pt")
        else:
            stale += 1
        if stale >= patience:
            break
    history = pd.DataFrame(rows)
    plot_history(history, output / "training_curves.png")
    summary = {"best_val_accuracy": best_accuracy, "best_epoch": int(history.loc[
        history.val_accuracy.idxmax(), "epoch"]), "epochs_run": len(rows),
        "batch": batch, "image_size": size, "learning_rate": lr, "augment": augment,
        "seed": seed, "device": device}
    save_json(output / "train_summary.json", summary)
    return summary


def load_cnn(checkpoint: str | Path, device: str = "auto"):
    device = device_name(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if len(state["class_names"]) != NUM_CLASSES:
        raise ValueError("CNN checkpoint does not have 43 class names")
    model = TrafficSignCNN().to(device)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model, int(state["image_size"]), device


def predict_loader(model, loader, device):
    labels, predictions, confidences = [], [], []
    model.eval()
    with torch.inference_mode():
        for images, targets in loader:
            probs = model(images.to(device)).softmax(1)
            confidence, pred = probs.max(1)
            labels.extend(targets.tolist())
            predictions.extend(pred.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())
    return np.array(labels), np.array(predictions), np.array(confidences)


def evaluate_cnn(manifest: str | Path, checkpoint: str | Path, output: str | Path,
                 split: str = "test", batch: int = 128, workers: int = 2,
                 device: str = "auto") -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model, size, device = load_cnn(checkpoint, device)
    loader = make_loader(manifest, split, size, batch, False, workers)
    truth, predicted, confidence = predict_loader(model, loader, device)
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predicted, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0)
    accuracy = float(np.mean(truth == predicted))
    summary = {"split": split, "samples": len(truth), "accuracy": accuracy,
               "precision_macro": float(precision), "recall_macro": float(recall),
               "f1_macro": float(f1)}
    save_json(output / f"cnn_{split}_metrics.json", summary)
    pd.DataFrame([summary]).to_csv(output / f"cnn_{split}_metrics.csv", index=False)
    report = classification_report(truth, predicted, labels=list(range(NUM_CLASSES)),
                                   target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(output / f"cnn_{split}_class_report.csv")
    matrix = confusion_matrix(truth, predicted, labels=list(range(NUM_CLASSES)))
    pd.DataFrame(matrix, index=range(NUM_CLASSES), columns=range(NUM_CLASSES)).to_csv(
        output / f"cnn_{split}_confusion.csv")
    fig, ax = plt.subplots(figsize=(12, 11))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set(xlabel="Predicted class ID", ylabel="True class ID", title=f"CNN {split} confusion matrix")
    ax.set_xticks(range(NUM_CLASSES), labels=range(NUM_CLASSES), fontsize=6)
    ax.set_yticks(range(NUM_CLASSES), labels=range(NUM_CLASSES), fontsize=6)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output / f"cnn_{split}_confusion.png", dpi=160)
    plt.close(fig)
    table = loader.dataset.rows[["path", "class_id"]].copy()
    table["predicted_id"] = predicted
    table["confidence"] = confidence
    table.to_csv(output / f"cnn_{split}_predictions.csv", index=False)
    return summary
