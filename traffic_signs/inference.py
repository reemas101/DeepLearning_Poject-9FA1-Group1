"""YOLO -> crop -> CNN inference, visualization, and GTSDB end-to-end evaluation."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from ultralytics import YOLO

from .cnn import image_transform, load_cnn
from .common import CLASS_NAMES, save_json


def intersection_over_union(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


class TrafficSignSystem:
    def __init__(self, yolo_checkpoint: str | Path, cnn_checkpoint: str | Path,
                 device: str = "auto", confidence: float = 0.25, image_size: int = 640):
        self.cnn, self.crop_size, self.device = load_cnn(cnn_checkpoint, device)
        self.yolo = YOLO(str(yolo_checkpoint))
        self.transform = image_transform(self.crop_size)
        self.confidence = confidence
        self.image_size = image_size
        if set(self.yolo.names.values()) != {"traffic_sign"}:
            raise ValueError("Detector checkpoint must be trained as one class: traffic_sign")

    def predict(self, image_bgr: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        if image_bgr is None or image_bgr.ndim != 3:
            raise ValueError("Expected a color road image")
        height, width = image_bgr.shape[:2]
        result = self.yolo.predict(source=image_bgr, conf=self.confidence,
                                   imgsz=self.image_size, verbose=False)[0]
        annotated, predictions = image_bgr.copy(), []
        for box in result.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].cpu().tolist())
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image_bgr[y1:y2, x1:x2]
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                probabilities = self.cnn(tensor).softmax(1)[0]
            class_id = int(probabilities.argmax().item())
            cnn_confidence = float(probabilities[class_id].item())
            detection_confidence = float(box.conf[0].item())
            prediction = {"box": [x1, y1, x2, y2], "class_id": class_id,
                          "class_name": CLASS_NAMES[class_id],
                          "detection_confidence": detection_confidence,
                          "classification_confidence": cnn_confidence}
            predictions.append(prediction)
            label = f"{CLASS_NAMES[class_id]} | det {detection_confidence:.2f} | cls {cnn_confidence:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            ty = max(text_size[1] + 7, y1)
            cv2.rectangle(annotated, (x1, ty - text_size[1] - 6),
                          (min(width, x1 + text_size[0] + 4), ty + 2), (0, 90, 0), -1)
            cv2.putText(annotated, label, (x1 + 2, ty - 3), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return annotated, predictions


def evaluate_integrated(system: TrafficSignSystem, annotations_csv: str | Path,
                        output: str | Path, split: str = "test", iou_threshold: float = 0.5,
                        example_count: int = 6) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(annotations_csv)
    rows = table[table.split == split]
    if rows.empty:
        raise ValueError(f"No annotated {split} images")
    # Include empty-label images in the evaluation denominator for false positives.
    image_dir = Path(rows.iloc[0].image).parent
    image_paths = sorted(image_dir.glob("*.jpg"))
    gt_by_image = {Path(name).name: group.to_dict("records") for name, group in rows.groupby("image")}
    total_gt = matched = correct = total_predictions = 0
    prediction_rows = []
    sample_dir = output / "integrated_examples"
    sample_dir.mkdir(exist_ok=True)
    for index, path in enumerate(image_paths):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Unreadable image: {path}")
        annotated, predictions = system.predict(image)
        if index < example_count:
            cv2.imwrite(str(sample_dir / path.name), annotated)
        ground_truth = gt_by_image.get(path.name, [])
        total_gt += len(ground_truth)
        total_predictions += len(predictions)
        candidates = []
        for pi, prediction in enumerate(predictions):
            for gi, truth in enumerate(ground_truth):
                iou = intersection_over_union(prediction["box"],
                                             [truth[k] for k in ("x1", "y1", "x2", "y2")])
                if iou >= iou_threshold:
                    candidates.append((iou, pi, gi))
        used_p, used_g = set(), set()
        matches = {}
        for iou, pi, gi in sorted(candidates, reverse=True):
            if pi not in used_p and gi not in used_g:
                used_p.add(pi)
                used_g.add(gi)
                matches[pi] = (gi, iou)
                matched += 1
                correct += int(predictions[pi]["class_id"] == int(ground_truth[gi]["class_id"]))
        for pi, prediction in enumerate(predictions):
            match = matches.get(pi)
            prediction_rows.append({"image": path.name, **prediction,
                                    "true_class_id": int(ground_truth[match[0]]["class_id"]) if match else None,
                                    "iou": match[1] if match else None,
                                    "matched": match is not None,
                                    "correct_class": bool(match and prediction["class_id"] ==
                                                          int(ground_truth[match[0]]["class_id"]))})
    summary = {"split": split, "images": len(image_paths), "ground_truth_signs": total_gt,
               "predictions": total_predictions, "matched_detections": matched,
               "correctly_classified_matches": correct,
               "detection_recall_at_iou": matched / total_gt if total_gt else 0.0,
               "classification_accuracy_on_matches": correct / matched if matched else 0.0,
               "end_to_end_recall": correct / total_gt if total_gt else 0.0,
               "end_to_end_precision": correct / total_predictions if total_predictions else 0.0,
               "iou_threshold": iou_threshold}
    pd.DataFrame(prediction_rows).to_csv(output / "integrated_predictions.csv", index=False)
    save_json(output / "integrated_metrics.json", summary)
    pd.DataFrame([summary]).to_csv(output / "integrated_metrics.csv", index=False)
    return summary
