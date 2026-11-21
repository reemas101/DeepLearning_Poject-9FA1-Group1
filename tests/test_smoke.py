"""Small fixtures exercise dataset conversion, model training, and multi-sign inference."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from traffic_signs import data
from traffic_signs.cnn import TrafficSignCNN, evaluate_cnn, image_transform, train_cnn
from traffic_signs.experiments import feature_selection_experiment
from traffic_signs.inference import TrafficSignSystem, evaluate_integrated


class ProjectSmokeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_gtsrb_track_split_and_independent_test(self):
        root = self.root

        class FakeGTSRB:
            def __init__(self, root, split, download):
                if split == "train":
                    self._samples = [(f"{root}/{label:05d}/{track:05d}_{frame:05d}.ppm", label)
                                     for label in range(43) for track in range(2) for frame in range(2)]
                else:
                    self._samples = [(f"{root}/test/{label:05d}.ppm", label)
                                     for label in range(43)]

        with patch.object(data, "GTSRB", FakeGTSRB):
            summary = data.prepare_gtsrb(root / "raw", root / "processed", download=False)
        frame = pd.read_csv(root / "processed" / "gtsrb_manifest.csv")
        self.assertEqual(summary["train_val_group_overlap"], 0)
        self.assertEqual(len(frame[frame.split == "test"]), 43)
        self.assertEqual(set(frame[frame.split == "train"].class_id), set(range(43)))

    def test_gtsdb_conversion_and_empty_labels(self):
        raw = self.root / "raw"
        raw.mkdir()
        for number in (0, 1, 600, 601):
            cv2.imwrite(str(raw / f"{number:05d}.ppm"), np.zeros((40, 60, 3), np.uint8))
        crop_folder = raw / "14"
        crop_folder.mkdir()
        cv2.imwrite(str(crop_folder / "00000.ppm"), np.zeros((12, 12, 3), np.uint8))
        (raw / "gt.txt").write_text("00000.ppm;5;5;15;15;1\n"
                                    "00600.ppm;2;2;12;12;1\n"
                                    "00600.ppm;25;5;35;15;14\n", encoding="utf-8")
        out = self.root / "processed"
        summary = data.prepare_gtsdb(raw, out, val_fraction=0.5, strict=False)
        self.assertEqual(summary["image_counts"], {"train": 1, "val": 1, "test": 2})
        self.assertEqual(summary["represented_class_ids"], [1, 14])
        labels = (out / "labels" / "test" / "00600.txt").read_text().splitlines()
        self.assertEqual(len(labels), 2)
        self.assertTrue(all(line.startswith("0 ") for line in labels))
        self.assertEqual((out / "labels" / "test" / "00601.txt").read_text(), "")
        gtsrb_rows = []
        for label in range(43):
            sample_path = self.root / f"gtsrb_{label}.png"
            Image.fromarray(np.full((16, 16, 3), label, np.uint8)).save(sample_path)
            gtsrb_rows.append({"path": str(sample_path), "class_id": label,
                               "group": str(label), "split": "train"})
        manifest = self.root / "gtsrb_manifest.csv"
        pd.DataFrame(gtsrb_rows).to_csv(manifest, index=False)
        compatibility = data.verify_label_compatibility(
            manifest, out / "gtsdb_annotations.csv", self.root / "compatibility")
        self.assertTrue(compatibility["numeric_id_subset_valid"])
        self.assertEqual(compatibility["represented_gtsdb_classes"], 2)
        self.assertTrue((self.root / "compatibility" /
                         "label_compatibility_contact_sheet.png").exists())

    def test_cnn_training_evaluation_and_multisign_pipeline(self):
        image_dir = self.root / "images"
        image_dir.mkdir()
        rows = []
        for split in ("train", "val", "test"):
            for label in (0, 1):
                for index in range(2):
                    path = image_dir / f"{split}_{label}_{index}.png"
                    array = np.full((32, 32, 3), label * 220 + index, np.uint8)
                    Image.fromarray(array).save(path)
                    rows.append({"path": str(path), "class_id": label, "group": split,
                                 "split": split})
        manifest = self.root / "manifest.csv"
        pd.DataFrame(rows).to_csv(manifest, index=False)
        output = self.root / "cnn"
        train_cnn(manifest, output, epochs=1, batch=2, size=32, workers=0)
        metrics = evaluate_cnn(manifest, output / "best.pt", output / "evaluation",
                               batch=2, workers=0)
        self.assertEqual(metrics["samples"], 4)
        self.assertTrue((output / "evaluation" / "cnn_test_confusion.png").exists())
        features = feature_selection_experiment(manifest, output / "best.pt",
                                                self.root / "features", batch=2, workers=0)
        self.assertIn(features["best_feature_count"], {32, 64, 128, 256})
        # Mock YOLO predictions to exercise two independent crops and annotations.
        class Box:
            def __init__(self, coords):
                self.xyxy = torch.tensor([coords], dtype=torch.float32)
                self.conf = torch.tensor([0.9])

        class Detector:
            def predict(self, **kwargs):
                return [type("Result", (), {"boxes": [Box([2, 2, 14, 14]),
                                                       Box([20, 2, 32, 14])]})()]

        system = TrafficSignSystem.__new__(TrafficSignSystem)
        system.cnn = TrafficSignCNN().eval()
        system.crop_size = 32
        system.device = "cpu"
        system.yolo = Detector()
        system.transform = image_transform(32)
        system.confidence = 0.25
        system.image_size = 640
        scene = np.zeros((40, 60, 3), np.uint8)
        annotated, predictions = system.predict(scene)
        self.assertEqual(annotated.shape, scene.shape)
        self.assertEqual(len(predictions), 2)
        test_dir = self.root / "gtsdb" / "images" / "test"
        test_dir.mkdir(parents=True)
        path = test_dir / "00600.jpg"
        cv2.imwrite(str(path), scene)
        annotations = self.root / "annotations.csv"
        pd.DataFrame([{"split": "test", "image": str(path), "source_image": "00600.ppm",
                       "x1": 2, "y1": 2, "x2": 14, "y2": 14, "class_id": 0},
                      {"split": "test", "image": str(path), "source_image": "00600.ppm",
                       "x1": 20, "y1": 2, "x2": 32, "y2": 14, "class_id": 1}]).to_csv(annotations, index=False)
        integrated = evaluate_integrated(system, annotations, self.root / "integrated")
        self.assertEqual(integrated["matched_detections"], 2)


if __name__ == "__main__":
    unittest.main()
