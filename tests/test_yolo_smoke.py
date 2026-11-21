"""Exercise the actual Ultralytics train/validate API on temporary images."""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from traffic_signs.data import prepare_gtsdb
from traffic_signs.yolo import evaluate_yolo, train_yolo


class YoloSmokeTest(unittest.TestCase):
    def test_one_epoch_train_and_test_validation(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            root = Path(folder)
            raw = root / "raw"
            raw.mkdir()
            annotations = []
            for number in (0, 1, 600, 601):
                image = np.zeros((96, 128, 3), np.uint8)
                cv2.rectangle(image, (25, 25), (65, 65), (0, 0, 255), -1)
                cv2.imwrite(str(raw / f"{number:05d}.ppm"), image)
                annotations.append(f"{number:05d}.ppm;25;25;65;65;14")
            (raw / "gt.txt").write_text("\n".join(annotations) + "\n", encoding="utf-8")
            prepared = root / "prepared"
            prepare_gtsdb(raw, prepared, val_fraction=0.5, strict=False)
            result = train_yolo(prepared / "gtsdb.yaml", root / "run", model_name="yolo11n.yaml",
                                epochs=1, image_size=64, batch=1, patience=1, device="cpu")
            self.assertTrue(Path(result["checkpoint"]).exists())
            metrics = evaluate_yolo(prepared / "gtsdb.yaml", result["checkpoint"],
                                    root / "evaluation", split="test", image_size=64,
                                    batch=1, device="cpu", examples=1)
            self.assertIn("map50_95", metrics)
            self.assertTrue((root / "evaluation" / "test_examples" / "00600.jpg").exists())


if __name__ == "__main__":
    unittest.main()
