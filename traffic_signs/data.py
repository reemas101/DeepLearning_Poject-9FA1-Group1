"""Prepare the official GTSRB and GTSDB benchmarks without test leakage."""
from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import pandas as pd
import yaml
from PIL import Image
from PIL import ImageDraw
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.datasets import GTSRB

from .common import CLASS_NAMES, NUM_CLASSES, save_json


def prepare_gtsrb(root: str | Path, processed: str | Path, seed: int = 42,
                  val_fraction: float = 0.2, download: bool = True) -> dict:
    """Split official training tracks by class; keep official test untouched."""
    root, processed = Path(root), Path(processed)
    train = GTSRB(root=str(root), split="train", download=download)
    test = GTSRB(root=str(root), split="test", download=download)
    records = []
    for path, label in train._samples:  # torchvision exposes the decoded file list here
        path = Path(path)
        match = re.match(r"^(\d+)_\d+$", path.stem)
        if not match:
            raise ValueError(f"Unrecognized GTSRB track filename: {path.name}")
        records.append({"path": str(path.resolve()), "class_id": int(label),
                        "group": f"{label}/{match.group(1)}"})
    frame = pd.DataFrame(records)
    if set(frame.class_id) != set(range(NUM_CLASSES)):
        raise ValueError("GTSRB train must contain all 43 official class IDs")
    val_groups = set()
    for label, part in frame.groupby("class_id"):
        groups = sorted(part.group.unique())
        if len(groups) < 2:
            raise ValueError(f"Class {label} has fewer than two tracks")
        _, selected = train_test_split(groups, test_size=val_fraction,
                                       random_state=seed)
        val_groups.update(selected)
    frame["split"] = frame.group.map(lambda x: "val" if x in val_groups else "train")
    test_rows = [{"path": str(Path(path).resolve()), "class_id": int(label),
                  "group": "official_test", "split": "test"} for path, label in test._samples]
    all_rows = pd.concat([frame, pd.DataFrame(test_rows)], ignore_index=True)
    if set(all_rows[all_rows.split == "test"].class_id) != set(range(NUM_CLASSES)):
        raise ValueError("GTSRB official test labels are incomplete")
    processed.mkdir(parents=True, exist_ok=True)
    manifest = processed / "gtsrb_manifest.csv"
    all_rows.to_csv(manifest, index=False)
    counts = all_rows.groupby(["split", "class_id"]).size().rename("count").reset_index()
    counts.to_csv(processed / "gtsrb_class_counts.csv", index=False)
    summary = {"source": "official torchvision GTSRB", "seed": seed,
               "split_counts": all_rows.split.value_counts().to_dict(),
               "train_val_group_overlap": len(set(frame[frame.split == "train"].group) &
                                              set(frame[frame.split == "val"].group))}
    save_json(processed / "gtsrb_summary.json", summary)
    return summary


class ManifestDataset(Dataset):
    def __init__(self, manifest: str | Path, split: str, transform=None):
        table = pd.read_csv(manifest)
        self.rows = table[table.split == split].reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"No {split} rows in {manifest}")
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        with Image.open(row.path) as image:
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(row.class_id)


def read_gtsdb_annotations(path: str | Path) -> dict[str, list[tuple[int, int, int, int, int]]]:
    annotations = defaultdict(list)
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.reader(handle, delimiter=";"):
            if not row or not row[0].strip():
                continue
            if len(row) != 6:
                raise ValueError(f"Expected six fields in {path}: {row}")
            name = Path(row[0].strip()).name
            x1, y1, x2, y2, label = map(int, row[1:])
            if not (0 <= label < NUM_CLASSES and x1 < x2 and y1 < y2):
                raise ValueError(f"Invalid GTSDB annotation: {row}")
            annotations[name].append((x1, y1, x2, y2, label))
    return dict(annotations)


def prepare_gtsdb(raw_dir: str | Path, processed: str | Path, seed: int = 42,
                  val_fraction: float = 0.15, strict: bool = True) -> dict:
    """Convert full 900-image archive to one-class YOLO detection labels.

    Images 00000..00599 are the historic train partition, 00600..00899 test.
    The original 43-class annotation is retained for integrated evaluation.
    """
    raw_dir, processed = Path(raw_dir), Path(processed)
    # FullIJCNN2013 contains class subfolders of cropped signs with duplicate
    # five-digit names. Only top-level PPM files are full road scenes.
    image_paths = sorted(raw_dir.glob("*.ppm"))
    image_paths = [p for p in image_paths if re.fullmatch(r"\d{5}", p.stem)]
    by_name = {p.name: p for p in image_paths}
    if len(by_name) != len(image_paths):
        raise ValueError("Duplicate full-scene image names found")
    if strict and (len(image_paths) != 900 or
                   set(by_name) != {f"{i:05d}.ppm" for i in range(900)}):
        raise ValueError("Expected FullIJCNN2013 with exactly 00000.ppm..00899.ppm")
    if not image_paths:
        raise FileNotFoundError(f"No full-scene PPM images under {raw_dir}")
    annotation_files = [p for p in raw_dir.rglob("gt.txt") if p.parent in {x.parent for x in image_paths}]
    if len(annotation_files) != 1:
        raise ValueError("Expected exactly one gt.txt beside the full-scene images")
    annotations = read_gtsdb_annotations(annotation_files[0])
    unknown_files = set(annotations) - set(by_name)
    if unknown_files:
        raise ValueError(f"Annotations refer to missing images: {sorted(unknown_files)[:5]}")
    train_pool = [p for p in image_paths if int(p.stem) < 600]
    test_pool = [p for p in image_paths if int(p.stem) >= 600]
    if strict and (len(train_pool) != 600 or len(test_pool) != 300):
        raise ValueError("Historic GTSDB 600/300 split is incomplete")
    if not train_pool or not test_pool:
        raise ValueError("Need both historic training and independent test images")
    train_paths, val_paths = train_test_split(train_pool, test_size=val_fraction,
                                             random_state=seed)
    splits = {"train": train_paths, "val": val_paths, "test": test_pool}
    processed.mkdir(parents=True, exist_ok=True)
    rows, class_counts = [], Counter()
    for split, paths in splits.items():
        (processed / "images" / split).mkdir(parents=True, exist_ok=True)
        (processed / "labels" / split).mkdir(parents=True, exist_ok=True)
        for path in sorted(paths):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Cannot read {path}")
            height, width = image.shape[:2]
            output_image = processed / "images" / split / f"{path.stem}.jpg"
            if not cv2.imwrite(str(output_image), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError(f"Cannot write {output_image}")
            label_lines = []
            for x1, y1, x2, y2, label in annotations.get(path.name, []):
                if not (0 <= x1 < x2 < width and 0 <= y1 < y2 < height):
                    raise ValueError(f"Box outside {path.name}: {(x1, y1, x2, y2)}")
                # GTSDB's right/bottom coordinates are inclusive.
                x2e, y2e = x2 + 1, y2 + 1
                label_lines.append(f"0 {(x1+x2e)/(2*width):.8f} {(y1+y2e)/(2*height):.8f} "
                                   f"{(x2e-x1)/width:.8f} {(y2e-y1)/height:.8f}")
                class_counts[label] += 1
                rows.append({"split": split, "image": str(output_image.resolve()),
                             "source_image": path.name, "x1": x1, "y1": y1,
                             "x2": x2e, "y2": y2e, "class_id": label})
            (processed / "labels" / split / f"{path.stem}.txt").write_text(
                "\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
    pd.DataFrame(rows, columns=["split", "image", "source_image", "x1", "y1", "x2", "y2", "class_id"]).to_csv(
        processed / "gtsdb_annotations.csv", index=False)
    config = {"path": str(processed.resolve()), "train": "images/train",
              "val": "images/val", "test": "images/test", "names": {0: "traffic_sign"}}
    (processed / "gtsdb.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    coverage = pd.DataFrame([{"class_id": i, "class_name": CLASS_NAMES[i],
                              "gtsdb_instances": class_counts[i], "present": class_counts[i] > 0}
                             for i in range(NUM_CLASSES)])
    coverage.to_csv(processed / "class_coverage.csv", index=False)
    summary = {"source": "GTSDB FullIJCNN2013", "seed": seed,
               "image_counts": {s: len(p) for s, p in splits.items()},
               "annotated_instances": len(rows),
               "represented_class_ids": sorted(i for i, n in class_counts.items() if n),
               "absent_class_ids": [i for i in range(NUM_CLASSES) if not class_counts[i]],
               "detector_classes": ["traffic_sign"]}
    save_json(processed / "gtsdb_summary.json", summary)
    return summary


def verify_label_compatibility(gtsrb_manifest: str | Path,
                               gtsdb_annotations: str | Path, output: str | Path) -> dict:
    """Audit numeric IDs and save GTSRB/GTSDB crop pairs for semantic inspection."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rb = pd.read_csv(gtsrb_manifest)
    db = pd.read_csv(gtsdb_annotations)
    rb_train = rb[rb.split == "train"]
    if set(rb_train.class_id) != set(range(NUM_CLASSES)):
        raise ValueError("GTSRB training manifest does not cover IDs 0..42")
    if not set(db.class_id).issubset(set(rb_train.class_id)):
        raise ValueError("GTSDB contains an ID absent from GTSRB")
    rows = []
    contact = Image.new("RGB", (4 * 270, 11 * 150), "white")
    draw = ImageDraw.Draw(contact)
    for label in range(NUM_CLASSES):
        rb_row = rb_train[rb_train.class_id == label].iloc[0]
        db_part = db[db.class_id == label]
        rows.append({"class_id": label, "class_name": CLASS_NAMES[label],
                     "gtsrb_train_count": int((rb_train.class_id == label).sum()),
                     "gtsdb_count": len(db_part), "gtsdb_present": not db_part.empty})
        x, y = (label % 4) * 270, (label // 4) * 150
        draw.text((x + 3, y + 3), f"{label}: {CLASS_NAMES[label][:31]}", fill="black")
        with Image.open(rb_row.path) as source:
            rb_image = source.convert("RGB")
            rb_image.thumbnail((110, 110))
            contact.paste(rb_image, (x + 5, y + 28))
        if not db_part.empty:
            sample = db_part.iloc[0]
            with Image.open(sample.image) as source:
                crop = source.convert("RGB").crop((int(sample.x1), int(sample.y1),
                                                   int(sample.x2), int(sample.y2)))
                crop.thumbnail((110, 110))
                contact.paste(crop, (x + 135, y + 28))
        else:
            draw.text((x + 135, y + 60), "absent in GTSDB", fill="gray")
    pd.DataFrame(rows).to_csv(output / "label_compatibility.csv", index=False)
    contact.save(output / "label_compatibility_contact_sheet.png")
    summary = {"numeric_id_subset_valid": True,
               "represented_gtsdb_classes": int(sum(row["gtsdb_present"] for row in rows)),
               "semantic_check": "Review contact sheet: left GTSRB, right GTSDB crop for each class"}
    save_json(output / "label_compatibility.json", summary)
    return summary
