# Traffic Sign Recognition for Autonomous Vehicles

**Group:** 9FA1-Group1

ARTI 502 final project: an Ultralytics YOLO detector finds traffic signs in road images, and a custom PyTorch CNN classifies each detected crop into one of the 43 GTSRB classes. The detector uses a single `traffic_sign` class because GTSDB does not reliably contain examples of every GTSRB class. Original GTSDB IDs are retained for integrated evaluation.

## Structure

| Path | Purpose |
| --- | --- |
| `traffic_signs/data.py` | GTSRB track split, GTSDB annotation conversion, dataset manifests |
| `traffic_signs/cnn.py` | CNN architecture, training, evaluation, curves, confusion matrix |
| `traffic_signs/yolo.py` | YOLO fine-tuning and detection evaluation |
| `traffic_signs/inference.py` | YOLO crop CNN pipeline and end-to-end evaluation |
| `traffic_signs/experiments.py` | Hyperparameter, augmentation, and feature-selection experiments |
| `scripts/` | Command-line entry points for preparing data, training, evaluation, inference, and the demo |
| `notebooks/Traffic_Sign_Recognition_Colab.ipynb` | Numbered Google Colab workflow |
| `data/raw/`, `data/processed/`, `outputs/` | Local datasets, conversions, and generated results (Git ignored) |

## Datasets and split policy

- **GTSRB:** `torchvision.datasets.GTSRB` downloads the 26,640-image `GTSRB-Training_fixed` archive used in the 2011 competition and the 12,630-image official final test archive. The competition training archive is a subset of the later full final training release. Training images are split into train and validation by *sign track* within each class, so adjacent images of one physical sign do not cross that boundary. The official test images are kept untouched until final evaluation.
- **GTSDB:** Download `FullIJCNN2013.zip` from the [official GTSDB archive](https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/published-archive.html). Extract it to `data/raw/gtsdb/FullIJCNN2013/`, containing `00000.ppm` through `00899.ppm` and `gt.txt`. Images 00000–00599 form the historical training pool; a seeded image-level 15% subset becomes validation. Images 00600–00899 are independent test images. This includes road images with no annotated signs; their YOLO labels are empty.
- `data/processed/gtsdb_yolo/class_coverage.csv` lists each original class ID and its count. The YOLO labels are all class 0 (`traffic_sign`), while `gtsdb_annotations.csv` retains the original IDs. Only annotated GTSDB signs are eligible for the 43-class integrated score. Unannotated or out-of-scope signs may still appear in road scenes.
- `data/processed/label_compatibility.csv` verifies that annotated IDs are contained in the GTSRB taxonomy. Its contact sheet places a GTSRB sample beside a GTSDB crop for every represented ID so the semantic mapping can be inspected visually. Empty GTSDB classes are marked absent.

The official benchmark descriptions are [GTSRB](https://benchmark.ini.rub.de/gtsrb_dataset.html) and [GTSDB](https://benchmark.ini.rub.de/gtsdb_dataset.html). The [torchvision GTSRB loader](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.GTSRB.html) provides the downloads. YOLO training and evaluation use the [Ultralytics Python API](https://docs.ultralytics.com/usage/python).

## Run in Google Colab

1. Upload the full project folder to `MyDrive/DeepLearning_Poject`, or set `GITHUB_URL` and `PROJECT_ROOT` in notebook section 1.
2. Place the official `FullIJCNN2013.zip` at `data/raw/FullIJCNN2013.zip`, or extract it as above. Allow several GB for archive, source images, converted JPGs, and models.
3. Open [`notebooks/Traffic_Sign_Recognition_Colab.ipynb`](notebooks/Traffic_Sign_Recognition_Colab.ipynb) in Colab, select a GPU runtime, and execute cells from top to bottom. The Drive mount is optional (`USE_DRIVE`).
4. The training and experiment cells may take substantial time. If Colab disconnects, the files under `outputs/` remain on Drive when the project is there. Results shown in the notebook come from the corresponding saved CSV, JSON, and image files.

## Run locally

Use Python 3.10–3.12, preferably with a CUDA GPU. From the project root:

```bash
python -m venv .venv
# activate .venv using your operating system's command
python -m pip install -r requirements.txt
python -m scripts.prepare_datasets
python -m scripts.train_cnn
python -m scripts.evaluate_cnn
python -m scripts.train_yolo
python -m scripts.evaluate_yolo
python -m scripts.evaluate_integrated
python -m scripts.predict path/to/road_image.jpg
python -m scripts.demo
```

Run optional validation experiments after the main system works:

```bash
python -m scripts.run_experiments cnn-tuning
python -m scripts.run_experiments yolo-tuning --epochs 6
python -m scripts.run_experiments augmentation
python -m scripts.run_experiments features
```

All scripts accept `--help`. CNN and YOLO training can use different batch sizes and image sizes. The experiments are intentionally small: three CNN configurations, three YOLO configurations, two augmentation settings, and four CNN embedding sizes. These are **experimental runs**, not claimed performance results.

## Generated outputs

- CNN: `outputs/cnn/best.pt`, `history.csv`, `training_curves.png`, and evaluation JSON/CSV/confusion images.
- YOLO: `outputs/yolo/train/weights/best.pt`, Ultralytics plots and `results.csv`, test metrics JSON/CSV, and detection examples.
- Integrated system: `outputs/integrated/integrated_metrics.json`, `integrated_predictions.csv`, and annotated examples.
- Experiments: validation comparisons, plots, best settings JSON, feature-selection test metrics, and augmentation ablation under `outputs/experiments/`.

No metric or checkpoint is included before an actual run. CNN accuracy, macro precision/recall/F1, and YOLO mAP measure different tasks and should be reported separately. The integrated metrics use one-to-one matching at IoU 0.5 and report correct class predictions relative to all ground-truth signs and all predicted boxes.

## Reproducibility and limitations

The default random seed is 42, and PyTorch deterministic cuDNN settings are enabled. GPU kernels, package versions, and Colab hardware may still cause minor differences. GTSDB has only 900 road images and sparse coverage of the 43 classes. A CNN trained on isolated GTSRB signs may perform less well on small, blurred detection crops. The demo is a research prototype and is not suitable for vehicle control.
