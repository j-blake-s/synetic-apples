# ApplesM5
## Breaking the Bottleneck: Synthetic Data as the New Foundation for Vision AI

This repository contains training images and scripts for the Synetic AI **ApplesM5** object detection project that was used in the **Breaking the Bottleneck: Synthetic Data as the New Foundation for Vision AI** white paper, using **Ultralytics YOLO12**. The core scripts allow you to train models using custom YAML datasets and evaluate the results using the provided train_metrics.py script.

The paper is available for download at https://synetic.ai/white-paper/breaking/benchmark .

---

## 📂 Repository Structure (Key Files)

| File                       | Purpose                                                                                     |
| -------------------------- | ------------------------------------------------------------------------------------------- |
| `applesm5-train-det.py`    | Trains YOLO12 detection models using specified datasets and hyperparameters.                |
| `FileCrawler.py`           | Recursively crawls directories to find image and label files. Used for evaluating datasets. |
| `train_metrics.py`         | Runs evaluations on trained YOLO12 models and computes mAP, precision, and recall metrics.  |
| `*.yaml` (dataset configs) | Define dataset splits, including training, validation, and test image directories.          |

---

## ⚙️ Setup

### 1. Install Dependencies

```bash
pip install ultralytics tqdm
```

Your environment should have **PyTorch** and GPU drivers properly configured.

---

## 🚀 Usage

### A. Training Models (`applesm5-train-det.py`)

To train object detection models using YOLO12:

```bash
python applesm5-train-det.py
```

Key things to configure:

- Edit the `dataNames` list to point to your dataset YAML files (e.g., `real`, `synetic+real`, etc.).
- YAML files should be placed at `/home/user/datasets/ApplesM5/`.
- Adjust `hyperparams`, `epochs`, and GPU `devices` as needed.
- The script trains multiple model variants (`yolo12n.yaml`, etc.) and saves results to the Ultralytics default `runs/detect/` folder.

---

### B. Dataset YAML Files

Example dataset YAML (`real.yaml`):

```yaml
path: /path/to/your/dataset
train: images/train
val: images/val
test: images/test
names:
  0: apple
```

Modify the paths in your YAML files to point to your dataset locations.

---

### C. Evaluating Models (`train_metrics.py`)

After training, you can evaluate your models on a validation dataset:

```bash
python train_metrics.py
```

Make sure to adjust the following in the script:

- `modelPaths`: list of trained YOLO12 model `.pt` files to evaluate.
- `pathValsDataset`: path to your validation images (`.png`/`.jpg`).

This will compute **mAP50**, **mAP50-95**, **precision**, and **recall** scores and print them to the console.

---

## ✅ Example Workflow

1. Prepare datasets and YAML config files.
2. Train detection models with `applesm5-train-det.py`.
3. Run `train_metrics.py` to benchmark models.
4. Iterate on your datasets and training parameters to improve performance.

---

## 🔧 Notes

- The training script assumes a multi-GPU setup (adjust the `devices` list if needed).
- The repo is tuned for an NVIDIA DGX or similar system with 8 GPUs but can be modified for single-GPU setups.
- Dataset YAML and trained model `.pt` files follow the **Ultralytics YOLO12** conventions.

---
