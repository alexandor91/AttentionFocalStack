# 🌌 FocDepthFormer

**Transformer with Latent LSTM for Depth Estimation from Focal Stack**

---

## 🧠 Abstract

Depth estimation is a fundamental problem in computer vision with wide-ranging applications in **augmented reality**, **robotics**, and **autonomous driving**.

This repository provides an end-to-end **PyTorch implementation** for **focal stack–based depth estimation**, based on our paper:

> *“FocDepthFormer: Transformer with Latent LSTM for Depth Estimation from Focal Stack.”*

### Key Contributions:

* 💡 **Vision Transformer backbone** enhanced with **Latent LSTM fusion** for multi-focus information integration
* 🧩 **Sequential focal stack dataset support** (not just single image inputs)
* ⚙️ **Unified training and evaluation** pipelines with **wandb** integration
* 👁️‍🗨️ Designed for **auto-focus and in-the-wild scenarios**

---

## ⚡ Web Demo

A demo is hosted on **Hugging Face Spaces**, powered by **Gradio**, allowing users to upload a focal stack and visualize the estimated depth map in real-time.

---

## 📦 Requirements

Install dependencies via:

```bash
pip install -r requirements.txt
```

### Core Dependencies:

* `torch`
* `torchvision`
* `wandb`
* `scipy`
* `numpy`
* `Pillow`
* `matplotlib`
* `tqdm`

---

## 🏗️ Project Structure

```
├── FOD/
│   ├── models/                # Transformer + LSTM architecture
│   ├── dataset_focalstack.py  # Dataset loader for sequential focal stacks
│   ├── Trainer.py             # Training and validation logic
│   ├── losses.py              # Scale-Shift-Invariant and gradient losses
│   └── api.py                 # Dataset utilities
├── config.json                # Configuration for dataset and model setup
├── train_focalstack.py        # Training script for focal stack model
├── run.py                     # Inference / testing script
└── README.md
```

---

## 🏦 Model Zoo

| Model Name                            | Description                        | Link          |
| ------------------------------------- | ---------------------------------- | ------------- |
| `FocusOnDepth_vit_base_patch16_384.p` | Vision Transformer + LSTM backbone | *Coming soon* |

Place pretrained weights in:

```
models/
```

Then, edit `config.json` to select the corresponding model type (e.g., `"type": "depth"`).

---

## 🎯 Running a Prediction

1️⃣ Place your **input focal stacks** (multiple `.png` or `.jpg` per scene) into:

```
input/
```

2️⃣ Run inference:

```bash
python run.py
```

3️⃣ The generated **depth maps** and optional **segmentation masks** will appear in:

```
output/
```

---

## 🔨 Training

### 🧩 Build the Dataset

This project supports training with mixed datasets:

* [Inria Movie 3D Dataset (Kaggle)](https://www.kaggle.com/datasets)
* [NYU Depth v2 Dataset (Kaggle)](https://www.kaggle.com/datasets)
* [PoseTrack Dataset (Kaggle)](https://www.kaggle.com/datasets)

Each sample should include:

* Sequential focal stack images
* Corresponding depth ground truth

---

### ⚙️ Configure `config.json`

The `config.json` defines:

* Dataset paths
* Data augmentations (resize, crop, rotation)
* Training hyperparameters (batch size, learning rate, epochs)

Example:

```json
{
  "General": {
    "seed": 42,
    "batch_size": 4
  },
  "Dataset": {
    "paths": {
      "path_dataset": "./data",
      "path_images": "images",
      "path_depths": "depths"
    },
    "transforms": {
      "resize": 384,
      "p_flip": 0.5,
      "p_crop": 0.3,
      "p_rot": 0.2
    }
  }
}
```

---

### 🚀 Start Training

Run:

```bash
python train_focalstack.py
```

* Uses `AutoFocusStackDataset` for sequential focal images
* Logs metrics and visualizations to **Weights & Biases (wandb)**

To verify input structure, set:

```json
"debug_batch_shape": true
```

in your config.

---

## 🧩 Key Features

✅ **Sequential focal stack input**
✅ **Latent LSTM encoder** for temporal-style integration
✅ **Scale & Shift Invariant Loss (SSI)**
✅ **Gradient smoothness regularization**
✅ **Supports multi-task outputs (depth + segmentation)**
✅ **Multi-dataset compatibility**

---

## 📊 Evaluation

Evaluate trained models using:

```bash
python evaluate.py
```

Metrics include **MAE**, **RMSE**, and **δ-accuracy**.

---

## 📚 Citation

If you use this repository, please cite our work:

```bibtex
@inproceedings{kang2024focdepthformer,
  title={FocDepthFormer: Transformer with Latent LSTM for Depth Estimation from Focal Stack},
  author={Kang, Xueyang and Han, Fengze and Fayjie, Abdur R and Vandewalle, Patrick and Khoshelham, Kourosh and Gong, Dong},
  booktitle={Australasian Joint Conference on Artificial Intelligence},
  pages={273--290},
  year={2024},
  organization={Springer}
}
```

---

## 🙏 Acknowledgements

Our work extends ideas from **Ranftl et al. (MiDaS: Towards Robust Monocular Depth Estimation)** and **FocusOnDepth code implementation**.
Special thanks to open-source contributors and dataset providers who made this research possible.

---

## 📧 Contact

**Author:** Xueyang (Alex) Kang
**Email:** xueyang.kang [at] unimelb.edu.au
**Affiliations:** KU Leuven – University of Melbourne

---

> “Focus beyond what the eye can see — learn depth from focus.”

