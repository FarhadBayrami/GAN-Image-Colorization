<div align="center">

# 🎨 Attribute-Conditioned Sketch Colorization
### A Paired Dataset for GAN-Based Character Colorization with Hair & Shirt Attribute Control

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="https://img.shields.io/badge/Resolution-512x512-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Combinations-25%20per%20sketch-green?style=flat-square"/>
  <img src="https://img.shields.io/badge/Task-Conditional%20Image%20Translation-orange?style=flat-square"/>
</p>

*Given a black-and-white character sketch plus a target hair color and shirt color, the model learns to produce the fully colored character. This repository contains the paired dataset and its preparation pipeline.*

</div>

---

<div align="center">
  <img src="assets/preview.png" alt="Dataset preview — one input sketch and four colored variants" width="700"/>
</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Color Palette](#-color-palette)
- [Getting Started](#-getting-started)
- [Regenerating the Dataset](#-regenerating-the-dataset)
- [Intended Use](#-intended-use)
- [Project Structure](#-project-structure)
- [Citation](#-citation)
- [Author](#-author)

---

## 🔬 Overview

**Attribute-conditioned sketch colorization** is the task of coloring a black-and-white character sketch according to specified categorical attributes — here, a target **hair color** and **shirt color**.

This dataset is designed for conditional image-to-image translation experiments such as **pix2pix-style GANs** or **conditional diffusion models**, where the condition is the sketch plus an attribute pair. Its small size and fixed palette make it a fast, controlled benchmark for studying attribute conditioning.

---

## 📦 Dataset

| Split | Sketches (inputs) | Colored images (targets) | Resolution | Format |
|-------|-------------------|--------------------------|------------|--------|
| Train | 10 | 250 | 512 × 512 | BMP (RGB) |
| Test  | 4  | 100 | 512 × 512 | BMP (RGB) |

Each sketch is colored in all **25 combinations** of 5 hair colors × 5 shirt colors, so every input maps to 25 targets. The pairing and attribute labels live in `metadata.csv` (one per split):

| Column | Meaning | Example |
|--------|---------|---------|
| `input`  | Sketch filename in `inputs/` | `0001.bmp` |
| `target` | Colored image filename in `targets/` | `0003.bmp` |
| `hair`   | Hair color of the target | `blue` |
| `shirt`  | Shirt color of the target | `red` |

---

## 🎨 Color Palette

**Variable colors** (hair and shirt):

| Color | Hex |
|-------|-----|
| Blue | `#3FAEE1` |
| Brown | `#BA6B4B` |
| Red | `#E84744` |
| Yellow | `#FBD04F` |
| Green | `#6FBB84` |

**Fixed colors** across all images: skin `#FBD3C2`, cheeks `#F39E9C`, line art `#000000`.

The full palette is in [`raw/palette.txt`](raw/palette.txt).

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/FarhadBayrami/GAN-Image-Colorization.git
cd GAN-Image-Colorization

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 🔄 Regenerating the Dataset

The `processed/` folder (~275 MB of uncompressed BMPs) is **not committed** — it is fully reproducible from `raw/` in a few seconds, keeping the repository small and fast to clone.

```bash
mkdir -p processed/train/inputs processed/train/targets processed/test/inputs processed/test/targets
jupyter notebook dataset-preparation.ipynb
```

Run the notebook twice: once with `phase = 'train'` and once with `phase = 'test'` (first cell). For each phase the pipeline:

1. Composites every RGBA source PNG onto a white background (alpha blending) and saves it as sequentially numbered BMP (`0001.bmp`, `0002.bmp`, …)
2. Flattens the per-sketch target folders into a single numbered `targets/` directory
3. Generates `metadata.csv` linking each target to its input sketch and its hair/shirt color attributes

---

## 🎯 Intended Use

This dataset is designed for **conditional image-to-image translation** experiments:
- pix2pix-style conditional GANs
- Conditional diffusion models
- Attribute-conditioning studies

The small size and fixed palette make it a fast, controlled benchmark for research and teaching.

---

## 📁 Project Structure

| Path | Description |
|------|-------------|
| `dataset-preparation.ipynb` | Pipeline: raw PNGs → processed BMPs + metadata.csv |
| `raw/palette.txt` | Full color palette definition |
| `raw/train/inputs/` | 10 source sketches (RGBA PNG) |
| `raw/train/targets/` | One folder per sketch, 25 colored PNGs each |
| `raw/test/` | Same layout, 4 sketches |
| `processed/` | Generated dataset (git-ignored, reproducible from `raw/`) |
| `assets/` | README images |
| `requirements.txt` | Python dependencies |
| `LICENSE` | MIT License |
| `CITATION.cff` | How to cite this work |
| `README.md` | Project documentation |

---

## 📚 Citation

If you use this dataset in your work, please cite:

| Field | Value |
|-------|-------|
| **Author** | Farhad Bayrami |
| **Title** | Attribute-Conditioned Sketch Colorization Dataset |
| **Year** | 2025 |
| **URL** | github.com/FarhadBayrami/GAN-Image-Colorization |

---

## 👤 Author

**Farhad Bayrami**
Machine Learning Engineer · MSc in Artificial Intelligence
📧 [farhad.bayrami@studio.unibo.it](mailto:farhad.bayrami@studio.unibo.it)
🔗 [GitHub](https://github.com/FarhadBayrami)

</div>