
# 🦋 Decoding Global Noctuidae Phenotypes via Vision Foundation Models

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![R-Spatial](https://img.shields.io/badge/R-GAMMs-276DC3.svg)](https://www.r-project.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Data](https://img.shields.io/badge/Data-GBIF-green.svg)](https://www.gbif.org/)

> **Official repository for the paper:** *"Decoding Global Macroevolutionary Patterns of Noctuidae Phenotypes: A High-Throughput Study Unveiling Dual-Track Environmental Filtering via Vision Foundation Models"*

This repository contains the code, models, and data processing pipeline for our global-scale macroecological study. By integrating Vision Foundation Models (**DINOv2**, **YOLOv10**, **CLIP**) with Spatial Generalized Additive Mixed Models (**GAMMs**), we extracted high-dimensional morphological axes from **>300,000** global Noctuidae specimens to decouple the "dual-track" mechanisms of environmental filtering. Our raw data and preprocessed dataset can be accessed at https://huggingface.co/datasets/Jaysaly/Noctuidae-pheno-env.

---

## ✨ Key Highlights

- **🌍 Massive Global Dataset:** High-throughput pipeline processing over 300,000 Noctuidae specimens worldwide.
- **🧠 VFM-Driven Phenomics:** Replaces traditional heuristics with self-supervised Vision Foundation Models (DINOv2) to extract `Dino PC1`—a robust, high-dimensional semantic axis.
- **⚙️ Automated Cleaning & Segmentation:** Employs a dual-model computer vision pipeline (CLIP for life-stage/angle classification + YOLOv10 for precise instance segmentation).
- **🧬 Dual-Track Decoupling:** Statistical framework to isolate "Phenotypic Filtering" (directional adaptation) from "Species Filtering" (taxonomic turnover) using evolutionary ablation.

---

## 📂 Repository Structure

```text
├── codes/
│   ├── 1_texture_edge.py                # Extracts texture and edge features (Pattern Complexity)
│   ├── 2_dino_extract.py                # Extracts high-dimensional DINOv2 embeddings (Dino PC1)
│   ├── 3_Pattern_characteristics.py     # Aggregates pattern characteristics
│   ├── 4_Global_Pattern_chara.py        # Processes global spatial mapping of traits
│   ├── 5_all_feature_env_GAMM.py        # Runs spatial GAMMs for environment-phenotype modeling
│   ├── 6_Extended_Factors_Plotting.py   # Generates visualizations for extended drivers
│   ├── 7_elevation_map.py               # Generates global elevation and topography overlays
│   ├── 8_1_raw_env_pics.py              # Plots raw environmental variable distributions
│   ├── 8_2_scaled_env_pics.py           # Plots standardized environmental variables
│   └── 9_genetic_effect_ablation.py     # Performs taxonomic ablation to decouple filtering tracks
├── comprehensive_statistics_summary_scaled.csv  # Standardized regression data
├── species_list.txt                             # Validated target species checklist
├── species_means_final.csv                      # Aggregated species-environment index dataset
└── README.md                                    # Project documentation
```

---

## 📊 Data Availability

The local summary datasets are provided directly in the root directory for quick statistical replication:
- `comprehensive_statistics_summary_scaled.csv`: Main matrix used for the GAMM modeling and statistical reporting.
- `species_means_final.csv`: Aggregated quantitative index connecting species records to their bioclimatic parameters.

**Full Image Dataset:** Due to GitHub's storage constraints, the complete raw and segmented image datasets (>300,000 images), along with the full environmental matrices, are hosted on HuggingFace:
👉 **[HuggingFace Dataset: Jaysaly/Noctuidae-pheno-env](https://huggingface.co/datasets/Jaysaly/Noctuidae-pheno-env)**

---

## 🚀 Usage Guide

### 1. Environment Setup

Clone the repository and install the required Python packages (PyTorch, OpenCV, scikit-learn, etc.) and R dependencies (mgcv) for spatial modeling.

```bash
git clone [https://github.com/your-username/Noctuidae-Phenomics.git](https://github.com/your-username/Noctuidae-Phenomics.git)
cd Noctuidae-Phenomics
# Install Python dependencies (requirements.txt or equivalent)
pip install -r requirements.txt
```

### 2. Execution Pipeline

The scripts in the `codes/` directory are numbered sequentially to match the analytical workflow described in the paper. 

**Part 1: Feature Extraction (Run sequentially)**
Execute scripts `1_texture_edge.py` through `3_Pattern_characteristics.py` to extract both traditional morphometrics and VFM embeddings from the segmented images.

```bash
python codes/1_texture_edge.py
python codes/2_dino_extract.py
```

**Part 2: Statistical Modeling & Ablation**
Run the GAMMs and decoupling analysis using scripts `5_all_feature_env_GAMM.py` and `9_genetic_effect_ablation.py`. Ensure the provided `.csv` data files are in the working directory.

**Part 3: Visualization & Mapping**
Scripts `6`, `7`, `8_1`, and `8_2` handle the generation of the global geographical maps, environmental gradients, and regression curve plotting.

---

## 📖 Citation

If you use this code, data, or the proposed dual-track framework in your research, please cite our paper:

we haven't get a publishment, plz contact us at kaijie.yu@zju.edu.cn

## 📄 License

This project is licensed under the **MIT License** - see the LICENSE file for details.
```
