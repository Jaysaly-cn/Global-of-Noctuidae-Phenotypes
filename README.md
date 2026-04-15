# 🦋 Decoding Global Noctuidae Phenotypes via Vision Foundation Models

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![R-Spatial](https://img.shields.io/badge/R-GAMMs-276DC3.svg)](https://www.r-project.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Data](https://img.shields.io/badge/Data-GBIF-green.svg)](https://www.gbif.org/)

> **Official repository for the paper:** *"Decoding Global Macroevolutionary Patterns of Noctuidae Phenotypes: A High-Throughput Study Unveiling Dual-Track Environmental Filtering via Vision Foundation Models"*

This repository contains the code, models, and data processing pipeline for our global-scale macroecological study. By integrating Vision Foundation Models (**DINOv2**, **YOLOv10**, **CLIP**) with Spatial Generalized Additive Mixed Models (**GAMMs**), we extracted high-dimensional morphological axes from **>300,000** global Noctuidae specimens to decouple the "dual-track" mechanisms of environmental filtering.

---

## ✨ Key Highlights

- **🌍 Massive Global Dataset:** High-throughput pipeline processing over 300,000 Noctuidae specimens worldwide.
- **🧠 VFM-Driven Phenomics:** Replaces traditional heuristics with self-supervised Vision Foundation Models (DINOv2) to extract `Dino PC1`—a robust, high-dimensional semantic axis.
- **⚙️ Automated Cleaning & Segmentation:** Employs a dual-model computer vision pipeline (CLIP for life-stage/angle classification + YOLOv10 for precise instance segmentation).
- **🧬 Dual-Track Decoupling:** Statistical framework to isolate "Phenotypic Filtering" (directional adaptation) from "Species Filtering" (taxonomic turnover) using evolutionary ablation.

---
