# Algorithmic Fairness in Machine Learning

IEEE-style research paper investigating bias and fairness across four real-world ML classification tasks. Analyses how demographic disparities manifest in model predictions and evaluates mitigation strategies.

## Overview

Four datasets, four fairness challenges:
- **Adult Census** — income prediction, protected attribute: sex & race
- **COMPAS** — recidivism risk scoring, protected attribute: race
- **Bank Marketing** — credit subscription, protected attribute: education & gender
- **Credit Default** — loan default prediction, protected attribute: marital status & age

Each model (M1–M4) is evaluated on both standard performance metrics (AUC, accuracy, F1) and fairness metrics (demographic parity, disparate impact, equalised odds).

## Tech Stack
- Python 3.x
- Scikit-learn
- Pandas / NumPy
- Matplotlib / Seaborn

## Structure
```
notebooks/     # Jupyter analysis notebooks
data/          # Datasets (COMPAS, Adult, Bank Marketing, Credit Default) + results CSVs
paper/         # IEEE research paper (docx), slides (pdf), result figures
```

## Key Results
See `paper/figures/` for fairness metric heatmaps and performance comparisons across all four models and demographic groups.

## Run
```bash
pip install scikit-learn pandas numpy matplotlib seaborn jupyter
jupyter notebook notebooks/
```

---
MSc Artificial Intelligence — NCI Dublin, 2025
