# Full Results Reference

This file contains every metric produced across all four datasets, all four classifiers, and all eight protected-attribute evaluations. It is intended as a downloadable reference for anyone who wants to work with the numbers without running the notebooks.

For the interpretive summary, see [README.md](README.md).

---

## Definitions

| Symbol | Metric | Notes |
|---|---|---|
| DPD | Demographic Parity Difference | Difference in positive prediction rates between reference and comparison group. Ideal = 0. |
| EOD | Equalised Odds Difference | Max difference in TPR and FPR across groups. Ideal = 0. |
| DI | Disparate Impact Ratio | Ratio of positive prediction rates (comparison / reference). Ideal = 1.0; below 0.8 = regulatory concern. |
| κ | Cohen's Kappa | Agreement beyond chance. Accounts for class imbalance better than raw accuracy. |

**Models:** LR = Logistic Regression · DT = Decision Tree · RF = Random Forest · XGB = XGBoost

**Shared training setup:** RANDOM_STATE = 42 · Stratified 80/20 split · GridSearchCV · StratifiedKFold(n_splits=5)

---

## M1 — Adult Census Income

**Task:** Predict whether annual income exceeds $50,000.  
**Source:** UCI ML Repository — 1994 US Census data.  
**Rows:** ~48,842 (train + test combined).  
**Class balance:** ~76% ≤$50k / ~24% >$50k.

### Standard Performance

| Model | Accuracy | Precision | Sensitivity | Specificity | F1 Macro | F1 Weighted | AUC-ROC | Cohen κ |
|---|---|---|---|---|---|---|---|---|
| XGBoost | **0.8787** | **0.7961** | 0.6630 | **0.9466** | **0.8229** | **0.8747** | **0.9326** | **0.6466** |
| Random Forest | 0.8657 | 0.8021 | 0.5825 | 0.9548 | 0.7951 | 0.8578 | 0.9184 | 0.5930 |
| Decision Tree | 0.8622 | 0.7874 | 0.5813 | 0.9506 | 0.7909 | 0.8546 | 0.9042 | 0.5843 |
| Logistic Regression | 0.8517 | 0.7324 | **0.5992** | 0.9311 | 0.7822 | 0.8463 | 0.9058 | 0.5656 |

### Fairness — Sex (Reference: Male, Comparison: Female)

| Model | +Rate Reference | +Rate Comparison | DPD | EOD | DI |
|---|---|---|---|---|---|
| Logistic Regression | 0.2566 | 0.0760 | 0.1806 | 0.1226 | 0.2962 |
| Decision Tree | 0.2273 | 0.0769 | 0.1504 | 0.0592 | 0.3384 |
| Random Forest | 0.2265 | 0.0699 | 0.1566 | 0.0834 | 0.3087 |
| XGBoost | 0.2596 | 0.0806 | 0.1790 | 0.1077 | 0.3104 |

All four models: DI well below 0.8. Sex is the most severe protected-attribute disparity in this study.

### Fairness — Race (Reference: White, Comparison: Non-White)

| Model | +Rate Reference | +Rate Comparison | DPD | EOD | DI |
|---|---|---|---|---|---|
| Logistic Regression | 0.2098 | 0.1140 | 0.0957 | 0.0919 | 0.5435 |
| Decision Tree | 0.1881 | 0.1098 | 0.0783 | 0.0418 | 0.5837 |
| Random Forest | 0.1845 | 0.1112 | 0.0733 | 0.0382 | 0.6027 |
| XGBoost | 0.2138 | 0.1140 | 0.0998 | 0.0866 | 0.5332 |

Random Forest produces the best racial DI ratio (0.603) but still falls below 0.8.

---

## M2 — Bank Marketing

**Task:** Predict whether a client will subscribe to a bank term deposit.  
**Source:** UCI ML Repository — Portuguese bank telemarketing campaign.  
**Rows:** 41,188.  
**Class balance:** ~89% no / ~11% yes (heavily imbalanced).

### Standard Performance

| Model | Accuracy | Precision | Sensitivity | Specificity | F1 Macro | F1 Weighted | AUC-ROC | Cohen κ |
|---|---|---|---|---|---|---|---|---|
| XGBoost | 0.8175 | 0.6625 | 0.3564 | **0.9484** | **0.6768** | **0.7957** | **0.7784** | **0.3653** |
| Decision Tree | **0.8177** | **0.6639** | 0.3557 | 0.9489 | 0.6767 | 0.7957 | 0.7424 | 0.3652 |
| Random Forest | 0.8165 | 0.6592 | 0.3527 | 0.9482 | 0.6745 | 0.7944 | 0.7698 | 0.3610 |
| Logistic Regression | 0.6795 | 0.3671 | **0.6202** | 0.6963 | 0.6166 | 0.7032 | 0.7081 | 0.2539 |

Logistic Regression trades specificity for sensitivity — useful if recall on the minority class matters, but overall weaker on AUC.

### Fairness — Education (Reference: Higher Education, Comparison: Lower/Other)

| Model | +Rate Reference | +Rate Comparison | DPD | EOD | DI |
|---|---|---|---|---|---|
| Logistic Regression | 0.3770 | 0.3586 | 0.0185 | 0.1080 | 0.9510 |
| Decision Tree | 0.1158 | 0.1305 | −0.0147 | 0.0405 | 1.1265 |
| Random Forest | 0.1138 | 0.1387 | −0.0249 | 0.0264 | 1.2188 |
| XGBoost | 0.1148 | 0.1378 | −0.0230 | 0.0252 | 1.2001 |

The negative DPD and DI > 1 for tree-based models indicate lower-education groups receive higher predicted subscription rates — a counterintuitive result that warrants investigation of interaction effects with job type and contact history.

### Fairness — Gender (Reference: Male, Comparison: Female)

| Model | +Rate Reference | +Rate Comparison | DPD | EOD | DI |
|---|---|---|---|---|---|
| Logistic Regression | 0.4546 | 0.3196 | 0.1350 | 0.1398 | 0.7031 |
| Decision Tree | 0.1245 | 0.1145 | 0.0100 | 0.0078 | 0.9199 |
| Random Forest | 0.1257 | 0.1134 | 0.0123 | 0.0119 | 0.9019 |
| XGBoost | 0.1257 | 0.1145 | 0.0112 | 0.0122 | 0.9108 |

Logistic Regression shows a 0.703 DI for gender, while tree-based models cluster near 0.91 — the largest within-dataset spread across model families in the study.

---

## M3 — COMPAS Recidivism

**Task:** Predict whether a defendant recidivates within two years of release.  
**Source:** ProPublica COMPAS analysis (2016).  
**Rows:** 7,214.  
**Class balance:** ~55% recidivist / ~45% non-recidivist.

### Standard Performance

| Model | Accuracy | Precision | Sensitivity | Specificity | F1 Macro | F1 Weighted | AUC-ROC | Cohen κ |
|---|---|---|---|---|---|---|---|---|
| XGBoost | **0.7020** | **0.7100** | 0.5836 | **0.8009** | **0.6931** | **0.6978** | **0.7362** | **0.3902** |
| Decision Tree | 0.6955 | 0.6775 | **0.6317** | 0.7489 | 0.6910 | 0.6944 | 0.7303 | 0.3827 |
| Logistic Regression | 0.6858 | 0.6611 | 0.6352 | 0.7281 | 0.6821 | 0.6852 | 0.7315 | 0.3645 |
| Random Forest | 0.6850 | 0.6935 | 0.5516 | 0.7964 | 0.6741 | 0.6795 | 0.7284 | 0.3540 |

Models cluster within ~2 percentage points of AUC — no single classifier dominates on COMPAS.

### Fairness — Race (Reference: African-American + Other, Comparison: Caucasian)

| Model | +Rate Reference | +Rate Comparison | DPD | EOD | DI |
|---|---|---|---|---|---|
| Logistic Regression | 0.5136 | 0.2927 | 0.2209 | 0.2461 | 0.5700 |
| Decision Tree | 0.5012 | 0.2787 | 0.2225 | 0.2759 | 0.5560 |
| Random Forest | 0.4307 | 0.2319 | 0.1988 | 0.2421 | 0.5383 |
| XGBoost | 0.4418 | 0.2459 | 0.1959 | 0.2434 | 0.5566 |

Racial bias in recidivism prediction is consistent across all classifiers. Random Forest shows the smallest gap but still reaches only DI = 0.538.

### Fairness — Sex (Reference: Male, Comparison: Female)

| Model | +Rate Reference | +Rate Comparison | DPD | EOD | DI |
|---|---|---|---|---|---|
| Logistic Regression | 0.4691 | 0.3004 | 0.1686 | 0.1320 | 0.6405 |
| Decision Tree | 0.4611 | 0.2661 | 0.1950 | 0.1759 | 0.5771 |
| Random Forest | 0.3912 | 0.2361 | 0.1552 | 0.1224 | 0.6034 |
| XGBoost | 0.4082 | 0.2275 | 0.1807 | 0.1748 | 0.5573 |

---

## M4 — Credit Card Default

**Task:** Predict whether a credit card holder will default next month.  
**Source:** UCI ML Repository — Taiwan credit data, October 2005.  
**Rows:** 30,000.  
**Class balance:** ~78% no default / ~22% default.

### Standard Performance

| Model | Accuracy | Precision | Sensitivity | Specificity | F1 Macro | F1 Weighted | AUC-ROC | Cohen κ |
|---|---|---|---|---|---|---|---|---|
| Decision Tree | **0.9028** | 0.6862 | 0.2522 | 0.9854 | 0.6581 | **0.8822** | 0.7910 | 0.3281 |
| Random Forest | 0.9020 | **0.6921** | 0.2349 | **0.9867** | 0.6489 | 0.8799 | **0.8114** | 0.3115 |
| XGBoost | 0.9013 | 0.6611 | 0.2543 | 0.9834 | **0.6569** | 0.8812 | 0.8090 | **0.3251** |
| Logistic Regression | 0.8341 | 0.3664 | **0.6487** | 0.8576 | 0.6850 | 0.8529 | 0.8018 | **0.3789** |

High accuracy reflects the class imbalance, not balanced classification quality. Logistic Regression's lower accuracy comes with far higher sensitivity.

### Fairness — Age (Reference: Older ≥40, Comparison: Younger <40) ⚠️

| Model | +Rate Reference | +Rate Comparison | DPD | EOD | DI |
|---|---|---|---|---|---|
| Logistic Regression | 0.8197 | 0.1805 | 0.6392 | 0.5857 | 0.2202 |
| Decision Tree | 0.2541 | 0.0349 | 0.2192 | 0.1719 | 0.1374 |
| Random Forest | 0.2295 | 0.0324 | 0.1971 | 0.1911 | 0.1412 |
| XGBoost | 0.2910 | 0.0358 | 0.2552 | 0.2392 | **0.1230** |

**The most extreme disparity in the study.** All four models assign older borrowers default probabilities between 7× and 8× higher than younger borrowers. Logistic Regression's DPD of 0.639 is the largest single demographic parity difference observed.

### Fairness — Marital Status (Reference: Single/Divorced, Comparison: Married)

| Model | +Rate Reference | +Rate Comparison | DPD | EOD | DI |
|---|---|---|---|---|---|
| Logistic Regression | 0.2314 | 0.1785 | 0.0529 | 0.0406 | 0.7714 |
| Decision Tree | 0.0478 | 0.0372 | 0.0106 | 0.0054 | 0.7778 |
| Random Forest | 0.0469 | 0.0326 | 0.0143 | 0.0186 | 0.6945 |
| XGBoost | 0.0509 | 0.0384 | 0.0125 | 0.0140 | 0.7547 |

Marital status disparities are more modest than age disparities but all four models still fall below 0.8 for at least one model.

---

## Cross-Dataset Summary

### Best AUC-ROC per Dataset

| Dataset | Best Model | AUC-ROC |
|---|---|---|
| Adult Income (M1) | XGBoost | **0.933** |
| Bank Marketing (M2) | XGBoost | 0.778 |
| COMPAS (M3) | XGBoost | 0.736 |
| Credit Default (M4) | Random Forest | 0.811 |

### Lowest Disparate Impact Ratio per Evaluation

| Dataset | Attribute | Model | DI Ratio |
|---|---|---|---|
| Adult Income | Sex | Logistic Regression | 0.296 |
| Adult Income | Race | XGBoost | 0.533 |
| Bank Marketing | Education | Logistic Regression | 0.951 (closest to fair) |
| Bank Marketing | Gender | Logistic Regression | 0.703 |
| COMPAS | Race | Random Forest | 0.538 |
| COMPAS | Sex | XGBoost | 0.557 |
| Credit Default | Age | XGBoost | **0.123** (most extreme) |
| Credit Default | Marital | Random Forest | 0.695 |

XGBoost achieves the best AUC in three of four datasets and the worst disparate impact ratio in two of four evaluations — a direct illustration of the performance–fairness tradeoff.
