# Algorithmic Fairness in ML

Four real-world datasets. Four different fairness problems. The question throughout: how much does a model's accuracy actually tell you when it's systematically wrong for some groups and not others?

We trained classifiers on Adult Census, COMPAS, Bank Marketing, and Credit Default, then measured both standard performance metrics and fairness metrics — demographic parity, equalised odds, disparate impact — broken down by the protected attribute in each dataset.

## What we found

**Adult Census (M1) — income prediction**  
XGBoost hit 87.9% accuracy and AUC 0.933, but the fairness numbers tell a different story. Men were predicted as high earners at nearly 3x the rate of women (26% vs 8%). The disparity was consistent across all four models, not just XGBoost — the problem is in the data, not the algorithm choice.

**COMPAS (M3) — recidivism risk**  
The most striking results. African-American defendants were flagged as high-risk at roughly double the rate of Caucasian defendants across every model (XGBoost: 44% vs 25%). Equalized odds difference of 0.24. This replicates the original ProPublica findings on a different model set, which at least confirms the methodology is picking up real disparities.

**Bank Marketing (M2) — credit subscription**  
Fairness gaps were present but smaller than M1 and M3. Education level was the bigger driver than gender for predicted outcomes.

**Credit Default (M4) — loan default**  
The most balanced across groups. Disparate impact ratios were around 0.75–0.78 for marital status — not clean, but noticeably better than the other three datasets.

The overall pattern: model choice matters less than dataset choice. The same algorithms produced very different fairness profiles depending on what the data encoded.

| Dataset | Protected attr | Demographic parity diff (XGBoost) | Disparate impact ratio |
|---|---|---|---|
| Adult Census | Sex | 0.179 | 0.310 |
| Adult Census | Race | 0.100 | 0.533 |
| COMPAS | Race | 0.196 | 0.557 |
| Credit Default | Marital status | 0.013 | 0.755 |

## Structure

```
notebooks/     — one notebook per dataset (M1–M4) + group synthesis
data/          — raw datasets + results CSVs per model/attribute
paper/         — IEEE paper, slides, figures
```

## Notebooks

```
member1_adult_income.ipynb       — M1: Adult Census
member2_credit_default.ipynb     — M2: Credit Default
member3_compas_recidivism.ipynb  — M3: COMPAS
member4_bank_marketing.ipynb     — M4: Bank Marketing
group_synthesis.ipynb            — combined analysis and comparison
```

## Stack

Python 3, scikit-learn, pandas, numpy, matplotlib, seaborn

```bash
pip install scikit-learn pandas numpy matplotlib seaborn jupyter
jupyter notebook notebooks/
```

---

MSc Artificial Intelligence, NCI Dublin 2025  
Group project — Joshua Thomas, et al.
