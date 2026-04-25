# Notebooks

Five Jupyter notebooks make up this project. Four member notebooks each handle one dataset; the fifth synthesises results across all four.

---

## Run Order

```
1. member1_adult_income.ipynb
2. member2_credit_default.ipynb
3. member3_compas_recidivism.ipynb
4. member4_bank_marketing.ipynb
5. group_synthesis.ipynb
```

The synthesis notebook reads the CSVs that the four member notebooks export to `data/results/`. Run the members first.

---

## Notebook Summaries

### `member1_adult_income.ipynb` — M1: Adult Census Income

**Dataset:** 1994 US Census, ~48,842 rows. Predict income > $50k.  
**Protected attributes:** Sex (Male/Female), Race (White/Non-White)  
**Reference implementation** for the group contract — use this as a template for section structure, metric export format, and model label conventions.

Key outputs:
- `data/results/m1_standard_metrics.csv`
- `data/results/m1_fairness_race.csv`
- `data/results/m1_fairness_sex.csv`
- `paper/figures/m1_*.png` (4 figures)

---

### `member2_credit_default.ipynb` — M2: Credit Card Default

**Dataset:** Taiwan credit card data, 30,000 rows. Predict default next month.  
**Protected attributes:** Age (binarised at 40), Marital Status (Single/Divorced vs Married)

Key outputs:
- `data/results/m2_standard_metrics.csv`
- `data/results/m4_fairness_age.csv`
- `data/results/m4_fairness_marital_status.csv`
- `paper/figures/m4_demographic_distributions.png`

Note: This notebook is numbered M2 in execution order but its results are labelled M4 in the CSV and figure naming convention.

---

### `member3_compas_recidivism.ipynb` — M3: COMPAS

**Dataset:** ProPublica COMPAS data, 7,214 rows. Predict two-year recidivism.  
**Protected attributes:** Race (African-American+Other vs Caucasian), Sex (Male/Female)

Key outputs:
- `data/results/m3_standard_metrics.csv`
- `data/results/m3_fairness_race.csv`
- `data/results/m3_fairness_sex.csv`
- `paper/figures/m3_*.png` (4 figures)

---

### `member4_bank_marketing.ipynb` — M4: Bank Marketing

**Dataset:** Portuguese bank telemarketing, 41,188 rows. Predict term deposit subscription.  
**Protected attributes:** Education (Higher vs Lower/Other), Gender (Male/Female proxy)

Key outputs:
- `data/results/m4_standard_metrics.csv`  *(Note: file prefix is `m2` in the results directory)*
- `data/results/m2_fairness_education.csv`
- `data/results/m2_fairness_gender.csv`
- `paper/figures/m2_*.png` (4 figures)

---

### `group_synthesis.ipynb` — Cross-Dataset Comparison

Reads all 12 CSV files from `data/results/` and produces the two group-level heatmaps:
- `paper/figures/group_disparate_impact_heatmap.png`
- `paper/figures/group_auc_heatmap.png`

Also generates the cross-dataset narrative comparing which models and attributes showed the largest disparities.

---

## Structure Inside Each Member Notebook

Every notebook follows the CRISP-DM framework per the [group contract](../group_notebook_contract.md):

| Section | Content |
|---|---|
| `## 1. Business Understanding` | Dataset context, research questions, fairness framing |
| `## 2. Data Understanding` | Shape, dtypes, class balance, demographic distributions |
| `## 3. Data Preparation` | Cleaning, encoding, protected attribute construction, train/test split |
| `## 4. Modelling` | GridSearchCV over LR, DT, RF, XGBoost with StratifiedKFold(5) |
| `## 5. Evaluation` | Standard metrics + fairness metrics, figure exports, CSV exports |
| `## 6. Deployment` | Artefact checks, summary of exportable outputs |

Each code cell is preceded by a markdown cell describing what it does and its expected output. Each code cell begins with a comment `# Cell N: description`.

---

## Shared Modelling Parameters

```python
RANDOM_STATE = 42
test_size = 0.20
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
```

Models compared:
- `sklearn.linear_model.LogisticRegression`
- `sklearn.tree.DecisionTreeClassifier`
- `sklearn.ensemble.RandomForestClassifier`
- `xgboost.XGBClassifier`

---

## Dependencies

```bash
pip install -r ../requirements.txt
```
