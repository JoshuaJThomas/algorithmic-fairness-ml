# Data Directory

This directory contains the raw input datasets and pre-computed result CSVs.

---

## Raw Datasets

### Adult Census Income (`adult/`)

| Property | Value |
|---|---|
| Source | [UCI ML Repository — Adult](https://archive.ics.uci.edu/dataset/2/adult) |
| Origin | 1994 US Census Bureau database |
| Licence | CC BY 4.0 |
| Rows | 48,842 (32,561 train + 16,281 test) |
| Target | `income` — binary: `>50K` / `<=50K` |
| Protected attributes used | `sex`, `race` |

Files:
- `adult.data` — training set (no header, comma-separated with spaces)
- `adult.test` — test set (same format, first row is a comment to skip)
- `adult.names` — feature descriptions

Column order (no header in file):
```
age, workclass, fnlwgt, education, education-num, marital-status,
occupation, relationship, race, sex, capital-gain, capital-loss,
hours-per-week, native-country, income
```

Cleaning applied in `notebooks/member1_adult_income.ipynb`:
- Strip whitespace from all string columns
- Remove rows with `?` in `workclass`, `occupation`, `native-country`
- Combine train and test sets, re-split stratified 80/20

---

### Bank Marketing (`bank_marketing/bank-additional/`)

| Property | Value |
|---|---|
| Source | [UCI ML Repository — Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) |
| Origin | Portuguese bank telephone campaigns, 2008–2013 |
| Licence | CC BY 4.0 |
| Rows | 41,188 (`bank-additional-full.csv`) |
| Target | `y` — binary: `yes` / `no` (term deposit subscription) |
| Protected attributes used | `education`, `gender` (derived from `marital`) |

Files:
- `bank-additional-full.csv` — full dataset (semicolon-separated)
- `bank-additional.csv` — 10% random sample
- `bank-additional-names.txt` — variable descriptions

Key columns: `age`, `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `day_of_week`, `duration`, `campaign`, `pdays`, `previous`, `poutcome`, 4 socioeconomic indicators, `y`

Cleaning applied in `notebooks/member4_bank_marketing.ipynb`:
- Encode `education` as binary: university/high school = Higher, rest = Lower/Other
- Derive `gender` proxy from `marital` and `age` distributions (noted as a limitation)
- One-hot encode categorical features

---

### COMPAS Recidivism (`compas/`)

| Property | Value |
|---|---|
| Source | [ProPublica COMPAS Analysis](https://github.com/propublica/compas-analysis) |
| Origin | Broward County, Florida court records |
| Licence | Public release (no stated restrictions) |
| Rows | 7,214 (after ProPublica filtering) |
| Target | `two_year_recid` — binary: 1 = recidivated within 2 years |
| Protected attributes used | `race`, `sex` |

File: `compas-scores-two-years.csv`

Key columns used: `age`, `c_charge_degree`, `race`, `sex`, `priors_count`, `days_b_screening_arrest`, `decile_score`, `two_year_recid`

Filtering applied in `notebooks/member3_compas_recidivism.ipynb`:
- Keep only rows where `days_b_screening_arrest` is between −30 and 30
- Remove rows where `is_recid = -1` (charge not filed)
- Keep only `charge_degree` in `{F, M}`
- Race grouped: African-American + Other vs Caucasian (mirrors ProPublica methodology)

---

### Credit Card Default (`credit_default/`)

| Property | Value |
|---|---|
| Source | [UCI ML Repository — Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) |
| Origin | Taiwan credit card holders, April–September 2005 |
| Licence | CC BY 4.0 |
| Rows | 30,000 |
| Target | `default_payment_next_month` — binary: 1 = default |
| Protected attributes used | `age` (binarised at 40), `marriage` |

File: `credit_default.xls` (requires `openpyxl` or `xlrd`)

Key columns: `LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`, `PAY_0`–`PAY_6`, `BILL_AMT1`–`BILL_AMT6`, `PAY_AMT1`–`PAY_AMT6`, `default_payment_next_month`

Cleaning applied in `notebooks/member2_credit_default.ipynb`:
- Drop ID column
- Binarise `AGE`: Older = ≥40, Younger = <40
- Binarise `MARRIAGE`: Single/Divorced = {1,3}, Married = {2}
- Remove undocumented `EDUCATION` values (0, 5, 6)

---

## Result CSVs (`results/`)

Pre-computed output from all four notebooks. These files are what the `group_synthesis.ipynb` reads.

### Standard Metrics Schema

```
model, accuracy, precision, sensitivity, specificity, f1_macro, f1_weighted, auc_roc, cohen_kappa
```

Files: `m1_standard_metrics.csv`, `m2_standard_metrics.csv`, `m3_standard_metrics.csv`, `m4_standard_metrics.csv`

### Fairness Metrics Schema

```
model, sensitive_attribute, reference_group, comparison_group,
positive_rate_reference, positive_rate_comparison,
demographic_parity_difference, equalized_odds_difference, disparate_impact_ratio
```

Files:

| File | Dataset | Attribute |
|---|---|---|
| `m1_fairness_race.csv` | Adult Income | Race (White vs Non-White) |
| `m1_fairness_sex.csv` | Adult Income | Sex (Male vs Female) |
| `m2_fairness_education.csv` | Bank Marketing | Education (Higher vs Lower) |
| `m2_fairness_gender.csv` | Bank Marketing | Gender (Male vs Female) |
| `m3_fairness_race.csv` | COMPAS | Race (Afr-Am+Other vs Caucasian) |
| `m3_fairness_sex.csv` | COMPAS | Sex (Male vs Female) |
| `m4_fairness_age.csv` | Credit Default | Age (Older vs Younger) |
| `m4_fairness_marital_status.csv` | Credit Default | Marital Status |

Model labels are consistent across all files:
- `Logistic Regression`
- `Decision Tree`
- `Random Forest`
- `XGBoost`
