# Group Notebook Contract

This document defines the minimum consistency required across all four member
notebooks so the group synthesis notebook can ingest results without manual
cleanup.

## Non-Negotiable Structure

Every member notebook must use the same top-level markdown headers in this order:

1. `## 1. Business Understanding`
2. `## 2. Data Understanding`
3. `## 3. Data Preparation`
4. `## 4. Modelling`
5. `## 5. Evaluation`
6. `## 6. Deployment`

Every code cell must begin with a comment in this style:

```python
# Cell N: [description]
```

Before each code cell, include a markdown cell that states:

- what the cell achieves
- the expected output

After each major result block, include a short markdown insight cell suitable for
reuse in the report.

## Shared Modelling Rules

These rules must match across all member notebooks:

- `RANDOM_STATE = 42`
- `warnings.filterwarnings("ignore")`
- stratified `80/20` train-test split
- `GridSearchCV`
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)`
- required models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - XGBoost
- standard metrics:
  - accuracy
  - precision
  - sensitivity
  - specificity
  - f1_macro
  - f1_weighted
  - auc_roc
  - cohen_kappa
- fairness metrics:
  - demographic_parity_difference
  - equalized_odds_difference
  - disparate_impact_ratio

The notebooks may differ in dataset-specific loading, cleaning, protected-attribute
construction, and preprocessing. The contract is shared outputs and shared
evaluation logic, not identical code.

## Shared CSV Schemas

### Standard Metrics CSV

Each member must export one standard metrics CSV using exactly these columns and
this column order:

```text
model,accuracy,precision,sensitivity,specificity,f1_macro,f1_weighted,auc_roc,cohen_kappa
```

Expected file names:

- `data/results/m1_standard_metrics.csv`
- `data/results/m2_standard_metrics.csv`
- `data/results/m3_standard_metrics.csv`
- `data/results/m4_standard_metrics.csv`

### Fairness Metrics CSV

Each fairness CSV must use exactly these columns and this column order:

```text
model,sensitive_attribute,reference_group,comparison_group,positive_rate_reference,positive_rate_comparison,demographic_parity_difference,equalized_odds_difference,disparate_impact_ratio
```

Expected file names:

- `data/results/m1_fairness_race.csv`
- `data/results/m1_fairness_sex.csv`
- `data/results/m2_fairness_gender.csv`
- `data/results/m2_fairness_education.csv`
- `data/results/m3_fairness_race.csv`
- `data/results/m3_fairness_sex.csv`
- `data/results/m4_fairness_age.csv`
- `data/results/m4_fairness_marital_status.csv`

If a member uses different protected attributes, only the file suffix changes.
The column schema stays fixed.

## Shared Figure Contract

Each member should export four figure files:

- demographic distribution figure
- target by demographics figure
- fairness metrics comparison figure
- performance comparison figure

Naming pattern:

- `paper/figures/mX_demographic_distributions.png`
- `paper/figures/mX_target_by_demographics.png` or the member-specific agreed label
- `paper/figures/mX_fairness_metrics.png`
- `paper/figures/mX_performance_comparison.png`

Important:

- The current Member 1 notebook uses `m1_income_by_demographics.png` because that
  file name is already part of the agreed deliverable list.
- For synthesis, prefer mapping figure meaning by convention rather than trying to
  infer it from the file name.

## Narrative Standard

Each notebook must make the same kinds of claims:

- treat the dataset as a benchmark or case study, not proof of causal discrimination
- distinguish predictive performance from fairness evaluation
- state that fairness metrics can disagree because they encode different notions of parity
- acknowledge dataset-specific limitations
- avoid claiming that the most accurate model is the fairest model without evidence

## Group Synthesis Requirements

The group synthesis notebook should assume:

- one standard metrics CSV per member
- two fairness CSVs per member
- four models per member
- consistent `model` labels across all CSVs:
  - `Logistic Regression`
  - `Decision Tree`
  - `Random Forest`
  - `XGBoost`

Do not let members rename models ad hoc. If one member uses `XGB`, `RF`, or
`LogReg`, the synthesis step becomes unnecessary cleanup work.

## Member 1 Baseline

Member 1 is the reference implementation for this contract:

- notebook: `notebooks/member1_adult_income.ipynb`
- standard metrics: `data/results/m1_standard_metrics.csv`
- fairness metrics:
  - `data/results/m1_fairness_race.csv`
  - `data/results/m1_fairness_sex.csv`

Use Member 1 to mirror:

- section ordering
- metric column names
- model labels
- export assertions
- deployment artifact checks
