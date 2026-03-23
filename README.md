# -Small-Business-Administration-loan-default-prediction
An end-to-end binary classification pipeline that predicts whether a U.S. Small Business Administration (SBA) loan will default. Built on 800K+ real loan records, the project covers the full ML lifecycle — from raw data cleaning to a production-ready scoring function validated on unseen holdout data.

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline Summary](#pipeline-summary)
- [Feature Engineering](#feature-engineering)
- [Models & Results](#models--results)
- [Scoring Function](#scoring-function)
- [How to Run](#how-to-run)
- [Requirements](#requirements)
- [Key Takeaways](#key-takeaways)

---

## Overview

Small businesses are a major driver of U.S. job creation, and SBA loan default prediction is critical for risk management. This project builds and evaluates two regularized linear models to predict loan default probability, with a focus on:

- Handling **class imbalance** (17.5% default rate)
- **Zero data leakage** — all encoders and scalers fit on training data only
- A **reusable scoring function** that scores new data without access to labels
- **Threshold optimization** via macro F1 to maximize minority class detection

---

## Dataset

- **Source:** U.S. Small Business Administration (SBA) loan records
- **Size:** 809,247 rows × 20 columns (raw) → 809,077 rows × 28 features (after cleaning & engineering)
- **Target:** `MIS_Status` — 0 = paid in full, 1 = defaulted (17.5% positive rate)
- **Features include:** loan amounts, number of employees, job creation/retention, industry codes (NAICS), urban/rural classification, franchise status, low-documentation flag

---

## Project Structure

```
sba-loan-default-prediction/
├── data/
│   ├── SBA_loans_project_1.csv.zip
│   └── SBA_loans_project_1_holdout_students_valid_no_labels.csv
├── artifacts/
│   ├── sklearn/
│   │   └── sklearn_artifacts.pkl       # model + encoders + scaler + threshold
│   └── h2o/
│       ├── GLM_model_python_...        # H2O binary model file
│       └── h2o_artifacts.pkl           # encoders + threshold
├── main_notebook.ipynb                 # data prep, feature engineering, training
├── scoring_notebook.ipynb              # scoring function + validation
├── predictions_sklearn.csv             # Kaggle submission — sklearn
├── predictions_h2o.csv                 # Kaggle submission — H2O
└── README.md
```

---

## Pipeline Summary

```
Raw Data (809K rows)
       ↓
Data Cleaning
  • Standardize binary flags (RevLineCr, LowDoc)
  • Extract NAICS 2-digit sector codes
  • Convert FranchiseCode → is_franchise binary flag
  • Drop uninformative columns (City, Bank, BalanceGross, Zip)
       ↓
Train / Validation / Test Split (70% / 15% / 15%) — stratified
       ↓
Categorical Encoding (fit on train only)
  • OHE → UrbanRural, NewExist
  • Target Encoding → State, NAICS_sector
       ↓
Feature Engineering (12 features)
       ↓
Scaling — StandardScaler (sklearn pipeline only)
       ↓
Model Tuning
  • sklearn LR — 50 combinations
  • H2O GLM   — 60 combinations
       ↓
Threshold Optimization (macro F1 on validation)
       ↓
Final Evaluation on Test Set
```

---

## Feature Engineering

12 engineered features were created — OHE, label encoding, and target encoding do not count toward this total.

| Feature | Description |
|---------|-------------|
| `sba_coverage_ratio` | SBA guaranteed amount / approved loan amount |
| `loan_per_employee` | Approved loan amount / (employees + 1) |
| `disbursement_ratio` | Disbursed amount / approved amount |
| `job_creation_ratio` | Jobs created / (employees + 1) |
| `log_GrAppv` | Log1p of approved loan amount |
| `log_NoEmp` | Log1p of number of employees |
| `log_DisbursementGross` | Log1p of disbursed amount |
| `retained_to_emp_ratio` | Jobs retained / (employees + 1) |
| `is_low_doc_franchise` | Interaction: low-doc AND franchise loan |
| `sba_appv_per_emp` | SBA approved amount / (employees + 1) |
| `total_jobs` | Jobs created + jobs retained |
| `GrAppv_bin_woe` | WOE-encoded quantile bin of approved loan amount |

---

## Models & Results

Both models were tuned using **AUCPR** as the selection metric on a held-out validation set.

### Hyperparameter Search

| Model | Combinations | Best Config |
|-------|-------------|-------------|
| sklearn Logistic Regression | 50 | L2, C=20, lbfgs solver |
| H2O GLM (binomial) | 60 | alpha=0.5, lambda=1e-6 |

### Final Test Set Performance

| Metric | sklearn LR | H2O GLM |
|--------|-----------|---------|
| **AUC** | 0.7403 | 0.7403 |
| **AUCPR** | 0.3861 | **0.3864** |
| **Log-loss** | 0.4103 | 0.4103 |
| **Threshold** | 0.29 | 0.29 |

### Confusion Matrix at Threshold = 0.29

| | Predicted 0 | Predicted 1 |
|--|------------|------------|
| **Actual 0** | 86,125 (TN) | 13,955 (FP) |
| **Actual 1** | 11,719 (FN) | 9,563 (TP) |

- **Recall:** ~45% of actual defaults correctly identified
- **Precision:** ~41% of flagged loans were actual defaults
- Threshold shifted from 0.50 → **0.29** to account for 17.5% class imbalance

### Kaggle Competition Results

| Submission | Public AUCPR |
|------------|-------------|
| sklearn LR | 0.37492 |
| **H2O GLM** | **0.37540** |

---

## Scoring Function

The `project_1_scoring()` function in `scoring_notebook.ipynb` scores new data for both model families:

```python
predictions = project_1_scoring(new_data, model_type='sklearn')
predictions = project_1_scoring(new_data, model_type='h2o')
```

**Input:** Pandas DataFrame with the same schema as training data, without `MIS_Status`

**Output:** Pandas DataFrame with columns:

| Column | Description |
|--------|-------------|
| `index` | Original record index |
| `label` | Predicted class (0 or 1) |
| `probability_0` | Probability of non-default |
| `probability_1` | Probability of default |

**Scoring guarantees:**
- No records dropped 
- No encoders/scalers fitted during scoring 
- No target column expected 
- Validated on 89,917 holdout records 

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/bass990/-Small-Business-Administration-loan-default-prediction.git
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the main notebook
Open `main_notebook.ipynb` and run all cells top to bottom.
This will clean the data, engineer features, tune both models, and save all artifacts.

### 4. Run the scoring notebook
Open `scoring_notebook.ipynb` and run all cells.
This validates the scoring function and generates Kaggle submission files.

---

## Requirements

```
pandas
numpy
scikit-learn
category_encoders
h2o
joblib
jupyter
```

> **Note:** H2O requires Java (JDK 11+). Install from https://www.java.com before running H2O cells.

---

## Key Takeaways

- **Low regularization wins:** Both models performed best with minimal regularization (high C / low lambda), suggesting the engineered features are informative and underfitting is the primary risk for linear models on this dataset
- **Threshold matters:** Shifting the decision threshold from 0.50 to 0.29 significantly improved minority class detection on imbalanced data
- **Linear model ceiling:** AUCPR of ~0.386 suggests non-linear relationships exist — tree-based models would likely push AUCPR to 0.55+ on this dataset
- **Both frameworks converge:** sklearn and H2O GLM produced virtually identical results (AUC difference < 0.001), confirming the feature set drives performance more than the implementation framework

---

## Author

**Name:** Mamadou Bassirou Diallo| University of Texas at Dallas  
 LinkedIn: `linkedin.com/in/mamadou9905`  
GitHub: `bass990`
Kaggle: `bass990` 
