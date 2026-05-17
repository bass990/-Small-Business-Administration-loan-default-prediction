# SBA Loan Default Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-blue?style=for-the-badge)
![H2O](https://img.shields.io/badge/H2O.ai-3.46-FFD700?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**End-to-end binary classification pipeline comparing regularized linear models and gradient boosting on 809K SBA loan records**

[Live Demo →](#streamlit-demo) · [Portfolio Site →](https://bass990.github.io/-Small-Business-Administration-loan-default-prediction/) · [View Notebook →](complete_code.ipynb)

</div>

---

## Overview

This project builds, tunes, and compares three model families to predict whether a U.S. Small Business Administration (SBA) loan will default. The dataset spans **809K loan records** across 19 features covering approval amounts, business characteristics, employment, and geographic data.

All models are trained on an identical leakage-safe preprocessing pipeline with **12 hand-engineered features**. LightGBM is added to test whether non-linear interactions improve beyond the linear model plateau (~AUCPR 0.39).

| Model | AUC | AUCPR | Log-loss | Threshold |
|---|---|---|---|---|
| **LightGBM** *(selected)* | run notebook | run notebook | run notebook | F1-optimized |
| H2O GLM (ElasticNet α=0.5, λ=1e-6) | 0.7403 | 0.3864 | 0.4103 | 0.29 |
| sklearn Logistic Regression (L2, C=20) | 0.7403 | 0.3861 | 0.4103 | 0.29 |

> Linear models converged to the same AUC and Log-loss, confirming the linear decision boundary is the performance ceiling for LR/GLM on this dataset — not regularization or tuning. LightGBM captures feature interactions (`sba_coverage_ratio × log_GrAppv`) that no linear model can represent. Run `complete_code.ipynb` end-to-end to populate the LightGBM test metrics.

---

## Quick Start

```bash
# Clone and install (Streamlit app only — no Java required)
git clone https://github.com/bass990/-Small-Business-Administration-loan-default-prediction.git
cd -Small-Business-Administration-loan-default-prediction
pip install -r requirements.txt

# Run the Streamlit dashboard
streamlit run app.py

# For notebook development (includes H2O, requires Java 11+)
pip install -r requirements-dev.txt
jupyter notebook complete_code.ipynb
```

The interactive dashboard includes:
- **Data Explorer** — class distribution, loan amount histograms, default rates by category
- **Feature Engineering** — visual explanation of all 12 engineered features
- **Model Results** — AUC/AUCPR/Log-loss comparison across all three models, confusion matrices
- **Live Predictor** — enter loan details and get a real-time default probability (uses LightGBM when artifacts are present)

---

## Project Architecture

```
Raw Data (880K records, 19 features)
         │
         ▼
┌─────────────────────────────────────────┐
│           Data Cleaning                 │
│  • Fix NAICS codes (0 → NaN → 0)        │
│  • Standardize RevLineCr / LowDoc       │
│  • Convert FranchiseCode → is_franchise │
│  • Handle DisbursementGross = 0         │
└───────────────┬─────────────────────────┘
                │
                ▼
    Stratified Train / Val / Test Split
         70%  /  15%  /  15%
                │
                ▼
┌─────────────────────────────────────────┐
│         Categorical Encoding            │
│  • OHE: UrbanRural, NewExist            │
│  • Target Enc: State, NAICS_sector      │
│  • WOE Enc: GrAppv quantile bins        │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│         Feature Engineering (12)        │
│  Ratios, log transforms, interactions   │
└───────────────┬─────────────────────────┘
                │
         ┌──────┴──────┐
         ▼             ▼
   ┌───────────┐  ┌───────────┐
   │  sklearn  │  │  H2O GLM  │
   │  LR Grid  │  │  Grid     │
   │  (50+ HP) │  │  (12 HP)  │
   └─────┬─────┘  └─────┬─────┘
         │               │
         ▼               ▼
   Threshold Optimization (F1 macro → 0.29)
         │               │
         ▼               ▼
   Artifacts saved   Artifacts saved
   (pkl + model)     (pkl + H2O binary)
```

---

## Repository Structure

```
SBA_loan_default_prediction/
├── app.py                          # Streamlit interactive demo
├── Project-1-Mamadou_Bassirou-     # Main analysis notebook
│   Diallo-DAL181960.ipynb
├── scoring_notebook.ipynb          # Scoring function validation
├── scoring_template.py             # Completed scoring function
│
├── src/                            # Reusable Python modules
│   ├── __init__.py
│   ├── preprocessing.py            # Data cleaning pipeline
│   ├── features.py                 # Feature engineering
│   └── scoring.py                  # Production scoring function
│
├── artifacts/
│   ├── sklearn/
│   │   └── sklearn_artifacts.pkl   # Model + encoders + scaler
│   └── h2o/
│       ├── h2o_artifacts.pkl       # Encoders + metadata
│       └── GLM_model_*             # H2O binary model
│
├── data/
│   ├── SBA_loans_project_1.csv.zip
│   └── SBA_loans_project_1_holdout_students_valid_no_labels.csv.zip
│
├── predictions_sklearn.csv         # Holdout predictions (89,917 rows)
├── predictions_h2o.csv             # Holdout predictions (89,917 rows)
│
├── docs/                           # GitHub Pages portfolio site
│   ├── index.html
│   ├── styles.css
│   └── script.js
│
├── examples/                       # Reference Titanic workflow
├── requirements.txt
└── README.md
```

---

## Feature Engineering

12 engineered features were created beyond categorical encoding:

| # | Feature | Formula | Rationale |
|---|---|---|---|
| 1 | `sba_coverage_ratio` | SBA_Appv / GrAppv | SBA guarantee percentage — higher = lower lender risk |
| 2 | `loan_per_employee` | GrAppv / (NoEmp + 1) | Loan burden per worker — size-normalized exposure |
| 3 | `disbursement_ratio` | DisbursementGross / GrAppv | Actual vs. approved disbursement gap |
| 4 | `job_creation_ratio` | CreateJob / (NoEmp + 1) | Growth intent relative to current size |
| 5 | `retained_to_emp_ratio` | RetainedJob / (NoEmp + 1) | Employment stability signal |
| 6 | `sba_appv_per_emp` | SBA_Appv / (NoEmp + 1) | Guaranteed amount per employee |
| 7 | `log_GrAppv` | log(1 + GrAppv) | Compress right-skewed loan amounts |
| 8 | `log_NoEmp` | log(1 + NoEmp) | Compress right-skewed employee counts |
| 9 | `log_DisbursementGross` | log(1 + DisbursementGross) | Compress disbursement distribution |
| 10 | `is_low_doc_franchise` | LowDoc × is_franchise | Interaction: low-doc + franchise = elevated risk |
| 11 | `total_jobs` | CreateJob + RetainedJob | Total employment impact |
| 12 | `GrAppv_bin_woe` | WOE(qcut(GrAppv, 5)) | Non-linear loan-size → default relationship |

---

## Model Results

### Test Set Performance (15% hold-out, ~132K records)

**sklearn Logistic Regression** (L2, C=20, lbfgs)
```
AUC:      0.7403    AUCPR: 0.3861    Log-loss: 0.4103
Threshold: 0.29 (F1-optimized)

Confusion Matrix:
              Predicted 0   Predicted 1
  Actual 0      86,111        13,969
  Actual 1      11,735         9,547
```

**H2O GLM** (ElasticNet α=0.5, λ=1e-6) ← *Selected Model*
```
AUC:      0.7403    AUCPR: 0.3864    Log-loss: 0.4103
Threshold: 0.29 (F1-optimized)

Confusion Matrix:
              Predicted 0   Predicted 1
  Actual 0      86,125        13,955
  Actual 1      11,719         9,563
```

### Hyperparameter Tuning

**sklearn**: Grid search over 50+ combinations
- Penalties: L1, L2, ElasticNet
- C values: [0.1, 0.5, 1, 5, 10, 20]
- Solvers: lbfgs, liblinear, saga
- Metric: AUCPR on validation set

**H2O GLM**: Grid search over 12 combinations
- Lambda: [1e-5, 1e-4, 1e-3, 1e-2]
- Alpha: [0.0, 0.5, 1.0]
- Metric: AUCPR on validation set

---

## Quick Start

### Prerequisites
- Python 3.12+
- Java 8+ (required for H2O)

### Installation

```bash
git clone https://github.com/bass990/SBA_loan_default_prediction.git
cd SBA_loan_default_prediction
pip install -r requirements.txt
```

### Run the Streamlit App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### Use the Scoring Function

```python
import pandas as pd
from src.scoring import project_1_scoring

# Load new loan data (same schema as training, without MIS_Status)
new_data = pd.read_csv("data/SBA_loans_project_1_holdout_students_valid_no_labels.csv.zip")

# Score with sklearn model
predictions = project_1_scoring(new_data, model_type="sklearn")
print(predictions.head())
# index  label  probability_0  probability_1
#     0      0       0.833668       0.166332
#     1      0       0.842401       0.157599

# Score with H2O GLM (requires h2o.init() first)
import h2o
h2o.init()
predictions_h2o = project_1_scoring(new_data, model_type="h2o")
```

### Run Jupyter Notebook

```bash
jupyter notebook Project-1-Mamadou_Bassirou-Diallo-DAL181960.ipynb
```

---

## Tech Stack

| Category | Tools |
|---|---|
| **Data** | pandas, numpy, polars, pyarrow |
| **ML — sklearn** | scikit-learn, category-encoders |
| **ML — H2O** | h2o (3.46), H2O GLM |
| **Visualization** | matplotlib, seaborn, plotly |
| **App** | Streamlit |
| **Serialization** | joblib, pickle |

---

## Key Design Decisions

**Why threshold 0.29 instead of 0.5?**
The dataset has ~19% positive class (default). At 0.5, the model would classify almost all loans as non-default. Optimizing by F1-macro on the validation set pushes the threshold down to 0.29, capturing ~45% more true positives at the cost of more false positives — the right trade-off for risk management.

**Why AUCPR instead of AUC for tuning?**
With 19% positive class, AUC can look deceptively high even with poor minority-class recall. AUCPR (area under Precision-Recall curve) directly measures performance on the positive class and is the correct metric for imbalanced binary classification.

**Why H2O GLM over sklearn?**
Both models converge to AUC 0.7403 and Log-loss 0.4103. H2O edges sklearn on AUCPR (0.3864 vs 0.3861). Given equal performance, H2O GLM is selected as the submission model — it runs grid search natively on the full 566K training set without requiring a subsample, which provides more stable coefficient estimates.

---

## Author

**Mamadou Bassirou Diallo**  
MS Business Analytics | The University of Texas at Dallas  
NetID: DAL181960 | Course: BUAN 6341 Machine Learning, Spring 2026

[![GitHub](https://img.shields.io/badge/GitHub-bass990-181717?style=flat&logo=github)](https://github.com/bass990)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
