# IT2011 — Artificial Intelligence and Machine Learning
## Handwritten Digit Recognition: Preprocessing, EDA & Model Training

> **Dataset:** UCI Optical Recognition of Handwritten Digits (Optdigits)  
> **Task:** 10-class classification (digits 0–9) from 8×8 pixel-block features  
> **Module:** IT2011 — Artificial Intelligence and Machine Learning, Progress Review I

---

## 1. Project Overview

This project covers the full supervised machine learning workflow applied to the UCI Optdigits dataset:

1. **Data preprocessing** — cleaning, outlier removal, scaling, feature selection, dimensionality reduction, and encoding
2. **Exploratory data analysis (EDA)** — visualisations and statistical summaries for every preprocessing stage
3. **Model training & evaluation** — six classifier experiments, each with `GridSearchCV` hyperparameter tuning, 5-fold stratified cross-validation, and held-out test evaluation

---

## 2. Dataset Description

| Property | Detail |
|---|---|
| **Name** | Optical Recognition of Handwritten Digits (Optdigits) |
| **Source** | UCI Machine Learning Repository |
| **DOI** | https://doi.org/10.24432/C50P49 |
| **Citation** | Alpaydin, E. & Kaynak, C. (1998). UCI Machine Learning Repository. |
| **Files** | `optdigits.tra` (train), `optdigits.tes` (test) |
| **Features** | 64 integer pixel-intensity values (range 0–16), from 8×8 non-overlapping blocks of a 32×32 bitmap |
| **Target** | Digit class 0–9 (10 classes) |
| **Instances** | 5 620 total — 3 823 train / 1 797 test |
| **Missing values** | None (verified in data cleaning notebook) |

---

## 3. Repository Structure

```
optdigits-classification/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                  # optdigits.tra, optdigits.tes, optdigits.names
│   ├── external/             # (unused — no external reference datasets needed)
│   └── processed/            # optdigits_train_processed.csv, optdigits_test_processed.csv
├── notebooks/
│   ├── preprocessing/        # one notebook per preprocessing technique + pipeline
│   │   ├── DataCleaning.ipynb
│   │   ├── OutlierRemoval.ipynb
│   │   ├── Scaling.ipynb
│   │   ├── FeatureSelection.ipynb
│   │   ├── PCA.ipynb
│   │   ├── Encoding.ipynb
│   │   └── preprocessing_pipeline.ipynb
│   ├── models/               # one notebook per model experiment
│   │   ├── DecisionTree.ipynb
│   │   ├── LogisticRegression.ipynb
│   │   ├── KNearestNeighbors.ipynb
│   │   ├── RandomForest.ipynb
│   │   ├── MLP_PCA.ipynb
│   │   └── SVC.ipynb
│   ├── evaluation/           # reserved for cross-model evaluation notebooks
│   └── pipeline.ipynb        # end-to-end comparison summary
└── results/
    ├── eda_visualizations/   # exported EDA plots (.png)
    ├── logs/
    ├── metrics/              # model_comparison.csv
    └── outputs/
```

---

## 4. Preprocessing Pipeline

Each preprocessing technique is developed in its own notebook and then integrated in `preprocessing_pipeline.ipynb`. All transformers are fitted **on the training split only** and applied to the test split to prevent data leakage.

| Step | Technique | Notebook | Key Detail |
|---|---|---|---|
| 1 | **Data Cleaning** | `DataCleaning.ipynb` | Verified zero missing values; removed duplicate rows; validated pixel range (0–16) |
| 2 | **Outlier Removal** | `OutlierRemoval.ipynb` | IQR-based detection on total per-row pixel intensity; applied to train only |
| 3 | **Feature Scaling** | `Scaling.ipynb` | `MinMaxScaler` (0–1), `fit_transform` on train, `transform` on test |
| 4 | **Feature Selection** | `FeatureSelection.ipynb` | `VarianceThreshold` + correlation-based redundancy removal; fit on train |
| 5 | **Dimensionality Reduction** | `PCA.ipynb` | PCA retaining ~95% variance; fit on train, `transform` on test |
| 6 | **Encoding** | `Encoding.ipynb` | `KBinsDiscretizer` on pixel intensities; `OneHotEncoder` on target column |

> **Important:** Do not use the precomputed PCA CSVs in `data/processed/` as model cross-validation input. Those components were fitted on the full training split and would leak information across CV folds. Each model notebook places its own transformers inside an sklearn `Pipeline` that is refit within every fold.

---

## 5. Exploratory Data Analysis (EDA)

Each preprocessing notebook includes at least one visualisation with a written interpretation:

| Visualisation | Notebook |
|---|---|
| Class distribution bar chart | `DataCleaning.ipynb` |
| Boxplots of per-row pixel intensity (before/after outlier removal) | `OutlierRemoval.ipynb` |
| Feature distribution histograms (before/after scaling) | `Scaling.ipynb` |
| Variance profile & correlation heatmap | `FeatureSelection.ipynb` |
| Cumulative explained-variance curve & 2D PCA scatter | `PCA.ipynb` |
| Bin-count bar charts & encoded feature distributions | `Encoding.ipynb` |

Exported PNG files are saved to `results/eda_visualizations/`.

---

## 6. Model Training

### 6.1 Training Strategy

All six model notebooks follow the same rigorous experimental protocol:

- **Cross-validation:** 5-fold `StratifiedKFold` (shuffled, `random_state=42`) to preserve class balance across folds
- **Hyperparameter tuning:** `GridSearchCV` over a defined `param_grid` for each model variant
- **Selection metric:** CV macro F1 (weights every digit class equally)
- **Evaluation metrics reported:** CV accuracy, CV macro F1, test accuracy, test precision (macro), test recall (macro), test macro F1
- **Leakage prevention:** All preprocessing steps (scaling, PCA, discretization) are wrapped inside `sklearn Pipeline` objects and refitted within each CV fold
- **Test set policy:** The official `optdigits.tes` split is held back entirely and used once — only for final evaluation after the best variant is selected via CV

### 6.2 Model Experiments

| Model | Notebook | Variants Compared | Best Preprocessing |
|---|---|---|---|
| Decision Tree | `DecisionTree.ipynb` | Raw pixels vs. after duplicate removal | Before/after duplicate removal (tied) |
| Logistic Regression | `LogisticRegression.ipynb` | Raw pixels vs. IQR outlier-removed | Before IQR outlier removal |
| K-Nearest Neighbors | `KNearestNeighbors.ipynb` | Raw pixels vs. MinMax scaled | Raw pixels |
| Random Forest | `RandomForest.ipynb` | All 64 features vs. VarianceThreshold filtered | VarianceThreshold (threshold=0.05) |
| MLP Neural Network | `MLP_PCA.ipynb` | MinMax scaled vs. MinMax + PCA 95% | MinMax scaled + PCA 95% |
| Support Vector Classifier | `SVC.ipynb` | Raw / scaled / discretized pixels | MinMax scaled (RBF kernel) |

### 6.3 Results Summary

| Model | Best Variant | CV Accuracy | CV Macro F1 | Test Accuracy | Test Macro F1 |
|---|---|---|---|---|---|
| **RBF SVC** | MinMax scaled | 99.11% | 0.9911 | **97.99%** | **0.9799** |
| **K-Nearest Neighbors** | Raw pixels | 98.51% | 0.9852 | 97.77% | 0.9777 |
| **Random Forest** | VarianceThreshold | 98.17% | 0.9818 | 97.22% | 0.9722 |
| **MLP Classifier** | MinMax + PCA 95% | 97.25% | 0.9726 | 95.77% | 0.9579 |
| **Logistic Regression** | Before outlier removal | 97.12% | 0.9712 | 95.27% | 0.9526 |
| **Decision Tree** | Before duplicate removal | 89.80% | 0.8982 | 85.64% | 0.8564 |

> **Best overall model:** RBF Support Vector Classifier on MinMax-scaled pixels — **97.99% test accuracy**, **0.9799 macro F1**.

Full per-class classification reports and confusion matrices are available inside each model notebook and in `results/metrics/model_comparison.csv`.

---

## 7. How to Run

### Setup
```bash
# Create and activate a virtual environment, then install dependencies
pip install -r requirements.txt
```

### Individual notebooks
1. Place `optdigits.tra` and `optdigits.tes` in `data/raw/`.
2. Open any notebook under `notebooks/preprocessing/` or `notebooks/models/` and run all cells top-to-bottom.

### Full pipeline (end-to-end)
```
1. Run  notebooks/preprocessing/preprocessing_pipeline.ipynb
          → generates data/processed/optdigits_train_processed.csv
                      data/processed/optdigits_test_processed.csv

2. Run each notebook in notebooks/models/
          → each notebook is self-contained and loads raw data independently

3. Update results/metrics/model_comparison.csv from the CV-selected rows
   displayed in each model notebook.

4. Run notebooks/pipeline.ipynb
          → renders the cross-model comparison summary table and charts
```

---

## 8. Output Files

| File | Description |
|---|---|
| `data/processed/optdigits_train_processed.csv` | Preprocessed training split (PCA components + target) |
| `data/processed/optdigits_test_processed.csv` | Preprocessed test split |
| `results/eda_visualizations/*.png` | EDA plots exported from preprocessing notebooks |
| `results/metrics/model_comparison.csv` | CV and test metrics snapshot for all six models |

---

## 9. Dependencies

Key libraries (see `requirements.txt` for pinned versions):

| Library | Purpose |
|---|---|
| `scikit-learn` | Preprocessing, pipelines, cross-validation, all classifiers |
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | EDA visualisations |
| `jupyter` | Notebook environment |
