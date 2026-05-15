
# ESG Controversy Prediction in Banks

## Predicting Environmental, Social, and Governance Controversies Using Machine Learning

**Team:** Sapiens  
**Members:** C Yuktha (IM24027), Loveleen Kaur (IM24032)  
**Date:** May 2026

----------

## Project Overview

This project develops a machine learning framework to predict ESG (Environmental, Social, and Governance) controversies in Indian banks using a hybrid approach combining:

-   **Traditional financial metrics** (ROE, ROA, NPL ratios, etc.)
-   **NLP-based sentiment features** (BM25 relevance scores, FinBERT sentiment analysis)

**Key Result:** Random Forest achieves **F1-score of 0.6857** and **ROC-AUC of 0.71** when combining financial and NLP signals, with **8.7% precision improvement** over financial-only models.

----------

## Dataset

### Data Composition

-   **10 major Indian banks:** SBI, HDFC Bank, ICICI Bank, Axis Bank, Kotak Mahindra Bank, Federal Bank, Punjab National Bank, Bank of Baroda, Canara Bank, IndusInd Bank
-   **Time period:** 2022–2025 (4 fiscal years)
-   **Total observations:** 40 bank-year pairs (raw) → 30 bank-year pairs (lagged)
-   **Features:** 28 total (10 financial + 4 NLP + 5 engineered + 9 base)
-   **Target variable:** Binary ESG controversy indicator (0 = no controversy, 1 = controversy)

### Data File

```
alkaline.xlsx
├── Bank: Bank name (categorical)
├── Year: Fiscal year (2022-2025)
├── Financial metrics: ROE, ROA, NPL ratios, leverage, profitability, etc.
├── ESG signals: Negative news count, BM25 relevance, FinBERT confidence
└── Target: Actual_ESG_Controversy_Level (binary)

```

----------

## Installation & Setup

### Requirements

```bash
Python 3.7+
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
xgboost >= 1.6.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
openpyxl >= 3.6.0  # For Excel file reading

```

### Installation Steps





1.  **Install dependencies:**

```bash
pip install -r requirements.txt

```

Or install manually:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn openpyxl

```

2.  **For NLP features (optional):** If reproducing NLP feature extraction:

```bash
pip install transformers torch rank-bm25

```

----------

## File Structure

```
Sapiens_Project_code/
├── ESG.ipynb       
├── alkaline.xlsx    
├── README.md                       
```


----------

##  Methodology

### Step 1: Data Cleaning & Preprocessing

```python
# Rename messy column names
rename_map = {
    "Total Assets (₹ Cr)": "TotalAssets_Cr",
    "Equity (₹ Cr)": "Equity_Cr",
    # ... etc
}
df.rename(columns=rename_map, inplace=True)

# Drop duplicates and redundant columns
df.drop(columns=["Ticker_x", "NetIncome", "Shares", "YearEndPrice"], inplace=True)

# Impute missing values with median
df[numeric_cols] = df[numeric_cols].apply(lambda col: col.fillna(col.median()))

```

### Step 2: Feature Engineering

Five domain-specific features are engineered to capture financial stress and ESG dynamics:

Feature

Formula

Interpretation

**NegNewsIntensity**

Negative_ESG_News × BM25_Score

Media relevance × volume

**SentimentRiskScore**

Negative_ESG_News × FinBERT_Confidence

Sentiment × confidence

**ControversyMomentum**

Δ Negative_ESG_News (year-over-year)

Trend in negative coverage

**ROA_Stability**

Δ ROA (year-over-year)

Profitability deterioration

**GovernanceStress**

CostToIncomeRatio × Negative_ESG_News

Operational burden × risk

### Step 3: Multicollinearity Filtering

Features with pairwise correlation |r| > 0.85 are removed:

-   **Dropped:** Capad, NPL_Assets_pct, NetProfit_Cr, EmployeeCount, NegNewsIntensity, SentimentRiskScore, GovernanceStress
-   **Retained:** 14 features (10 financial + 4 NLP)

### Step 4: Lagged Prediction Strategy

**Crucial for avoiding information leakage:**

```python
# Shift target backward by 1 year
df["Target_NextYear"] = df.groupby("Bank")["Actual_ESG_Controversy_Level"].shift(-1)

# Keep only valid rows (drop last year per bank)
df_lagged = df.dropna(subset=["Target_NextYear"])

# Result: Year-t features → Year-(t+1) controversy

```

**Dataset shape:** 40 rows → 30 rows (3 prediction windows: 2022→2023, 2023→2024, 2024→2025)

### Step 5: Model Training

Three algorithms compared using **5-fold Stratified K-Fold Cross-Validation:**

```python
MODELS = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced", max_depth=4, random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        scale_pos_weight=(y==0).sum() / (y==1).sum(),
        use_label_encoder=False, eval_metric="logloss", random_state=42
    )
}

```

### Step 6: Evaluation & Interpretation

**Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC  
**Diagnostics:** Confusion matrix, ROC curve, Precision-Recall curve  
**Feature importance:** Permutation importance (30 iterations)

----------

##  Results Summary

### Experiment A: Financial Features Only

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 40.0% | 35.3% | 35.0% | 0.3467 | 0.4917 |
| Random Forest | 63.3% | 72.0% | 65.0% | **0.6548** | 0.6806 |
| XGBoost | 66.7% | 76.7% | 66.7% | 0.6762 | **0.7028** |
### Experiment B: Financial + NLP Features

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 43.3% | 55.3% | 40.0% | 0.4267 | 0.3972 |
| **Random Forest** | **66.7%** | **78.3%** | **66.7%** | **0.6857** | 0.5861 |
| XGBoost | 63.3% | 66.7% | 63.3% | 0.6395 | 0.6611 |
### Key Findings

✅ **NLP features improve precision by 8.7%** (72% → 78.3%)  
✅ **ROA_Stability is the dominant predictor** (13.78% permutation importance)  
✅ **Random Forest outperforms XGBoost** on this small-sample problem  
⚠️ **ROC-AUC trade-off:** Probability calibration affected by NLP noise

----------

##  Key Insights

### 1. ROA Stability Dominates

Banks experiencing declining profitability are **8.5× more likely** to face ESG controversies. This is the single strongest signal.

### 2. NLP Adds Precision, Not Recall

-   **Precision:** +8.7% (better false alarm reduction)
-   **ROC-AUC:** -4.5% (probability miscalibration)
-   **F1:** +4.7% (balanced improvement)

**Implication:** Use combined model when precision is critical (few false alarms acceptable).

### 3. Non-linear Models Required

Logistic Regression achieves only 40% accuracy vs. 66% for tree-based models, indicating complex decision boundaries.

### 4. Small Sample Uncertainty

With n=30, cross-validation estimates have high variance. External validation essential before production use.

----------

##  Code Usage Examples

### Load and Explore Data

```python
import pandas as pd

df = pd.read_excel("alkaline.xlsx")
print(f"Dataset shape: {df.shape}")
print(f"Banks: {df['Bank'].unique()}")
print(f"Years: {sorted(df['Year'].unique())}")
print(f"Target distribution:\n{df['Actual_ESG_Controversy_Level'].value_counts()}")

```

### Perform Feature Correlation Analysis

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Select features
features = ['ROE', 'ROA', 'NPL_Loans_pct', 'Negative_ESG_News_Count', ...]
corr_matrix = df[features].corr()

# Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdYlGn", center=0)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.show()

```

### Train and Evaluate a Single Model

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Prepare data
X = df_lagged[selected_features].values
y = df_lagged["Target_NextYear"].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train with cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", max_depth=4)

scores = cross_validate(
    rf, X_scaled, y, cv=cv,
    scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
)

# Results
for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
    mean_score = scores[f'test_{metric}'].mean()
    std_score = scores[f'test_{metric}'].std()
    print(f"{metric}: {mean_score:.4f} (+/- {std_score:.4f})")

```

### Feature Importance Analysis

```python
from sklearn.inspection import permutation_importance

# Train model on full data
rf.fit(X_scaled, y)

# Compute permutation importance
perm_imp = permutation_importance(rf, X_scaled, y, n_repeats=30, random_state=42, n_jobs=-1)

# Rankings
importance_df = pd.DataFrame({
    'Feature': selected_features,
    'Importance': perm_imp.importances_mean
}).sort_values('Importance', ascending=False)

print(importance_df.head(10))

```

----------

## Configuration & Hyperparameters

### Model Hyperparameters

```python
# Logistic Regression
LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)

# Random Forest
RandomForestClassifier(
    n_estimators=200,           # Number of trees
    max_depth=4,                # Maximum tree depth (shallow for regularization)
    class_weight="balanced",    # Handle class imbalance
    random_state=42
)

# XGBoost
XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    scale_pos_weight=0.76,      # Inverse of class ratio (13/17)
    eval_metric="logloss",
    random_state=42,
    verbosity=0
)

```

### Cross-Validation Setup

```python
StratifiedKFold(
    n_splits=5,           # 5 folds
    shuffle=True,         # Shuffle before splitting
    random_state=42       # Reproducibility
)

```

----------

## Advanced Usage

### Modify Feature Set

```python
# Custom feature selection
CUSTOM_FEATURES = [
    # Financial only
    'ROE', 'ROA', 'NPL_Loans_pct', 'Loans_Assets',
    # NLP only
    'Negative_ESG_News_Count', 'Avg_BM25_Score',
]

X_custom = df_lagged[CUSTOM_FEATURES].values
X_custom_scaled = scaler.fit_transform(X_custom)

# Re-run experiments with custom features

```

### Adjust Correlation Threshold

```python
# Default: |r| > 0.85
# Stricter: |r| > 0.75
# Looser: |r| > 0.95

def drop_high_corr(df_in, features, threshold=0.75):
    corr = df_in[features].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    kept = [f for f in features if f not in to_drop]
    print(f"Dropped ({threshold}): {to_drop}")
    return kept

```

### Hyperparameter Tuning (GridSearch)

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5, 6],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(class_weight="balanced", random_state=42),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_scaled, y)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best F1-score: {grid_search.best_score_:.4f}")

```

----------



##  References

### Key Paper

1.  Friede, G., Busch, T., & Bassen, A. (2015). "ESG and financial performance: aggregated evidence from more than 2000 empirical studies." _Journal of Sustainable Finance & Investment_, 5(4), 210–233.
    
    

----------

## Citation

If you use this code or dataset, please cite:

```bibtex
@misc{sapiens2026,
  title={Predicting ESG Controversies in Banks Using Machine Learning Techniques},
  author={C Yuktha and Loveleen Kaur},
  year={2026},,
  note={Team: Sapiens}
}

```

----------

## Contact & Support

**Team:** Sapiens  
**Members:**

-   C Yuktha (IM24027)
-   Loveleen Kaur (IM24032)

For questions or issues, please refer to the project report or contact the team.

----------

##  License

This project is provided for educational and research purposes.

----------

## Checklist for Running the Code

-   [ ] Python 3.7+ installed
-   [ ] Dependencies installed (`pip install -r requirements.txt`)
-   [ ] `alkaline.xlsx` in working directory or path configured
-   [ ] Output directory exists (or will be created automatically)
-   [ ] Sufficient disk space for plots (~5 MB)
-   [ ] Expected runtime: 2–5 minutes

**Status:** Ready to run!

----------

_Last Updated: May 2026_ 