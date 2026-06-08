# Credit Default Risk Analysis
### Do demographic factors predict credit card default?

A machine learning analysis of the UCI Taiwan Credit Default dataset, examining whether demographic characteristics (gender, education, marital status, age) are meaningful predictors of credit card default — or whether financial behavior tells the real story.

---

## The Question

Banks want to know who will default. The instinctive answer might be to look at who someone *is* — their age, education, marital status. This project asks whether that instinct holds up under scrutiny, and what the data actually suggest about fairness in credit risk modeling.

**Short answer:** Demographics are weak predictors. Credit utilization behavior is the dominant signal, and it's distributed nearly uniformly across demographic groups.

---

## Dataset

**UCI Machine Learning Repository — Default of Credit Card Clients**  
30,000 credit card clients from a Taiwanese bank, collected during Taiwan's early 2000s credit crisis.

| Feature | Description |
|---|---|
| `limit_bal` | Initial credit limit offered |
| `sex` | Gender (M/F) |
| `education` | Education level |
| `marital` | Marital status |
| `age` | Age in years |
| `pay_1` – `pay_6` | Repayment status over 6 months |
| `bill_amt1` – `bill_amt6` | Bill statement amounts over 6 months |
| `pay_amt1` – `pay_amt6` | Previous payment amounts over 6 months |
| `delinquent` | **Target** — did the client default? (Y/N) |

**Preprocessing notes:**
- ~850 records removed due to missing education/marital status values
- Dataset sex-balanced to ~50/50 (raw data: 60% female) using random sampling, cross-referenced against Taiwan's 2005 census demographics
- Final working dataset: ~23,150 records

---

## Methods

Three model families applied across three feature subsets:

| Model | Feature Subset |
|---|---|
| Linear Regression | All features |
| Logistic Regression | All features / Demographics only / Credit behavior only |
| K-Nearest Neighbors (k=3, k=5) | All features / Demographics only / Credit behavior only |

Standard sklearn preprocessing pipeline: `StandardScaler` applied before logistic regression and KNN. Train/test split: 80/20.

---

## Key Findings

**1. Demographics alone are near-useless predictors.**  
Linear and logistic regression models trained only on demographic features (gender, education, marital status, age) produce near-zero R² scores. The models cannot reliably identify defaulters from demographics alone.

**2. Credit behavior is where the signal lives.**  
Models trained on credit utilization features (payment history, bill amounts, repayment amounts) significantly outperform demographic-only models across all model types.

**3. KNN captures something regression misses.**  
KNN classifiers modestly outperform regression on the demographic-only subset, suggesting weak but non-zero cluster structure in the demographic data — though the absolute performance remains low.

**4. Default risk is distributed uniformly across demographic groups.**  
Correlation analysis across demographic categories shows no meaningful concentration of default risk in any single demographic group. This finding has direct implications for responsible credit modeling: demographic features add little predictive power while introducing fairness risk.

**5. The prediction problem should be reframed.**  
The original framing ("predict creditworthiness from demographics") is both less accurate and less equitable than a behavioral framing. A model focused on repayment history and credit utilization is both more predictive and less prone to demographic bias.

---

## Results Summary

| Model | Features | Test Accuracy | Recall (default class) |
|---|---|---|---|
| Logistic Regression | All | ~0.78 | ~0.23 |
| Logistic Regression | Demographics only | ~0.77 | ~0.00 |
| Logistic Regression | Credit behavior only | ~0.78 | ~0.22 |
| KNN (k=3) | All | ~0.80 | ~0.34 |
| KNN (k=3) | Demographics only | ~0.78 | ~0.05 |
| KNN (k=3) | Credit behavior only | ~0.81 | ~0.35 |

*Note: Low recall on the default class reflects class imbalance (~22% default rate) and the inherent difficulty of the prediction problem. Models with high overall accuracy may still fail to identify actual defaulters.*

---

## Responsible AI Takeaway

This analysis deliberately reframes a common fintech ML problem. Models that incorporate demographic features to predict creditworthiness risk encoding historical inequities without meaningful predictive benefit. The data here suggest that a demographic-blind model based on behavioral financial signals is both more accurate and more equitable — and that the right evaluation metric for this use case is recall on the default class, not overall accuracy.

---

## Stack

```
Python 3.x
pandas
numpy
scikit-learn (LinearRegression, LogisticRegression, KNeighborsClassifier)
matplotlib
```

---

## Usage

```bash
git clone https://github.com/slam-wise/credit-default-risk-analysis.git
cd credit-default-risk-analysis
pip install -r requirements.txt
jupyter notebook analysis.ipynb
```

Dataset available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).

---

## Author

slamwise

[github.com/slam-wise](https://github.com/slam-wise)
