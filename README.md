# 🩺 Diabetes Risk Prediction & Clinical Triage System
*Screening Decision Support System for Rural Health Clinics | Kaggle Pima Indians Diabetes Database*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit_Learn-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Executive Summary & Problem Statement
A rural health clinic with limited daily laboratory capacity can afford blood glucose confirmation tests for only **30% of daily walk-ins**. This project implements an end-to-end Machine Learning pipeline to triage patients based on routine physical measurements[cite: 12]. 

The primary objective is to **maximize the detection of true diabetic individuals (Recall)** while maintaining an efficient positive predictive value (Precision) under the strict **30% operational testing threshold**[cite: 12].

---

## 📊 Key Project Metrics & Dataset
* **Total Patients:** 768 records[cite: 18]
* **Features:** 8 physiological inputs[cite: 18]
* **Diabetic Prevalence:** 34.9% (268 positive cases)[cite: 18]
* **Operational Constraint:** 30% daily testing cap[cite: 12]
* **Winning Model Recall:** **72.2%** (Decision Tree Classifier tuned to top-30% risk percentile)[cite: 21]

---

## 🛠️ 1. Data Preprocessing & Hidden Missing Values
Standard statistical inspection revealed 0 explicit `NaN` values. However, clinical domain analysis identified that zero values in physiological indicators are biologically impossible. These hidden missing values were imputed using **feature-specific medians** to safeguard against heavy positive skewness[cite: 18]:

| Feature | Zero Count | % Missing | Imputation Strategy |
| :--- | :---: | :---: | :--- |
| **Insulin** | 374 | 48.7% | Median = 125.0 $\mu U/ml$[cite: 18] |
| **SkinThickness** | 227 | 29.6% | Median = 29.0 mm[cite: 18] |
| **BloodPressure** | 35 | 4.6% | Median = 72.0 mm Hg[cite: 18] |
| **BMI** | 11 | 1.4% | Median = 32.3 $kg/m^2$[cite: 18] |
| **Glucose** | 5 | 0.7% | Median = 117.0 mg/dl[cite: 18] |

---

## 📈 Exploratory Data Analysis & Feature Dynamics
Bivariate correlation analyses confirm that **Glucose** exhibits the strongest individual association with diabetic status ($r = 0.47$), followed by **BMI** ($r = 0.29$), **Age** ($r = 0.24$), and **Pregnancies** ($r = 0.22$)[cite: 18].

<p align="center">
  <img src="outputs/figures/plot_03_correlation_heatmap.png" width="600" alt="Correlation Heatmap">
</p>

---

## 🏆 Model Evaluation & 30% Threshold Tuning
In clinical triage, **False Negatives incur far greater harm than False Positives** (missing a diabetic patient delays critical intervention)[cite: 21]. Under the 30% testing capacity constraint, models were adjusted to rank patients by predicted risk score and flag the top 30%[cite: 20, 21].

| Model Architecture | Default Accuracy | Default ROC-AUC | Tuned Cutoff | Precision @ 30% | Recall @ 30% | F1 @ 30% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **KNN ($k=5$)** | 0.753 | 0.789 | 0.600 | 66.0% | 61.1% | 0.635[cite: 20] |
| **Logistic Regression** | 0.708 | **0.813** | 0.499 | 57.4% | 50.0% | 0.535[cite: 20] |
| **Decision Tree (depth=5)** | **0.760** | 0.761 | **0.517** | **63.9%** | **72.2% 🏆** | **0.678 🏆**[cite: 20] |

<p align="center">
  <img src="outputs/figures/plot_08_model_comparison.png" width="700" alt="Model Comparison">
</p>

---
📁 Repository Structure
```text
diabetes-risk-prediction/
├── figures/                    # Generated visualization plots
│   ├── plot_01_outcome_distribution.png
│   ├── plot_02_feature_histograms.png
│   ├── plot_03_correlation_heatmap.png
│   ├── plot_04_boxplots_by_outcome.png
│   ├── plot_05_confusion_matrices.png
│   ├── plot_06_roc_curves.png
│   ├── plot_07_confusion_matrices_tuned.png
│   └── plot_08_model_comparison.png
├── outputs/                    # Project reports & major deliverables
│   └── reports/
│       └── Diabetes_Risk_Prediction_Report.pdf
├── src/                        # Core Python scripts
│   ├── diabetes_prediction.py  # End-to-end ML training & evaluation pipeline
│   └── generate_pdf_report.py  # ReportLab script for generating clinical PDF
├── diabetes.csv                # Raw Kaggle Pima Indians Diabetes dataset
├── requirements.txt            # Python library dependencies
└── README.md                   # Project documentation


 **Clone the repository:**

   git clone [https://github.com/kotha-yasaswini/diabetes-risk-prediction.git](https://github.com/kotha-yasaswini/diabetes-risk-prediction.git)
