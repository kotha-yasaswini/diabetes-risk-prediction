# 🩺 Diabetes Risk Prediction & Clinical Triage System
*Screening Decision Support System for Rural Health Clinics | Kaggle Pima Indians Diabetes Database*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit_Learn-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Executive Summary & Problem Statement
A rural health clinic with limited daily laboratory capacity can afford blood glucose confirmation tests for only **30% of daily walk-ins**[cite: 1, 12]. This project implements an end-to-end Machine Learning pipeline to triage patients based on routine physical measurements[cite: 1, 12]. 

The primary objective is to **maximize the detection of true diabetic individuals (Recall)** while maintaining an efficient positive predictive value (Precision) under the strict **30% operational testing threshold**[cite: 1, 12].

---

## 📊 Key Project Metrics & Dataset
* **Total Patients:** 768 records[cite: 18]
* **Features:** 8 physiological inputs[cite: 18]
* **Diabetic Prevalence:** 34.9% (268 positive cases)[cite: 18]
* **Operational Constraint:** 30% daily testing cap
* **Winning Model Recall:** **72.2%** (Decision Tree Classifier tuned to top-30% risk percentile)[cite: 21]

---

## 📁 Repository Structure
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
