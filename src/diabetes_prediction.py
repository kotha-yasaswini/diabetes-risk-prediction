"""
Diabetes Risk Prediction
Dataset: Pima Indians Diabetes Database
Business Goal: Maximize detection of real diabetic patients (Recall)
               while keeping false alarms low (Precision)
               — with a lab capacity constraint of 30% walk-ins per day.
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (saves PNGs)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)

RANDOM_STATE = 42
DATA_PATH    = r"c:\Users\APPLE\Desktop\dataset\diabetes.csv"
OUT_DIR      = r"C:\Users\APPLE\.gemini\antigravity\brain\bcd7942c-21c1-4773-ae71-eab2d2ea4573\scratch"

# =============================================================================
# 2. LOAD & INSPECT DATA
# =============================================================================
print("=" * 60)
print("STEP 1 — LOAD AND INSPECT DATA")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\nShape : {df.shape}  (rows x columns)")
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:\n", df.dtypes)
print("\nBasic statistics:")
print(df.describe())
print("\nMissing values (NaN):", df.isnull().sum().sum())

# =============================================================================
# 3. HANDLE HIDDEN MISSING VALUES (zeros in medical columns)
# =============================================================================
print("\n" + "=" * 60)
print("STEP 2 — HANDLE HIDDEN MISSING VALUES (zeros -> NaN)")
print("=" * 60)

zero_not_allowed = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

print("\nZero counts BEFORE replacement:")
for col in zero_not_allowed:
    n_zeros = (df[col] == 0).sum()
    pct = n_zeros / len(df) * 100
    print(f"  {col:<30} {n_zeros:>4} zeros  ({pct:.1f}%)")

df[zero_not_allowed] = df[zero_not_allowed].replace(0, np.nan)

for col in zero_not_allowed:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)   # direct assignment — pandas 3.x safe
    print(f"  {col:<30} imputed with median = {median_val:.2f}")

print("\nMissing values AFTER imputation:", df.isnull().sum().sum())

# =============================================================================
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3 — EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nOutcome distribution:")
print(df["Outcome"].value_counts())
print(f"\nClass balance: {df['Outcome'].mean()*100:.1f}% diabetic")

# --- Plot 1: Outcome Distribution ---
fig, ax = plt.subplots(figsize=(5, 4))
df["Outcome"].value_counts().plot(kind="bar", color=["steelblue", "salmon"], ax=ax)
ax.set_title("Outcome Distribution")
ax.set_xlabel("Outcome (0=No, 1=Yes)")
ax.set_ylabel("Count")
ax.set_xticklabels(["Not Diabetic (0)", "Diabetic (1)"], rotation=0)
for p in ax.patches:
    ax.annotate(str(int(p.get_height())),
                (p.get_x() + p.get_width() / 2, p.get_height() + 5),
                ha="center")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}\\plot_01_outcome_distribution.png", dpi=120)
plt.close()
print("  Saved: plot_01_outcome_distribution.png")

# --- Plot 2: Feature Histograms ---
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
features = df.columns.tolist()
for i, col in enumerate(features):
    ax = axes[i // 3][i % 3]
    df[col].hist(bins=30, ax=ax, color="steelblue", edgecolor="white")
    ax.set_title(col)
    ax.set_ylabel("Count")
plt.suptitle("Feature Distributions (after imputation)", fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}\\plot_02_feature_histograms.png", dpi=120)
plt.close()
print("  Saved: plot_02_feature_histograms.png")

# --- Plot 3: Correlation Heatmap ---
fig, ax = plt.subplots(figsize=(9, 7))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            mask=mask, linewidths=0.5, ax=ax)
ax.set_title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}\\plot_03_correlation_heatmap.png", dpi=120)
plt.close()
print("  Saved: plot_03_correlation_heatmap.png")

# --- Plot 4: Boxplots by Outcome ---
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
feature_cols = [c for c in df.columns if c != "Outcome"]
for i, col in enumerate(feature_cols):
    ax = axes[i // 4][i % 4]
    df.boxplot(column=col, by="Outcome", ax=ax,
               boxprops=dict(color="steelblue"),
               medianprops=dict(color="red"))
    ax.set_title(col)
    ax.set_xlabel("Outcome")
plt.suptitle("Feature Distribution by Outcome", fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}\\plot_04_boxplots_by_outcome.png", dpi=120)
plt.close()
print("  Saved: plot_04_boxplots_by_outcome.png")

# =============================================================================
# 5. FEATURE SCALING & TRAIN/TEST SPLIT
# =============================================================================
print("\n" + "=" * 60)
print("STEP 4 — FEATURE SCALING & STRATIFIED TRAIN/TEST SPLIT")
print("=" * 60)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain set : {X_train.shape[0]} rows  "
      f"(diabetic: {y_train.sum()}, non-diabetic: {(y_train==0).sum()})")
print(f"Test  set : {X_test.shape[0]} rows  "
      f"(diabetic: {y_test.sum()}, non-diabetic: {(y_test==0).sum()})")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print("\nStandardScaler applied — features now have mean~0, std~1")

# =============================================================================
# 6. TRAIN THREE MODELS
# =============================================================================
print("\n" + "=" * 60)
print("STEP 5 — TRAIN KNN, LOGISTIC REGRESSION, DECISION TREE")
print("=" * 60)

models = {
    "KNN (k=5)":               KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression":     LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree":           DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
}

results = {}
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred  = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)[:, 1]

    results[name] = {
        "model":     model,
        "y_pred":    y_pred,
        "y_proba":   y_proba,
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_proba),
    }
    print(f"  [{name}]  trained OK")

# =============================================================================
# 7. EVALUATE — STANDARD THRESHOLD (0.5)
# =============================================================================
print("\n" + "=" * 60)
print("STEP 6 — EVALUATION AT DEFAULT THRESHOLD = 0.50")
print("=" * 60)

print(f"\n{'Model':<25} {'Accuracy':>10} {'Precision':>10} "
      f"{'Recall':>8} {'F1':>8} {'ROC-AUC':>8}")
print("-" * 73)
for name, r in results.items():
    print(f"{name:<25} {r['accuracy']:>10.3f} {r['precision']:>10.3f} "
          f"{r['recall']:>8.3f} {r['f1']:>8.3f} {r['roc_auc']:>8.3f}")

# --- Confusion Matrices ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, r) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, r["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["True 0", "True 1"])
    ax.set_title(f"{name}\nAcc={r['accuracy']:.2f}  Recall={r['recall']:.2f}")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
plt.suptitle("Confusion Matrices (threshold = 0.50)", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}\\plot_05_confusion_matrices.png", dpi=120)
plt.close()
print("\n  Saved: plot_05_confusion_matrices.png")

# --- ROC Curves ---
fig, ax = plt.subplots(figsize=(7, 6))
colors = ["steelblue", "darkorange", "green"]
for (name, r), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
    ax.plot(fpr, tpr, label=f"{name}  AUC={r['roc_auc']:.3f}", color=color)
ax.plot([0,1],[0,1],"k--", linewidth=0.8, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — All Models")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}\\plot_06_roc_curves.png", dpi=120)
plt.close()
print("  Saved: plot_06_roc_curves.png")

# =============================================================================
# 8. THRESHOLD TUNING (Business Constraint: flag only 30% of patients)
# =============================================================================
print("\n" + "=" * 60)
print("STEP 7 — THRESHOLD TUNING (flag top 30% most likely diabetic)")
print("=" * 60)
print("\nBusiness rule: clinic can test only 30% of walk-ins per day.")
print("Strategy: rank patients by predicted probability, flag top 30%.\n")

n_test = len(y_test)
top_30_cutoff = int(np.ceil(n_test * 0.30))

print(f"{'Model':<25} {'Threshold':>10} {'Precision@30%':>14} "
      f"{'Recall@30%':>12} {'Flagged':>8}")
print("-" * 73)

tuned_results = {}
for name, r in results.items():
    proba = r["y_proba"]
    sorted_proba = np.sort(proba)[::-1]
    threshold    = sorted_proba[top_30_cutoff - 1]
    y_pred_tuned = (proba >= threshold).astype(int)

    n_flagged  = y_pred_tuned.sum()
    prec_tuned = precision_score(y_test, y_pred_tuned)
    rec_tuned  = recall_score(y_test, y_pred_tuned)
    f1_tuned   = f1_score(y_test, y_pred_tuned)

    tuned_results[name] = {
        "threshold": threshold,
        "precision": prec_tuned,
        "recall":    rec_tuned,
        "f1":        f1_tuned,
        "flagged":   n_flagged,
        "y_pred":    y_pred_tuned,
    }
    print(f"{name:<25} {threshold:>10.3f} {prec_tuned:>14.3f} "
          f"{rec_tuned:>12.3f} {n_flagged:>8}")

# --- Confusion Matrices at tuned threshold ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, r) in zip(axes, tuned_results.items()):
    cm = confusion_matrix(y_test, r["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", ax=ax,
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["True 0", "True 1"])
    ax.set_title(f"{name}\nPrec={r['precision']:.2f}  Recall={r['recall']:.2f}")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
plt.suptitle("Confusion Matrices (top-30% threshold, business-tuned)", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}\\plot_07_confusion_matrices_tuned.png", dpi=120)
plt.close()
print("\n  Saved: plot_07_confusion_matrices_tuned.png")

# =============================================================================
# 9. COMPREHENSIVE METRIC SUMMARY TABLE
# =============================================================================
print("\n" + "=" * 60)
print("STEP 8 — FULL METRIC COMPARISON TABLE")
print("=" * 60)

rows = []
for name in results:
    r  = results[name]
    rt = tuned_results[name]
    rows.append({
        "Model":        name,
        "Acc (0.5)":    round(r["accuracy"],  3),
        "Prec (0.5)":   round(r["precision"], 3),
        "Recall (0.5)": round(r["recall"],    3),
        "F1 (0.5)":     round(r["f1"],        3),
        "ROC-AUC":      round(r["roc_auc"],   3),
        "Prec (30%)":   round(rt["precision"],3),
        "Recall (30%)": round(rt["recall"],   3),
        "F1 (30%)":     round(rt["f1"],       3),
    })

summary_df = pd.DataFrame(rows).set_index("Model")
print(summary_df.to_string())

# =============================================================================
# 10. WINNING MODEL SELECTION
# =============================================================================
print("\n" + "=" * 60)
print("STEP 9 — WINNING MODEL SELECTION (Business Criteria)")
print("=" * 60)

best_name = max(tuned_results,
                key=lambda n: (tuned_results[n]["recall"], results[n]["roc_auc"]))

print(f"""
Business Priority Ranking:
  1. Recall (catching real diabetics)  -- HIGHEST priority
  2. Precision (avoid wasting test slots)
  3. ROC-AUC (overall discrimination)

WINNER: {best_name}
  Threshold (top-30% rule) : {tuned_results[best_name]['threshold']:.3f}
  Precision@30%            : {tuned_results[best_name]['precision']:.3f}
  Recall@30%               : {tuned_results[best_name]['recall']:.3f}
  F1@30%                   : {tuned_results[best_name]['f1']:.3f}
  ROC-AUC                  : {results[best_name]['roc_auc']:.3f}

Interpretation:
  Flags the {tuned_results[best_name]['flagged']} highest-risk patients out of {n_test} tested.
  Recall@30%    = {tuned_results[best_name]['recall']:.1%} of all true diabetics are caught.
  Precision@30% = {tuned_results[best_name]['precision']:.1%} of flagged patients are truly diabetic.
""")

# --- Bar chart: Model comparison ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

model_names = list(tuned_results.keys())
recalls  = [tuned_results[n]["recall"]    for n in model_names]
precs    = [tuned_results[n]["precision"] for n in model_names]
aucs     = [results[n]["roc_auc"]         for n in model_names]

x = np.arange(len(model_names))
width = 0.25
bars1 = axes[0].bar(x - width, recalls, width, label="Recall@30%",    color="steelblue")
bars2 = axes[0].bar(x,         precs,   width, label="Precision@30%", color="salmon")
bars3 = axes[0].bar(x + width, aucs,    width, label="ROC-AUC",       color="green")
axes[0].set_xticks(x)
axes[0].set_xticklabels(model_names, rotation=12, ha="right")
axes[0].set_ylim(0, 1.15)
axes[0].set_title("Model Comparison @ 30% Threshold")
axes[0].legend()
axes[0].set_ylabel("Score")
for bar in list(bars1) + list(bars2) + list(bars3):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.01,
                 f"{bar.get_height():.2f}", ha="center", fontsize=8)

# Winner metrics bar
categories = ["Accuracy", "Precision@30%", "Recall@30%", "F1@30%", "ROC-AUC"]
values_w = [
    results[best_name]["accuracy"],
    tuned_results[best_name]["precision"],
    tuned_results[best_name]["recall"],
    tuned_results[best_name]["f1"],
    results[best_name]["roc_auc"],
]
colors_w = ["steelblue","salmon","green","purple","orange"]
axes[1].barh(categories, values_w, color=colors_w)
axes[1].set_xlim(0, 1.1)
axes[1].set_title(f"Winner: {best_name} — All Metrics")
for i, v in enumerate(values_w):
    axes[1].text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}\\plot_08_model_comparison.png", dpi=120)
plt.close()
print("  Saved: plot_08_model_comparison.png")

# =============================================================================
# 11. CLASSIFICATION REPORT FOR WINNER
# =============================================================================
print("\n" + "=" * 60)
print(f"STEP 10 — CLASSIFICATION REPORT: {best_name}")
print("=" * 60)
print(classification_report(y_test, tuned_results[best_name]["y_pred"],
                             target_names=["Not Diabetic", "Diabetic"]))

# =============================================================================
# 12. SAVE SUMMARY CSV
# =============================================================================
summary_df.to_csv(f"{OUT_DIR}\\model_summary.csv")
print(f"Summary CSV saved to: {OUT_DIR}\\model_summary.csv")
print("\n" + "=" * 60)
print("PROJECT COMPLETE — All outputs saved!")
print("=" * 60)
