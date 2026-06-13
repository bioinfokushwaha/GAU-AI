# ============================================================
# AI PIPELINE FOR:
# Comprehensive Model Metric Visualizations
#
# OUTPUTS:
#   - plot_bar_comparison.png
#   - plot_heatmap.png
#   - plot_radar_chart.png
#   - plot_roc_curve.png
#   - plot_pr_curve.png
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import warnings
warnings.filterwarnings('ignore')

print("\nLoading model_evaluation_metrics.csv...")
try:
    df = pd.read_csv("model_evaluation_metrics.csv")
except FileNotFoundError:
    print("Error: 'model_evaluation_metrics.csv' not found. Please run your training script first.")
    exit()

# ============================================================
# 1. GROUPED BAR COMPARISON PLOT
# ============================================================
print("Generating Bar Comparison Plot...")
plt.figure(figsize=(14, 6))

# Exclude Log Loss since its scale is vastly different (lower is better, unbounded vs 0-1)
df_bar = df.drop(columns=["Log_Loss"])
df_melted = df_bar.melt(id_vars="Model", var_name="Metric", value_name="Score")

sns.barplot(data=df_melted, x="Metric", y="Score", hue="Model", palette="viridis")
plt.title("Model Comparison: Classification Metrics", fontsize=16)
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
plt.ylim(0.85, 1.0) # Zoomed in for better visibility of high-performing metrics
plt.tight_layout()
plt.savefig("plot_bar_comparison.png", dpi=300)
plt.close()

# ============================================================
# 2. HEATMAP
# ============================================================
print("Generating Metrics Heatmap...")
plt.figure(figsize=(10, 6))
df_heat = df.set_index("Model")

sns.heatmap(df_heat, annot=True, cmap="YlGnBu", fmt=".4f", linewidths=.5)
plt.title("Model Evaluation Metrics Heatmap", fontsize=16)
plt.tight_layout()
plt.savefig("plot_heatmap.png", dpi=300)
plt.close()

# ============================================================
# 3. RADAR CHART (SPIDER CHART)
# ============================================================
print("Generating Radar Chart...")
metrics = df_bar.columns[1:].tolist()
N = len(metrics)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=12)

for i, row in df_bar.iterrows():
    values = row[metrics].tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=row['Model'])
    ax.fill(angles, values, alpha=0.1)

plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.title("Model Performance Radar Chart", size=16, y=1.1)
plt.tight_layout()
plt.savefig("plot_radar_chart.png", dpi=300)
plt.close()

# ============================================================
# 4. MULTI-CLASS ROC AND PR CURVES
# ============================================================
print("\nRecreating test dataset to calculate Multi-Class ROC & PR Curves...")
try:
    geno = pd.read_csv("ml_input.raw", sep=r"\s+")
    pheno = pd.read_csv("breed_labels.txt", sep="\t")
    geno["IID"] = geno["IID"].astype(str).str.strip()
    pheno["IID"] = pheno["IID"].astype(str).str.strip()
    data = pd.merge(geno, pheno, on="IID")
    
    meta_cols = ["FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE"]
    snp_cols = [c for c in data.columns if c not in meta_cols + ["Breed"]]
    X = data[snp_cols].fillna(data[snp_cols].mean())
    y = data["Breed"]
    
    le = joblib.load("breed_label_encoder.pkl")
    y_encoded = le.transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # Binarize labels for multi-class curve calculations (One-vs-Rest Micro-Averaging)
    classes = range(len(le.classes_))
    y_test_bin = label_binarize(y_test, classes=classes)
    
    fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
    fig_pr, ax_pr = plt.subplots(figsize=(10, 8))

    print("Calculating Micro-Averaged curves for each model...")
    for model_name in df['Model']:
        try:
            model = joblib.load(f"{model_name}_breed_classifier.pkl")
            y_proba = model.predict_proba(X_test)
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.4f})')
            
            # PR Curve
            precision, recall, _ = precision_recall_curve(y_test_bin.ravel(), y_proba.ravel())
            pr_auc = average_precision_score(y_test_bin, y_proba, average="micro")
            ax_pr.plot(recall, precision, lw=2, label=f'{model_name} (AP = {pr_auc:.4f})')
            
        except Exception as e:
            print(f"--> Warning: Could not generate curves for {model_name}. Reason: {e}")

    # Save ROC Curve
    ax_roc.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    ax_roc.set(xlim=[0.0, 1.0], ylim=[0.0, 1.05], xlabel='False Positive Rate', ylabel='True Positive Rate', title='Micro-Average ROC Curve Comparison')
    ax_roc.legend(loc="lower right", fontsize=12)
    fig_roc.savefig("plot_roc_curve.png", dpi=300, bbox_inches='tight')

    # Save PR Curve
    ax_pr.set(xlim=[0.0, 1.0], ylim=[0.0, 1.05], xlabel='Recall', ylabel='Precision', title='Micro-Average Precision-Recall Curve Comparison')
    ax_pr.legend(loc="lower left", fontsize=12)
    fig_pr.savefig("plot_pr_curve.png", dpi=300, bbox_inches='tight')

    print("\nAll visualizations saved successfully in your directory!")

except Exception as e:
    print(f"\nCould not generate ROC/PR curves. Error recreating test data: {e}")