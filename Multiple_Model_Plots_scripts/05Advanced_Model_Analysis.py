# ============================================================
# AI PIPELINE FOR:
# Advanced Model Analysis (Confusion Matrices & Feature Importance)
#
# INPUT:
#   - Saved models (.pkl files)
#   - breed_label_encoder.pkl
#   - ml_input.raw & breed_labels.txt (to recreate the test set)
#
# OUTPUT:
#   - Confusion Matrix plots for SVM, XGBoost, CatBoost
#   - Top Important SNPs CSVs and Bar Plots for XGBoost, CatBoost
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# ============================================================
# STEP 1: RECREATE TEST DATASET
# ============================================================
print("\nLoading data to recreate the test set...")

geno = pd.read_csv("ml_input.raw", sep=r"\s+")
pheno = pd.read_csv("breed_labels.txt", sep="\t")

# Clean IDs
geno["IID"] = geno["IID"].astype(str).str.strip()
pheno["IID"] = pheno["IID"].astype(str).str.strip()

# Merge
data = pd.merge(geno, pheno, on="IID")

meta_cols = ["FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE"]
snp_cols = [c for c in data.columns if c not in meta_cols + ["Breed"]]

X = data[snp_cols]
X = X.fillna(X.mean())
y = data["Breed"]

# Load the previously fitted LabelEncoder
try:
    le = joblib.load("breed_label_encoder.pkl")
except FileNotFoundError:
    print("Error: 'breed_label_encoder.pkl' not found.")
    exit()

y_encoded = le.transform(y)

# Split data exactly the same way as in the training script
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print(f"Test set successfully recreated with {X_test.shape[0]} samples.")

# ============================================================
# STEP 2: PLOT CONFUSION MATRICES FOR TOP MODELS
# ============================================================
top_models = ["SVM", "XGBoost", "CatBoost"]

print("\nGenerating Confusion Matrices...")

for model_name in top_models:
    try:
        model = joblib.load(f"{model_name}_breed_classifier.pkl")
        y_pred = model.predict(X_test)
        
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt="d", 
            cmap="Blues", 
            xticklabels=le.classes_, 
            yticklabels=le.classes_
        )
        
        plt.title(f"Confusion Matrix: {model_name}", fontsize=14)
        plt.xlabel("Predicted Breed", fontsize=12)
        plt.ylabel("Actual Breed", fontsize=12)
        plt.tight_layout()
        
        cm_filename = f"confusion_matrix_{model_name}.png"
        plt.savefig(cm_filename, dpi=300)
        plt.close()
        print(f"--> Saved {cm_filename}")
        
    except FileNotFoundError:
        print(f"--> Warning: Model file for {model_name} not found. Skipping...")

# ============================================================
# STEP 3: EXTRACT FEATURE IMPORTANCE (SNPs)
# ============================================================
tree_models = ["XGBoost", "CatBoost"]

print("\nExtracting Top Important SNPs...")

for model_name in tree_models:
    try:
        model = joblib.load(f"{model_name}_breed_classifier.pkl")
        
        # Extract feature importances
        importances = model.feature_importances_
        
        # Create DataFrame mapping SNPs to their importance scores
        importance_df = pd.DataFrame({
            "SNP": X.columns,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)
        
        # Save full list to CSV
        csv_filename = f"top_snps_{model_name}.csv"
        importance_df.to_csv(csv_filename, index=False)
        
        # Plot Top 20
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df.head(20), x="Importance", y="SNP", palette="Reds_r")
        plt.title(f"Top 20 Most Important SNPs ({model_name})", fontsize=14)
        plt.xlabel("Importance Score", fontsize=12)
        plt.ylabel("SNP Marker", fontsize=12)
        plt.tight_layout()
        
        plot_filename = f"feature_importance_{model_name}.png"
        plt.savefig(plot_filename, dpi=300)
        plt.close()
        print(f"--> Saved {csv_filename} and {plot_filename}")
        
    except FileNotFoundError:
        print(f"--> Warning: Model file for {model_name} not found. Skipping...")