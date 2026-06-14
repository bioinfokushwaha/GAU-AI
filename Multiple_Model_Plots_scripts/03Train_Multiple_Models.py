# ============================================================
# AI PIPELINE FOR:
# Training Multiple Machine Learning Models for Breed Identification
#
# MODELS INCLUDED:
#   - Random Forest (RF)
#   - Support Vector Machine (SVM)
#   - K-Nearest Neighbors (KNN)
#   - XGBoost
#   - LightGBM
#   - CatBoost
#
# OUTPUT:
#   - Individual .pkl model files
#   - Accuracy summary report
# ============================================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, roc_auc_score, log_loss,
    matthews_corrcoef, cohen_kappa_score
)

# Import Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# ============================================================
# STEP 1: LOAD AND MERGE DATA
# ============================================================

print("\nLoading genotype and phenotype data...")

geno = pd.read_csv("ml_input.raw", sep=r"\s+")
pheno = pd.read_csv("breed_labels.txt", sep="\t")

# Clean IDs
geno["IID"] = geno["IID"].astype(str).str.strip()
pheno["IID"] = pheno["IID"].astype(str).str.strip()

# Merge
data = pd.merge(geno, pheno, on="IID")
print("\nMerged dataset shape:", data.shape)

# ============================================================
# STEP 2: EXTRACT FEATURES AND LABELS
# ============================================================

meta_cols = ["FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE"]
snp_cols = [c for c in data.columns if c not in meta_cols + ["Breed"]]

X = data[snp_cols]
# Replace missing values with column means
X = X.fillna(X.mean())

y = data["Breed"]

# ============================================================
# STEP 3: ENCODE LABELS
# ============================================================

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Save encoder so it can be reused for predictions later
joblib.dump(le, "breed_label_encoder.pkl")

print("\nBreed Classes:", le.classes_)

# ============================================================
# STEP 4: TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples : {X_test.shape[0]}")

# ============================================================
# STEP 5: INITIALIZE MODELS
# ============================================================

models = {
    "RandomForest": RandomForestClassifier(n_estimators=500, max_features="sqrt", n_jobs=-1, random_state=42),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_jobs=-1),
    "LightGBM": LGBMClassifier(random_state=42, n_jobs=-1),
    "CatBoost": CatBoostClassifier(iterations=500, verbose=0, random_state=42, thread_count=-1)
}

# ============================================================
# STEP 6: TRAIN, EVALUATE AND SAVE MODELS
# ============================================================

metrics_data = []

print("\nStarting Model Training...\n" + "="*40)

for name, model in models.items():
    print(f"Training {name}...")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
    lloss = log_loss(y_test, y_proba)
    mcc = matthews_corrcoef(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    
    metrics_data.append({
        "Model": name,
        "Accuracy": acc,
        "Weighted_F1": f1,
        "ROC_AUC": roc_auc,
        "Log_Loss": lloss,
        "MCC": mcc,
        "Cohen_Kappa": kappa
    })
    
    print(f"--> Accuracy: {acc:.4f} | F1: {f1:.4f} | Log Loss: {lloss:.4f}\n")
    
    # Save the model
    model_filename = f"{name}_breed_classifier.pkl"
    joblib.dump(model, model_filename)

# ============================================================
# STEP 7: SUMMARY OF RESULTS AND EXPORT METRICS
# ============================================================

print("="*40)
print("MODEL TRAINING COMPLETE - EVALUATION SUMMARY:")
print("="*40)

metrics_df = pd.DataFrame(metrics_data)
metrics_df = metrics_df.sort_values(by="Accuracy", ascending=False)

print(metrics_df.to_string(index=False))

# Export to CSV for plotting
metrics_df.to_csv("model_evaluation_metrics.csv", index=False)
print("\nMetrics exported successfully to model_evaluation_metrics.csv!")