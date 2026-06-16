#!/usr/bin/env python3

###############################################################
# GAU-AI
# Breed Identification + Breed Composition Prediction
# python predict_gauai.py --geno new_animals.raw --modeldir GauAI_Models --outdir RESULTS
###############################################################

import pandas as pd
import numpy as np
import argparse
import os
import sys
import joblib
###############################################################
# ARGUMENTS
###############################################################

parser = argparse.ArgumentParser(
    description="GAU-AI Breed Prediction Pipeline"
)

parser.add_argument(
    "--geno",
    required=True,
    help="Input PLINK .raw file"
)

parser.add_argument(
    "--modeldir",
    default="GAUAI_MODELS",
    help="Directory containing trained models"
)

parser.add_argument(
    "--outdir",
    default="GAUAI_RESULTS",
    help="Output directory"
)

args = parser.parse_args()

###############################################################
# CREATE OUTPUT DIRECTORY
###############################################################

os.makedirs(
    args.outdir,
    exist_ok=True
)

###############################################################
# REQUIRED FILES
###############################################################

required_files = [

    "breed_classifier.pkl",

    "breed_composition_model.pkl",

    "breed_label_encoder.pkl",

    "training_snp_order.pkl",

    "composition_labels.pkl"

]

for file in required_files:

    path = os.path.join(
        args.modeldir,
        file
    )

    if not os.path.exists(path):

        print(
            f"\nERROR: Missing {path}"
        )

        sys.exit(1)

###############################################################
# LOAD MODELS
###############################################################

print("\nLoading models...")

breed_model = joblib.load(
    os.path.join(
        args.modeldir,
        "breed_classifier.pkl"
    )
)

composition_model = joblib.load(
    os.path.join(
        args.modeldir,
        "breed_composition_model.pkl"
    )
)

label_encoder = joblib.load(
    os.path.join(
        args.modeldir,
        "breed_label_encoder.pkl"
    )
)

training_snps = joblib.load(
    os.path.join(
        args.modeldir,
        "training_snp_order.pkl"
    )
)

breed_columns = joblib.load(
    os.path.join(
        args.modeldir,
        "composition_labels.pkl"
    )
)

print("Models loaded successfully")

###############################################################
# LOAD GENOTYPE DATA
###############################################################

print(
    f"\nLoading genotype file:\n{args.geno}"
)

geno = pd.read_csv(
    args.geno,
    sep=r"\s+"
)

print(
    f"Animals: {geno.shape[0]}"
)

###############################################################
# EXTRACT SAMPLE IDS
###############################################################

if "IID" not in geno.columns:

    print(
        "\nERROR: IID column not found"
    )

    sys.exit(1)

animal_ids = geno["IID"].astype(str)

###############################################################
# BUILD FEATURE MATRIX
###############################################################

print(
    "\nAligning SNPs..."
)

X = pd.DataFrame()

missing_snps = 0

for snp in training_snps:

    if snp in geno.columns:

        X[snp] = geno[snp]

    else:

        X[snp] = np.nan

        missing_snps += 1

print(
    f"Missing SNPs: {missing_snps:,}"
)

###############################################################
# NUMERIC CONVERSION
###############################################################

X = X.apply(
    pd.to_numeric,
    errors="coerce"
)

###############################################################
# IMPUTATION
###############################################################

X = X.fillna(
    X.mean()
)

X = X.fillna(0)

print(
    f"Feature matrix: {X.shape}"
)

###############################################################
# BREED PREDICTION
###############################################################

print(
    "\nPredicting breeds..."
)

breed_pred = breed_model.predict(
    X
)

predicted_breeds = (

    label_encoder
    .inverse_transform(
        breed_pred
    )
)

###############################################################
# CONFIDENCE SCORES
###############################################################

if hasattr(
    breed_model,
    "predict_proba"
):

    probs = breed_model.predict_proba(
        X
    )

    confidence = probs.max(
        axis=1
    )

else:

    confidence = np.repeat(
        np.nan,
        len(predicted_breeds)
    )

###############################################################
# SAVE BREED PREDICTIONS
###############################################################

breed_results = pd.DataFrame({

    "IID":
        animal_ids,

    "Predicted_Breed":
        predicted_breeds,

    "Confidence":
        np.round(
            confidence,
            4
        )
})

breed_results.to_csv(

    os.path.join(
        args.outdir,
        "breed_predictions.csv"
    ),

    index=False
)

###############################################################
# BREED COMPOSITION PREDICTION
###############################################################

print(
    "\nPredicting breed composition..."
)

composition_pred = (

    composition_model
    .predict(X)
)

###############################################################
# REMOVE NEGATIVE VALUES
###############################################################

composition_pred[
    composition_pred < 0
] = 0

###############################################################
# NORMALIZE TO SUM = 1
###############################################################

row_sum = composition_pred.sum(
    axis=1,
    keepdims=True
)

row_sum[
    row_sum == 0
] = 1

composition_pred = (
    composition_pred /
    row_sum
)

###############################################################
# CREATE COMPOSITION TABLE
###############################################################

composition_df = pd.DataFrame(

    composition_pred,

    columns=breed_columns
)

composition_df.insert(
    0,
    "IID",
    animal_ids
)

###############################################################
# CONVERT TO %
###############################################################

for col in breed_columns:

    composition_df[col] = (
        composition_df[col] * 100
    ).round(2)

###############################################################
# SAVE COMPOSITION
###############################################################

composition_df.to_csv(

    os.path.join(
        args.outdir,
        "breed_composition_predictions.csv"
    ),

    index=False
)

###############################################################
# TOP BREEDS SUMMARY
###############################################################

summary_rows = []

for i in range(
    len(composition_df)
):

    row = composition_df.iloc[i]

    values = row[
        breed_columns
    ].sort_values(
        ascending=False
    )

    summary_rows.append({

        "IID":
            row["IID"],

        "Breed1":
            values.index[0],

        "Breed1_%":
            round(
                values.iloc[0],
                2
            ),

        "Breed2":
            values.index[1],

        "Breed2_%":
            round(
                values.iloc[1],
                2
            ),

        "Breed3":
            values.index[2],

        "Breed3_%":
            round(
                values.iloc[2],
                2
            )
    })

summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(

    os.path.join(
        args.outdir,
        "breed_composition_summary.csv"
    ),

    index=False
)

###############################################################
# DISPLAY FIRST 10 ANIMALS
###############################################################

print(
    "\n===================================="
)

print(
    "Prediction Summary"
)

print(
    "===================================="
)

for i in range(
    min(
        10,
        len(breed_results)
    )
):

    print(
        f"\nAnimal: {breed_results.iloc[i]['IID']}"
    )

    print(
        f"Breed : {breed_results.iloc[i]['Predicted_Breed']}"
    )

    if not np.isnan(
        breed_results.iloc[i]['Confidence']
    ):

        print(
            f"Confidence: "
            f"{breed_results.iloc[i]['Confidence']:.3f}"
        )

###############################################################
# FINISHED
###############################################################

print(
    "\n===================================="
)

print(
    "GAU-AI PREDICTION COMPLETED"
)

print(
    "===================================="
)

print(
    "\nOutput Files:"
)

print(
    os.path.join(
        args.outdir,
        "breed_predictions.csv"
    )
)

print(
    os.path.join(
        args.outdir,
        "breed_composition_predictions.csv"
    )
)

print(
    os.path.join(
        args.outdir,
        "breed_composition_summary.csv"
    )
)
