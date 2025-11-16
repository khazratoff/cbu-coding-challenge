#!/usr/bin/env python3
"""
credit_pipeline.py
Full training pipeline: preprocessing -> PCA -> SMOTEENN -> LightGBM
Includes SHAP explainability and saving artifacts.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, classification_report

from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
import shap

# -------------------------------
# Configuration
# -------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
PCA_VARIANCE = 0.95  # keep 95% variance
MODEL_OUTPUT = "final_lgbm.pkl"
PREPROCESSOR_OUTPUT = "preprocessor.pkl"
SHAP_PLOT_OUTPUT = "shap_summary.png"

# -------------------------------
# Helper: load your data here
# -------------------------------
# Replace this with actual loading; expects df with columns and 'default' target
# Example:
# df = pd.read_csv("data/credit_data.csv")

# For testing purpose raise if df not defined
from data_pipeline import load_data
try:
    df = load_data()
except NameError:
    raise RuntimeError("Please set `df` variable to your pandas DataFrame before running the script.")

# -------------------------------
# Features & target
# -------------------------------
TARGET = 'default'
features = [c for c in df.columns if c != TARGET]

X = df[features].copy()
y = df[TARGET].astype(int).copy()

# -------------------------------
# Identify numeric / categorical columns
# -------------------------------
num_features = X.select_dtypes(include=["number"]).columns.tolist()
cat_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

# Remove target-col-like from features if erroneously included
if TARGET in num_features:
    num_features.remove(TARGET)
if TARGET in cat_features:
    cat_features.remove(TARGET)

print(f"Numeric features: {len(num_features)}  Categorical features: {len(cat_features)}")

# -------------------------------
# Train/test split (important: do it BEFORE resampling)
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

# -------------------------------
# Preprocessor
# -------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        # ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_features)
    ],
    remainder="drop"
)

# -------------------------------
# Build imbalanced pipeline (preprocess -> PCA -> SMOTEENN -> LGBM)
# Note: Use imblearn Pipeline so resampling step is supported
# -------------------------------
pca = PCA(n_components=PCA_VARIANCE, svd_solver="full")
resampler = SMOTEENN(random_state=RANDOM_STATE)

lgbm = LGBMClassifier(
    n_estimators=1500,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    num_leaves=60,
    min_child_samples=40,
    reg_alpha=3,
    reg_lambda=3,
    class_weight="balanced",
    random_state=RANDOM_STATE
)

pipeline = ImbPipeline(steps=[
    ("preprocess", preprocessor),
    ("pca", pca),
    ("resample", resampler),
    ("model", lgbm)
])

# -------------------------------
# Train
# -------------------------------
print("Training pipeline ...")
pipeline.fit(X_train, y_train)

# -------------------------------
# Evaluate
# -------------------------------
preds_proba = pipeline.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, preds_proba)
print(f"Test AUC: {auc:.4f}")

# Optional: classification report at threshold 0.5
preds = (preds_proba >= 0.5).astype(int)
print(classification_report(y_test, preds))

# ROC curve save
fpr, tpr, thr = roc_curve(y_test, preds_proba)
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
plt.plot([0,1],[0,1], linestyle='--', alpha=0.6)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.close()

# -------------------------------
# Save artifacts
# -------------------------------
joblib.dump(pipeline, MODEL_OUTPUT)
print(f"Saved pipeline to {MODEL_OUTPUT}")

# Save preprocessor separately if needed
joblib.dump(preprocessor, PREPROCESSOR_OUTPUT)
print(f"Saved preprocessor to {PREPROCESSOR_OUTPUT}")

# -------------------------------
# SHAP explainability
# -------------------------------
print("Computing SHAP values (may take some time)...")

# We need to extract the fitted LGBM and the transformed training matrix for SHAP
fitted_model = pipeline.named_steps['model']
preproc_step = pipeline.named_steps['preprocess']

# Transform a subset for speed and stability
X_train_trans = preproc_step.transform(X_train)

# Use TreeExplainer
explainer = shap.TreeExplainer(fitted_model)
shap_values = explainer.shap_values(X_train_trans,)

# Create feature names for transformed matrix
try:
    transformed_feature_names = preproc_step.get_feature_names_out()
except Exception:
    # Fallback: build names
    num_names = num_features
    cat_names = list(preproc_step.named_transformers_["cat"].get_feature_names_out(cat_features)) if len(cat_features)>0 else []
    transformed_feature_names = np.array(num_names + cat_names)

# Summary plot
plt.figure(figsize=(10,8))
shap.summary_plot(shap_values, X_train_trans, feature_names=transformed_feature_names, show=False)
plt.tight_layout()
plt.savefig(SHAP_PLOT_OUTPUT)
plt.close()
print(f"Saved SHAP summary to {SHAP_PLOT_OUTPUT}")

# Optional: aggregate SHAP to original features if one-hot expanded
# This part aggregates OHE names back to base features (if OHE used)
try:
    tf_names = list(transformed_feature_names)
    shap_vals_mean = np.mean(np.abs(shap_values), axis=0)
    agg = {}
    for nm, val in zip(tf_names, shap_vals_mean):
        if '__' in nm:
            # sklearn get_feature_names_out uses format: "cat__<col>_<level>" or similar
            base = nm.split("__")[1] if '__' in nm else nm
            # sometimes OneHotEncoder uses 'col_level'
            if '_' in base and base.split('_')[0] in cat_features:
                base = base.split('_')[0]
        else:
            base = nm
        agg[base] = agg.get(base, 0) + val
    agg_df = pd.DataFrame([ (k,v) for k,v in agg.items() ], columns=["feature","shap_abs_mean"]).sort_values(by='shap_abs_mean', ascending=False)
    agg_df.to_csv('shap_aggregated.csv', index=False)
    print('Saved aggregated SHAP importances to shap_aggregated.csv')
except Exception as e:
    print('Failed to aggregate SHAP:', e)

print('Done.')