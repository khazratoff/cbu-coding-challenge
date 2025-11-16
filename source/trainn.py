import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

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
# 1) SPLIT DATA (BEFORE SMOTEENN TO AVOID LEAKAGE)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
num_features = X.select_dtypes(include=["number"]).columns.tolist()
cat_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

# Remove target-col-like from features if erroneously included
if TARGET in num_features:
    num_features.remove(TARGET)
if TARGET in cat_features:
    cat_features.remove(TARGET)

# 2) PREPROCESSOR
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        # ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ],
    remainder="drop"
)

# 3) BUILD PIPELINE
model_pipeline = Pipeline([
    ("preprocess", preprocess),                  
    ("pca", PCA(n_components=0.95)),             # keep components that explain 95% variance
    ("smoteenn", SMOTEENN(random_state=42)),     # resampling AFTER preprocessing
    ("model", LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        max_depth=-1,
        random_state=42
    ))
])

# 4) TRAIN
model_pipeline.fit(X_train, y_train)

# 5) PREDICT
preds = model_pipeline.predict_proba(X_test)[:, 1]
print("AUC:", roc_auc_score(y_test, preds))
