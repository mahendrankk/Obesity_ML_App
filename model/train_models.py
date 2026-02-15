# Training script for ML Assignment 2
# This script trains 6 different ML models on the Obesity dataset
# and saves the trained models and results
#
# Project-specific notes:
# - Author: Student submission (ML Assignment 2)

import os
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# paths setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(PROJECT_DIR, "ObesityDataSet_raw_and_data_sinthetic.csv")
SAVE_DIR = BASE_DIR  # save everything inside model/ folder

# ========== STEP 1: Load and preprocess data ==========
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print("Classes:", df["NObeyesdad"].nunique())

# separate features and target
X = df.drop(columns=["NObeyesdad"])
y = df["NObeyesdad"]

# encode the target variable (convert text labels to numbers)
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)
n_classes = len(le_target.classes_)

# find categorical and numerical columns
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(include=["number"]).columns.tolist()
print("Categorical columns:", cat_cols)
print("Numerical columns:", num_cols)

# encode categorical features using LabelEncoder
feature_label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    feature_label_encoders[col] = le

# scale numerical features using StandardScaler
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# ========== STEP 2: Split into train and test ==========
# using 80-20 split with stratify to keep class balance
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y_encoded, np.arange(len(y_encoded)),
    test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# ========== STEP 3: Define all 6 models ==========
model_registry = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "kNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost (Ensemble)": XGBClassifier(
        n_estimators=100, random_state=42,
        use_label_encoder=False, eval_metric="mlogloss"
    ),
}

# ========== STEP 4: Train each model and calculate metrics ==========
all_results = {}
print("\n" + "=" * 60)
print("TRAINING MODELS...")
print("=" * 60)

for name, model in model_registry.items():
    print(f"\nTraining: {name}")
    model.fit(X_train, y_train)

    # make predictions
    y_pred = model.predict(X_test)

    # get probability scores (needed for AUC calculation)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
    else:
        y_prob = np.zeros((len(y_test), n_classes))

    # calculate all 6 metrics
    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
    except:
        auc = 0.0
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    metrics = {
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "MCC": round(mcc, 4),
    }
    all_results[name] = metrics

    print(f"  Accuracy: {acc:.4f}")
    print(f"  AUC: {auc:.4f}")
    print(f"  F1: {f1:.4f}")

    # save the trained model
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
    joblib.dump(model, os.path.join(SAVE_DIR, f"{safe_name}.joblib"))
    print(f"  Model saved!")

# ========== STEP 5: Save all artifacts ==========
print("\nSaving preprocessing artifacts...")

joblib.dump(le_target, os.path.join(SAVE_DIR, "label_encoder_target.joblib"))
joblib.dump(feature_label_encoders, os.path.join(SAVE_DIR, "label_encoders_features.joblib"))
joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler.joblib"))
joblib.dump(cat_cols, os.path.join(SAVE_DIR, "cat_cols.joblib"))
joblib.dump(num_cols, os.path.join(SAVE_DIR, "num_cols.joblib"))
joblib.dump(list(X.columns), os.path.join(SAVE_DIR, "feature_columns.joblib"))

# save test data (raw/unencoded) so the streamlit app can use it
test_df = df.iloc[idx_test].reset_index(drop=True)
test_df.to_csv(os.path.join(SAVE_DIR, "test_data.csv"), index=False)

# save metrics as csv
metrics_df = pd.DataFrame(all_results).T
metrics_df.index.name = "ML Model Name"
metrics_df.to_csv(os.path.join(SAVE_DIR, "metrics.csv"))

# print final comparison
print("\n" + "=" * 60)
print("RESULTS COMPARISON")
print("=" * 60)
print(metrics_df.to_string())
print("\nDone! All models saved to:", SAVE_DIR)
