# Streamlit app for Obesity Level Classification
# ML Assignment 2
#
# Project-specific notes:
# - Author: Student submission (ML Assignment 2)

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)

# page setup
st.set_page_config(page_title="Obesity Level Classification", page_icon="🏋️", layout="wide")

# paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# model files
MODEL_FILES = {
    "Logistic Regression": "Logistic_Regression.joblib",
    "Decision Tree": "Decision_Tree.joblib",
    "kNN": "kNN.joblib",
    "Naive Bayes": "Naive_Bayes.joblib",
    "Random Forest (Ensemble)": "Random_Forest_Ensemble.joblib",
    "XGBoost (Ensemble)": "XGBoost_Ensemble.joblib",
}


# load all the saved models and preprocessing stuff
@st.cache_resource
def load_models_and_data():
    models = {}
    for name, filename in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            models[name] = joblib.load(path)

    le_target = joblib.load(os.path.join(MODEL_DIR, "label_encoder_target.joblib"))
    le_features = joblib.load(os.path.join(MODEL_DIR, "label_encoders_features.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    cat_cols = joblib.load(os.path.join(MODEL_DIR, "cat_cols.joblib"))
    num_cols = joblib.load(os.path.join(MODEL_DIR, "num_cols.joblib"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_columns.joblib"))
    metrics_df = pd.read_csv(os.path.join(MODEL_DIR, "metrics.csv"), index_col=0)

    return models, le_target, le_features, scaler, cat_cols, num_cols, feature_cols, metrics_df


# function to preprocess uploaded csv for prediction
def preprocess_input_data(df, le_target, le_features, scaler, cat_cols, num_cols, feature_cols):
    # check if target column exists (supports both column names)
    target_col = None
    if "NObeyesdad" in df.columns:
        target_col = "NObeyesdad"
    elif "Obesity Level" in df.columns:
        target_col = "Obesity Level"

    has_target = target_col is not None

    if has_target:
        y_raw = df[target_col]
        X_raw = df.drop(columns=[target_col])
        y_encoded = le_target.transform(y_raw)
    else:
        X_raw = df.copy()
        y_encoded = None

    # encode categorical columns
    for col in cat_cols:
        if col in X_raw.columns:
            X_raw[col] = le_features[col].transform(X_raw[col])

    # scale numerical columns
    num_cols_present = [c for c in num_cols if c in X_raw.columns]
    if num_cols_present:
        X_raw[num_cols_present] = scaler.transform(X_raw[num_cols_present])

    # make sure columns are in right order
    X_raw = X_raw[feature_cols]

    return X_raw, y_encoded, has_target


# main app
def main():
    st.title("🏋️ Obesity Level Classification")
    st.write("""
    Predict obesity levels based on eating habits and physical condition.
    Dataset: [Obesity Levels (UCI/Kaggle)](https://www.kaggle.com/datasets/fatemehmehrparvar/obesity-levels)
    — 7 Classes, 16 Features, 2111 Instances, 6 ML Models
    """)
    st.divider()

    # load everything
    try:
        loaded_models, le_target, le_features, scaler, cat_cols, num_cols, feature_cols, saved_metrics = load_models_and_data()
    except Exception as e:
        st.error(f"Error loading models: {e}. Please run model/train_models.py first.")
        return

    class_names = list(le_target.classes_)

    # sidebar - model selection
    st.sidebar.header("Settings")
    selected_model = st.sidebar.selectbox("Select Model", list(MODEL_FILES.keys()))

    st.sidebar.divider()

    # sidebar - file upload
    st.sidebar.header("Upload Test Data")
    st.sidebar.write("Upload a CSV with same features. Include 'NObeyesdad' or 'Obesity Level' column for evaluation.")
    uploaded_file = st.sidebar.file_uploader("Choose CSV file", type=["csv"])

    # create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "All Models Comparison", "Selected Model Details",
        "Upload Evaluation", "Dataset Info"
    ])

    # ---- TAB 1: Compare all models ----
    with tab1:
        st.subheader("Model Comparison")
        st.write("Metrics from the test set (20% of data):")

        # show metrics table with color highlighting
        styled = saved_metrics.style.highlight_max(axis=0, color="#90EE90").highlight_min(axis=0, color="#FFB6C1").format("{:.4f}")
        st.dataframe(styled, use_container_width=True)

        # bar chart
        st.subheader("Bar Chart Comparison")
        metric_choice = st.selectbox("Pick a metric:", saved_metrics.columns.tolist())

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
        bars = ax.bar(saved_metrics.index, saved_metrics[metric_choice], color=colors)
        ax.set_ylabel(metric_choice)
        ax.set_title(f"{metric_choice} - All Models")
        plt.xticks(rotation=25, ha="right", fontsize=9)
        for bar, val in zip(bars, saved_metrics[metric_choice]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.4f}", ha="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

    # ---- TAB 2: Selected model details ----
    with tab2:
        st.subheader(f"Details: {selected_model}")

        if selected_model in loaded_models:
            model = loaded_models[selected_model]

            # show metrics
            if selected_model in saved_metrics.index:
                row = saved_metrics.loc[selected_model]
                cols = st.columns(6)
                for i, (metric, value) in enumerate(row.items()):
                    cols[i].metric(metric, f"{value:.4f}")

            # confusion matrix using test data
            test_path = os.path.join(MODEL_DIR, "test_data.csv")
            if os.path.exists(test_path):
                test_df = pd.read_csv(test_path)
                X_test, y_test, _ = preprocess_input_data(
                    test_df, le_target, le_features, scaler, cat_cols, num_cols, feature_cols
                )
                y_pred = model.predict(X_test)

                # plot confusion matrix
                st.write("### Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=class_names, yticklabels=class_names, ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title(f"Confusion Matrix - {selected_model}")
                plt.xticks(rotation=45, ha="right", fontsize=9)
                plt.yticks(rotation=0, fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)

                # classification report
                st.write("### Classification Report")
                report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
                st.dataframe(pd.DataFrame(report).T.style.format("{:.4f}"), use_container_width=True)
        else:
            st.warning(f"Model '{selected_model}' not found.")

    # ---- TAB 3: Upload and evaluate ----
    with tab3:
        st.subheader("Evaluate Uploaded Data")

        if uploaded_file is not None:
            try:
                df_up = pd.read_csv(uploaded_file)
                st.write(f"**Uploaded data:** {df_up.shape[0]} rows, {df_up.shape[1]} columns")
                st.dataframe(df_up.head(10), use_container_width=True)

                X_up, y_up, has_target = preprocess_input_data(
                    df_up, le_target, le_features, scaler, cat_cols, num_cols, feature_cols
                )

                model = loaded_models[selected_model]
                y_pred_up = model.predict(X_up)
                pred_labels = le_target.inverse_transform(y_pred_up)

                # show predictions
                result_df = df_up.copy()
                result_df["Predicted_Obesity_Level"] = pred_labels
                st.write(f"### Predictions ({selected_model})")
                st.dataframe(result_df, use_container_width=True)

                # if target column was present, show evaluation metrics too
                if has_target and y_up is not None:
                    if hasattr(model, "predict_proba"):
                        y_prob = model.predict_proba(X_up)
                    else:
                        y_prob = np.zeros((len(y_up), len(class_names)))

                    # calculate metrics
                    acc = accuracy_score(y_up, y_pred_up)
                    try:
                        auc = roc_auc_score(y_up, y_prob, multi_class="ovr", average="weighted")
                    except:
                        auc = 0.0
                    prec = precision_score(y_up, y_pred_up, average="weighted", zero_division=0)
                    rec = recall_score(y_up, y_pred_up, average="weighted", zero_division=0)
                    f1_val = f1_score(y_up, y_pred_up, average="weighted", zero_division=0)
                    mcc = matthews_corrcoef(y_up, y_pred_up)

                    st.write("### Evaluation Metrics")
                    cols = st.columns(6)
                    cols[0].metric("Accuracy", f"{acc:.4f}")
                    cols[1].metric("AUC", f"{auc:.4f}")
                    cols[2].metric("Precision", f"{prec:.4f}")
                    cols[3].metric("Recall", f"{rec:.4f}")
                    cols[4].metric("F1", f"{f1_val:.4f}")
                    cols[5].metric("MCC", f"{mcc:.4f}")

                    # confusion matrix
                    st.write("### Confusion Matrix")
                    cm = confusion_matrix(y_up, y_pred_up)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                                xticklabels=class_names, yticklabels=class_names, ax=ax)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    ax.set_title(f"Confusion Matrix - {selected_model}")
                    plt.xticks(rotation=45, ha="right", fontsize=9)
                    plt.yticks(rotation=0, fontsize=9)
                    plt.tight_layout()
                    st.pyplot(fig)

                    # classification report
                    st.write("### Classification Report")
                    report = classification_report(y_up, y_pred_up, target_names=class_names, output_dict=True)
                    st.dataframe(pd.DataFrame(report).T.style.format("{:.4f}"), use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.info("Upload a CSV file from the sidebar to get predictions.")

    # ---- TAB 4: Dataset info ----
    with tab4:
        st.subheader("About the Dataset")
        st.write("""
        **Dataset:** Estimation of Obesity Levels Based on Eating Habits and Physical Condition

        **Source:** [Kaggle](https://www.kaggle.com/datasets/fatemehmehrparvar/obesity-levels)

        **DOI:** 10.24432/C5H31Z

        **Size:** 2,111 instances | 16 features | 7 target classes

        **Target Classes:**
        Insufficient_Weight, Normal_Weight, Overweight_Level_I, Overweight_Level_II,
        Obesity_Type_I, Obesity_Type_II, Obesity_Type_III

        **Features:**
        | # | Feature | Description | Type |
        |---|---------|-------------|------|
        | 1 | Gender | Male / Female | Categorical |
        | 2 | Age | Age in years | Continuous |
        | 3 | Height | Height in meters | Continuous |
        | 4 | Weight | Weight in kg | Continuous |
        | 5 | family_history_with_overweight | Family overweight history | Binary |
        | 6 | FAVC | High caloric food consumption | Binary |
        | 7 | FCVC | Vegetable consumption frequency | Integer (1-3) |
        | 8 | NCP | Number of main meals | Continuous |
        | 9 | CAEC | Food between meals | Categorical |
        | 10 | SMOKE | Smoker or not | Binary |
        | 11 | CH2O | Daily water intake | Continuous |
        | 12 | SCC | Monitors calorie intake | Binary |
        | 13 | FAF | Physical activity frequency | Continuous |
        | 14 | TUE | Technology usage time | Integer |
        | 15 | CALC | Alcohol consumption | Categorical |
        | 16 | MTRANS | Transportation used | Categorical |

        **Note:** 77% of data was synthetically generated using SMOTE, 23% was collected from
        users in Colombia, Peru and Mexico.
        """)


if __name__ == "__main__":
    main()
