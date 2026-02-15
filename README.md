# Obesity Level Classification — ML Assignment 2

## a. Problem Statement

Obesity is a growing health issue worldwide. It can lead to many serious diseases like diabetes, heart problems etc. So it is important to predict obesity levels early using machine learning.

In this project, we classify individuals into **7 obesity levels** using 16 features about their eating habits, physical activity and lifestyle. We trained 6 different ML models, compared their performance and built a Streamlit web app to demonstrate the results.

---

## b. Dataset Description

| Property | Value |
|---|---|
| **Dataset Name** | Estimation of Obesity Levels Based on Eating Habits and Physical Condition |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/fatemehmehrparvar/obesity-levels) |
| **DOI** | `10.24432/C5H31Z` |
| **Instances** | 2,111 |
| **Features** | 16 (8 numerical + 8 categorical) |
| **Target Variable** | `NObeyesdad` — Obesity Level (7 classes) |
| **Missing Values** | None |
| **License** | CC BY 4.0 |

**Target Classes (7):**
- Insufficient_Weight
- Normal_Weight
- Overweight_Level_I
- Overweight_Level_II
- Obesity_Type_I
- Obesity_Type_II
- Obesity_Type_III

**Feature Descriptions:**

| # | Feature | Description | Type |
|---|---------|-------------|------|
| 1 | Gender | Male / Female | Categorical |
| 2 | Age | Age in years | Continuous |
| 3 | Height | Height in meters | Continuous |
| 4 | Weight | Weight in kg | Continuous |
| 5 | family_history_with_overweight | Family overweight history | Binary |
| 6 | FAVC | Frequent high caloric food consumption | Binary |
| 7 | FCVC | Vegetable consumption frequency (1-3) | Integer |
| 8 | NCP | Number of main meals daily | Continuous |
| 9 | CAEC | Food consumption between meals | Categorical |
| 10 | SMOKE | Smoker or not | Binary |
| 11 | CH2O | Daily water intake | Continuous |
| 12 | SCC | Monitors calorie intake | Binary |
| 13 | FAF | Physical activity frequency | Continuous |
| 14 | TUE | Technology device usage time | Integer |
| 15 | CALC | Alcohol consumption frequency | Categorical |
| 16 | MTRANS | Primary transportation used | Categorical |

**Note:** 77% of the data was synthetically generated using SMOTE (Weka); 23% was collected directly from users in Mexico, Peru, and Colombia via a web platform.

**Citation:**  
Palechor, F.M. & De la Hoz Manotas, A. (2019). *Dataset for estimation of obesity levels based on eating habits and physical condition in individuals from Colombia, Peru and Mexico.* Data in Brief, 25, 104344.

---

## c. Models Used

We trained all 6 models on the same data using an **80/20 train-test split** (stratified, random_state=42). Categorical features were label-encoded and numerical features were standard-scaled.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :------------- | :--------: | :---------: | :-------: | :----: | :----: | :-----: |
| Logistic Regression | 0.8723 | 0.9873 | 0.8719 | 0.8723 | 0.8702 | 0.8515 |
| Decision Tree | 0.9054 | 0.9450 | 0.9079 | 0.9054 | 0.9062 | 0.8897 |
| kNN | 0.8345 | 0.9627 | 0.8336 | 0.8345 | 0.8236 | 0.8094 |
| Naive Bayes | 0.5981 | 0.9000 | 0.6465 | 0.5981 | 0.5732 | 0.5435 |
| Random Forest (Ensemble) | **0.9574** | 0.9973 | **0.9609** | **0.9574** | **0.9580** | **0.9507** |
| XGBoost (Ensemble) | 0.9527 | **0.9975** | 0.9557 | 0.9527 | 0.9532 | 0.9451 |

**Best overall model: Random Forest** (highest Accuracy, Precision, Recall, F1, MCC)  
**Best AUC: XGBoost** (0.9975)

---

## d. Observations

| ML Model Name | Observation about model performance |
| :------------- | :---------------------------------- |
| Logistic Regression | Good baseline model with 87.23% accuracy. It works well for multi-class problems. AUC is 0.9873 which means it can separate classes nicely. It struggles a bit with similar classes like Overweight I and II because it only uses linear boundaries. |
| Decision Tree | Better than Logistic Regression at 90.54% accuracy. It can handle non-linear patterns in the data which helps. But AUC is slightly lower (0.945) probably because of some overfitting on the training data. |
| kNN | Moderate accuracy of 83.45%. It doesn't work as well here because the dataset has mixed feature types (categorical + numerical) and 16 features is quite a lot for distance-based methods. Even with scaling, performance is limited. |
| Naive Bayes | Worst performer with only 59.81% accuracy. This makes sense because Naive Bayes assumes all features are independent, but in our dataset features like Weight, Height, and FAF are clearly related to each other. So the assumption is violated. |
| Random Forest (Ensemble) | **Best model overall** with 95.74% accuracy. Using 100 trees reduces overfitting and captures complex patterns well. AUC of 0.9973 is almost perfect. MCC of 0.9507 also shows it classifies all 7 classes very well. |
| XGBoost (Ensemble) | Very close to Random Forest at 95.27% accuracy. Has the **highest AUC (0.9975)** which means its probability predictions are very good. The boosting approach helps in distinguishing between similar classes. Slightly lower accuracy than RF might be because we used default hyperparameters. |

---

## Project Structure

```
project-folder/
│── app.py                          # Streamlit web application
│── requirements.txt                # Python dependencies
│── README.md                       # This file
│── ObesityDataSet_raw_and_data_sinthetic.csv  # Dataset
│── model/
    ├── train_models.py             # Model training script
    ├── *.joblib                    # Saved trained models & preprocessors
    ├── metrics.csv                 # Evaluation metrics
    └── test_data.csv               # Test split for app evaluation
```

## How to Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Train models (generates model/*.joblib files)
python model/train_models.py

# Launch Streamlit app (from project root)
streamlit run app.py
```

## Live App

🔗 **Streamlit App:** [Streamlit - Obesity Level Classification](https://obesitymlapp-oxtc4gyzg5hlcsgvv4o3jf.streamlit.app/)

## GitHub Repository

🔗 **Repository:** [Obesity_ML_App](https://github.com/mahendrankk/Obesity_ML_App)
