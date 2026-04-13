# Student Academic Performance Predictor

## Executive Summary

This project develops a machine learning pipeline to predict student academic outcomes (Graduate, Dropout, or Enrolled) using the "Predict Students' Dropout and Academic Success" dataset. The primary objective is to build a predictive model that can assist higher education institutions in early identification of students at risk of dropping out, enabling timely intervention.

The project followed a complete ML pipeline: data preprocessing, exploratory data analysis, feature engineering, model development, and evaluation. Four classification algorithms were trained and evaluated: Logistic Regression, Decision Tree, Random Forest, and K-Nearest Neighbors (KNN). The Random Forest classifier achieved the best performance with an accuracy of 76.7% and a weighted F1-score of 75.4%. The final model was deployed as a Streamlit web application.

## Problem Statement

Student dropout rates in higher education are a significant challenge, leading to:
- Financial losses for institutions
- Reduced workforce readiness
- Personal setbacks for students

Traditional methods of identifying at-risk students are often reactive. There is a need for a proactive, data-driven solution that can:
- Predict student outcomes early in their academic journey
- Identify key factors contributing to dropout risk
- Provide educators and administrators with actionable insights for student support services

## Dataset Source

**Dataset Name:** Predict Students' Dropout and Academic Success

**Source:** Kaggle

**Domain:** Education

**File Format:** CSV (; delimiter)

**Total Samples:** 4,424 student records

**Total Features:** 37 (36 input features + 1 target variable)

**Target Variable:** Target (3 classes: Graduate, Dropout, Enrolled)

**Class Distribution:**
| Class | Count | Percentage |
|-------|-------|------------|
| Graduate | 2,209 | 49.9% |
| Dropout | 1,421 | 32.1% |
| Enrolled | 794 | 18.0% |

## Methodology

### Data Preprocessing

The following steps were performed in the Jupyter notebook `1_Dataset_Preprocessing.ipynb`:

1. **Data Loading:** The dataset was loaded using pandas with semicolon separator
2. **Missing Values Check:** No missing values were found in any column
3. **Duplicate Removal:** No duplicate records were found
4. **Target Encoding:** The target variable was label-encoded as:
   - Dropout → 0
   - Enrolled → 1
   - Graduate → 2
5. **Train-Test Split:** 80% training, 20% testing with stratification to preserve class distribution
6. **Feature Scaling:** StandardScaler was applied to numerical features for models sensitive to scale (Logistic Regression, KNN)

### Models Developed (4 Models)

| Model | Justification |
|-------|----------------|
| Logistic Regression | Baseline linear model; interpretable, fast, works well for probabilistic classification |
| Decision Tree | Non-linear, easy to visualize, handles mixed data types |
| Random Forest | Ensemble method; reduces overfitting, captures complex interactions, handles class imbalance |
| K-Nearest Neighbors (KNN) | Instance-based learning; effective for smaller datasets with clear clusters |

### Evaluation Metrics

Since this is a multi-class classification problem, the following weighted metrics were used:
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-Score (weighted)
- Confusion Matrix (for class-wise performance)

## Results

### Model Performance Comparison

| Model | Accuracy | Precision (weighted) | Recall (weighted) | F1-Score (weighted) |
|-------|----------|----------------------|-------------------|----------------------|
| Random Forest | 76.72% | 75.44% | 76.72% | 75.39% |
| Logistic Regression | 76.84% | 75.00% | 76.84% | 75.31% |
| Decision Tree | 69.72% | 70.19% | 69.72% | 69.94% |
| KNN | 66.78% | 64.83% | 66.78% | 65.53% |

**Best Model:** Random Forest (selected based on highest F1-Score)

### Detailed Classification Report - Random Forest

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Dropout | 0.81 | 0.75 | 0.78 | 284 |
| Enrolled | 0.57 | 0.37 | 0.45 | 159 |
| Graduate | 0.78 | 0.92 | 0.85 | 442 |
| Macro avg | 0.72 | 0.68 | 0.69 | 885 |
| Weighted avg | 0.75 | 0.77 | 0.75 | 885 |

### Confusion Matrix Interpretation

- **Graduate class:** Best performance (92% recall) – model identifies most graduates correctly
- **Dropout class:** Good performance (75% recall, 81% precision)
- **Enrolled class:** Weakest performance (37% recall) – many "Enrolled" students are misclassified as either Dropout or Graduate, indicating this class is harder to distinguish

## Demonstration of the Application (Streamlit Deployment)

The final model has been deployed using Streamlit, allowing users to input student data and receive real-time predictions.

### Features of the Web App:
- User-friendly sidebar input form for all 36 features
- Real-time prediction using the saved Random Forest model
- Output displays the predicted category: Dropout, Enrolled, or Graduate
- Model files saved: `scaler.pkl`, `label_encoder.pkl`, `best_model.pkl`

### How to Run:

# Install required packages
pip install streamlit pandas numpy scikit-learn joblib

# Navigate to project folder
cd ML-Project

# Run the Streamlit app
 py -m streamlit run app.py
