# 📊 Academic Performance Prediction Using Machine Learning

This project aims to predict student academic performance by classifying them into performance quartiles (Q1 to Q4) using various machine learning models. The goal is not only to compare model performance but also to interpret model decisions using explainability tools like SHAP and LIME.

## 📁 Dataset

The dataset used (`data_academic_performance.csv`) contains anonymized student features and their final grade scores (`G_SC`). The dataset was cleaned and engineered for supervised learning by binning the continuous grade into four categories using `pd.qcut`.

### 🧹 Preprocessing Steps
- Removed irrelevant columns: `['COD_S11', 'Cod_SPro', 'PERCENTILE', '2ND_DECILE', 'QUARTILE', 'Unnamed: 9']`
- Label encoding of categorical features
- Binning of target variable (`G_SC`) into 4 quartiles (`G_SC_binned`)

## 🔍 Problem Statement

**Objective**: Predict the quartile (Q1 to Q4) a student belongs to, based on academic and background attributes.

## 🧠 Models Used

| Model                    | Description                               |
|--------------------------|-------------------------------------------|
| RandomForestClassifier   | Ensemble tree-based model                 |
| KNeighborsClassifier     | Instance-based learning (K=4)             |
| GradientBoostingClassifier | Boosted decision trees                  |
| LogisticRegressionCV     | Regularized logistic regression (CV=5)    |
| SVC (Support Vector Machine) | Kernel-based model with probability output |
| DecisionTreeClassifier   | Interpretable tree model                  |
| 1D Convolutional Neural Network (CNN) | Deep learning on reshaped tabular data |

## 📈 Evaluation Metrics

Each model is evaluated using the following metrics:
- Accuracy
- Weighted F1 Score
- Confusion Matrix
- ROC-AUC (micro average)
- Classification Report (per-class precision, recall, F1)

## 📊 Visualizations

- 📌 Confusion Matrices
- 📌 ROC Curves (micro-average)
- 📌 Feature Importance (Gini, Permutation)
- 📌 SHAP Summary & Dependence Plots
- 📌 LIME Instance Explanation

## 🧪 Explainability Tools

| Tool | Use |
|------|-----|
| **SHAP** | Global + local explanations using TreeExplainer |
| **LIME** | Local explanation for one instance (LogisticRegressionCV) |

## 🧬 Feature Importance

Feature importance was extracted using:
- `.feature_importances_` from tree-based models
- `permutation_importance` from sklearn
- `coef_` values from LogisticRegressionCV
- SHAP value magnitude for Random Forest

## 📦 Dependencies

Make sure to install the following:

```bash
pip install pandas scikit-learn matplotlib seaborn shap lime tensorflow
