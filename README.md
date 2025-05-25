## 📊 Credit Scoring Model – Loan Default Classification

This project aims to develop and evaluate machine learning models to predict the probability of loan default based on customer-level credit data. The analysis was performed as part of a Financial Data Analytics (FDA) course project, using a cleaned dataset sourced from [Kaggle](https://www.kaggle.com/datasets/parisrohan/credit-score-classification/data).

---

## 📁 Dataset Overview

The dataset used for this project includes several borrower-level features like:

* Personal and demographic information
* Financial indicators (Income, Monthly Balance, etc.)
* Credit behavior data (Outstanding Debt, Credit Mix, Credit History, etc.)
* Credit score class (target variable: **Good**, **Standard**, or **Poor**)

> The dataset was first cleaned externally and then imported as `cleaned_dataset.csv`.

---

## 🔍 1. Exploratory Data Analysis (EDA)

The notebook begins with a thorough EDA to understand the structure, content, and integrity of the data.

Key tasks:

* Checked missing values and class distributions
* Used **Seaborn** and **Matplotlib** for visual insights
* Identified outliers and feature distributions
* Reviewed correlations and feature importance based on intuition and visuals

EDA revealed:

* Class imbalance across the credit score categories
* Presence of potential outliers in numerical variables
* Importance of engineered features like `Credit Mix`, `Credit History`, and `Outstanding Debt`

---

## 🧪 2. Preprocessing

We conducted several preprocessing steps to make the data suitable for modeling:

* **Label Encoding** for categorical variables like `Credit Mix`, `Payment of Minimum Amount`, and `Occupation`
* **Standardization** of numerical features using `StandardScaler`
* Addressed **class imbalance** using **SMOTE** (Synthetic Minority Oversampling Technique)
* Ensured that transformed data retained interpretability and balanced class distributions

---

## 🏗️ 3. Model Development

The project builds and compares a range of classification models:

### ✅ Baseline Models

* **Logistic Regression**
* **Decision Tree Classifier**
* **Random Forest Classifier**
* **Naive Bayes**

### ✅ Advanced Ensemble Models

* **XGBoost**
* **CatBoost**
* **LightGBM**
* **Gradient Boosting**

Each model was trained using stratified training/testing splits and evaluated using:

* **Accuracy**
* **F1-Score**
* **Confusion Matrix**
* **Classification Report**

---

## 🧠 4. Feature Engineering and Outlier Handling

The team conducted iterations to improve model performance:

* **Feature Engineering**: Derived new features such as:

  * Ratio-based features (e.g., Monthly Balance vs. Income)
  * Combined indicators (e.g., Credit Mix + Payment Behavior)
* **Outlier Removal**: Used IQR-based filtering to remove extreme values in key numerical features

These iterations led to significant improvements in model generalizability and F1-scores.

---

## 🔁 5. Model Comparison & Results

Three separate modeling strategies were compared:

1. **Baseline**: Raw features without modifications
2. **Feature Engineered Model**: With added composite features
3. **Outlier Removed Model**: Cleaned outliers before modeling

Model with **feature engineering and SMOTE balancing** yielded the best results, with the **CatBoost** and **XGBoost classifiers** achieving:

* **F1-Score**: \~0.94 on test set
* **Improved precision and recall** across all three credit classes

A detailed comparison table and confusion matrix plots were included to justify the model selection.

---

## 📈 6. Performance Visualization

The notebook includes multiple visualizations for interpretability:

* Confusion Matrices (Seaborn heatmaps)
* F1-score comparisons across models
* Precision and recall breakdown
* Class-wise model performance visualization

---

## 📌 Key Takeaways

* **Class imbalance and feature scale issues** significantly affect model performance; addressing them using **SMOTE** and **standardization** is essential.
* **Tree-based ensemble models** (CatBoost, XGBoost, LightGBM) outperformed simpler classifiers.
* **Feature engineering** improved the model’s capacity to distinguish nuanced borrower behaviors.

---

## 🛠️ Tools & Libraries

* Python
* Google Colab
* Pandas, Numpy
* Scikit-learn
* Matplotlib, Seaborn
* SMOTE (Imbalanced-learn)
* CatBoost, XGBoost, LightGBM

---

## 🧑‍💻 Authors

* Group 7 – Financial Data Analytics (FDA)
* Course Project (Spring 2025)

---

