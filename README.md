---

## ⚙️ Modeling Workflow

Our machine learning pipeline for predicting BNPL loan defaults follows a clean and structured workflow:

---

### 1. 🧹 Data Preprocessing

* **Data Source**: We used the `Credit Score Classification` dataset, which includes both **traditional financial variables** (e.g., income, age, debt) and **behavioral indicators** (e.g., credit inquiries, payment patterns).
* **Steps Taken**:

  * Loaded the data using `pandas.read_csv()`
  * Checked for **missing values** and removed incomplete rows using `dropna()`
  * Applied **label encoding** to convert categorical variables into numeric format using `LabelEncoder` from `sklearn.preprocessing`
  * Ensured all features were numeric and suitable for model training

> This preprocessing step ensures that our ML models receive clean, structured, and numerical data.

---

### 2. 🧪 Feature Engineering

* We examined all features and retained only those with predictive relevance.
* Converted categorical features to numeric, where necessary.
* No synthetic features were added, but we ensured that:

  * **Credit mix**, **payment behavior**, and **current loan activity** were preserved
  * Class labels (0 = low risk, 1 = medium risk, 2 = high risk) were encoded correctly

---

### 3. 📊 Data Splitting

* We split the dataset using an 80:20 **train-test split** to evaluate model generalization.
* This was done using:

  ```python
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  ```

> A fixed random state ensures reproducibility of our results.

---

### 4. 🤖 Model Training

We trained and tested **six models** using Scikit-learn and specialized libraries:

| Model                 | Description                                                             |
| --------------------- | ----------------------------------------------------------------------- |
| **Random Forest**     | Tree-based ensemble method with good performance on tabular data        |
| **XGBoost**           | Gradient boosting algorithm optimized for speed and accuracy            |
| **LightGBM**          | Faster gradient boosting using histogram-based approach                 |
| **CatBoost**          | Handles categorical variables efficiently; robust with default settings |
| **Stacked (2-model)** | Combines Random Forest + CatBoost                                       |
| **Stacked (3-model)** | Combines Random Forest + LightGBM + CatBoost                            |

Model stacking was implemented using:

```python
from sklearn.ensemble import StackingClassifier
```

---

### 5. 📈 Evaluation Metrics

To evaluate model performance, we used:

* **Accuracy** – Overall correctness
* **F1 Score (Weighted)** – Balance between precision and recall across all classes
* **AUC-ROC** – How well the model separates classes (especially Class 2: high-risk)

Best result came from the **3-model stacked ensemble**:

* **Accuracy**: 79.1%
* **F1 Score**: 79.1%
* **AUC-ROC**: 91.5%

> These metrics reflect strong generalization and precise high-risk borrower detection.

---

### 6. 🔍 Key Insights from Modeling

* Our stacked model achieved **80% recall** for high-risk users (Class 2), helping minimize default-related losses.
* Affirm’s real-life practice of using **ensemble models** for credit scoring was mirrored effectively.
* The model also showed conservative behavior toward low-risk users — a trade-off between safety and opportunity.

---

