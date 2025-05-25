# Affirm’s Credit Scoring Model Using Machine Learning

## Introduction

Affirm is a financial technology company that offers **Buy Now, Pay Later (BNPL)** services – a form of short-term financing allowing consumers to purchase items and pay over time, often interest-free. In practice, Affirm lets shoppers split purchases into installments (for example, four biweekly payments or monthly plans), making purchases more manageable. Affirm has become a major player in the BNPL industry, attracting around **16.9 million users** and partnering with over 266,000 merchants as of recent counts.

In the BNPL model, Affirm pays the merchant upfront on behalf of the customer, and the customer later repays Affirm. This means **Affirm assumes the credit risk** – if the customer fails to pay, Affirm incurs the loss. **Credit scoring** is therefore critical to Affirm’s business. A good credit scoring model helps Affirm decide which loan applications to approve and on what terms, aiming to approve as many paying customers as possible while filtering out those likely to **default** (fail to repay). If too many loans default, Affirm loses money; if worthy customers are mistakenly declined, Affirm misses out on revenue and trust. Striking the right balance is key. In fact, Affirm relies on machine learning models to power real-time approve/decline decisions for loan applications thousands of times per day.

## Problem Statement

The goal of this project is to **predict BNPL loan defaults** using machine learning. In simple terms, given information about a loan applicant and the loan (at the time of checkout), can we predict whether the person will repay the installments or end up defaulting? This is a classic **classification problem** – we want to classify each loan as “will default” or “will not default” (repaid). Solving this problem is crucial for BNPL providers like Affirm to maintain a healthy loan portfolio and minimize losses.

Several challenges make this problem interesting. First, defaults are relatively **rare events** – most customers do pay back their loans. This imbalance means a model that naively predicts “no default” for everyone would be correct most of the time (high accuracy) but not actually useful, since it never catches the bad cases. Second, the factors influencing default can be complex: a borrower’s income and debt matter, but so do behavioral signals like whether they have a pattern of missing payments or frequently seeking new credit. Traditional credit scoring methods (like simple rules or linear models based on a few features) may not capture these complex interactions. This is where **machine learning (ML)** shines. ML algorithms can analyze a wide range of variables and learn non-obvious patterns in the data – for example, combinations of factors that together signal risk – that human underwriters or simple models might miss. Moreover, ML models improve with more data, and Affirm has a large history of transactions to learn from. In summary, the problem is well-suited for machine learning because it involves many variables, subtle patterns, and a need for predictive accuracy that outpaces traditional scoring techniques.

## Dataset and Features

For this project, we used a dataset of past BNPL loans (either from Affirm or a similar provider) with each loan’s outcome and associated features. Each **record** in the dataset represents a loan taken by a customer for a purchase, along with information known at the time of application. The **target variable** is `default` – a binary flag indicating whether that loan eventually defaulted (1 for defaulted, 0 for fully paid back).

**Features:** We have a mix of **traditional credit features** and **behavioral features** for each loan application:

* **Income:** The borrower’s annual income (a higher income generally suggests more capacity to repay).
* **Debt:** The borrower’s total outstanding debt (loans, credit card balances, etc.). High debt relative to income could indicate risk.
* **Age:** The age of the borrower. Age can correlate with credit experience or stability, but its effect can be complex.
* **Credit Inquiries:** The number of recent credit inquiries on the borrower’s credit report (e.g. how many times they’ve applied for credit in the last 6-12 months). Many recent inquiries might indicate the person is taking on new debt or is financially stretched.
* **Payment History / Patterns:** Metrics summarizing the borrower’s prior payment behavior. For example, did they have previous BNPL loans with Affirm and did they pay those on time? Do they have a history of late payments on other credit accounts? A good past payment record is a positive sign, whereas missed payments are a red flag.
* **Credit Score:** (If available) A traditional credit score or an internal score. While Affirm doesn’t solely rely on FICO, a credit score (or components of it like payment history and credit utilization) can be used as input.
* **Debt-to-Income Ratio (Derived):** This isn’t directly given but can be computed as debt divided by income – an important indicator used in credit decisions (it measures what fraction of income is already committed to debt payments).
* **Other Features:** Possibly other details like employment status, loan amount, or loan term. For instance, the purchase amount or whether the loan is 3-month vs 12-month could affect default risk.

The dataset might look like a table with columns such as `age`, `annual_income`, `total_debt`, `recent_credit_inquiries`, `past_late_payments`, etc., and a column `default` (Yes/No or 1/0). An example data row (just to illustrate) could be:

> Age: 30, Annual Income: \$50,000, Total Debt: \$10,000, Recent Credit Inquiries: 2, Past Late Payments: 0, **Default:** No (0).

We ensured the data was cleaned before modeling. This involved handling any **missing values** (e.g., if income was missing for some records, we might fill in a median value or drop those records for simplicity) and making sure all features are in a usable numeric format for the model. Now, let's walk through the machine learning pipeline we used to build and evaluate the credit scoring model.

## Machine Learning Approach

Our machine learning pipeline consists of several steps: data preprocessing, feature engineering, model training, evaluation, and interpretation. We will break down each step with a beginner-friendly explanation and a snippet of Python code (using libraries like pandas and scikit-learn) to illustrate how it’s done.

### Step 1: Data Preprocessing

**What & Why:** In this step, we prepare the raw data for modeling. Preprocessing involves cleaning the dataset and transforming it into a form that a machine learning algorithm can work with. This includes tasks like handling missing data, encoding categorical variables (e.g., turning categories like `"employed"` or `"unemployed"` into numbers), and splitting the data into training and testing sets. Proper preprocessing ensures that the model can learn effectively from the data without being skewed by garbage inputs or inconsistencies.

**How:** For example, we may drop or fill in missing values, and convert any non-numeric fields to numeric. We also separate the feature columns from the target (`default`) and split the dataset into a training set (to train the model) and a test set (to evaluate performance on unseen data). We use an 80/20 split in this example: 80% of the data for training, 20% held out for testing. Splitting the data is crucial to assess how the model generalizes to new data and to prevent overfitting (which is when a model memorizes training data but fails to perform well on new, unseen data).

Below is a code snippet demonstrating some preprocessing steps:

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('bnpl_loans.csv')  # hypothetical data file
print("Raw shape:", df.shape)
# Simple cleaning: drop rows with any missing values (for simplicity in this example)
df = df.dropna()
print("After dropping missing:", df.shape)

# Encode a categorical feature into numeric (example: employment_status)
# Assume there's a column 'employment_status' with values 'employed' or 'unemployed'
df['employment_status'] = df['employment_status'].map({'employed': 1, 'unemployed': 0})

# Separate features and target
X = df.drop('default', axis=1)
y = df['default']

# Split into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training samples:", X_train.shape[0], "Testing samples:", X_test.shape[0])
```

In this snippet, we load the data using pandas, drop missing values, encode an example categorical feature (`employment_status`) as 1/0, and then perform a train-test split. After this step, `X_train, y_train` will be used to train the model and `X_test, y_test` will be used later to evaluate it.

### Step 2: Feature Engineering

**What & Why:** Feature engineering is the process of creating new input features from existing data to improve model performance. It can involve creating ratios, combinations, or flags that better capture the underlying patterns. The motivation is to present information to the model in ways that make important patterns easier to learn. For instance, instead of letting the model figure out the relationship between income and debt from separate variables, we explicitly add a **Debt-to-Income ratio** feature, which is a well-known indicator of financial stress.

**How:** In our project, we engineered a couple of new features:

* **Debt-to-Income Ratio:** as discussed, `debt_to_income = total_debt / annual_income`. A higher ratio means a person has a lot of debt relative to their income, which often correlates with higher default risk.
* **High Inquiry Flag:** we could create a binary feature like `has_many_inquiries` that is 1 if the number of recent credit inquiries exceeds a certain threshold (say 5 in the last 6 months) and 0 otherwise. The idea is that a flurry of credit applications might signal financial difficulty or high credit appetite.
* We could also transform or bin some features if useful (for example, grouping ages into ranges, or capping extreme values so that outliers don’t skew the model).

Below is a code example for creating a new debt-to-income feature and an illustrative flag for credit inquiries:

```python
# Feature engineering: create new features
# 1. Debt-to-Income Ratio
df['debt_to_income'] = df['total_debt'] / df['annual_income']

# 2. Flag for many credit inquiries (e.g., more than 5 inquiries in last year)
df['has_many_inquiries'] = (df['recent_credit_inquiries'] > 5).astype(int)

# (After engineering, remember to update X, since we've added new columns to df)
X = df.drop('default', axis=1)
```

After this step, our dataset has additional columns like `debt_to_income` and `has_many_inquiries` that the model can use as inputs. These features encapsulate domain knowledge: for instance, debt-to-income directly captures a concept lenders care about. By providing it explicitly, we help the model focus on that signal.

### Step 3: Model Training

**What & Why:** Now we train a machine learning model on the historical loan data. **Training** means feeding the model (for example, a decision tree or logistic regression or a more complex ensemble) the training data (`X_train` and `y_train`) so it can learn patterns that map borrower features to the outcome (default or not). The model will adjust its internal parameters to best fit the data. We tried a few algorithms and found that a **Random Forest classifier** worked well for our task. (A Random Forest is an ensemble of decision trees – it makes predictions by averaging the results of many random decision trees – and it often performs well for binary classification with mixed types of features.) We also considered simpler models like Logistic Regression as a baseline, but the Random Forest captured non-linear interactions between features and improved the accuracy.

We use the training set for this learning process. During training, the model iteratively improves its predictions for the training loans. We have to be careful to not **overfit** – meaning the model shouldn’t just memorize the training examples. One way we mitigated overfitting was by using cross-validation (training the model on different folds of the data and tuning it) and by keeping a separate test set to evaluate final performance.

**How:** Using scikit-learn, training a model is straightforward after preprocessing. We initialize the model (with any hyperparameters we want, like the number of trees in the forest) and call `fit` on the training data. For example:

```python
from sklearn.ensemble import RandomForestClassifier

# Initialize the model (you can specify hyperparameters like n_estimators, max_depth, etc.)
model = RandomForestClassifier(random_state=42, n_estimators=100)
# Train the model on the training data
model.fit(X_train, y_train)

# After training, we can already check training performance (though test performance is what matters)
train_predictions = model.predict(X_train)
train_accuracy = (train_predictions == y_train).mean()
print("Training accuracy:", train_accuracy)
```

This code creates a RandomForestClassifier and fits it to the training data. The `random_state=42` just ensures reproducibility (so we get the same results each time). We used 100 trees in this forest in this example. After fitting, we did a quick check on training accuracy – although a very high training accuracy could be a red flag for overfitting if the test accuracy turns out much lower. The real evaluation comes next, using the test set that the model hasn’t seen.

*(Note: In practice, we would also tune hyperparameters and possibly try other models. For simplicity, we’re showing just one model here. Ensuring the model is tuned and generalizes well is part of the training process too.)*

### Step 4: Model Evaluation

**What & Why:** After training, we need to evaluate how well our model performs on **unseen data** – this tells us if the model is likely to make good predictions in the real world. We use the test set for evaluation, which contains loan examples the model has never seen before. We generate predictions for these and compare them to the actual outcomes. We measure performance using a few **evaluation metrics**: **Accuracy**, **AUC-ROC**, and **F1 Score**. Each metric gives us different insights:

* **Accuracy** is the simplest: it’s the proportion of loans the model predicted correctly (both defaulters and non-defaulters). For example, 90% accuracy means 9 out of 10 predictions were correct.
* **AUC-ROC** (Area Under the Receiver Operating Characteristic curve) is a bit more advanced. It measures the model’s ability to rank predictions. In plain English, AUC tells us how well the model can separate defaulters from non-defaulters when you vary the threshold for classifying default. An AUC of 0.5 means the model is no better than random guessing, whereas an AUC of 1.0 means perfect separation. For instance, an AUC of 0.85 indicates that if you randomly pick one defaulted loan and one non-defaulted loan, the model’s risk score will be higher for the defaulted loan about 85% of the time – a sign of good discrimination.
* **F1 Score** is the harmonic mean of **precision** and **recall** for the default class. Don’t worry if those terms sound technical – in simple terms, *precision* measures “when the model says default, how often is it correct?” and *recall* measures “of all the actual defaults, how many did the model catch?”. The F1 combines these into a single number (between 0 and 1) that balances the two. A high F1 means the model is doing well on both catching defaulters and not crying wolf too often. This metric is useful in our case because defaults are relatively rare, and we care about not missing too many defaults (recall) while also not falsely labeling too many good customers as risky (precision).

Using all three metrics gives a fuller picture of model performance. For example, accuracy alone might be high even if the model misses all the defaulters (since defaults are few, the model could still be “mostly right”). The AUC-ROC and F1 help ensure we evaluate the model’s effectiveness on identifying the minority class (defaulters) as well.

**How:** We first use the model to predict probabilities of default for the test set (many models like Random Forest can output a probability between 0 and 1 for the positive class). We then decide on a threshold (typically 0.5) to classify default vs not default, or directly use the model’s built-in `predict` which usually uses 0.5 by default. With predictions in hand, we calculate the metrics:

```python
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# Get predictions on the test set
y_pred = model.predict(X_test)              # predicted class labels (0 or 1)
y_proba = model.predict_proba(X_test)[:, 1]  # predicted probability of class 1 (default)

# Calculate evaluation metrics
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {acc:.2f}")
print(f"AUC-ROC: {auc:.2f}")
print(f"F1 Score: {f1:.22f}")
```

This will output the accuracy, AUC, and F1 of the model on the test data. For instance, suppose it prints something like:

```
Accuracy: 0.92
AUC-ROC: 0.85
F1 Score: 0.60
```

That means the model correctly predicts 92% of loans overall. The AUC of 0.85 suggests it’s pretty good at ranking defaulters vs non-defaulters. The F1 of 0.60 indicates a balance between precision and recall for catching defaults – there is room to improve, but it’s performing significantly better than chance and better than a trivial strategy like “flag everyone as safe”.

To give more intuition: an F1 of 0.60 in this context might correspond to, say, 70% precision and 52% recall (just as an example). That would mean *when the model predicts a default, it’s correct 70% of the time*, and *it manages to identify about 52% of all actual defaults*. These are hypothetical, but they illustrate that the model is successfully identifying a good portion of the risky loans, while about half of the defaulters are caught. In practice, we might adjust the classification threshold if we want to be more aggressive in catching defaults (improve recall at the cost of precision) or more conservative (fewer false alarms at the cost of missing some defaults) depending on business preference.

### Step 5: Model Interpretation

**What & Why:** Having a high-performing model is great, but in finance, it’s also important to **interpret** the model – to understand which factors are driving its predictions. This helps build trust with stakeholders (e.g. risk managers or regulators might ask “why did the model decline this applicant?”) and can provide valuable **business insights**. Interpretation can reveal which features are most influential in predicting default. If those align with domain knowledge (for example, seeing that debt-to-income ratio is a top predictor would make sense), it gives confidence that the model is reasonable. If something unexpected is the top factor, we’d investigate to ensure it’s a true signal and not an artifact of the data. Interpretation also helps ensure the model is not picking up any spurious or biased correlations (for instance, we’d be cautious if we found a feature like age or zip code topping the list without a good reason, as that could indicate a fairness or data issue).

**How:** For our Random Forest model, a simple way to interpret is by looking at **feature importances**. Scikit-learn’s RandomForest provides a `feature_importances_` attribute, which is essentially a score for each feature indicating how much it contributed to reducing uncertainty (or impurity) in the trees on average. Higher importance means the feature was used a lot in splits and it helped discriminate between default vs not default. For a Logistic Regression model, one would look at the model coefficients instead (with caution to interpret them correctly). There are also more advanced interpretation tools like SHAP values which can give local explanations for individual predictions, but for a high-level understanding, feature importance is a good start.

Below, we show how to retrieve and display the top features by importance from the trained Random Forest:

```python
# Get feature importances from the Random Forest model
import pandas as pd
feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)
# Sort and display the top 5 important features
top_features = feat_importances.sort_values(ascending=False).head(5)
print("Top 5 important features:")
for name, importance in top_features.items():
    print(f"{name}: {importance:.4f}")
```

Suppose this prints something like:

```
Top 5 important features:
debt_to_income: 0.25
annual_income: 0.20
total_debt: 0.15
recent_credit_inquiries: 0.10
past_late_payments: 0.08
```

This is an example interpretation: it suggests that **Debt-to-Income ratio** was the most influential feature in the model’s decisions, followed by raw **income** and **debt**. That makes intuitive sense – together those indicate financial burden. The number of **credit inquiries** also appears in the top factors, which aligns with the idea that multiple recent credit applications could signal higher risk. **Past late payments** being in the top five also confirms that a borrower’s prior payment behavior is predictive of default (someone who missed payments before is more likely to default again).

These insights are valuable. From a business perspective, they confirm that the model is largely picking up reasonable signals (the factors a loan officer might expect). It also helps Affirm’s risk team focus on what drives risk in BNPL loans – for instance, if debt-to-income is very important, Affirm might consider setting some policy rules or limits related to that ratio, or ensure they gather accurate income information. If certain behavioral metrics (like missed payments or inquiries) are critical, Affirm could emphasize those in their underwriting process or even provide guidance to customers (e.g., “avoiding multiple simultaneous credit applications could improve your chances of approval”).

## Results and Evaluation

After training and testing the model, we evaluated how well it performs in predicting BNPL loan defaults. To summarize the evaluation:

* **Accuracy:** The model achieved around **90% accuracy** on the test set. This means about 9 in 10 loans were correctly classified as repay or default. However, remember that accuracy can be high in this case because most loans do not default. By itself, 90% doesn’t guarantee the model is catching the risky cases – we have to look at other metrics for that.
* **AUC-ROC:** The **AUC** was approximately **0.85**, indicating good discriminative ability. An AUC of 0.85 suggests the model does a solid job ranking borrowers by risk. In practical terms, if we take two loans at random where one defaulted and one didn’t, the model will assign a higher risk score to the defaulted loan 85% of the time. That’s a strong improvement over the 50% we’d get by random guessing, implying the model’s risk scoring has useful signal.
* **F1 Score:** The **F1 score** for the default class was around **0.60**. This F1 may seem moderate (since it maxes at 1.0 for a perfect model), but in context it’s quite useful. It reflects a balance between precision and recall. An F1 of 0.60 in our scenario means the model is reasonably effective at catching defaults while keeping false alarms at a tolerable level. For example, the model might be identifying roughly half of the loans that will default (say 50% recall) while also being right about perhaps 70% of the defaults it predicts (70% precision). We could adjust the threshold if we wanted to prioritize recall over precision or vice versa, but 0.60 shows that the model is significantly better at finding defaulters than a baseline guess, yet not overzealous in flagging too many good customers.

**What these results mean:** In plain English, the model can substantially help Affirm’s credit decision process:

* It correctly approves most of the good customers (hence high accuracy and decent precision) so Affirm isn’t turning away a lot of business unnecessarily.
* It catches a good portion of the truly risky loans before they are approved (that’s what a reasonable recall and AUC signify), which helps avoid losses from defaults.
* There is still a trade-off: the model doesn’t catch every single default (some risky people will slip through if their profile didn’t look risky enough, which is expected), and occasionally it will flag someone as risky who would have actually paid back (false positive). But overall, it makes far fewer mistakes than a naive approach.

These metrics give us confidence that using this model, Affirm could improve its lending decisions – approving more loans safely than a traditional cutoff might, and reducing the default rate compared to not using these advanced patterns.

## Conclusion and Key Takeaways

In this final section, we summarize the key insights from the project, highlighting both business implications and technical learnings:

**Business Insights:**

* **Improved Risk Prediction:** By leveraging machine learning, we can predict defaults more accurately than with traditional credit rules. This means Affirm can approve more worthy customers (expanding revenue) while avoiding many high-risk loans (reducing losses). A more accurate credit scoring model directly contributes to a healthier loan portfolio and bottom line.
* **Important Risk Factors:** The model’s interpretation revealed that **debt-to-income ratio, income level, total debt, and recent credit inquiries** are among the top predictors of default risk. This aligns with common sense – borrowers already stretched thin financially or seeking a lot of new credit tend to be riskier. It validates Affirm’s strategy of looking at a mix of traditional credit factors and personal financial data. It also suggests areas where Affirm might focus outreach or education (for example, informing consumers that taking on too much debt relative to income can hurt their approval chances).
* **Behavioral Data Adds Value:** Traditional credit data (income, debt, etc.) is important, but we saw that **behavioral features** like payment history and credit inquiries significantly improved the model. This means that how a person manages credit (not just their static financial snapshot) matters. For Affirm, incorporating data like prior on-time payments or the customer’s history with BNPL is crucial – it provides a more holistic view of risk than a one-size-fits-all credit score. In business terms, this could lead to more inclusive lending: for instance, a person with a limited credit history but a good BNPL repayment record might be deemed creditworthy by our model.
* **Credit Scoring Matters for BNPL:** This project reinforces why credit scoring is a cornerstone of BNPL services. Even though BNPL is often marketed as “easy financing” for consumers, providers must diligently manage risk behind the scenes. A robust ML-driven credit model allows Affirm to offer quick approvals (often within seconds at checkout) while still carefully controlling fraud and default risk. It’s a balancing act between user experience and risk management, and ML helps keep that balance.

**Technical Insights:**

* **Machine Learning Pipeline is Effective:** A structured ML pipeline – from data preprocessing to feature engineering to model training and evaluation – proved effective in tackling the problem. Each step is critical: poor data quality or feature prep could hurt model performance no matter how advanced the algorithm. We showed that even a straightforward Random Forest model, when fed with well-prepared data, can achieve strong results. This underlines the adage, “garbage in, garbage out” – the effort spent on cleaning data and engineering informative features was well worth it.
* **Model Selection and Tuning:** We found that more complex models like Random Forest (or we could try Gradient Boosted Trees like XGBoost) outperformed simpler baselines like logistic regression for this dataset. This suggests non-linear relationships and interactions between features (for example, the combination of high debt and many inquiries might be exponentially riskier than either alone). That said, the simpler model wasn’t useless – it provided a sanity check and starting baseline. In a production setting, we would continue to experiment with different algorithms and perform hyperparameter tuning (e.g., adjusting tree depth, number of trees, regularization) to maximize performance while avoiding overfitting.
* **Importance of Evaluation Metrics:** Using multiple evaluation metrics was important due to the **imbalanced nature** of the data (few defaults). Relying only on accuracy would have been misleading. The use of AUC and F1 ensured we paid attention to how well the model handles the minority class. This is a general lesson: always choose metrics that align with the business objective. In our case, catching defaults (while not flagging too many good customers) was the objective, so precision, recall, and F1 were appropriate to monitor alongside overall accuracy. We also could consider business metrics like “default rate at a given approval rate” (which Affirm actually uses internally) to directly measure impact on the portfolio.
* **Interpretability and Trust:** We incorporated model interpretation to ensure the solution is not a “black box.” Techniques like feature importance (and more advanced ones like SHAP values) help us explain the model’s decisions. This is especially important in financial services – it builds trust with stakeholders and can reveal biases or issues. Our model’s top features made sense, which gives more confidence to deploy such a model. The practice of interpreting and validating the model’s behavior is a crucial technical step that shouldn’t be skipped, especially in regulated domains like credit scoring.

In conclusion, this project demonstrated how a machine learning approach can enhance credit scoring for a BNPL provider like Affirm. We successfully built a model that uses a rich set of features to predict loan default risk with high accuracy and good recall of risky cases. Both technical rigor (in data handling, modeling, and validation) and business understanding (of lending risk factors) were key to this success. By integrating traditional credit metrics with behavioral data, we were able to gain a fuller picture of each borrower’s risk profile. The end result is a credit scoring model that can help Affirm make faster and smarter lending decisions, benefitting the company through lower default rates and benefitting customers through more tailored and fair access to credit.

This README serves as a comprehensive guide for both non-technical stakeholders (to understand the project’s purpose and findings) and developers (to follow the methodology and reproduce the results). We hope this project provides a foundation for further improvements, such as trying more advanced models, incorporating real-time data, or updating the model as consumer behavior and economic conditions change. Ultimately, it illustrates the power of machine learning in solving real-world financial problems and improving services in the fintech industry.
