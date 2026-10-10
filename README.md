# Credit Card Fraud Detection

An end-to-end machine learning project for identifying fraudulent credit card transactions in a highly imbalanced dataset.

## Project Overview

Credit card fraud creates significant financial losses for banks, payment providers, and customers. Since fraudulent transactions represent only a very small proportion of all transactions, a fraud detection system must identify rare fraudulent patterns while minimizing false alarms on legitimate transactions.

This project applies exploratory data analysis, data preprocessing, class-imbalance handling, model training, hyperparameter optimization, and performance evaluation to build machine learning models capable of detecting fraudulent credit card transactions.

The project was completed as a capstone machine learning project using the publicly available Credit Card Fraud Detection dataset.

## Business Problem

The primary business objective is to detect fraudulent transactions as accurately as possible so that financial institutions can:

- Reduce financial losses caused by unauthorized transactions
- Identify suspicious transactions in near real time
- Improve customer trust and transaction security
- Reduce the operational cost of manually reviewing transactions
- Balance fraud detection with the customer experience by limiting false positives

Fraud detection is a challenging classification problem because missing a fraudulent transaction can be costly, while incorrectly blocking a legitimate transaction can negatively affect customers.

## Project Objectives

The key objectives of this project are to:

1. Understand the structure and characteristics of credit card transaction data.
2. Analyze the distribution of fraudulent and legitimate transactions.
3. Identify the challenges caused by severe class imbalance.
4. Prepare the data for machine learning.
5. Apply techniques such as feature scaling and SMOTE to improve model learning.
6. Train and compare multiple classification algorithms.
7. Tune model hyperparameters to improve predictive performance.
8. Evaluate models using metrics appropriate for fraud detection.
9. Identify the most important features associated with fraudulent transactions.

## Dataset

The project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) originally published by Machine Learning Group at ULB.

### Dataset Summary

| Property | Value |
|---|---:|
| Total transactions | 284,807 |
| Fraudulent transactions | 492 |
| Legitimate transactions | 284,315 |
| Total features | 30 input features |
| Target variable | `Class` |
| Fraud rate | Approximately 0.173% |
| Legitimate transaction rate | Approximately 99.827% |

The dataset contains transactions made by European cardholders over a two-day period in September 2013.

### Features

The dataset includes:

- `Time` — Seconds elapsed between each transaction and the first transaction
- `V1` through `V28` — Principal Component Analysis-transformed features
- `Amount` — Transaction amount
- `Class` — Target variable:
  - `0` = Legitimate transaction
  - `1` = Fraudulent transaction

The original transaction attributes were transformed using PCA. As a result, most feature names are anonymized, which helps protect sensitive financial information while still allowing machine learning analysis.

## Why Class Imbalance Matters

Only 492 of the 284,807 transactions are fraudulent. This means that fraudulent transactions account for approximately 0.173% of the dataset.

A model that predicts every transaction as legitimate would achieve more than 99% accuracy while detecting no fraud at all. Therefore, accuracy alone is not an appropriate measure of success for this problem.

The project focuses on metrics that provide more meaningful insight into fraud detection performance:

- Precision
- Recall
- F1-score
- Confusion matrix
- ROC-AUC
- Precision-Recall AUC

For fraud detection, recall is especially important because it measures the proportion of actual fraudulent transactions that the model successfully identifies.

## Project Workflow

The project follows a complete machine learning workflow.

### 1. Data Loading

The transaction data is loaded into a pandas DataFrame for analysis and modeling.

```python
df = pd.read_csv("creditcard.csv")
```

The dataset is inspected using:

- `head()`
- `shape`
- `describe()`
- `info()`
- Missing-value checks
- Target-class distribution analysis

### 2. Exploratory Data Analysis

The exploratory analysis focuses on understanding:

- Dataset dimensions
- Data types
- Missing values
- Statistical distributions
- Fraudulent versus legitimate transaction counts
- Transaction amount distribution
- Time distribution
- Differences between fraud and non-fraud transactions
- Feature relationships and correlations

The analysis confirms that the dataset is extremely imbalanced and that fraudulent transactions have different patterns across several transformed features.

### 3. Data Quality Checks

The project checks for:

- Null values
- Duplicate records
- Invalid data types
- Unexpected target values
- Distribution irregularities
- Potential outliers

These checks help ensure that the data is suitable for model training.

### 4. Feature Preparation

The target variable is separated from the explanatory variables:

```python
X = df.drop("Class", axis=1)
y = df["Class"]
```

The `Time` and `Amount` features require particular attention because they are measured on different scales from the PCA-transformed features.

Feature scaling is used to improve the behavior of scale-sensitive machine learning algorithms, especially Logistic Regression and K-Nearest Neighbors.

### 5. Train-Test Split

The dataset is divided into training and testing subsets. A stratified split is important because it preserves the proportion of fraudulent transactions in both subsets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
```

The test data is kept separate so that final model performance can be evaluated on unseen transactions.

### 6. Handling Class Imbalance with SMOTE

The project applies Synthetic Minority Over-sampling Technique, commonly known as SMOTE, to improve the representation of fraudulent transactions in the training data.

SMOTE creates synthetic examples of the minority class rather than simply duplicating existing fraud records.

Important implementation principle:

> Oversampling should be applied only to the training data, never before the train-test split.

Applying SMOTE before splitting the dataset can cause information leakage and result in overly optimistic evaluation results.

### 7. Model Training

Several machine learning algorithms are explored and compared, including:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- AdaBoost Classifier
- XGBoost Classifier
- K-Nearest Neighbors

These algorithms provide a range of approaches:

- Logistic Regression provides a strong and interpretable baseline.
- Decision Trees capture non-linear decision boundaries.
- Random Forest combines multiple decision trees to improve stability.
- AdaBoost focuses successively on difficult training examples.
- XGBoost uses gradient boosting to build powerful predictive models.
- K-Nearest Neighbors classifies transactions based on nearby examples.

### 8. Hyperparameter Optimization

Hyperparameter optimization is used to identify better model configurations.

The project explores tools and techniques such as:

- `GridSearchCV`
- `RandomizedSearchCV`
- Bayesian optimization with `BayesSearchCV`
- Stratified cross-validation

Potential hyperparameters include:

- Number of estimators
- Tree depth
- Learning rate
- Minimum samples per split
- Minimum samples per leaf
- Regularization strength
- Number of neighbors
- Subsample ratio

Stratified cross-validation is used so that each validation fold maintains a representative fraud-to-legitimate transaction ratio.

### 9. Model Evaluation

The models are evaluated using fraud-focused classification metrics.

#### Precision

Precision measures how many transactions predicted as fraudulent are actually fraudulent.

```text
Precision = True Positives / (True Positives + False Positives)
```

High precision helps reduce the number of legitimate transactions incorrectly flagged as fraud.

#### Recall

Recall measures how many actual fraudulent transactions are correctly detected.

```text
Recall = True Positives / (True Positives + False Negatives)
```

High recall is important when failing to detect fraud is more expensive than reviewing an additional legitimate transaction.

#### F1-Score

The F1-score is the harmonic mean of precision and recall.

```text
F1-score = 2 × (Precision × Recall) / (Precision + Recall)
```

It is useful when both false positives and false negatives matter.

#### Confusion Matrix

The confusion matrix contains:

- True Positives — Fraud correctly identified as fraud
- True Negatives — Legitimate transactions correctly identified as legitimate
- False Positives — Legitimate transactions incorrectly flagged as fraud
- False Negatives — Fraudulent transactions missed by the model

#### ROC-AUC

ROC-AUC measures the model's ability to distinguish between fraudulent and legitimate transactions across different classification thresholds.

#### Precision-Recall AUC

Because fraud is extremely rare, Precision-Recall AUC is particularly useful. It provides a more informative view of performance on the minority class than accuracy alone.

## Technology Stack

### Programming Language

- Python

### Data Analysis

- pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Machine Learning

- scikit-learn
- XGBoost
- scikit-optimize

### Development Environment

- Jupyter Notebook
- Google Colab

## Repository Structure

```text
Credit-Card-Fraud-Detection/
│
├── Capstone_Project_(Credit_Card_Fraud_Detection)_-_Abhijeet_Singh_[1].ipynb
├── Credit_card_fraud_detection_Starter_code+[1].ipynb
└── README.md
```

### Main Notebook

`Capstone_Project_(Credit_Card_Fraud_Detection)_-_Abhijeet_Singh_[1].ipynb`

This notebook contains the complete project workflow, including:

- Data loading
- Exploratory data analysis
- Data preparation
- Class-imbalance analysis
- Feature processing
- Model development
- Hyperparameter tuning
- Model evaluation
- Visualizations
- Feature-importance analysis

### Starter Notebook

`Credit_card_fraud_detection_Starter_code+[1].ipynb`

This notebook contains the initial project structure and starter code used during development.

## How to Run the Project

### Option 1: Run with Google Colab

1. Open the main notebook in Google Colab.
2. Upload the dataset or connect the notebook to Google Drive.
3. Download the Credit Card Fraud Detection dataset from Kaggle.
4. Place `creditcard.csv` in the notebook's working directory.
5. Run the notebook cells in order.

### Option 2: Run Locally

Clone the repository:

```bash
git clone https://github.com/Abhijeet-Singh-2001/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

Install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn xgboost scikit-optimize jupyter
```

Download the dataset from Kaggle and place the following file in the project directory:

```text
creditcard.csv
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open the main notebook and execute the cells sequentially.

## Key Findings

The project demonstrates several important findings:

- Credit card fraud detection is a highly imbalanced classification problem.
- Accuracy is not sufficient for evaluating fraud detection models.
- Precision, recall, F1-score, and Precision-Recall AUC provide more useful insights.
- Fraudulent transactions exhibit distinguishable patterns across several PCA-transformed features.
- Feature scaling is important for scale-sensitive algorithms.
- Resampling techniques such as SMOTE can improve minority-class learning when applied correctly.
- Ensemble models such as Random Forest and XGBoost are well suited to capturing complex, non-linear relationships.
- The final decision threshold should depend on the business cost of false positives and false negatives.

## Practical Business Interpretation

In a real-world banking environment, the model would not necessarily make a final yes-or-no decision by itself. Instead, it could generate a fraud probability for every transaction.

A bank could then define different actions based on that probability:

- Low-risk transactions: Approve automatically
- Medium-risk transactions: Request additional authentication
- High-risk transactions: Temporarily block and send for manual review

This approach allows the business to tune the trade-off between customer convenience and fraud prevention.

## Limitations

This project has several limitations:

- The dataset represents transactions collected over only two days.
- The feature names and original business meanings are anonymized.
- The dataset may not represent current fraud patterns.
- The model is evaluated on historical data and may not generalize to every financial institution.
- The cost of false positives and false negatives is not explicitly modeled.
- SMOTE creates synthetic training examples that may not perfectly represent real fraudulent behavior.
- A production deployment would require monitoring for data drift and changing fraud patterns.

## Future Improvements

Potential improvements include:

- Use time-based validation to better simulate real-world deployment.
- Apply cost-sensitive learning based on the financial impact of each error.
- Optimize the classification threshold according to business requirements.
- Compare additional imbalance-handling techniques such as:
  - Random undersampling
  - Balanced Random Forest
  - Class-weighted models
  - ADASYN
  - EasyEnsemble
- Calibrate model probabilities for more reliable risk scores.
- Add explainability using SHAP or permutation importance.
- Track model performance over time to identify concept drift.
- Build a real-time prediction API using Flask or FastAPI.
- Create a dashboard using Streamlit or Power BI.
- Package the final model with reproducible preprocessing steps.
- Add automated testing and continuous integration.
- Deploy the model using a cloud platform.

## Skills Demonstrated

This project demonstrates practical experience in:

- Python programming
- Data cleaning and validation
- Exploratory data analysis
- Statistical analysis
- Data visualization
- Feature scaling
- Imbalanced classification
- SMOTE oversampling
- Supervised machine learning
- Ensemble learning
- Hyperparameter optimization
- Cross-validation
- Model evaluation
- Business-focused metric selection
- Technical documentation

## Resume-Ready Project Summary

> Developed an end-to-end credit card fraud detection system using Python and machine learning on 284,807 transactions with only 492 fraudulent cases. Performed exploratory data analysis, feature preprocessing, imbalance handling with SMOTE, model training, hyperparameter optimization, and evaluation using precision, recall, F1-score, ROC-AUC, and Precision-Recall AUC. Compared Logistic Regression, Decision Tree, Random Forest, AdaBoost, K-Nearest Neighbors, and XGBoost models to identify effective approaches for rare-event fraud classification.

## Conclusion

This project presents a complete machine learning workflow for detecting fraudulent credit card transactions in a highly imbalanced dataset.

The most important lesson is that fraud detection should not be treated as a standard accuracy-based classification problem. A useful solution must carefully manage class imbalance, select appropriate evaluation metrics, control false positives and false negatives, and align the final decision threshold with business requirements.

The project provides a strong foundation for extending the work into a production-ready fraud monitoring system with real-time prediction, explainable risk scores, automated alerts, and ongoing model monitoring.
