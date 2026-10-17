# Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to leave their service provider. This project covers data preprocessing, exploratory data analysis, feature engineering, model comparison, hyperparameter tuning, threshold optimization, and an interactive Streamlit web application.

## Project Overview

Customer churn is a major challenge for telecom companies. Predicting which customers are likely to leave can help businesses identify at-risk customers and improve customer retention strategies.

This project uses historical telecom customer data to build a binary classification model that predicts customer churn.

## Objectives

- Analyze customer demographics, services, contracts, and billing information.
- Clean and preprocess the dataset.
- Perform exploratory data analysis (EDA).
- Engineer additional features to improve model performance.
- Train and compare multiple machine learning models.
- Optimize the Random Forest model using GridSearchCV.
- Evaluate the model using classification metrics.
- Build an interactive web application using Streamlit.

## Dataset

**Dataset:** Telco Customer Churn

**Source:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

The dataset contains customer information, subscribed services, account details, billing information, tenure, and churn status.

The target variable is `Churn`:
- `1`: Customer churned.
- `0`: Customer did not churn.

The unique customer identifier is removed during preprocessing.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Joblib
- Streamlit
- Google Colab / Jupyter Notebook

## Project Workflow

### 1. Data Cleaning
- Converted `TotalCharges` into a numeric format.
- Handled missing values.
- Encoded the target variable.
- Removed the customer identifier.

### 2. Exploratory Data Analysis
Analyzed churn patterns across contract types, customer tenure, internet services, billing methods, and other customer characteristics.

### 3. Feature Engineering

Created the following features:

- **NumberOfServices:** Counts selected services subscribed to by each customer.
- **TenureGroup:** Groups customers based on their tenure.
- **AverageMonthlyCost:** Calculates average monthly cost using total charges and tenure, with special handling for zero tenure.

### 4. Data Preprocessing
- Applied `StandardScaler` to numerical features.
- Applied `OneHotEncoder` to categorical features.
- Used Scikit-learn's `ColumnTransformer` to combine preprocessing steps.

### 5. Model Training and Comparison

Compared three classification models:
- Logistic Regression
- Decision Tree
- Random Forest

Random Forest was further optimized using `GridSearchCV` with five-fold cross-validation and F1-score as the scoring metric.

### 6. Hyperparameter Tuning

The selected Random Forest parameters were:

- `n_estimators`: 100
- `max_depth`: 5
- `min_samples_split`: 10
- `min_samples_leaf`: 1

### 7. Threshold Optimization

Experimented with different classification thresholds and selected a threshold of **0.58** to balance precision and recall for churn prediction.

The threshold should ideally be selected using validation data or cross-validation before final evaluation on an untouched test set.

## Model Performance

The selected Random Forest model achieved the following experimental test results at a threshold of 0.58:

| Metric | Score |
|---|---:|
| Accuracy | 77.40% |
| Churn Precision | 55.67% |
| Churn Recall | 73.53% |
| Churn F1-Score | 63.36% |
| ROC-AUC | Approximately 0.835 |

These results are specific to the current experiment and may vary with different data splits or model configurations.

## Feature Importance

The model identified several important predictive features, including:

- Contract type, particularly month-to-month contracts
- Online security subscription
- Customer tenure
- Internet service type
- Technical support subscription
- Total charges

Feature importance describes the variables used by the model for prediction and does not necessarily imply causation.

## Streamlit Web Application

The interactive application allows users to enter customer details and receive:
- A predicted churn or non-churn outcome.
- An estimated churn probability.

The application uses the saved machine learning pipeline to preprocess the input and generate predictions.

## Run the Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/khushalsharma1214/customer_churn_prediction.git
cd customer_churn_prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

## Repository Structure

```text
customer_churn_prediction/
├── app.py
├── churn_pipeline.pkl
├── requirements.txt
├── README.md
└── customer_churn_prediction.ipynb
```

The notebook filename shown above is an example; the actual filename may differ.

## Limitations

- Predictions are estimates and cannot guarantee future customer behavior.
- Model performance depends on the quality and representativeness of the dataset.
- The model may require retraining or validation before use with data from other telecom providers.
- The classification threshold affects false positives and false negatives.
- Customer data should be handled responsibly and securely.

## Future Improvements

- Improve threshold selection using a dedicated validation set.
- Compare additional classification algorithms.
- Use SHAP for model explainability.
- Add personalized customer retention recommendations.
- Deploy the application to a cloud platform.
- Monitor performance as new data becomes available.

## Author

**Khushal Sharma**

GitHub: [khushalsharma1214](https://github.com/khushalsharma1214)

---

This project demonstrates an end-to-end machine learning workflow, from data cleaning and feature engineering to model evaluation and interactive prediction.
