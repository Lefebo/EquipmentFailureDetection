# 🛠️ Equipment Failure Prediction Using Logistic Regression

## 📌 Project Overview

This project focuses on predicting equipment failure within the next 30 days using historical sensor, operational, and maintenance data. The goal is to enable **predictive maintenance**, reduce unexpected downtime, and improve operational efficiency through data-driven decision-making.

A complete end-to-end data science pipeline was developed, including data cleaning, feature engineering, preprocessing, and machine learning model training using **Logistic Regression**.

---

## 🎯 Business Objective

Industrial equipment failures can lead to:
- High maintenance costs
- Production downtime
- Operational inefficiencies

This project aims to:
- Predict equipment failure in advance (30-day horizon)
- Identify key risk factors (pressure, temperature, service history, etc.)
- Support proactive maintenance strategies

---

## 📊 Dataset Description

The dataset contains equipment operational and maintenance records such as:

- `customer_id` – Unique customer identifier  
- `device_id` – Equipment identifier  
- `pressure` – Operational pressure readings  
- `avg_temp` – Average operating temperature  
- `noise_level` – Equipment noise level (low/medium/high)  
- `contract_type` – Service contract category  
- `service_calls_12m` – Number of service calls in last 12 months  
- `install_date` – Installation date of equipment  
- `last_service` – Last maintenance date  
- `failure_next_30d` – Target variable (1 = failure, 0 = no failure)

---

## ⚙️ Data Preprocessing & Cleaning

Key preprocessing steps included:

- Handling missing values using median imputation
- Cleaning and converting inconsistent numerical formats (e.g., pressure column parsing)
- Converting date columns into datetime format
- Encoding categorical variables using one-hot encoding
- Feature scaling using StandardScaler
- Handling placeholder and invalid values (e.g., `#NUM!`, nulls)

---

## 🧠 Feature Engineering

New predictive features were created:

- **Age of device** = Current date – installation date
- Transformation of service history into usable numeric patterns
- Encoding of operational conditions (noise level, contract type)

These features improved model interpretability and predictive power.

---

## 🤖 Machine Learning Model

A **Logistic Regression** model was used due to:

- Strong interpretability for business use cases
- Ability to estimate failure probability
- Baseline performance benchmarking for classification problems

### Model Workflow:
- Train-test split (90/10)
- Model training using sklearn LogisticRegression
- Prediction on unseen test data
- Evaluation using classification report

---

## 📈 Evaluation Metrics

The model was evaluated using:

- Precision
- Recall
- F1-score
- Accuracy

```python
classification_report(y_true, y_pred)
