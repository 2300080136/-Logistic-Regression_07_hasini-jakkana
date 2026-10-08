# Telco Customer Churn Prediction Using Logistic Regression

## Project Overview

This project predicts whether a telecom customer is likely to churn (leave the company) using a **Logistic Regression** machine learning model.

The project uses customer information such as tenure, services, contract type, payment method, monthly charges, and total charges to predict customer churn.

---

## Dataset

**Dataset Name:** Telco Customer Churn Dataset

**Dataset File:**
`WA_Fn-UseC_-Telco-Customer-Churn (1).csv`

The dataset contains customer information and a target variable called `Churn`.

- `Churn = Yes` → Customer is likely to leave
- `Churn = No` → Customer stays

---

## Objective

The main objectives of this project are:

1. Load and understand the customer churn dataset.
2. Identify relevant input features.
3. Handle missing values.
4. Convert categorical data into numerical data.
5. Split the dataset into training and testing data.
6. Build a Logistic Regression model.
7. Predict customer churn.
8. Calculate churn probabilities.
9. Evaluate the model.
10. Analyze the confusion matrix.
11. Interpret the results in business terms.
12. Suggest an improvement to the model.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

## Project Workflow

### 1. Load the Dataset

The dataset is loaded using Pandas.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn (1).csv")

print(df.head())
print(df.shape)
print(df.columns)
print(df.info())
