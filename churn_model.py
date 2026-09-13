"""
Customer Churn Prediction
Telco Customer Churn Dataset (Kaggle)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# 1. Load data
df = pd.read_csv(r"C:\Users\LOQ\Desktop\WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(df.info())
print(df.head())


# 2. Clean data
# TotalCharges was stored as text with some blank values -> convert to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Missing TotalCharges all belong to brand-new customers (tenure = 0) -> fill with 0
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Drop customerID (unique identifier, not useful for the model)
df = df.drop("customerID", axis=1)


# 3. Exploratory Data Analysis (EDA)
print(df["Churn"].value_counts())
print(df.groupby("Contract")["Churn"].value_counts())
print(df.groupby("PaymentMethod")["Churn"].value_counts())

sns.histplot(data=df, x="tenure", hue="Churn", bins=30)
plt.title("Customer tenure vs Churn")
plt.show()

# Key EDA findings:
# - Data is imbalanced: ~26% churn vs ~74% no churn
# - Month-to-month contracts churn far more (43%) than two-year contracts (3%)
# - Electronic check payment method has the highest churn rate (45%)
# - New customers (low tenure) churn far more than long-term customers


# 4. Encode categorical columns
binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
for col in binary_cols:
    print(col, df[col].unique())

df["gender"] = df["gender"].map({"Female": 0, "Male": 1})
df["Partner"] = df["Partner"].map({"No": 0, "Yes": 1})
df["Dependents"] = df["Dependents"].map({"No": 0, "Yes": 1})
df["PhoneService"] = df["PhoneService"].map({"No": 0, "Yes": 1})
df["PaperlessBilling"] = df["PaperlessBilling"].map({"No": 0, "Yes": 1})
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# One-Hot Encode multi-category columns (drop_first avoids redundant/multicollinear columns)
df = pd.get_dummies(df, columns=[
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod"
], drop_first=True)

print(df.shape)
print(df.columns.tolist())


# 5. Split features/target and train/test sets
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)


# 6. Model 1: Logistic Regression (baseline)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Logistic Regression (baseline)")
print(classification_report(y_test, y_pred))


# 7. Model 2: Random Forest (comparison)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Random Forest")
print(classification_report(y_test, y_pred_rf))


# 8. Model 3: Logistic Regression with class_weight='balanced' (final choice)
# Chosen as the final model: prioritizes catching churners (higher recall)
# over overall accuracy, since missing a real churner is costlier for the
# business than contacting a customer who was not actually going to leave.
model_balanced = LogisticRegression(max_iter=1000, class_weight='balanced')
model_balanced.fit(X_train, y_train)
y_pred_balanced = model_balanced.predict(X_test)

print("Logistic Regression (balanced) - FINAL MODEL")
print(classification_report(y_test, y_pred_balanced))