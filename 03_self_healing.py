"""
Self-healing mechanism demonstration.

This script simulates data drift detection and retraining
as described in Chapter 3 and Chapter 4 of the project report.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv("startup_funding.csv")

# Basic preprocessing
data = data.dropna()
X = data.drop("status", axis=1)
y = data["status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initial model training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Initial evaluation
y_pred = model.predict(X_test)
initial_accuracy = accuracy_score(y_test, y_pred)

print("Initial Model Accuracy:", initial_accuracy)

# ---------------- SELF-HEALING LOGIC ---------------- #

ACCURACY_THRESHOLD = 0.65

def retrain_model(updated_data):
    updated_data = updated_data.dropna()
    X_new = updated_data.drop("status", axis=1)
    y_new = updated_data["status"]

    X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
        X_new, y_new, test_size=0.2, random_state=42
    )

    new_model = RandomForestClassifier(random_state=42)
    new_model.fit(X_train_new, y_train_new)

    y_pred_new = new_model.predict(X_test_new)
    new_accuracy = accuracy_score(y_test_new, y_pred_new)

    print("Retrained Model Accuracy:", new_accuracy)
    return new_model, new_accuracy

# Simulated monitoring check
if initial_accuracy < ACCURACY_THRESHOLD:
    print("Accuracy dropped. Triggering self-healing retraining...")
    model, updated_accuracy = retrain_model(data)
else:
    print("Model performance stable. No retraining required.")
