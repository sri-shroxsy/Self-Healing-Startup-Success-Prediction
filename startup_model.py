"""
Self-Healing Startup Success Prediction Model
Academic demonstration code
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load Dataset
# This is a placeholder dataset name.
# Actual dataset used during experimentation may vary.
data = pd.read_csv("startup_funding.csv")

# Step 2: Basic Data Preparation
# Remove rows with missing values for simplicity
data = data.dropna()

# Convert target variable
# Status: Operating / Acquired = 1, Closed = 0
data["status"] = data["status"].apply(
    lambda x: 1 if x in ["Operating", "Acquired"] else 0
)

# Select features (simplified for academic demonstration)
X = data.drop("status", axis=1)
y = data["status"]

# Convert categorical variables using dummy encoding
X = pd.get_dummies(X, drop_first=True)

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 5: Prediction & Evaluation
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

# Step 6: Self-Healing Simulation
# If accuracy drops below a threshold, retraining is triggered
THRESHOLD = 0.60

if accuracy < THRESHOLD:
    print("Performance drop detected. Retraining model...")
    model.fit(X_train, y_train)
    print("Model retrained successfully.")
else:
    print("Model performance is stable. No retraining required.")
