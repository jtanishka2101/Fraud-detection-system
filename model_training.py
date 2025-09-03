# model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

# -------------------------------
# 1. Load Cleaned Dataset
# -------------------------------
df = pd.read_csv("creditcard.csv")
print("Dataset loaded successfully!")
print("Shape:", df.shape)

# -------------------------------
# 2. Split Features & Target
# -------------------------------
X = df.drop("Class", axis=1)  # Features
y = df["Class"]               # Target

# Split into Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set:", X_train.shape, "Test set:", X_test.shape)

# -------------------------------
# 3. Logistic Regression (Baseline)
# -------------------------------
log_reg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

print("\n--- Logistic Regression Performance ---")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Precision:", precision_score(y_test, y_pred_log))
print("Recall:", recall_score(y_test, y_pred_log))
print("F1-score:", f1_score(y_test, y_pred_log))
print("\nClassification Report:\n", classification_report(y_test, y_pred_log))

# -------------------------------
# 4. Random Forest (Better Performance)
# -------------------------------
rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\n--- Random Forest Performance ---")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("F1-score:", f1_score(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

# -------------------------------
# 5. Save the Best Model
# -------------------------------
# (Choose Random Forest as it's usually better for this dataset)
joblib.dump(rf, "fraud_detection_model.pkl")
print("\nBest model (Random Forest) saved as fraud_detection_model.pkl")
