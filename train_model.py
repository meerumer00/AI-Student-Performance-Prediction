
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load the generated sample dataset
data = pd.read_csv("sample_students.csv")

# Input features and target
X = data[
    ["attendance", "internal_marks", "assignment_score", "study_hours"]
]
y = data["performance"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Train the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate the model on unseen test records
predictions = model.predict(X_test)

print("Model Evaluation Report")
print("-----------------------")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Save the trained model
joblib.dump(model, "student_model.pkl")
print("\nModel saved successfully as student_model.pkl")
