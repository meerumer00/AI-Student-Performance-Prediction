
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
data = pd.read_csv("students.csv")

print("Dataset Loaded Successfully!")
print("Total Students:", len(data))

# Input Features
X = data[
    [
        "attendance",
        "internal_marks",
        "assignment_score",
        "study_hours"
    ]
]

# Target
y = data["performance"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Train Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nModel Training Completed!")
print("Training Students:", len(X_train))
print("Testing Students:", len(X_test))
print("Model Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save Model
joblib.dump(model, "student_model.pkl")

print("\nModel Saved Successfully!")
