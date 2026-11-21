
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------------- LOAD DATASET ----------------

data = pd.read_csv("sample_students.csv")

print("Dataset Loaded Successfully!")
print("Total Students:", len(data))

# ---------------- INPUT FEATURES ----------------

X = data[
    [
        "attendance",
        "internal_marks",
        "assignment_score",
        "study_hours"
    ]
]

# ---------------- TARGET ----------------

y = data["performance"]

# ---------------- SPLIT DATASET ----------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nDataset Split Successfully!")
print("Training Students:", len(X_train))
print("Testing Students:", len(X_test))

# ---------------- TRAIN MODEL ----------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------- PREDICTIONS ----------------

predictions = model.predict(X_test)

# ---------------- MODEL ACCURACY ----------------

accuracy = accuracy_score(y_test, predictions)

print("\nModel Training Completed!")
print("Model Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# ---------------- SAVE MODEL ----------------

joblib.dump(model, "student_model.pkl")

print("\nModel Saved Successfully!")
print("Updated model: student_model.pkl")
