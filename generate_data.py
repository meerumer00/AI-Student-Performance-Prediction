
import pandas as pd
import random

# Keep results consistent every time
random.seed(42)

# Create student records
students = []

for i in range(1000):

    attendance = random.randint(40, 100)
    internal_marks = random.randint(20, 100)
    assignment_score = random.randint(20, 100)
    study_hours = random.randint(0, 10)

    # Calculate overall performance score
    average = (
        attendance * 0.30
        + internal_marks * 0.30
        + assignment_score * 0.25
        + study_hours * 10 * 0.15
    )

    # Classify student performance
    if average >= 75:
        performance = "Good"

    elif average >= 50:
        performance = "Average"

    else:
        performance = "Poor"

    # Store student data
    students.append({
        "student_name": f"Student_{i+1}",
        "attendance": attendance,
        "internal_marks": internal_marks,
        "assignment_score": assignment_score,
        "study_hours": study_hours,
        "performance": performance
    })


# Convert records into DataFrame
df = pd.DataFrame(students)

# Save dataset as CSV
df.to_csv("sample_students.csv", index=False)

# Display results
print("1000 Student Dataset Created Successfully!")

print("\nFirst 5 Student Records:")
print(df.head())

print("\nPerformance Counts:")
print(df["performance"].value_counts())

print("\nTotal Students:", len(df))
print("\nDataset saved as sample_students.csv")
