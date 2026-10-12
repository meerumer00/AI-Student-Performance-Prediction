
import pandas as pd
import random

random.seed(42)

students = []

for i in range(100):
    attendance = random.randint(40, 100)
    internal_marks = random.randint(20, 100)
    assignment_score = random.randint(20, 100)
    study_hours = random.randint(0, 10)

    average = (
        attendance * 0.30
        + internal_marks * 0.30
        + assignment_score * 0.25
        + study_hours * 10 * 0.15
    )

    if average >= 75:
        performance = "Good"
    elif average >= 50:
        performance = "Average"
    else:
        performance = "Poor"

    students.append({
        "attendance": attendance,
        "internal_marks": internal_marks,
        "assignment_score": assignment_score,
        "study_hours": study_hours,
        "performance": performance
    })

df = pd.DataFrame(students)
df.to_csv("sample_students.csv", index=False)

print("100 sample student records created successfully!")
print(df.head())
print("\nPerformance counts:")
print(df["performance"].value_counts())
