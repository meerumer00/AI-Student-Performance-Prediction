
import streamlit as st
import pandas as pd
import joblib

# Load trained AI model
model = joblib.load("student_model.pkl")

st.set_page_config(
    page_title="AI Student Performance System",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 AI Student Performance Prediction System")
st.write(
    "Predict student performance and get personalized "
    "improvement suggestions."
)

# ---------------- STUDENT PREDICTION ----------------

st.header("📝 Enter Student Details")

student_name = st.text_input("Student Name")

attendance = st.number_input(
    "Attendance (%)", min_value=0, max_value=100, value=85
)

internal_marks = st.number_input(
    "Internal Marks (out of 100)",
    min_value=0, max_value=100, value=75
)

assignment_score = st.number_input(
    "Assignment Score (out of 100)",
    min_value=0, max_value=100, value=80
)

study_hours = st.number_input(
    "Study Hours per Day", min_value=0, max_value=24, value=4
)

if st.button("Predict Performance"):

    student = pd.DataFrame([{
        "attendance": attendance,
        "internal_marks": internal_marks,
        "assignment_score": assignment_score,
        "study_hours": study_hours
    }])

    prediction = str(model.predict(student)[0]).strip()
    result = prediction.lower()

    st.subheader("Prediction Result")
    st.success(f"Predicted Performance: {prediction}")

    st.subheader("💡 Personalized Improvement Suggestions")

    suggestions = []

    if result in ["poor", "low"]:
        suggestions.extend([
            "Focus on weak subjects and ask teachers for help.",
            "Create a daily study timetable and revise regularly."
        ])
    elif result == "average":
        suggestions.extend([
            "Practise more questions to improve your performance.",
            "Revise class notes daily and prepare early for exams."
        ])
    elif result == "good":
        suggestions.extend([
            "Excellent progress! Maintain your study routine.",
            "Try advanced questions and practice mock tests."
        ])

    if attendance < 75:
        suggestions.append(
            "Improve attendance and attend classes regularly."
        )

    if internal_marks < 60:
        suggestions.append(
            "Revise important topics and practise exam questions."
        )

    if assignment_score < 60:
        suggestions.append(
            "Complete assignments on time and review feedback."
        )

    if study_hours < 2:
        suggestions.append(
            "Try following a consistent daily study schedule."
        )

    if not suggestions:
        suggestions.append(
            "Continue regular revision and maintain your study habits."
        )

    for suggestion in dict.fromkeys(suggestions):
        st.write(f"📌 {suggestion}")

    if student_name.strip():
        st.caption(f"Student: {student_name.strip()}")

# ---------------- TEACHER DASHBOARD ----------------

st.divider()
st.header("👨‍🏫 Teacher Dashboard")

try:
    data = pd.read_csv("students.csv")

    total_students = len(data)
    good_count = (data["performance"] == "Good").sum()
    average_count = (data["performance"] == "Average").sum()
    poor_count = (data["performance"] == "Poor").sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Students", total_students)
    col2.metric("Good", int(good_count))
    col3.metric("Average", int(average_count))
    col4.metric("Poor", int(poor_count))

    # Performance chart
    st.subheader("📊 Student Performance Chart")

    chart_data = pd.DataFrame({
        "Performance": ["Good", "Average", "Poor"],
        "Students": [
            int(good_count),
            int(average_count),
            int(poor_count)
        ]
    })

    st.bar_chart(
        chart_data.set_index("Performance")
    )

    # Student records
    st.subheader("📋 Student Records")

    search_name = st.text_input(
        "Search students by performance or feature value",
        placeholder="e.g. Average"
    )

    if search_name.strip():
        filtered_data = data[
            data.astype(str).apply(
                lambda column: column.str.contains(
                    search_name.strip(),
                    case=False,
                    na=False
                )
            ).any(axis=1)
        ]
    else:
        filtered_data = data

    st.dataframe(
        filtered_data,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Dashboard records are generated sample data, "
        "not actual college student records."
    )

except FileNotFoundError:
    st.error(
        "sample_students.csv not found. "
        "Please check that the file is in your project folder."
    )
