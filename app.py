import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="AI Student Performance System",
    page_icon="🎓",
    layout="wide"
)


# ---------------- LOAD AI MODEL ----------------

@st.cache_resource
def load_model():
    return joblib.load("student_model.pkl")


try:
    model = load_model()
except Exception as e:
    st.error(f"AI model could not be loaded: {e}")
    st.stop()


# ---------------- CATEGORY STANDARDIZATION ----------------

def standardize_performance(value):
    value = str(value).strip().lower()

    category_mapping = {
        "good": "Good",
        "average": "Average",
        "moderate": "Average",
        "poor": "Poor",
        "low": "Poor",
        "at risk": "Poor",
        "at-risk": "Poor",
        "at_risk": "Poor"
    }

    return category_mapping.get(value, "Unknown")


# ---------------- APP TITLE ----------------

st.title("🎓 AI Student Performance Prediction System")

st.write(
    "Predict student performance and get personalized "
    "improvement suggestions."
)


# ---------------- STUDENT PREDICTION ----------------

st.header("📝 Enter Student Details")

student_name = st.text_input("Student Name")

attendance = st.number_input(
    "Attendance (%)",
    min_value=0,
    max_value=100,
    value=85
)

internal_marks = st.number_input(
    "Internal Marks (out of 100)",
    min_value=0,
    max_value=100,
    value=75
)

assignment_score = st.number_input(
    "Assignment Score (out of 100)",
    min_value=0,
    max_value=100,
    value=80
)

study_hours = st.number_input(
    "Study Hours per Day",
    min_value=0,
    max_value=24,
    value=4
)

if st.button("Predict Performance", type="primary"):

    student = pd.DataFrame([{
        "attendance": attendance,
        "internal_marks": internal_marks,
        "assignment_score": assignment_score,
        "study_hours": study_hours
    }])

    try:
        prediction = str(model.predict(student)[0]).strip()
        result = standardize_performance(prediction)

        st.subheader("📊 Prediction Result")

        if result == "Good":
            st.success("Predicted Performance: Good")

        elif result == "Average":
            st.warning("Predicted Performance: Average")

        elif result == "Poor":
            st.error("Predicted Performance: Poor")

        else:
            st.info(f"Model Prediction: {prediction}")

        st.subheader("💡 Personalized Improvement Suggestions")

        suggestions = []

        if result == "Poor":
            suggestions.extend([
                "Focus on weak subjects and ask teachers for help.",
                "Create a daily study timetable and revise regularly.",
                "Seek additional academic support if needed."
            ])

        elif result == "Average":
            suggestions.extend([
                "Practise more questions to improve your performance.",
                "Revise class notes daily and prepare early for exams.",
                "Set weekly learning goals and track your progress."
            ])

        elif result == "Good":
            suggestions.extend([
                "Excellent progress! Maintain your study routine.",
                "Try advanced questions and practise mock tests."
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

    except Exception as e:
        st.error(f"Prediction failed: {e}")


# ---------------- TEACHER DASHBOARD ----------------

st.divider()
st.header("👨‍🏫 Teacher Dashboard")

try:
    data = pd.read_csv("students.csv")

    if "performance" not in data.columns:
        st.error(
            "The students.csv file must contain a 'performance' column."
        )

    else:
        # Standardize performance categories
        data["performance"] = data["performance"].apply(
            standardize_performance
        )

        # Calculate counts from actual student records
        total_students = len(data)

        good_count = int(
            (data["performance"] == "Good").sum()
        )

        average_count = int(
            (data["performance"] == "Average").sum()
        )

        poor_count = int(
            (data["performance"] == "Poor").sum()
        )

        unknown_count = int(
            (data["performance"] == "Unknown").sum()
        )

        # Dashboard metrics
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Students", total_students)
        col2.metric("Good", good_count)
        col3.metric("Average", average_count)
        col4.metric("Poor", poor_count)

        if unknown_count > 0:
            st.warning(
                f"{unknown_count} student record(s) have an "
                "unrecognized performance category."
            )

        # Performance chart
        st.subheader("📊 Student Performance Chart")

        chart_data = pd.DataFrame({
            "Performance": ["Good", "Average", "Poor"],
            "Students": [
                good_count,
                average_count,
                poor_count
            ]
        })

        st.bar_chart(
            chart_data.set_index("Performance")
        )

        # Student records
        st.subheader("📋 Student Records")

        search_text = st.text_input(
            "Search students by performance or feature value",
            placeholder="e.g. Average, Good, Poor, or 85"
        )

        if search_text.strip():
            filtered_data = data[
                data.astype(str).apply(
                    lambda column: column.str.contains(
                        search_text.strip(),
                        case=False,
                        na=False,
                        regex=False
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
        "students.csv not found. Make sure it is in your project folder."
    )

except Exception as e:
    st.error(f"Could not load the student dashboard: {e}")
