
import streamlit as st
import pandas as pd
import joblib
import sqlite3
from datetime import datetime
from pathlib import Path

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="AI Student Performance System",
    page_icon="🎓",
    layout="wide"
)

# ---------------- DATABASE CONFIGURATION ----------------

DB_FILE = Path(__file__).parent / "students.db"

def get_connection():
    return sqlite3.connect(DB_FILE)

def initialize_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_name TEXT NOT NULL,
                attendance REAL NOT NULL,
                internal_marks REAL NOT NULL,
                assignment_score REAL NOT NULL,
                study_hours REAL NOT NULL,
                performance TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

def save_student(name, attendance, marks, assignment, hours, result):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO students (
                student_name, attendance, internal_marks,
                assignment_score, study_hours,
                performance, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            name, attendance, marks, assignment,
            hours, result,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))

def get_students():
    with get_connection() as conn:
        return pd.read_sql_query(
            "SELECT * FROM students ORDER BY id DESC",
            conn
        )

def update_student(student_id, name, attendance, marks,
                   assignment, hours, performance):
    with get_connection() as conn:
        conn.execute("""
            UPDATE students
            SET student_name=?, attendance=?,
                internal_marks=?, assignment_score=?,
                study_hours=?, performance=?
            WHERE id=?
        """, (
            name, attendance, marks, assignment,
            hours, performance, student_id
        ))

def delete_student(student_id):
    with get_connection() as conn:
        conn.execute(
            "DELETE FROM students WHERE id=?",
            (student_id,)
        )

initialize_database()

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

    mapping = {
        "good": "Good",
        "average": "Average",
        "moderate": "Average",
        "poor": "Poor",
        "low": "Poor",
        "at risk": "Poor",
        "at-risk": "Poor",
        "at_risk": "Poor"
    }

    return mapping.get(value, "Unknown")

# ---------------- APP TITLE ----------------

st.title("🎓 AI Student Performance Prediction System")

st.write(
    "Predict student performance, save student records "
    "and manage academic information."
)

# ---------------- STUDENT PREDICTION ----------------

st.header("📝 Enter Student Details")

with st.form("prediction_form"):

    student_name = st.text_input("Student Name")

    attendance = st.number_input(
        "Attendance (%)", 0.0, 100.0, 85.0
    )

    internal_marks = st.number_input(
        "Internal Marks (out of 100)", 0.0, 100.0, 75.0
    )

    assignment_score = st.number_input(
        "Assignment Score (out of 100)", 0.0, 100.0, 80.0
    )

    study_hours = st.number_input(
        "Study Hours per Day", 0.0, 24.0, 4.0
    )

    predict_button = st.form_submit_button(
        "Predict & Save Student",
        type="primary"
    )

if predict_button:

    if not student_name.strip():
        st.error("Please enter the student name.")

    else:
        student = pd.DataFrame([{
            "attendance": attendance,
            "internal_marks": internal_marks,
            "assignment_score": assignment_score,
            "study_hours": study_hours
        }])

        try:
            prediction = model.predict(student)[0]
            result = standardize_performance(prediction)

            if result == "Unknown":
                st.error(f"Unrecognized model result: {prediction}")

            else:
                # Automatically save the student record
                save_student(
                    student_name.strip(),
                    attendance,
                    internal_marks,
                    assignment_score,
                    study_hours,
                    result
                )

                st.success(
                    f"Prediction: {result} | Student data saved successfully!"
                )

                st.subheader("💡 Improvement Suggestions")

                suggestions = []

                if result == "Good":
                    suggestions.extend([
                        "Excellent progress! Maintain your study routine.",
                        "Try advanced questions and mock tests."
                    ])

                elif result == "Average":
                    suggestions.extend([
                        "Practise more questions.",
                        "Revise class notes daily.",
                        "Set weekly learning goals."
                    ])

                elif result == "Poor":
                    suggestions.extend([
                        "Focus on weak subjects.",
                        "Create a daily study timetable.",
                        "Seek additional academic support."
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
                        "Complete assignments on time."
                    )

                if study_hours < 2:
                    suggestions.append(
                        "Follow a consistent daily study schedule."
                    )

                for suggestion in dict.fromkeys(suggestions):
                    st.write("📌", suggestion)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------------- TEACHER DASHBOARD ----------------

st.divider()
st.header("👨‍🏫 Teacher Dashboard")

data = get_students()

total = len(data)

good = int((data["performance"] == "Good").sum())
average = int((data["performance"] == "Average").sum())
poor = int((data["performance"] == "Poor").sum())

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Students", total)
col2.metric("Good", good)
col3.metric("Average", average)
col4.metric("Poor", poor)

# ---------------- PERFORMANCE CHART ----------------

st.subheader("📊 Student Performance Chart")

if total > 0:
    chart_data = pd.DataFrame({
        "Performance": ["Good", "Average", "Poor"],
        "Students": [good, average, poor]
    })

    st.bar_chart(
        chart_data.set_index("Performance")
    )
else:
    st.info("No student records available yet.")

# ---------------- SEARCH STUDENTS ----------------

st.subheader("🔍 Search Student")

search_text = st.text_input(
    "Search by student name or performance"
)

if search_text.strip():
    filtered_data = data[
        data.astype(str).apply(
            lambda col: col.str.contains(
                search_text.strip(),
                case=False,
                regex=False,
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

# ---------------- DOWNLOAD RECORDS ----------------

st.subheader("📥 Download Student Records")

csv_data = data.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Student Data (CSV)",
    data=csv_data,
    file_name="student_records.csv",
    mime="text/csv",
    disabled=data.empty
)

# ---------------- EDIT STUDENT ----------------

st.divider()
st.subheader("✏️ Edit Student Record")

if not data.empty:

    selected_id = st.selectbox(
        "Select Student ID to Edit",
        data["id"].tolist()
    )

    selected = data[data["id"] == selected_id].iloc[0]

    with st.form("edit_student_form"):

        edit_name = st.text_input(
            "Student Name",
            value=str(selected["student_name"])
        )

        edit_attendance = st.number_input(
            "Attendance",
            min_value=0.0,
            max_value=100.0,
            value=float(selected["attendance"]),
            key="edit_attendance"
        )

        edit_marks = st.number_input(
            "Internal Marks",
            min_value=0.0,
            max_value=100.0,
            value=float(selected["internal_marks"]),
            key="edit_marks"
        )

        edit_assignment = st.number_input(
            "Assignment Score",
            min_value=0.0,
            max_value=100.0,
            value=float(selected["assignment_score"]),
            key="edit_assignment"
        )

        edit_hours = st.number_input(
            "Study Hours",
            min_value=0.0,
            max_value=24.0,
            value=float(selected["study_hours"]),
            key="edit_hours"
        )

        edit_performance = st.selectbox(
            "Performance",
            ["Good", "Average", "Poor"],
            index=["Good", "Average", "Poor"].index(
                selected["performance"]
            )
        )

        update_button = st.form_submit_button("Update Student")

    if update_button:
        if not edit_name.strip():
            st.error("Student name cannot be empty.")
        else:
            update_student(
                selected_id,
                edit_name.strip(),
                edit_attendance,
                edit_marks,
                edit_assignment,
                edit_hours,
                edit_performance
            )

            st.success("Student record updated successfully!")
            st.rerun()

# ---------------- DELETE STUDENT ----------------

st.subheader("🗑️ Delete Student Record")

if not data.empty:

    delete_id = st.selectbox(
        "Select Student ID to Delete",
        data["id"].tolist(),
        key="delete_student_id"
    )

    confirm_delete = st.checkbox(
        "I confirm that I want to delete this student record."
    )

    if st.button("Delete Selected Student", type="secondary"):

        if confirm_delete:
            delete_student(delete_id)
            st.success("Student record deleted successfully!")
            st.rerun()
        else:
            st.warning("Please confirm before deleting.")

else:
    st.info("No saved student records yet.")

# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "AI Student Performance Prediction System | "
    "Powered by Python, Streamlit, SQLite and Machine Learning"
)
