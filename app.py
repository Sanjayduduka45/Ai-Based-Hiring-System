
import streamlit as st
import pandas as pd
import joblib


# PAGE CONFIG


st.set_page_config(
    page_title="AI Hiring Prediction System",
    page_icon="📄",
    layout="wide"
)

# =====================================================
# LOAD MODEL
# =====================================================

model = joblib.load("Ai_based_hiring_system.pkl")

# =====================================================
# HEADER
# =====================================================

st.title("📄 AI-Based Hiring Prediction System")

st.markdown("""
This application predicts whether a candidate is likely to be **Hired** or **Rejected**
based on resume information using a trained Machine Learning model.
""")

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("📝 Candidate Information")

skills = st.sidebar.text_area(
    "Skills",
    placeholder="Python, SQL, Machine Learning, TensorFlow"
)

experience = st.sidebar.number_input(
    "Experience (Years)",
    min_value=0,
    max_value=50,
    value=2
)

education = st.sidebar.selectbox(
    "Education",
    [
        "B.Tech",
        "B.Sc",
        "M.Tech",
        "M.Sc",
        "MBA",
        "PhD"
    ]
)

certification = st.sidebar.selectbox(
    "Certifications",
    [
        "No Certifications",
        "AWS Certified",
        "Google ML",
        "Deep Learning Specialization",
        "Azure Fundamentals",
        "Data Science Professional"
    ]
)

job_role = st.sidebar.selectbox(
    "Job Role",
    [
        "Data Scientist",
        "Machine Learning Engineer",
        "Software Engineer",
        "AI Researcher",
        "Cybersecurity Analyst",
        "Data Analyst"
    ]
)

salary = st.sidebar.number_input(
    "Salary Expectation ($)",
    min_value=0,
    max_value=1000000,
    value=50000
)

projects = st.sidebar.number_input(
    "Projects Count",
    min_value=0,
    max_value=100,
    value=2
)

predict_btn = st.sidebar.button("🚀 Predict Hiring Decision")

# =====================================================
# MAIN CONTENT
# =====================================================

col1, col2 = st.columns(2)

with col1:

    st.subheader("📋 Candidate Profile")

    st.write(f"**Skills:** {skills}")
    st.write(f"**Experience:** {experience} Years")
    st.write(f"**Education:** {education}")
    st.write(f"**Certification:** {certification}")
    st.write(f"**Job Role:** {job_role}")
    st.write(f"**Salary Expectation:** ${salary:,.0f}")
    st.write(f"**Projects Count:** {projects}")

with col2:

    st.subheader("🎯 Prediction Result")

    if predict_btn:

        input_df = pd.DataFrame({

            "Skills": [skills],

            "Experience (Years)": [experience],

            "Education": [education],

            "Certifications": [certification],

            "Job Role": [job_role],

            "Salary Expectation ($)": [salary],

            "Projects Count": [projects]

        })

        prediction = model.predict(input_df)[0]

        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:

            st.success("✅ Candidate Likely To Be Hired")

        else:

            st.error("❌ Candidate Likely To Be Rejected")

        st.metric(
            "Hiring Probability",
            f"{probability:.2%}"
        )

        st.progress(float(probability))

# =====================================================
# PROJECT INFORMATION
# =====================================================

st.markdown("---")

st.subheader("📊 Model Information")

col3, col4, col5 = st.columns(3)

with col3:
    st.metric("Accuracy", "97.0%")

with col4:
    st.metric("F1 Score", "98.1%")

with col5:
    st.metric("ROC-AUC", "99.6%")

st.markdown("---")

st.subheader("🔍 Project Overview")

st.info(
    """
    This project uses Machine Learning and NLP techniques
    to automate resume screening and hiring prediction.

    Features Used:
    • Skills (TF-IDF NLP)
    • Experience
    • Education
    • Certifications
    • Job Role
    • Salary Expectation
    • Projects Count

    Best Model:
    • XGBoost Classifier
    """
)

st.markdown("---")

st.subheader("💡 How It Works")

st.write("""
1. Enter candidate information from the sidebar.
2. Click **Predict Hiring Decision**.
3. The trained XGBoost model analyzes the candidate profile.
4. The system predicts whether the candidate is likely to be hired.
5. The hiring probability is displayed instantly.
""")

st.markdown("---")

st.caption(
    "Developed using Python, Streamlit, NLP (TF-IDF), XGBoost, and Scikit-Learn."
)

