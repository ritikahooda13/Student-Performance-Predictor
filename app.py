import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Page configuration
st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="centered")

# App Title
st.title("🎓 Student Performance Predictor")
st.write("Predict your Final Marks based on academic and personal factors.")
st.markdown("---")

# Load data and train model behind the scenes
@st.cache_data
def get_trained_model():
    df = pd.read_csv('student_marks.csv')
    X = df[['Study_Hours', 'Attendance', 'Previous_Score', 'Assignments_Score']]
    y = df['Final_Marks']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = get_trained_model()

# User Inputs (Sidebar or Main Page)
st.subheader("📝 Enter Student Details:")

study_hours = st.slider("Daily Study Hours", min_value=2.0, max_value=10.0, value=5.0, step=0.1)
attendance = st.slider("Attendance (%)", min_value=60, max_value=100, value=85, step=1)
previous_score = st.number_input("Previous Exam Score (Out of 100)", min_value=40, max_value=100, value=75, step=1)
assignments_score = st.number_input("Assignments Score (Out of 100)", min_value=50, max_value=100, value=80, step=1)

st.markdown("---")

# Prediction Button
if st.button("🔮 Predict Final Marks", type="primary"):
    # Prepare input for model
    input_data = np.array([[study_hours, attendance, previous_score, assignments_score]])
    
    # Predict
    prediction = model.predict(input_data)[0]
    
    # Display Result
    st.balloons()
    st.success(f"### 🎯 Predicted Final Marks: **{prediction:.1f} / 100**")
    
    # Feedback messages based on performance
    if prediction >= 85:
        st.info("🌟 Excellent performance! Keep it up.")
    elif prediction >= 60:
        st.info("👍 Good job! Room for improvement with more study hours.")
    else:
        st.warning("⚠️ Warning: Student might need extra support and guidance.")