import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate 500 samples
n_samples = 500

study_hours = np.random.uniform(2, 10, n_samples).round(1)
attendance = np.random.randint(60, 100, n_samples)
previous_score = np.random.randint(40, 100, n_samples)
assignments_score = np.random.randint(50, 100, n_samples)

# Base formula for Final Marks with some realistic random noise
final_marks = (
    (study_hours * 2.5) + 
    (attendance * 0.3) + 
    (previous_score * 0.4) + 
    (assignments_score * 0.2) + 
    np.random.normal(0, 3, n_samples)
)

# Clip marks between 0 and 100
final_marks = np.clip(final_marks, 0, 100).round(1)

# Create DataFrame
df = pd.DataFrame({
    'Study_Hours': study_hours,
    'Attendance': attendance,
    'Previous_Score': previous_score,
    'Assignments_Score': assignments_score,
    'Final_Marks': final_marks
})

# Save to CSV
df.to_csv('student_marks.csv', index=False)
print("Dataset 'student_marks.csv' successfully created with 500 records!")