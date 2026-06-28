import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import os

# --- STEP 1: DATASET GENERATION (Agar file nahi hai toh automatic banegi) ---
if not os.path.exists('student_marks.csv'):
    print("Generating 'student_marks.csv' dataset...")
    np.random.seed(42)
    n_samples = 500

    study_hours = np.random.uniform(2, 10, n_samples).round(1)
    attendance = np.random.randint(60, 100, n_samples)
    previous_score = np.random.randint(40, 100, n_samples)
    assignments_score = np.random.randint(50, 100, n_samples)

    final_marks = (
        (study_hours * 2.5) + 
        (attendance * 0.3) + 
        (previous_score * 0.4) + 
        (assignments_score * 0.2) + 
        np.random.normal(0, 3, n_samples)
    )
    final_marks = np.clip(final_marks, 0, 100).round(1)

    df = pd.DataFrame({
        'Study_Hours': study_hours,
        'Attendance': attendance,
        'Previous_Score': previous_score,
        'Assignments_Score': assignments_score,
        'Final_Marks': final_marks
    })
    df.to_csv('student_marks.csv', index=False)
    print("Dataset successfully created!\n")

# --- STEP 2: MODEL TRAINING & EVALUATION ---
# 1. Load the dataset
df = pd.read_csv('student_marks.csv')

# 2. Separate Features (X) and Target Variable (y)
X = df[['Study_Hours', 'Attendance', 'Previous_Score', 'Assignments_Score']]
y = df['Final_Marks']

# 3. Split dataset into training and testing set (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Building (Random Forest Regressor)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Model Prediction
y_pred = model.predict(X_test)

# 6. Evaluation Metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(y_test, y_pred)

print("="*40)
print("     MODEL EVALUATION METRICS     ")
print("="*40)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R2 Score (Accuracy): {r2 * 100:.2f}%")
print("="*40)