"""
Dataset Generator: Employee Productivity & Burnout
Generates a realistic synthetic dataset for data scientists to work with.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

N = 1000  # number of records

departments = ['Engineering', 'Data Science', 'Marketing', 'HR', 'Finance', 'Operations']
job_levels = ['Junior', 'Mid', 'Senior', 'Lead', 'Manager']
work_modes = ['Remote', 'Hybrid', 'On-site']
genders = ['Male', 'Female', 'Non-binary']

data = {
    'employee_id': [f'EMP{str(i).zfill(4)}' for i in range(1, N+1)],
    'age': np.random.randint(22, 60, N),
    'gender': np.random.choice(genders, N, p=[0.48, 0.48, 0.04]),
    'department': np.random.choice(departments, N),
    'job_level': np.random.choice(job_levels, N, p=[0.25, 0.30, 0.25, 0.12, 0.08]),
    'years_experience': np.random.randint(0, 35, N),
    'work_mode': np.random.choice(work_modes, N, p=[0.40, 0.35, 0.25]),
    'weekly_work_hours': np.random.normal(45, 8, N).clip(30, 80).astype(int),
    'meetings_per_week': np.random.randint(2, 25, N),
    'tasks_completed_per_week': np.random.randint(5, 40, N),
    'collaboration_score': np.random.uniform(1, 10, N).round(2),
    'training_hours_per_year': np.random.randint(0, 100, N),
    'salary_usd': np.random.randint(35000, 180000, N),
    'last_promotion_months_ago': np.random.randint(0, 72, N),
    'manager_rating': np.random.uniform(1, 5, N).round(1),
    'self_rating': np.random.uniform(1, 5, N).round(1),
    'sick_days_per_year': np.random.randint(0, 30, N),
    'overtime_frequency': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Always'], N),
    'work_life_balance_score': np.random.uniform(1, 10, N).round(2),
    'stress_level': np.random.randint(1, 11, N),
    'sleep_hours_avg': np.random.normal(6.5, 1.2, N).clip(4, 10).round(1),
    'exercise_days_per_week': np.random.randint(0, 7, N),
    'num_direct_reports': np.random.randint(0, 12, N),
    'project_success_rate': np.random.uniform(0.4, 1.0, N).round(2),
    'remote_tools_proficiency': np.random.randint(1, 11, N),
}

df = pd.DataFrame(data)

# Introduce realistic missing values
for col in ['manager_rating', 'training_hours_per_year', 'sleep_hours_avg', 'exercise_days_per_week']:
    mask = np.random.rand(N) < 0.05
    df.loc[mask, col] = np.nan

# Create target: burnout_risk (0 = Low, 1 = High) based on realistic factors
burnout_score = (
    (df['weekly_work_hours'] > 55).astype(int) * 2 +
    (df['stress_level'] > 7).astype(int) * 3 +
    (df['work_life_balance_score'] < 4).astype(int) * 2 +
    (df['sick_days_per_year'] > 15).astype(int) * 1 +
    (df['sleep_hours_avg'] < 6).astype(int) * 2 +
    (df['overtime_frequency'].isin(['Often', 'Always'])).astype(int) * 2 +
    (df['last_promotion_months_ago'] > 48).astype(int) * 1 +
    (df['manager_rating'] < 2.5).astype(int) * 1
)
df['burnout_risk'] = (burnout_score >= 5).astype(int)

# Create productivity score (regression target)
df['productivity_score'] = (
    df['tasks_completed_per_week'] * 2.5 +
    df['collaboration_score'] * 3 +
    df['project_success_rate'] * 20 +
    df['manager_rating'].fillna(3) * 4 -
    df['stress_level'] * 1.5 +
    np.random.normal(0, 5, N)
).clip(0, 100).round(2)

df.to_csv('employee_data.csv', index=False)
print(f"✅ Dataset generated: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"   Burnout Rate: {df['burnout_risk'].mean()*100:.1f}%")
print(f"   Missing values per column:")
print(df.isnull().sum()[df.isnull().sum() > 0])
