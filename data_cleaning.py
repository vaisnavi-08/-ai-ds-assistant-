"""
Module 3: AI-Powered Data Cleaning Pipeline
Automatically detects and fixes data quality issues.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')
import os

os.makedirs('outputs', exist_ok=True)

def clean_pipeline(df: pd.DataFrame, target_col: str = 'burnout_risk'):
    print("\n" + "="*60)
    print("       AI-POWERED DATA CLEANING PIPELINE")
    print("="*60)

    df = df.copy()
    report = {}

    # ── Step 1: Drop ID columns ───────────────────────────────
    id_cols = [c for c in df.columns if 'id' in c.lower()]
    df.drop(columns=id_cols, inplace=True)
    print(f"\n✅ Step 1 — Dropped ID columns: {id_cols}")
    report['dropped_id_cols'] = id_cols

    # ── Step 2: Handle missing values ─────────────────────────
    num_cols  = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols  = df.select_dtypes(include='object').columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)

    missing_before = df.isnull().sum().sum()

    # Numeric: median imputation
    if num_cols:
        num_imputer = SimpleImputer(strategy='median')
        df[num_cols] = num_imputer.fit_transform(df[num_cols])

    # Categorical: most-frequent imputation
    if cat_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    missing_after = df.isnull().sum().sum()
    print(f"\n✅ Step 2 — Missing value imputation")
    print(f"   Before: {missing_before} missing  →  After: {missing_after} missing")
    report['missing_before'] = int(missing_before)
    report['missing_after']  = int(missing_after)

    # ── Step 3: Remove outliers (IQR) ─────────────────────────
    outlier_count = 0
    outlier_cols  = ['weekly_work_hours', 'salary_usd', 'sick_days_per_year',
                     'meetings_per_week', 'tasks_completed_per_week']
    for col in outlier_cols:
        if col not in df.columns:
            continue
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        before = len(df)
        df = df[(df[col] >= lower) & (df[col] <= upper)]
        outlier_count += before - len(df)
    print(f"\n✅ Step 3 — Outlier removal (IQR method)")
    print(f"   Removed {outlier_count} outlier rows  →  {len(df)} rows remaining")
    report['outliers_removed'] = outlier_count

    # ── Step 4: Encode categoricals ───────────────────────────
    encoders = {}
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    print(f"\n✅ Step 4 — Label Encoding applied to: {cat_cols}")
    report['encoded_cols'] = cat_cols

    # ── Step 5: Feature Engineering ───────────────────────────
    if 'weekly_work_hours' in df.columns and 'meetings_per_week' in df.columns:
        df['focus_time_ratio'] = (
            (df['weekly_work_hours'] - df['meetings_per_week']) /
            df['weekly_work_hours'].replace(0, 1)
        ).round(4)

    if 'tasks_completed_per_week' in df.columns and 'weekly_work_hours' in df.columns:
        df['tasks_per_hour'] = (
            df['tasks_completed_per_week'] / df['weekly_work_hours'].replace(0, 1)
        ).round(4)

    if 'stress_level' in df.columns and 'work_life_balance_score' in df.columns:
        df['wellbeing_index'] = (
            (df['work_life_balance_score'] - df['stress_level'] / 2)
        ).round(4)

    print(f"\n✅ Step 5 — Feature Engineering")
    print(f"   Added: focus_time_ratio, tasks_per_hour, wellbeing_index")
    report['new_features'] = ['focus_time_ratio', 'tasks_per_hour', 'wellbeing_index']

    # ── Step 6: Scale features ────────────────────────────────
    feature_cols = [c for c in df.columns if c != target_col]
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
    print(f"\n✅ Step 6 — StandardScaler applied to {len(feature_cols)} feature columns")

    # Save
    df.to_csv('outputs/cleaned_data.csv', index=False)
    df_scaled.to_csv('outputs/scaled_data.csv', index=False)
    print(f"\n💾 Saved: cleaned_data.csv  &  scaled_data.csv")

    print(f"\n{'='*60}")
    print(f"   CLEANING COMPLETE  —  Final shape: {df.shape}")
    print(f"{'='*60}")
    return df, df_scaled, scaler, encoders

if __name__ == '__main__':
    df = pd.read_csv('employee_data.csv')
    clean_pipeline(df)
