"""
🚀 MAIN PIPELINE — AI-Powered Data Science Assistant
Full end-to-end workflow: Data → EDA → Clean → AutoML → Deep Learning → Viz → Explain
"""

import os, sys, time
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── All imports at top level (fixes SyntaxError) ──────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_dataset import *
from auto_eda import auto_eda
from data_cleaning import clean_pipeline
from automl_pipeline import run_automl
from deep_learning import build_and_train
from visualization import generate_visualizations
from explainability import explain_model

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║      🤖 AI-POWERED DATA SCIENCE ASSISTANT PIPELINE          ║
║         Employee Productivity & Burnout Prediction          ║
╚══════════════════════════════════════════════════════════════╝
"""

def step(n, title):
    print(f"\n{'─'*60}")
    print(f"  STEP {n}: {title}")
    print(f"{'─'*60}")

def main():
    print(BANNER)
    total_start = time.time()

    # ── STEP 1: Generate Dataset ──────────────────────────────
    step(1, "GENERATE SYNTHETIC DATASET")
    df_raw = pd.read_csv(os.path.join(os.path.dirname(__file__), 'employee_data.csv'))
    print(f"✅ Dataset loaded: {df_raw.shape}")

    # ── STEP 2: Auto EDA ──────────────────────────────────────
    step(2, "AUTOMATED EXPLORATORY DATA ANALYSIS")
    num_cols, cat_cols = auto_eda(df_raw, target_col='burnout_risk')

    # ── STEP 3: Data Cleaning ─────────────────────────────────
    step(3, "AI-POWERED DATA CLEANING")
    df_clean, df_scaled, scaler, encoders = clean_pipeline(df_raw, target_col='burnout_risk')

    # ── STEP 4: AutoML ────────────────────────────────────────
    step(4, "AUTOML — AUTOMATED MODEL SELECTION")
    best_model, results_df, X_test, y_test, feature_cols = run_automl(df_clean, target_col='burnout_risk')

    # ── STEP 5: Deep Learning ─────────────────────────────────
    step(5, "DEEP LEARNING — TENSORFLOW / KERAS")
    dl_model, history, dl_auc = build_and_train(df_scaled, epochs=50)

    # ── STEP 6: Visualizations ────────────────────────────────
    step(6, "INTERACTIVE VISUALIZATIONS")
    generate_visualizations(df_raw, df_clean)

    # ── STEP 7: Model Explainability ──────────────────────────
    step(7, "MODEL EXPLAINABILITY")
    explain_model(best_model, X_test, y_test, feature_cols)

    # ── FINAL SUMMARY ─────────────────────────────────────────
    elapsed = time.time() - total_start
    best_auc  = results_df.iloc[0]['Test_AUC']
    best_name = results_df.iloc[0]['Model']

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                  ✅ PIPELINE COMPLETE                        ║
╠══════════════════════════════════════════════════════════════╣
║  Total Time       : {elapsed:.1f}s
║  Dataset Size     : {df_raw.shape[0]} rows x {df_raw.shape[1]} cols
║  Best ML Model    : {best_name}
║  Best ML AUC      : {best_auc:.4f}
║  Deep Learning AUC: {dl_auc:.4f}
╠══════════════════════════════════════════════════════════════╣
║  OUTPUT FILES:                                              ║
║  📁 outputs/eda/           — EDA plots (4 charts)           ║
║  📁 outputs/automl/        — Model comparison + ROC         ║
║  📁 outputs/deep_learning/ — Training curves                ║
║  📁 outputs/visualizations/— Plotly dashboards + Seaborn    ║
║  📁 outputs/explainability/— SHAP + Feature importance      ║
╚══════════════════════════════════════════════════════════════╝
""")

if __name__ == '__main__':
    main()
