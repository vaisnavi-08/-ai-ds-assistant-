"""
Module 2: Automated Exploratory Data Analysis (Auto EDA)
Generates comprehensive statistical summaries and visualizations automatically.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os

OUTPUT_DIR = 'outputs/eda'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def auto_eda(df: pd.DataFrame, target_col: str = 'burnout_risk'):
    print("\n" + "="*60)
    print("       AUTO EXPLORATORY DATA ANALYSIS REPORT")
    print("="*60)

    # ── 1. Basic Info ──────────────────────────────────────────
    print(f"\n📊 DATASET OVERVIEW")
    print(f"   Rows        : {df.shape[0]}")
    print(f"   Columns     : {df.shape[1]}")
    print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    print(f"   Duplicates  : {df.duplicated().sum()}")

    # ── 2. Column Types ───────────────────────────────────────
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    print(f"\n   Numeric columns  : {len(num_cols)}")
    print(f"   Categorical cols : {len(cat_cols)}")

    # ── 3. Missing Values ─────────────────────────────────────
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print(f"\n🔍 MISSING VALUES")
    if missing.empty:
        print("   No missing values found!")
    else:
        for col, cnt in missing.items():
            print(f"   {col:<35} {cnt:>4} ({cnt/len(df)*100:.1f}%)")

    # ── 4. Target Distribution ────────────────────────────────
    print(f"\n🎯 TARGET COLUMN: {target_col}")
    vc = df[target_col].value_counts()
    for val, cnt in vc.items():
        bar = '█' * int(cnt / len(df) * 40)
        print(f"   Class {val}: {bar} {cnt} ({cnt/len(df)*100:.1f}%)")

    # ── 5. Numeric Stats ──────────────────────────────────────
    print(f"\n📈 NUMERIC FEATURES — DESCRIPTIVE STATS")
    stats = df[num_cols].describe().T[['mean', 'std', 'min', '50%', 'max']]
    stats.columns = ['Mean', 'Std', 'Min', 'Median', 'Max']
    print(stats.round(2).to_string())

    # ── 6. Plots ──────────────────────────────────────────────
    print(f"\n🎨 Generating EDA plots...")

    sns.set_style("whitegrid")
    palette = sns.color_palette("husl", 8)

    # Plot 1: Target distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Target Variable Analysis', fontsize=14, fontweight='bold')
    df[target_col].value_counts().plot.bar(ax=axes[0], color=palette[:2], edgecolor='white')
    axes[0].set_title('Burnout Risk Distribution')
    axes[0].set_xlabel('Burnout Risk (0=Low, 1=High)')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=0)

    key_cols = ['stress_level', 'weekly_work_hours', 'work_life_balance_score', 'productivity_score']
    for col in key_cols:
        if col in df.columns:
            axes[1].set_visible(False)
            break
    sns.kdeplot(data=df, x='stress_level', hue=target_col, ax=axes[1], fill=True, palette='husl')
    axes[1].set_title('Stress Level by Burnout Risk')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/01_target_distribution.png', dpi=120, bbox_inches='tight')
    plt.close()

    # Plot 2: Correlation heatmap
    fig, ax = plt.subplots(figsize=(14, 10))
    corr_cols = [c for c in num_cols if c != 'employee_id' and df[c].dtype != object]
    corr_matrix = df[corr_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=ax, annot_kws={'size': 7}, linewidths=0.5)
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/02_correlation_heatmap.png', dpi=120, bbox_inches='tight')
    plt.close()

    # Plot 3: Key numeric distributions
    key_numeric = ['age', 'weekly_work_hours', 'stress_level',
                   'work_life_balance_score', 'productivity_score', 'salary_usd']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Key Feature Distributions', fontsize=14, fontweight='bold')
    for i, col in enumerate(key_numeric):
        if col in df.columns:
            ax = axes[i // 3][i % 3]
            sns.histplot(df[col].dropna(), kde=True, ax=ax, color=palette[i], edgecolor='white')
            ax.set_title(col.replace('_', ' ').title())
            ax.set_xlabel('')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/03_feature_distributions.png', dpi=120, bbox_inches='tight')
    plt.close()

    # Plot 4: Categorical analysis
    cat_plot_cols = ['department', 'job_level', 'work_mode', 'overtime_frequency']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Burnout Risk by Categorical Features', fontsize=14, fontweight='bold')
    for i, col in enumerate(cat_plot_cols):
        if col in df.columns:
            ax = axes[i // 2][i % 2]
            ct = df.groupby(col)[target_col].mean().sort_values(ascending=False)
            ct.plot.bar(ax=ax, color=sns.color_palette('RdYlGn_r', len(ct)), edgecolor='white')
            ax.set_title(f'Avg Burnout Risk by {col.replace("_"," ").title()}')
            ax.set_ylabel('Burnout Rate')
            ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/04_categorical_burnout.png', dpi=120, bbox_inches='tight')
    plt.close()

    print(f"   ✅ Plots saved to: {OUTPUT_DIR}/")
    return num_cols, cat_cols

if __name__ == '__main__':
    df = pd.read_csv('employee_data.csv')
    auto_eda(df, target_col='burnout_risk')
