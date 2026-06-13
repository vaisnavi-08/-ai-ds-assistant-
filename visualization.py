"""
Module 6: Interactive Visualizations (Plotly + Seaborn)
Generates rich dashboards and charts for insights communication.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
import os

OUTPUT_DIR = 'outputs/visualizations'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_visualizations(df_raw: pd.DataFrame, df_clean: pd.DataFrame = None):
    print("\n" + "="*60)
    print("       INTERACTIVE VISUALIZATIONS — PLOTLY + SEABORN")
    print("="*60)

    sns.set_style("whitegrid")
    sns.set_palette("husl")

    # ── Plot 1: Burnout Risk Dashboard (Plotly) ───────────────
    print("\n🎨 Generating Plot 1: Burnout Risk Dashboard...")
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'Burnout by Department', 'Stress vs Work Hours',
            'Work-Life Balance Distribution', 'Burnout by Work Mode',
            'Salary vs Productivity', 'Overtime vs Burnout'
        ),
        specs=[[{"type": "bar"}, {"type": "scatter"}, {"type": "histogram"}],
               [{"type": "bar"}, {"type": "scatter"},  {"type": "bar"}]]
    )

    # Burnout by department
    dept_burn = df_raw.groupby('department')['burnout_risk'].mean().reset_index()
    fig.add_trace(go.Bar(
        x=dept_burn['department'], y=dept_burn['burnout_risk'],
        marker_color='#e74c3c', name='Burnout Rate', showlegend=False
    ), row=1, col=1)

    # Stress vs Work Hours (scatter)
    colors = ['#2ecc71' if v == 0 else '#e74c3c' for v in df_raw['burnout_risk']]
    fig.add_trace(go.Scatter(
        x=df_raw['weekly_work_hours'], y=df_raw['stress_level'],
        mode='markers', marker=dict(color=colors, size=4, opacity=0.5),
        name='Employees', showlegend=False
    ), row=1, col=2)

    # Work-life balance
    fig.add_trace(go.Histogram(
        x=df_raw['work_life_balance_score'], nbinsx=20,
        marker_color='#3498db', showlegend=False
    ), row=1, col=3)

    # Burnout by work mode
    wm_burn = df_raw.groupby('work_mode')['burnout_risk'].mean().reset_index()
    fig.add_trace(go.Bar(
        x=wm_burn['work_mode'], y=wm_burn['burnout_risk'],
        marker_color='#9b59b6', showlegend=False
    ), row=2, col=1)

    # Salary vs productivity
    fig.add_trace(go.Scatter(
        x=df_raw['salary_usd'], y=df_raw['productivity_score'],
        mode='markers', marker=dict(color='#f39c12', size=4, opacity=0.4),
        showlegend=False
    ), row=2, col=2)

    # Overtime vs burnout
    ot_burn = df_raw.groupby('overtime_frequency')['burnout_risk'].mean().reset_index()
    fig.add_trace(go.Bar(
        x=ot_burn['overtime_frequency'], y=ot_burn['burnout_risk'],
        marker_color='#1abc9c', showlegend=False
    ), row=2, col=3)

    fig.update_layout(
        title_text="🔥 Employee Burnout Risk — Interactive Dashboard",
        title_font_size=16,
        height=700,
        template='plotly_white'
    )
    fig.write_html(f'{OUTPUT_DIR}/burnout_dashboard.html')
    print("   ✅ burnout_dashboard.html")

    # ── Plot 2: Productivity Heatmap (Seaborn) ────────────────
    print("\n🎨 Generating Plot 2: Productivity Heatmap...")
    fig2, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig2.suptitle('Productivity & Burnout Deep Dive', fontsize=15, fontweight='bold')

    # Heatmap: avg productivity by dept × job_level
    pivot = df_raw.pivot_table(
        values='productivity_score', index='department',
        columns='job_level', aggfunc='mean'
    )
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd',
                ax=axes[0], linewidths=0.5, cbar_kws={'label': 'Avg Productivity'})
    axes[0].set_title('Avg Productivity Score\nDepartment × Job Level')
    axes[0].set_xlabel('Job Level')
    axes[0].set_ylabel('Department')

    # Box plot: productivity by work mode
    df_raw.boxplot(column='productivity_score', by='work_mode', ax=axes[1],
                   patch_artist=True)
    axes[1].set_title('Productivity by Work Mode')
    axes[1].set_xlabel('Work Mode')
    axes[1].set_ylabel('Productivity Score')
    plt.suptitle('')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/productivity_heatmap.png', dpi=130, bbox_inches='tight')
    plt.close()
    print("   ✅ productivity_heatmap.png")

    # ── Plot 3: Violin plots (Seaborn) ───────────────────────
    print("\n🎨 Generating Plot 3: Violin Plots...")
    fig3, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig3.suptitle('Feature Distribution by Burnout Risk', fontsize=14, fontweight='bold')

    for i, col in enumerate(['stress_level', 'weekly_work_hours', 'sleep_hours_avg']):
        if col in df_raw.columns:
            sns.violinplot(data=df_raw, x='burnout_risk', y=col, ax=axes[i],
                           palette=['#2ecc71', '#e74c3c'], inner='box')
            axes[i].set_title(col.replace('_', ' ').title())
            axes[i].set_xlabel('Burnout Risk (0=Low, 1=High)')
            axes[i].set_xticklabels(['Low Risk', 'High Risk'])

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/violin_plots.png', dpi=130, bbox_inches='tight')
    plt.close()
    print("   ✅ violin_plots.png")

    # ── Plot 4: Plotly Sunburst ───────────────────────────────
    print("\n🎨 Generating Plot 4: Sunburst Chart...")
    sun_df = df_raw.groupby(
        ['department', 'work_mode', 'overtime_frequency']
    )['burnout_risk'].mean().reset_index()
    sun_df['burnout_pct'] = (sun_df['burnout_risk'] * 100).round(1)

    fig4 = px.sunburst(
        sun_df, path=['department', 'work_mode', 'overtime_frequency'],
        values='burnout_pct', color='burnout_pct',
        color_continuous_scale='RdYlGn_r',
        title='Burnout Risk Breakdown — Department > Work Mode > Overtime'
    )
    fig4.update_layout(height=600, template='plotly_white')
    fig4.write_html(f'{OUTPUT_DIR}/sunburst_burnout.html')
    print("   ✅ sunburst_burnout.html")

    # ── Plot 5: Pairplot key features (Seaborn) ───────────────
    print("\n🎨 Generating Plot 5: Pairplot...")
    pair_cols = ['stress_level', 'weekly_work_hours', 'work_life_balance_score',
                 'productivity_score', 'burnout_risk']
    pair_df = df_raw[pair_cols].copy()
    pair_df['burnout_risk'] = pair_df['burnout_risk'].map({0: 'Low Risk', 1: 'High Risk'})

    g = sns.pairplot(pair_df, hue='burnout_risk', palette={'Low Risk': '#2ecc71', 'High Risk': '#e74c3c'},
                     diag_kind='kde', plot_kws={'alpha': 0.4, 's': 20})
    g.figure.suptitle('Pairplot — Key Features by Burnout Risk', y=1.02, fontsize=13, fontweight='bold')
    g.figure.savefig(f'{OUTPUT_DIR}/pairplot.png', dpi=110, bbox_inches='tight')
    plt.close()
    print("   ✅ pairplot.png")

    print(f"\n✅ All visualizations saved to: {OUTPUT_DIR}/")

if __name__ == '__main__':
    df = pd.read_csv('employee_data.csv')
    generate_visualizations(df)
