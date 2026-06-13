"""
Module 7: Model Explainability — SHAP Values & Feature Importance
Makes the black-box model interpretable for stakeholders.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')
import os

OUTPUT_DIR = 'outputs/explainability'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def explain_model(model, X_test: pd.DataFrame, y_test, feature_cols: list):
    print("\n" + "="*60)
    print("       MODEL EXPLAINABILITY — SHAP + FEATURE IMPORTANCE")
    print("="*60)

    # ── 1. Feature Importance (built-in) ─────────────────────
    print("\n📊 Feature Importance Analysis...")
    if hasattr(model, 'feature_importances_'):
        fi = pd.Series(model.feature_importances_, index=feature_cols)
        fi_sorted = fi.sort_values(ascending=False)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle('Model Explainability — Feature Importance', fontsize=14, fontweight='bold')

        # Top 20 bar chart
        colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, 20))
        fi_sorted.head(20).plot.barh(ax=axes[0], color=colors, edgecolor='white')
        axes[0].set_title('Top 20 Feature Importances')
        axes[0].set_xlabel('Importance Score')
        axes[0].invert_yaxis()
        for i, (val, label) in enumerate(zip(fi_sorted.head(20), fi_sorted.head(20).index)):
            axes[0].text(val + 0.001, i, f'{val:.3f}', va='center', fontsize=8)

        # Cumulative importance
        cum_imp = fi_sorted.cumsum() / fi_sorted.sum() * 100
        axes[1].plot(range(1, len(cum_imp)+1), cum_imp.values, color='#3498db', lw=2)
        axes[1].axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% threshold')
        axes[1].axhline(y=95, color='orange', linestyle='--', alpha=0.7, label='95% threshold')
        n80 = (cum_imp <= 80).sum() + 1
        axes[1].axvline(x=n80, color='red', linestyle=':', alpha=0.5)
        axes[1].fill_between(range(1, len(cum_imp)+1), cum_imp.values, alpha=0.1, color='#3498db')
        axes[1].set_title('Cumulative Feature Importance')
        axes[1].set_xlabel('Number of Features')
        axes[1].set_ylabel('Cumulative Importance (%)')
        axes[1].legend()
        axes[1].set_ylim(0, 105)

        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/feature_importance.png', dpi=130, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Feature importance saved")
        print(f"   Top 5 features driving burnout predictions:")
        for feat, score in fi_sorted.head(5).items():
            bar = '█' * int(score * 200)
            print(f"   {feat:<35} {bar} {score:.4f}")

    # ── 2. SHAP Values ────────────────────────────────────────
    print("\n🔍 Attempting SHAP analysis...")
    try:
        import shap
        shap.initjs()

        if hasattr(model, 'feature_importances_'):
            explainer  = shap.TreeExplainer(model)
            X_sample   = X_test.iloc[:200] if len(X_test) > 200 else X_test
            shap_vals  = explainer.shap_values(X_sample)

            # Handle binary classification (shap returns list)
            if isinstance(shap_vals, list):
                sv = shap_vals[1]
            else:
                sv = shap_vals

            # SHAP summary plot
            fig, axes = plt.subplots(1, 2, figsize=(18, 8))
            plt.sca(axes[0])
            shap.summary_plot(sv, X_sample, feature_names=feature_cols,
                              plot_type='bar', show=False, max_display=20)
            axes[0].set_title('SHAP Feature Importance (Mean |SHAP|)', fontsize=12)

            plt.sca(axes[1])
            shap.summary_plot(sv, X_sample, feature_names=feature_cols,
                              show=False, max_display=20)
            axes[1].set_title('SHAP Value Distribution', fontsize=12)

            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/shap_summary.png', dpi=120, bbox_inches='tight')
            plt.close()
            print("   ✅ SHAP summary plot saved")

            # SHAP dependence plot for top feature
            top_feat = fi_sorted.index[0]
            top_idx  = list(feature_cols).index(top_feat)
            fig, ax  = plt.subplots(figsize=(10, 6))
            shap.dependence_plot(top_idx, sv, X_sample.values,
                                 feature_names=feature_cols, ax=ax, show=False)
            ax.set_title(f'SHAP Dependence Plot — {top_feat}', fontsize=13)
            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/shap_dependence.png', dpi=120, bbox_inches='tight')
            plt.close()
            print(f"   ✅ SHAP dependence plot saved for: {top_feat}")

    except ImportError:
        print("   ⚠️  SHAP not installed — skipping SHAP analysis")
        print("      Install with: pip install shap")
    except Exception as e:
        print(f"   ⚠️  SHAP error: {e}")

    # ── 3. Prediction Analysis ────────────────────────────────
    print("\n📈 Prediction Confidence Analysis...")
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Prediction Analysis', fontsize=14, fontweight='bold')

    # Confidence distribution
    axes[0].hist(y_proba[y_test == 0], bins=30, alpha=0.6, color='#2ecc71', label='Low Risk (Actual)', edgecolor='white')
    axes[0].hist(y_proba[y_test == 1], bins=30, alpha=0.6, color='#e74c3c', label='High Risk (Actual)', edgecolor='white')
    axes[0].axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold')
    axes[0].set_title('Prediction Confidence Distribution')
    axes[0].set_xlabel('Predicted Probability (Burnout)')
    axes[0].set_ylabel('Count')
    axes[0].legend()

    # Correct vs incorrect predictions
    correct   = (y_pred == y_test.values).astype(int)
    axes[1].pie([correct.sum(), len(correct) - correct.sum()],
                labels=['Correct', 'Incorrect'],
                colors=['#2ecc71', '#e74c3c'], autopct='%1.1f%%',
                startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    axes[1].set_title(f'Prediction Accuracy\n({correct.sum()}/{len(correct)} correct)')

    # Threshold analysis
    thresholds = np.arange(0.1, 0.9, 0.05)
    accs, precisions, recalls = [], [], []
    for t in thresholds:
        p = (y_proba >= t).astype(int)
        from sklearn.metrics import precision_score, recall_score, accuracy_score
        accs.append(accuracy_score(y_test, p))
        precisions.append(precision_score(y_test, p, zero_division=0))
        recalls.append(recall_score(y_test, p, zero_division=0))
    axes[2].plot(thresholds, accs,       label='Accuracy',  color='#3498db', lw=2)
    axes[2].plot(thresholds, precisions, label='Precision', color='#2ecc71', lw=2)
    axes[2].plot(thresholds, recalls,    label='Recall',    color='#e74c3c', lw=2)
    axes[2].axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Default (0.5)')
    axes[2].set_title('Metric vs Decision Threshold')
    axes[2].set_xlabel('Threshold')
    axes[2].set_ylabel('Score')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/prediction_analysis.png', dpi=120, bbox_inches='tight')
    plt.close()
    print("   ✅ Prediction analysis saved")
    print(f"\n✅ All explainability outputs saved to: {OUTPUT_DIR}/")
