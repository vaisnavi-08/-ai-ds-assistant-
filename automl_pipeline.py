"""
Module 4: AutoML Pipeline
Automatically trains, compares, and selects the best ML model.
Uses scikit-learn ensemble of models to simulate AutoML behavior.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               ExtraTreesClassifier, AdaBoostClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, accuracy_score)
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')
import os

OUTPUT_DIR = 'outputs/automl'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_automl(df: pd.DataFrame, target_col: str = 'burnout_risk'):
    print("\n" + "="*60)
    print("         AUTOML — AUTOMATED MODEL SELECTION")
    print("="*60)

    # ── Prepare data ──────────────────────────────────────────
    feature_cols = [c for c in df.columns if c != target_col and c != 'productivity_score']
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n📊 Data Split")
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Test : {X_test.shape[0]} samples")
    print(f"   Features: {X_train.shape[1]}")

    # ── Define models ─────────────────────────────────────────
    models = {
        'Random Forest'         : RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting'     : GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Extra Trees'           : ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'AdaBoost'              : AdaBoostClassifier(n_estimators=100, random_state=42),
        'Logistic Regression'   : LogisticRegression(max_iter=1000, random_state=42),
        'KNN'                   : KNeighborsClassifier(n_neighbors=7),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    print(f"\n🤖 Training & Evaluating {len(models)} Models...")
    print(f"   {'Model':<25} {'CV AUC':>8} {'CV Acc':>8} {'Test AUC':>9} {'Test Acc':>9}")
    print(f"   {'-'*25} {'-'*8} {'-'*8} {'-'*9} {'-'*9}")

    trained_models = {}
    for name, model in models.items():
        # Cross-validation
        cv_auc  = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc').mean()
        cv_acc  = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy').mean()

        # Train on full training set
        model.fit(X_train, y_train)
        y_pred      = model.predict(X_test)
        y_proba     = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        test_auc    = roc_auc_score(y_test, y_proba)
        test_acc    = accuracy_score(y_test, y_pred)

        trained_models[name] = model
        results.append({
            'Model': name, 'CV_AUC': cv_auc, 'CV_Acc': cv_acc,
            'Test_AUC': test_auc, 'Test_Acc': test_acc
        })
        print(f"   {name:<25} {cv_auc:>8.4f} {cv_acc:>8.4f} {test_auc:>9.4f} {test_acc:>9.4f}")

    results_df = pd.DataFrame(results).sort_values('Test_AUC', ascending=False)

    # ── Best Model ────────────────────────────────────────────
    best_name  = results_df.iloc[0]['Model']
    best_model = trained_models[best_name]
    best_auc   = results_df.iloc[0]['Test_AUC']
    best_acc   = results_df.iloc[0]['Test_Acc']

    print(f"\n🏆 BEST MODEL: {best_name}")
    print(f"   Test AUC      : {best_auc:.4f}")
    print(f"   Test Accuracy : {best_acc:.4f}")

    y_pred_best  = best_model.predict(X_test)
    y_proba_best = best_model.predict_proba(X_test)[:, 1]

    print(f"\n📋 Classification Report ({best_name}):")
    print(classification_report(y_test, y_pred_best, target_names=['Low Risk', 'High Risk']))

    # ── Plots ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('AutoML — Model Evaluation', fontsize=14, fontweight='bold')

    # Model comparison
    colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(results_df))]
    axes[0].barh(results_df['Model'], results_df['Test_AUC'], color=colors, edgecolor='white')
    axes[0].set_title('Model Comparison (Test AUC)')
    axes[0].set_xlabel('AUC Score')
    axes[0].axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    axes[0].legend()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_best)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=['Low Risk', 'High Risk'],
                yticklabels=['Low Risk', 'High Risk'])
    axes[1].set_title(f'Confusion Matrix\n{best_name}')
    axes[1].set_ylabel('Actual')
    axes[1].set_xlabel('Predicted')

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba_best)
    axes[2].plot(fpr, tpr, color='#2ecc71', lw=2, label=f'ROC (AUC = {best_auc:.3f})')
    axes[2].plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Random')
    axes[2].fill_between(fpr, tpr, alpha=0.1, color='#2ecc71')
    axes[2].set_title(f'ROC Curve — {best_name}')
    axes[2].set_xlabel('False Positive Rate')
    axes[2].set_ylabel('True Positive Rate')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/automl_evaluation.png', dpi=120, bbox_inches='tight')
    plt.close()

    # Feature importance
    if hasattr(best_model, 'feature_importances_'):
        fi = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 8))
        fi.head(20).plot.barh(ax=ax, color='#3498db', edgecolor='white')
        ax.set_title(f'Top 20 Feature Importances — {best_name}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Importance Score')
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/feature_importance.png', dpi=120, bbox_inches='tight')
        plt.close()
        print(f"\n🔑 Top 5 Features:")
        for feat, score in fi.head(5).items():
            print(f"   {feat:<35} {score:.4f}")

    # Save results & model
    results_df.to_csv(f'{OUTPUT_DIR}/model_comparison.csv', index=False)
    joblib.dump(best_model, f'{OUTPUT_DIR}/best_model.pkl')
    print(f"\n💾 Saved: model_comparison.csv, best_model.pkl, automl_evaluation.png")

    return best_model, results_df, X_test, y_test, feature_cols

if __name__ == '__main__':
    df = pd.read_csv('outputs/cleaned_data.csv')
    run_automl(df)
