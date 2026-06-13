"""
Module 5: Deep Learning Model (MLP Neural Network via scikit-learn)
Trains a neural network for burnout risk prediction.
Compatible with Python 3.14+ where TensorFlow is not yet available.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os

OUTPUT_DIR = 'outputs/deep_learning'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def build_and_train(df: pd.DataFrame, target_col: str = 'burnout_risk', epochs: int = 50):
    print("\n" + "="*60)
    print("       DEEP LEARNING — MLP NEURAL NETWORK")
    print("="*60)

    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, roc_auc_score

    # ── Prepare data ──────────────────────────────────────────
    feature_cols = [c for c in df.columns if c != target_col and c != 'productivity_score']
    X = df[feature_cols].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )

    print(f"\n📊 Data Splits")
    print(f"   Train : {X_train.shape[0]} samples")
    print(f"   Val   : {X_val.shape[0]} samples")
    print(f"   Test  : {X_test.shape[0]} samples")
    print(f"   Features: {X_train.shape[1]}")

    # ── Build MLP Model ───────────────────────────────────────
    print(f"\n🧠 Model Architecture:")
    print(f"   Input({X_train.shape[1]}) → Dense(256) → Dense(128) → Dense(64) → Dense(32) → Output(1)")

    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        batch_size=32,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=epochs,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=10,
        verbose=False
    )

    # ── Train ─────────────────────────────────────────────────
    print(f"\n⚡ Training Neural Network ({epochs} epochs max)...")
    model.fit(X_train, y_train)
    actual_epochs = len(model.loss_curve_)
    print(f"   Stopped at epoch: {actual_epochs}")

    # ── Evaluate ──────────────────────────────────────────────
    y_pred       = model.predict(X_test)
    y_proba      = model.predict_proba(X_test)[:, 1]
    roc_auc      = roc_auc_score(y_test, y_proba)
    accuracy     = (y_pred == y_test).mean()

    print(f"\n📊 Test Results:")
    print(f"   Accuracy : {accuracy:.4f}")
    print(f"   AUC      : {roc_auc:.4f}")

    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))

    # ── Plots ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Neural Network Training Results', fontsize=14, fontweight='bold')

    # Loss curve
    axes[0].plot(model.loss_curve_, color='#e74c3c', lw=2, label='Train Loss')
    if hasattr(model, 'validation_scores_') and model.validation_scores_:
        val_loss = [1 - s for s in model.validation_scores_]
        axes[0].plot(val_loss, color='#3498db', lw=2, label='Val Loss')
    axes[0].set_title('Loss Curve')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # Validation score curve
    if hasattr(model, 'validation_scores_') and model.validation_scores_:
        axes[1].plot(model.validation_scores_, color='#2ecc71', lw=2, label='Val Accuracy')
        axes[1].set_title('Validation Accuracy Curve')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, f'Final AUC: {roc_auc:.4f}\nAccuracy: {accuracy:.4f}',
                    ha='center', va='center', fontsize=16, transform=axes[1].transAxes,
                    bbox=dict(boxstyle='round', facecolor='#2ecc71', alpha=0.3))
        axes[1].set_title('Final Results')
        axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/training_curves.png', dpi=120, bbox_inches='tight')
    plt.close()

    print(f"\n💾 Saved: training_curves.png")
    print(f"   Final Test AUC: {roc_auc:.4f}")

    return model, None, roc_auc

if __name__ == '__main__':
    df = pd.read_csv('outputs/scaled_data.csv')
    build_and_train(df, epochs=50)
