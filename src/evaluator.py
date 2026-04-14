"""
=============================================================================
FILE: src/evaluator.py
PROJECT: AI-Powered Cybersecurity Threat Detection System
AUTHOR: Your Name

DESCRIPTION:
    Evaluates model performance using key cybersecurity-relevant metrics.

METRICS EXPLAINED (for beginners):
    ┌────────────────────────────────────────────────────────────────────┐
    │  CONFUSION MATRIX                                                  │
    │                                                                    │
    │           Predicted: NORMAL  |  Predicted: ATTACK                 │
    │  Actual NORMAL    TN (Good)  |  FP (False Alarm) ← we want low    │
    │  Actual ATTACK    FN (Miss!) |  TP (Good)        ← we want high   │
    │                                                                    │
    │  Accuracy   = (TP + TN) / Total       — Overall correctness       │
    │  Precision  = TP / (TP + FP)          — When we say "Attack",     │
    │                                         how often are we right?   │
    │  Recall     = TP / (TP + FN)          — Of all real attacks,      │
    │                                         how many did we catch?    │
    │  F1-Score   = 2 * (P * R) / (P + R)  — Balance of P and R        │
    │                                                                    │
    │  IN CYBERSECURITY: High Recall > High Precision                    │
    │  Missing an attack (FN) is far worse than a false alarm (FP)       │
    └────────────────────────────────────────────────────────────────────┘

    ROC-AUC Score:
        → Measures how well the model separates Normal vs Attack
        → AUC = 1.0 = Perfect | AUC = 0.5 = Random guessing
=============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)


# ─────────────────────────────────────────────
# SECTION 1: Full Evaluation Report
# ─────────────────────────────────────────────
def evaluate_model(model,
                   X_test: np.ndarray,
                   y_test: np.ndarray,
                   model_name: str = "Model",
                   save_dir: str = None) -> dict:
    """
    Complete evaluation of a trained classification model.

    Args:
        model      : Trained sklearn model with .predict() method
        X_test     : Test features
        y_test     : True test labels
        model_name : Name for display in reports
        save_dir   : Directory to save evaluation reports

    Returns:
        dict containing all metrics
    """
    print(f"\n[EVALUATOR] Evaluating {model_name}...")
    print("=" * 55)

    y_pred = model.predict(X_test)

    # For Random Forest: get probability scores (for ROC)
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]

    # ── Core Metrics ──────────────────────────────────────────────────────────
    acc       = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    auc       = roc_auc_score(y_test, y_prob) if y_prob is not None else None

    print(f"\n  Model         : {model_name}")
    print(f"  Accuracy      : {acc   * 100:.2f}%")
    print(f"  Precision     : {precision * 100:.2f}%  (Of all flagged attacks, how many were real?)")
    print(f"  Recall        : {recall    * 100:.2f}%  (Of all real attacks, how many did we catch?)")
    print(f"  F1-Score      : {f1        * 100:.2f}%  (Balanced score of Precision + Recall)")
    if auc:
        print(f"  ROC-AUC       : {auc:.4f}  (1.0 = perfect, 0.5 = random guessing)")

    # ── Classification Report ─────────────────────────────────────────────────
    report = classification_report(
        y_test, y_pred,
        target_names=["NORMAL (0)", "ATTACK (1)"],
        digits=4
    )
    print(f"\n  Full Classification Report:\n{report}")

    # ── Save Report to File ───────────────────────────────────────────────────
    metrics = {
        "model_name": model_name,
        "accuracy":   round(acc, 6),
        "precision":  round(precision, 6),
        "recall":     round(recall, 6),
        "f1_score":   round(f1, 6),
        "roc_auc":    round(auc, 6) if auc else None,
        "n_test_samples": len(y_test),
        "n_attacks_detected": int((y_pred == 1).sum()),
        "n_attacks_actual":   int((y_test == 1).sum()),
        "false_positives":    int(((y_pred == 1) & (y_test == 0)).sum()),
        "false_negatives":    int(((y_pred == 0) & (y_test == 1)).sum()),
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        report_path = os.path.join(save_dir, f"{model_name.replace(' ', '_')}_report.txt")
        json_path   = os.path.join(save_dir, f"{model_name.replace(' ', '_')}_metrics.json")

        with open(report_path, "w") as f:
            f.write(f"Model: {model_name}\n")
            f.write("=" * 55 + "\n")
            f.write(f"Accuracy  : {acc * 100:.4f}%\n")
            f.write(f"Precision : {precision * 100:.4f}%\n")
            f.write(f"Recall    : {recall * 100:.4f}%\n")
            f.write(f"F1-Score  : {f1 * 100:.4f}%\n")
            if auc:
                f.write(f"ROC-AUC   : {auc:.6f}\n")
            f.write("\n" + report)

        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"\n[EVALUATOR] Reports saved:\n  {report_path}\n  {json_path}")

    print("=" * 55)
    return metrics


# ─────────────────────────────────────────────
# SECTION 2: Plot Confusion Matrix
# ─────────────────────────────────────────────
def plot_confusion_matrix(y_test: np.ndarray,
                          y_pred: np.ndarray,
                          model_name: str,
                          save_path: str = None) -> None:
    """
    Plot and save a confusion matrix heatmap.

    The confusion matrix shows:
        TP: Attack correctly detected as Attack    → GOOD
        TN: Normal correctly detected as Normal    → GOOD
        FP: Normal incorrectly flagged as Attack   → False Alarm
        FN: Attack missed (detected as Normal)     → DANGEROUS

    WHY: In cybersecurity, FN (missed attacks) are most dangerous.
         We prefer FP (false alarms) over FN.
    """
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Predicted: NORMAL", "Predicted: ATTACK"],
        yticklabels=["Actual: NORMAL",    "Actual: ATTACK"],
        linewidths=2, linecolor="white",
        annot_kws={"size": 14, "weight": "bold"}
    )

    # Color-code each cell label
    labels = ["True Negative\n(✓ Correct)", "False Positive\n(⚠ False Alarm)",
              "False Negative\n(✗ MISSED!)", "True Positive\n(✓ Detected)"]
    for i, text in enumerate(ax.texts):
        row, col = divmod(i, 2)
        text.set_text(f"{text.get_text()}\n{labels[row * 2 + col]}")

    ax.set_title(f"Confusion Matrix — {model_name}\nAI Cybersecurity Threat Detection",
                 fontsize=12, fontweight="bold", pad=15)
    ax.set_ylabel("Actual Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[EVALUATOR] Confusion matrix saved: {save_path}")
    else:
        plt.show()


# ─────────────────────────────────────────────
# SECTION 3: Plot ROC Curve
# ─────────────────────────────────────────────
def plot_roc_curve(y_test: np.ndarray,
                   y_prob: np.ndarray,
                   model_name: str,
                   save_path: str = None) -> None:
    """
    Plot and save the ROC (Receiver Operating Characteristic) curve.

    WHAT IS ROC CURVE?
        - X-axis: False Positive Rate (FP / (FP + TN)) → false alarm rate
        - Y-axis: True Positive Rate (TP / (TP + FN))  → detection rate
        - A perfect model hugs the top-left corner
        - AUC (area under curve) = 1.0 is perfect

    WHY: ROC curve shows model performance across ALL thresholds.
         You can tune the threshold based on business need:
         → Lower threshold = catch more attacks (higher recall, more FP)
         → Higher threshold = fewer false alarms (lower recall, fewer FP)
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#2196F3", lw=2.5,
            label=f"{model_name} (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.5)")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2196F3")

    # Mark optimal threshold point (Youden's J statistic)
    optimal_idx = np.argmax(tpr - fpr)
    ax.scatter(fpr[optimal_idx], tpr[optimal_idx], color="red", zorder=5, s=80,
               label=f"Optimal Threshold = {thresholds[optimal_idx]:.3f}")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (False Alarm Rate)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Threat Detection Rate)", fontsize=11)
    ax.set_title(f"ROC Curve — {model_name}\nAI Cybersecurity Threat Detection",
                 fontsize=12, fontweight="bold", pad=15)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3, linestyle="--")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[EVALUATOR] ROC curve saved: {save_path}")
    else:
        plt.show()


# ─────────────────────────────────────────────
# SECTION 4: Compare Multiple Models
# ─────────────────────────────────────────────
def compare_models(metrics_list: list, save_path: str = None) -> None:
    """
    Create a bar chart comparing multiple models across all metrics.

    Args:
        metrics_list : List of dicts from evaluate_model()
        save_path    : Save comparison chart
    """
    if not metrics_list:
        print("[EVALUATOR] No metrics to compare.")
        return

    metric_cols = ["accuracy", "precision", "recall", "f1_score"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]
    models = [m["model_name"] for m in metrics_list]

    x = np.arange(len(models))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (col, label, color) in enumerate(zip(metric_cols, metric_labels, colors)):
        vals = [m[col] * 100 for m in metrics_list]
        bars = ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha='center', va='bottom', fontsize=8, fontweight="bold")

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Model Comparison — AI Cybersecurity Threat Detection\n"
                 "Accuracy | Precision | Recall | F1-Score",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim([0, 115])
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[EVALUATOR] Comparison chart saved: {save_path}")
    else:
        plt.show()
