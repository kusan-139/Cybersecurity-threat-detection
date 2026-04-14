"""
=============================================================================
FILE: src/model_trainer.py
PROJECT: AI-Powered Cybersecurity Threat Detection System
AUTHOR: Your Name

DESCRIPTION:
    This module trains TWO complementary threat detection models:

    MODEL 1 — Random Forest Classifier (Supervised Learning)
    ─────────────────────────────────────────────────────────
    • Learns from LABELED data (Normal vs Attack)
    • Best for: detecting known, labeled attack patterns
    • Output: Binary classification (0=Normal, 1=Attack)
    • Strength: High accuracy, interpretable feature importances
    • Weakness: Cannot detect novel/unknown attacks without labels

    MODEL 2 — Isolation Forest (Unsupervised Anomaly Detection)
    ────────────────────────────────────────────────────────────
    • Learns from UNLABELED data (no labels needed)
    • Best for: detecting unknown/zero-day threats
    • Output: Anomaly score (negative = more anomalous)
    • Strength: Detects novelties even without labeled attack data
    • Weakness: Higher false positive rate

    WHY TWO MODELS?
    Real SOC teams use multi-layered detection systems.
    Combining supervised + unsupervised detection provides:
    - Better coverage of known AND unknown threats
    - Redundancy (if one model misses, the other catches)
    - More realistic industry architecture
=============================================================================
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest


# ─────────────────────────────────────────────
# SECTION 1: Train Random Forest (Supervised)
# ─────────────────────────────────────────────
def train_random_forest(X_train: np.ndarray,
                        y_train: np.ndarray,
                        save_path: str = None,
                        **kwargs) -> RandomForestClassifier:
    """
    Train a Random Forest Classifier for threat detection.

    HOW RANDOM FOREST WORKS:
        1. Creates many Decision Trees (n_estimators=100 by default)
        2. Each tree is trained on a random subset of data
        3. Each tree uses random subset of features at each split
        4. Final prediction = majority vote of all trees
        → Ensemble of weak learners creates a strong learner

    HYPERPARAMETERS:
        n_estimators    : Number of trees in the forest (more = better but slower)
        max_depth       : Maximum depth of each tree (prevents overfitting)
        min_samples_leaf: Minimum samples required in a leaf node
        class_weight    : 'balanced' → handles imbalanced datasets automatically
        n_jobs          : -1 → use all CPU cores for parallel training

    Args:
        X_train   : Training features (scaled)
        y_train   : Training labels (0/1)
        save_path : Path to save trained model (.pkl file)
        **kwargs  : Override default hyperparameters

    Returns:
        Trained RandomForestClassifier
    """
    print("\n[MODEL TRAINER] Training Random Forest Classifier...")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Features        : {X_train.shape[1]}")
    print(f"  Attack samples  : {y_train.sum()} ({y_train.mean()*100:.1f}%)")

    # Default hyperparameters (tuned for cybersecurity use case)
    params = {
        "n_estimators":     100,      # 100 trees — good balance of speed vs accuracy
        "max_depth":        20,       # Prevent overfitting
        "min_samples_leaf": 2,        # Ensure generalization
        "class_weight":     "balanced", # Handle class imbalance (more normal than attack)
        "n_jobs":           -1,       # Parallel on all cores
        "random_state":     42,       # Reproducibility
        "verbose":          0,
    }
    params.update(kwargs)  # Allow caller to override any param

    rf = RandomForestClassifier(**params)
    rf.fit(X_train, y_train)

    print(f"[MODEL TRAINER] Random Forest trained ✓")
    print(f"  Training Accuracy : {rf.score(X_train, y_train) * 100:.2f}%")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(rf, save_path)
        print(f"[MODEL TRAINER] Model saved: {save_path}")

    return rf


# ─────────────────────────────────────────────
# SECTION 2: Train Isolation Forest (Unsupervised)
# ─────────────────────────────────────────────
def train_isolation_forest(X_train: np.ndarray,
                           contamination: float = 0.20,
                           save_path: str = None,
                           **kwargs) -> IsolationForest:
    """
    Train an Isolation Forest for anomaly/novelty detection.

    HOW ISOLATION FOREST WORKS:
        1. Builds many random trees (like Random Forest)
        2. For each datapoint, counts how many splits are needed to isolate it
        3. NORMAL points: take more splits to isolate (mixed with others)
        4. ANOMALY points: isolated quickly (stand out from the crowd)
        5. Anomaly score = 1/average_path_length  → higher = more anomalous

    THINK OF IT LIKE:
        Imagine finding "Waldo" in a crowd vs finding a man in a red cape.
        The man in a red cape (anomaly) stands out immediately — few splits needed.

    HYPERPARAMETERS:
        n_estimators  : Number of isolation trees (100 is standard)
        contamination : Expected fraction of anomalies in training data
                        → Set to your estimated attack percentage
        max_samples   : "auto" uses min(256, n_samples)
        random_state  : Reproducibility

    Args:
        X_train      : Training features (can use only NORMAL samples for semi-supervised)
        contamination: Expected anomaly fraction (0.20 = 20% attacks expected)
        save_path    : Path to save the model
        **kwargs     : Override hyperparameters

    Returns:
        Trained IsolationForest
    """
    print("\n[MODEL TRAINER] Training Isolation Forest (Anomaly Detector)...")
    print(f"  Training samples  : {X_train.shape[0]}")
    print(f"  Features          : {X_train.shape[1]}")
    print(f"  Contamination     : {contamination} ({contamination*100:.0f}% expected attacks)")

    params = {
        "n_estimators":  100,
        "contamination": contamination,
        "max_samples":   "auto",
        "random_state":  42,
        "n_jobs":        -1,
        "verbose":       0,
    }
    params.update(kwargs)

    iso_forest = IsolationForest(**params)
    iso_forest.fit(X_train)   # Does NOT need y_train labels

    print(f"[MODEL TRAINER] Isolation Forest trained ✓")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(iso_forest, save_path)
        print(f"[MODEL TRAINER] Model saved: {save_path}")

    return iso_forest


# ─────────────────────────────────────────────
# SECTION 3: Load Saved Models
# ─────────────────────────────────────────────
def load_model(model_path: str):
    """
    Load a previously trained model from disk.

    Args:
        model_path: Path to the .pkl file

    Returns:
        Loaded model (RandomForest or IsolationForest)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}\nRun training first.")

    model = joblib.load(model_path)
    print(f"[MODEL TRAINER] Loaded model from: {model_path}")
    return model


# ─────────────────────────────────────────────
# SECTION 4: Get Anomaly Scores (from IF)
# ─────────────────────────────────────────────
def get_anomaly_scores(iso_forest: IsolationForest, X: np.ndarray) -> np.ndarray:
    """
    Get anomaly scores from a trained Isolation Forest.

    Scores are negative:
        → Closer to -1 = highly anomalous (likely attack)
        → Closer to  0 = borderline
        → Positive     = normal

    Returns:
        Array of anomaly scores
    """
    # decision_function returns: lower (more negative) = more anomalous
    scores = iso_forest.decision_function(X)
    return scores
