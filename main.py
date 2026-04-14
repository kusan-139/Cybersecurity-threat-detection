"""
=============================================================================
FILE: main.py
PROJECT: AI-Powered Cybersecurity Threat Detection System
AUTHOR: Your Name
DATE  : 2024

DESCRIPTION:
    Master runner script that orchestrates the complete ML pipeline:

    PHASE 1  → Environment check
    PHASE 2  → Dataset generation / loading
    PHASE 3  → Data inspection & EDA
    PHASE 4  → Preprocessing (clean, encode, scale, split)
    PHASE 5  → Feature engineering (importance, selection)
    PHASE 6  → Model training (Random Forest + Isolation Forest)
    PHASE 7  → Model evaluation (accuracy, F1, ROC, confusion matrix)
    PHASE 8  → Threat detection & alert generation
    PHASE 9  → Visualization (6+ charts)
    PHASE 10 → Summary report

HOW TO RUN:
    python main.py

EXPECTED OUTPUT:
    - models/random_forest.pkl
    - models/isolation_forest.pkl
    - models/scaler.pkl
    - outputs/alert_log.csv
    - outputs/evaluation_report.txt
    - images/*.png  (6+ charts)
    - data/raw/synthetic_network_traffic.csv
=============================================================================
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

# ── Fix Windows console encoding (cp1252 → UTF-8) so Unicode chars print fine
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

warnings.filterwarnings("ignore")   # Suppress sklearn deprecation warnings

# ─────────────────────────────────────────────────────────────────────────────
# Ensure project root is in Python path
# ─────────────────────────────────────────────────────────────────────────────
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ─────────────────────────────────────────────────────────────────────────────
# Import all project modules
# ─────────────────────────────────────────────────────────────────────────────
from src.data_loader        import generate_synthetic_dataset, inspect_dataset, save_sample, load_dataset
from src.preprocessor       import preprocess
from src.feature_engineering import get_feature_importances, select_top_features, plot_correlation_heatmap
from src.model_trainer       import train_random_forest, train_isolation_forest, get_anomaly_scores
from src.evaluator           import evaluate_model, plot_confusion_matrix, plot_roc_curve, compare_models
from src.alert_generator     import generate_alerts, print_alert_summary
from src.visualizer          import (plot_class_distribution, plot_attack_distribution,
                                      plot_anomaly_scores, plot_alert_severity_dashboard,
                                      plot_feature_distributions)
from src.threat_detector     import ThreatDetector


# ─────────────────────────────────────────────────────────────────────────────
# PATH CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
PATHS = {
    # Data
    "raw_data":         "data/raw/synthetic_network_traffic.csv",
    "sample_data":      "data/raw/sample_data.csv",
    "processed_data":   "data/processed/cleaned_traffic.csv",

    # Models
    "rf_model":         "models/random_forest.pkl",
    "if_model":         "models/isolation_forest.pkl",
    "scaler":           "models/scaler.pkl",

    # Outputs
    "alert_log":        "outputs/alert_log.csv",
    "eval_report":      "outputs/",

    # Images
    "img_class_dist":   "images/01_class_distribution.png",
    "img_attack_dist":  "images/02_attack_distribution.png",
    "img_corr":         "images/03_correlation_heatmap.png",
    "img_feat_imp":     "images/04_feature_importance.png",
    "img_conf_matrix_rf": "images/05_confusion_matrix_rf.png",
    "img_roc_rf":       "images/06_roc_curve_rf.png",
    "img_anomaly":      "images/07_anomaly_scores.png",
    "img_alert_dash":   "images/08_alert_dashboard.png",
    "img_feat_dist":    "images/09_feature_distributions.png",
    "img_model_compare": "images/10_model_comparison.png",
}

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "n_samples":          50000,   # Number of synthetic samples to generate
    "top_features":       20,      # Top N features to use for training
    "test_size":          0.25,    # 25% for testing
    "contamination":      0.20,    # Expected attack fraction for Isolation Forest
    "rf_n_estimators":    100,     # Number of trees in Random Forest
    "use_real_dataset":   False,   # Set True if you have a real CICIDS/NSL-KDD CSV
    "real_dataset_path":  "data/raw/",  # Path to real CSV if above is True
}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline():
    start_time = time.time()

    print("\n" + "█" * 65)
    print("█  AI-POWERED CYBERSECURITY THREAT DETECTION SYSTEM         █")
    print("█  Version 1.0.0 | Dual-Model Detection (RF + IF)           █")
    print("█" * 65 + "\n")

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 1: Create directories
    # ──────────────────────────────────────────────────────────────────────────
    print("=" * 65)
    print("  PHASE 1: ENVIRONMENT SETUP")
    print("=" * 65)
    for folder in ["data/raw", "data/processed", "models", "outputs", "images", "docs", "notebooks"]:
        os.makedirs(folder, exist_ok=True)
    print("[SETUP] All directories ready ✓")

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 2: Dataset Generation / Loading
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  PHASE 2: DATASET LOADING")
    print("=" * 65)

    if CONFIG["use_real_dataset"]:
        # Load real CICIDS-2017 or NSL-KDD dataset
        from src.preprocessor import preprocess_cicids
        csv_files = [f for f in os.listdir(CONFIG["real_dataset_path"]) if f.endswith(".csv")]
        if not csv_files:
            print("[WARN] No CSV found. Falling back to synthetic data.")
            df = generate_synthetic_dataset(CONFIG["n_samples"], PATHS["raw_data"])
        else:
            df = load_dataset(os.path.join(CONFIG["real_dataset_path"], csv_files[0]))
    else:
        # Use synthetic dataset (always works, no download needed)
        if os.path.exists(PATHS["raw_data"]):
            print(f"[DATA] Loading existing dataset: {PATHS['raw_data']}")
            df = pd.read_csv(PATHS["raw_data"])
        else:
            df = generate_synthetic_dataset(CONFIG["n_samples"], PATHS["raw_data"])

    # Keep a reference to the original df for attack_type column later
    df_original = df.copy()

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 3: Data Inspection (EDA)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  PHASE 3: DATA INSPECTION (EDA)")
    print("=" * 65)
    inspect_dataset(df)
    save_sample(df, PATHS["sample_data"])

    # Visualize class and attack distributions
    plot_class_distribution(df, save_path=PATHS["img_class_dist"])
    plot_attack_distribution(df, save_path=PATHS["img_attack_dist"])
    plot_feature_distributions(df,
                               features=["duration", "src_bytes", "dst_bytes", "count",
                                         "serror_rate", "same_srv_rate", "diff_srv_rate",
                                         "num_failed_logins", "hot"],
                               save_path=PATHS["img_feat_dist"])

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 4: Preprocessing
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  PHASE 4: DATA PREPROCESSING")
    print("=" * 65)
    data = preprocess(df, scaler_save_path=PATHS["scaler"])

    X_train      = data["X_train"]
    X_test       = data["X_test"]
    y_train      = data["y_train"]
    y_test       = data["y_test"]
    feature_names = data["feature_names"]
    scaler        = data["scaler"]

    # Save cleaned data
    cleaned_df = pd.DataFrame(
        np.vstack([X_train, X_test]), columns=feature_names
    )
    cleaned_df.to_csv(PATHS["processed_data"], index=False)
    print(f"[PREPROC] Cleaned data saved: {PATHS['processed_data']}")

    # Correlation heatmap (on original numeric df before scaling)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    plot_correlation_heatmap(df[numeric_cols], PATHS["img_corr"])

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 5: Feature Engineering
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  PHASE 5: FEATURE ENGINEERING")
    print("=" * 65)
    importance_df = get_feature_importances(
        X_train, y_train, feature_names,
        top_n=CONFIG["top_features"],
        plot_save_path=PATHS["img_feat_imp"]
    )

    X_train_sel, X_test_sel, selected_features = select_top_features(
        X_train, X_test, feature_names, importance_df,
        top_n=CONFIG["top_features"]
    )

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 6: Model Training
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  PHASE 6: MODEL TRAINING")
    print("=" * 65)

    # Train Random Forest (Supervised)
    rf_model = train_random_forest(
        X_train_sel, y_train,
        save_path=PATHS["rf_model"],
        n_estimators=CONFIG["rf_n_estimators"]
    )

    # Train Isolation Forest (Unsupervised)
    if_model = train_isolation_forest(
        X_train_sel,
        contamination=CONFIG["contamination"],
        save_path=PATHS["if_model"]
    )

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 7: Model Evaluation
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  PHASE 7: MODEL EVALUATION")
    print("=" * 65)

    # Evaluate Random Forest
    rf_metrics = evaluate_model(
        rf_model, X_test_sel, y_test,
        model_name="Random Forest",
        save_dir=PATHS["eval_report"]
    )

    # Confusion Matrix
    y_pred_rf = rf_model.predict(X_test_sel)
    plot_confusion_matrix(y_test, y_pred_rf, "Random Forest", save_path=PATHS["img_conf_matrix_rf"])

    # ROC Curve
    y_prob_rf = rf_model.predict_proba(X_test_sel)[:, 1]
    plot_roc_curve(y_test, y_prob_rf, "Random Forest", save_path=PATHS["img_roc_rf"])

    # Evaluate Isolation Forest (map its -1/+1 output to 0/1)
    if_raw_pred  = if_model.predict(X_test_sel)
    y_pred_if    = (if_raw_pred == -1).astype(int)
    if_metrics   = evaluate_model(
        type("DummyModel", (), {"predict": lambda self, X: (if_model.predict(X) == -1).astype(int)})(),
        X_test_sel, y_test,
        model_name="Isolation Forest",
        save_dir=PATHS["eval_report"]
    )

    # Anomaly score visualization
    anomaly_scores = get_anomaly_scores(if_model, X_test_sel)
    plot_anomaly_scores(anomaly_scores, y_test, save_path=PATHS["img_anomaly"])

    # Model comparison chart
    compare_models([rf_metrics, if_metrics], save_path=PATHS["img_model_compare"])

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 8: Threat Detection & Alert Generation
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  PHASE 8: THREAT DETECTION & ALERT GENERATION")
    print("=" * 65)

    # Set up the unified ThreatDetector
    # NOTE: X_test_sel is already scaled by the StandardScaler from preprocessing.
    #       We pass pre_scaled=True so ThreatDetector skips re-scaling.
    detector = ThreatDetector()
    detector.set_models(rf_model, if_model, scaler, feature_names=selected_features)

    # Run detection on test set (data is already scaled)
    detection_results = detector.predict(X_test_sel, mode="combined", pre_scaled=True)

    # Generate alert log using test subset of original df
    test_size = len(X_test_sel)
    df_test_original = df_original.tail(test_size).reset_index(drop=True)

    # Align lengths
    if len(df_test_original) != test_size:
        df_test_original = df_original.sample(test_size, random_state=42).reset_index(drop=True)

    y_final_pred = detection_results["final_prediction"].values
    y_final_prob = detection_results["rf_confidence"].values / 100.0  # back to 0-1

    alert_df = generate_alerts(
        y_pred=y_final_pred,
        y_prob=y_final_prob,
        original_df=df_test_original,
        attack_type_col="attack_type",
        save_path=PATHS["alert_log"]
    )
    print_alert_summary(alert_df)

    # Alert dashboard
    plot_alert_severity_dashboard(alert_df, save_path=PATHS["img_alert_dash"])

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 9: Demo — Predict on Individual Samples
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  PHASE 9: REAL-TIME DETECTION DEMO")
    print("=" * 65)

    print("\n  [DEMO] Testing with 3 sample network connections:\n")

    # Sample 1: Normal traffic
    sample_normal = X_test_sel[y_test == 0][0].reshape(1, -1)
    r1 = detector.predict(sample_normal, pre_scaled=True)
    print(f"  Sample 1 (should be NORMAL):")
    print(f"    RF Prediction: {'ATTACK' if r1['rf_prediction'][0] else 'NORMAL'} "
          f"| Confidence: {r1['rf_confidence'][0]}%")
    print(f"    IF Score: {r1['if_anomaly_score'][0]:.4f} | Final: {r1['threat_label'][0]}")

    # Sample 2: Attack traffic
    sample_attack = X_test_sel[y_test == 1][0].reshape(1, -1)
    r2 = detector.predict(sample_attack, pre_scaled=True)
    print(f"\n  Sample 2 (should be ATTACK):")
    print(f"    RF Prediction: {'ATTACK' if r2['rf_prediction'][0] else 'NORMAL'} "
          f"| Confidence: {r2['rf_confidence'][0]}%")
    print(f"    IF Score: {r2['if_anomaly_score'][0]:.4f} | Final: {r2['threat_label'][0]}")

    # Sample 3: Random from test set
    random_idx = np.random.randint(0, len(X_test_sel))
    sample_rand = X_test_sel[random_idx].reshape(1, -1)
    r3 = detector.predict(sample_rand, pre_scaled=True)
    print(f"\n  Sample 3 (random, true label = {'ATTACK' if y_test[random_idx] else 'NORMAL'}):")
    print(f"    RF Prediction: {'ATTACK' if r3['rf_prediction'][0] else 'NORMAL'} "
          f"| Confidence: {r3['rf_confidence'][0]}%")
    print(f"    IF Score: {r3['if_anomaly_score'][0]:.4f} | Final: {r3['threat_label'][0]}")

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 10: Final Summary
    # ──────────────────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print("\n" + "█" * 65)
    print("█  PIPELINE COMPLETE — FINAL SUMMARY                        █")
    print("█" * 65)
    print(f"\n  ✅  Total Runtime        : {elapsed:.1f} seconds")
    print(f"  ✅  Dataset Size         : {len(df):,} samples")
    print(f"  ✅  Features Used        : {len(selected_features)}")
    print(f"  ✅  RF Accuracy          : {rf_metrics['accuracy'] * 100:.2f}%")
    print(f"  ✅  RF F1-Score          : {rf_metrics['f1_score'] * 100:.2f}%")
    print(f"  ✅  RF ROC-AUC           : {rf_metrics['roc_auc']:.4f}")
    print(f"  ✅  Threats Detected     : {(y_final_pred == 1).sum():,}")
    print(f"  ✅  Alerts Generated     : {len(alert_df):,}")
    print(f"  ✅  Critical Alerts      : {(alert_df['severity'] == 'CRITICAL').sum() if len(alert_df) > 0 else 0}")
    print(f"\n  📁  Saved files:")
    for name, path in PATHS.items():
        full_path = os.path.abspath(path)
        if os.path.exists(full_path):
            size = os.path.getsize(full_path)
            print(f"      {path:<45} ({size / 1024:.1f} KB)")



    return {
        "rf_metrics":   rf_metrics,
        "if_metrics":   if_metrics,
        "alert_count":  len(alert_df),
        "detector":     detector,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = run_pipeline()
