"""
=============================================================================
FILE: src/threat_detector.py
PROJECT: AI-Powered Cybersecurity Threat Detection System
AUTHOR: Your Name

DESCRIPTION:
    The ThreatDetector class is the unified prediction interface.
    It combines Random Forest + Isolation Forest for dual-layer detection.

    USAGE:
        detector = ThreatDetector()
        detector.load_models(rf_path="models/random_forest.pkl",
                             if_path="models/isolation_forest.pkl",
                             scaler_path="models/scaler.pkl")

        # Predict on new samples
        results = detector.predict(X_new, feature_names)
        print(results)
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler


class ThreatDetector:
    """
    Unified threat detection interface combining two ML models.

    ARCHITECTURE:
        Input → Scaler → [Random Forest] → Supervised Prediction
                      → [Isolation Forest] → Anomaly Score
                      → Combined Decision → Final Threat Label
    """

    def __init__(self):
        self.rf_model:     RandomForestClassifier = None
        self.if_model:     IsolationForest        = None
        self.scaler:       StandardScaler         = None
        self.feature_names: list                  = None
        self._is_ready = False

    # ── Load Models ──────────────────────────────────────────────────────────
    def load_models(self, rf_path: str, if_path: str, scaler_path: str) -> None:
        """Load trained models and scaler from disk."""
        print("\n[THREAT DETECTOR] Loading models...")
        if not all(os.path.exists(p) for p in [rf_path, if_path, scaler_path]):
            raise FileNotFoundError(
                "One or more model files not found.\n"
                "Please run main.py first to train and save the models."
            )
        self.rf_model = joblib.load(rf_path)
        self.if_model = joblib.load(if_path)
        self.scaler   = joblib.load(scaler_path)
        self._is_ready = True
        print("[THREAT DETECTOR] All models loaded successfully ✓")

    # ── Set Models Directly ───────────────────────────────────────────────────
    def set_models(self, rf_model, if_model, scaler, feature_names: list = None) -> None:
        """Set models directly (after training, without saving/loading)."""
        self.rf_model      = rf_model
        self.if_model      = if_model
        self.scaler        = scaler
        self.feature_names = feature_names
        self._is_ready     = True

    # ── Predict ───────────────────────────────────────────────────────────────
    def predict(self, X: np.ndarray, mode: str = "combined", pre_scaled: bool = False) -> pd.DataFrame:
        """
        Run threat detection on input features.

        MODES:
            'rf'       → Use only Random Forest (supervised)
            'if'       → Use only Isolation Forest (anomaly detection)
            'combined' → Use both models; flag as threat if either detects an attack

        Args:
            X          : Feature matrix (numpy array)
            mode       : Detection mode ('rf', 'if', 'combined')
            pre_scaled : If True, skip internal StandardScaler (data already scaled)

        Returns:
            pd.DataFrame with columns:
                - rf_prediction      (0/1 from Random Forest)
                - rf_confidence      (probability score %)
                - if_anomaly_score   (Isolation Forest score)
                - if_prediction      (0=normal, 1=anomaly from IF)
                - final_prediction   (combined decision)
                - threat_label       (NORMAL or THREAT)
        """
        if not self._is_ready:
            raise RuntimeError("Models not loaded. Call load_models() or set_models() first.")

        # Scale input only if raw (unscaled) data is passed in
        X_scaled = X if pre_scaled else self.scaler.transform(X)

        results = pd.DataFrame(index=range(len(X)))

        # ── Random Forest ─────────────────────────────────────────────────────
        if mode in ["rf", "combined"]:
            rf_preds   = self.rf_model.predict(X_scaled)
            rf_probs   = self.rf_model.predict_proba(X_scaled)[:, 1]
            results["rf_prediction"] = rf_preds
            results["rf_confidence"] = np.round(rf_probs * 100, 2)
        else:
            results["rf_prediction"] = 0
            results["rf_confidence"] = 0.0

        # ── Isolation Forest ──────────────────────────────────────────────────
        if mode in ["if", "combined"]:
            if_scores   = self.if_model.decision_function(X_scaled)
            if_preds    = self.if_model.predict(X_scaled)   # +1=normal, -1=anomaly
            if_bin      = (if_preds == -1).astype(int)       # Convert: 1=anomaly
            results["if_anomaly_score"] = np.round(if_scores, 6)
            results["if_prediction"]    = if_bin
        else:
            results["if_anomaly_score"] = 0.0
            results["if_prediction"]    = 0

        # ── Combined Decision ─────────────────────────────────────────────────
        # Flag as THREAT if EITHER model detects an attack
        results["final_prediction"] = (
            (results["rf_prediction"] == 1) | (results["if_prediction"] == 1)
        ).astype(int)

        results["threat_label"] = results["final_prediction"].map(
            {0: "NORMAL", 1: "⚠ THREAT DETECTED"}
        )

        return results

    # ── Predict Single Sample ─────────────────────────────────────────────────
    def predict_single(self, feature_dict: dict) -> dict:
        """
        Predict on a single network flow sample.

        Args:
            feature_dict: Dict mapping feature_name → value

        Returns:
            Dict with prediction result
        """
        if not self._is_ready:
            raise RuntimeError("Models not loaded.")

        if self.feature_names:
            X = np.array([[feature_dict.get(f, 0) for f in self.feature_names]])
        else:
            X = np.array([list(feature_dict.values())])

        result_df = self.predict(X)
        row = result_df.iloc[0].to_dict()

        print(f"\n[THREAT DETECTOR] Single Sample Analysis:")
        print(f"  RF Prediction  : {'ATTACK' if row['rf_prediction'] == 1 else 'NORMAL'}")
        print(f"  RF Confidence  : {row['rf_confidence']}%")
        print(f"  IF Anomaly Score: {row['if_anomaly_score']:.4f}")
        print(f"  Final Verdict  : {row['threat_label']}")

        return row
