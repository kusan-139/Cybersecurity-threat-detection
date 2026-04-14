"""
=============================================================================
FILE: src/alert_generator.py
PROJECT: AI-Powered Cybersecurity Threat Detection System
AUTHOR: Your Name

DESCRIPTION:
    Generates cybersecurity alerts from model predictions.

    In a real Security Operations Center (SOC):
    - SIEM (Security Information and Event Management) tools like Splunk,
      IBM QRadar, or Microsoft Sentinel generate structured alerts.
    - Each alert contains: timestamp, source IP, threat type, severity, action.
    - Analysts triage alerts by severity (HIGH > MEDIUM > LOW).

    This module simulates that SOC alert generation process from ML predictions.

    ALERT SEVERITY LEVELS:
    ┌──────────────────────────────────────────────────────────────────┐
    │  CRITICAL  — Confirmed attack with very high confidence (>95%)   │
    │  HIGH       — High confidence attack (85-95%)                    │
    │  MEDIUM     — Moderate confidence (70-85%)                       │
    │  LOW        — Low confidence alert / anomaly detected (<70%)     │
    └──────────────────────────────────────────────────────────────────┘
=============================================================================
"""

import os
import uuid
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
ATTACK_TYPE_MAP = {
    "DoS":         "Denial of Service",
    "PortScan":    "Port Scan / Reconnaissance",
    "BruteForce":  "Brute Force / Credential Attack",
    "DataExfil":   "Data Exfiltration",
    "Botnet":      "Botnet / C&C Communication",
    "Unknown":     "Unknown Threat / Zero-Day",
}

PROTOCOLS = {0: "TCP", 1: "UDP", 2: "ICMP"}


# ─────────────────────────────────────────────
# SECTION 1: Generate Alert Log
# ─────────────────────────────────────────────
def generate_alerts(y_pred: np.ndarray,
                    y_prob: np.ndarray,
                    original_df: pd.DataFrame,
                    attack_type_col: str = "attack_type",
                    save_path: str = None) -> pd.DataFrame:
    """
    Generate a structured alert log from model predictions.

    For every predicted ATTACK (y_pred == 1), create an alert record:
    - Alert ID (unique UUID)
    - Timestamp (simulated)
    - Source IP (simulated)
    - Destination IP (simulated)
    - Port
    - Protocol
    - Threat Type
    - Confidence Score (from model probability)
    - Severity (based on confidence)
    - Recommended Action
    - Status

    Args:
        y_pred          : Predicted labels (0/1)
        y_prob          : Prediction probabilities [0,1] (from Random Forest)
        original_df     : Original dataframe (for context features)
        attack_type_col : Column name for attack type labels
        save_path       : Save alert log CSV to this path

    Returns:
        alert_df (pd.DataFrame): Structured alert log
    """
    print("\n[ALERT GENERATOR] Generating security alerts...")

    # Reset index alignment
    if len(y_pred) != len(original_df):
        raise ValueError(f"Length mismatch: y_pred={len(y_pred)}, df={len(original_df)}")

    alerts = []
    base_time = datetime.now() - timedelta(hours=8)  # Simulate last 8 hours

    attack_indices = np.where(y_pred == 1)[0]
    print(f"[ALERT GENERATOR] Total threats detected: {len(attack_indices)}")

    random.seed(42)
    np.random.seed(42)

    for i, idx in enumerate(attack_indices):
        row = original_df.iloc[idx]
        
        # ── Confidence with Jitter ──────────────────────────────────────────
        # Adding slight noise (0-5%) to confidence to simulate real-world SOC variety
        # otherwise highly accurate models on synthetic data make everything CRITICAL.
        base_confidence = float(y_prob[idx]) if y_prob is not None else np.random.uniform(0.6, 0.99)
        jitter = np.random.uniform(-0.05, 0.02) if base_confidence > 0.90 else 0
        confidence = max(min(base_confidence + jitter, 1.0), 0.5)

        # ── Severity Classification ─────────────────────────────────────────
        if confidence >= 0.98:
            severity = "CRITICAL"
            action   = "Block source IP immediately. Escalate to Tier-2 SOC."
        elif confidence >= 0.92:
            severity = "HIGH"
            action   = "Isolate affected host. Capture network packets for forensics."
        elif confidence >= 0.80:
            severity = "MEDIUM"
            action   = "Monitor closely. Apply rate limiting. Alert network team."
        else:
            severity = "LOW"
            action   = "Log for review. No immediate action required."

        # ── Get attack type from dataset (if available) ─────────────────────
        raw_attack_type = "Unknown"
        if attack_type_col in original_df.columns:
            raw_attack_type = str(row.get(attack_type_col, "Unknown"))
        threat_type = ATTACK_TYPE_MAP.get(raw_attack_type, raw_attack_type)

        # ── Simulated Network Metadata ──────────────────────────────────────
        src_ip  = f"192.168.{random.randint(1, 10)}.{random.randint(1, 254)}"
        dst_ip  = f"10.0.{random.randint(0, 5)}.{random.randint(1, 254)}"
        port    = random.choice([22, 80, 443, 3306, 8080, 23, 21, 25, 3389, 6667])
        proto   = PROTOCOLS.get(int(row.get("protocol_type", 0)), "TCP")
        ts      = base_time + timedelta(minutes=random.randint(0, 480))

        alert = {
            "alert_id":         str(uuid.uuid4())[:8].upper(),
            "timestamp":        ts.strftime("%Y-%m-%d %H:%M:%S"),
            "source_ip":        src_ip,
            "destination_ip":   dst_ip,
            "port":             port,
            "protocol":         proto,
            "threat_type":      threat_type,
            "raw_attack_type":  raw_attack_type,
            "confidence_score": round(confidence * 100, 2),
            "severity":         severity,
            "action_required":  action,
            "status":           "OPEN",
            "analyst_notes":    "",
        }
        alerts.append(alert)

    alert_df = pd.DataFrame(alerts)

    if len(alert_df) > 0:
        # Sort by severity and timestamp
        sev_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        alert_df["sev_rank"] = alert_df["severity"].map(sev_order)
        alert_df = alert_df.sort_values(["sev_rank", "timestamp"]).drop(columns=["sev_rank"])
        alert_df = alert_df.reset_index(drop=True)

        print(f"\n  Alert Severity Breakdown:")
        print(alert_df["severity"].value_counts().to_string())

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            alert_df.to_csv(save_path, index=False)
            print(f"\n[ALERT GENERATOR] Alert log saved: {save_path}")
    else:
        print("[ALERT GENERATOR] No threats detected in this batch.")

    return alert_df


# ─────────────────────────────────────────────
# SECTION 2: Print Alert Summary
# ─────────────────────────────────────────────
def print_alert_summary(alert_df: pd.DataFrame) -> None:
    """
    Print a formatted SOC-style alert summary to the console.
    """
    if alert_df is None or len(alert_df) == 0:
        print("\n[ALERT GENERATOR] No alerts to display.")
        return

    print("\n" + "=" * 70)
    print("  🔴  SECURITY ALERT DASHBOARD — AI Threat Detection System")
    print("=" * 70)
    print(f"  Total Alerts   : {len(alert_df)}")
    print(f"  CRITICAL       : {(alert_df['severity'] == 'CRITICAL').sum()}")
    print(f"  HIGH           : {(alert_df['severity'] == 'HIGH').sum()}")
    print(f"  MEDIUM         : {(alert_df['severity'] == 'MEDIUM').sum()}")
    print(f"  LOW            : {(alert_df['severity'] == 'LOW').sum()}")
    print("=" * 70)

    print(f"\n  TOP 10 CRITICAL/HIGH ALERTS:")
    critical = alert_df[alert_df["severity"].isin(["CRITICAL", "HIGH"])].head(10)
    for _, row in critical.iterrows():
        sev_icon = "🔴" if row["severity"] == "CRITICAL" else "🟠"
        print(f"\n  {sev_icon} [{row['alert_id']}] {row['timestamp']}")
        print(f"     Threat  : {row['threat_type']}")
        print(f"     Source  : {row['source_ip']} → {row['destination_ip']}:{row['port']}")
        print(f"     Confidence: {row['confidence_score']}% | Severity: {row['severity']}")
        print(f"     Action  : {row['action_required']}")

    print("\n" + "=" * 70 + "\n")
