"""
=============================================================================
FILE: web/app.py
PROJECT: AI-Powered Cybersecurity Threat Detection System — Web Dashboard
DESCRIPTION:
    Flask web server that powers the SOC dashboard.
    Reads all pre-generated ML outputs and serves them as a live dashboard.

HOW TO RUN:
    python web/app.py
    Then open: http://localhost:5000
=============================================================================
"""

import os
import sys
import json
import random
import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, jsonify
import pandas as pd

app = Flask(__name__)

# ── Paths (relative to project root) ─────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALERT_CSV  = os.path.join(BASE_DIR, "outputs", "alert_log.csv")
RF_JSON    = os.path.join(BASE_DIR, "outputs", "Random_Forest_metrics.json")
IF_JSON    = os.path.join(BASE_DIR, "outputs", "Isolation_Forest_metrics.json")
IMAGES_DIR = os.path.join(BASE_DIR, "images")


# ── Helper: load alert log ────────────────────────────────────────────────────
def load_alerts():
    if os.path.exists(ALERT_CSV):
        df = pd.read_csv(ALERT_CSV)
        return df
    return pd.DataFrame()


# ── Helper: load metrics ───────────────────────────────────────────────────────
def load_metrics():
    rf = {}
    if_m = {}
    if os.path.exists(RF_JSON):
        with open(RF_JSON) as f:
            rf = json.load(f)
    if os.path.exists(IF_JSON):
        with open(IF_JSON) as f:
            if_m = json.load(f)
    return rf, if_m


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Main SOC dashboard page."""
    df = load_alerts()
    rf, if_m = load_metrics()

    stats = {
        "total_alerts":    len(df),
        "critical":        int((df["severity"] == "CRITICAL").sum()) if len(df) else 0,
        "high":            int((df["severity"] == "HIGH").sum())     if len(df) else 0,
        "medium":          int((df["severity"] == "MEDIUM").sum())   if len(df) else 0,
        "low":             int((df["severity"] == "LOW").sum())       if len(df) else 0,
        "unique_src_ips":  int(df["source_ip"].nunique())             if len(df) else 0,
        "avg_confidence":  round(df["confidence_score"].mean(), 1)    if len(df) else 0,
        "rf_accuracy":     round(rf.get("accuracy", 0) * 100, 2),
        "rf_f1":           round(rf.get("f1_score", 0) * 100, 2),
        "rf_auc":          round(rf.get("roc_auc", 0), 4),
        "if_accuracy":     round(if_m.get("accuracy", 0) * 100, 2),
        "if_f1":           round(if_m.get("f1_score", 0) * 100, 2),
    }
    return render_template("index.html", stats=stats)


@app.route("/api/alerts")
def api_alerts():
    """Return latest 100 alerts as JSON for live feed."""
    df = load_alerts()
    if len(df) == 0:
        return jsonify([])
    latest = df.head(100).fillna("")
    return jsonify(latest.to_dict(orient="records"))


@app.route("/api/attack-distribution")
def api_attack_distribution():
    """Attack type counts for the doughnut chart."""
    df = load_alerts()
    if len(df) == 0:
        return jsonify({})
    counts = df["raw_attack_type"].value_counts().to_dict()
    return jsonify(counts)


@app.route("/api/severity-distribution")
def api_severity_distribution():
    """Severity counts for chart."""
    df = load_alerts()
    if len(df) == 0:
        return jsonify({})
    counts = df["severity"].value_counts().to_dict()
    return jsonify(counts)


@app.route("/api/timeline")
def api_timeline():
    """Hourly alert counts for the timeline chart."""
    df = load_alerts()
    if len(df) == 0:
        return jsonify({"labels": [], "values": []})
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.strftime("%H:%M")
    hourly = df.groupby("hour").size().reset_index(name="count")
    return jsonify({
        "labels": hourly["hour"].tolist(),
        "values": hourly["count"].tolist()
    })


@app.route("/api/top-ips")
def api_top_ips():
    """Top 10 attacking source IPs."""
    df = load_alerts()
    if len(df) == 0:
        return jsonify({"ips": [], "counts": []})
    top = df["source_ip"].value_counts().head(10)
    return jsonify({"ips": top.index.tolist(), "counts": top.values.tolist()})


@app.route("/api/metrics")
def api_metrics():
    """Model metrics as JSON."""
    rf, if_m = load_metrics()
    return jsonify({"random_forest": rf, "isolation_forest": if_m})


@app.route("/api/live-feed")
def api_live_feed():
    """
    Simulates a real-time new threat alert arriving.
    In production this would come from a streaming source.
    """
    attack_types = [
        "Denial of Service", "Port Scan / Reconnaissance",
        "Brute Force / Credential Attack", "Data Exfiltration", "Botnet / C&C"
    ]
    severities = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    weights    = [0.4, 0.3, 0.2, 0.1]

    severity = random.choices(severities, weights=weights)[0]
    conf = {
        "CRITICAL": round(random.uniform(95, 100), 1),
        "HIGH":     round(random.uniform(85, 95),  1),
        "MEDIUM":   round(random.uniform(70, 85),  1),
        "LOW":      round(random.uniform(50, 70),  1),
    }[severity]

    alert = {
        "alert_id":       "".join(random.choices("0123456789ABCDEF", k=8)),
        "timestamp":      datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_ip":      f"192.168.{random.randint(1,10)}.{random.randint(1,254)}",
        "destination_ip": f"10.0.{random.randint(0,5)}.{random.randint(1,254)}",
        "port":           random.choice([22, 80, 443, 3306, 8080, 21, 25, 3389]),
        "protocol":       random.choice(["TCP", "UDP", "ICMP"]),
        "threat_type":    random.choice(attack_types),
        "confidence_score": conf,
        "severity":       severity,
        "action_required": {
            "CRITICAL": "Block source IP immediately. Escalate to Tier-2 SOC.",
            "HIGH":     "Isolate affected host. Capture packets for forensics.",
            "MEDIUM":   "Monitor closely. Apply rate limiting.",
            "LOW":      "Log for review. No immediate action required.",
        }[severity],
        "status": "OPEN",
    }
    return jsonify(alert)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  AI Cybersecurity Threat Detection — Web Dashboard")
    print("  Open your browser at: http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
