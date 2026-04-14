"""
=============================================================================
FILE: src/visualizer.py
PROJECT: AI-Powered Cybersecurity Threat Detection System
AUTHOR: Your Name

DESCRIPTION:
    Creates professional security visualizations for GitHub portfolio.
    All charts are publication-quality and saved as PNGs in the images/ folder.

CHARTS GENERATED:
    1. Class Distribution (Normal vs Attack)
    2. Attack Type Distribution (bar chart)
    3. Anomaly Score Distribution (Isolation Forest output)
    4. Predicted vs Actual Side-by-Side
    5. Alert Severity Dashboard
    6. Feature Distribution Comparison (Normal vs Attack)
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


# ─────────────────────────────────────────────────────────────────────────────
# Global Style Configuration
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":       120,
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  10,
    "axes.spines.top":  False,
    "axes.spines.right": False,
})
PALETTE = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]


# ─────────────────────────────────────────────
# CHART 1: Class Distribution Pie Chart
# ─────────────────────────────────────────────
def plot_class_distribution(df: pd.DataFrame,
                            label_col: str = "label",
                            save_path: str = None) -> None:
    """
    Plot a pie chart showing the ratio of Normal vs Attack traffic.
    WHY: Shows class imbalance — important for understanding model bias.
    """
    counts = df[label_col].value_counts().sort_index()
    labels = ["Normal Traffic", "Attack Traffic"]
    colors = ["#4CAF50", "#F44336"]
    explode = (0, 0.1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Pie chart
    wedges, texts, autotexts = axes[0].pie(
        counts.values, labels=labels, autopct="%1.1f%%",
        colors=colors, explode=explode, startangle=90,
        textprops={"fontsize": 11}, pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 2}
    )
    for at in autotexts:
        at.set_fontweight("bold")
        at.set_color("white")
    axes[0].set_title("Network Traffic Class Distribution", fontweight="bold", pad=15)

    # Bar chart
    bars = axes[1].bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.5)
    for bar, count in zip(bars, counts.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                     f"{count:,}", ha="center", va="bottom", fontweight="bold")
    axes[1].set_title("Sample Count — Normal vs Attack", fontweight="bold", pad=15)
    axes[1].set_ylabel("Number of Records")
    axes[1].grid(axis="y", alpha=0.3, linestyle="--")

    plt.suptitle("AI-Powered Cybersecurity Threat Detection\nDataset Class Distribution",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save_or_show(save_path, "Class distribution chart")


# ─────────────────────────────────────────────
# CHART 2: Attack Type Bar Chart
# ─────────────────────────────────────────────
def plot_attack_distribution(df: pd.DataFrame,
                             attack_col: str = "attack_type",
                             save_path: str = None) -> None:
    """
    Plot distribution of different attack types in the dataset.
    WHY: Understanding which attacks dominate helps set detection priorities.
    """
    if attack_col not in df.columns:
        print(f"[VISUALIZER] Column '{attack_col}' not found, skipping.")
        return

    counts = df[attack_col].value_counts().head(15)

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(counts))]
    bars = ax.barh(range(len(counts)), counts.values, color=colors, edgecolor="white", height=0.7)

    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(counts.index.tolist(), fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Records")
    ax.set_title("Attack Type Distribution\nAI Cybersecurity Threat Detection",
                 fontweight="bold", pad=15)

    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
                f"{count:,}", va="center", fontsize=9)

    ax.grid(axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout()
    _save_or_show(save_path, "Attack distribution chart")


# ─────────────────────────────────────────────
# CHART 3: Anomaly Score Distribution
# ─────────────────────────────────────────────
def plot_anomaly_scores(anomaly_scores: np.ndarray,
                        y_true: np.ndarray,
                        save_path: str = None) -> None:
    """
    Plot Isolation Forest anomaly scores for Normal vs Attack samples.

    WHY: Shows how well the Isolation Forest separates normal from attack.
         Good separation = overlapping distributions are minimal.
         Scores < threshold → flagged as anomaly (attack).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    normal_scores = anomaly_scores[y_true == 0]
    attack_scores = anomaly_scores[y_true == 1]

    # Histogram overlap
    axes[0].hist(normal_scores, bins=60, color="#4CAF50", alpha=0.6, label="Normal Traffic")
    axes[0].hist(attack_scores, bins=60, color="#F44336", alpha=0.6, label="Attack Traffic")
    axes[0].axvline(x=0, color="black", linestyle="--", lw=1.5, label="Decision Threshold (0)")
    axes[0].set_xlabel("Anomaly Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Isolation Forest Anomaly Score\nDistribution by Class", fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.3, linestyle="--")

    # KDE (Kernel Density Estimate)
    from scipy import stats
    x_range = np.linspace(anomaly_scores.min(), anomaly_scores.max(), 300)
    kde_normal = stats.gaussian_kde(normal_scores)
    kde_attack = stats.gaussian_kde(attack_scores)
    axes[1].plot(x_range, kde_normal(x_range), color="#4CAF50", lw=2.5, label="Normal")
    axes[1].plot(x_range, kde_attack(x_range), color="#F44336", lw=2.5, label="Attack")
    axes[1].fill_between(x_range, kde_normal(x_range), alpha=0.1, color="#4CAF50")
    axes[1].fill_between(x_range, kde_attack(x_range), alpha=0.1, color="#F44336")
    axes[1].axvline(x=0, color="black", linestyle="--", lw=1.5, label="Threshold")
    axes[1].set_title("KDE — Anomaly Score Density\nNormal vs Attack", fontweight="bold")
    axes[1].set_xlabel("Anomaly Score")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    axes[1].grid(alpha=0.3, linestyle="--")

    plt.suptitle("Isolation Forest Anomaly Detection\nAI-Powered Cybersecurity Threat Detection",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save_or_show(save_path, "Anomaly score chart")


# ─────────────────────────────────────────────
# CHART 4: Alert Severity Dashboard
# ─────────────────────────────────────────────
def plot_alert_severity_dashboard(alert_df: pd.DataFrame, save_path: str = None) -> None:
    """
    Create a comprehensive alert dashboard visualization.
    Includes severity distribution, threat types, and confidence scores.
    """
    if alert_df is None or len(alert_df) == 0:
        print("[VISUALIZER] No alerts to plot.")
        return

    fig = plt.figure(figsize=(15, 10))
    fig.patch.set_facecolor("#0D1117")

    sev_colors = {
        "CRITICAL": "#FF1744",
        "HIGH":     "#FF6D00",
        "MEDIUM":   "#FFAB00",
        "LOW":      "#00E676"
    }

    # ── Plot 1: Severity Donut Chart ─────────────────────────────────────────
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_facecolor("#161B22")
    sev_counts = alert_df["severity"].value_counts().reindex(
        ["CRITICAL", "HIGH", "MEDIUM", "LOW"], fill_value=0
    )
    colors_list = [sev_colors[s] for s in sev_counts.index]
    wedges, texts, autotexts = ax1.pie(
        sev_counts.values, labels=sev_counts.index, autopct="%1.0f%%",
        colors=colors_list, startangle=90,
        wedgeprops={"edgecolor": "#0D1117", "linewidth": 2.5, "width": 0.55},
        textprops={"color": "white", "fontsize": 9}
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
    ax1.set_title("Alert Severity", color="white", fontweight="bold", pad=10)

    # ── Plot 2: Severity Bar Chart ────────────────────────────────────────────
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set_facecolor("#161B22")
    bars = ax2.bar(sev_counts.index, sev_counts.values, color=colors_list, edgecolor="#0D1117", linewidth=1.5)
    for bar, count in zip(bars, sev_counts.values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(count), ha="center", va="bottom", color="white", fontweight="bold")
    ax2.set_title("Alerts by Severity Level", color="white", fontweight="bold", pad=10)
    ax2.tick_params(colors="white")
    ax2.grid(axis="y", alpha=0.2, color="white", linestyle="--")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#30363D")

    # ── Plot 3: Threat Types ──────────────────────────────────────────────────
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_facecolor("#161B22")
    if "raw_attack_type" in alert_df.columns:
        type_counts = alert_df["raw_attack_type"].value_counts().head(8)
        colors_t = PALETTE[:len(type_counts)]
        bars3 = ax3.barh(range(len(type_counts)), type_counts.values, color=colors_t, edgecolor="#0D1117")
        ax3.set_yticks(range(len(type_counts)))
        ax3.set_yticklabels(type_counts.index, color="white", fontsize=8)
        ax3.set_title("Detected Threat Types", color="white", fontweight="bold", pad=10)
        ax3.tick_params(colors="white")
        ax3.grid(axis="x", alpha=0.2, color="white", linestyle="--")
        for spine in ax3.spines.values():
            spine.set_edgecolor("#30363D")
        ax3.invert_yaxis()

    # ── Plot 4: Confidence Score Distribution ────────────────────────────────
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_facecolor("#161B22")
    conf_by_sev = {s: alert_df[alert_df["severity"] == s]["confidence_score"]
                   for s in ["CRITICAL", "HIGH", "MEDIUM", "LOW"] if s in alert_df["severity"].values}
    for sev, data in conf_by_sev.items():
        ax4.hist(data, bins=20, alpha=0.7, label=sev, color=sev_colors[sev])
    ax4.set_xlabel("Confidence Score (%)", color="white")
    ax4.set_ylabel("Count", color="white")
    ax4.set_title("Confidence Score Distribution\nby Severity", color="white", fontweight="bold", pad=10)
    ax4.legend(facecolor="#161B22", edgecolor="#30363D", labelcolor="white")
    ax4.tick_params(colors="white")
    ax4.grid(alpha=0.2, color="white", linestyle="--")

    # ── Plot 5: Top Source IPs ────────────────────────────────────────────────
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_facecolor("#161B22")
    top_ips = alert_df["source_ip"].value_counts().head(8)
    ax5.barh(range(len(top_ips)), top_ips.values, color="#2196F3", edgecolor="#0D1117")
    ax5.set_yticks(range(len(top_ips)))
    ax5.set_yticklabels(top_ips.index, color="white", fontsize=8)
    ax5.set_title("Top Attacking Source IPs", color="white", fontweight="bold", pad=10)
    ax5.tick_params(colors="white")
    ax5.grid(axis="x", alpha=0.2, color="white", linestyle="--")
    ax5.invert_yaxis()

    # ── Plot 6: Timeline Summary Box ─────────────────────────────────────────
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_facecolor("#161B22")
    ax6.axis("off")
    stats_text = (
        f"📊  THREAT DETECTION SUMMARY\n"
        f"{'─' * 32}\n"
        f"Total Alerts   :  {len(alert_df):>6,}\n"
        f"CRITICAL        :  {(alert_df['severity'] == 'CRITICAL').sum():>6,}\n"
        f"HIGH            :  {(alert_df['severity'] == 'HIGH').sum():>6,}\n"
        f"MEDIUM          :  {(alert_df['severity'] == 'MEDIUM').sum():>6,}\n"
        f"LOW             :  {(alert_df['severity'] == 'LOW').sum():>6,}\n"
        f"{'─' * 32}\n"
        f"Avg Confidence  :  {alert_df['confidence_score'].mean():>5.1f}%\n"
        f"Max Confidence  :  {alert_df['confidence_score'].max():>5.1f}%\n"
        f"Unique Src IPs  :  {alert_df['source_ip'].nunique():>6,}\n"
        f"{'─' * 32}\n"
        f"Status: MONITORING ACTIVE 🟢"
    )
    ax6.text(0.5, 0.5, stats_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment="center", horizontalalignment="center",
             color="white", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.8", facecolor="#21262D", edgecolor="#30363D", linewidth=1.5))

    fig.suptitle("🔐  AI-Powered Cybersecurity Threat Detection — Security Operations Dashboard",
                 color="white", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    _save_or_show(save_path, "Alert severity dashboard")


# ─────────────────────────────────────────────
# CHART 5: Feature Distribution Comparison
# ─────────────────────────────────────────────
def plot_feature_distributions(df: pd.DataFrame,
                                label_col: str = "label",
                                features: list = None,
                                save_path: str = None) -> None:
    """
    Compare feature distributions between Normal and Attack traffic.
    WHY: Shows visually why certain features are good predictors.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != label_col]

    if features:
        plot_cols = [c for c in features if c in numeric_cols][:9]
    else:
        plot_cols = numeric_cols[:9]

    if not plot_cols:
        return

    n_cols = 3
    n_rows = (len(plot_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, col in enumerate(plot_cols):
        ax = axes[i]
        normal_vals  = df[df[label_col] == 0][col].dropna()
        attack_vals  = df[df[label_col] == 1][col].dropna()
        ax.hist(normal_vals, bins=40, alpha=0.6, color="#4CAF50", label="Normal", density=True)
        ax.hist(attack_vals, bins=40, alpha=0.6, color="#F44336", label="Attack", density=True)
        ax.set_title(col, fontweight="bold")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, linestyle="--")

    for j in range(len(plot_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Distribution: Normal vs Attack Traffic\nAI Cybersecurity Threat Detection",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save_or_show(save_path, "Feature distributions chart")


# ─────────────────────────────────────────────
# HELPER: Save or Show
# ─────────────────────────────────────────────
def _save_or_show(save_path: str, chart_name: str) -> None:
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=plt.gcf().get_facecolor())
        plt.close()
        print(f"[VISUALIZER] {chart_name} saved: {save_path}")
    else:
        plt.show()
