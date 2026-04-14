"""
=============================================================================
FILE: src/feature_engineering.py
PROJECT: AI-Powered Cybersecurity Threat Detection System
AUTHOR: Your Name

DESCRIPTION:
    Feature Engineering improves model performance by:
    1. Removing redundant/highly correlated features
    2. Selecting the most important features using Random Forest
    3. Providing correlation analysis tools

WHY THIS IS IMPORTANT:
    - Too many features → model overfits, slower training
    - Correlated features → redundant info, wastes computation
    - Best features → fewer dimensions, better accuracy, faster prediction

    This is one of the highest-impact steps in an ML pipeline.
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ─────────────────────────────────────────────
# SECTION 1: Get Feature Importances
# ─────────────────────────────────────────────
def get_feature_importances(X_train: np.ndarray,
                            y_train: np.ndarray,
                            feature_names: list,
                            top_n: int = 20,
                            plot_save_path: str = None) -> pd.DataFrame:
    """
    Calculate and rank feature importances using a Random Forest.

    HOW IT WORKS:
        Random Forest internally tracks how much each feature
        reduces impurity (Gini Index) across all decision trees.
        Features that split the data most effectively get higher scores.

    Args:
        X_train       : Training feature matrix
        y_train       : Training labels
        feature_names : List of feature column names
        top_n         : Show top N features
        plot_save_path: Save importance bar chart to this path

    Returns:
        importance_df (pd.DataFrame): Features ranked by importance
    """
    print(f"\n[FEATURE ENG] Calculating feature importances (top {top_n})...")

    # Quick RF just for feature importance (not the main model)
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    importances = rf.feature_importances_
    importance_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print(f"\n  Top {min(top_n, len(importance_df))} features:")
    print(importance_df.head(top_n).to_string(index=False))

    if plot_save_path:
        _plot_feature_importance(importance_df, top_n, plot_save_path)

    return importance_df


# ─────────────────────────────────────────────
# SECTION 2: Select Top Features
# ─────────────────────────────────────────────
def select_top_features(X_train: np.ndarray,
                        X_test: np.ndarray,
                        feature_names: list,
                        importance_df: pd.DataFrame,
                        top_n: int = 20) -> tuple:
    """
    Reduce X_train and X_test to only the top N most important features.

    WHY: Using fewer, more informative features:
        - Reduces overfitting
        - Speeds up training
        - Often improves accuracy (by removing noise)

    Args:
        X_train       : Full training feature matrix
        X_test        : Full test feature matrix
        feature_names : Full list of feature names
        importance_df : DataFrame from get_feature_importances()
        top_n         : Number of top features to keep

    Returns:
        (X_train_selected, X_test_selected, selected_feature_names)
    """
    top_features = importance_df["feature"].head(top_n).tolist()
    feature_indices = [feature_names.index(f) for f in top_features if f in feature_names]

    X_train_sel = X_train[:, feature_indices]
    X_test_sel  = X_test[:,  feature_indices]

    print(f"\n[FEATURE ENG] Selected {len(feature_indices)} features out of {len(feature_names)}")
    print(f"[FEATURE ENG] New X_train shape: {X_train_sel.shape}")
    return X_train_sel, X_test_sel, top_features


# ─────────────────────────────────────────────
# SECTION 3: Remove Highly Correlated Features
# ─────────────────────────────────────────────
def remove_correlated_features(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """
    Remove features that are highly correlated with each other.

    WHY: If Feature A and Feature B are 99% correlated, they carry the
         same information. Keeping both wastes resources and can confuse
         the model (multicollinearity).

    Args:
        df        : Input dataframe (numeric features only)
        threshold : Correlation threshold above which to remove a feature

    Returns:
        Reduced dataframe with highly correlated features removed
    """
    print(f"\n[FEATURE ENG] Removing features with correlation > {threshold}...")
    corr_matrix = df.corr().abs()

    # Upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation above threshold
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    print(f"[FEATURE ENG] Dropping {len(to_drop)} highly correlated features: {to_drop[:10]}...")
    return df.drop(columns=to_drop)


# ─────────────────────────────────────────────
# SECTION 4: Plot Feature Importance
# ─────────────────────────────────────────────
def _plot_feature_importance(importance_df: pd.DataFrame, top_n: int, save_path: str) -> None:
    """Internal helper: Create and save a feature importance horizontal bar chart."""
    top = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top)))
    bars = ax.barh(range(len(top)), top["importance"].values, color=colors[::-1], edgecolor="white")

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance Score", fontsize=11)
    ax.set_title(f"Top {top_n} Most Important Features\n(AI Cybersecurity Threat Detection)", 
                 fontsize=13, fontweight="bold", pad=15)

    # Annotate bars with values
    for i, bar in enumerate(bars):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{bar.get_width():.4f}', va='center', ha='left', fontsize=7, color='gray')

    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[FEATURE ENG] Feature importance chart saved: {save_path}")


# ─────────────────────────────────────────────
# SECTION 5: Correlation Heatmap
# ─────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame, save_path: str, top_n_cols: int = 20) -> None:
    """
    Generate a correlation heatmap for the top N numeric features.

    WHY: A heatmap visually shows which features are related to each other.
         Dark red = strongly positively correlated
         Dark blue = strongly negatively correlated
         White = no correlation

    Args:
        df          : DataFrame with numeric features
        save_path   : Where to save the heatmap PNG
        top_n_cols  : Limit to this many columns for readability
    """
    print(f"\n[FEATURE ENG] Generating correlation heatmap...")
    numeric_df = df.select_dtypes(include=[np.number]).iloc[:, :top_n_cols]

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(numeric_df.corr(), dtype=bool))  # Hide upper triangle
    sns.heatmap(
        numeric_df.corr(),
        mask=mask,
        annot=False,
        fmt=".1f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Feature Correlation Heatmap\n(AI Cybersecurity Threat Detection)",
                 fontsize=13, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[FEATURE ENG] Correlation heatmap saved: {save_path}")
