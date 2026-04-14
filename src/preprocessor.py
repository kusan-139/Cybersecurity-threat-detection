"""
=============================================================================
FILE: src/preprocessor.py
PROJECT: AI-Powered Cybersecurity Threat Detection System
AUTHOR: Your Name

DESCRIPTION:
    This module handles all data preprocessing steps:
    - Removing null/missing values
    - Removing infinite values (common in CICIDS dataset)
    - Removing duplicate rows
    - Encoding categorical variables
    - Splitting into train/test sets
    - Feature scaling (StandardScaler)

WHY THIS IS IMPORTANT:
    ML models cannot handle:
      ✗ NaN (missing) values
      ✗ Infinite values
      ✗ String columns (only numbers allowed)
      ✗ Features with vastly different scales (e.g., 0-1 vs 0-1,000,000)
    
    This module fixes all of those issues.
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os


# ─────────────────────────────────────────────
# SECTION 1: Full Preprocessing Pipeline
# ─────────────────────────────────────────────
def preprocess(df: pd.DataFrame, scaler_save_path: str = None) -> dict:
    """
    Full preprocessing pipeline for cybersecurity dataset.

    Steps:
        1. Drop irrelevant columns (labels/identifiers)
        2. Clean data (nulls, inf, duplicates)
        3. Encode remaining categoricals
        4. Split into X (features) and y (labels)
        5. Train/test split
        6. Scale features

    Args:
        df              (pd.DataFrame): Raw input dataframe
        scaler_save_path (str)        : Path to save the fitted StandardScaler

    Returns:
        dict: {
            'X_train', 'X_test', 'y_train', 'y_test',
            'feature_names', 'scaler',
            'X_train_raw', 'X_test_raw'   ← unscaled versions
        }
    """
    print("\n[PREPROCESSOR] Starting preprocessing pipeline...")
    df = df.copy()

    # ── Step 1: Separate target label ────────────────────────────────────────
    if "label" not in df.columns:
        raise ValueError("Dataset must have a 'label' column (0=Normal, 1=Attack)")

    y = df["label"].values.astype(int)

    # ── Step 2: Drop non-feature columns ─────────────────────────────────────
    # These columns are metadata, not features for the model
    cols_to_drop = ["label", "attack_type"]
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    print(f"[PREPROCESSOR] Dropped label/meta columns. Remaining: {df.shape[1]} features")

    # ── Step 3: Handle infinite values ───────────────────────────────────────
    # CICIDS-2017 is known for having Inf values in flow rate columns
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    if inf_count > 0:
        print(f"[PREPROCESSOR] Found {inf_count} infinite values → replacing with NaN")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # ── Step 4: Handle missing (NaN) values ──────────────────────────────────
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        print(f"[PREPROCESSOR] Found {null_count} null values → filling with column median")
        # Fill numeric columns with median (robust to outliers)
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].fillna(df[col].median())

    # ── Step 5: Handle remaining string/categorical columns ──────────────────
    # Convert any leftover string columns using label encoding
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        print(f"[PREPROCESSOR] Encoding {len(cat_cols)} categorical column(s): {cat_cols}")
        for col in cat_cols:
            df[col] = pd.Categorical(df[col]).codes

    # ── Step 6: Remove duplicate rows ────────────────────────────────────────
    before = len(df)
    dup_mask = ~df.duplicated()                # True = keep, False = duplicate
    df = df[dup_mask].reset_index(drop=True)
    y  = y[dup_mask.values]                    # Keep same rows in y
    after = len(df)
    if before != after:
        print(f"[PREPROCESSOR] Removed {before - after} duplicate rows")

    # ── Step 7: Record feature names before converting to numpy ──────────────
    feature_names = df.columns.tolist()
    X = df.values.astype(np.float64)

    print(f"[PREPROCESSOR] Final feature matrix: {X.shape} | Labels: {y.shape}")
    print(f"[PREPROCESSOR] Attack count: {y.sum()} | Normal count: {(y == 0).sum()}")

    # ── Step 8: Train / Test Split ────────────────────────────────────────────
    # stratify=y ensures both classes appear in same ratio in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,     # 75% train, 25% test
        random_state=42,    # Reproducible split
        stratify=y          # Preserve class ratio
    )
    print(f"[PREPROCESSOR] Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    # ── Step 9: Feature Scaling ───────────────────────────────────────────────
    # WHY: Random Forest doesn't strictly need scaling, but Isolation Forest
    #      and many other algorithms do. It's good practice to always scale.
    # StandardScaler: (x - mean) / std_dev  →  z-score normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # Fit ONLY on train data!
    X_test_scaled  = scaler.transform(X_test)         # Apply same transform to test

    if scaler_save_path:
        os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
        joblib.dump(scaler, scaler_save_path)
        print(f"[PREPROCESSOR] Scaler saved to: {scaler_save_path}")

    print("[PREPROCESSOR] Preprocessing complete ✓\n")

    return {
        "X_train":      X_train_scaled,
        "X_test":       X_test_scaled,
        "y_train":      y_train,
        "y_test":       y_test,
        "feature_names": feature_names,
        "scaler":       scaler,
        "X_train_raw":  X_train,   # Unscaled — for Isolation Forest anomaly demo
        "X_test_raw":   X_test,
    }


# ─────────────────────────────────────────────
# SECTION 2: CICIDS-2017 Specific Preprocessing
# ─────────────────────────────────────────────
def preprocess_cicids(df: pd.DataFrame, scaler_save_path: str = None) -> dict:
    """
    Specialized preprocessing for the CICIDS-2017 dataset.

    CICIDS-specific issues:
        - Label column is called ' Label' (with a leading space)
        - Contains values like 'BENIGN', 'DoS Hulk', 'PortScan', etc.
        - Has many Inf values in Flow Bytes/s and Flow Packets/s columns
        - Column names have leading spaces

    Usage:
        df = load_dataset("data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
        result = preprocess_cicids(df)
    """
    print("\n[PREPROCESSOR] Running CICIDS-2017 specific preprocessing...")
    df = df.copy()

    # Strip column name spaces (CICIDS issue)
    df.columns = df.columns.str.strip()

    # CICIDS label column is 'Label'
    if "Label" not in df.columns:
        raise ValueError("Expected 'Label' column not found in CICIDS dataset")

    # Map BENIGN → 0, everything else → 1
    df["label"] = (df["Label"] != "BENIGN").astype(int)
    df["attack_type"] = df["Label"]
    df.drop(columns=["Label"], inplace=True)

    print(f"[PREPROCESSOR] CICIDS Labels mapped: {df['label'].value_counts().to_dict()}")
    return preprocess(df, scaler_save_path)
