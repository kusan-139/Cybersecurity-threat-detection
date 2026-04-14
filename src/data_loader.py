"""
=============================================================================
FILE: src/data_loader.py
PROJECT: AI-Powered Cybersecurity Threat Detection System
AUTHOR: Your Name

DESCRIPTION:
    This module handles all dataset-related operations:
    - Generating a realistic synthetic dataset (for demo/offline use)
    - Loading CSV datasets from disk (CICIDS, NSL-KDD, etc.)
    - Basic sanity checks and dataset inspection

WHY THIS IS IMPORTANT:
    Without clean, representative data, the AI model cannot learn correctly.
    This is Step 1 of any ML pipeline — "Garbage In, Garbage Out."
=============================================================================
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm


# ─────────────────────────────────────────────
# SECTION 1: Generate Synthetic Dataset
# ─────────────────────────────────────────────
def generate_synthetic_dataset(n_samples: int = 50000, save_path: str = None) -> pd.DataFrame:
    """
    Generates a realistic synthetic network traffic dataset for training.

    WHY: Real datasets like CICIDS-2017 are 2-3 GB. For classroom/GitHub demos,
         we generate a synthetic dataset that mirrors real network traffic features.

    FEATURES GENERATED (Based on CICIDS-2017 feature set):
        - duration           : Length of connection in seconds
        - protocol_type      : TCP=0, UDP=1, ICMP=2
        - src_bytes          : Bytes sent from source
        - dst_bytes          : Bytes sent from destination
        - land               : 1 if src/dst host+port are same (flag for loopback attack)
        - wrong_fragment     : Number of wrong fragments
        - urgent             : Number of urgent packets
        - hot                : Number of "hot" indicators
        - num_failed_logins  : Failed login attempts
        - logged_in          : 1 if successfully logged in
        - num_compromised    : Number of compromised conditions
        - root_shell         : 1 if root shell obtained
        - su_attempted       : 1 if su root command attempted
        - num_root           : Number of root accesses
        - num_file_creations : Number of file creation operations
        - num_shells         : Number of shell prompts
        - num_access_files   : Number of operations on access control files
        - num_outbound_cmds  : Number of outbound commands
        - is_host_login      : 1 if login is to host accounts
        - is_guest_login     : 1 if login is guest
        - count              : Number of connections to the same host in 2 seconds
        - srv_count          : Number of connections to the same service
        - serror_rate        : % of connections with SYN errors
        - srv_serror_rate    : % of same-service connections with SYN errors
        - rerror_rate        : % of connections with REJ errors
        - srv_rerror_rate    : % of same-service connections with REJ errors
        - same_srv_rate      : % of connections to same service
        - diff_srv_rate      : % of connections to different services
        - label              : 0 = NORMAL / 1 = ATTACK

    ATTACK TYPES SIMULATED:
        - DoS (Denial of Service)
        - Port Scan
        - Brute Force
        - Data Exfiltration
        - Botnet
    """
    print(f"\n[DATA LOADER] Generating synthetic dataset with {n_samples} samples...")

    np.random.seed(42)  # Reproducibility

    # ── Normal Traffic (70% of dataset) ──────────────────────────────────────
    n_normal = int(n_samples * 0.70)
    normal = {
        "duration":           np.random.exponential(scale=2.0,  size=n_normal).clip(0, 100),
        "protocol_type":      np.random.choice([0, 1, 2], size=n_normal, p=[0.6, 0.3, 0.1]),
        "src_bytes":          np.random.exponential(scale=5000,  size=n_normal).clip(0, 100000),
        "dst_bytes":          np.random.exponential(scale=3000,  size=n_normal).clip(0, 80000),
        "land":               np.zeros(n_normal, dtype=int),
        "wrong_fragment":     np.zeros(n_normal, dtype=int),
        "urgent":             np.zeros(n_normal, dtype=int),
        "hot":                np.random.poisson(lam=1, size=n_normal),
        "num_failed_logins":  np.zeros(n_normal, dtype=int),
        "logged_in":          np.ones(n_normal, dtype=int),
        "num_compromised":    np.zeros(n_normal, dtype=int),
        "root_shell":         np.zeros(n_normal, dtype=int),
        "su_attempted":       np.zeros(n_normal, dtype=int),
        "num_root":           np.zeros(n_normal, dtype=int),
        "num_file_creations": np.random.poisson(lam=0.5, size=n_normal),
        "num_shells":         np.zeros(n_normal, dtype=int),
        "num_access_files":   np.random.poisson(lam=0.2, size=n_normal),
        "num_outbound_cmds":  np.zeros(n_normal, dtype=int),
        "is_host_login":      np.zeros(n_normal, dtype=int),
        "is_guest_login":     np.zeros(n_normal, dtype=int),
        "count":              np.random.randint(1, 10,  size=n_normal),
        "srv_count":          np.random.randint(1, 10,  size=n_normal),
        "serror_rate":        np.random.uniform(0, 0.1, size=n_normal),
        "srv_serror_rate":    np.random.uniform(0, 0.1, size=n_normal),
        "rerror_rate":        np.random.uniform(0, 0.1, size=n_normal),
        "srv_rerror_rate":    np.random.uniform(0, 0.1, size=n_normal),
        "same_srv_rate":      np.random.uniform(0.7, 1.0, size=n_normal),
        "diff_srv_rate":      np.random.uniform(0, 0.3,   size=n_normal),
        "attack_type":        ["NORMAL"] * n_normal,
        "label":              np.zeros(n_normal, dtype=int),
    }

    # ── Attack Traffic (30% of dataset) ──────────────────────────────────────
    attack_types = ["DoS", "PortScan", "BruteForce", "DataExfil", "Botnet"]
    attack_probs = [0.40,  0.25,       0.15,         0.10,        0.10]
    n_attack = n_samples - n_normal
    chosen_attacks = np.random.choice(attack_types, size=n_attack, p=attack_probs)

    attacks = {
        "duration":           np.random.exponential(scale=0.5,   size=n_attack).clip(0, 50),
        "protocol_type":      np.random.choice([0, 1, 2], size=n_attack, p=[0.7, 0.2, 0.1]),
        "src_bytes":          np.random.exponential(scale=50000,  size=n_attack).clip(0, 500000),
        "dst_bytes":          np.random.exponential(scale=100,    size=n_attack).clip(0, 5000),
        "land":               np.random.choice([0, 1], size=n_attack, p=[0.85, 0.15]),
        "wrong_fragment":     np.random.poisson(lam=2, size=n_attack),
        "urgent":             np.random.poisson(lam=1, size=n_attack),
        "hot":                np.random.poisson(lam=5, size=n_attack),
        "num_failed_logins":  np.random.poisson(lam=3, size=n_attack),
        "logged_in":          np.random.choice([0, 1], size=n_attack, p=[0.8, 0.2]),
        "num_compromised":    np.random.poisson(lam=2, size=n_attack),
        "root_shell":         np.random.choice([0, 1], size=n_attack, p=[0.7, 0.3]),
        "su_attempted":       np.random.choice([0, 1], size=n_attack, p=[0.6, 0.4]),
        "num_root":           np.random.poisson(lam=1, size=n_attack),
        "num_file_creations": np.random.poisson(lam=2, size=n_attack),
        "num_shells":         np.random.poisson(lam=1, size=n_attack),
        "num_access_files":   np.random.poisson(lam=2, size=n_attack),
        "num_outbound_cmds":  np.random.poisson(lam=1, size=n_attack),
        "is_host_login":      np.random.choice([0, 1], size=n_attack, p=[0.5, 0.5]),
        "is_guest_login":     np.random.choice([0, 1], size=n_attack, p=[0.4, 0.6]),
        "count":              np.random.randint(50, 512, size=n_attack),
        "srv_count":          np.random.randint(50, 512, size=n_attack),
        "serror_rate":        np.random.uniform(0.5, 1.0, size=n_attack),
        "srv_serror_rate":    np.random.uniform(0.5, 1.0, size=n_attack),
        "rerror_rate":        np.random.uniform(0.4, 1.0, size=n_attack),
        "srv_rerror_rate":    np.random.uniform(0.4, 1.0, size=n_attack),
        "same_srv_rate":      np.random.uniform(0, 0.3,   size=n_attack),
        "diff_srv_rate":      np.random.uniform(0.7, 1.0, size=n_attack),
        "attack_type":        chosen_attacks.tolist(),
        "label":              np.ones(n_attack, dtype=int),
    }

    # ── Combine and Shuffle ───────────────────────────────────────────────────
    df_normal  = pd.DataFrame(normal)
    df_attacks = pd.DataFrame(attacks)
    df = pd.concat([df_normal, df_attacks], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

    print(f"[DATA LOADER] Dataset generated: {len(df)} rows | {df['label'].sum()} attacks | "
          f"{(df['label'] == 0).sum()} normal")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"[DATA LOADER] Saved to: {save_path}")

    return df


# ─────────────────────────────────────────────
# SECTION 2: Load Dataset from CSV File
# ─────────────────────────────────────────────
def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load a CSV dataset from disk.

    Args:
        filepath (str): Path to the CSV file (e.g., CICIDS-2017, NSL-KDD)

    Returns:
        pd.DataFrame: Loaded dataset

    Common CICIDS-2017 issues handled:
        - Column names with leading/trailing spaces
        - Mixed type columns
        - Very large files (read in chunks if needed)
    """
    print(f"\n[DATA LOADER] Loading dataset from: {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}\n"
                                f"Please download the dataset and place it in 'data/raw/'")

    # Strip leading/trailing spaces from column names (common CICIDS issue)
    df = pd.read_csv(filepath, low_memory=False)
    df.columns = df.columns.str.strip()

    print(f"[DATA LOADER] Loaded {len(df)} rows × {len(df.columns)} columns")
    return df


# ─────────────────────────────────────────────
# SECTION 3: Dataset Inspection
# ─────────────────────────────────────────────
def inspect_dataset(df: pd.DataFrame) -> None:
    """
    Print a detailed summary of the dataset for initial understanding.

    WHY: Before doing anything with data, you must understand:
         - How many rows and columns?
         - Are there missing values?
         - What are the data types?
         - What does the class distribution look like?
    """
    print("\n" + "=" * 60)
    print("  DATASET INSPECTION REPORT")
    print("=" * 60)
    print(f"\n  Shape        : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Memory Usage : {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\n  First 3 rows:\n{df.head(3)}")
    print(f"\n  Data Types:\n{df.dtypes.value_counts()}")
    print(f"\n  Null Values (top 10):\n{df.isnull().sum().sort_values(ascending=False).head(10)}")

    if "label" in df.columns:
        print(f"\n  Class Distribution:")
        print(df["label"].value_counts())
        pct = df["label"].value_counts(normalize=True) * 100
        print(f"\n  Attack %  : {pct.get(1, 0):.2f}%")
        print(f"  Normal %  : {pct.get(0, 0):.2f}%")

    if "attack_type" in df.columns:
        print(f"\n  Attack Types:\n{df['attack_type'].value_counts()}")

    print("=" * 60 + "\n")


# ─────────────────────────────────────────────
# SECTION 4: Save Sample Data for GitHub Demo
# ─────────────────────────────────────────────
def save_sample(df: pd.DataFrame, save_path: str, n: int = 5000) -> None:
    """
    Save a small sample of the dataset for GitHub demo purposes.

    WHY: Real datasets are 2-3 GB and can't be pushed to GitHub.
         A 5000-row sample lets reviewers run the project instantly.

    Args:
        df       : Full dataframe
        save_path: Path to save the sample CSV
        n        : Number of rows to include (default 5000)
    """
    sample = df.sample(n=min(n, len(df)), random_state=42)
    sample.to_csv(save_path, index=False)
    print(f"[DATA LOADER] Sample ({n} rows) saved to: {save_path}")
