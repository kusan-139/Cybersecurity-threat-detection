# 🔐 AI-Powered Cybersecurity Threat Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.1-150458?style=for-the-badge&logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

**A production-grade AI/ML system for detecting cybersecurity threats using dual-model architecture: Random Forest (supervised) + Isolation Forest (unsupervised anomaly detection).**

[📊 View Results](#-results) · [🚀 Quick Start](#-quick-start) · [🏗️ Architecture](#-architecture) · [📁 Project Structure](#-project-structure)

</div>

---

## 📌 What Is This Project?

This project simulates a **Security Operations Center (SOC)** threat detection system — the kind used by banks, IT companies, and government organizations to protect their networks in real time.

### 🧠 Simple Explanation
Imagine your company's network as a busy highway. Millions of data packets travel through it every second. This AI system acts as an **intelligent surveillance camera** — automatically detecting suspicious traffic without needing human analysts to check every packet.

### ⚙️ Technical Explanation
An end-to-end ML pipeline that:
1. Ingests **network flow records** (packet-level features)
2. **Preprocesses** data (cleans, normalizes, engineers features)
3. Trains **two complementary models**:
   - **Random Forest**: Supervised learner for known attack patterns
   - **Isolation Forest**: Unsupervised anomaly detector for unknown threats
4. Generates **structured SOC-style alerts** with severity levels
5. Produces **publication-quality visualizations** for analysis

---

## 🎯 Problems This Solves

| Problem | Solution |
|---------|----------|
| Manual monitoring impossible at scale | Automated AI detection |
| Alert fatigue for SOC analysts | Confidence-scored priority alerts |
| Zero-day / unknown attacks missed | Unsupervised anomaly detection |
| High false positive rate | Dual-model consensus reduces errors |
| No explainability | Feature importance shows *why* a threat was flagged |

---

## 🏢 Industry Context

| Industry | Use Case | This System's Equivalent |
|----------|----------|--------------------------|
| **Banks** | Fraud detection | Flagging abnormal transaction patterns |
| **IT Companies** | Intrusion Detection (IDS) | Random Forest on network flows |
| **E-Commerce** | Account takeover prevention | Brute force attack detection |
| **Hospitals** | Ransomware early warning | Anomaly detection on file operations |
| **Telecom** | DDoS mitigation | High-count/high-error-rate flagging |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT LAYER                                  │
│    Network Flow Dataset (CSV) — 50,000 samples                  │
│    28 features: duration, bytes, flags, error rates, etc.       │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│                 PREPROCESSING MODULE                            │
│  ✓ Remove nulls & duplicates                                    │
│  ✓ Handle infinite values (CICIDS-2017 issue)                   │
│  ✓ Encode categorical variables                                 │
│  ✓ StandardScaler normalization                                 │
│  ✓ Stratified 75/25 train/test split                            │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│              FEATURE ENGINEERING MODULE                         │
│  ✓ Correlation heatmap analysis                                 │
│  ✓ Feature importance via Random Forest                         │
│  ✓ Select top 20 most predictive features                       │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────────────┐
│                   MODEL TRAINING LAYER                           │
│                                                                  │
│  ┌─────────────────────┐    ┌──────────────────────────────────┐ │
│  │   Random Forest     │    │     Isolation Forest             │ │
│  │  (Supervised)       │    │    (Unsupervised/Anomaly)        │ │
│  │  100 trees          │    │  contamination=0.20              │ │
│  │  class_weight=auto  │    │  Detects zero-day threats        │ │
│  └─────────────────────┘    └──────────────────────────────────┘ │
└──────────────────────┬───────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│              PREDICTION & ALERT ENGINE                          │
│  Combined: threat if RF OR IF predicts attack                   │
│  CRITICAL (>95%) → HIGH (>85%) → MEDIUM (>70%) → LOW            │
│  Output: alert_log.csv with timestamp, IP, severity, action     │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│              VISUALIZATION DASHBOARD                            │
│  10 professional charts saved as PNG                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
AI-Cybersecurity-Threat-Detection/
│
├── 📂 data/
│   ├── raw/
│   │   ├── synthetic_network_traffic.csv    ← Auto-generated dataset
│   │   └── sample_data.csv                  ← 5000-row GitHub demo sample
│   └── processed/
│       └── cleaned_traffic.csv              ← After preprocessing
│
├── 📂 src/                                  ← Core Python modules
│   ├── __init__.py
│   ├── data_loader.py                       ← Dataset generation & loading
│   ├── preprocessor.py                      ← Cleaning, encoding, scaling
│   ├── feature_engineering.py               ← Importance ranking, selection
│   ├── model_trainer.py                     ← Train RF + IF models
│   ├── evaluator.py                         ← Metrics, plots, reports
│   ├── alert_generator.py                   ← SOC-style alert generation
│   ├── visualizer.py                        ← 10 professional plots
│   └── threat_detector.py                   ← Unified prediction class
│
├── 📂 models/                               ← Saved trained models
│   ├── random_forest.pkl
│   ├── isolation_forest.pkl
│   └── scaler.pkl
│
├── 📂 outputs/                              ← Generated reports
│   ├── alert_log.csv                        ← Security alerts
│   ├── Random_Forest_report.txt             ← Evaluation report
│   └── Random_Forest_metrics.json           ← JSON metrics
│
├── 📂 images/                               ← All visualization PNGs
│   ├── 01_class_distribution.png
│   ├── 02_attack_distribution.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_feature_importance.png
│   ├── 05_confusion_matrix_rf.png
│   ├── 06_roc_curve_rf.png
│   ├── 07_anomaly_scores.png
│   ├── 08_alert_dashboard.png          ← Dark SOC dashboard
│   ├── 09_feature_distributions.png
│   └── 10_model_comparison.png
│
├── 📂 notebooks/                            ← Jupyter notebooks for learning
│   ├── 01_EDA.ipynb
│   ├── 02_Model_Training.ipynb
│   └── 03_Threat_Detection_Demo.ipynb
│
├── 📂 docs/                                 ← Documentation
│   └── project_report.md
│
├── main.py                                  ← Master pipeline runner ← START HERE
├── requirements.txt                         ← All dependencies
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Cybersecurity-Threat-Detection.git
cd AI-Cybersecurity-Threat-Detection
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Machine Learning Pipeline

```bash
python main.py
```

That's it! The pipeline will automatically:
- ✅ Generate a 50,000-sample synthetic network traffic dataset
- ✅ Preprocess and engineer features
- ✅ Train Random Forest + Isolation Forest
- ✅ Evaluate model performance
- ✅ Generate a security alert log
- ✅ Save 10 visualization charts

### Step 5: Start the SOC Dashboard

```bash
python web/app.py
```

After the server starts, open your browser and navigate to **http://localhost:5000** to view the live Security Operations Center dashboard.

---

## 📊 Results

### Model Performance

| Metric | Random Forest | Isolation Forest |
|--------|:---:|:---:|
| **Accuracy** | ~97% | ~82% |
| **Precision** | ~96% | ~78% |
| **Recall** | ~97% | ~80% |
| **F1-Score** | ~97% | ~79% |
| **ROC-AUC** | ~0.997 | — |

> *Results vary slightly due to random seed. Random Forest consistently outperforms Isolation Forest on labeled data — expected behavior.*

### Why Two Models?
| Scenario | Random Forest | Isolation Forest |
|----------|:---:|:---:|
| Known attack patterns | ✅ Excellent | ⚠️ Moderate |
| Zero-day / unknown threats | ❌ Cannot detect | ✅ Detects anomalies |
| No labeled training data | ❌ Needs labels | ✅ Unsupervised |
| **Combined (best of both)** | **✅** | **✅** |

---

## 🔍 Dataset

This project uses a **synthetic dataset** that mirrors real-world network traffic features from:
- **CICIDS-2017** (Canadian Institute for Cybersecurity)
- **NSL-KDD** benchmark dataset

### Simulated Attack Types
| Attack | Description | % of Attacks |
|--------|-------------|:---:|
| **DoS** | Denial of Service — floods network | 40% |
| **PortScan** | Reconnaissance — scanning for open ports | 25% |
| **BruteForce** | Credential stuffing / password attacks | 15% |
| **DataExfil** | Unauthorized data extraction | 10% |
| **Botnet** | Command & Control communication | 10% |

### Using Real Datasets (Optional)
To use the real CICIDS-2017 dataset:
1. Download from: https://www.unb.ca/cic/datasets/ids-2017.html
2. Place CSV files in `data/raw/`
3. In `main.py`, set: `CONFIG["use_real_dataset"] = True`

---

## 📈 Visualizations

| # | Chart | What It Shows |
|---|-------|---------------|
| 1 | Class Distribution | Normal vs Attack ratio (pie + bar) |
| 2 | Attack Type Distribution | Which attacks dominate the dataset |
| 3 | Correlation Heatmap | Feature relationships |
| 4 | Feature Importance | Top 20 most predictive features |
| 5 | Confusion Matrix | TP, TN, FP, FN breakdown |
| 6 | ROC Curve | Model discrimination at all thresholds |
| 7 | Anomaly Scores | Isolation Forest score distribution |
| 8 | **Alert Dashboard** | Dark SOC-style dashboard |
| 9 | Feature Distributions | Normal vs Attack feature comparison |
| 10 | Model Comparison | RF vs IF across all metrics |

---

## 🔔 Alert System

Generated alert log (`outputs/alert_log.csv`) includes:

| Column | Example |
|--------|---------|
| `alert_id` | `A3F9B1` |
| `timestamp` | `2024-01-15 14:32:07` |
| `source_ip` | `192.168.3.142` |
| `destination_ip` | `10.0.1.88` |
| `port` | `22` (SSH) |
| `protocol` | `TCP` |
| `threat_type` | `Brute Force / Credential Attack` |
| `confidence_score` | `97.3%` |
| `severity` | `CRITICAL` |
| `action_required` | `Block source IP immediately. Escalate to Tier-2 SOC.` |

---

## 🧰 Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.9+** | Core language |
| **Pandas** | Data manipulation |
| **NumPy** | Numerical operations |
| **Scikit-learn** | ML models (RF + IF) |
| **Matplotlib** | Static visualizations |
| **Seaborn** | Statistical plots |
| **Joblib** | Model serialization |
| **TQDM** | Progress bars |

---

## 📚 Learning Objectives

By studying this project you will learn:
- ✅ Real-world ML pipeline design
- ✅ Supervised vs Unsupervised learning comparison
- ✅ Class imbalance handling (`class_weight='balanced'`)
- ✅ Feature engineering and selection techniques
- ✅ Model evaluation beyond accuracy (Precision, Recall, F1, AUC)
- ✅ Confusion matrix interpretation for security context
- ✅ Anomaly detection with Isolation Forest
- ✅ Professional Python project structure

---

## 🚀 GitHub Publishing Guide

### Step 1: Initialize Repository
```bash
git init
git add .
git commit -m "🔐 Initial commit: AI Cybersecurity Threat Detection System"
```

### Step 2: Create GitHub Repository
1. Go to [github.com/new](https://github.com/new)
2. Repository name: `AI-Cybersecurity-Threat-Detection`
3. Set to **Public**
4. Do NOT initialize with README (you already have one)

### Step 3: Push to GitHub
```bash
git remote add origin https://github.com/YOUR_USERNAME/AI-Cybersecurity-Threat-Detection.git
git branch -M main
git push -u origin main
```

### Step 4: Make It Stand Out
- Add **topics/tags**: `cybersecurity`, `machine-learning`, `anomaly-detection`, `python`, `random-forest`, `network-security`
- Pin the repository on your profile
- Add the **images/** folder so charts show in README

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 👨‍💻 Author

**Your Name**  
- GitHub: [@your_username](https://github.com/your_username)
- LinkedIn: [your-linkedin](https://linkedin.com/in/your-linkedin)

---

## ⭐ If This Helped You...

Give this repository a ⭐ — it helps other students find this project!

---

*Built as part of an AI/ML portfolio project for cybersecurity applications.*
