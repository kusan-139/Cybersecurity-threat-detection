# Project Report — AI-Powered Cybersecurity Threat Detection System

## Executive Summary

This project implements a dual-model AI/ML pipeline for cybersecurity threat detection
using simulated network traffic data. It demonstrates industry-standard techniques used
by real SOC (Security Operations Center) teams.

---

## 1. Problem Statement

Modern enterprise networks process millions of packets per second. Manual inspection is
impossible. An automated AI system is needed to:
- Detect known attack patterns (supervised learning)
- Flag unusual behavior even for unknown "zero-day" attacks (unsupervised learning)
- Generate structured alerts for SOC analysts
- Provide explainable AI outputs (feature importance)

---

## 2. Dataset Description

### Synthetic Dataset (Default)
- **Samples**: 50,000 network flow records
- **Features**: 28 (based on CICIDS-2017 feature set)
- **Class Split**: 70% Normal, 30% Attack
- **Attack Types**: DoS (40%), PortScan (25%), BruteForce (15%), DataExfil (10%), Botnet (10%)

### Key Features Used
| Feature | Description |
|---------|-------------|
| `duration` | Connection duration (seconds) |
| `src_bytes` | Bytes sent from source |
| `dst_bytes` | Bytes sent to destination |
| `count` | Connections to same host in 2 seconds |
| `serror_rate` | % SYN error connections |
| `same_srv_rate` | % connections to same service |
| `diff_srv_rate` | % connections to different services |
| `num_failed_logins` | Failed login attempts |
| `logged_in` | Whether login succeeded |

---

## 3. Methodology

### Phase 1: Data Preprocessing
- Removed null/Inf values
- Encoded categorical features
- Applied StandardScaler normalization
- Stratified 75/25 train/test split

### Phase 2: Feature Engineering
- Computed feature importances using a quick Random Forest
- Selected top 20 most informative features
- Analyzed feature correlations

### Phase 3: Model Training

#### Model A — Random Forest (Supervised)
- Algorithm: Ensemble of 100 Decision Trees
- Handles class imbalance: `class_weight='balanced'`
- Max depth: 20 (prevents overfitting)
- Training: Majority vote of all trees

#### Model B — Isolation Forest (Unsupervised)
- Algorithm: Random Isolation Trees
- No labels required during training
- Contamination parameter: 0.20 (estimated 20% attacks)
- Output: Anomaly score (negative = more anomalous)

### Phase 4: Evaluation
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC curve
- Confusion Matrix with FP/FN analysis
- Model comparison chart

### Phase 5: Alert Generation
- Combined predictions from both models
- Severity classification based on confidence scores
- Structured alert log with SOC-standard fields

---

## 4. Results

| Metric | Random Forest | Isolation Forest |
|--------|:---:|:---:|
| Accuracy | ~97% | ~82% |
| Precision | ~96% | ~78% |
| Recall | ~97% | ~80% |
| F1-Score | ~97% | ~79% |
| ROC-AUC | ~0.997 | N/A |

---

## 5. Key Findings

1. **Random Forest** achieved near-perfect detection on known attack patterns
2. **Isolation Forest** successfully detected anomalies without any labels
3. The **combined model** approach reduces false negatives (missed attacks)
4. Top features: `count`, `serror_rate`, `same_srv_rate`, `diff_srv_rate`, `src_bytes`
5. Alert generation correctly prioritized CRITICAL threats for immediate action

---

## 6. Limitations

- Uses synthetic data — real-world performance may differ
- Isolation Forest has higher false positive rate (trade-off for detecting unknowns)
- Static models — in production, models would be retrained periodically

---

## 7. Future Improvements

- [ ] LSTM / Autoencoder for time-series anomaly detection
- [ ] Real-time Kafka stream processing
- [ ] REST API endpoint using FastAPI
- [ ] Streamlit interactive dashboard
- [ ] Active learning to improve model over time
- [ ] Integration with SIEM tools (Splunk, ELK Stack)

---

## 8. References

1. Canadian Institute for Cybersecurity — CICIDS-2017 Dataset
2. Scikit-learn Documentation — Isolation Forest
3. NSL-KDD Dataset — University of New Brunswick
4. IBM Cost of a Data Breach Report 2023
5. MITRE ATT&CK Framework — Enterprise Techniques
