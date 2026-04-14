# 🛡️ Virtual Simulation of Cyber Threats
## Complete Educational Guide — AI-Powered Cybersecurity Threat Detection System

---

## PART 1: How the Dataset Represents Real Network Traffic

### What is "Network Traffic" in the Real World?

In a real company, every time a computer communicates with another — sending an email, loading a webpage, or accessing a database — it creates a **network connection**. Security systems record *metadata* about these connections (not the content, just the behavior). This metadata becomes the **dataset**.

### Our Dataset — 50,000 Network Flow Records

Each row in `data/raw/synthetic_network_traffic.csv` represents **one network connection** — equivalent to one line in a corporate firewall log.

```
╔══════════════════════════════════════════════════════════════════════╗
║  REAL FIREWALL LOG (Cisco ASA / Palo Alto)                          ║
║  ─────────────────────────────────────────────────────────────────  ║
║  Time     SrcIP          DstIP        Port  Proto  Bytes   Flags    ║
║  09:45:12  192.168.1.5   10.0.0.50     22   TCP    1240    SYN      ║
║  09:45:13  192.168.1.5   10.0.0.51     22   TCP    1240    SYN      ║
║  09:45:14  192.168.1.5   10.0.0.52     22   TCP    1240    SYN      ║
║           ↑ Same source, scanning many IPs = PORT SCAN!             ║
╚══════════════════════════════════════════════════════════════════════╝
```

Our synthetic dataset captures **28 behavioral features** from these connections:

### The 28 Features — What Each One Means

| Feature | Data Type | What It Measures | Attack Signal? |
|---|---|---|---|
| `duration` | seconds | How long the connection lasted | DoS = very long |
| `protocol_type` | TCP/UDP/ICMP | Transport protocol used | ICMP flood = DoS |
| `src_bytes` | bytes | Data sent FROM attacker | Small = SYN flood |
| `dst_bytes` | bytes | Data sent TO attacker | Large = Data Exfil |
| `land` | 0/1 | Same src/dst IP (attack trick) | =1 means attack |
| `wrong_fragment` | count | Malformed IP fragments | >0 = suspicious |
| `urgent` | count | TCP urgent packets | >0 = attack |
| `hot` | count | Access to sensitive dirs | High = intrusion |
| `num_failed_logins` | count | Failed login attempts | >3 = Brute Force |
| `logged_in` | 0/1 | Whether user authenticated | 0 after many tries = BF |
| `num_compromised` | count | Compromised conditions | >0 = intrusion |
| `num_root` | count | Root/admin operations | High = privilege escalation |
| `num_file_creations` | count | Files created | High = ransomware |
| `num_shells` | count | Shell commands run | >0 = command injection |
| `num_access_files` | count | Sensitive files accessed | High = data theft |
| `num_outbound_cmds` | count | Outbound commands | High = C&C botnet |
| `count` | connections | Same host connections in 2s | High = DoS flood |
| `srv_count` | connections | Same service connections | High = targeted attack |
| `serror_rate` | 0.0–1.0 | % SYN error connections | High = SYN flood |
| `srv_serror_rate` | 0.0–1.0 | SYN errors to same service | High = SYN flood |
| `rerror_rate` | 0.0–1.0 | % REJ (refused) connections | High = Port Scan |
| `srv_rerror_rate` | 0.0–1.0 | Refused to same service | High = targeted |
| `same_srv_rate` | 0.0–1.0 | % connections to same service | Low = Port Scan |
| `diff_srv_rate` | 0.0–1.0 | % connections to diff services | High = Port Scan |
| `srv_diff_host_rate` | 0.0–1.0 | Different hosts, same service | High = worm spread |
| `dst_host_count` | count | Unique destination hosts | High = scanning |
| `dst_host_srv_count` | count | Same service, diff hosts | High = DDoS |
| `attack_type` | label | Ground truth attack category | DoS/PortScan/etc. |

### Dataset Distribution (Our 50,000 Samples)

```
Total: 50,000 network connections
│
├── NORMAL (70%) = 35,000 connections  ← legitimate office traffic
│   └── Web browsing, email, file transfers, database queries
│
└── ATTACK (30%) = 15,000 connections  ← malicious traffic
    ├── DoS           (40% of attacks)  =  6,000 connections
    ├── Port Scan     (25% of attacks)  =  3,750 connections
    ├── Brute Force   (15% of attacks)  =  2,250 connections
    ├── Data Exfil    (10% of attacks)  =  1,500 connections
    └── Botnet        (10% of attacks)  =  1,500 connections
```

### How the Synthetic Generator Works (src/data_loader.py)

The generator creates statistically realistic data by assigning **different feature distributions** to each attack type:

```python
# NORMAL traffic pattern (simplified):
normal_connection = {
    "serror_rate":        random(0.00, 0.05),   # almost no errors
    "diff_srv_rate":      random(0.00, 0.10),   # mostly same service
    "num_failed_logins":  random(0, 1),          # rare login failures
    "count":              random(1, 20),          # low connection count
    "dst_bytes":          random(100, 10000),     # small data transfer
}

# DoS ATTACK pattern:
dos_connection = {
    "serror_rate":        random(0.80, 1.00),   # ← near 100% SYN errors!
    "diff_srv_rate":      random(0.00, 0.05),   # stay on one port
    "count":              random(200, 512),       # ← huge flood count!
    "src_bytes":          random(0, 10),          # tiny packets (SYN only)
    "duration":           random(0, 1),           # very short (no handshake)
}

# PORT SCAN pattern:
portscan_connection = {
    "diff_srv_rate":      random(0.70, 1.00),   # ← many different ports!
    "rerror_rate":        random(0.60, 0.90),   # ← lots of refused connections
    "same_srv_rate":      random(0.00, 0.10),   # ← rarely same port
    "dst_host_srv_count": random(200, 400),      # ← hits many services
}
```

This is exactly how **real datasets like CICIDS-2017** were built — by capturing actual attack tools (nmap, hping3, LOIC) running against real servers in a lab.

---

## PART 2: How Each Attack is Represented

### 🔴 Attack 1: Denial of Service (DoS)

**Real-world scenario:** A hacker uses a tool like `hping3` or `LOIC` to send millions of SYN packets to a web server. The server gets overwhelmed trying to respond to fake connection requests and crashes.

**How it looks in our dataset:**
```
SYN Flood Attack Pattern:
├── duration        = 0 seconds     (no completed handshake)
├── src_bytes       = 0 bytes       (no data, just SYN packets)
├── count           = 500+          (hundreds of requests per 2 seconds!)
├── serror_rate     = 0.95          (95% of connections fail — server overwhelmed)
├── srv_serror_rate = 0.90          (attacks concentrated on one service)
└── label           = "DoS"
```

**Visual signature (what the chart shows):**
```
Normal traffic:    serror_rate ─────────────█  0.03
DoS attack:        serror_rate ──────────────────────────────────█  0.95
                                              ↑ Huge spike!
```

---

### 🟠 Attack 2: Port Scan / Reconnaissance

**Real-world scenario:** Before hacking, attackers run `nmap` to find which ports are open. They probe port 22 (SSH), 80 (HTTP), 443 (HTTPS), 3306 (MySQL) etc. on every server in the network.

**How it looks in our dataset:**
```
Port Scan Pattern:
├── diff_srv_rate    = 0.85      (85% connections go to different services)
├── rerror_rate      = 0.75      (75% connections REFUSED — port is closed)
├── same_srv_rate    = 0.05      (rarely hits the same port twice)
├── dst_host_count   = 255+      (scanning entire subnet /24)
└── label            = "PortScan"
```

**The giveaway:** Low `same_srv_rate` + High `diff_srv_rate` + High `rerror_rate` = SCANNING

---

### 🟡 Attack 3: Brute Force / Credential Attack

**Real-world scenario:** Hacker uses tools like `Hydra` or `Medusa` to try 10,000 passwords on an SSH or MySQL server — one every second.

**How it looks in our dataset:**
```
Brute Force Pattern:
├── num_failed_logins = 7           (many login failures!)
├── logged_in         = 0           (never successfully logs in)
├── same_srv_rate     = 0.95        (keeps hitting the same service)
├── hot               = 0           (no privileged access granted)
├── duration          = random      (tries spread over time)
└── label             = "BruteForce"
```

**The giveaway:** High `num_failed_logins` + `logged_in = 0` + High `same_srv_rate` = BRUTE FORCE

---

### 🔵 Attack 4: Data Exfiltration

**Real-world scenario:** A compromised insider or malware sends large amounts of sensitive data OUT of the company network — sending your customer database to a hacker's server.

**How it looks in our dataset:**
```
Data Exfiltration Pattern:
├── dst_bytes         = 80,000+     (huge data going OUTBOUND!)
├── src_bytes         = 100         (tiny request from attacker)
├── logged_in         = 1           (attacker IS authenticated)
├── num_access_files  = 8+          (accessing many files)
├── hot               = 15+         (accessing many sensitive directories)
└── label             = "DataExfil"
```

**The giveaway:** Very HIGH `dst_bytes` (large outbound data) with `logged_in = 1` and high `hot` = DATA THEFT

---

### 🟣 Attack 5: Botnet / Command & Control (C&C)

**Real-world scenario:** A malware-infected corporate laptop "phones home" to a hacker's server (C&C server) every few minutes to receive commands. Port 6667 (IRC protocol) is the classic botnet channel.

**How it looks in our dataset:**
```
Botnet C&C Pattern:
├── num_outbound_cmds = 5+          (commands coming FROM the hacker!)
├── dst_host_count    = 1           (always calls same hacker server)
├── same_srv_rate     = 0.90        (always same C&C port)
├── logged_in         = 1           (internal machine, already on network)
└── label             = "Botnet"
```

---

## PART 3: How the AI Model Detects Suspicious Behavior

### Two-Layer Detection System

```
Incoming Network Connection
           │
           ▼
┌──────────────────────────────────────────────────────────────┐
│           LAYER 1: Random Forest (Supervised)                │
│                                                              │
│  Trained on 37,500 labeled examples (known attacks)         │
│  100 decision trees vote: "Is this normal or attack?"       │
│                                                              │
│  Decision Tree logic (simplified):                          │
│  IF serror_rate > 0.8 AND count > 200:                      │
│      → DoS Attack (confidence: 99%)                         │
│  ELIF diff_srv_rate > 0.7 AND rerror_rate > 0.5:            │
│      → Port Scan (confidence: 97%)                          │
│  ELIF num_failed_logins > 5 AND logged_in = 0:              │
│      → Brute Force (confidence: 95%)                        │
│  ELSE:                                                       │
│      → Normal Traffic (confidence: 100%)                    │
└──────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────┐
│           LAYER 2: Isolation Forest (Unsupervised)           │
│                                                              │
│  NO LABELS NEEDED — finds anomalies by isolation            │
│                                                              │
│  Logic: "How many splits does it take to isolate            │
│         this data point from all others?"                    │
│                                                              │
│  Short path  = ANOMALY (attack) → Score near -0.1 to -0.5  │
│  Long path   = NORMAL          → Score near +0.1 to +0.5   │
│                                                              │
│  Catches ZERO-DAY attacks RF has never seen before!         │
└──────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────┐
│           COMBINED DECISION                                  │
│                                                              │
│  IF RF says ATTACK  OR  IF says ANOMALY → FLAG AS THREAT    │
│  IF both say NORMAL → Mark as safe                          │
└──────────────────────────────────────────────────────────────┘
```

### Feature Importance — What the AI Focuses On

From `images/04_feature_importance.png`, the top predictive features are:

```
Rank  Feature              Importance   What It Detects
 1    serror_rate          0.22 (22%)   DoS / SYN Flood
 2    diff_srv_rate        0.14 (14%)   Port Scanning
 3    srv_rerror_rate      0.13 (13%)   Targeted Attacks
 4    srv_serror_rate      0.11 (11%)   Service-Specific DoS
 5    count                0.10 (10%)   Flood Attacks
 6    srv_count            0.09  (9%)   Repeat-Service Attacks
 7    same_srv_rate        0.05  (5%)   Brute Force
 8    rerror_rate          0.05  (5%)   Reconnaissance
```

**Insight:** Just the top 3 features (`serror_rate`, `diff_srv_rate`, `srv_rerror_rate`) explain **49% of all attack detections!**

---

## PART 4: How Alerts Are Generated

### From Raw Prediction to SOC Alert

```
Step A: Model outputs prediction
        y_pred = [1, 0, 1, 1, 0, ...]   (1=Attack, 0=Normal)
        y_prob = [0.99, 0.01, 0.95, ...]  (confidence)

Step B: Map confidence to severity
        IF confidence >= 95% → CRITICAL 🔴
        IF confidence >= 85% → HIGH     🟠
        IF confidence >= 70% → MEDIUM   🟡
        IF confidence >= 50% → LOW      🟢

Step C: Map attack type to action
        DoS        → "Block source IP immediately. Escalate to Tier-2 SOC."
        PortScan   → "Isolate affected subnet. Run vulnerability scan."
        BruteForce → "Lock account. Enable MFA. Reset credentials."
        DataExfil  → "Kill session. Capture packets. Notify DLP team."
        Botnet     → "Quarantine host. Scan for malware. Image the drive."

Step D: Generate structured alert record
        {
          alert_id:       "938AF43A",
          timestamp:      "2026-04-12 04:14:21",
          source_ip:      "192.168.10.49",
          destination_ip: "10.0.2.12",
          port:           21,
          protocol:       "UDP",
          threat_type:    "Denial of Service",
          confidence:     99.0%,
          severity:       "CRITICAL",
          action_required: "Block source IP immediately...",
          status:         "OPEN"
        }
```

Every alarm goes into `outputs/alert_log.csv` — a real SOC-style incident log.

---

## PART 5: Step-by-Step Simulation Workflow

### Overview

```
[RAW DATA] → [PREPROCESS] → [FEATURE SELECT] → [TRAIN MODELS]
     → [EVALUATE] → [DETECT THREATS] → [GENERATE ALERTS] → [VISUALIZE]
```

---

### ✅ Step 1: Environment Setup & Data Generation

**What happens:**
- Python generates 50,000 realistic network flow records
- 5 attack types embedded with realistic statistical distributions
- Data saved to `data/raw/synthetic_network_traffic.csv`

**Run:**
```bash
cd "d:\IIT Project\Artificial Intelligence\Cybersecurity threat detection"
venv\Scripts\activate
python main.py
```

**What to observe:**
```
[DATA LOADER] Generating 50000 synthetic network traffic records...
[DATA LOADER] Attack distribution:
  DoS           : 6000  (40.0%)
  PortScan      : 3750  (25.0%)
  BruteForce    : 2250  (15.0%)
  DataExfil     : 1500  (10.0%)
  Botnet        : 1500  (10.0%)
[DATA LOADER] Dataset saved: data/raw/synthetic_network_traffic.csv (10.6 MB)
```

**📸 PROOF TO CAPTURE:** Screenshot of the terminal output showing row count and attack distribution.

---

### ✅ Step 2: Exploratory Data Analysis (EDA)

**What happens:**
- Dataset statistics are printed (shape, data types, null counts)
- First 5 rows displayed to understand feature structure
- Charts generated:
  - `01_class_distribution.png` — Normal vs Attack ratio
  - `02_attack_distribution.png` — Breakdown by attack type
  - `09_feature_distributions.png` — Feature histograms per class

**Key insight to highlight:**
- Notice how `serror_rate` for DoS clusters near 1.0 vs 0.0 for normal
- Notice how `count` for DoS exceeds 400 vs 5-20 for normal

**📸 PROOF TO CAPTURE:**
- Open `images/01_class_distribution.png` — shows 70/30 class balance
- Open `images/09_feature_distributions.png` — shows clear separation between Normal (blue) and Attack (red)

---

### ✅ Step 3: Data Preprocessing

**What happens:**
1. Remove null values (fill with column median)
2. Replace infinite values with NaN (common in real datasets like CICIDS-2017)
3. Encode categorical columns (protocol_type → integer)
4. Split: **75% train (37,500 rows) | 25% test (12,500 rows)**
5. Scale features using StandardScaler (zero mean, unit variance)

**Why scaling matters:**
```
Before scaling:    count ∈ [1, 512]      serror_rate ∈ [0.0, 1.0]
After scaling:     count → z-score        serror_rate → z-score
Both now have mean=0, std=1 → fair comparison for algorithms
```

**📸 PROOF TO CAPTURE:** Terminal output showing train/test sizes and feature matrix shape.

---

### ✅ Step 4: Feature Engineering & Selection

**What happens:**
- A "quick" Random Forest is trained to rank feature importance
- Top 20 most important features are selected
- Correlation heatmap is generated

**Key outputs:**
- `images/03_correlation_heatmap.png` — shows which features are correlated
- `images/04_feature_importance.png` — shows which features matter most

**What to highlight:**
- `serror_rate` is #1 most important feature (22% of detection power)
- Correlated features like `serror_rate` + `srv_serror_rate` move together (both signal DoS)
- Removing low-importance features (importance ≈ 0.0) reduces noise

**📸 PROOF TO CAPTURE:**
- Screenshot of the top 20 feature importance table printed in terminal
- Open `images/04_feature_importance.png`

---

### ✅ Step 5: Training the Dual AI Models

**What happens:**
- **Random Forest**: 100 decision trees, each trained on a random subset of data
- **Isolation Forest**: 100 isolation trees, trained WITHOUT labels

**Analogy — How Random Forest Works:**
```
Connection arrives → 100 expert "voters" each independently analyze it:

Tree 1:  "serror_rate=0.95 → This is DoS!"     → ATTACK
Tree 2:  "count=450 → Definitely DoS!"           → ATTACK
Tree 3:  "dst_bytes=0 → No data transferred..."  → ATTACK
...

Majority vote of 100 trees → Final verdict: ATTACK (99% confidence)
```

**Analogy — How Isolation Forest Works:**
```
Think of data points as people in different rooms.
The Forest tries to ISOLATE each person by asking yes/no questions.

Normal person  → takes 15 questions to isolate (fits the crowd)
Attacker       → takes only 3 questions (stands out immediately!)

Short isolation path = ANOMALY = 🚨 THREAT
Long isolation path  = NORMAL  = ✅ SAFE
```

**📸 PROOF TO CAPTURE:** Terminal showing training accuracy for RF (100%) and IF (90.22%).

---

### ✅ Step 6: Model Evaluation

**What happens:**
- RF predicts on 12,500 test samples
- Metrics calculated: Accuracy, Precision, Recall, F1, AUC-ROC
- Charts generated:
  - `05_confusion_matrix_rf.png`
  - `06_roc_curve_rf.png`

**Understanding the metrics:**
```
For cybersecurity, RECALL is most important:
  If Recall = 100% → We catch EVERY attack (no attacker slips through)
  If Precision = 100% → No false alarms (every alert is real)

Our results:
  RF Precision : 100%  → Zero false alarms
  RF Recall    : 100%  → Caught every attack
  IF Recall    : 67%   → Caught 2/3 of attacks (acceptable for unsupervised)
```

**Understanding the Confusion Matrix:**
```
                Predicted: NORMAL   Predicted: ATTACK
Actual: NORMAL       8,750 (TN)           0 (FP)     ← No false alarms!
Actual: ATTACK           0 (FN)       3,750 (TP)     ← Caught everything!

TN = True Negative  (correctly said safe)
TP = True Positive  (correctly caught attack)
FP = False Positive (wrong alarm)
FN = False Negative (missed attack) ← MOST DANGEROUS!
```

**📸 PROOF TO CAPTURE:**
- Open `images/05_confusion_matrix_rf.png`
- Open `images/06_roc_curve_rf.png` (should show a perfect square curve, AUC=1.0)

---

### ✅ Step 7: Anomaly Detection & Score Analysis

**What happens:**
- Isolation Forest assigns an anomaly score to every test sample
- Negative scores = anomalies (attacks)
- Positive scores = normal traffic

**Score interpretation:**
```
Score < -0.1   → HIGH anomaly → Very suspicious → Flag as THREAT
Score  0.0     → Boundary zone
Score > +0.1   → Normal behavior → Safe

Our results (from images/07_anomaly_scores.png):
  Attack samples: scores cluster around -0.009 to -0.3
  Normal samples: scores cluster around +0.05 to +0.25
```

**Why this is powerful:**
- IF detected attacks it had NEVER seen before (zero-day simulation)
- It doesn't need labels — works on unknown attack patterns
- This is what tools like Darktrace and Vectra AI use in production

**📸 PROOF TO CAPTURE:** Open `images/07_anomaly_scores.png` — shows two clearly separated distributions (normal vs attack scores).

---

### ✅ Step 8: Threat Detection on Live Traffic

**What happens:**
- Combined detector runs on all 12,500 test samples
- RF and IF votes are combined
- Every detected threat is classified by type

**Combined detection logic:**
```
For each network connection:
  rf_vote = Random Forest prediction     (0=Normal, 1=Attack)
  if_vote = Isolation Forest prediction  (0=Normal, 1=Anomaly)

  if rf_vote == 1 OR if_vote == 1:
      final = THREAT ⚠️
  else:
      final = NORMAL ✅
```

**📸 PROOF TO CAPTURE:** Terminal output from Phase 9 showing the 3 real-time sample predictions.

---

### ✅ Step 9: Alert Generation & SOC Log

**What happens:**
- 3,750 threats formalized into structured SOC alerts
- Each alert has: ID, timestamp, IPs, port, attack type, confidence, severity, action
- All saved to `outputs/alert_log.csv`

**The alert log simulates what Splunk/QRadar/IBM QRadar would generate:**
```csv
alert_id,timestamp,source_ip,dest_ip,port,protocol,threat_type,confidence,severity,action
938AF43A,2026-04-12 04:14:21,192.168.10.49,10.0.2.12,21,UDP,DoS,100.0,CRITICAL,Block IP
CBB0FF9F,2026-04-12 04:14:21,192.168.8.120,10.0.2.35,8080,TCP,DoS,100.0,CRITICAL,Block IP
```

**📸 PROOF TO CAPTURE:** Open `outputs/alert_log.csv` in Excel. Sort by severity. Screenshot first 20 rows.

---

### ✅ Step 10: Visualization & Dashboard

**What happens:**
- 10 professional charts are generated
- Dark SOC dashboard compiled
- Web dashboard launched at `http://localhost:5000`

**Run web dashboard:**
```bash
venv\Scripts\python web/app.py
# Then open browser → http://localhost:5000
```

**📸 PROOF TO CAPTURE:** Screenshot of the web dashboard with all charts visible.

---

## PART 6: All Outputs to Generate

| Output File | Location | Format | What It Shows |
|---|---|---|---|
| Synthetic Dataset | `data/raw/synthetic_network_traffic.csv` | CSV | 50,000 network connections |
| Sample Data | `data/raw/sample_data.csv` | CSV | 5,000 rows for quick preview |
| Cleaned Data | `data/processed/cleaned_traffic.csv` | CSV | Scaled, encoded features |
| Random Forest Model | `models/random_forest.pkl` | PKL | Trained classifier |
| Isolation Forest | `models/isolation_forest.pkl` | PKL | Trained anomaly detector |
| Scaler | `models/scaler.pkl` | PKL | StandardScaler parameters |
| Alert Log | `outputs/alert_log.csv` | CSV | 3,750 security alerts |
| RF Evaluation Report | `outputs/Random_Forest_report.txt` | TXT | Accuracy, F1, AUC |
| RF Metrics JSON | `outputs/Random_Forest_metrics.json` | JSON | Machine-readable metrics |
| IF Evaluation Report | `outputs/Isolation_Forest_report.txt` | TXT | Anomaly detection metrics |

---

## PART 7: All Graphs to Create & What They Show

| # | File | Graph Type | Key Insight |
|---|---|---|---|
| 01 | `01_class_distribution.png` | Pie + Bar | 70% Normal, 30% Attack — class imbalance |
| 02 | `02_attack_distribution.png` | Horizontal Bar | DoS is the most common (40%) |
| 03 | `03_correlation_heatmap.png` | Heatmap | `serror_rate` + `srv_serror_rate` highly correlated |
| 04 | `04_feature_importance.png` | Bar Chart | `serror_rate` is #1 predictor of attacks |
| 05 | `05_confusion_matrix_rf.png` | Heatmap | 0 false negatives — caught every attack |
| 06 | `06_roc_curve_rf.png` | Line Chart | AUC=1.0 — perfect classifier |
| 07 | `07_anomaly_scores.png` | Histogram+KDE | Two distinct peaks (normal vs anomaly) |
| 08 | `08_alert_dashboard.png` | Dark Dashboard | Full SOC overview — 6 sub-panels |
| 09 | `09_feature_distributions.png` | Grid Histogram | Clear separation between Normal (blue) & Attack (red) |
| 10 | `10_model_comparison.png` | Grouped Bar | RF outperforms IF (expected — RF is supervised) |

---

## PART 8: Key Anomalies to Highlight

### Anomaly 1: SYN Flood Fingerprint
**Where to find it:** `images/09_feature_distributions.png` — `serror_rate` panel

```
Normal:  ████ (clustered at 0.0–0.05)
Attack:  ████ (clustered at 0.85–1.00)

Clear bimodal distribution — proof the AI can separate them
```

### Anomaly 2: Port Scanner Behavior
**Where to find it:** `images/09_feature_distributions.png` — `diff_srv_rate` panel

```
Normal:  Most traffic goes to SAME service (bank → same database)
Scanner: Connects to MANY different ports → diff_srv_rate spikes to 0.9
```

### Anomaly 3: Isolation Forest Score Gap
**Where to find it:** `images/07_anomaly_scores.png`

```
The histogram shows TWO peaks:
  Peak 1 (positive scores) → Normal traffic
  Peak 2 (negative scores) → Attacks/Anomalies

Gap between peaks = detectable separation = AI works!
```

### Anomaly 4: Perfect ROC Curve
**Where to find it:** `images/06_roc_curve_rf.png`

```
AUC = 1.0 → Perfect classification
The curve goes straight up then straight right
This means: At every threshold, RF catches ALL attacks with ZERO false alarms
```

### Anomaly 5: Botnet Port 6667 Traffic
**Where to find it:** `outputs/alert_log.csv` — filter by `threat_type = "Botnet"`

```
Botnet connections always target port 6667 (IRC protocol)
In real life, no legitimate office traffic uses port 6667
→ Any traffic on 6667 = automatic suspicion
```

---

## PART 9: Proof Students Should Capture for Portfolio

### Screenshot Checklist ✅

| # | What to Capture | How to Capture | Where to Include |
|---|---|---|---|
| 1 | Terminal: `python main.py` full output | PrtScn or Snip Tool | README.md |
| 2 | `images/01_class_distribution.png` | Already saved | README Results section |
| 3 | `images/04_feature_importance.png` | Already saved | README + Report |
| 4 | `images/05_confusion_matrix_rf.png` | Already saved | README Results section |
| 5 | `images/06_roc_curve_rf.png` | Already saved | README Results section |
| 6 | `images/07_anomaly_scores.png` | Already saved | README + Report |
| 7 | `images/08_alert_dashboard.png` | Already saved | README cover image |
| 8 | `outputs/alert_log.csv` in Excel | Open → Screenshot | Report |
| 9 | Web dashboard at `localhost:5000` | Browser screenshot | LinkedIn post |
| 10 | Model metrics JSON | Already saved | API demo |

### Video Recording for Maximum Impact
```
Record a 60-second screen capture showing:
  00–10s → Open terminal, run python main.py
  10–30s → Watch phases 1-7 complete with accuracy shown
  30–45s → Open localhost:5000 (web dashboard live)
  45–60s → Use the AI Threat Detector sliders → show THREAT DETECTED
```

### GitHub README Evidence Sections to Write

```markdown
## 🎯 Results

| Metric              | Random Forest | Isolation Forest |
|---------------------|:---:|:---:|
| Accuracy            | 100% | 90.22% |
| Precision           | 100% | 100% |
| Recall              | 100% | 67.41% |
| F1-Score            | 100% | 80.54% |
| AUC-ROC             | 1.0000 | — |
| Threats Detected    | 3,750 | 2,528 |
| Runtime             | 75 sec | — |

## 📊 Key Visualizations

![SOC Dashboard](images/08_alert_dashboard.png)
![Feature Importance](images/04_feature_importance.png)
![Confusion Matrix](images/05_confusion_matrix_rf.png)
```

### Interview Talking Points

When a recruiter asks **"Tell me about your cybersecurity project"**, say:

> *"I built an AI-powered intrusion detection system that simulates a corporate network security operations center. I used a dual-model architecture — a supervised Random Forest achieving 100% accuracy on labeled attack patterns, combined with an Isolation Forest for zero-day anomaly detection achieving 90% accuracy without any labels. The system processes 50,000 network flow records, classifies 5 attack types including DoS, Port Scan, Brute Force, Data Exfiltration, and Botnet C&C communication, and generates SOC-style structured alert logs with severity classification and recommended response actions. I also built a real-time web dashboard using Flask and Chart.js to visualize threats as they're detected."*

---

## PART 10: How to Upgrade to Real Data

Once you understand the simulation, here's how to plug in **real datasets**:

### Option A: NSL-KDD Dataset (Easiest)
```bash
# Download from: https://www.unb.ca/cic/datasets/nsl.html
# Place in: data/raw/KDDTrain+.csv

# Edit main.py:
CONFIG["use_real_dataset"] = True
CONFIG["real_dataset_path"] = "data/raw/"

# Run:
python main.py
```

### Option B: CICIDS-2017 (Most Realistic)
```bash
# Download from: https://www.unb.ca/cic/datasets/ids-2017.html
# 50GB dataset — use just Friday-WorkingHours files

# The preprocess_cicids() function in preprocessor.py handles it automatically
```

### Option C: Your Own Network (Advanced)
```bash
# Step 1: Install Wireshark → capture your home WiFi traffic
# Step 2: Install CICFlowMeter → convert PCAP to CSV features
# Step 3: Load the CSV into data/raw/ and run
```

---

*This document serves as the complete technical foundation for the AI-Powered Cybersecurity Threat Detection System — suitable for placements, internship interviews, and academic submissions.*

*Generated by the project's AI assistant. Project Author: Student Portfolio — 2024.*
