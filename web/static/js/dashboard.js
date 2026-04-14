/* ============================================================
   dashboard.js — AI Cybersecurity SOC Dashboard JavaScript
   Handles: charts, live feed, AI detector, tables, clock
   ============================================================ */

"use strict";

// ── Chart Color Palette ────────────────────────────────────────────────────
const COLORS = {
  critical: "#ff1744",
  high:     "#ff6d00",
  medium:   "#ffab00",
  low:      "#00e676",
  blue:     "#2196f3",
  cyan:     "#00bcd4",
  purple:   "#9c27b0",
  teal:     "#009688",
  indigo:   "#3f51b5",
};

const CHART_DEFAULTS = {
  plugins: {
    legend: { labels: { color: "#8b949e", font: { family: "Inter", size: 11 } } },
    tooltip: {
      backgroundColor: "#1c2128",
      borderColor: "#30363d",
      borderWidth: 1,
      titleColor: "#e6edf3",
      bodyColor: "#8b949e",
      padding: 10,
    }
  },
  animation: { duration: 800, easing: "easeOutQuart" },
};

Chart.defaults.color = "#8b949e";
Chart.defaults.borderColor = "#30363d";

// ─────────────────────────────────────────────────────────────────────────────
// 1. CLOCK
// ─────────────────────────────────────────────────────────────────────────────
function startClock() {
  const el = document.getElementById("navTime");
  function update() {
    const now = new Date();
    el.textContent = now.toLocaleTimeString("en-GB", { hour12: false });
  }
  update();
  setInterval(update, 1000);
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. COUNT-UP ANIMATION
// ─────────────────────────────────────────────────────────────────────────────
function animateCountUp() {
  document.querySelectorAll(".count-up").forEach(el => {
    const target  = parseInt(el.dataset.target || "0");
    const dur     = 1200;
    const fps     = 60;
    const step    = target / (dur / (1000 / fps));
    let   current = 0;
    const timer   = setInterval(() => {
      current += step;
      if (current >= target) { current = target; clearInterval(timer); }
      el.textContent = Math.round(current).toLocaleString();
    }, 1000 / fps);
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. TIMELINE CHART
// ─────────────────────────────────────────────────────────────────────────────
async function initTimelineChart() {
  const data = await fetchJSON("/api/timeline");
  if (!data || !data.labels) return;

  new Chart(document.getElementById("timelineChart"), {
    type: "line",
    data: {
      labels: data.labels,
      datasets: [{
        label: "Alert Count",
        data:  data.values,
        borderColor: COLORS.blue,
        backgroundColor: "rgba(33,150,243,0.08)",
        borderWidth: 2.5,
        pointBackgroundColor: COLORS.blue,
        pointRadius: 4,
        pointHoverRadius: 6,
        tension: 0.4,
        fill: true,
      }]
    },
    options: {
      ...CHART_DEFAULTS,
      responsive: true,
      maintainAspectRatio: true,
      scales: {
        x: { grid: { color: "rgba(48,54,61,0.6)" }, ticks: { maxTicksLimit: 12 } },
        y: { grid: { color: "rgba(48,54,61,0.6)" }, beginAtZero: true }
      }
    }
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. ATTACK TYPE DOUGHNUT
// ─────────────────────────────────────────────────────────────────────────────
async function initAttackChart() {
  const data = await fetchJSON("/api/attack-distribution");
  if (!data) return;

  const labels = Object.keys(data);
  const values = Object.values(data);
  const bgColors = [COLORS.critical, COLORS.high, COLORS.blue, COLORS.purple, COLORS.cyan, COLORS.teal];

  new Chart(document.getElementById("attackChart"), {
    type: "doughnut",
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: bgColors.slice(0, labels.length),
        borderColor: "#161b22",
        borderWidth: 3,
        hoverOffset: 6,
      }]
    },
    options: {
      ...CHART_DEFAULTS,
      responsive: true,
      cutout: "62%",
      plugins: {
        ...CHART_DEFAULTS.plugins,
        legend: { position: "bottom", labels: { ...CHART_DEFAULTS.plugins.legend.labels, boxWidth: 10, padding: 12 } }
      }
    }
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. SEVERITY DOUGHNUT
// ─────────────────────────────────────────────────────────────────────────────
async function initSeverityChart() {
  const data = await fetchJSON("/api/severity-distribution");
  if (!data) return;

  const order    = ["CRITICAL", "HIGH", "MEDIUM", "LOW"];
  const sevColors = { CRITICAL: COLORS.critical, HIGH: COLORS.high, MEDIUM: COLORS.medium, LOW: COLORS.low };
  const labels   = order.filter(k => data[k] !== undefined);
  const values   = labels.map(k => data[k]);
  const colors   = labels.map(k => sevColors[k]);

  new Chart(document.getElementById("severityChart"), {
    type: "doughnut",
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: colors,
        borderColor: "#161b22",
        borderWidth: 3,
        hoverOffset: 6,
      }]
    },
    options: {
      ...CHART_DEFAULTS,
      responsive: true,
      cutout: "62%",
      plugins: {
        ...CHART_DEFAULTS.plugins,
        legend: { position: "bottom", labels: { ...CHART_DEFAULTS.plugins.legend.labels, boxWidth: 10, padding: 12 } }
      }
    }
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. TOP IPs HORIZONTAL BAR
// ─────────────────────────────────────────────────────────────────────────────
async function initIPChart() {
  const data = await fetchJSON("/api/top-ips");
  if (!data || !data.ips) return;

  const gradient = data.counts.map((_, i) => {
    const t = i / Math.max(data.counts.length - 1, 1);
    return `rgba(33, 150, 243, ${1 - t * 0.6})`;
  });

  new Chart(document.getElementById("ipChart"), {
    type: "bar",
    data: {
      labels: data.ips,
      datasets: [{
        label: "Alert Count",
        data:  data.counts,
        backgroundColor: gradient,
        borderColor: COLORS.blue,
        borderWidth: 1,
        borderRadius: 4,
      }]
    },
    options: {
      ...CHART_DEFAULTS,
      responsive: true,
      indexAxis: "y",
      maintainAspectRatio: true,
      scales: {
        x: { grid: { color: "rgba(48,54,61,0.6)" }, beginAtZero: true },
        y: { grid: { color: "rgba(48,54,61,0.6)" }, ticks: { font: { family: "JetBrains Mono", size: 11 } } }
      },
      plugins: { ...CHART_DEFAULTS.plugins, legend: { display: false } }
    }
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. ALERTS TABLE
// ─────────────────────────────────────────────────────────────────────────────
async function initAlertsTable() {
  const alerts = await fetchJSON("/api/alerts");
  const tbody  = document.getElementById("alertTableBody");
  const badge  = document.getElementById("alertCount");

  if (!alerts || !alerts.length) {
    tbody.innerHTML = '<tr><td colspan="7" class="table-loading">No alerts found. Run main.py first.</td></tr>';
    return;
  }

  badge.textContent = `Latest ${alerts.length} alerts`;

  const sevColor = { CRITICAL: "var(--critical)", HIGH: "var(--high)", MEDIUM: "var(--medium)", LOW: "var(--low)" };

  tbody.innerHTML = alerts.slice(0, 50).map(a => `
    <tr>
      <td style="font-family:var(--font-mono);color:var(--text-mono);">${a.alert_id || "—"}</td>
      <td>${a.timestamp || "—"}</td>
      <td style="color:var(--accent-cyan);">${a.source_ip || "—"} → ${a.destination_ip || ""}:${a.port || ""}</td>
      <td style="font-weight:500;">${a.threat_type || "Unknown"}</td>
      <td>
        <div style="display:flex;align-items:center;gap:6px;">
          <div style="height:4px;width:${Math.round(a.confidence_score || 0)}px;max-width:60px;
                      background:${sevColor[a.severity] || "var(--accent-blue)"};border-radius:2px;"></div>
          <span>${a.confidence_score || 0}%</span>
        </div>
      </td>
      <td>
        <span class="sev-badge-table sev-${a.severity}"
              style="background:${sevColor[a.severity]}22;color:${sevColor[a.severity]};
                     border:1px solid ${sevColor[a.severity]}44;">
          ${a.severity || "—"}
        </span>
      </td>
      <td style="color:var(--text-secondary);max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">
        ${a.action_required || "—"}
      </td>
    </tr>
  `).join("");
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. LIVE THREAT FEED (simulated real-time)
// ─────────────────────────────────────────────────────────────────────────────
const feedList = [];
const MAX_FEED = 12;

async function fetchLiveThreat() {
  try {
    const alert = await fetchJSON("/api/live-feed");
    if (!alert) return;

    feedList.unshift(alert);
    if (feedList.length > MAX_FEED) feedList.pop();

    renderFeed();
    updateTicker(alert);
    updateKPICounters(alert.severity);
  } catch (_) {}
}

/**
 * Dynamically increments the KPI counters at the top of the dashboard.
 * @param {string} severity - The severity level of the new alert.
 */
function updateKPICounters(severity) {
  const sevKey = severity.toLowerCase();
  const kpiMap = {
    'critical': '.kpi-critical .kpi-value',
    'high':     '.kpi-high .kpi-value',
    'medium':   '.kpi-medium .kpi-value',
    'low':      '.kpi-low .kpi-value',
    'total':    '.kpi-total .kpi-value'
  };

  // 1. Increment the specific severity counter
  const sevEl = document.querySelector(kpiMap[sevKey]);
  if (sevEl) {
    let val = parseInt(sevEl.textContent.replace(/,/g, '')) || 0;
    sevEl.textContent = (val + 1).toLocaleString();
  }

  // 2. Increment the total alerts counter
  const totalEl = document.querySelector(kpiMap['total']);
  if (totalEl) {
    let val = parseInt(totalEl.textContent.replace(/,/g, '')) || 0;
    totalEl.textContent = (val + 1).toLocaleString();
  }
}

function renderFeed() {
  const container = document.getElementById("threatFeed");
  if (!feedList.length) return;

  container.innerHTML = feedList.map(a => `
    <div class="feed-item">
      <span class="feed-sev sev-${a.severity}">${a.severity}</span>
      <div class="feed-details">
        <div class="feed-threat">${a.threat_type}</div>
        <div class="feed-meta">${a.source_ip} → ${a.destination_ip}:${a.port} · ${a.confidence_score}% · ${a.timestamp.slice(11,19)}</div>
      </div>
    </div>
  `).join("");
}

function updateTicker(alert) {
  const el = document.getElementById("tickerContent");
  el.textContent += `  ⚠ [${alert.severity}] ${alert.threat_type} — ${alert.source_ip} → Port ${alert.port} — ${alert.confidence_score}% confidence  |`;
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. AI THREAT DETECTOR (Slider-based simulation)
// ─────────────────────────────────────────────────────────────────────────────
function updateSlider(id, value) {
  document.getElementById(id).textContent = parseFloat(value).toFixed(2);
}

function runDetection() {
  const btn = document.getElementById("detectBtn");
  const result = document.getElementById("detectionResult");

  btn.textContent = "⟳ Analyzing...";
  btn.disabled = true;

  setTimeout(() => {
    // Read slider values
    const serror    = parseFloat(document.getElementById("serrorRate").value);
    const diffSrv   = parseFloat(document.getElementById("diffSrvRate").value);
    const count     = parseInt(document.getElementById("connCount").value);
    const sameSrv   = parseFloat(document.getElementById("sameSrvRate").value);
    const failed    = parseInt(document.getElementById("failedLogins").value);
    const rerror    = parseFloat(document.getElementById("rerrorRate").value);

    // Rule-based heuristic mirroring the ML model's top features
    let threatScore = 0;
    if (serror  > 0.5)  threatScore += 40;
    if (diffSrv > 0.6)  threatScore += 30;
    if (count   > 200)  threatScore += 20;
    if (sameSrv < 0.3)  threatScore += 15;
    if (failed  >= 3)   threatScore += 25;
    if (rerror  > 0.5)  threatScore += 20;

    // Normalize to 0–100
    threatScore = Math.min(threatScore, 100);
    const normalScore = 100 - threatScore;

    let verdict, cssClass, icon, threatType;

    if (threatScore >= 70) {
      verdict = `THREAT DETECTED — Confidence: ${threatScore}%`;
      if (serror > 0.5 && count > 200) threatType = "DoS / DDoS Attack";
      else if (failed >= 3)            threatType = "Brute Force / Credential Attack";
      else if (diffSrv > 0.6)          threatType = "Port Scan / Reconnaissance";
      else                             threatType = "Suspicious Activity";
      cssClass = "result-threat";
      icon = "⚠️";
    } else if (threatScore >= 35) {
      verdict  = `SUSPICIOUS — Threat Score: ${threatScore}%`;
      threatType = "Anomalous Traffic Pattern";
      cssClass = "result-medium";
      icon = "🟡";
    } else {
      verdict  = `NORMAL TRAFFIC — Threat Score: ${threatScore}%`;
      threatType = "Benign Connection";
      cssClass = "result-normal";
      icon = "✅";
    }

    result.style.display = "block";
    result.className = `detection-result ${cssClass}`;
    result.innerHTML = `
      <div style="font-size:16px;margin-bottom:8px;">${icon} ${verdict}</div>
      <div style="font-size:11px;opacity:0.8;">Classification: <strong>${threatType}</strong></div>
      <div style="font-size:11px;opacity:0.8;margin-top:4px;">
        Normal Score: ${normalScore}% | Threat Score: ${threatScore}%
      </div>
      <div style="margin-top:8px;height:6px;background:rgba(255,255,255,0.1);border-radius:3px;overflow:hidden;">
        <div style="height:100%;width:${threatScore}%;background:${threatScore>=70?'#ff1744':threatScore>=35?'#ffab00':'#00e676'};border-radius:3px;transition:width 0.5s;"></div>
      </div>
    `;

    btn.textContent = "🔍 ANALYZE CONNECTION";
    btn.disabled = false;
  }, 900);
}

// ─────────────────────────────────────────────────────────────────────────────
// HELPER: fetch JSON
// ─────────────────────────────────────────────────────────────────────────────
async function fetchJSON(url) {
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (e) {
    console.warn(`[Dashboard] Failed to fetch ${url}:`, e.message);
    return null;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// INIT — runs on page load
// ─────────────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
  // Immediate
  startClock();
  animateCountUp();

  // Async chart and data load
  await Promise.all([
    initTimelineChart(),
    initAttackChart(),
    initSeverityChart(),
    initIPChart(),
    initAlertsTable(),
    fetchLiveThreat(),  // first live feed item
  ]);

  // Simulate new live threats every 4 seconds
  setInterval(fetchLiveThreat, 4000);
});
