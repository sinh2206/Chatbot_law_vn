const chartState = {
  timeseries: null,
  histogram: null,
  domain: null,
  level: null,
  runHistory: null,
};

const palette = {
  green: "#00534f",
  mint: "#3fa18a",
  amber: "#cc7c1f",
  blue: "#2968a7",
  red: "#b84040",
};

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: HTTP ${response.status}`);
  }
  return response.json();
}

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatMs(value) {
  return `${Math.round(value)} ms`;
}

function renderSummary(summary) {
  const cards = [
    { label: "Tổng câu hỏi", value: summary.total_questions ?? 0 },
    { label: "Avg latency", value: formatMs(summary.avg_response_ms ?? 0) },
    { label: "P95 latency", value: formatMs(summary.p95_response_ms ?? 0) },
    { label: "Fallback rate", value: formatPercent(summary.fallback_rate ?? 0) },
    { label: "Multi-domain rate", value: formatPercent(summary.multi_domain_rate ?? 0) },
    { label: "RAG hit rate", value: formatPercent(summary.rag_hit_rate ?? 0) },
    { label: "Throughput 24h", value: `${(summary.throughput_qph_24h ?? 0).toFixed(2)} q/h` },
    {
      label: "Hữu ích",
      value: formatPercent(summary.feedback?.helpful_ratio ?? 0),
    },
    {
      label: "Avg rating",
      value: (summary.feedback?.avg_rating ?? 0).toFixed(2),
    },
    { label: "RAG queries", value: summary.rag_query_count ?? 0 },
    { label: "Max latency", value: formatMs(summary.max_response_ms ?? 0) },
    {
      label: "Không hữu ích",
      value: formatPercent(summary.feedback?.unhelpful_ratio ?? 0),
    },
  ];

  const container = document.getElementById("summaryCards");
  container.innerHTML = "";

  cards.forEach((item) => {
    const node = document.createElement("article");
    node.className = "metric-card";
    node.innerHTML = `<div class="label">${item.label}</div><div class="value">${item.value}</div>`;
    container.appendChild(node);
  });
}

function renderAlerts(alerts) {
  const panel = document.getElementById("alertsPanel");
  panel.innerHTML = "";

  if (!alerts.length) {
    return;
  }

  alerts.forEach((alert) => {
    const node = document.createElement("div");
    node.className = `alert-item ${alert.level}`;
    node.textContent = `${alert.message}: ${Number(alert.value).toFixed(2)} (ngưỡng ${alert.threshold})`;
    panel.appendChild(node);
  });
}

function upsertChart(key, canvasId, config) {
  if (chartState[key]) {
    chartState[key].destroy();
  }
  chartState[key] = new Chart(document.getElementById(canvasId), config);
}

function renderTimeseries(items) {
  const labels = items.map((item) => item.day);
  const latency = items.map((item) => Number(item.avg_response_ms || 0));
  const fallbackRate = items.map((item) => Number(item.fallback_rate || 0) * 100);

  upsertChart("timeseries", "timeseriesChart", {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Avg latency (ms)",
          data: latency,
          borderColor: palette.green,
          backgroundColor: "rgba(0, 83, 79, 0.15)",
          yAxisID: "y",
          tension: 0.25,
        },
        {
          label: "Fallback rate (%)",
          data: fallbackRate,
          borderColor: palette.amber,
          backgroundColor: "rgba(204, 124, 31, 0.15)",
          yAxisID: "y1",
          tension: 0.25,
        },
      ],
    },
    options: {
      responsive: true,
      interaction: { mode: "index", intersect: false },
      scales: {
        y: { beginAtZero: true, title: { display: true, text: "Latency (ms)" } },
        y1: {
          beginAtZero: true,
          position: "right",
          grid: { drawOnChartArea: false },
          title: { display: true, text: "Fallback (%)" },
        },
      },
    },
  });
}

function renderHistogram(items) {
  upsertChart("histogram", "histogramChart", {
    type: "bar",
    data: {
      labels: items.map((item) => item.bucket),
      datasets: [
        {
          label: "Số câu",
          data: items.map((item) => item.count),
          backgroundColor: "rgba(63, 161, 138, 0.45)",
          borderColor: palette.mint,
          borderWidth: 1,
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        y: { beginAtZero: true },
      },
    },
  });
}

function renderDomainBreakdown(items) {
  upsertChart("domain", "domainChart", {
    type: "bar",
    data: {
      labels: items.map((item) => item.domain),
      datasets: [
        {
          label: "Fallback rate (%)",
          data: items.map((item) => Number(item.fallback_rate || 0) * 100),
          backgroundColor: "rgba(184, 64, 64, 0.35)",
          borderColor: palette.red,
          borderWidth: 1,
        },
        {
          label: "Retrieval hit rate (%)",
          data: items.map((item) => Number(item.retrieval_hit_rate || 0) * 100),
          backgroundColor: "rgba(41, 104, 167, 0.35)",
          borderColor: palette.blue,
          borderWidth: 1,
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        y: { beginAtZero: true, max: 100 },
      },
    },
  });
}

function renderLevelAccuracy(latestRun) {
  const levels = latestRun?.levels || [];
  upsertChart("level", "levelChart", {
    type: "bar",
    data: {
      labels: levels.map((item) => `Level ${item.level}`),
      datasets: [
        {
          label: "Accuracy (%)",
          data: levels.map((item) => Number(item.accuracy || 0) * 100),
          backgroundColor: "rgba(0, 83, 79, 0.35)",
          borderColor: palette.green,
          borderWidth: 1,
        },
        {
          label: "Avg score (x10)",
          data: levels.map((item) => Number(item.avg_score || 0) * 10),
          backgroundColor: "rgba(63, 161, 138, 0.35)",
          borderColor: palette.mint,
          borderWidth: 1,
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        y: { beginAtZero: true, max: 100 },
      },
    },
  });
}

function renderRunHistory(items) {
  const ordered = [...items].reverse();
  upsertChart("runHistory", "runHistoryChart", {
    type: "bar",
    data: {
      labels: ordered.map((item) => item.run_id.slice(0, 6)),
      datasets: [
        {
          label: "Accuracy (%)",
          data: ordered.map((item) => Number(item.accuracy || 0) * 100),
          backgroundColor: "rgba(41, 104, 167, 0.35)",
          borderColor: palette.blue,
          borderWidth: 1,
        },
        {
          label: "Fallback (%)",
          data: ordered.map((item) => Number(item.fallback_rate || 0) * 100),
          backgroundColor: "rgba(204, 124, 31, 0.35)",
          borderColor: palette.amber,
          borderWidth: 1,
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        y: { beginAtZero: true, max: 100 },
      },
    },
  });
}

function renderLowFeedback(items) {
  const tbody = document.querySelector("#lowFeedbackTable tbody");
  tbody.innerHTML = "";

  if (!items.length) {
    const tr = document.createElement("tr");
    tr.innerHTML = '<td colspan="5">Chưa có dữ liệu phản hồi tiêu cực.</td>';
    tbody.appendChild(tr);
    return;
  }

  items.forEach((item) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${item.updated_at || "-"}</td>
      <td>${(item.question || "").slice(0, 180)}</td>
      <td>${item.fallback_used ? "Có" : "Không"}</td>
      <td>${item.rating ?? (item.helpful === false ? "Không hữu ích" : "-")}</td>
      <td>${item.comment || "-"}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderTestHistory(items) {
  const tbody = document.querySelector("#testHistoryTable tbody");
  tbody.innerHTML = "";

  if (!items.length) {
    const tr = document.createElement("tr");
    tr.innerHTML = '<td colspan="6">Chưa có lịch sử test.</td>';
    tbody.appendChild(tr);
    return;
  }

  items.forEach((item) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${item.run_id.slice(0, 8)}</td>
      <td>${item.mode}</td>
      <td>${item.total_cases}</td>
      <td>${formatPercent(item.accuracy || 0)}</td>
      <td>${Number(item.avg_score || 0).toFixed(2)}</td>
      <td>${formatMs(item.avg_response_ms || 0)}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderFrequentFailures(items) {
  const tbody = document.querySelector("#frequentFailuresTable tbody");
  tbody.innerHTML = "";

  if (!items.length) {
    const tr = document.createElement("tr");
    tr.innerHTML = '<td colspan="4">Chưa có dữ liệu fail test.</td>';
    tbody.appendChild(tr);
    return;
  }

  items.forEach((item) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${item.case_id}</td>
      <td>${(item.question || "").slice(0, 180)}</td>
      <td>${item.fail_count}</td>
      <td>${Number(item.avg_score || 0).toFixed(2)}</td>
    `;
    tbody.appendChild(tr);
  });
}

async function loadDashboard() {
  const [summary, timeseries, histogram, domainBreakdown, lowFeedback, latestTest, testHistory, frequentFailures, alerts] =
    await Promise.all([
      fetchJson("/metrics/summary"),
      fetchJson("/metrics/timeseries?days=14"),
      fetchJson("/metrics/latency-histogram"),
      fetchJson("/metrics/domain-breakdown?days=30"),
      fetchJson("/metrics/low-feedback?limit=15"),
      fetchJson("/metrics/tests/latest"),
      fetchJson("/metrics/tests/history?limit=20"),
      fetchJson("/metrics/tests/frequent-failures?limit=20"),
      fetchJson("/metrics/alerts"),
    ]);

  renderSummary(summary);
  renderAlerts(alerts.items || []);
  renderTimeseries(timeseries.items || []);
  renderHistogram(histogram.items || []);
  renderDomainBreakdown(domainBreakdown.items || []);
  renderLevelAccuracy(latestTest.available ? latestTest : { levels: [] });
  renderRunHistory(testHistory.items || []);
  renderLowFeedback(lowFeedback.items || []);
  renderTestHistory(testHistory.items || []);
  renderFrequentFailures(frequentFailures.items || []);
}

async function refresh() {
  try {
    await loadDashboard();
  } catch (error) {
    const panel = document.getElementById("alertsPanel");
    panel.innerHTML = `<div class="alert-item critical">Lỗi tải dashboard: ${error.message}</div>`;
  }
}

document.getElementById("refreshBtn").addEventListener("click", refresh);
refresh();
setInterval(refresh, 30000);
