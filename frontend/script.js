const STORAGE_KEY = "legal_chat_history_v2";
const API_BASE = window.location.protocol.startsWith("http")
  ? ""
  : "http://localhost:8000";

const chatHistoryEl = document.getElementById("chatHistory");
const chatFormEl = document.getElementById("chatForm");
const questionInputEl = document.getElementById("questionInput");
const sendBtnEl = document.getElementById("sendBtn");
const clearBtnEl = document.getElementById("clearBtn");
const domainSelectEl = document.getElementById("domainSelect");
const statusBadgeEl = document.getElementById("statusBadge");
const messageTemplateEl = document.getElementById("messageTemplate");

let pendingTypingNode = null;

function createMessageElement(role, content) {
  const node = messageTemplateEl.content.firstElementChild.cloneNode(true);
  node.classList.add(role);
  node.querySelector(".bubble").textContent = content;
  return node;
}

function appendMessage(role, content, extras = null) {
  const node = createMessageElement(role, content);

  if (extras && role === "assistant") {
    const bubble = node.querySelector(".bubble");
    const meta = document.createElement("div");
    meta.className = "meta-block";

    const details = [];
    if (typeof extras.responseTimeMs === "number") {
      details.push(`Latency: ${extras.responseTimeMs} ms`);
    }
    if (extras.fallbackUsed) {
      details.push("Mode: web fallback");
    }
    if (details.length) {
      const line = document.createElement("div");
      line.textContent = details.join(" | ");
      meta.appendChild(line);
    }

    const citationLines = (extras.citations || []).slice(0, 8);
    if (citationLines.length) {
      const citationBlock = document.createElement("div");
      citationBlock.textContent = `Trích dẫn: ${citationLines.join(" | ")}`;
      meta.appendChild(citationBlock);
    }

    const sources = extras.sources || [];
    if (sources.length) {
      const sourceContainer = document.createElement("div");
      sourceContainer.textContent = "Nguồn web:";
      meta.appendChild(sourceContainer);

      sources.slice(0, 5).forEach((source) => {
        const row = document.createElement("div");
        const link = document.createElement("a");
        link.href = source.url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = source.title || source.url;
        row.appendChild(link);
        meta.appendChild(row);
      });
    }

    if (meta.childNodes.length) {
      bubble.appendChild(meta);
    }

    if (extras.interactionId) {
      const feedbackRow = document.createElement("div");
      feedbackRow.className = "feedback-row";
      feedbackRow.dataset.interactionId = extras.interactionId;

      const helpfulBtn = document.createElement("button");
      helpfulBtn.type = "button";
      helpfulBtn.className = "feedback-btn";
      helpfulBtn.dataset.helpful = "true";
      helpfulBtn.textContent = "Hữu ích";

      const unhelpfulBtn = document.createElement("button");
      unhelpfulBtn.type = "button";
      unhelpfulBtn.className = "feedback-btn";
      unhelpfulBtn.dataset.helpful = "false";
      unhelpfulBtn.textContent = "Không hữu ích";

      feedbackRow.appendChild(helpfulBtn);
      feedbackRow.appendChild(unhelpfulBtn);
      bubble.appendChild(feedbackRow);
    }
  }

  chatHistoryEl.appendChild(node);
  scrollToBottom();
  persistHistory();
}

function scrollToBottom() {
  chatHistoryEl.scrollTop = chatHistoryEl.scrollHeight;
}

function setLoading(isLoading) {
  sendBtnEl.disabled = isLoading;
  questionInputEl.disabled = isLoading;

  if (isLoading) {
    pendingTypingNode = createMessageElement("assistant", "Đang xử lý");
    pendingTypingNode.querySelector(".bubble").classList.add("typing");
    chatHistoryEl.appendChild(pendingTypingNode);
    scrollToBottom();
  } else if (pendingTypingNode) {
    pendingTypingNode.remove();
    pendingTypingNode = null;
  }
}

async function sendQuestion(question) {
  const payload = {
    question,
    domain: domainSelectEl.value || null,
  };

  const response = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const errorJson = await response.json();
      detail = errorJson.detail || detail;
    } catch (_err) {
      // Ignore JSON parsing errors for error payload.
    }
    throw new Error(detail);
  }

  return response.json();
}

async function sendFeedback(interactionId, helpful) {
  const response = await fetch(`${API_BASE}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      interaction_id: interactionId,
      helpful,
    }),
  });

  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const errorJson = await response.json();
      detail = errorJson.detail || detail;
    } catch (_err) {
      // Keep default detail.
    }
    throw new Error(detail);
  }
}

function persistHistory() {
  const snapshot = [];
  chatHistoryEl.querySelectorAll(".message").forEach((messageNode) => {
    if (messageNode === pendingTypingNode) {
      return;
    }

    const role = messageNode.classList.contains("user")
      ? "user"
      : messageNode.classList.contains("error")
        ? "error"
        : "assistant";

    snapshot.push({
      role,
      html: messageNode.innerHTML,
    });
  });

  localStorage.setItem(STORAGE_KEY, JSON.stringify(snapshot));
}

function restoreHistory() {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) {
    appendMessage(
      "assistant",
      "Xin chào, bạn có thể hỏi về doanh nghiệp, hộ tịch, căn cước, đất đai và thuế."
    );
    return;
  }

  try {
    const items = JSON.parse(raw);
    if (!Array.isArray(items)) {
      throw new Error("invalid storage");
    }

    items.forEach((item) => {
      if (!item || !item.role || !item.html) {
        return;
      }
      const node = document.createElement("article");
      node.className = `message ${item.role}`;
      node.innerHTML = item.html;
      chatHistoryEl.appendChild(node);
    });

    scrollToBottom();
  } catch (_err) {
    localStorage.removeItem(STORAGE_KEY);
  }
}

function clearHistory() {
  chatHistoryEl.innerHTML = "";
  localStorage.removeItem(STORAGE_KEY);
  appendMessage(
    "assistant",
    "Đã bắt đầu phiên chat mới. Hãy nhập câu hỏi pháp lý của bạn."
  );
}

async function checkConnection() {
  try {
    const response = await fetch(`${API_BASE}/health`, { method: "GET" });
    if (!response.ok) {
      throw new Error("unhealthy");
    }
    statusBadgeEl.textContent = "Online";
    statusBadgeEl.classList.remove("offline");
    statusBadgeEl.classList.add("online");
  } catch (_err) {
    statusBadgeEl.textContent = "Offline";
    statusBadgeEl.classList.remove("online");
    statusBadgeEl.classList.add("offline");
  }
}

chatFormEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInputEl.value.trim();
  if (!question) {
    return;
  }

  appendMessage("user", question);
  questionInputEl.value = "";
  setLoading(true);

  try {
    const data = await sendQuestion(question);
    appendMessage("assistant", data.answer, {
      interactionId: data.interaction_id,
      citations: data.citations || [],
      sources: data.sources || [],
      fallbackUsed: data.fallback_used || false,
      responseTimeMs: data.response_time_ms,
    });
  } catch (error) {
    appendMessage("error", `Không thể lấy phản hồi từ backend: ${error.message}`);
  } finally {
    setLoading(false);
    questionInputEl.focus();
    checkConnection();
  }
});

chatHistoryEl.addEventListener("click", async (event) => {
  const target = event.target;
  if (!(target instanceof HTMLButtonElement)) {
    return;
  }
  if (!target.classList.contains("feedback-btn")) {
    return;
  }

  const row = target.closest(".feedback-row");
  if (!row || !row.dataset.interactionId || row.dataset.sent === "true") {
    return;
  }

  const helpful = target.dataset.helpful === "true";
  try {
    await sendFeedback(row.dataset.interactionId, helpful);
    row.dataset.sent = "true";
    row.querySelectorAll(".feedback-btn").forEach((button) => {
      button.disabled = true;
      if (button === target) {
        button.classList.add("active");
      }
    });
    persistHistory();
  } catch (error) {
    appendMessage("error", `Không thể gửi phản hồi: ${error.message}`);
  }
});

clearBtnEl.addEventListener("click", clearHistory);

questionInputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatFormEl.requestSubmit();
  }
});

restoreHistory();
checkConnection();
setInterval(checkConnection, 30000);
