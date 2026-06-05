const messagesEl = document.querySelector("#messages");
const formEl = document.querySelector("#chatForm");
const promptEl = document.querySelector("#promptInput");
const domainEl = document.querySelector("#domainSelect");
const newChatButton = document.querySelector("#newChatButton");

const chatEndpoint = "/chat";

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function addMessage(role, text, sources = [], meta = {}) {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "U" : "L";

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  const paragraph = document.createElement("p");
  paragraph.innerHTML = escapeHtml(text).replaceAll("\n", "<br>");
  bubble.appendChild(paragraph);

  if (sources.length) {
    const list = document.createElement("div");
    list.className = "source-list";
    for (const source of sources.slice(0, 5)) {
      const chip = document.createElement("div");
      chip.className = "source-chip";
      const sourceText = [
        source.citation || source.title || source.source || source.source_file || "source",
      ]
        .filter(Boolean)
        .join(" | ");
      if (source.url) {
        const link = document.createElement("a");
        link.href = source.url;
        link.target = "_blank";
        link.rel = "noreferrer";
        link.textContent = sourceText;
        chip.appendChild(link);
      } else {
        chip.textContent = sourceText;
      }
      list.appendChild(chip);
    }
    bubble.appendChild(list);
  }

  article.appendChild(avatar);
  article.appendChild(bubble);
  messagesEl.appendChild(article);
  scrollToBottom();
  return article;
}

function setLoading(isLoading) {
  document.body.classList.toggle("is-loading", isLoading);
}

function normalizeResponse(data) {
  if (typeof data === "string") {
    return { answer: data, sources: [] };
  }

  return {
    answer:
      data.answer ||
      data.response ||
      data.message ||
      data.text ||
      "Không có nội dung trả lời trong response.",
    sources: data.sources || data.results || data.context || [],
    mode: data.mode || "",
    reason: data.reason || "",
    localUsed: Boolean(data.local_used),
    geminiUsed: Boolean(data.gemini_used),
  };
}

function buildErrorAnswer(question, error) {
  return [
    "Chưa kết nối được backend API.",
    "",
    `Câu hỏi vừa nhập: ${question}`,
    "",
    `Lỗi: ${error.message || error}`,
    "",
    "Hãy kiểm tra backend đang chạy tại http://localhost:8000.",
  ].join("\n");
}

async function callChatApi(payload) {
  const response = await fetch(chatEndpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }

  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return normalizeResponse(await response.json());
  }

  return normalizeResponse(await response.text());
}

function resizePrompt() {
  promptEl.style.height = "auto";
  promptEl.style.height = `${Math.min(promptEl.scrollHeight, 180)}px`;
}

newChatButton.addEventListener("click", () => {
  messagesEl.innerHTML = "";
  addMessage("assistant", "Chatbot Luật VN xin chào, rất vui được giúp đỡ bạn.");
  promptEl.focus();
});

promptEl.addEventListener("input", resizePrompt);

promptEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    formEl.requestSubmit();
  }
});

formEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = promptEl.value.trim();
  if (!question) {
    return;
  }

  const payload = {
    message: question,
    query: question,
    domain: domainEl.value,
    top_k: 5,
    gemini_fallback: true,
  };

  addMessage("user", question);
  promptEl.value = "";
  resizePrompt();
  setLoading(true);

  const pending = addMessage("assistant", "Đang xử lý...");
  try {
    const result = await callChatApi(payload);
    pending.remove();
    addMessage("assistant", result.answer, result.sources, {
      mode: result.mode,
      reason: result.reason,
      localUsed: result.localUsed,
      geminiUsed: result.geminiUsed,
    });
  } catch (error) {
    pending.remove();
    addMessage("assistant", buildErrorAnswer(question, error), [], { mode: "gemini_error" });
  } finally {
    setLoading(false);
    promptEl.focus();
  }
});

resizePrompt();
