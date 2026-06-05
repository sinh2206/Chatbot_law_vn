const messagesEl = document.querySelector("#messages");
const formEl = document.querySelector("#chatForm");
const promptEl = document.querySelector("#promptInput");
const endpointEl = document.querySelector("#endpointInput");
const saveEndpointButton = document.querySelector("#saveEndpointButton");
const domainEl = document.querySelector("#domainSelect");
const topKEl = document.querySelector("#topKInput");
const fallbackEl = document.querySelector("#fallbackToggle");
const newChatButton = document.querySelector("#newChatButton");

const endpointKey = "chatbot-law-vn-endpoint";
const savedEndpoint = localStorage.getItem(endpointKey);

if (savedEndpoint) {
  endpointEl.value = savedEndpoint;
}

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

function addMessage(role, text, sources = []) {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "U" : "L";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = `<p>${escapeHtml(text).replaceAll("\n", "<br>")}</p>`;

  if (sources.length) {
    const list = document.createElement("div");
    list.className = "source-list";
    for (const source of sources.slice(0, 5)) {
      const chip = document.createElement("div");
      chip.className = "source-chip";
      const sourceText = [
        source.source_file || source.source || "source",
        source.domain ? `domain=${source.domain}` : "",
        Number.isFinite(source.score) ? `score=${source.score.toFixed(4)}` : "",
      ]
        .filter(Boolean)
        .join(" | ");
      chip.textContent = sourceText;
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
      "Khong co noi dung tra loi trong response.",
    sources: data.sources || data.results || data.context || [],
  };
}

function buildDemoAnswer(question) {
  return [
    "Chua ket noi duoc endpoint web API.",
    "",
    `Cau hoi vua nhap: ${question}`,
    "",
    "Frontend da san sang gui POST den endpoint da cau hinh. Local RAG/Gemini fallback van can duoc expose qua mot API web rieng neu muon chat truc tiep tren trinh duyet.",
  ].join("\n");
}

async function callChatApi(payload) {
  const endpoint = endpointEl.value.trim();
  const response = await fetch(endpoint, {
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

saveEndpointButton.addEventListener("click", () => {
  localStorage.setItem(endpointKey, endpointEl.value.trim());
  saveEndpointButton.blur();
});

newChatButton.addEventListener("click", () => {
  messagesEl.innerHTML = "";
  addMessage("assistant", "Chao ban. Hay nhap cau hoi phap luat, minh se uu tien can cu noi bo truoc.");
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
    top_k: Number(topKEl.value || 5),
    gemini_fallback: fallbackEl.checked,
  };

  addMessage("user", question);
  promptEl.value = "";
  resizePrompt();
  setLoading(true);

  const pending = addMessage("assistant", "Dang xu ly...");
  try {
    const result = await callChatApi(payload);
    pending.remove();
    addMessage("assistant", result.answer, result.sources);
  } catch (error) {
    pending.remove();
    addMessage("assistant", buildDemoAnswer(question));
  } finally {
    setLoading(false);
    promptEl.focus();
  }
});

resizePrompt();
