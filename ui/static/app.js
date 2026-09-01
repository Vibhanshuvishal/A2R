// A2R v2 Single-Page Application Client Logic

let activeSessionId = "";
let currentEventSource = null;

// DOM Elements
const sessionListEl = document.getElementById("session-list");
const chatMessagesEl = document.getElementById("chat-messages");
const welcomeScreenEl = document.getElementById("welcome-screen");
const queryInputEl = document.getElementById("query-input");
const chatFormEl = document.getElementById("chat-form");
const sendBtnEl = document.getElementById("send-btn");
const typingIndicatorEl = document.getElementById("typing-indicator");
const typingTextEl = document.getElementById("typing-text");
const newChatBtnEl = document.getElementById("new-chat-btn");
const currentChatTitleEl = document.getElementById("current-chat-title");
const renameChatBtnEl = document.getElementById("rename-chat-btn");
const clearCacheBtnEl = document.getElementById("clear-cache-btn");
const cacheStatsEl = document.getElementById("cache-stats");
const cacheSizeEl = document.getElementById("cache-size");
const modelNameEl = document.getElementById("model-name");
const sidebarToggleEl = document.getElementById("sidebar-toggle");
const sidebarEl = document.getElementById("sidebar");
const toastEl = document.getElementById("toast");

// Toast Notification
function showToast(message) {
  toastEl.textContent = message;
  toastEl.classList.add("show");
  setTimeout(() => toastEl.classList.remove("show"), 3500);
}

// Auto-expand input textarea
queryInputEl.addEventListener("input", function () {
  this.style.height = "auto";
  this.style.height = Math.min(this.scrollHeight, 160) + "px";
});

// Keydown Enter to send (Shift+Enter for newline)
queryInputEl.addEventListener("keydown", function (e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    chatFormEl.dispatchEvent(new Event("submit"));
  }
});

// Sidebar mobile toggle
sidebarToggleEl.addEventListener("click", () => {
  sidebarEl.classList.toggle("open");
});

// Fetch & Update Cache & Model Stats
async function refreshStats() {
  try {
    const healthRes = await fetch("/health");
    if (healthRes.ok) {
      const health = await healthRes.json();
      if (health.model_status) {
        modelNameEl.textContent = `${health.model_status.model} (${health.model_status.detail})`;
      }
      if (health.cache) {
        cacheStatsEl.textContent = `${health.cache.hits} hits (${Math.round(health.cache.hit_rate * 100)}%)`;
        cacheSizeEl.textContent = `${health.cache.cache_size} entries`;
      }
    }
  } catch (err) {
    console.error("Failed to refresh health/stats:", err);
  }
}

// Load Sessions
async function loadSessions() {
  try {
    const res = await fetch("/sessions");
    if (!res.ok) return;
    const sessions = await res.json();
    sessionListEl.innerHTML = "";

    if (sessions.length === 0) {
      // Create first default session
      await createNewSession("Initial Chat");
      return;
    }

    sessions.forEach((sess) => {
      const item = document.createElement("div");
      item.className = `session-item ${sess.id === activeSessionId ? "active" : ""}`;
      item.dataset.id = sess.id;
      item.innerHTML = `
        <span class="title-text">${escapeHtml(sess.title)}</span>
        <button class="delete-btn" title="Delete conversation">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="3 6 5 6 21 6"></polyline>
            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
          </svg>
        </button>
      `;

      item.addEventListener("click", (e) => {
        if (e.target.closest(".delete-btn")) {
          deleteSession(sess.id);
          return;
        }
        switchSession(sess.id);
      });

      sessionListEl.appendChild(item);
    });

    if (!activeSessionId && sessions.length > 0) {
      switchSession(sessions[0].id);
    }
  } catch (err) {
    console.error("Failed to load sessions:", err);
  }
}

// Create New Session
async function createNewSession(title = "New Conversation") {
  try {
    const res = await fetch("/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    });
    if (!res.ok) return;
    const data = await res.json();
    activeSessionId = data.id;
    await loadSessions();
    await switchSession(data.id);
  } catch (err) {
    console.error("Failed to create session:", err);
  }
}

// Switch Session
async function switchSession(sessionId) {
  if (currentEventSource) {
    currentEventSource.close();
    currentEventSource = null;
  }
  activeSessionId = sessionId;

  // Highlight active
  document.querySelectorAll(".session-item").forEach((el) => {
    el.classList.toggle("active", el.dataset.id === sessionId);
  });

  try {
    const res = await fetch(`/sessions/${sessionId}`);
    if (!res.ok) return;
    const data = await res.json();
    currentChatTitleEl.textContent = data.session.title || "Conversation";

    chatMessagesEl.innerHTML = "";
    if (!data.messages || data.messages.length === 0) {
      chatMessagesEl.appendChild(welcomeScreenEl);
      bindPills();
      return;
    }

    data.messages.forEach((msg) => {
      appendMessage(msg.role, msg.content, msg.metadata, msg.query_id);
    });
    scrollToBottom();
  } catch (err) {
    console.error("Failed to switch session:", err);
  }
}

// Delete Session
async function deleteSession(sessionId) {
  try {
    const res = await fetch(`/sessions/${sessionId}`, { method: "DELETE" });
    if (!res.ok) return;
    if (activeSessionId === sessionId) {
      activeSessionId = "";
    }
    await loadSessions();
  } catch (err) {
    console.error("Failed to delete session:", err);
  }
}

// Rename Session
renameChatBtnEl.addEventListener("click", async () => {
  if (!activeSessionId) return;
  const currentTitle = currentChatTitleEl.textContent;
  const newTitle = prompt("Enter new title for conversation:", currentTitle);
  if (newTitle && newTitle.trim()) {
    try {
      const res = await fetch(`/sessions/${activeSessionId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: newTitle.trim() }),
      });
      if (res.ok) {
        currentChatTitleEl.textContent = newTitle.trim();
        await loadSessions();
      }
    } catch (err) {
      console.error("Failed to rename session:", err);
    }
  }
});

// Clear Cache Button
clearCacheBtnEl.addEventListener("click", async () => {
  try {
    const res = await fetch("/cache/clear", { method: "POST" });
    if (res.ok) {
      showToast("Semantic query cache flushed.");
      await refreshStats();
    }
  } catch (err) {
    console.error("Failed to clear cache:", err);
  }
});

// New Chat Button
newChatBtnEl.addEventListener("click", () => createNewSession("New Conversation"));

// Pill buttons on welcome screen
function bindPills() {
  document.querySelectorAll(".pill-btn").forEach((btn) => {
    btn.onclick = () => {
      queryInputEl.value = btn.dataset.query;
      chatFormEl.dispatchEvent(new Event("submit"));
    };
  });
}
bindPills();

// Helper to escape HTML
function escapeHtml(str) {
  return (str || "").replace(/[&<>"']/g, (m) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[m]);
}

function scrollToBottom() {
  chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight;
}

// Append message bubble
function appendMessage(role, content, metadata = {}, queryId = "") {
  if (welcomeScreenEl.parentNode === chatMessagesEl) {
    welcomeScreenEl.remove();
  }

  const row = document.createElement("div");
  row.className = `message-row ${role === "user" ? "user-row" : "assistant-row"}`;

  const avatar = document.createElement("div");
  avatar.className = `avatar ${role === "user" ? "user-avatar" : "assistant-avatar"}`;
  avatar.textContent = role === "user" ? "U" : "A2R";

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";

  if (role === "assistant" && metadata.source_badge) {
    const colorClass = `badge-${metadata.source_badge_color || "green"}`;
    const badge = document.createElement("div");
    badge.className = `badge-tag ${colorClass}`;
    badge.textContent = metadata.source_badge;
    bubble.appendChild(badge);
  }

  const contentEl = document.createElement("div");
  contentEl.className = "message-text";
  contentEl.textContent = content;
  bubble.appendChild(contentEl);

  // Citations if any
  if (role === "assistant" && metadata.sources && metadata.sources.length > 0) {
    const citationsBox = document.createElement("div");
    citationsBox.className = "citations-box";
    citationsBox.innerHTML = `
      <div class="citations-header">📚 Referenced Internal Documents (${metadata.sources.length})</div>
      <div class="citations-list">
        ${metadata.sources.map((s) => `<div class="citation-item">${escapeHtml(s)}</div>`).join("")}
      </div>
    `;
    bubble.appendChild(citationsBox);
  }

  // Feedback buttons
  if (role === "assistant" && queryId) {
    const actions = document.createElement("div");
    actions.className = "message-actions";
    actions.innerHTML = `
      <button class="feedback-btn accept-btn" title="Helpful answer">✓ Helpful</button>
      <button class="feedback-btn reject-btn" title="Not helpful answer">✗ Not helpful</button>
    `;

    const acceptBtn = actions.querySelector(".accept-btn");
    const rejectBtn = actions.querySelector(".reject-btn");

    acceptBtn.onclick = () => submitFeedback(queryId, "accept", acceptBtn, rejectBtn);
    rejectBtn.onclick = () => submitFeedback(queryId, "reject", rejectBtn, acceptBtn);
    bubble.appendChild(actions);
  }

  row.appendChild(avatar);
  row.appendChild(bubble);
  chatMessagesEl.appendChild(row);
  scrollToBottom();
  return { row, bubble, contentEl };
}

// Submit Reinforcement Feedback
async function submitFeedback(queryId, signal, activeBtn, otherBtn) {
  try {
    const res = await fetch("/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query_id: queryId, signal }),
    });
    if (res.ok) {
      const data = await res.json();
      activeBtn.classList.add(signal === "accept" ? "accepted" : "rejected");
      otherBtn.disabled = true;
      activeBtn.disabled = true;
      showToast(`Router learned from feedback! New weight: ${data.new_weight?.toFixed(2) || "updated"}`);
    } else {
      showToast("Feedback already recorded for this query.");
    }
  } catch (err) {
    console.error("Feedback failed:", err);
  }
}

// Handle Query Submission with Token Streaming
chatFormEl.addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = queryInputEl.value.trim();
  if (!query) return;

  if (!activeSessionId) {
    await createNewSession("New Conversation");
  }

  // Render User Message
  appendMessage("user", query);
  queryInputEl.value = "";
  queryInputEl.style.height = "auto";
  sendBtnEl.disabled = true;

  // Show Typing Indicator
  typingTextEl.textContent = "Analyzing query & routing...";
  typingIndicatorEl.style.display = "flex";
  scrollToBottom();

  let assistantBubble = null;
  let assistantTextEl = null;
  let accumulatedText = "";
  let queryResult = null;

  // Connect to SSE stream
  const url = `/query-stream?query=${encodeURIComponent(query)}&session_id=${encodeURIComponent(activeSessionId)}`;
  const es = new EventSource(url);
  currentEventSource = es;

  es.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data);

      if (payload.event === "status") {
        typingTextEl.textContent = payload.message || "Processing...";
      } else if (payload.event === "route") {
        typingTextEl.textContent = `Routing to ${payload.pipeline}...`;
      } else if (payload.event === "cache_hit") {
        typingTextEl.textContent = `Instant Semantic Cache Hit (${Math.round(payload.similarity * 100)}% match)...`;
      } else if (payload.event === "token") {
        if (!assistantBubble) {
          typingIndicatorEl.style.display = "none";
          const res = appendMessage("assistant", "");
          assistantBubble = res.bubble;
          assistantTextEl = res.contentEl;
        }
        accumulatedText += payload.token;
        assistantTextEl.textContent = accumulatedText;
        scrollToBottom();
      } else if (payload.event === "done") {
        queryResult = payload.result;
        es.close();
        currentEventSource = null;
        sendBtnEl.disabled = false;
        typingIndicatorEl.style.display = "none";

        if (queryResult) {
          // If no tokens were streamed yet (e.g. immediate return)
          if (!assistantBubble) {
            const res = appendMessage("assistant", queryResult.answer, {
              source_badge: queryResult.source_badge,
              source_badge_color: queryResult.source_badge_color,
              sources: queryResult.sources,
            }, queryResult.query_id);
          } else {
            // Add badge and citations to existing bubble
            if (queryResult.source_badge) {
              const colorClass = `badge-${queryResult.source_badge_color || "green"}`;
              const badge = document.createElement("div");
              badge.className = `badge-tag ${colorClass}`;
              badge.textContent = queryResult.source_badge;
              assistantBubble.insertBefore(badge, assistantTextEl);
            }

            if (queryResult.sources && queryResult.sources.length > 0) {
              const citationsBox = document.createElement("div");
              citationsBox.className = "citations-box";
              citationsBox.innerHTML = `
                <div class="citations-header">📚 Referenced Internal Documents (${queryResult.sources.length})</div>
                <div class="citations-list">
                  ${queryResult.sources.map((s) => `<div class="citation-item">${escapeHtml(s)}</div>`).join("")}
                </div>
              `;
              assistantBubble.appendChild(citationsBox);
            }

            // Feedback buttons
            if (queryResult.query_id) {
              const actions = document.createElement("div");
              actions.className = "message-actions";
              actions.innerHTML = `
                <button class="feedback-btn accept-btn" title="Helpful answer">✓ Helpful</button>
                <button class="feedback-btn reject-btn" title="Not helpful answer">✗ Not helpful</button>
              `;
              const acceptBtn = actions.querySelector(".accept-btn");
              const rejectBtn = actions.querySelector(".reject-btn");
              acceptBtn.onclick = () => submitFeedback(queryResult.query_id, "accept", acceptBtn, rejectBtn);
              rejectBtn.onclick = () => submitFeedback(queryResult.query_id, "reject", rejectBtn, acceptBtn);
              assistantBubble.appendChild(actions);
            }
          }
        }
        refreshStats();
        loadSessions();
      } else if (payload.event === "error") {
        es.close();
        currentEventSource = null;
        sendBtnEl.disabled = false;
        typingIndicatorEl.style.display = "none";
        appendMessage("assistant", `⚠️ Error: ${payload.message}`);
      }
    } catch (err) {
      console.error("SSE parse error:", err);
    }
  };

  es.onerror = (err) => {
    console.error("SSE error:", err);
    es.close();
    currentEventSource = null;
    sendBtnEl.disabled = false;
    typingIndicatorEl.style.display = "none";
    if (!assistantBubble) {
      appendMessage("assistant", "⚠️ Connection error or generation interrupted.");
    }
  };
});

// Initial boot
(async () => {
  await loadSessions();
  await refreshStats();
})();
