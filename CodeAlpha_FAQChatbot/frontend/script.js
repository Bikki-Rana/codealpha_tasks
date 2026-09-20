const messagesEl = document.getElementById("messages");
const formEl = document.getElementById("chatForm");
const inputEl = document.getElementById("messageInput");
const loadingEl = document.getElementById("loadingIndicator");
const clearBtn = document.getElementById("clearBtn");
const suggestionsEl = document.getElementById("suggestions");

// Same-origin by default since the backend serves this file directly
// (see README - run via `uvicorn backend.main:app`). Change this if you
// ever host the frontend separately from the API.
const API_BASE = window.location.origin;

// One session per browser tab. Conversation history lives server-side,
// keyed by this id - see backend/chatbot.py for what "history" does and
// does not do (it's retrieval, not a stateful LLM conversation).
function getSessionId() {
  let id = sessionStorage.getItem("faq_session_id");
  if (!id) {
    id = crypto.randomUUID();
    sessionStorage.setItem("faq_session_id", id);
  }
  return id;
}

const sessionId = getSessionId();

function addBubble(role, text, meta) {
  const bubble = document.createElement("div");
  bubble.className = `bubble ${role}`;
  bubble.textContent = text;

  if (meta) {
    const metaEl = document.createElement("span");
    metaEl.className = "meta";
    metaEl.textContent = meta;
    bubble.appendChild(metaEl);
  }

  messagesEl.appendChild(bubble);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function setLoading(isLoading) {
  loadingEl.classList.toggle("hidden", !isLoading);
  document.getElementById("sendBtn").disabled = isLoading;
}

async function sendMessage(message) {
  if (!message.trim()) return;

  addBubble("user", message);
  inputEl.value = "";
  setLoading(true);

  try {
    const response = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: sessionId }),
    });

    if (!response.ok) {
      throw new Error(`Server returned ${response.status}`);
    }

    const data = await response.json();
    const meta = data.category
      ? `${data.category} - confidence ${(data.confidence * 100).toFixed(0)}%`
      : null;

    addBubble("bot", data.answer, meta);
  } catch (err) {
    console.error(err);
    addBubble("bot", "Something went wrong reaching the server. Is the backend running?");
  } finally {
    setLoading(false);
  }
}

formEl.addEventListener("submit", (e) => {
  e.preventDefault();
  sendMessage(inputEl.value);
});

suggestionsEl.addEventListener("click", (e) => {
  if (e.target.classList.contains("chip")) {
    sendMessage(e.target.textContent);
  }
});

clearBtn.addEventListener("click", async () => {
  try {
    await fetch(`${API_BASE}/clear`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId }),
    });
  } catch (err) {
    console.error("Failed to clear server-side history:", err);
  }
  messagesEl.innerHTML = "";
});

// Greet on load
addBubble("bot", "Hi! Ask me anything about JRU admissions, fees, exams, hostel, library, or placements.");
