const API_BASE = ""; // same origin as the FastAPI server serving this page

const sourceLangSel = document.getElementById("source-lang");
const targetLangSel = document.getElementById("target-lang");
const sourceText = document.getElementById("source-text");
const outputText = document.getElementById("output-text");
const charCount = document.getElementById("char-count");
const latencyNote = document.getElementById("latency-note");
const errorBanner = document.getElementById("error-banner");
const loadingIndicator = document.getElementById("loading-indicator");
const historyList = document.getElementById("history-list");

function showError(message) {
  errorBanner.textContent = message;
  errorBanner.classList.remove("hidden");
  setTimeout(() => errorBanner.classList.add("hidden"), 6000);
}

async function loadLanguages() {
  try {
    const res = await fetch(`${API_BASE}/languages`);
    if (!res.ok) throw new Error("Failed to load languages");
    const data = await res.json();
    for (const lang of data.languages) {
      const opt1 = document.createElement("option");
      opt1.value = lang.name;
      opt1.textContent = lang.name;
      sourceLangSel.appendChild(opt1);

      const opt2 = document.createElement("option");
      opt2.value = lang.name;
      opt2.textContent = lang.name;
      targetLangSel.appendChild(opt2);
    }
    sourceLangSel.value = "English";
    targetLangSel.value = "Hindi";
  } catch (err) {
    showError("[ERROR] Could not reach the backend. Is it running?");
  }
}

// --- Debounce helper -------------------------------------------------
const AUTO_TRANSLATE_DELAY_MS = 700;
let debounceTimer = null;

function scheduleAutoTranslate() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => performTranslate({ saveHistory: false }), AUTO_TRANSLATE_DELAY_MS);
}

sourceText.addEventListener("input", () => {
  charCount.textContent = `${sourceText.value.length} characters`;
  if (!sourceText.value.trim()) {
    outputText.textContent = "";
    latencyNote.textContent = "";
    clearTimeout(debounceTimer);
    return;
  }
  scheduleAutoTranslate();
});

sourceLangSel.addEventListener("change", () => {
  if (sourceText.value.trim()) performTranslate({ saveHistory: false });
});
targetLangSel.addEventListener("change", () => {
  if (sourceText.value.trim()) performTranslate({ saveHistory: false });
});

document.getElementById("clear-btn").addEventListener("click", () => {
  clearTimeout(debounceTimer);
  sourceText.value = "";
  charCount.textContent = "0 characters";
  outputText.textContent = "";
  latencyNote.textContent = "";
});

document.getElementById("swap-btn").addEventListener("click", () => {
  const srcVal = sourceLangSel.value;
  const tgtVal = targetLangSel.value;
  if ([...sourceLangSel.options].some(o => o.value === tgtVal) &&
      [...targetLangSel.options].some(o => o.value === srcVal)) {
    sourceLangSel.value = tgtVal;
    targetLangSel.value = srcVal;
  } else {
    showError("[ERROR] Unsupported language pair for swap.");
    return;
  }
  if (outputText.textContent.trim()) {
    sourceText.value = outputText.textContent;
    charCount.textContent = `${sourceText.value.length} characters`;
    outputText.textContent = "";
  }
  if (sourceText.value.trim()) performTranslate({ saveHistory: false });
});

document.getElementById("copy-btn").addEventListener("click", async () => {
  const text = outputText.textContent;
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    showError("[ERROR] Could not copy to clipboard.");
  }
});

document.getElementById("detect-btn").addEventListener("click", async () => {
  const text = sourceText.value.trim();
  if (!text) {
    showError("[ERROR] Empty input.");
    return;
  }
  try {
    const res = await fetch(`${API_BASE}/detect`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    if (data.detected_language) {
      sourceLangSel.value = data.detected_language;
      performTranslate({ saveHistory: false });
    } else {
      showError("Could not confidently detect the language - please select it manually.");
    }
  } catch {
    showError("[ERROR] Detection failed.");
  }
});

let requestSeq = 0;
const translateBtn = document.getElementById("translate-btn");

async function performTranslate({ saveHistory }) {
  const text = sourceText.value.trim();
  const source_language = sourceLangSel.value;
  const target_language = targetLangSel.value;

  if (!text) return;
  if (source_language === target_language) {
    showError("[ERROR] Source and target language must differ.");
    return;
  }

  const thisRequest = ++requestSeq;
  loadingIndicator.classList.remove("hidden");
  translateBtn.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/translate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, source_language, target_language, save_history: saveHistory }),
    });
    const data = await res.json();

    if (thisRequest !== requestSeq) return; // a newer request has since started

    if (!res.ok) {
      showError(data.detail || "[ERROR] Translation failed.");
      return;
    }
    outputText.textContent = data.translated_text;
    latencyNote.textContent = `${data.latency_ms} ms - ${data.chunks_used} chunk(s)`;
    if (saveHistory) loadHistory();
  } catch (err) {
    if (thisRequest === requestSeq) showError("[ERROR] Could not reach the backend.");
  } finally {
    if (thisRequest === requestSeq) {
      loadingIndicator.classList.add("hidden");
      translateBtn.disabled = false;
    }
  }
}

translateBtn.addEventListener("click", () => {
  clearTimeout(debounceTimer);
  if (!sourceText.value.trim()) { showError("[ERROR] Empty input."); return; }
  performTranslate({ saveHistory: true });
});

async function loadHistory() {
  try {
    const res = await fetch(`${API_BASE}/history`);
    const data = await res.json();
    historyList.innerHTML = "";
    for (const record of [...data.records].reverse().slice(0, 20)) {
      const li = document.createElement("li");
      li.innerHTML = `<div class="meta">${record.source_language} &rarr; ${record.target_language} - ${new Date(record.timestamp).toLocaleString()}</div>
        <div>${record.source_text}</div>
        <div><strong>${record.translated_text}</strong></div>`;
      historyList.appendChild(li);
    }
  } catch {
    // fail silently - history is a convenience feature
  }
}

document.getElementById("clear-history-btn").addEventListener("click", async () => {
  try {
    await fetch(`${API_BASE}/history`, { method: "DELETE" });
    loadHistory();
  } catch {
    showError("[ERROR] Could not clear history.");
  }
});

loadLanguages();
loadHistory();