const API = "";

let cameras = [];
let activeCameraId = null;
const chatHistory = {}; // camera_id -> [{role, text, ts}]

const el = {
  cameraList: document.getElementById("camera-list"),
  viewerTitle: document.getElementById("viewer-title"),
  viewerLocation: document.getElementById("viewer-location"),
  viewerStream: document.getElementById("viewer-stream"),
  viewerPlaceholder: document.getElementById("viewer-placeholder"),
  chatLog: document.getElementById("chat-log"),
  chatEmpty: document.getElementById("chat-empty"),
  promptChips: document.getElementById("prompt-chips"),
  chatForm: document.getElementById("chat-form"),
  chatInput: document.getElementById("chat-input"),
  chatSend: document.getElementById("chat-send"),
  modelStatus: document.getElementById("model-status"),
  clock: document.getElementById("clock"),
};

function tickClock() {
  const now = new Date();
  el.clock.textContent = now.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}
setInterval(tickClock, 1000);
tickClock();

async function pollStatus() {
  try {
    const res = await fetch(`${API}/api/status`);
    const data = await res.json();
    if (data.model_ready) {
      el.modelStatus.className = "pill pill-ready";
      el.modelStatus.innerHTML = `<span class="dot"></span> Model ready`;
    } else if (data.model_error) {
      el.modelStatus.className = "pill pill-error";
      el.modelStatus.innerHTML = `<span class="dot"></span> Model error`;
    } else {
      el.modelStatus.className = "pill pill-loading";
      el.modelStatus.innerHTML = `<span class="dot"></span> Loading model…`;
    }
  } catch (e) {
    // ignore transient errors
  }
}
setInterval(pollStatus, 3000);
pollStatus();

async function loadCameras() {
  const res = await fetch(`${API}/api/cameras`);
  cameras = await res.json();
  renderCameraList();
  if (cameras.length) {
    selectCamera(cameras[0].id);
  }
}

function renderCameraList() {
  el.cameraList.innerHTML = "";
  for (const cam of cameras) {
    const item = document.createElement("div");
    item.className = "camera-item" + (cam.id === activeCameraId ? " active" : "");
    item.onclick = () => selectCamera(cam.id);
    item.innerHTML = `
      <img class="camera-thumb" src="${API}/api/cameras/${cam.id}/snapshot" />
      <div class="camera-meta">
        <div class="camera-name">${cam.name}</div>
        <div class="camera-location">${cam.location}</div>
        <div class="camera-online"><span class="dot"></span>${cam.online ? "Online" : "Offline"}</div>
      </div>
    `;
    el.cameraList.appendChild(item);
  }
}

function refreshThumbnails() {
  const imgs = el.cameraList.querySelectorAll(".camera-thumb");
  imgs.forEach((img, i) => {
    const cam = cameras[i];
    if (cam) img.src = `${API}/api/cameras/${cam.id}/snapshot?t=${Date.now()}`;
  });
}
setInterval(refreshThumbnails, 4000);

function selectCamera(camId) {
  activeCameraId = camId;
  const cam = cameras.find((c) => c.id === camId);
  if (!cam) return;

  el.viewerTitle.textContent = cam.name;
  el.viewerLocation.textContent = cam.location;
  el.viewerStream.src = `${API}/api/cameras/${camId}/stream`;
  el.viewerStream.classList.add("visible");
  el.viewerPlaceholder.style.display = "none";

  renderCameraList();
  renderChat();
}

async function loadPrompts() {
  const res = await fetch(`${API}/api/prompts`);
  const prompts = await res.json();
  el.promptChips.innerHTML = "";
  for (const p of prompts) {
    const chip = document.createElement("div");
    chip.className = "chip";
    chip.textContent = p;
    chip.onclick = () => {
      el.chatInput.value = p;
      el.chatForm.requestSubmit();
    };
    el.promptChips.appendChild(chip);
  }
}

function renderChat() {
  const history = chatHistory[activeCameraId] || [];
  el.chatLog.innerHTML = "";
  if (history.length === 0) {
    el.chatLog.innerHTML = `<div class="chat-empty">No conversation yet for this camera. Try one of the prompts below, or ask your own question.</div>`;
    return;
  }
  for (const msg of history) {
    appendMessageEl(msg.role, msg.text, msg.thinking);
  }
  el.chatLog.scrollTop = el.chatLog.scrollHeight;
}

function appendMessageEl(role, text, thinking) {
  const bubble = document.createElement("div");
  bubble.className = `msg msg-${role}` + (thinking ? " thinking" : "");
  bubble.textContent = text;
  el.chatLog.appendChild(bubble);
  el.chatLog.scrollTop = el.chatLog.scrollHeight;
  return bubble;
}

el.chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = el.chatInput.value.trim();
  if (!question || !activeCameraId) return;

  if (!chatHistory[activeCameraId]) chatHistory[activeCameraId] = [];
  chatHistory[activeCameraId].push({ role: "user", text: question });
  renderChat();
  el.chatInput.value = "";
  el.chatSend.disabled = true;

  const thinkingBubble = appendMessageEl("ai", "Analyzing current frame…", true);

  try {
    const res = await fetch(`${API}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ camera_id: activeCameraId, question }),
    });
    const data = await res.json();
    thinkingBubble.remove();

    if (!res.ok) {
      const errText = data.detail || "Something went wrong.";
      chatHistory[activeCameraId].push({ role: "ai", text: errText });
    } else {
      chatHistory[activeCameraId].push({ role: "ai", text: data.answer });
    }
  } catch (err) {
    thinkingBubble.remove();
    chatHistory[activeCameraId].push({
      role: "ai",
      text: "Could not reach the analysis server.",
    });
  }

  renderChat();
  el.chatSend.disabled = false;
});

loadCameras();
loadPrompts();
