const API = "";

let cameras = [];
let activeCameraId = null;
let currentView = "dashboard";
const chatHistory = {}; // camera_id -> [{role, text}]

const el = {
  clock: document.getElementById("clock"),
  modelStatus: document.getElementById("model-status"),
  railStatus: document.getElementById("rail-status"),
  topbarTitle: document.getElementById("topbar-title"),

  dashboardGrid: document.getElementById("dashboard-grid"),

  chatStrip: document.getElementById("chat-strip"),
  viewerTitle: document.getElementById("viewer-title"),
  viewerLocation: document.getElementById("viewer-location"),
  viewerStream: document.getElementById("viewer-stream"),
  viewerPlaceholder: document.getElementById("viewer-placeholder"),
  chatLog: document.getElementById("chat-log"),
  promptChips: document.getElementById("prompt-chips"),
  chatForm: document.getElementById("chat-form"),
  chatInput: document.getElementById("chat-input"),
  chatSend: document.getElementById("chat-send"),

  camerasTable: document.getElementById("cameras-table"),
  btnAddCamera: document.getElementById("btn-add-camera"),

  modalOverlay: document.getElementById("modal-overlay"),
  modalClose: document.getElementById("modal-close"),
  btnCancelCamera: document.getElementById("btn-cancel-camera"),
  cameraForm: document.getElementById("camera-form"),
  cameraFormError: document.getElementById("camera-form-error"),
  btnSubmitCamera: document.getElementById("btn-submit-camera"),
};

const VIEW_TITLES = {
  dashboard: "Dashboard",
  chat: "Chat",
  alerts: "Alerts",
  archive: "Archive",
  cameras: "Cameras",
};

/* ---------- clock + model status ---------- */

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
      el.railStatus.className = "rail-status ready";
    } else if (data.model_error) {
      el.modelStatus.className = "pill pill-error";
      el.modelStatus.innerHTML = `<span class="dot"></span> Model error`;
      el.railStatus.className = "rail-status error";
    } else {
      el.modelStatus.className = "pill pill-loading";
      el.modelStatus.innerHTML = `<span class="dot"></span> Loading model…`;
      el.railStatus.className = "rail-status";
    }
  } catch (e) {
    // ignore transient errors
  }
}
setInterval(pollStatus, 3000);
pollStatus();

/* ---------- view routing ---------- */

document.querySelectorAll(".rail-btn").forEach((btn) => {
  if (btn.classList.contains("disabled")) return;
  btn.addEventListener("click", () => switchView(btn.dataset.view));
});

function switchView(view) {
  currentView = view;
  document.querySelectorAll(".rail-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.view === view);
  });
  document.querySelectorAll(".view").forEach((v) => {
    v.hidden = v.id !== `view-${view}`;
  });
  el.topbarTitle.textContent = VIEW_TITLES[view] || "";

  if (view === "cameras") renderCamerasTable();
}

/* ---------- data loading ---------- */

async function loadCameras() {
  const res = await fetch(`${API}/api/cameras`);
  cameras = await res.json();
  renderDashboard();
  renderChatStrip();
  if (currentView === "cameras") renderCamerasTable();

  if (!activeCameraId && cameras.length) {
    selectCamera(cameras[0].id);
  }
}

function refreshThumbnails() {
  document.querySelectorAll("[data-thumb-for]").forEach((img) => {
    const camId = img.dataset.thumbFor;
    img.src = `${API}/api/cameras/${camId}/snapshot?t=${Date.now()}`;
  });
}
setInterval(refreshThumbnails, 4000);
setInterval(loadCameras, 15000);

/* ---------- dashboard ---------- */

function renderDashboard() {
  el.dashboardGrid.innerHTML = "";
  if (cameras.length === 0) {
    el.dashboardGrid.innerHTML = `<div class="empty-state">No cameras yet. Go to Cameras to add one.</div>`;
    return;
  }
  for (const cam of cameras) {
    const tile = document.createElement("div");
    tile.className = "tile";
    tile.onclick = () => {
      switchView("chat");
      selectCamera(cam.id);
    };
    tile.innerHTML = `
      <div class="tile-frame">
        ${
          cam.online
            ? `<img data-thumb-for="${cam.id}" src="${API}/api/cameras/${cam.id}/snapshot" />`
            : `<div class="tile-offline">No signal</div>`
        }
        <div class="tile-live"><span class="dot"></span>LIVE</div>
      </div>
      <div class="tile-meta">
        <div class="tile-name">${escapeHtml(cam.name)}</div>
        <div class="tile-location">${escapeHtml(cam.location)}</div>
        <div class="tile-tags">
          ${(cam.zone_tags || []).map((t) => `<span class="tile-tag">${escapeHtml(t)}</span>`).join("")}
        </div>
      </div>
    `;
    el.dashboardGrid.appendChild(tile);
  }
}

/* ---------- chat view ---------- */

function renderChatStrip() {
  el.chatStrip.innerHTML = "";
  for (const cam of cameras) {
    const item = document.createElement("div");
    item.className = "strip-item" + (cam.id === activeCameraId ? " active" : "");
    item.onclick = () => selectCamera(cam.id);
    item.innerHTML = `
      <img class="strip-thumb" data-thumb-for="${cam.id}" src="${API}/api/cameras/${cam.id}/snapshot" />
      <div class="strip-label">${escapeHtml(cam.name)}</div>
    `;
    el.chatStrip.appendChild(item);
  }
}

function selectCamera(camId) {
  activeCameraId = camId;
  const cam = cameras.find((c) => c.id === camId);
  if (!cam) return;

  el.viewerTitle.textContent = cam.name;
  el.viewerLocation.textContent = cam.location;
  el.viewerStream.src = `${API}/api/cameras/${camId}/stream`;
  el.viewerStream.classList.add("visible");
  el.viewerPlaceholder.style.display = "none";

  renderChatStrip();
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
      chatHistory[activeCameraId].push({ role: "ai", text: data.detail || "Something went wrong." });
    } else {
      chatHistory[activeCameraId].push({ role: "ai", text: data.answer });
    }
  } catch (err) {
    thinkingBubble.remove();
    chatHistory[activeCameraId].push({ role: "ai", text: "Could not reach the analysis server." });
  }

  renderChat();
  el.chatSend.disabled = false;
});

/* ---------- cameras CRUD ---------- */

function renderCamerasTable() {
  el.camerasTable.innerHTML = "";
  if (cameras.length === 0) {
    el.camerasTable.innerHTML = `<div class="empty-state">No cameras yet. Click "+ Add Camera" to create one.</div>`;
    return;
  }
  for (const cam of cameras) {
    const row = document.createElement("div");
    row.className = "cam-row";
    row.innerHTML = `
      <img class="cam-row-thumb" data-thumb-for="${cam.id}" src="${API}/api/cameras/${cam.id}/snapshot" />
      <div class="cam-row-info">
        <div class="cam-row-name">${escapeHtml(cam.name)}</div>
        <div class="cam-row-sub">${escapeHtml(cam.location)} &middot; ${cam.online ? "Online" : "Offline"}</div>
        <div class="cam-row-tags">
          ${(cam.zone_tags || []).map((t) => `<span class="tile-tag">${escapeHtml(t)}</span>`).join("")}
        </div>
      </div>
      <div class="cam-row-actions">
        <button class="btn-icon" data-action="view">View</button>
        <button class="btn-icon danger" data-action="delete">Delete</button>
      </div>
    `;
    row.querySelector('[data-action="view"]').onclick = () => {
      switchView("chat");
      selectCamera(cam.id);
    };
    row.querySelector('[data-action="delete"]').onclick = () => deleteCamera(cam.id, cam.name);
    el.camerasTable.appendChild(row);
  }
}

async function deleteCamera(camId, name) {
  if (!confirm(`Remove camera "${name}"? This cannot be undone.`)) return;
  await fetch(`${API}/api/cameras/${camId}`, { method: "DELETE" });
  if (activeCameraId === camId) activeCameraId = null;
  await loadCameras();
}

el.btnAddCamera.addEventListener("click", () => openCameraModal());
el.modalClose.addEventListener("click", () => closeCameraModal());
el.btnCancelCamera.addEventListener("click", () => closeCameraModal());
el.modalOverlay.addEventListener("click", (e) => {
  if (e.target === el.modalOverlay) closeCameraModal();
});

document.querySelectorAll(".field-tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    if (tab.classList.contains("field-tab-disabled")) return;
    document.querySelectorAll(".field-tab").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
  });
});

function openCameraModal() {
  el.cameraForm.reset();
  el.cameraFormError.hidden = true;
  el.modalOverlay.hidden = false;
}

function closeCameraModal() {
  el.modalOverlay.hidden = true;
}

el.cameraForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  el.cameraFormError.hidden = true;
  el.btnSubmitCamera.disabled = true;
  el.btnSubmitCamera.textContent = "Adding…";

  const formData = new FormData(el.cameraForm);

  try {
    const res = await fetch(`${API}/api/cameras`, {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Failed to add camera.");
    }
    closeCameraModal();
    await loadCameras();
  } catch (err) {
    el.cameraFormError.textContent = err.message;
    el.cameraFormError.hidden = false;
  } finally {
    el.btnSubmitCamera.disabled = false;
    el.btnSubmitCamera.textContent = "Add Camera";
  }
});

/* ---------- utils ---------- */

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}

/* ---------- init ---------- */

loadCameras();
loadPrompts();
