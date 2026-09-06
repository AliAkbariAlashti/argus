const API = "";

let cameras = [];
let activeCameraId = null;
let currentView = "dashboard";
// One global conversation covering every camera, not one per camera —
// switching the focused camera only changes the live view, never wipes
// context. Each entry: {role, text, thinking?, camerasUsed?}
const chatLog = [];

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

  btnClearChat: document.getElementById("btn-clear-chat"),

  camerasTable: document.getElementById("cameras-table"),
  btnAddCamera: document.getElementById("btn-add-camera"),
  railCamerasBadge: document.getElementById("rail-cameras-badge"),

  eventsList: document.getElementById("events-list"),
  eventsCharts: document.getElementById("events-charts"),
  eventsSearch: document.getElementById("events-search"),
  eventsCameraFilter: document.getElementById("events-camera-filter"),
  eventsSeverityFilter: document.getElementById("events-severity-filter"),
  railAlertsBadge: document.getElementById("rail-alerts-badge"),

  alertRuleForm: document.getElementById("alert-rule-form"),
  alertRuleCamera: document.getElementById("alert-rule-camera"),
  alertRuleTarget: document.getElementById("alert-rule-target"),
  alertRulesList: document.getElementById("alert-rules-list"),

  healthPage: document.getElementById("health-page"),

  modalOverlay: document.getElementById("modal-overlay"),
  modalClose: document.getElementById("modal-close"),
  btnCancelCamera: document.getElementById("btn-cancel-camera"),
  cameraForm: document.getElementById("camera-form"),
  cameraFormError: document.getElementById("camera-form-error"),
  btnSubmitCamera: document.getElementById("btn-submit-camera"),

  editModalOverlay: document.getElementById("edit-modal-overlay"),
  editModalClose: document.getElementById("edit-modal-close"),
  btnCancelEditCamera: document.getElementById("btn-cancel-edit-camera"),
  editCameraForm: document.getElementById("edit-camera-form"),
  editCameraFormError: document.getElementById("edit-camera-form-error"),
  btnSubmitEditCamera: document.getElementById("btn-submit-edit-camera"),
};

const VIEW_TITLES = {
  dashboard: "Dashboard",
  chat: "Chat",
  alerts: "Alerts",
  archive: "Archive",
  cameras: "Cameras",
  health: "System Health",
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
  if (view === "health") loadHealth();
  if (view === "alerts") {
    loadEvents();
    loadAlertRules();
    markEventsSeen();
  }

  // The live MJPEG connection stays open as long as the <img> has a src,
  // even while its view is hidden — stop it when navigating away, and
  // resume it when coming back to Chat with a camera already selected.
  if (view === "chat" && activeCameraId) {
    if (!el.viewerStream.src) {
      el.viewerStream.src = `${API}/api/cameras/${activeCameraId}/stream`;
    }
  } else {
    el.viewerStream.removeAttribute("src");
    el.viewerStream.classList.remove("visible");
  }
}

/* ---------- data loading ---------- */

async function loadCameras() {
  const res = await fetch(`${API}/api/cameras`);
  cameras = await res.json();
  renderDashboard();
  renderChatStrip();
  renderRailBadge();
  populateCameraSelects();
  if (currentView === "cameras") renderCamerasTable();

  if (!activeCameraId && cameras.length) {
    selectCamera(cameras[0].id);
  }
}

function renderRailBadge() {
  const online = cameras.filter((c) => c.online).length;
  el.railCamerasBadge.hidden = cameras.length === 0;
  el.railCamerasBadge.textContent = `${online}/${cameras.length}`;
  el.railCamerasBadge.classList.toggle("warn", online < cameras.length);
}

function refreshThumbnails() {
  // Hidden views stay in the DOM (just [hidden]), so scope this to the
  // active view only — otherwise every camera's snapshot gets refetched
  // redundantly for the dashboard grid, chat strip, and cameras table at
  // once, multiplying request load for no visible benefit.
  const activeViewEl = document.getElementById(`view-${currentView}`);
  if (!activeViewEl) return;
  activeViewEl.querySelectorAll("[data-thumb-for]").forEach((img) => {
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
  // Only changes which feed is focused in the viewer — the chat below it
  // is one continuous conversation across every camera, so this never
  // touches chatLog.
  activeCameraId = camId;
  const cam = cameras.find((c) => c.id === camId);
  if (!cam) return;

  el.viewerTitle.textContent = cam.name;
  el.viewerLocation.textContent = cam.location;
  el.viewerStream.src = `${API}/api/cameras/${camId}/stream`;
  el.viewerStream.classList.add("visible");
  el.viewerPlaceholder.style.display = "none";

  renderChatStrip();
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

async function loadChatHistory() {
  try {
    const res = await fetch(`${API}/api/chat/history`);
    if (!res.ok) return;
    const rows = await res.json();
    chatLog.length = 0;
    for (const row of rows) {
      chatLog.push({
        role: row.role === "assistant" ? "ai" : "user",
        text: row.text,
        camerasUsed: row.cameras_used,
        snapshot: row.snapshot,
      });
    }
    renderChat();
  } catch (e) {
    // A missing transcript shouldn't block using the app.
  }
}

function renderChat() {
  el.chatLog.innerHTML = "";
  if (chatLog.length === 0) {
    el.chatLog.innerHTML = `<div class="chat-empty">No conversation yet. Ask about a specific camera, or something across all of them.</div>`;
    return;
  }
  for (const msg of chatLog) {
    appendMessageEl(msg.role, msg.text, msg.thinking, msg.camerasUsed, msg.snapshot);
  }
  el.chatLog.scrollTop = el.chatLog.scrollHeight;
}

function camerasUsedLabel(camIds) {
  if (!camIds || camIds.length === 0) return null;
  const names = camIds
    .map((id) => cameras.find((c) => c.id === id)?.name)
    .filter(Boolean);
  if (names.length === 0) return null;
  return names.length === 1 ? `📷 ${names[0]}` : `📷 ${names.join(", ")}`;
}

function appendMessageEl(role, text, thinking, camerasUsed, snapshot) {
  const bubble = document.createElement("div");
  bubble.className = `msg msg-${role}` + (thinking ? " thinking" : "");

  const body = document.createElement("div");
  body.textContent = text;
  bubble.appendChild(body);

  // Show the frame the answer was actually based on, so the claim is
  // checkable rather than something the user has to take on faith.
  if (snapshot) {
    const img = document.createElement("img");
    img.className = "msg-snapshot";
    img.src = snapshot;
    img.alt = "Frame analyzed";
    bubble.appendChild(img);
  }

  const label = camerasUsedLabel(camerasUsed);
  if (label || (role === "ai" && !thinking)) {
    const meta = document.createElement("div");
    meta.className = "msg-meta";

    if (label) {
      const span = document.createElement("span");
      span.textContent = label;
      meta.appendChild(span);
    }

    if (role === "ai" && !thinking) {
      const copyBtn = document.createElement("button");
      copyBtn.className = "msg-copy";
      copyBtn.textContent = "Copy";
      copyBtn.onclick = async () => {
        try {
          await navigator.clipboard.writeText(text);
          copyBtn.textContent = "Copied";
          setTimeout(() => (copyBtn.textContent = "Copy"), 1500);
        } catch (e) {
          copyBtn.textContent = "Failed";
          setTimeout(() => (copyBtn.textContent = "Copy"), 1500);
        }
      };
      meta.appendChild(copyBtn);
    }

    bubble.appendChild(meta);
  }

  el.chatLog.appendChild(bubble);
  el.chatLog.scrollTop = el.chatLog.scrollHeight;
  return bubble;
}

el.btnClearChat.addEventListener("click", async () => {
  if (chatLog.length && !confirm("Clear the whole conversation?")) return;
  try {
    await fetch(`${API}/api/chat/history`, { method: "DELETE" });
  } catch (e) {
    // Clearing the view is still the right outcome if the call fails.
  }
  chatLog.length = 0;
  renderChat();
});

el.chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = el.chatInput.value.trim();
  if (!question) return;

  chatLog.push({ role: "user", text: question });
  renderChat();
  el.chatInput.value = "";
  el.chatSend.disabled = true;

  const thinkingBubble = appendMessageEl("ai", "Analyzing…", true);

  try {
    const res = await fetch(`${API}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, focused_camera_id: activeCameraId }),
    });
    const data = await res.json();
    thinkingBubble.remove();

    if (!res.ok) {
      chatLog.push({ role: "ai", text: data.detail || "Something went wrong." });
    } else {
      chatLog.push({
        role: "ai",
        text: data.answer,
        camerasUsed: data.cameras_used,
        snapshot: data.snapshot,
      });
    }
  } catch (err) {
    thinkingBubble.remove();
    chatLog.push({ role: "ai", text: "Could not reach the analysis server." });
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
        <button class="btn-icon" data-action="edit">Edit</button>
        <button class="btn-icon danger" data-action="delete">Delete</button>
      </div>
    `;
    row.querySelector('[data-action="view"]').onclick = () => {
      switchView("chat");
      selectCamera(cam.id);
    };
    row.querySelector('[data-action="edit"]').onclick = () => openEditModal(cam);
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

  // Video uploads can be several MB; give it real headroom but never hang
  // the modal forever if the server or network stalls.
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 60000);

  try {
    const res = await fetch(`${API}/api/cameras`, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Failed to add camera.");
    }
    closeCameraModal();
    await loadCameras();
  } catch (err) {
    el.cameraFormError.textContent =
      err.name === "AbortError"
        ? "The request timed out. Please try again."
        : err.message;
    el.cameraFormError.hidden = false;
  } finally {
    clearTimeout(timeoutId);
    el.btnSubmitCamera.disabled = false;
    el.btnSubmitCamera.textContent = "Add Camera";
  }
});

/* ---------- edit camera ---------- */

let editingCameraId = null;

function openEditModal(cam) {
  editingCameraId = cam.id;
  const f = el.editCameraForm;
  f.elements.name.value = cam.name || "";
  f.elements.location.value = cam.location || "";
  f.elements.zone_tags.value = (cam.zone_tags || []).join(", ");
  f.elements.description.value = cam.description || "";
  el.editCameraFormError.hidden = true;
  el.editModalOverlay.hidden = false;
}

function closeEditModal() {
  el.editModalOverlay.hidden = true;
  editingCameraId = null;
}

el.editModalClose.addEventListener("click", closeEditModal);
el.btnCancelEditCamera.addEventListener("click", closeEditModal);
el.editModalOverlay.addEventListener("click", (e) => {
  if (e.target === el.editModalOverlay) closeEditModal();
});

el.editCameraForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!editingCameraId) return;

  el.editCameraFormError.hidden = true;
  el.btnSubmitEditCamera.disabled = true;
  el.btnSubmitEditCamera.textContent = "Saving…";

  const f = el.editCameraForm;
  const payload = {
    name: f.elements.name.value.trim(),
    location: f.elements.location.value.trim(),
    zone_tags: f.elements.zone_tags.value
      .split(",")
      .map((t) => t.trim())
      .filter(Boolean),
    description: f.elements.description.value.trim(),
  };

  try {
    const res = await fetch(`${API}/api/cameras/${editingCameraId}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Failed to save changes.");
    closeEditModal();
    await loadCameras();
  } catch (err) {
    el.editCameraFormError.textContent = err.message;
    el.editCameraFormError.hidden = false;
  } finally {
    el.btnSubmitEditCamera.disabled = false;
    el.btnSubmitEditCamera.textContent = "Save changes";
  }
});

/* ---------- health ---------- */

async function loadHealth() {
  el.healthPage.innerHTML = `<div class="empty-state">Loading…</div>`;
  try {
    const res = await fetch(`${API}/api/health`);
    const data = await res.json();
    renderHealth(data);
  } catch (err) {
    el.healthPage.innerHTML = `<div class="empty-state">Could not reach the server.</div>`;
  }
}

function renderHealth(data) {
  const gpu = data.gpu || {};
  const gpuPct = gpu.available && gpu.total_mb
    ? Math.round((gpu.used_mb / gpu.total_mb) * 100)
    : 0;

  const modelState = data.model.ready
    ? `<span class="health-ok">Ready</span>`
    : data.model.error
    ? `<span class="health-bad">Error</span>`
    : `<span class="health-warn">Loading…</span>`;

  el.healthPage.innerHTML = `
    <div class="health-grid">
      <div class="health-card">
        <div class="health-card-title">Model</div>
        <div class="health-row"><span>Status</span><span>${modelState}</span></div>
        <div class="health-row"><span>Model</span><span class="health-mono">${escapeHtml(data.model.id)}</span></div>
        ${
          data.model.error
            ? `<div class="health-error">${escapeHtml(data.model.error)}</div>`
            : ""
        }
      </div>

      <div class="health-card">
        <div class="health-card-title">GPU</div>
        ${
          gpu.available
            ? `
          <div class="health-row"><span>Device</span><span class="health-mono">${escapeHtml(gpu.name)}</span></div>
          <div class="health-row"><span>Memory</span><span class="health-mono">${gpu.used_mb} / ${gpu.total_mb} MB</span></div>
          <div class="health-bar"><div class="health-bar-fill" style="width:${gpuPct}%"></div></div>
        `
            : `<div class="health-muted">No CUDA device detected.</div>`
        }
      </div>

      <div class="health-card health-card-wide">
        <div class="health-card-title">Cameras</div>
        ${
          (data.cameras || []).length === 0
            ? `<div class="health-muted">No cameras configured.</div>`
            : data.cameras
                .map(
                  (c) => `
          <div class="health-row">
            <span>${escapeHtml(c.name)}</span>
            <span>
              <span class="health-mono">${c.fps.toFixed(1)} fps</span>
              ${c.online ? `<span class="health-ok">online</span>` : `<span class="health-bad">offline</span>`}
            </span>
          </div>`
                )
                .join("")
        }
      </div>
    </div>
  `;
}

/* ---------- events (alerts) ---------- */
// The badge/seen tracking lives in localStorage, per-browser — there's no
// concept of "read" server-side, this is just a lightweight "anything new
// since I last looked" indicator on the rail icon.
//
// Two separate polls, deliberately not one: `recentUnfiltered` (unfiltered,
// small, every 10s) drives the badge and the critical-event toast — those
// must stay correct regardless of whatever search/camera/severity filter is
// currently applied to the visible list. `latestEvents` (filtered, fetched
// on open/filter-change) drives what's actually rendered below.

const EVENTS_SEEN_KEY = "sentinel_events_last_seen_id";
let latestEvents = [];
let recentUnfiltered = [];
let alertRules = [];
const eventFilters = { q: "", camera_id: "", severity: "" };

function getLastSeenEventId() {
  try {
    return parseInt(localStorage.getItem(EVENTS_SEEN_KEY) || "0", 10) || 0;
  } catch (e) {
    return 0;
  }
}

function markEventsSeen() {
  if (!recentUnfiltered.length) return;
  const newestId = Math.max(...recentUnfiltered.map((e) => e.id));
  try {
    localStorage.setItem(EVENTS_SEEN_KEY, String(newestId));
  } catch (e) {
    // Private-browsing/blocked storage shouldn't break the view.
  }
  renderAlertsBadge();
}

async function pollRecentEvents() {
  try {
    const res = await fetch(`${API}/api/events?limit=30`);
    if (!res.ok) return;
    const rows = await res.json();
    const seenIds = new Set(recentUnfiltered.map((e) => e.id));
    const newCritical = rows.find((e) => e.severity === "critical" && !seenIds.has(e.id));
    recentUnfiltered = rows;
    renderAlertsBadge();
    if (newCritical) notifyCritical(newCritical);
  } catch (e) {
    // Transient errors shouldn't break other views.
  }
}
setInterval(pollRecentEvents, 10000);

function buildEventsQuery() {
  const params = new URLSearchParams({ limit: "300" });
  if (eventFilters.q) params.set("q", eventFilters.q);
  if (eventFilters.camera_id) params.set("camera_id", eventFilters.camera_id);
  if (eventFilters.severity) params.set("severity", eventFilters.severity);
  return params.toString();
}

async function loadEvents() {
  try {
    const res = await fetch(`${API}/api/events?${buildEventsQuery()}`);
    if (!res.ok) return;
    latestEvents = await res.json();
    renderEvents();
    renderEventCharts();
  } catch (e) {
    // Transient errors shouldn't break other views.
  }
}

let searchDebounceTimer = null;
el.eventsSearch.addEventListener("input", () => {
  clearTimeout(searchDebounceTimer);
  searchDebounceTimer = setTimeout(() => {
    eventFilters.q = el.eventsSearch.value.trim();
    loadEvents();
  }, 300);
});
el.eventsCameraFilter.addEventListener("change", () => {
  eventFilters.camera_id = el.eventsCameraFilter.value;
  loadEvents();
});
el.eventsSeverityFilter.addEventListener("change", () => {
  eventFilters.severity = el.eventsSeverityFilter.value;
  loadEvents();
});

function renderAlertsBadge() {
  const lastSeen = getLastSeenEventId();
  const unseen = recentUnfiltered.filter((e) => e.id > lastSeen && e.severity !== "info");
  el.railAlertsBadge.hidden = unseen.length === 0;
  el.railAlertsBadge.textContent = unseen.length;
  el.railAlertsBadge.classList.toggle("warn", unseen.some((e) => e.severity === "critical"));
}

function eventCameraLabel(camId) {
  return cameras.find((c) => c.id === camId)?.name || "Unknown camera";
}

function renderEvents() {
  el.eventsList.innerHTML = "";
  if (latestEvents.length === 0) {
    el.eventsList.innerHTML = `<div class="empty-state">No events match the current filters.</div>`;
    return;
  }
  for (const ev of latestEvents) {
    const row = document.createElement("div");
    row.className = `event-row event-${ev.severity}`;
    const time = ev.created_at ? new Date(ev.created_at).toLocaleString() : "";
    row.innerHTML = `
      ${
        ev.snapshot
          ? `<img class="event-thumb" src="${ev.snapshot}" />`
          : `<div class="event-thumb"></div>`
      }
      <div class="event-info">
        <div class="event-top">
          <span class="event-severity event-severity-${ev.severity}">${escapeHtml(ev.severity)}</span>
          <span class="event-camera">${escapeHtml(eventCameraLabel(ev.camera_id))}</span>
          <span class="event-time">${time}</span>
        </div>
        <div class="event-category">${escapeHtml(ev.category)}</div>
        <div class="event-summary">${escapeHtml(ev.summary)}</div>
      </div>
    `;
    el.eventsList.appendChild(row);
  }
}

/* ---------- events: charts ---------- */
// Categorical hues validated (dataviz skill's validate_palette.js) against
// this app's actual card surface (#131926) in dark mode — worst adjacent
// CVD ΔE 8.4, worst normal-vision ΔE 19.3, all >=3:1 contrast. Every bar
// also carries a direct text label, so identity never rests on color alone.
const CAMERA_CHART_COLORS = [
  "#3987e5", "#d95926", "#199e70", "#c98500",
  "#d55181", "#008300", "#9085e9", "#e66767",
];

function chartBarRow(label, count, max, color) {
  const pct = max > 0 ? (count / max) * 100 : 0;
  return `
    <div class="chart-bar-row">
      <span class="chart-bar-label">${escapeHtml(label)}</span>
      <div class="chart-bar-track">
        <div class="chart-bar-fill" style="width:${pct}%; background:${color}"></div>
      </div>
      <span class="chart-bar-count">${count}</span>
    </div>`;
}

function renderEventCharts() {
  el.eventsCharts.innerHTML = "";
  if (latestEvents.length === 0) {
    el.eventsCharts.innerHTML = `<div class="empty-state">No events yet to chart.</div>`;
    return;
  }

  // Severity carries the app's existing status colors (same as the list
  // rows below) - critical/warning/info, never reused for camera identity.
  const severityCounts = { critical: 0, warning: 0, info: 0 };
  for (const e of latestEvents) {
    if (severityCounts[e.severity] !== undefined) severityCounts[e.severity]++;
  }
  const severityColors = { critical: "var(--red)", warning: "var(--amber)", info: "var(--border)" };
  const severityMax = Math.max(...Object.values(severityCounts));

  const severityCard = document.createElement("div");
  severityCard.className = "chart-card";
  severityCard.innerHTML = `
    <div class="chart-card-title">Events by severity</div>
    <div class="chart-bars">
      ${Object.entries(severityCounts)
        .map(([sev, count]) => chartBarRow(sev, count, severityMax, severityColors[sev]))
        .join("")}
    </div>
  `;
  el.eventsCharts.appendChild(severityCard);

  const cameraCounts = {};
  for (const e of latestEvents) {
    cameraCounts[e.camera_id] = (cameraCounts[e.camera_id] || 0) + 1;
  }
  const sortedCams = Object.entries(cameraCounts).sort((a, b) => b[1] - a[1]).slice(0, 8);
  const camMax = Math.max(...sortedCams.map(([, c]) => c));

  const cameraCard = document.createElement("div");
  cameraCard.className = "chart-card";
  cameraCard.innerHTML = `
    <div class="chart-card-title">Events by camera</div>
    <div class="chart-bars">
      ${sortedCams
        .map(([camId, count], i) =>
          chartBarRow(
            eventCameraLabel(camId),
            count,
            camMax,
            CAMERA_CHART_COLORS[i % CAMERA_CHART_COLORS.length]
          )
        )
        .join("")}
    </div>
  `;
  el.eventsCharts.appendChild(cameraCard);
}

/* ---------- events: critical toast + sound ---------- */

function notifyCritical(event) {
  showToast(`⚠ ${event.category} — ${eventCameraLabel(event.camera_id)}`);
  playAlertSound();
}

function showToast(text) {
  const toast = document.createElement("div");
  toast.className = "event-toast";
  toast.textContent = text;
  document.body.appendChild(toast);
  requestAnimationFrame(() => toast.classList.add("show"));
  setTimeout(() => {
    toast.classList.remove("show");
    setTimeout(() => toast.remove(), 300);
  }, 6000);
}

let audioCtx = null;
function playAlertSound() {
  // WebAudio, not an audio file - no asset to ship, no CDN dependency. Browsers
  // block audio until the page has had a user gesture; if that hasn't happened
  // yet this just silently no-ops and the toast still shows.
  try {
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = "sine";
    osc.frequency.value = 880;
    gain.gain.setValueAtTime(0.0001, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.3, audioCtx.currentTime + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + 0.5);
    osc.connect(gain).connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.5);
  } catch (e) {
    // Audio blocked/unsupported - the toast alone is still a valid alert.
  }
}

/* ---------- alert rules ---------- */

function populateCameraSelects() {
  const optionsHtml =
    `<option value="">All cameras</option>` +
    cameras.map((c) => `<option value="${c.id}">${escapeHtml(c.name)}</option>`).join("");
  const camFilterVal = el.eventsCameraFilter.value;
  const ruleCamVal = el.alertRuleCamera.value;
  el.eventsCameraFilter.innerHTML = optionsHtml;
  el.alertRuleCamera.innerHTML = optionsHtml;
  el.eventsCameraFilter.value = camFilterVal;
  el.alertRuleCamera.value = ruleCamVal;
}

async function loadAlertRules() {
  try {
    const res = await fetch(`${API}/api/alert-rules`);
    if (!res.ok) return;
    alertRules = await res.json();
    renderAlertRulesList();
  } catch (e) {
    // Transient errors shouldn't break other views.
  }
}

function renderAlertRulesList() {
  el.alertRulesList.innerHTML = "";
  if (alertRules.length === 0) {
    el.alertRulesList.innerHTML = `<div class="empty-state">No custom alert rules yet.</div>`;
    return;
  }
  for (const rule of alertRules) {
    const row = document.createElement("div");
    row.className = "alert-rule-row" + (rule.enabled ? "" : " disabled");
    const camLabel = rule.camera_id ? eventCameraLabel(rule.camera_id) : "All cameras";
    row.innerHTML = `
      <span class="alert-rule-target">"${escapeHtml(rule.target)}"</span>
      <span class="alert-rule-camera">${escapeHtml(camLabel)}</span>
      <label class="alert-rule-toggle">
        <input type="checkbox" ${rule.enabled ? "checked" : ""} />
        <span>${rule.enabled ? "Enabled" : "Disabled"}</span>
      </label>
      <button class="btn-icon danger" data-action="delete">Delete</button>
    `;
    row.querySelector('input[type="checkbox"]').addEventListener("change", (e) => {
      toggleAlertRule(rule.id, e.target.checked);
    });
    row.querySelector('[data-action="delete"]').addEventListener("click", () => {
      deleteAlertRule(rule.id);
    });
    el.alertRulesList.appendChild(row);
  }
}

async function toggleAlertRule(ruleId, enabled) {
  await fetch(`${API}/api/alert-rules/${ruleId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled }),
  });
  await loadAlertRules();
}

async function deleteAlertRule(ruleId) {
  await fetch(`${API}/api/alert-rules/${ruleId}`, { method: "DELETE" });
  await loadAlertRules();
}

el.alertRuleForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const target = el.alertRuleTarget.value.trim();
  if (!target) return;
  const camera_id = el.alertRuleCamera.value || null;
  await fetch(`${API}/api/alert-rules`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ camera_id, target }),
  });
  el.alertRuleTarget.value = "";
  await loadAlertRules();
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
renderChat();
loadChatHistory();
pollRecentEvents();
loadAlertRules();
