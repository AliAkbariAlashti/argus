const API = "";

let cameras = [];
let activeCameraId = null;
let currentView = "chat";
// One global conversation covering every camera, not one per camera —
// switching the focused camera only changes the live view, never wipes
// context. Each entry: {role, text, thinking?, camerasUsed?}
const chatLog = [];
let chatSessions = [];
let activeChatSessionId = localStorage.getItem("argus-chat-session");

const el = {
  clock: document.getElementById("clock"),
  modelStatus: document.getElementById("model-status"),
  railStatus: document.getElementById("rail-status"),
  topbarTitle: document.getElementById("topbar-title"),

  statRow: document.getElementById("stat-row"),
  searchForm: document.getElementById("search-form"),
  searchInput: document.getElementById("search-input"),
  searchSubmit: document.getElementById("search-submit"),
  searchChips: document.getElementById("search-chips"),
  searchResult: document.getElementById("search-result"),
  recentQueries: document.getElementById("recent-queries"),
  fleetStrip: document.getElementById("fleet-strip"),
  qaAddFile: document.getElementById("qa-add-file"),
  qaAddCamera: document.getElementById("qa-add-camera"),
  qaAddRule: document.getElementById("qa-add-rule"),

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
  eventsKpis: document.getElementById("events-kpis"),
  eventsLiveState: document.getElementById("events-live-state"),
  eventsLiveToggle: document.getElementById("events-live-toggle"),
  eventsLastUpdated: document.getElementById("events-last-updated"),
  eventsStreamCount: document.getElementById("events-stream-count"),
  eventsMarkSeen: document.getElementById("events-mark-seen"),
  eventsClearFilters: document.getElementById("events-clear-filters"),
  eventsFocusCritical: document.getElementById("events-focus-critical"),
  eventsFocusRecent: document.getElementById("events-focus-recent"),
  eventsSearch: document.getElementById("events-search"),
  eventsCameraFilter: document.getElementById("events-camera-filter"),
  eventsSeverityFilter: document.getElementById("events-severity-filter"),
  railAlertsBadge: document.getElementById("rail-alerts-badge"),
  historyAskForm: document.getElementById("history-ask-form"),
  historyAskInput: document.getElementById("history-ask-input"),
  historyAskAnswer: document.getElementById("history-ask-answer"),

  alertRuleForm: document.getElementById("alert-rule-form"),
  alertRuleCamera: document.getElementById("alert-rule-camera"),
  alertRuleSource: document.getElementById("alert-rule-source"),
  alertRuleClass: document.getElementById("alert-rule-class"),
  alertRuleTarget: document.getElementById("alert-rule-target"),
  alertRuleSeverity: document.getElementById("alert-rule-severity"),
  alertRuleCpuNote: document.getElementById("alert-rule-cpu-note"),
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

const appShell = document.querySelector(".app");
const sidebarToggle = document.getElementById("sidebar-toggle");
const railCollapse = document.getElementById("rail-collapse");

function setSidebarCollapsed(collapsed) {
  appShell.classList.toggle("rail-collapsed", collapsed);
  sidebarToggle.setAttribute("aria-expanded", String(!collapsed));
  localStorage.setItem("argus-sidebar-collapsed", collapsed ? "1" : "0");
}
railCollapse.addEventListener("click", () => setSidebarCollapsed(true));
sidebarToggle.addEventListener("click", () => setSidebarCollapsed(!appShell.classList.contains("rail-collapsed")));
setSidebarCollapsed(localStorage.getItem("argus-sidebar-collapsed") === "1" || innerWidth < 700);

const VIEW_TITLES = {
  dashboard: "Overview",
  chat: "Argus Agent",
  alerts: "Activity",
  rules: "Alert Rules",
  archive: "Archive",
  directory: "Cameras",
  health: "System Health",
  runtime: "AI setup",
  cpu: "CPU tools",
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
      el.modelStatus.innerHTML = `<span class="dot"></span> Agent ready`;
      el.railStatus.className = "rail-status ready";
    } else if (data.configured === false) {
      el.modelStatus.className = "pill pill-loading";
      el.modelStatus.innerHTML = `<span class="dot"></span> AI not connected`;
      el.railStatus.className = "rail-status";
    } else if (data.model_error) {
      el.modelStatus.className = "pill pill-error";
      el.modelStatus.innerHTML = `<span class="dot"></span> Model error`;
      el.railStatus.className = "rail-status error";
    } else {
      el.modelStatus.className = "pill pill-loading";
      el.modelStatus.innerHTML = `<span class="dot"></span> Test AI connection`;
      el.railStatus.className = "rail-status";
    }
  } catch (e) {
    el.modelStatus.className = "pill pill-error";
    el.modelStatus.innerHTML = `<span class="dot"></span> Server unavailable`;
  }
}
setInterval(pollStatus, 3000);
pollStatus();

/* ---------- view routing ---------- */

document.querySelectorAll(".rail-btn").forEach((btn) => {
  if (btn.classList.contains("disabled") || !btn.dataset.view) return;
  btn.addEventListener("click", () => switchView(btn.dataset.view));
});

function switchView(view) {
  currentView = view;
  document.querySelectorAll(".rail-btn[data-view]").forEach((b) => {
    b.classList.toggle("active", b.dataset.view === view);
  });
  document.querySelectorAll(".view").forEach((v) => {
    v.hidden = v.id !== `view-${view}`;
  });
  el.topbarTitle.textContent = VIEW_TITLES[view] || "";

  if (view === "directory") renderCamerasTable();
  if (view === "health") loadHealth();
  if (view === "runtime") loadRuntime();
  if (view === "cpu" && typeof loadCpu === "function") loadCpu(true);
  if (view === "dashboard") renderFleetStrip();
  if (view === "alerts") {
    loadEvents();
    markEventsSeen();
  }
  if (view === "rules") {
    loadAlertRules();
    if (!cpuClasses.length) loadCpuClasses();
  }

  // The live MJPEG connection stays open as long as the <img> has a src,
  // even while its view is hidden — stop it when navigating away, and
  // resume it when coming back to Chat with a camera already selected.
  el.viewerStream.removeAttribute("src");
  el.viewerStream.classList.remove("visible");
}

/* ---------- data loading ---------- */

async function loadCameras() {
  let res;
  try {
    res = await fetch(`${API}/api/cameras`);
    if (!res.ok) throw new Error("Sources unavailable");
    cameras = await res.json();
  } catch (_) {
    el.fleetStrip.innerHTML = '<div class="empty-state">Cannot connect to your cameras. Check system health; we will retry automatically.</div>';
    return;
  }
  renderFleetStrip();
  renderChatStrip();
  renderRailBadge();
  renderStatRow();
  populateCameraSelects();
  if (currentView === "directory") renderCamerasTable();

  if (!cameras.some(c => c.id === activeCameraId)) activeCameraId = null;
  if (!activeCameraId && cameras.length) {
    selectCamera(cameras[0].id);
  } else if (activeCameraId) {
    const active = cameras.find(c => c.id === activeCameraId);
    document.getElementById("viewer-source-label").textContent = sourceLabel(active);
    if (!active.online) selectCamera(activeCameraId);
  } else {
    el.viewerStream.removeAttribute("src");
    el.viewerStream.classList.remove("visible");
    el.viewerTitle.textContent = "Add a source to begin";
    el.viewerLocation.textContent = "";
    el.viewerPlaceholder.style.display = "block";
    el.viewerPlaceholder.textContent = "Upload a video in Cameras to start exploring.";
    document.getElementById("viewer-source-label").textContent = "NO SOURCE";
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

/* ==========================================================================
   Search / Dashboard
   ========================================================================== */

const STAT_ICONS = {
  events: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 2a6 6 0 00-6 6v4l-2 4h16l-2-4V8a6 6 0 00-6-6z"/></svg>',
  sources: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M17 10.5V7a1 1 0 00-1-1H4a1 1 0 00-1 1v10a1 1 0 001 1h12a1 1 0 001-1v-3.5l4 4v-11l-4 4z"/></svg>',
  footage: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 3"/></svg>',
};

let recentEventCount24h = 0;

function renderStatRow() {
  const online = cameras.filter((c) => c.online).length;
  el.statRow.innerHTML = `
    <div class="stat">
      <div class="stat-label">${STAT_ICONS.events} Recent events</div>
      <div class="stat-value">${recentEventCount24h}</div>
      <div class="stat-sub">last 24h · latest 30 events</div>
    </div>
    <div class="stat">
      <div class="stat-label">${STAT_ICONS.sources} Sources</div>
      <div class="stat-value">${cameras.length}</div>
      <div class="stat-sub">${online} online now</div>
    </div>
    <div class="stat">
      <div class="stat-label">${STAT_ICONS.footage} Analysis mode</div>
      <div class="stat-value">Visual<span class="stat-unit"> Q&A</span></div>
      <div class="stat-sub">Answers with saved snapshots</div>
    </div>
  `;
}

function renderFleetStrip() {
  el.fleetStrip.innerHTML = "";
  if (cameras.length === 0) {
    el.fleetStrip.innerHTML = `<div class="empty-state-inline">No sources yet — add one below.</div>`;
    return;
  }
  for (const cam of cameras) {
    const tile = document.createElement("button");
    tile.className = "fleet-tile";
    tile.setAttribute("aria-label", `Open ${cam.name} in live analyst`);
    tile.onclick = () => {
      switchView("chat");
      selectCamera(cam.id);
    };
    tile.innerHTML = `
      <div class="fleet-tile-frame">
        ${
          cam.online
            ? `<img data-thumb-for="${cam.id}" src="${API}/api/cameras/${cam.id}/snapshot" />`
            : `<div class="fleet-tile-offline">No signal</div>`
        }
        <div class="fleet-tile-live">${sourceLabel(cam)}</div>
      </div>
      <div class="fleet-tile-name"><span>${escapeHtml(cam.name)}</span><span class="tile-arrow">↗</span></div><div class="fleet-tile-location">${escapeHtml(cam.location || "No location set")}</div>
    `;
    el.fleetStrip.appendChild(tile);
  }
}

const recentQueries = [];

function renderRecentQueries() {
  if (recentQueries.length === 0) {
    el.recentQueries.innerHTML = `<div class="empty-state-inline">Nothing asked yet this session.</div>`;
    return;
  }
  el.recentQueries.innerHTML = "";
  for (const q of recentQueries.slice(0, 8)) {
    const item = document.createElement("div");
    item.className = "recent-query-item";
    item.innerHTML = `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 3"/></svg>
      <span class="recent-query-text">${escapeHtml(q)}</span>
    `;
    item.onclick = () => {
      el.searchInput.value = q;
      el.searchForm.requestSubmit();
    };
    el.recentQueries.appendChild(item);
  }
}

el.searchForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = el.searchInput.value.trim();
  if (!question || chatBusy) return;
  recentQueries.unshift(question);
  renderRecentQueries();
  document.getElementById("chat-scope").value = "fleet";
  switchView("chat");
  el.chatInput.value = question;
  el.chatForm.requestSubmit();
});

el.qaAddFile.addEventListener("click", () => openCameraModal());
el.qaAddCamera.addEventListener("click", () => switchView("directory"));
el.qaAddRule.addEventListener("click", () => switchView("rules"));

/* ---------- chat view ---------- */

function renderChatStrip() {
  el.chatStrip.innerHTML = "";
  for (const cam of cameras) {
    const item = document.createElement("button");
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
  document.getElementById("viewer-source-label").textContent = sourceLabel(cam);
  el.viewerStream.removeAttribute("src");
  el.viewerStream.classList.remove("visible");

  renderChatStrip();
}

async function loadPrompts() {
  const res = await fetch(`${API}/api/prompts`);
  const prompts = await res.json();
  el.promptChips.innerHTML = "";
  el.searchChips.innerHTML = "";
  for (const p of prompts) {
    if (el.promptChips.childElementCount < 3) {
      const chip = document.createElement("button");
      chip.className = "chip";
      chip.textContent = p;
      chip.onclick = () => {
        el.chatInput.value = p;
        el.chatForm.requestSubmit();
      };
      el.promptChips.appendChild(chip);
    }

    if (el.searchChips.childElementCount < 4) {
      const searchChip = document.createElement("button");
      searchChip.className = "chip";
      searchChip.textContent = p;
      searchChip.onclick = () => {
        el.searchInput.value = p;
        el.searchForm.requestSubmit();
      };
      el.searchChips.appendChild(searchChip);
    }
  }
}

async function loadChatHistory() {
  try {
    if (chatBusy) return;
    if (!activeChatSessionId) return;
    const [res, actionRes] = await Promise.all([
      fetch(`${API}/api/chat/history?session_id=${encodeURIComponent(activeChatSessionId)}`),
      fetch(`${API}/api/agent/actions?session_id=${encodeURIComponent(activeChatSessionId)}`),
    ]);
    if (!res.ok) return;
    const rows = await res.json(), actions = actionRes.ok ? await actionRes.json() : [];
    if (chatBusy) return;
    chatLog.length = 0;
    for (const row of rows) {
      chatLog.push({
        role: row.role === "assistant" ? "ai" : "user",
        text: row.text,
        camerasUsed: row.cameras_used,
        snapshot: row.snapshot,
      });
    }
    const lastAssistant = [...chatLog].reverse().find(message => message.role === "ai");
    if (lastAssistant && actions.length) lastAssistant.pendingActions = actions;
    renderChat();
  } catch (e) {
    // A missing transcript shouldn't block using the app.
  }
}

function renderChatSessions() {
  const container = document.getElementById("chat-session-list");
  container.innerHTML = chatSessions.map(session => `<div class="chat-session${session.id === activeChatSessionId ? " active" : ""}"><button data-session-id="${escapeHtml(session.id)}" title="${escapeHtml(session.title)}">${escapeHtml(session.title)}</button><button data-delete-session="${escapeHtml(session.id)}" title="Delete conversation" aria-label="Delete ${escapeHtml(session.title)}">×</button></div>`).join("");
}

async function createChatSession() {
  const res = await fetch(`${API}/api/chat/sessions`, {method:"POST"});
  if (!res.ok) throw new Error("Could not create a conversation");
  const session = await res.json();
  chatSessions.unshift(session);
  activeChatSessionId = session.id;
  localStorage.setItem("argus-chat-session", session.id);
  chatLog.length = 0;
  renderChatSessions();
  renderChat();
  return session;
}

async function loadChatSessions() {
  try {
    const res = await fetch(`${API}/api/chat/sessions`);
    if (!res.ok) throw new Error("Could not load conversations");
    chatSessions = await res.json();
    if (!chatSessions.length) await createChatSession();
    if (!chatSessions.some(session => session.id === activeChatSessionId)) activeChatSessionId = chatSessions[0].id;
    localStorage.setItem("argus-chat-session", activeChatSessionId);
    renderChatSessions();
    await loadChatHistory();
  } catch (_) {
    chatLog.length = 0;
    renderChat();
  }
}

document.getElementById("chat-session-list").addEventListener("click", async event => {
  const deleteButton = event.target.closest("[data-delete-session]");
  if (deleteButton) {
    const id = deleteButton.dataset.deleteSession;
    const res = await fetch(`${API}/api/chat/sessions/${encodeURIComponent(id)}`, {method:"DELETE"});
    if (!res.ok) return;
    chatSessions = chatSessions.filter(session => session.id !== id);
    if (activeChatSessionId === id) {
      if (!chatSessions.length) await createChatSession();
      else { activeChatSessionId = chatSessions[0].id; localStorage.setItem("argus-chat-session", activeChatSessionId); await loadChatHistory(); }
    }
    renderChatSessions();
    return;
  }
  const button = event.target.closest("[data-session-id]");
  if (!button || chatBusy) return;
  activeChatSessionId = button.dataset.sessionId;
  localStorage.setItem("argus-chat-session", activeChatSessionId);
  renderChatSessions();
  await loadChatHistory();
  switchView("chat");
});

function renderChat() {
  el.chatLog.innerHTML = "";
  el.promptChips.hidden = chatLog.length > 0;
  if (chatLog.length === 0) {
    el.chatLog.innerHTML = `<div class="chat-empty"><span class="empty-analyst">✳</span><strong>What should Argus handle?</strong><p>Investigate live cameras, search historical activity, verify evidence, check system health, or manage your monitoring workflow.</p><div class="agent-capability-row"><span>LIVE VISION</span><span>HISTORY</span><span>CPU</span><span>EVIDENCE</span></div></div>`;
    return;
  }
  for (const msg of chatLog) {
    appendMessageEl(msg.role, msg.text, msg.thinking, msg.camerasUsed, msg.snapshot, msg.scope, msg.evidence, msg.pendingActions);
  }
  el.chatLog.scrollTop = el.chatLog.scrollHeight;
}

function camerasUsedLabel(camIds) {
  if (!camIds || camIds.length === 0) return null;
  const names = camIds
    .map((id) => cameras.find((c) => c.id === id)?.name)
    .filter(Boolean);
  if (names.length === 0) return null;
  return names.length === 1 ? names[0] : names.join(", ");
}

function appendMessageEl(role, text, thinking, camerasUsed, snapshot, scope, evidence, pendingActions) {
  const bubble = document.createElement("div");
  bubble.className = `msg msg-${role}` + (thinking ? " thinking" : "");

  const body = document.createElement("div");
  body.textContent = text;
  bubble.appendChild(body);

  if (role === "ai" && scope && !thinking) {
    const tool = document.createElement("div");
    tool.className = "msg-tool";
    tool.textContent = ({camera_health: "CAMERA HEALTH", observations: "OBSERVATION SEARCH", verification: "VLM VERIFICATION", metadata: "CAMERA DIRECTORY", cpu: "CPU ANALYSIS", fleet: "LIVE FLEET", camera: "LIVE CAMERA"})[scope] || scope.replaceAll("_", " ").toUpperCase();
    bubble.insertBefore(tool, body);
  }

  // Show the frame the answer was actually based on, so the claim is
  // checkable rather than something the user has to take on faith.
  if (snapshot) {
    const img = document.createElement("img");
    img.className = "msg-snapshot";
    img.src = snapshot;
    img.alt = "Open the frame used for this answer";
    img.tabIndex = 0;
    img.setAttribute("role", "button");
    img.onclick = () => openEvidence(snapshot, camerasUsedLabel(camerasUsed) || "Analysis snapshot");
    img.onkeydown = e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); img.click(); } };
    bubble.appendChild(img);
  }

  if (evidence?.length) {
    const list = document.createElement("div");
    list.className = "msg-evidence-list";
    for (const item of evidence) {
      const evidenceItem = document.createElement("span");
      const time = item.timestamp ? new Date(item.timestamp).toLocaleString() : "Stored evidence";
      evidenceItem.textContent = `${item.camera_name || item.camera_id} · ${time}`;
      evidenceItem.title = `Observation ${item.observation_id || ""}`;
      list.appendChild(evidenceItem);
    }
    bubble.appendChild(list);
  }

  if (pendingActions?.length) {
    const actions = document.createElement("div");
    actions.className = "agent-actions";
    for (const action of pendingActions) {
      const card = document.createElement("div");
      card.className = `agent-action ${action.status}`;
      card.innerHTML = `<div><strong>${escapeHtml(action.summary)}</strong><span>${escapeHtml(action.action.replaceAll("_", " "))}</span></div><div class="agent-action-controls"></div>`;
      const controls = card.querySelector(".agent-action-controls");
      if (action.status === "pending") {
        for (const [decision, label] of [["reject", "Reject"], ["approve", "Approve"]]) {
          const button = document.createElement("button");
          button.type = "button";
          button.className = decision === "approve" ? "btn-primary" : "btn-ghost";
          button.textContent = label;
          button.onclick = async () => {
            controls.querySelectorAll("button").forEach(item => item.disabled = true);
            const res = await fetch(`${API}/api/agent/actions/${encodeURIComponent(action.id)}/decision`, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({decision})});
            if (res.ok) Object.assign(action, await res.json());
            else action.status = "failed";
            renderChat();
            loadCameras();
          };
          controls.appendChild(button);
        }
      } else {
        const state = document.createElement("span");
        state.textContent = action.status;
        controls.appendChild(state);
      }
      actions.appendChild(card);
    }
    bubble.appendChild(actions);
  }

  const label = camerasUsedLabel(camerasUsed);
  if (label || (role === "ai" && !thinking)) {
    const meta = document.createElement("div");
    meta.className = "msg-meta";

    if (label) {
      const span = document.createElement("span");
      span.textContent = `📷 ${label}`;
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
  if (chatBusy) return;
  if (chatLog.length && !confirm("Clear the whole conversation?")) return;
  try {
    const res = await fetch(`${API}/api/chat/history?session_id=${encodeURIComponent(activeChatSessionId)}`, { method: "DELETE" });
    if (!res.ok) throw new Error("Could not clear conversation");
  } catch (e) {
    document.getElementById("chat-progress").hidden = false;
    document.getElementById("chat-progress").textContent = "Could not clear the conversation. Please retry.";
    return;
  }
  chatLog.length = 0;
  renderChat();
});

document.getElementById("new-chat").addEventListener("click", async () => {
  if (chatBusy) return;
  try {
    await createChatSession();
    switchView("chat");
    if (innerWidth < 700) setSidebarCollapsed(true);
    el.chatInput.focus();
  } catch (error) {
    const progress = document.getElementById("chat-progress");
    progress.hidden = false;
    progress.textContent = "Could not start a new conversation. Please retry.";
  }
});

let chatBusy = false;

async function readChatStream(question, onEvent) {
  const res = await fetch(`${API}/api/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, focused_camera_id: activeCameraId, session_id: activeChatSessionId,
      scope: document.getElementById("chat-scope").value }),
  });
  if (!res.ok) {
    const data = await res.json();
    throw new Error(typeof data.detail === "string" ? data.detail : "Please check your question and selected source.");
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "", answer = null;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let boundary;
      while ((boundary = buffer.indexOf("\n\n")) !== -1) {
        const block = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const type = block.match(/^event: (.+)$/m)?.[1];
        const raw = block.match(/^data: (.+)$/m)?.[1];
        if (!raw) continue;
        const data = JSON.parse(raw);
        if (type === "error") throw new Error(data.detail);
        if (type === "answer") answer = data;
        onEvent(type, data);
      }
    }
  } finally { await reader.cancel().catch(() => {}); reader.releaseLock(); }
  if (!answer) throw new Error("The connection ended before the answer was saved. Please retry.");
  return answer;
}

el.chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = el.chatInput.value.trim();
  if (!question || chatBusy) return;
  chatBusy = true;
  chatLog.push({ role: "user", text: question });
  renderChat();
  el.chatInput.value = "";
  el.chatSend.disabled = el.searchSubmit.disabled = el.btnClearChat.disabled = true;
  const progress = document.getElementById("chat-progress");
  progress.hidden = false;
  const started = Date.now();
  let status = "Reading recent frames", streamed = "";
  const updateProgress = () => progress.textContent = `${status} · ${Math.floor((Date.now() - started) / 1000)}s`;
  updateProgress();
  const timer = setInterval(updateProgress, 1000);
  const bubble = appendMessageEl("ai", "Planning tools and gathering evidence…", true);
  try {
    const data = await readChatStream(question, (type, data) => {
      if (type === "status") status = data.text;
      if (type === "token") {
        status = "Writing answer";
        streamed += data.text;
        bubble.classList.remove("thinking");
        bubble.firstChild.textContent = streamed;
        el.chatLog.scrollTop = el.chatLog.scrollHeight;
      }
    });
    chatLog.push({ role: "ai", text: data.answer, camerasUsed: data.cameras_used, snapshot: data.snapshot, scope: data.scope, evidence: data.evidence, pendingActions: data.pending_actions });
    const resultType = data.evidence?.length ? `${data.evidence.length} evidence item${data.evidence.length === 1 ? "" : "s"}` : data.snapshot ? "Evidence attached" : (data.scope || "Completed");
    progress.textContent = `Agent complete · ${data.elapsed_seconds}s · ${resultType}`;
  } catch (err) {
    chatLog.push({ role: "ai", text: err.message || "Could not reach the analysis server." });
    el.chatInput.value = question;
    progress.textContent = "Answer unavailable. Your question is ready to retry.";
  } finally {
    clearInterval(timer);
    chatBusy = false;
    el.chatSend.disabled = el.searchSubmit.disabled = el.btnClearChat.disabled = false;
    renderChat();
    el.chatInput.focus();
  }
});

function sourceLabel(cam) {
  return !cam.online ? "OFFLINE" : cam.source_type === "file" ? "DEMO FOOTAGE" : "LIVE CAMERA";
}

function openEvidence(src, caption) {
  document.getElementById("evidence-image").src = src;
  document.getElementById("evidence-caption").textContent = caption + " · Saved analysis frame";
  document.getElementById("evidence-dialog").showModal();
}
document.getElementById("evidence-close").onclick = () => document.getElementById("evidence-dialog").close();
document.getElementById("evidence-dialog").onclick = e => { if (e.target === e.currentTarget) e.currentTarget.close(); };
el.viewerStream.onload = () => { el.viewerStream.classList.add("visible"); el.viewerPlaceholder.style.display = "none"; };
el.viewerStream.onerror = () => { el.viewerStream.classList.remove("visible"); el.viewerPlaceholder.style.display = "block"; el.viewerPlaceholder.textContent = "Source unavailable. Reopen the camera to reconnect."; };
document.getElementById("open-analyst").onclick = () => switchView("chat");
document.getElementById("view-all-events").onclick = () => switchView("alerts");
document.addEventListener("keydown", e => {
  if (e.key === "Escape") { closeCameraModal(); closeEditModal(); }
});

function renderOverviewActivity() {
  const container = document.getElementById("overview-activity");
  container.innerHTML = "";
  if (!recentUnfiltered.length) {
    container.innerHTML = '<div class="activity-empty"><span>◎</span><strong>A quiet moment.</strong><p>Detections will appear here as Argus observes your cameras.</p></div>';
    return;
  }
  for (const ev of recentUnfiltered.slice(0, 4)) {
    const row = document.createElement("button");
    row.className = "activity-item";
    row.innerHTML = `<span class="activity-dot ${escapeHtml(ev.severity)}"></span><div><strong>${escapeHtml(ev.category)}</strong><p>${escapeHtml(eventCameraLabel(ev.camera_id))}</p><time>${escapeHtml(new Date(ev.created_at).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"}))}</time></div><span>↗</span>`;
    row.onclick = () => ev.snapshot ? openEvidence(ev.snapshot, `${eventCameraLabel(ev.camera_id)} · ${new Date(ev.created_at).toLocaleString()}`) : switchView("alerts");
    container.appendChild(row);
  }
}

/* ---------- directory (camera CRUD) ---------- */

function renderCamerasTable() {
  el.camerasTable.innerHTML = "";
  if (cameras.length === 0) {
    el.camerasTable.innerHTML = `<div class="empty-state">No sources yet. Click "+ Add source" to create one.</div>`;
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
  if (!confirm(`Remove source "${name}"? This cannot be undone.`)) return;
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

let cameraSourceType = "file";

function setCameraSourceType(sourceType) {
  cameraSourceType = sourceType;
  document.querySelectorAll(".field-tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.source === sourceType));
  const videoField = document.getElementById("field-video");
  const rtspField = document.getElementById("field-rtsp");
  const videoInput = videoField.querySelector("input");
  const rtspInput = rtspField.querySelector("input");
  videoField.hidden = sourceType !== "file";
  rtspField.hidden = sourceType !== "rtsp";
  videoInput.disabled = sourceType !== "file";
  videoInput.required = sourceType === "file";
  rtspInput.disabled = sourceType !== "rtsp";
  rtspInput.required = sourceType === "rtsp";
}

document.querySelectorAll(".field-tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    setCameraSourceType(tab.dataset.source);
  });
});

function openCameraModal() {
  el.cameraForm.reset();
  setCameraSourceType("file");
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
  formData.set("source_type", cameraSourceType);

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
      throw new Error(data.detail || "Failed to add source.");
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
    el.btnSubmitCamera.textContent = "Add source";
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
  f.elements.source_type.value = cam.source_type || "file";
  f.elements.source_path.value = cam.source_type === "rtsp" ? cam.source_path || "" : "";
  document.getElementById("edit-field-rtsp").hidden = cam.source_type !== "rtsp";
  f.elements.source_path.required = cam.source_type === "rtsp";
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
    source_type: f.elements.source_type.value,
  };
  if (payload.source_type === "rtsp") payload.source_path = f.elements.source_path.value.trim();

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
        <div class="health-card-title">Sources</div>
        ${
          (data.cameras || []).length === 0
            ? `<div class="health-muted">No sources configured.</div>`
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
let eventLimit = 300;
let eventsLivePaused = false;

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

function renderEventKpis() {
  const critical = recentUnfiltered.filter((event) => event.severity === "critical").length;
  const warning = recentUnfiltered.filter((event) => event.severity === "warning").length;
  const sources = new Set(recentUnfiltered.map((event) => event.camera_id)).size;
  const latest = recentUnfiltered[0]?.created_at;
  el.eventsKpis.innerHTML = [
    ["Visible events", latestEvents.length, "Current stream filter"],
    ["Critical", critical, "Recent unfiltered events"],
    ["Warnings", warning, "Recent unfiltered events"],
    ["Reporting sources", sources, latest ? `Latest ${new Date(latest).toLocaleTimeString()}` : "No events yet"],
  ].map(([label, value, note]) => `<div class="activity-kpi"><span class="stat-label">${escapeHtml(label)}</span><strong>${escapeHtml(String(value))}</strong><small>${escapeHtml(note)}</small></div>`).join("");
}

function updateEventsLiveState() {
  el.eventsLiveState.classList.toggle("paused", eventsLivePaused);
  el.eventsLiveState.innerHTML = `<span class="live-dot"></span> ${eventsLivePaused ? "PAUSED" : "LIVE"}`;
  el.eventsLiveToggle.textContent = eventsLivePaused ? "Resume stream" : "Pause stream";
}

async function pollRecentEvents() {
  try {
    const res = await fetch(`${API}/api/events?limit=30`);
    if (!res.ok) return;
    const rows = await res.json();
    const seenIds = new Set(recentUnfiltered.map((e) => e.id));
    const newCritical = rows.find((e) => e.severity === "critical" && !seenIds.has(e.id));
    recentUnfiltered = rows;
    renderOverviewActivity();
    if (currentView === "alerts" && !eventsLivePaused) loadEvents();
    renderAlertsBadge();
    renderEventKpis();

    const dayAgo = Date.now() - 24 * 60 * 60 * 1000;
    recentEventCount24h = rows.filter((e) => e.created_at && new Date(e.created_at).getTime() >= dayAgo).length;
    if (currentView === "dashboard") renderStatRow();

    if (newCritical) notifyCritical(newCritical);
  } catch (e) {
    // Transient errors shouldn't break other views.
  }
}
setInterval(pollRecentEvents, 3000);

function buildEventsQuery() {
  const params = new URLSearchParams({ limit: String(eventLimit) });
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
    renderEventKpis();
    el.eventsLastUpdated.textContent = `Updated ${new Date().toLocaleTimeString()}`;
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

el.eventsLiveToggle.addEventListener("click", () => {
  eventsLivePaused = !eventsLivePaused;
  updateEventsLiveState();
  if (!eventsLivePaused && currentView === "alerts") loadEvents();
});
el.eventsMarkSeen.addEventListener("click", markEventsSeen);
el.eventsFocusCritical.addEventListener("click", () => {
  eventFilters.severity = "critical";
  el.eventsSeverityFilter.value = "critical";
  loadEvents();
});
el.eventsFocusRecent.addEventListener("click", () => {
  eventLimit = 30;
  loadEvents();
});
el.eventsClearFilters.addEventListener("click", () => {
  eventLimit = 300;
  eventFilters.q = eventFilters.camera_id = eventFilters.severity = "";
  el.eventsSearch.value = "";
  el.eventsCameraFilter.value = "";
  el.eventsSeverityFilter.value = "";
  loadEvents();
});
updateEventsLiveState();

el.historyAskForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = el.historyAskInput.value.trim();
  if (!question) return;
  el.historyAskAnswer.hidden = false;
  el.historyAskAnswer.textContent = "Searching…";
  try {
    const res = await fetch(`${API}/api/events/ask?${new URLSearchParams({ question })}`);
    const data = await res.json();
    if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Could not answer that question.");
    el.historyAskAnswer.textContent = data.answer;
    latestEvents = data.events;
    renderEvents();
  } catch (err) {
    el.historyAskAnswer.textContent = err.message;
  }
});

function renderAlertsBadge() {
  const lastSeen = getLastSeenEventId();
  const unseen = recentUnfiltered.filter((e) => e.id > lastSeen && e.severity !== "info");
  el.railAlertsBadge.hidden = unseen.length === 0;
  el.railAlertsBadge.textContent = unseen.length;
  el.railAlertsBadge.classList.toggle("warn", unseen.some((e) => e.severity === "critical"));
}

function eventCameraLabel(camId) {
  return cameras.find((c) => c.id === camId)?.name || "Unknown source";
}

function renderEvents() {
  el.eventsList.innerHTML = "";
  el.eventsStreamCount.textContent = `${latestEvents.length} event${latestEvents.length === 1 ? "" : "s"}`;
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
    const thumb = row.querySelector("img");
    if (thumb) {
      thumb.tabIndex = 0;
      thumb.setAttribute("role", "button");
      thumb.alt = "Open event evidence";
      thumb.onclick = () => openEvidence(ev.snapshot, `${eventCameraLabel(ev.camera_id)} · ${time}`);
      thumb.onkeydown = e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); thumb.click(); } };
    }
    el.eventsList.appendChild(row);
  }
}

/* ---------- events: charts ---------- */
// Categorical hues validated (dataviz skill's validate_palette.js) against
// this app's actual card surface (#12141d) in dark mode.
const CAMERA_CHART_COLORS = ["#eee", "#bbb", "#888", "#666", "#ddd", "#aaa", "#999", "#777"];

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
    <div class="chart-card-title">Events by source</div>
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
    `<option value="">All sources</option>` +
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

let cpuClasses = [];
let cpuDefaultSeverity = {};
async function loadCpuClasses() {
  try {
    const res = await fetch(`${API}/api/alert-rules/cpu-classes`);
    if (!res.ok) return;
    const data = await res.json();
    cpuClasses = data.classes;
    cpuDefaultSeverity = data.default_severity || {};
    el.alertRuleClass.innerHTML = cpuClasses.map(c => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join("");
    el.alertRuleCpuNote.textContent = data.available
      ? "Matches are sampled and should be reviewed."
      : "Object detection model is not available on this server yet.";
    applyCpuDefaultSeverity();
  } catch (e) {
    // Transient errors shouldn't break other views.
  }
}

function applyCpuDefaultSeverity() {
  const suggested = cpuDefaultSeverity[el.alertRuleClass.value];
  if (suggested) el.alertRuleSeverity.value = suggested;
}
el.alertRuleClass.addEventListener("change", applyCpuDefaultSeverity);

function updateAlertRuleFormMode() {
  const isCpu = el.alertRuleSource.value === "cpu";
  el.alertRuleClass.hidden = !isCpu;
  el.alertRuleTarget.hidden = isCpu;
  el.alertRuleTarget.required = !isCpu;
  el.alertRuleCpuNote.hidden = !isCpu;
  el.alertRuleSeverity.hidden = !isCpu; // VLM rule severity is fixed server-side ("warning")
  if (isCpu) applyCpuDefaultSeverity();
}
el.alertRuleSource.addEventListener("change", updateAlertRuleFormMode);
updateAlertRuleFormMode();

function renderAlertRulesList() {
  el.alertRulesList.innerHTML = "";
  if (alertRules.length === 0) {
    el.alertRulesList.innerHTML = `<div class="empty-state">No custom alert rules yet.</div>`;
    return;
  }
  for (const rule of alertRules) {
    const row = document.createElement("div");
    row.className = "alert-rule-row" + (rule.enabled ? "" : " disabled");
    const camLabel = rule.camera_id ? eventCameraLabel(rule.camera_id) : "All sources";
    const sourceLabel = rule.source === "cpu" ? "CPU" : "AI model";
    const severityOptions = ["info", "warning", "critical"]
      .map(s => `<option value="${s}" ${rule.severity === s ? "selected" : ""}>${s[0].toUpperCase()}${s.slice(1)}</option>`)
      .join("");
    row.innerHTML = `
      <span class="alert-rule-target">"${escapeHtml(rule.target)}" <em>(${sourceLabel})</em></span>
      <span class="alert-rule-camera">${escapeHtml(camLabel)}</span>
      ${rule.source === "cpu"
        ? `<select class="events-select alert-rule-severity" data-action="severity">${severityOptions}</select>`
        : `<span class="alert-rule-camera">Warning</span>`}
      <label class="alert-rule-toggle">
        <input type="checkbox" ${rule.enabled ? "checked" : ""} />
        <span>${rule.enabled ? "Enabled" : "Disabled"}</span>
      </label>
      <button class="btn-icon danger" data-action="delete">Delete</button>
    `;
    row.querySelector('input[type="checkbox"]').addEventListener("change", (e) => {
      updateAlertRule(rule.id, { enabled: e.target.checked });
    });
    row.querySelector('[data-action="severity"]')?.addEventListener("change", (e) => {
      updateAlertRule(rule.id, { severity: e.target.value });
    });
    row.querySelector('[data-action="delete"]').addEventListener("click", () => {
      deleteAlertRule(rule.id);
    });
    el.alertRulesList.appendChild(row);
  }
}

async function updateAlertRule(ruleId, patch) {
  const res = await fetch(`${API}/api/alert-rules/${ruleId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  if (!res.ok) { alert("Could not update the rule. Please retry."); }
  await loadAlertRules();
}

async function deleteAlertRule(ruleId) {
  const res = await fetch(`${API}/api/alert-rules/${ruleId}`, { method: "DELETE" });
  if (!res.ok) { alert("Could not delete the rule. Please retry."); }
  await loadAlertRules();
}

el.alertRuleForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const source = el.alertRuleSource.value;
  const target = source === "cpu" ? el.alertRuleClass.value : el.alertRuleTarget.value.trim();
  if (!target) return;
  const camera_id = el.alertRuleCamera.value || null;
  const body = { camera_id, source, target };
  if (source === "cpu") body.severity = el.alertRuleSeverity.value;
  const res = await fetch(`${API}/api/alert-rules`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) { const data = await res.json().catch(() => ({})); alert(typeof data.detail === "string" ? data.detail : "Could not create the rule."); return; }
  el.alertRuleTarget.value = "";
  await loadAlertRules();
});

/* ---------- utils ---------- */

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}

/* ---------- Hardware-independent AI setup ---------- */
let runtimeState = { settings: {}, hardware: {}, profiles: [], active_profile_id: null };
let runtimeEditingId = null;

function runtimeFields() {
  const provider = document.getElementById("runtime-provider").value;
  document.getElementById("runtime-api-fields").hidden = provider !== "compatible";
  document.getElementById("runtime-embedded-note").hidden = provider !== "embedded";
  document.getElementById("runtime-test").disabled = provider === "none";
}
document.getElementById("runtime-provider").onchange = runtimeFields;
document.getElementById("setup-shortcut").onclick = () => switchView("runtime");

function runtimeProviderLabel(provider) {
  return {none: "CPU tools only", compatible: "Compatible vision API", embedded: "Embedded Qwen"}[provider] || provider;
}

function runtimeProfileStatus(profile, settings) {
  if (profile.provider === "none") return ["CPU ready", "ok"];
  if (profile.active && settings.ready) return ["Ready", "ok"];
  if (profile.last_test_status === "failed") return ["Test failed", "failed"];
  if (profile.last_test_status === "passed") return ["Verified · retest on use", "pending"];
  return ["Needs test", "pending"];
}

function renderRuntimeProfiles() {
  const list = document.getElementById("runtime-profiles-list");
  const profiles = runtimeState.profiles || [];
  document.getElementById("runtime-setup-count").textContent = `${profiles.length} setup${profiles.length === 1 ? "" : "s"}`;
  if (!profiles.length) {
    list.innerHTML = `<div class="runtime-empty">No saved vision connections yet.<br />Create one to start testing a model.</div>`;
    return;
  }
  list.innerHTML = profiles.map(profile => {
    const [status, statusClass] = runtimeProfileStatus(profile, runtimeState.settings);
    const detail = profile.provider === "none" ? "OpenCV and CPU tools" : `${profile.model || "No model selected"} · ${profile.base_url || "Embedded runtime"}`;
    return `<article class="runtime-profile ${profile.active ? "active" : ""}">
      <div class="runtime-profile-main"><div class="runtime-profile-title"><strong>${escapeHtml(profile.name)}</strong>${profile.active ? `<em>active</em>` : ""}</div><div class="runtime-profile-detail">${escapeHtml(detail)}</div></div>
      <div class="runtime-profile-actions"><span class="runtime-status ${statusClass}">${escapeHtml(status)}</span><button type="button" class="btn-ghost" data-runtime-action="edit" data-profile-id="${escapeHtml(profile.id)}">Edit</button>${profile.active ? "" : `<button type="button" class="btn-ghost" data-runtime-action="activate" data-profile-id="${escapeHtml(profile.id)}">Use</button>`}<button type="button" class="btn-ghost" data-runtime-action="delete" data-profile-id="${escapeHtml(profile.id)}">Delete</button></div>
    </article>`;
  }).join("");
}

function renderRuntimeActive() {
  const root = document.getElementById("runtime-active-card");
  const settings = runtimeState.settings || {};
  const profile = (runtimeState.profiles || []).find(item => item.id === runtimeState.active_profile_id);
  if (!profile) {
    root.innerHTML = `<div class="runtime-active-main"><div class="runtime-active-icon">CPU</div><div class="runtime-active-copy"><strong>CPU tools only</strong><span>OpenCV analysis is available without a vision model</span></div></div><div class="runtime-active-meta"><div><strong>Active path</strong><span class="runtime-status ok">Available now</span></div><button class="btn-ghost" type="button" data-runtime-action="new">Add vision setup</button></div>`;
    return;
  }
  const [status, statusClass] = runtimeProfileStatus(profile, settings);
  const subtitle = profile.provider === "none" ? "OpenCV analysis is available without a vision model" : `${runtimeProviderLabel(profile.provider)} · ${profile.model || "model not selected"}`;
  const action = profile.active && settings.ready ? "" : `<button class="btn-primary" type="button" data-runtime-action="edit" data-profile-id="${escapeHtml(profile.id)}">Configure</button>`;
  root.innerHTML = `<div class="runtime-active-main"><div class="runtime-active-icon">AI</div><div class="runtime-active-copy"><strong>${escapeHtml(profile.name)}</strong><span>${escapeHtml(subtitle)}</span></div></div><div class="runtime-active-meta"><div><strong>Current state</strong><span class="runtime-status ${statusClass}">${escapeHtml(status)}</span></div>${action}</div>`;
}

function renderRuntimePipeline() {
  const settings = runtimeState.settings || {};
  const profile = (runtimeState.profiles || []).find(item => item.id === runtimeState.active_profile_id);
  const ai = settings.provider && settings.provider !== "none";
  const steps = [
    ["Setup", ai ? "Saved" : "CPU mode", ai],
    ["Endpoint", ai ? (settings.provider === "embedded" ? "Embedded" : (settings.base_url ? "Configured" : "Missing")) : "Skipped", !ai || settings.provider === "embedded" || Boolean(settings.base_url)],
    ["Model", ai ? (settings.model || "Missing") : "No model", !ai || Boolean(settings.model)],
    ["Vision test", ai ? (settings.ready ? "Passed" : (profile?.last_test_status === "failed" ? "Failed" : "Needs test")) : "Not required", !ai || settings.ready],
  ];
  document.getElementById("runtime-pipeline").innerHTML = steps.map(([name, value, ok]) => `<div class="runtime-pipeline-step ${ok ? "ok" : (value === "Failed" ? "failed" : "")}"><strong>${escapeHtml(name)}</strong><span>${escapeHtml(value)}</span></div>`).join("");
  const tested = profile?.last_tested_at ? new Date(profile.last_tested_at).toLocaleString([], {month:"short", day:"numeric", hour:"2-digit", minute:"2-digit"}) : "Awaiting test";
  document.getElementById("runtime-pipeline-time").textContent = tested;
}

function renderRuntimeCapabilities() {
  const settings = runtimeState.settings || {};
  const hardware = runtimeState.hardware || {};
  document.getElementById("runtime-hardware").textContent = `${hardware.cpu_count || "—"} CPU threads · ${hardware.nvidia_gpu || "NVIDIA GPU not detected"}. ${hardware.recommendation || ""}`;
  document.getElementById("runtime-capabilities").innerHTML = [
    ["Camera viewing & uploads", "Available · CPU"],
    ["Motion, watch areas & quality", "Available · OpenCV CPU"],
    ["Measurement reports & CSV export", "Available · no model"],
    ["Saved activity & snapshots", "Available · database"],
    ["Visual chat", settings.ready ? "Connected" : "Needs tested vision connection"],
    ["Vision-model detections", settings.ready && settings.monitor_enabled ? "Enabled" : "Not enabled"],
    ["NVIDIA GPU", hardware.nvidia_gpu || "Not detected"],
    ["Historical video search", "Not included in this MVP"],
  ].map(([name, value]) => `<div class="runtime-capability"><span>${escapeHtml(name)}</span><span>${escapeHtml(value)}</span></div>`).join("");
}

function renderRuntimeUsage(usage) {
  usage = usage || {total_requests: 0, last_24h: 0, last_7d: 0, daily: []};
  document.getElementById("runtime-stats").innerHTML = [["Last 24h", usage.last_24h, "chat answers"], ["Last 7 days", usage.last_7d, "chat answers"], ["All time", usage.total_requests, "saved answers"]].map(([label, value, sub]) => `<div class="runtime-stat"><div class="runtime-stat-label">${label}</div><div class="runtime-stat-value">${value}</div><div class="runtime-stat-sub">${sub}</div></div>`).join("");
  document.getElementById("runtime-usage-updated").textContent = usage.last_response_at ? `Last answer ${new Date(usage.last_response_at).toLocaleString([], {month:"short", day:"numeric", hour:"2-digit", minute:"2-digit"})}` : "No requests yet";
  const daily = usage.daily || [];
  const max = Math.max(1, ...daily.map(item => item.requests));
  document.getElementById("runtime-usage-chart").innerHTML = daily.length ? daily.map(item => `<div class="runtime-chart-bar" title="${item.requests} answer${item.requests === 1 ? "" : "s"}"><i style="height:${Math.max(4, Math.round(item.requests / max * 100))}%"></i><span>${escapeHtml(item.date.slice(5))}</span></div>`).join("") : `<span class="runtime-chart-empty">Usage will appear after the first visual answer.</span>`;
}

function fillRuntimeForm(profile) {
  profile = profile || {name: "", provider: "none", base_url: "", model: "", allow_remote: false, monitor_enabled: false, has_api_key: false};
  document.getElementById("runtime-name").value = profile.name || "";
  document.getElementById("runtime-provider").value = profile.provider || "none";
  document.getElementById("runtime-url").value = profile.base_url || "";
  document.getElementById("runtime-model").value = profile.model || "";
  document.getElementById("runtime-key").value = "";
  document.getElementById("runtime-key").placeholder = profile.has_api_key ? "Key saved · leave blank to keep" : "Optional for a local server";
  document.getElementById("runtime-remote").checked = Boolean(profile.allow_remote);
  document.getElementById("runtime-monitor").checked = Boolean(profile.monitor_enabled);
  runtimeFields();
}

function openRuntimeEditor(profile) {
  runtimeEditingId = profile?.id || null;
  document.getElementById("runtime-editor").hidden = false;
  document.getElementById("runtime-editor-eyebrow").textContent = profile ? "EDIT SAVED CONNECTION" : "NEW CONNECTION";
  document.getElementById("runtime-editor-title").textContent = profile ? `Edit ${profile.name}` : "Create an AI setup";
  fillRuntimeForm(profile);
  document.getElementById("runtime-editor").scrollIntoView({behavior:"smooth", block:"start"});
}

async function loadRuntime() {
  try {
    const [res, usageRes] = await Promise.all([fetch(`${API}/api/runtime`), fetch(`${API}/api/runtime/usage`)]);
    if (!res.ok) throw new Error("Could not load AI settings.");
    runtimeState = await res.json();
    renderRuntimeActive(); renderRuntimeProfiles(); renderRuntimePipeline(); renderRuntimeCapabilities();
    if (usageRes.ok) renderRuntimeUsage(await usageRes.json());
    if (!document.getElementById("runtime-editor").hidden && runtimeEditingId) {
      const profile = (runtimeState.profiles || []).find(item => item.id === runtimeEditingId);
      if (profile) fillRuntimeForm(profile);
    }
  } catch (err) { document.getElementById("runtime-message").textContent = err.message; }
}

async function saveRuntime(activate = true) {
  const res = await fetch(`${API}/api/runtime`, {method: "PUT", headers: {"Content-Type":"application/json"}, body: JSON.stringify({
    name: document.getElementById("runtime-name").value.trim(),
    profile_id: runtimeEditingId,
    create_new: !runtimeEditingId,
    activate,
    provider: document.getElementById("runtime-provider").value,
    base_url: document.getElementById("runtime-url").value.trim(),
    model: document.getElementById("runtime-model").value.trim(),
    api_key: document.getElementById("runtime-key").value || null,
    allow_remote: document.getElementById("runtime-remote").checked,
    monitor_enabled: document.getElementById("runtime-monitor").checked,
  })});
  const data = await res.json();
  if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Check the AI connection settings.");
  document.getElementById("runtime-key").value = "";
}
let runtimeBusy = false;
async function configureAI(test) {
  if (runtimeBusy) return;
  runtimeBusy = true;
  const msg = document.getElementById("runtime-message");
  const save = document.getElementById("runtime-save"), testButton = document.getElementById("runtime-test");
  save.disabled = testButton.disabled = true;
  msg.textContent = test ? "Saving and testing a synthetic image. This may take a moment…" : "Saving setup…";
  try {
    const existing = (runtimeState.profiles || []).find(item => item.id === runtimeEditingId);
    await saveRuntime(test || !existing || existing.active);
    if (test) {
      const res = await fetch(`${API}/api/runtime/test`, {method:"POST"});
      const data = await res.json();
      if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Connection test failed.");
      msg.textContent = `${data.message} (${data.seconds}s)`;
    } else { msg.textContent = "Setup saved. Use it from the list when you are ready."; document.getElementById("runtime-editor").hidden = true; }
    await loadRuntime();
    await pollStatus();
    if (test) document.getElementById("runtime-editor").hidden = true;
  } catch (err) { msg.textContent = err.message; }
  finally { runtimeBusy = false; save.disabled = false; runtimeFields(); }
}
document.getElementById("runtime-form").onsubmit = e => { e.preventDefault(); configureAI(false); };
document.getElementById("runtime-test").onclick = () => configureAI(true);
document.getElementById("runtime-new").onclick = () => openRuntimeEditor();
document.getElementById("runtime-cancel").onclick = () => { document.getElementById("runtime-editor").hidden = true; runtimeEditingId = null; };
document.getElementById("preset-ollama").onclick = () => { document.getElementById("runtime-provider").value = "compatible"; document.getElementById("runtime-url").value = "http://localhost:11434/v1"; document.getElementById("runtime-model").value = "moondream"; runtimeFields(); };
document.getElementById("preset-llama").onclick = () => { document.getElementById("runtime-provider").value = "compatible"; document.getElementById("runtime-url").value = "http://localhost:8080/v1"; document.getElementById("runtime-model").value = "your-vision-model"; runtimeFields(); };
document.getElementById("runtime-active-card").onclick = e => { const button = e.target.closest("[data-runtime-action]"); if (!button) return; const profile = (runtimeState.profiles || []).find(item => item.id === button.dataset.profileId); openRuntimeEditor(profile); };
document.getElementById("runtime-profiles-list").onclick = async e => {
  const button = e.target.closest("[data-runtime-action]");
  if (!button) return;
  const action = button.dataset.runtimeAction, id = button.dataset.profileId;
  const profile = (runtimeState.profiles || []).find(item => item.id === id);
  if (action === "new") return openRuntimeEditor();
  if (action === "edit") return openRuntimeEditor(profile);
  if (action === "activate") {
    const res = await fetch(`${API}/api/runtime/profiles/${encodeURIComponent(id)}/activate`, {method:"POST"});
    if (!res.ok) { document.getElementById("runtime-message").textContent = "Could not activate that setup."; return; }
    await loadRuntime(); await pollStatus(); return;
  }
  if (action === "delete" && profile && confirm(`Delete the ${profile.name} setup?`)) {
    const res = await fetch(`${API}/api/runtime/profiles/${encodeURIComponent(id)}`, {method:"DELETE"});
    if (!res.ok) { document.getElementById("runtime-message").textContent = "Could not delete that setup."; return; }
    await loadRuntime(); await pollStatus();
  }
};

/* ---------- init ---------- */

loadCameras();
loadPrompts();
renderChat();
renderRecentQueries();
renderStatRow();
loadChatSessions();
pollRecentEvents();

// Deep-linkable views, e.g. sentinelvision/?view=alerts — lets a specific
// page be bookmarked or shared instead of always landing on Search.
const deepLinkView = new URLSearchParams(location.search).get("view");
if (deepLinkView && VIEW_TITLES[deepLinkView]) {
  switchView(deepLinkView);
}
