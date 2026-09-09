/* CPU tools run on every supported server without a vision model. */
let cpuRows = [], cpuId = null, cpuPollBusy = false, cpuSamples = [], cpuLastSample = null;
let detectionLogLastId = null, detectionLogPollBusy = false;
const cpuEl = id => document.getElementById(`cpu-${id}`);

async function loadCpu(reset = false) {
  if (cpuPollBusy) return;
  cpuPollBusy = true;
  try {
    const res = await fetch(`${API}/api/cpu`);
    if (!res.ok) throw new Error('CPU analysis is unavailable. Check server health.');
    cpuRows = await res.json();
    const select = cpuEl('camera');
    const signature = cpuRows.map(r => r.camera_id + r.name).join('|');
    if (select.dataset.signature !== signature) {
      select.replaceChildren(...cpuRows.map(r => new Option(r.name, r.camera_id)));
      select.dataset.signature = signature;
    }
    if (!cpuRows.some(r => r.camera_id === cpuId)) { cpuId = cpuRows[0]?.camera_id || null; reset = true; }
    select.value = cpuId || '';
    const row = cpuRows.find(r => r.camera_id === cpuId);
    cpuEl('save').disabled = !row;
    if (!row) {
      cpuEl('engine-state').textContent = 'Add a video source in Cameras to get started.';
      cpuEl('preview').removeAttribute('src');
      cpuEl('preview-message').hidden = false;
      cpuEl('preview-message').textContent = 'No sources configured';
      cpuEl('metrics').replaceChildren();
      cpuEl('download').disabled = true;
      return;
    }
    if (reset) populateCpuForm(row.settings);
    renderCpu(row);
  } catch (err) { cpuEl('engine-state').textContent = err.message; }
  finally { cpuPollBusy = false; }
}

function populateCpuForm(s) {
  for (const [id, key] of [['enabled','enabled'],['motion-alerts','motion_alerts'],['quality-alerts','quality_alerts'],['face-alerts','face_alerts'],['people-alerts','people_alerts'],['object-alerts','object_alerts']]) cpuEl(id).checked = s[key];
  cpuEl('threshold').value = +(s.motion_threshold * 100).toFixed(2);
  cpuEl('cooldown').value = s.cooldown_seconds;
  cpuEl('dark').value = s.dark_threshold;
  cpuEl('blur').value = s.blur_threshold;
  for (const key of ['x','y','width','height']) cpuEl(key).value = Math.round(s.area[key] * 100);
  drawCpuArea();
}

function renderCpu(row) {
  const s = row.status;
  const valid = row.settings.enabled && s.online && !s.warming_up && s.motion_percent !== undefined;
  cpuEl('engine-state').textContent = !row.settings.enabled ? 'CPU analysis disabled for this source' : !s.online ? 'Source offline' : s.warming_up ? (s.message || 'Warming up · collecting frames') : `OpenCV · ${s.analysis_ms} ms / sample · ${new Date(s.checked_at).toLocaleTimeString()}`;
  const metrics = [
    ['Motion in watch area', valid ? `${s.motion_percent}%` : '—', valid ? (s.motion ? 'Above trigger' : 'Below trigger') : 'Waiting for data'],
    ['Brightness', valid ? `${s.brightness}` : '—', valid ? (s.low_light ? 'Low-light indicator' : 'Above low-light threshold') : 'Scale: 0–255'],
    ['Edge detail', valid ? `${s.sharpness}` : '—', valid ? (s.low_detail ? 'Low detail / possible blur' : 'Above detail threshold') : 'Not a calibrated blur score'],
    ['Motion regions', valid ? String(s.boxes.length) : '—', 'Connected pixel regions, not people'],
    ['Faces detected', valid && s.face_count !== undefined ? String(s.face_count) : '—', 'Haar cascade, sampled every few seconds'],
    ['People detected', valid && s.people_count !== undefined ? String(s.people_count) : '—', 'HOG pedestrian detector, sampled every few seconds'],
    ['Objects detected', valid && s.objects !== undefined ? String(s.objects.length) : '—', valid && s.objects && s.objects.length ? [...new Set(s.objects.map(o => o.class))].join(', ') : 'Sampled every few seconds'],
  ];
  cpuEl('metrics').innerHTML = metrics.map(([label,value,note]) => `<div class="stat"><div class="stat-label">${escapeHtml(label)}</div><div class="stat-value">${escapeHtml(value)}</div><div class="stat-sub">${escapeHtml(note)}</div></div>`).join('');
  cpuEl('download').disabled = !valid;
  if (valid) {
    cpuEl('preview').src = `${API}/api/cpu/${cpuId}/snapshot?t=${Date.now()}`;
    if (s.sampled_at !== cpuLastSample) {
      cpuSamples.push(s.motion_percent); cpuSamples = cpuSamples.slice(-60); cpuLastSample = s.sampled_at;
      cpuEl('spark').innerHTML = cpuSamples.map(v => `<div style="height:${Math.max(2, Math.min(100, v))}%" title="${v}% changed pixels"></div>`).join('');
    }
  } else {
    cpuEl('preview').removeAttribute('src');
    cpuEl('preview-message').hidden = false;
    cpuEl('preview-message').textContent = !row.settings.enabled ? 'Enable CPU analysis to see measurements.' : s.message || 'Waiting for fresh CPU samples…';
  }
}

async function loadDetectionLog() {
  const logEl = document.getElementById('detection-log');
  if (!cpuId) { logEl.innerHTML = ''; return; }
  if (detectionLogPollBusy) return;
  detectionLogPollBusy = true;
  try {
    const res = await fetch(`${API}/api/detections?camera_id=${cpuId}&limit=100`);
    if (!res.ok) return;
    const rows = await res.json();
    detectionLogLastId = rows[0]?.id ?? detectionLogLastId;
    if (rows.length === 0) {
      logEl.innerHTML = '<div class="empty-state-inline">No detections logged yet. Enable face, people, or object detection above to start collecting samples.</div>';
      return;
    }
    logEl.innerHTML = rows.map(r => {
      const counts = [];
      if (r.face_count) counts.push(`${r.face_count} face${r.face_count === 1 ? '' : 's'}`);
      if (r.people_count) counts.push(`${r.people_count} ${r.people_count === 1 ? 'person' : 'people'} (HOG)`);
      const objectSummary = r.objects.length
        ? r.objects.map(o => `${escapeHtml(o.class)} ${(o.confidence * 100).toFixed(0)}%`).join(', ')
        : '';
      const summary = [...counts, objectSummary].filter(Boolean).join(' · ') || 'No detections this sample';
      return `<div class="detection-log-row"><span class="detection-log-time">${new Date(r.created_at).toLocaleTimeString()}</span><span class="detection-log-summary">${summary}</span></div>`;
    }).join('');
  } catch (e) {
    // Transient errors shouldn't break the rest of the page.
  } finally { detectionLogPollBusy = false; }
}

cpuEl('preview').onload = () => { cpuEl('preview-message').hidden = true; };
cpuEl('preview').onerror = () => { cpuEl('preview-message').hidden = false; cpuEl('preview-message').textContent = 'No current analysis frame. Retrying…'; };
cpuEl('camera').onchange = () => { cpuId = cpuEl('camera').value; cpuSamples = []; cpuLastSample = null; detectionLogLastId = null; cpuEl('spark').replaceChildren(); cpuEl('save-message').textContent = ''; const row = cpuRows.find(r => r.camera_id === cpuId); if (row) { populateCpuForm(row.settings); renderCpu(row); } loadDetectionLog(); };
function drawCpuArea() {
  const rect = document.getElementById('cpu-zone-rect');
  for (const key of ['x','y','width','height']) rect.setAttribute(key, cpuEl(key).value);
}
for (const key of ['x','y','width','height']) cpuEl(key).oninput = drawCpuArea;
cpuEl('reset-area').onclick = () => { cpuEl('x').value = cpuEl('y').value = 0; cpuEl('width').value = cpuEl('height').value = 100; drawCpuArea(); };
const editor = document.getElementById('cpu-zone-editor');
let dragStart = null;
function point(event) { const box = editor.getBoundingClientRect(); return [Math.max(0, Math.min(100, (event.clientX-box.left)/box.width*100)), Math.max(0, Math.min(100,(event.clientY-box.top)/box.height*100))]; }
editor.onpointerdown = e => { if (!cpuId) return; dragStart = point(e); editor.setPointerCapture(e.pointerId); };
editor.onpointermove = e => {
  if (!dragStart) return;
  const end = point(e);
  const x = Math.min(95, Math.round(Math.min(dragStart[0], end[0]))), y = Math.min(95, Math.round(Math.min(dragStart[1], end[1])));
  cpuEl('x').value = x; cpuEl('y').value = y;
  cpuEl('width').value = Math.min(100-x, Math.max(5, Math.round(Math.abs(end[0]-dragStart[0]))));
  cpuEl('height').value = Math.min(100-y, Math.max(5, Math.round(Math.abs(end[1]-dragStart[1]))));
  drawCpuArea();
};
editor.onpointerup = editor.onpointercancel = () => { dragStart = null; };

cpuEl('form').onsubmit = async e => {
  e.preventDefault();
  if (!cpuId || cpuEl('save').disabled) return;
  const target = cpuId;
  cpuEl('save').disabled = true;
  cpuEl('save-message').textContent = 'Saving…';
  const body = {enabled: cpuEl('enabled').checked, motion_alerts: cpuEl('motion-alerts').checked, quality_alerts: cpuEl('quality-alerts').checked, face_alerts: cpuEl('face-alerts').checked, people_alerts: cpuEl('people-alerts').checked, object_alerts: cpuEl('object-alerts').checked, motion_threshold: Number(cpuEl('threshold').value)/100, cooldown_seconds: Number(cpuEl('cooldown').value), dark_threshold: Number(cpuEl('dark').value), blur_threshold: Number(cpuEl('blur').value), area: Object.fromEntries(['x','y','width','height'].map(k => [k, Number(cpuEl(k).value)/100]))};
  try {
    const res = await fetch(`${API}/api/cpu/${target}`, {method:'PUT', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
    const data = await res.json();
    if (!res.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Check the thresholds and ensure the watch area fits inside the frame.');
    cpuEl('save-message').textContent = 'Saved. Collecting fresh samples for this watch area.';
    cpuSamples = []; cpuLastSample = null;
    await loadCpu();
  } catch (err) { cpuEl('save-message').textContent = err.message; }
  finally { cpuEl('save').disabled = false; }
};

async function downloadFile(url, filename) {
  const response = await fetch(url);
  if (!response.ok) throw new Error('Download unavailable. Wait for fresh analysis and retry.');
  const objectUrl = URL.createObjectURL(await response.blob());
  const link = document.createElement('a'); link.href = objectUrl; link.download = filename; link.click();
  setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
}
cpuEl('download').onclick = async () => { try { await downloadFile(`${API}/api/cpu/${cpuId}/snapshot`, `argus-${cpuId}-analysis.jpg`); } catch (err) { cpuEl('save-message').textContent = err.message; } };
document.getElementById('export-events').onclick = async () => { try { await downloadFile(`${API}/api/events/export?${buildEventsQuery()}`, 'argus-events.csv'); } catch (err) { el.eventsList.innerHTML = `<div class="empty-state">${escapeHtml(err.message)}</div>`; } };
setInterval(() => { if (currentView === 'cpu') loadCpu(); }, 1000);
setInterval(() => { if (currentView === 'cpu') loadDetectionLog(); }, 3000);

document.getElementById('preset-ollama').onclick = () => {
  document.getElementById('runtime-provider').value = 'compatible';
  document.getElementById('runtime-url').value = 'http://localhost:11434/v1';
  document.getElementById('runtime-model').value = 'gemma3:4b';
  document.getElementById('runtime-remote').checked = false;
  document.getElementById('runtime-key').value = '';
  runtimeFields();
  document.getElementById('runtime-message').textContent = 'Preset filled, not saved. Follow the Ollama guide to install and pull the vision model.';
};
document.getElementById('preset-llama').onclick = () => {
  document.getElementById('runtime-provider').value = 'compatible';
  document.getElementById('runtime-url').value = 'http://localhost:8080/v1';
  document.getElementById('runtime-model').value = 'argus-vision';
  document.getElementById('runtime-remote').checked = false;
  document.getElementById('runtime-key').value = '';
  runtimeFields();
  document.getElementById('runtime-message').textContent = 'Preset filled, not saved. Start llama-server with the argus-vision alias from the documentation.';
};
// app.js initializes deep links before this file runs. Resume a CPU deep link now.
if (currentView === 'cpu') { loadCpu(true); loadDetectionLog(); }
