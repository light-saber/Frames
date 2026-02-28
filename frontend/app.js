'use strict';

// ── State ─────────────────────────────────────────────────────────────────────

const state = {
  photos: [],          // all PhotoAnalysis objects from server
  activeFilter: 'all',
  sortBy: 'score_desc',
  analysing: false,
  colorSettings: null,
};

const EMPTY_STATES = {
  pending:   ['Nothing left to decide',  'All photos have been marked keep or reject.'],
  keep:      ['No photos kept yet',       'Mark photos with ✓ Keep to see them here.'],
  reject:    ['No photos rejected',       'Mark photos with ✗ Reject to see them here.'],
  duplicate: ['No duplicates found',      'The perceptual hash analysis found no near-identical shots.'],
  all:       ['No photos found',          'Check that the folder contains supported RAW files.'],
};

// ── DOM refs ──────────────────────────────────────────────────────────────────

const $ = id => document.getElementById(id);
const welcomeScreen   = $('welcome-screen');
const photoView       = $('photo-view');
const photoGrid       = $('photo-grid');
const statsStrip      = $('stats-strip');
const progressWrap    = $('progress-bar-wrap');
const progressBar     = $('progress-bar');
const progressLabel   = $('progress-label');
const exportOverlay   = $('export-overlay');
const exportProgressBar = $('export-progress-bar');
const exportLabel     = $('export-label');

// ── Init ──────────────────────────────────────────────────────────────────────

async function init() {
  setupSidebar();
  await checkOllamaStatus();
  await loadPhotos();
  await loadColorSettings();
}

// ── Sidebar setup ─────────────────────────────────────────────────────────────

function setupSidebar() {
  $('browse-raw-btn').addEventListener('click', async () => {
    const res = await fetch('/api/folder-dialog', { method: 'POST' });
    const data = await res.json();
    if (data.path) {
      $('folder-input').value = data.path;
      onFolderSet(data.path);
    }
  });

  $('folder-input').addEventListener('change', () => {
    const path = $('folder-input').value.trim();
    if (path) onFolderSet(path);
  });

  $('folder-input').addEventListener('keydown', e => {
    if (e.key === 'Enter') {
      const path = $('folder-input').value.trim();
      if (path) onFolderSet(path);
    }
  });

  $('analyse-btn').addEventListener('click', startAnalysis);

  $('browse-export-btn').addEventListener('click', async () => {
    const res = await fetch('/api/folder-dialog', { method: 'POST' });
    const data = await res.json();
    if (data.path) $('export-input').value = data.path;
  });

  $('export-btn').addEventListener('click', startExport);

  $('recheck-ollama-btn').addEventListener('click', checkOllamaStatus);

  $('use-ai-checkbox').addEventListener('change', async e => {
    await fetch('/api/use-ai', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled: e.target.checked }),
    });
  });

  $('threshold-slider').addEventListener('input', e => {
    $('threshold-val').textContent = e.target.value;
  });

  $('apply-threshold-btn').addEventListener('click', async () => {
    const threshold = parseInt($('threshold-slider').value);
    await fetch('/api/batch/threshold', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ threshold, mode: 'pending_only' }),
    });
    await loadPhotos();
  });

  $('keep-all-btn').addEventListener('click', async () => {
    await fetch('/api/batch/keep-all', { method: 'POST' });
    await loadPhotos();
  });

  $('reset-btn').addEventListener('click', async () => {
    await fetch('/api/batch/reset', { method: 'POST' });
    await loadPhotos();
  });

  $('reset-colors-btn').addEventListener('click', async () => {
    const res = await fetch('/api/settings/reset', { method: 'POST' });
    state.colorSettings = await res.json();
    renderColorSliders();
  });

  // Tab bar
  document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
      state.activeFilter = tab.dataset.filter;
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      renderGrid();
    });
  });

  // Sort
  $('sort-select').addEventListener('change', e => {
    state.sortBy = e.target.value;
    renderGrid();
  });
}

// ── Folder handling ───────────────────────────────────────────────────────────

async function onFolderSet(path) {
  const res = await fetch('/api/folder', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path }),
  });

  if (res.ok) {
    $('analyse-btn').disabled = false;
    $('ai-section').style.display = 'block';
  } else {
    $('analyse-btn').disabled = true;
  }
}

// ── Ollama status ─────────────────────────────────────────────────────────────

async function checkOllamaStatus() {
  try {
    const res = await fetch('/api/ollama-status');
    const data = await res.json();
    const dot = $('ollama-status-dot');
    const cb  = $('use-ai-checkbox');
    if (data.alive) {
      dot.className = 'status-dot green';
      dot.textContent = '● Ollama running';
      cb.disabled = false;
    } else {
      dot.className = 'status-dot red';
      dot.textContent = '● Ollama not detected';
      cb.disabled = true;
      cb.checked = false;
    }
  } catch (e) {
    // server not ready yet
  }
}

// ── Load photos ───────────────────────────────────────────────────────────────

async function loadPhotos() {
  try {
    const res = await fetch('/api/photos');
    state.photos = await res.json();
  } catch {
    state.photos = [];
  }

  if (state.photos.length > 0) {
    showPhotoView();
  } else {
    showWelcome();
  }
}

async function loadColorSettings() {
  const res = await fetch('/api/settings');
  state.colorSettings = await res.json();
  renderColorSliders();
}

// ── Analysis ──────────────────────────────────────────────────────────────────

function startAnalysis() {
  if (state.analysing) return;
  state.analysing = true;
  state.photos = [];
  $('analyse-btn').textContent = '↺ Re-analysing…';
  $('analyse-btn').disabled = true;

  showPhotoView();
  progressWrap.style.display = 'block';
  setProgress(0, 1, 'Starting analysis…');

  const es = new EventSource('/api/analyze');
  es.onmessage = e => {
    const data = JSON.parse(e.data);

    if (data.done === true) {
      es.close();
      finishAnalysis();
      return;
    }

    if (data.photo) {
      // Update or add
      const idx = state.photos.findIndex(p => p.filename === data.photo.filename);
      if (idx >= 0) state.photos[idx] = data.photo;
      else state.photos.push(data.photo);
    }

    if (data.total > 0) {
      setProgress(data.done, data.total, `Analysing… ${data.done}/${data.total}`);
    }

    renderGrid();
  };

  es.onerror = () => {
    es.close();
    finishAnalysis();
  };
}

function finishAnalysis() {
  state.analysing = false;
  $('analyse-btn').textContent = '↺ Re-analyse';
  $('analyse-btn').disabled = false;
  progressWrap.style.display = 'none';
  $('batch-section').style.display = 'block';
  updateExportSection();
  renderStatsStrip();
  renderGrid();
  renderStepIndicator();
}

function setProgress(done, total, label) {
  const pct = total > 0 ? Math.round((done / total) * 100) : 0;
  progressBar.style.setProperty('--progress', `${pct}%`);
  progressLabel.textContent = label;
}

// ── Export ────────────────────────────────────────────────────────────────────

async function startExport() {
  const exportFolder = $('export-input').value.trim();
  if (!exportFolder) {
    alert('Please select an export folder first.');
    return;
  }

  exportOverlay.style.display = 'flex';
  exportProgressBar.style.setProperty('--progress', '0%');
  exportLabel.textContent = 'Starting export…';

  const url = `/api/export?export_folder=${encodeURIComponent(exportFolder)}`;
  const es = new EventSource(url);

  es.onmessage = e => {
    const data = JSON.parse(e.data);
    if (data.done === true) {
      es.close();
      exportOverlay.style.display = 'none';
      alert(`Export complete! ${data.total || ''} photos saved.`);
      return;
    }
    if (data.total > 0) {
      const pct = Math.round((data.done / data.total) * 100);
      exportProgressBar.style.setProperty('--progress', `${pct}%`);
      exportLabel.textContent = `Exporting ${data.filename || ''}… ${data.done}/${data.total}`;
    }
  };

  es.onerror = () => {
    es.close();
    exportOverlay.style.display = 'none';
  };
}

// ── Render helpers ────────────────────────────────────────────────────────────

function showWelcome() {
  welcomeScreen.style.display = 'block';
  photoView.style.display = 'none';
}

function showPhotoView() {
  welcomeScreen.style.display = 'none';
  photoView.style.display = 'block';
  renderStepIndicator();
  renderStatsStrip();
  renderGrid();
}

function updateExportSection() {
  const hasKeeps = state.photos.some(p => p.status === 'keep');
  $('export-section').style.display = hasKeeps ? 'block' : 'none';
}

// ── Step indicator ────────────────────────────────────────────────────────────

function renderStepIndicator() {
  const folderSet = !!$('folder-input').value.trim();
  const analyzed  = state.photos.length > 0;
  const hasKeeps  = state.photos.some(p => p.status === 'keep');

  let step = 1;
  if (folderSet) step = 2;
  if (analyzed)  step = 3;
  if (hasKeeps)  step = 4;

  const steps = [[1,'Load'],[2,'Analyse'],[3,'Cull'],[4,'Export']];
  const container = $('step-indicator');
  container.innerHTML = '';

  steps.forEach(([num, label], i) => {
    const item = document.createElement('div');
    item.className = 'step-item';

    let dotChar, dotColor, lblColor;
    if (num < step) {
      dotChar = '✓'; dotColor = 'var(--green-keep)'; lblColor = 'var(--text-muted)';
    } else if (num === step) {
      dotChar = '●'; dotColor = 'var(--accent-gold)'; lblColor = 'var(--accent-gold)';
    } else {
      dotChar = '○'; dotColor = 'var(--text-dimmed)'; lblColor = 'var(--text-dimmed)';
    }

    item.innerHTML = `
      <span class="step-dot" style="color:${dotColor}">${dotChar}</span>
      <span class="step-label" style="color:${lblColor}">${label}</span>
    `;
    container.appendChild(item);

    if (i < steps.length - 1) {
      const line = document.createElement('div');
      line.className = 'step-line';
      container.appendChild(line);
    }
  });
}

// ── Stats strip ───────────────────────────────────────────────────────────────

function renderStatsStrip() {
  const photos = state.photos;
  const total   = photos.length;
  const kept    = photos.filter(p => p.status === 'keep').length;
  const rejected = photos.filter(p => p.status === 'reject').length;
  const pending = photos.filter(p => p.status === 'pending').length;
  const dupes   = photos.filter(p => p.is_duplicate).length;
  const avg     = total > 0 ? (photos.reduce((s, p) => s + p.overall_score, 0) / total).toFixed(0) : 0;

  const item = (num, label) =>
    `<span class="stat-item-num">${num}</span><span class="stat-item-lbl">${label}</span>`;

  statsStrip.innerHTML =
    item(total, 'photos') +
    item(kept, 'kept') +
    item(rejected, 'rejected') +
    item(pending, 'pending') +
    item(dupes, 'dupes') +
    item(avg, 'avg score');

  // Update tab counts
  document.querySelectorAll('.tab').forEach(tab => {
    const filter = tab.dataset.filter;
    let count;
    switch (filter) {
      case 'all':       count = total; break;
      case 'pending':   count = pending; break;
      case 'keep':      count = kept; break;
      case 'reject':    count = rejected; break;
      case 'duplicate': count = dupes; break;
    }
    tab.textContent = `${filter.charAt(0).toUpperCase() + filter.slice(1)} (${count})`;
  });
}

// ── Photo grid ────────────────────────────────────────────────────────────────

function filteredPhotos() {
  let photos = state.photos.slice();
  const f = state.activeFilter;
  if (f === 'duplicate') photos = photos.filter(p => p.is_duplicate);
  else if (f !== 'all')  photos = photos.filter(p => p.status === f);

  const s = state.sortBy;
  if (s === 'score_desc') photos.sort((a, b) => b.overall_score - a.overall_score);
  else if (s === 'score_asc') photos.sort((a, b) => a.overall_score - b.overall_score);
  else if (s === 'name_asc')  photos.sort((a, b) => a.filename.localeCompare(b.filename));
  else if (s === 'name_desc') photos.sort((a, b) => b.filename.localeCompare(a.filename));

  return photos;
}

function renderGrid() {
  const photos = filteredPhotos();
  photoGrid.innerHTML = '';

  if (photos.length === 0) {
    const [title, desc] = EMPTY_STATES[state.activeFilter] || EMPTY_STATES.all;
    photoGrid.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">◻</div>
        <div class="empty-title">${title}</div>
        <div class="empty-desc">${desc}</div>
      </div>`;
    return;
  }

  photos.forEach(photo => {
    photoGrid.appendChild(buildCard(photo));
  });
}

function scoreColor(score) {
  if (score >= 70) return 'var(--green-keep)';
  if (score >= 45) return 'var(--amber-pending)';
  return 'var(--red-reject)';
}

function metricRow(label, value, color) {
  const pct = Math.min(100, Math.max(0, value));
  return `
    <div class="metric-row">
      <span class="metric-label">${label}</span>
      <div class="metric-bar-bg">
        <div class="metric-bar-fill" style="width:${pct.toFixed(0)}%;background:${color};"></div>
      </div>
      <span class="metric-val">${pct.toFixed(0)}</span>
    </div>`;
}

function buildCard(photo) {
  const card = document.createElement('div');
  card.className = `photo-card status-${photo.status}`;
  card.dataset.filename = photo.filename;

  const fname = photo.filename.length > 22
    ? photo.filename.slice(0, 22) + '…'
    : photo.filename;

  const dupHtml = photo.is_duplicate
    ? `<span class="dup-tag">dup</span>` : '';

  const sColor = scoreColor(photo.overall_score);

  let metrics = metricRow('S', photo.sharpness, 'var(--accent-gold)');
  metrics += metricRow('E', photo.exposure, 'var(--blue-exposure)');
  if (photo.ai_score != null) {
    metrics += metricRow('AI', photo.ai_score, 'var(--purple-dupe)');
  }

  const keepLabel = photo.status === 'keep'   ? '↩ kept'  : '✓ Keep';
  const rejLabel  = photo.status === 'reject' ? '↩ unrej' : '✗ Reject';
  const keepActive  = photo.status === 'keep'   ? 'active-keep'   : '';
  const rejActive   = photo.status === 'reject' ? 'active-reject'  : '';

  card.innerHTML = `
    <div class="thumb-wrap">
      <img src="/api/photo/${encodeURIComponent(photo.filename)}/thumbnail"
           alt="${photo.filename}" loading="lazy">
      <div class="score-overlay" style="color:${sColor};">${photo.overall_score.toFixed(0)}</div>
      <div class="card-metrics">${metrics}</div>
    </div>
    <div class="card-footer">
      <span class="filename">${fname}</span>${dupHtml}
    </div>
    <div class="card-actions">
      <button class="card-btn ${keepActive}" data-action="keep">${keepLabel}</button>
      <button class="card-btn ${rejActive}"  data-action="reject">${rejLabel}</button>
    </div>`;

  // Button handlers
  card.querySelectorAll('.card-btn').forEach(btn => {
    btn.addEventListener('click', () => toggleStatus(photo, btn.dataset.action, card));
  });

  return card;
}

async function toggleStatus(photo, action, card) {
  const newStatus = photo.status === action ? 'pending' : action;
  await fetch(`/api/photo/${encodeURIComponent(photo.filename)}/status`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ status: newStatus }),
  });
  photo.status = newStatus;

  // Update card border
  card.className = `photo-card status-${newStatus}`;

  // Update button labels/classes
  const [keepBtn, rejBtn] = card.querySelectorAll('.card-btn');
  keepBtn.textContent  = newStatus === 'keep'   ? '↩ kept'  : '✓ Keep';
  rejBtn.textContent   = newStatus === 'reject' ? '↩ unrej' : '✗ Reject';
  keepBtn.className = `card-btn${newStatus === 'keep'   ? ' active-keep'   : ''}`;
  rejBtn.className  = `card-btn${newStatus === 'reject' ? ' active-reject' : ''}`;

  renderStatsStrip();
  updateExportSection();
  renderStepIndicator();

  // If we're in a filtered view, card may no longer belong — re-render
  if (state.activeFilter !== 'all') {
    renderGrid();
  }
}

// ── Color sliders ─────────────────────────────────────────────────────────────

const COLOR_SLIDER_DEFS = [
  { key: 'brightness',        label: 'Brightness',         min: -0.3,  max: 0.3,  step: 0.01 },
  { key: 'contrast',          label: 'Contrast',            min: 0.7,   max: 1.5,  step: 0.01 },
  { key: 'saturation_boost',  label: 'Saturation',          min: 0.5,   max: 1.8,  step: 0.05 },
  { key: 'highlight_recovery',label: 'Highlight recovery',  min: 0.0,   max: 0.3,  step: 0.01 },
  { key: 'shadow_lift',       label: 'Shadow lift',         min: 0.0,   max: 0.15, step: 0.005 },
  { key: 'sharpening',        label: 'Sharpening',          min: 0.0,   max: 1.0,  step: 0.05 },
];

function renderColorSliders() {
  const container = $('color-sliders');
  container.innerHTML = '';
  if (!state.colorSettings) return;

  COLOR_SLIDER_DEFS.forEach(def => {
    const val = state.colorSettings[def.key];
    const row = document.createElement('div');
    row.className = 'color-slider-row';

    const valId = `cs-val-${def.key}`;
    row.innerHTML = `
      <span class="color-slider-label">${def.label}</span>
      <input type="range" min="${def.min}" max="${def.max}" step="${def.step}"
             value="${val}" class="slider" style="flex:1;">
      <span class="color-slider-val" id="${valId}">${parseFloat(val).toFixed(2)}</span>`;

    const input = row.querySelector('input');
    const valSpan = row.querySelector(`#${valId}`);
    input.addEventListener('input', async () => {
      const v = parseFloat(input.value);
      valSpan.textContent = v.toFixed(2);
      state.colorSettings[def.key] = v;
      await fetch('/api/settings', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ settings: { [def.key]: v } }),
      });
    });

    container.appendChild(row);
  });
}

// ── Kick off ──────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', init);
