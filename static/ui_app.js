const state = {
  latest: null,
  reviewItems: [],
  sectionOptions: [],
  finalLoaded: false,
};

const el = (id) => document.getElementById(id);
const uploadForm = el('uploadForm');
const documentInput = el('documentInput');
const fileLabel = el('fileLabel');
const uploadError = el('uploadError');
const startButton = el('startButton');
const uploadPanel = el('uploadPanel');
const statusPanel = el('statusPanel');
const reviewPanel = el('reviewPanel');
const finalPanel = el('finalPanel');
const failedPanel = el('failedPanel');
const stepList = el('stepList');
const progressBar = el('progressBar');
const progressPercent = el('progressPercent');
const progressDetail = el('progressDetail');
const phaseLabel = el('phaseLabel');
const statusTitle = el('statusTitle');
const statusDetail = el('statusDetail');
const messageLog = el('messageLog');
const reviewList = el('reviewList');
const reviewCount = el('reviewCount');
const approveAllButton = el('approveAllButton');
const continueButton = el('continueButton');
const finalDocument = el('finalDocument');
const downloadButton = el('downloadButton');
const downloadTxtButton = el('downloadTxtButton');
const newJobButton = el('newJobButton');
const retryButton = el('retryButton');
const failedTitle = el('failedTitle');
const failedDetail = el('failedDetail');

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function textDirection(value) {
  const text = String(value ?? '');
  const rtl = (text.match(/[֐-ࣿיִ-﷽ﹰ-ﻼ]/g) || []).length;
  const ltr = (text.match(/[A-Za-z]/g) || []).length;
  if (!rtl) return 'ltr';
  if (!ltr) return 'rtl';
  return rtl >= Math.max(2, Math.floor(ltr * 0.45)) ? 'rtl' : 'ltr';
}

function dirAttrs(value, extraClass = '') {
  const dir = textDirection(value);
  const classes = [`text-${dir}`, extraClass].filter(Boolean).join(' ');
  return `dir="${dir}" class="${classes}"`;
}

function show(panel, visible) {
  panel.classList.toggle('hidden', !visible);
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let message = `Request failed: ${response.status}`;
    try {
      const payload = await response.json();
      if (payload?.error) message = payload.error;
    } catch (_error) {
      // Keep the status-based fallback when the response body is not JSON.
    }
    throw new Error(message);
  }
  return response.json();
}

function renderSteps(steps = []) {
  stepList.innerHTML = steps.map((step, index) => `
    <li class="step-item ${escapeHtml(step.status)}">
      <div class="step-dot">${step.status === 'done' ? '✓' : index + 1}</div>
      <div>
        <p class="step-label">${escapeHtml(step.label)}</p>
        <p class="step-detail">${escapeHtml(step.detail || '')}</p>
      </div>
    </li>
  `).join('');
}

function renderProgress(snapshot) {
  const progress = snapshot.progress || { percent: 0, detail: 'Ready' };
  const percent = Number(progress.percent || 0);
  progressBar.style.width = `${percent}%`;
  progressPercent.textContent = `${percent}%`;
  progressDetail.textContent = progress.detail || '';
  phaseLabel.textContent = snapshot.phase_label || 'Ready';
  statusTitle.textContent = snapshot.phase_label || 'Working';
  statusDetail.textContent = progress.detail || '';
}

function renderMessages(messages = []) {
  messageLog.innerHTML = messages.slice(-12).reverse().map((item) => `
    <div ${dirAttrs(item.message || '', `message-item ${item.kind || ''}`)}>${escapeHtml(item.message || '')}</div>
  `).join('');
}

function sectionChip(label) {
  return `<span ${dirAttrs(label, 'section-chip')}>${escapeHtml(label)}</span>`;
}

function sectionPicker(item) {
  const selected = new Set(item.sections || []);
  return `
    <details class="area-picker">
      <summary>Change document area</summary>
      <div class="area-grid">
        ${state.sectionOptions.map((option) => `
          <label>
            <input type="checkbox" data-section-option="${escapeHtml(option.id)}" ${selected.has(option.id) ? 'checked' : ''}>
            <span ${dirAttrs(option.label, 'picker-label')}>${escapeHtml(option.label)}</span>
          </label>
        `).join('')}
      </div>
    </details>
  `;
}

function renderReview() {
  const openIndexes = new Set();
  reviewList.querySelectorAll('[data-review-index]').forEach((card) => {
    if (card.querySelector('details.area-picker')?.open) openIndexes.add(card.dataset.reviewIndex);
  });

  const approved = state.reviewItems.filter((item) => item.approved).length;
  reviewCount.textContent = `${approved} of ${state.reviewItems.length} approved`;
  continueButton.disabled = approved !== state.reviewItems.length || state.reviewItems.length === 0;
  reviewList.innerHTML = state.reviewItems.map((item) => `
    <article class="review-card ${item.approved ? 'approved' : ''}" data-review-index="${item.index}">
      <header class="review-card-header">
        <div>
          <h3 class="review-card-title">${escapeHtml(item.title)}</h3>
          <div class="chip-row">${(item.section_labels || []).map(sectionChip).join('')}</div>
        </div>
        <button class="approve-toggle" type="button">${item.approved ? 'Approved' : 'Approve'}</button>
      </header>
      <div class="review-card-body">${item.html || ''}</div>
      <footer class="review-card-footer">
        ${sectionPicker(item)}
      </footer>
    </article>
  `).join('');

  if (openIndexes.size) {
    reviewList.querySelectorAll('[data-review-index]').forEach((card) => {
      if (openIndexes.has(card.dataset.reviewIndex)) {
        const details = card.querySelector('details.area-picker');
        if (details) details.open = true;
      }
    });
  }
}

function syncSectionLabels(item) {
  item.section_labels = (item.sections || []).map((sectionId) => {
    const option = state.sectionOptions.find((candidate) => candidate.id === sectionId);
    return option ? option.label : sectionId;
  });
}

reviewList.addEventListener('click', (event) => {
  const button = event.target.closest('.approve-toggle');
  if (!button) return;
  const card = button.closest('[data-review-index]');
  const index = Number(card.dataset.reviewIndex);
  const item = state.reviewItems.find((candidate) => candidate.index === index);
  if (!item) return;
  if (!(item.sections || []).length) {
    button.textContent = 'Choose an area first';
    setTimeout(() => renderReview(), 900);
    return;
  }
  item.approved = !item.approved;
  renderReview();
});

reviewList.addEventListener('change', (event) => {
  if (!event.target.matches('[data-section-option]')) return;
  const card = event.target.closest('[data-review-index]');
  const index = Number(card.dataset.reviewIndex);
  const item = state.reviewItems.find((candidate) => candidate.index === index);
  if (!item) return;
  const sectionId = event.target.dataset.sectionOption;
  if (event.target.checked) {
    if (!(item.sections || []).includes(sectionId)) {
      item.sections = [...(item.sections || []), sectionId];
    }
  } else {
    item.sections = (item.sections || []).filter((id) => id !== sectionId);
  }
  syncSectionLabels(item);
  item.approved = false;
  renderReview();
});

approveAllButton.addEventListener('click', () => {
  state.reviewItems = state.reviewItems.map((item) => ({ ...item, approved: (item.sections || []).length > 0 }));
  renderReview();
});

continueButton.addEventListener('click', async () => {
  continueButton.disabled = true;
  const payload = {
    items: state.reviewItems.map((item) => ({ index: item.index, sections: item.sections, approved: item.approved })),
  };
  const response = await fetch('/api/review', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: 'Could not submit review.' }));
    uploadError.textContent = error.error || 'Could not submit review.';
    continueButton.disabled = false;
  }
});

async function loadFinal() {
  if (state.finalLoaded) return;
  const response = await fetch('/api/final');
  if (!response.ok) return;
  const payload = await response.json();
  finalDocument.innerHTML = payload.html || '';
  state.finalLoaded = true;
}

finalDocument.addEventListener('click', async (event) => {
  const button = event.target.closest('[data-hierarchy-shift]');
  if (!button) return;
  event.preventDefault();
  event.stopPropagation();
  if (button.disabled) return;

  const resultIndex = Number(button.dataset.resultIndex);
  const entryIndex = Number(button.dataset.entryIndex);
  const direction = String(button.dataset.hierarchyShift || '');
  if (!Number.isFinite(resultIndex) || !Number.isFinite(entryIndex) || !direction) return;

  button.disabled = true;
  try {
    const payload = await fetchJson('/api/final/hierarchy', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ result_index: resultIndex, entry_index: entryIndex, direction }),
    });
    finalDocument.innerHTML = payload.html || '';
    state.finalLoaded = true;
  } catch (err) {
    window.alert(err.message || 'Could not update hierarchy.');
    button.disabled = false;
  }
});

async function resetAndShowUpload() {
  await fetch('/api/reset', { method: 'POST' });
  state.finalLoaded = false;
  finalDocument.innerHTML = '';
  show(uploadPanel, true);
  show(statusPanel, false);
  show(reviewPanel, false);
  show(finalPanel, false);
  show(failedPanel, false);
  startButton.disabled = false;
  startButton.textContent = 'Start';
  uploadError.textContent = '';
}

newJobButton.addEventListener('click', resetAndShowUpload);
retryButton.addEventListener('click', resetAndShowUpload);

function applySnapshot(snapshot) {
  const prevStatus = state.latest?.status;
  state.latest = snapshot;
  renderSteps(snapshot.steps || []);
  renderProgress(snapshot);
  renderMessages(snapshot.messages || []);

  const idle = snapshot.status === 'idle';
  const awaitingReview = snapshot.status === 'awaiting_review';
  const completed = snapshot.status === 'completed';
  const failed = snapshot.status === 'failed';
  const working = !idle && !awaitingReview && !completed && !failed;

  show(uploadPanel, idle);
  show(statusPanel, working);
  show(reviewPanel, awaitingReview);
  show(finalPanel, completed);
  show(failedPanel, failed);

  if (failed) {
    failedTitle.textContent = snapshot.phase_label || 'Stopped';
    failedDetail.textContent = snapshot.error || 'The process stopped unexpectedly.';
  }

  if (awaitingReview) {
    if (prevStatus !== 'awaiting_review') {
      state.reviewItems = (snapshot.review_items || []).map((item) => ({ ...item }));
      state.sectionOptions = snapshot.section_options || [];
    }
    renderReview();
  }

  if (completed) {
    loadFinal();
  } else {
    state.finalLoaded = false;
    finalDocument.innerHTML = '';
  }
}

function startEvents() {
  if (!window.EventSource) {
    setInterval(async () => {
      const response = await fetch('/api/state');
      applySnapshot(await response.json());
    }, 1000);
    return;
  }
  const events = new EventSource('/api/events');
  events.onmessage = (event) => applySnapshot(JSON.parse(event.data));
  events.onerror = () => {
    events.close();
    setInterval(async () => {
      const response = await fetch('/api/state');
      applySnapshot(await response.json());
    }, 1500);
  };
}

documentInput.addEventListener('change', () => {
  const file = documentInput.files[0];
  fileLabel.textContent = file ? file.name : 'Choose a DOCX file';
});

uploadForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  uploadError.textContent = '';
  startButton.disabled = true;
  startButton.textContent = 'Starting...';
  const formData = new FormData(uploadForm);
  const response = await fetch('/api/upload', { method: 'POST', body: formData });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: 'Upload failed.' }));
    uploadError.textContent = error.error || 'Upload failed.';
    startButton.disabled = false;
    startButton.textContent = 'Start';
    return;
  }
  show(uploadPanel, false);
  show(statusPanel, true);
});

downloadButton.addEventListener('click', () => {
  window.location.href = '/api/download';
});

downloadTxtButton.addEventListener('click', () => {
  window.location.href = '/api/download-rag-txt';
});

fetch('/api/state').then((response) => response.json()).then(applySnapshot).finally(startEvents);
