const milvusState = {
  docs: [],
  selectedDocIds: new Set(),
  latestJob: null,
};

const milvusEl = (id) => document.getElementById(id);

const docList = milvusEl('milvusDocList');
const docSearch = milvusEl('milvusDocSearch');
const selectionCount = milvusEl('milvusSelectionCount');
const selectAllButton = milvusEl('milvusSelectAll');
const clearAllButton = milvusEl('milvusClearAll');
const ingestForm = milvusEl('milvusIngestForm');
const ingestButton = milvusEl('milvusIngestButton');
const ingestError = milvusEl('milvusIngestError');
const queryForm = milvusEl('milvusQueryForm');
const queryButton = milvusEl('milvusQueryButton');
const queryError = milvusEl('milvusQueryError');
const phaseLabel = milvusEl('milvusPhaseLabel');
const progressPercent = milvusEl('milvusProgressPercent');
const progressBar = milvusEl('milvusProgressBar');
const progressDetail = milvusEl('milvusProgressDetail');
const stepList = milvusEl('milvusStepList');
const messageLog = milvusEl('milvusMessageLog');
const hitsContainer = milvusEl('milvusHits');
const rerankedContainer = milvusEl('milvusRerankedDocs');
const summaryCard = milvusEl('milvusSummaryCard');
const resetButton = milvusEl('milvusResetButton');

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
  const className = [`text-${dir}`, extraClass].filter(Boolean).join(' ');
  return `dir="${dir}" class="${className}"`;
}

function show(element, visible) {
  if (!element) return;
  element.classList.toggle('hidden', !visible);
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let message = `Request failed: ${response.status}`;
    try {
      const payload = await response.json();
      if (payload?.error) message = payload.error;
    } catch (_error) {
      // Fall back to the status-based message.
    }
    throw new Error(message);
  }
  return response.json();
}

function numberLabel(value) {
  return new Intl.NumberFormat('en-US').format(Number(value || 0));
}

function renderSteps(steps = []) {
  stepList.innerHTML = steps.map((step, index) => `
    <li class="step-item ${escapeHtml(step.status || 'pending')}">
      <div class="step-dot">${step.status === 'done' ? '✓' : index + 1}</div>
      <div>
        <p class="step-label">${escapeHtml(step.label || '')}</p>
        <p class="step-detail">${escapeHtml(step.detail || '')}</p>
      </div>
    </li>
  `).join('');
}

function renderProgress(snapshot) {
  const progress = snapshot?.progress || { percent: 0, detail: 'Ready' };
  const percent = Number(progress.percent || 0);
  phaseLabel.textContent = snapshot?.phase_label || 'Ready';
  progressPercent.textContent = `${percent}%`;
  progressBar.style.width = `${percent}%`;
  progressDetail.textContent = progress.detail || 'Ready';
}

function renderMessages(messages = []) {
  if (!messages.length) {
    messageLog.innerHTML = '<div class="milvus-empty milvus-empty-inline">No ingestion activity yet.</div>';
    return;
  }
  messageLog.innerHTML = messages.slice(-14).reverse().map((entry) => `
    <div ${dirAttrs(entry.message || '', `message-item ${entry.kind || 'info'}`)}>${escapeHtml(entry.message || '')}</div>
  `).join('');
}

function renderSummary(result) {
  if (!result) {
    summaryCard.innerHTML = '';
    show(summaryCard, false);
    return;
  }
  summaryCard.innerHTML = `
    <div class="milvus-summary-grid">
      <div>
        <span>Collection</span>
        <strong>${escapeHtml(result.collection_name || '')}</strong>
      </div>
      <div>
        <span>Documents</span>
        <strong>${numberLabel(result.document_count)}</strong>
      </div>
      <div>
        <span>Chunks</span>
        <strong>${numberLabel(result.chunk_count)}</strong>
      </div>
      <div>
        <span>Embedding dim</span>
        <strong>${numberLabel(result.embedding_dimension)}</strong>
      </div>
    </div>
    <p class="milvus-summary-note">${result.recreated ? 'The target collection was recreated before ingestion.' : 'A fresh collection was created for this ingestion.'}</p>
  `;
  show(summaryCard, true);
}

function renderDocList() {
  const needle = String(docSearch.value || '').trim().toLowerCase();
  const visibleDocs = needle
    ? milvusState.docs.filter((doc) => doc.name.toLowerCase().includes(needle))
    : milvusState.docs.slice();

  selectionCount.textContent = `${milvusState.selectedDocIds.size} selected`;

  if (!visibleDocs.length) {
    docList.innerHTML = '<div class="milvus-empty">No RAG TXT documents match this search.</div>';
    return;
  }

  docList.innerHTML = visibleDocs.map((doc) => {
    const selected = milvusState.selectedDocIds.has(doc.id);
    return `
      <label class="milvus-doc-card ${selected ? 'selected' : ''}">
        <input type="checkbox" data-doc-id="${escapeHtml(doc.id)}" ${selected ? 'checked' : ''}>
        <div class="milvus-doc-copy">
          <div class="milvus-doc-title-row">
            <strong ${dirAttrs(doc.name)}>${escapeHtml(doc.name)}</strong>
            <span class="milvus-doc-chip">${numberLabel(doc.rag_chunk_count)} TXT</span>
          </div>
          <p class="milvus-doc-meta">${numberLabel(doc.total_characters)} characters</p>
        </div>
      </label>
    `;
  }).join('');
}

async function loadDocuments() {
  try {
    milvusState.docs = await fetchJson('/api/milvus/documents');
  } catch (error) {
    docList.innerHTML = `<div class="milvus-empty">${escapeHtml(error.message || 'Documents could not be loaded.')}</div>`;
    return;
  }
  renderDocList();
}

docList.addEventListener('change', (event) => {
  const checkbox = event.target.closest('[data-doc-id]');
  if (!checkbox) return;
  const docId = checkbox.dataset.docId;
  if (checkbox.checked) milvusState.selectedDocIds.add(docId);
  else milvusState.selectedDocIds.delete(docId);
  renderDocList();
});

docSearch.addEventListener('input', renderDocList);

selectAllButton.addEventListener('click', () => {
  milvusState.docs.forEach((doc) => milvusState.selectedDocIds.add(doc.id));
  renderDocList();
});

clearAllButton.addEventListener('click', () => {
  milvusState.selectedDocIds.clear();
  renderDocList();
});

function renderMilvusHits(hits = []) {
  if (!hits.length) {
    hitsContainer.innerHTML = '<div class="milvus-empty">No Milvus hits to display.</div>';
    return;
  }
  hitsContainer.innerHTML = hits.map((hit) => `
    <article class="milvus-hit-card">
      <div class="milvus-hit-head">
        <span class="milvus-rank-pill">#${escapeHtml(hit.rank)}</span>
        <span class="milvus-score-pill">${Number(hit.milvus_score || 0).toFixed(4)}</span>
      </div>
      <h4 ${dirAttrs(hit.document_name || '')}>${escapeHtml(hit.document_name || 'Unknown document')}</h4>
      <div class="milvus-chip-row">
        <span class="milvus-chip">${escapeHtml(hit.file_name || '')}</span>
        <span class="milvus-chip">${escapeHtml(hit.section_id || 'No section')}</span>
      </div>
      <p class="milvus-hit-path" ${dirAttrs(hit.hierarchy_path || '')}>${escapeHtml(hit.hierarchy_path || 'No hierarchy path')}</p>
      <div class="milvus-hit-text" ${dirAttrs(hit.text || '')}>${escapeHtml(hit.text || '')}</div>
    </article>
  `).join('');
}

function renderRerankedDocs(documents = []) {
  if (!documents.length) {
    rerankedContainer.innerHTML = '<div class="milvus-empty">No reranked documents to display.</div>';
    return;
  }
  rerankedContainer.innerHTML = documents.map((doc, index) => `
    <article class="milvus-hit-card milvus-hit-card-accent">
      <div class="milvus-hit-head">
        <span class="milvus-rank-pill">#${index + 1}</span>
        <span class="milvus-score-pill">${Number(doc.reranker_score || 0).toFixed(4)}</span>
      </div>
      <h4 ${dirAttrs(doc.document_name || '')}>${escapeHtml(doc.document_name || 'Unknown document')}</h4>
      <div class="milvus-chip-row">
        <span class="milvus-chip">${escapeHtml(doc.file_name || '')}</span>
        <span class="milvus-chip">Milvus ${Number(doc.milvus_score || 0).toFixed(4)}</span>
      </div>
      <p class="milvus-hit-path" ${dirAttrs(doc.hierarchy_path || '')}>${escapeHtml(doc.hierarchy_path || 'No hierarchy path')}</p>
      <div class="milvus-hit-text" ${dirAttrs(doc.text || '')}>${escapeHtml(doc.text || '')}</div>
    </article>
  `).join('');
}

function applySnapshot(snapshot) {
  milvusState.latestJob = snapshot;
  renderSteps(snapshot.steps || []);
  renderProgress(snapshot);
  renderMessages(snapshot.messages || []);
  renderSummary(snapshot.result || null);
  const isBusy = snapshot.status === 'queued' || snapshot.status === 'running';
  ingestButton.disabled = isBusy;
  ingestButton.textContent = isBusy ? 'Ingestion Running...' : 'Ingest Into Milvus';
  resetButton.disabled = isBusy;
  if (snapshot.status === 'failed') {
    ingestError.textContent = snapshot.error || 'The ingestion stopped unexpectedly.';
  } else {
    ingestError.textContent = '';
  }
}

function startMilvusEvents() {
  if (!window.EventSource) {
    setInterval(async () => {
      const snapshot = await fetchJson('/api/milvus/state');
      applySnapshot(snapshot);
    }, 1200);
    return;
  }

  const events = new EventSource('/api/milvus/events');
  events.onmessage = (event) => {
    applySnapshot(JSON.parse(event.data));
  };
  events.onerror = () => {
    events.close();
    setInterval(async () => {
      const snapshot = await fetchJson('/api/milvus/state');
      applySnapshot(snapshot);
    }, 1500);
  };
}

ingestForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  ingestError.textContent = '';
  const selectedIds = [...milvusState.selectedDocIds];
  if (!selectedIds.length) {
    ingestError.textContent = 'Select at least one document to ingest.';
    return;
  }

  ingestButton.disabled = true;
  ingestButton.textContent = 'Starting...';
  try {
    await fetchJson('/api/milvus/ingest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        doc_ids: selectedIds,
        collection_name: milvusEl('collectionName').value,
        milvus_uri: milvusEl('milvusUri').value,
        milvus_db_name: milvusEl('milvusDbName').value,
        milvus_token: milvusEl('milvusToken').value,
      }),
    });
  } catch (error) {
    ingestError.textContent = error.message || 'The ingestion request failed.';
  } finally {
    ingestButton.disabled = false;
    ingestButton.textContent = 'Ingest Into Milvus';
  }
});

queryForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  queryError.textContent = '';

  const topK = Number(milvusEl('milvusTopK').value || 0);
  const topN = Number(milvusEl('milvusTopN').value || 0);

  queryButton.disabled = true;
  queryButton.textContent = 'Retrieving...';
  try {
    const payload = await fetchJson('/api/milvus/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: milvusEl('milvusQuestion').value,
        doc_ids: [...milvusState.selectedDocIds],
        collection_name: milvusEl('collectionName').value,
        milvus_uri: milvusEl('milvusUri').value,
        milvus_db_name: milvusEl('milvusDbName').value,
        milvus_token: milvusEl('milvusToken').value,
        top_k: topK,
        top_n: topN,
      }),
    });
    renderMilvusHits(payload.milvus_hits || []);
    renderRerankedDocs(payload.reranked_documents || []);
  } catch (error) {
    queryError.textContent = error.message || 'The retrieval request failed.';
  } finally {
    queryButton.disabled = false;
    queryButton.textContent = 'Retrieve';
  }
});

resetButton.addEventListener('click', async () => {
  try {
    await fetchJson('/api/milvus/reset', { method: 'POST' });
    applySnapshot(await fetchJson('/api/milvus/state'));
    ingestError.textContent = '';
  } catch (error) {
    ingestError.textContent = error.message || 'The Milvus status could not be reset.';
  }
});

loadDocuments();
fetchJson('/api/milvus/state').then(applySnapshot).finally(startMilvusEvents);