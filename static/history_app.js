// ── Helpers ──────────────────────────────────────────────────────────────
function escHtml(v) {
  return String(v ?? '').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#039;');
}
function el(id) { return document.getElementById(id); }
function show(elem, visible) { if (elem) elem.classList.toggle('hidden', !visible); }
function textDir(v) {
  const t = String(v ?? '');
  const rtl = (t.match(/[֐-ࣿיִ-﷽ﹰ-ﻼ]/g)||[]).length;
  const ltr = (t.match(/[A-Za-z]/g)||[]).length;
  if (!rtl) return 'ltr'; if (!ltr) return 'rtl';
  return rtl >= Math.max(2, Math.floor(ltr * 0.45)) ? 'rtl' : 'ltr';
}
function dirAttr(v) { const d = textDir(v); return `dir="${d}" class="text-${d}"`; }
function formatDate(value, mode = 'date') {
  if (!value) return '';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '';
  if (mode === 'time') return date.toLocaleTimeString('en-GB');
  if (mode === 'datetime') return date.toLocaleString('en-GB');
  return date.toLocaleDateString('en-GB', { day:'numeric', month:'short', year:'numeric' });
}
async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Request failed: ${res.status}`);
  return res.json();
}

// ── State ──────────────────────────────────────────────────────────────────
const state = {
  docs: [],
  activeDocId: null,
  loadedChunks: null,
  loadedLogs: null,
  loadedJsons: null,
  allLogs: [],
};

// ── DOM refs ───────────────────────────────────────────────────────────────
const docList      = el('docList');
const docCount     = el('docCount');
const histWelcome  = el('histWelcome');
const histDetail   = el('histDetail');
const detailName   = el('detailName');
const detailMeta   = el('detailMeta');
const detailEyebrow= el('detailEyebrow');
const detailDownloads = el('detailDownloads');
const histTabs     = el('histTabs');
const chunksList   = el('chunksList');
const logsList     = el('logsList');
const jsonsList    = el('jsonsList');
const assetsList   = el('assetsList');
const logStageFilter = el('logStageFilter');
const logFailedOnly  = el('logFailedOnly');
const finalOutputContainer = el('finalOutputContainer');

// ── Tab logic ──────────────────────────────────────────────────────────────
const TABS = ['final', 'chunks', 'logs', 'jsons', 'assets'];

histTabs.addEventListener('click', (e) => {
  const btn = e.target.closest('.hist-tab');
  if (!btn) return;
  const tab = btn.dataset.tab;
  histTabs.querySelectorAll('.hist-tab').forEach((b) => b.classList.toggle('active', b === btn));
  TABS.forEach((t) => el(`tab-${t}`).classList.toggle('hidden', t !== tab));
  if (tab === 'chunks' && !state.loadedChunks) loadChunks();
  if (tab === 'logs'   && !state.loadedLogs)   loadLogs();
  if (tab === 'jsons'  && !state.loadedJsons)  loadJsons();
  if (tab === 'assets')                         loadAssets();
});

// ── Document list ──────────────────────────────────────────────────────────
async function loadDocList() {
  try {
    state.docs = await fetchJson('/api/history');
  } catch (err) {
    docCount.textContent = 'Unavailable';
    docList.innerHTML = '<div class="hist-empty">History could not be loaded. Refresh the page after the server restarts.</div>';
    return;
  }
  docCount.textContent = `${state.docs.length} document${state.docs.length !== 1 ? 's' : ''}`;

  if (!state.docs.length) {
    docList.innerHTML = `<div class="hist-empty">No processed documents found.<br><a href="/">Process your first document &#8594;</a></div>`;
    return;
  }

  docList.innerHTML = state.docs.map((doc) => {
    const date = formatDate(doc.created_at);
    const chunkCount = Number(doc.chunk_count || 0);
    const badges = [
      doc.has_classification && `<span class="hist-badge">Classified</span>`,
      doc.has_section_json   && `<span class="hist-badge done">Output ready</span>`,
      doc.has_llm_logs       && `<span class="hist-badge log">LLM logs</span>`,
      doc.has_assets         && `<span class="hist-badge">Assets</span>`,
      !doc.has_docx          && `<span class="hist-badge warn">Artifacts only</span>`,
    ].filter(Boolean).join('');
    return `
      <button class="hist-doc-card" data-doc-id="${escHtml(doc.id)}" type="button">
        <div class="hist-doc-name" ${dirAttr(doc.name)}>${escHtml(doc.name)}</div>
        <div class="hist-doc-meta">${escHtml(date)} · ${chunkCount} part${chunkCount === 1 ? '' : 's'}</div>
        <div class="hist-doc-badges">${badges}</div>
      </button>`;
  }).join('');
}

docList.addEventListener('click', (e) => {
  const card = e.target.closest('.hist-doc-card');
  if (!card) return;
  selectDoc(card.dataset.docId);
});

async function selectDoc(docId) {
  if (state.activeDocId === docId) return;
  state.activeDocId = docId;
  state.loadedChunks = null;
  state.loadedLogs = null;
  state.loadedJsons = null;
  state.allLogs = [];

  // Highlight selected card
  docList.querySelectorAll('.hist-doc-card').forEach((c) => c.classList.toggle('active', c.dataset.docId === docId));

  show(histWelcome, false);
  show(histDetail, true);

  const doc = state.docs.find((d) => d.id === docId) || { name: docId };
  detailName.textContent = doc.name;
  detailName.setAttribute('dir', textDir(doc.name));
  detailEyebrow.textContent = 'Document';
  const date = formatDate(doc.created_at, 'datetime');
  detailMeta.textContent = [date, doc.chunk_count && `${doc.chunk_count} parts`].filter(Boolean).join(' · ');

  // Download buttons
  detailDownloads.innerHTML = `
    ${doc.has_docx ? `<a class="secondary-button hist-dl-btn" href="/api/history/${encodeURIComponent(docId)}/download/docx">&#8659; DOCX</a>` : ''}
    ${doc.has_classification ? `<a class="secondary-button hist-dl-btn" href="/api/history/${encodeURIComponent(docId)}/download/classification">&#8659; Classification</a>` : ''}
    ${doc.has_inspection     ? `<a class="secondary-button hist-dl-btn" href="/api/history/${encodeURIComponent(docId)}/download/inspection">&#8659; Inspection</a>` : ''}
    ${doc.has_section_json   ? `<a class="secondary-button hist-dl-btn" href="/api/history/${encodeURIComponent(docId)}/download/section_json">&#8659; Section JSON</a>` : ''}
    ${doc.has_section_json   ? `<a class="secondary-button hist-dl-btn" href="/api/history/${encodeURIComponent(docId)}/download/rag-txt">&#8659; RAG TXT</a>` : ''}
  `;

  // Reset to Final tab
  histTabs.querySelectorAll('.hist-tab').forEach((b) => b.classList.toggle('active', b.dataset.tab === 'final'));
  TABS.forEach((t) => el(`tab-${t}`).classList.toggle('hidden', t !== 'final'));

  loadFinal();
}

// ── Final output ───────────────────────────────────────────────────────────
async function loadFinal() {
  finalOutputContainer.innerHTML = '<div class="hist-loading">Loading final document…</div>';
  const docId = state.activeDocId;
  try {
    const data = await fetchJson(`/api/history/${encodeURIComponent(docId)}/final`);
    if (docId !== state.activeDocId) return;
    finalOutputContainer.innerHTML = data.html || '<div class="hist-empty">Empty document.</div>';
  } catch (err) {
    if (docId !== state.activeDocId) return;
    finalOutputContainer.innerHTML = '<div class="hist-empty">Final document not available for this entry.</div>';
  }
}

// ── Chunks ─────────────────────────────────────────────────────────────────
async function loadChunks() {
  chunksList.innerHTML = '<div class="hist-loading">Loading chunks…</div>';
  const docId = state.activeDocId;
  let chunks = [];
  try {
    chunks = await fetchJson(`/api/history/${encodeURIComponent(docId)}/chunks`);
  } catch (err) {
    if (docId !== state.activeDocId) return;
    chunksList.innerHTML = '<div class="hist-empty">Chunks could not be loaded for this document.</div>';
    return;
  }
  if (docId !== state.activeDocId) return;
  if (!Array.isArray(chunks)) chunks = [];
  state.loadedChunks = chunks;

  if (!chunks.length) {
    chunksList.innerHTML = '<div class="hist-empty">No chunks found for this document.</div>';
    return;
  }

  chunksList.innerHTML = chunks.map((chunk, i) => {
    const reviewed = chunk.review_action === 'modified'
      ? '<span class="hist-badge warn">Modified in review</span>'
      : chunk.review_action === 'confirmed' ? '<span class="hist-badge done">Confirmed</span>' : '';
    const labels = (chunk.section_labels || []).map((l) => `<span class="section-chip" ${dirAttr(l)}>${escHtml(l)}</span>`).join('');
    return `
      <details class="hist-chunk-card" ${i === 0 ? 'open' : ''}>
        <summary class="hist-chunk-summary">
          <span class="hist-chunk-num">Part ${i + 1}</span>
          <span class="hist-chunk-file">${escHtml(chunk.name)}</span>
          <span class="chip-row">${labels}</span>
          ${reviewed}
        </summary>
        <div class="hist-chunk-body">${chunk.html || '<div class="hist-empty">No preview available.</div>'}</div>
      </details>`;
  }).join('');
}

// ── LLM Logs ───────────────────────────────────────────────────────────────
function stageLabel(s) { return {classify:'Classify', inspect:'Inspect', extract:'Extract'}[s] || 'Unknown'; }

function renderLogs() {
  const filterStage = logStageFilter.value;
  const filterFailed = logFailedOnly.checked;
  const filtered = state.allLogs.filter((e) => {
    if (filterStage && e.stage !== filterStage) return false;
    if (filterFailed && e.success !== false) return false;
    return true;
  });

  if (!filtered.length) {
    logsList.innerHTML = '<div class="hist-empty">No log entries match the current filter.</div>';
    return;
  }

  logsList.innerHTML = filtered.map((entry, i) => {
    const ts = formatDate(entry.ts, 'time');
    const statusClass = entry.success === false ? 'fail' : 'ok';
    const statusLabel = entry.success === false ? '✗ Failed' : '✓ OK';
    const stageBadge = `<span class="hist-badge stage-${entry.stage || ''}">${stageLabel(entry.stage)}</span>`;
    const subjectDir = textDir(entry.subject || '');

    const messages = (entry.messages || []);
    const systemMsg = messages.find((m) => m.role === 'system');
    const userMsgs  = messages.filter((m) => m.role === 'user');
    const lastUser  = userMsgs[userMsgs.length - 1];
    const attempt = Number(entry.attempt || 1);

    return `
      <details class="hist-log-card ${statusClass}">
        <summary class="hist-log-summary">
          <span class="hist-log-status ${statusClass}">${statusLabel}</span>
          ${stageBadge}
          <span class="hist-log-subject" dir="${subjectDir}">${escHtml(entry.subject || '')}</span>
          <span class="hist-log-ts">${escHtml(ts)}</span>
          ${attempt > 1 ? `<span class="hist-badge warn">retry ${attempt}</span>` : ''}
        </summary>
        <div class="hist-log-body">
          <div class="hist-log-meta">
            <span><strong>Model:</strong> ${escHtml(entry.model || '')}</span>
            <span><strong>Attempt:</strong> ${escHtml(String(entry.attempt || 1))}</span>
            ${entry.section_id ? `<span><strong>Section:</strong> ${escHtml(entry.section_id)}</span>` : ''}
          </div>
          ${entry.error ? `<div class="hist-log-error"><strong>Error:</strong> ${escHtml(entry.error)}</div>` : ''}
          ${systemMsg ? `
          <details class="hist-log-block">
            <summary>System prompt (${systemMsg.content?.length || 0} chars)</summary>
            <pre class="hist-log-pre">${escHtml((systemMsg.content || '').trim())}</pre>
          </details>` : ''}
          ${lastUser ? `
          <details class="hist-log-block" open>
            <summary>User prompt (${lastUser.content?.length || 0} chars)</summary>
            <pre class="hist-log-pre" ${dirAttr(lastUser.content || '')}>${escHtml((lastUser.content || '').trim())}</pre>
          </details>` : ''}
          <details class="hist-log-block" open>
            <summary>Response (${entry.response?.length || 0} chars)</summary>
            <pre class="hist-log-pre" ${dirAttr(entry.response || '')}>${escHtml((entry.response || '').trim())}</pre>
          </details>
        </div>
      </details>`;
  }).join('');
}

async function loadLogs() {
  logsList.innerHTML = '<div class="hist-loading">Loading LLM logs…</div>';
  const docId = state.activeDocId;
  let logs = [];
  try {
    logs = await fetchJson(`/api/history/${encodeURIComponent(docId)}/llm-logs`);
  } catch (err) {
    if (docId !== state.activeDocId) return;
    logsList.innerHTML = '<div class="hist-empty">LLM logs could not be loaded for this document.</div>';
    return;
  }
  if (docId !== state.activeDocId) return;
  state.allLogs = Array.isArray(logs) ? logs : [];
  state.loadedLogs = true;

  if (!state.allLogs.length) {
    logsList.innerHTML = '<div class="hist-empty">No LLM call logs found. Logs are created on new pipeline runs.</div>';
    return;
  }
  renderLogs();
}

logStageFilter.addEventListener('change', renderLogs);
logFailedOnly.addEventListener('change', renderLogs);

// ── JSONs ──────────────────────────────────────────────────────────────────
async function loadJsons() {
  jsonsList.innerHTML = '<div class="hist-loading">Loading JSON files…</div>';
  const docId = state.activeDocId;
  let data = {};
  try {
    data = await fetchJson(`/api/history/${encodeURIComponent(docId)}`);
  } catch (err) {
    if (docId !== state.activeDocId) return;
    jsonsList.innerHTML = '<div class="hist-empty">JSON files could not be loaded for this document.</div>';
    return;
  }
  if (docId !== state.activeDocId) return;
  state.loadedJsons = true;

  const sections = [
    { key: 'classification', label: 'Classification output', download: 'classification' },
    { key: 'inspection',     label: 'Column header inspection', download: 'inspection' },
    { key: 'section_json',   label: 'Section JSON output',    download: 'section_json' },
  ];

  jsonsList.innerHTML = sections.map(({ key, label, download }) => {
    const payload = data[key];
    if (!payload) return `<div class="hist-json-block missing"><p class="hist-section-label">${escHtml(label)}</p><p class="hist-empty">Not available</p></div>`;
    const jsonStr = JSON.stringify(payload, null, 2);
    return `
      <div class="hist-json-block">
        <div class="hist-json-header">
          <p class="hist-section-label">${escHtml(label)}</p>
          <a class="secondary-button hist-dl-btn" href="/api/history/${encodeURIComponent(docId)}/download/${download}">&#8659; Download</a>
        </div>
        <details class="hist-json-viewer">
          <summary>Show JSON (${jsonStr.length.toLocaleString()} chars)</summary>
          <pre class="hist-json-pre">${escHtml(jsonStr)}</pre>
        </details>
      </div>`;
  }).join('');
}

// ── Assets ─────────────────────────────────────────────────────────────────
async function loadAssets() {
  assetsList.innerHTML = '<div class="hist-loading">Loading assets…</div>';
  const docId = state.activeDocId;
  let data = {};
  try {
    data = await fetchJson(`/api/history/${encodeURIComponent(docId)}`);
  } catch (err) {
    if (docId !== state.activeDocId) return;
    assetsList.innerHTML = '<div class="hist-empty">Assets could not be loaded for this document.</div>';
    return;
  }
  if (docId !== state.activeDocId) return;
  const classification = data.classification;
  const knownAssets = Array.isArray(data.assets) ? data.assets.filter((asset) => asset && asset.id) : [];
  const assetSet = new Map(knownAssets.map((asset) => [String(asset.id), asset]));
  if (classification?.results) {
    for (const result of classification.results) {
      const text = JSON.stringify(result);
      for (const m of text.matchAll(/EM\d{6}/g)) {
        if (!assetSet.has(m[0])) assetSet.set(m[0], { id: m[0] });
      }
    }
  }
  if (data.section_json?.results) {
    const text = JSON.stringify(data.section_json.results);
    for (const m of text.matchAll(/EM\d{6}/g)) {
      if (!assetSet.has(m[0])) assetSet.set(m[0], { id: m[0] });
    }
  }

  const assets = [...assetSet.values()].sort((a, b) => String(a.id).localeCompare(String(b.id)));
  if (!assets.length) {
    assetsList.innerHTML = '<div class="hist-empty">No embedded assets found in this document.</div>';
    return;
  }

  assetsList.innerHTML = assets.map((asset) => {
    const assetId = String(asset.id);
    const label = asset.name || assetId;
    return `
    <div class="hist-asset-card">
      <div class="hist-asset-preview">
        <img src="/api/history/${encodeURIComponent(docId)}/asset/${encodeURIComponent(assetId)}"
             alt="${escHtml(assetId)}"
             data-asset-preview>
      </div>
      <div class="hist-asset-name" title="${escHtml(label)}">${escHtml(label)}</div>
      <a class="hist-asset-dl" href="/api/history/${encodeURIComponent(docId)}/asset/${encodeURIComponent(assetId)}" download>&#8659;</a>
    </div>
  `;
  }).join('');
  assetsList.querySelectorAll('[data-asset-preview]').forEach((img) => {
    img.addEventListener('error', () => {
      img.parentElement.innerHTML = '<div class="hist-asset-placeholder">Preview unavailable</div>';
    }, { once: true });
  });
}

// ── Init ───────────────────────────────────────────────────────────────────
loadDocList();
