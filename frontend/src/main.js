import { cancelInvestigation, getBootstrap, getInvestigation, getWorkspaceInvestigation, isApiLocal, listInvestigations, runInvestigation } from './api.js';
import { createFallbackBootstrap, emptyInvestigation, normalizeBootstrap, normalizeInvestigation, summarizeInvestigations, formatStageLabel } from './model.js';

const app = document.querySelector('#app');

const state = {
  loading: true,
  error: null,
  bootstrap: createFallbackBootstrap(),
  investigations: [],
  activeInvestigationId: null,
  requestedInvestigationId: null,
  selectedStage: 'answer',
  selectedReport: 'analyst',
  drawer: null,
  guideDraft: '',
  commandDraft: '',
  view: 'home',
  apiState: isApiLocal() ? 'local' : 'remote',
  form: {
    question: 'Why did sales decline last quarter?',
    datasetPath: createFallbackBootstrap().defaultDatasetPath,
    mode: 'guided',
  },
};

const RUN_POLL_INTERVAL_MS = 2000;
const RUN_POLL_TIMEOUT_MS = 300000;

function parseRoute() {
  const hash = window.location.hash.replace(/^#/, '') || 'home';
  const [view, id] = hash.split('/');
  state.view = view || 'home';
  state.requestedInvestigationId = id || null;
  state.activeInvestigationId = id || state.activeInvestigationId;
}

function setRoute(view, id = '') {
  window.location.hash = id ? `#${view}/${id}` : `#${view}`;
}

function getInvestigationUrl(id = getActiveInvestigation()?.id) {
  if (!id) return window.location.href;
  const base = `${window.location.origin}${window.location.pathname}`;
  return `${base}#investigations/${encodeURIComponent(id)}`;
}

function sleep(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function getActiveInvestigation() {
  if (!state.activeInvestigationId) return null;
  return state.investigations.find((item) => item.id === state.activeInvestigationId) || null;
}

function mergeInvestigationRecords(existing, incoming) {
  if (!existing) return incoming;
  const merged = { ...existing, ...incoming };
  const hasArray = (value) => Array.isArray(value) && value.length > 0;
  const hasAnswer = (value) => value?.direct && value.direct !== 'The investigation has not reached a direct answer yet.';
  const hasReport = (value) =>
    value &&
    Object.values(value).some((item) => {
      const result = text(item);
      return result && result !== 'No report available yet.';
    });

  merged.progress = hasArray(incoming.progress) ? incoming.progress : existing.progress;
  merged.findings = hasArray(incoming.findings) ? incoming.findings : existing.findings;
  merged.evidence = hasArray(incoming.evidence) ? incoming.evidence : existing.evidence;
  merged.journey = hasArray(incoming.journey) ? incoming.journey : existing.journey;
  merged.snapshots = hasArray(incoming.snapshots) ? incoming.snapshots : existing.snapshots;
  merged.tasks = hasArray(incoming.tasks) ? incoming.tasks : existing.tasks;
  merged.hypotheses = hasArray(incoming.hypotheses) ? incoming.hypotheses : existing.hypotheses;
  merged.recommendations = hasArray(incoming.recommendations) ? incoming.recommendations : existing.recommendations;
  merged.reports = hasReport(incoming.reports) ? incoming.reports : existing.reports;
  merged.answer = hasAnswer(incoming.answer) ? incoming.answer : existing.answer;
  merged.dataQuality = incoming.dataQuality?.issues || incoming.dataQuality?.rows || incoming.dataQuality?.columns ? incoming.dataQuality : existing.dataQuality;
  merged.confidence = incoming.confidence?.label && incoming.confidence.label !== 'Unknown' ? incoming.confidence : existing.confidence;
  merged.analysisPlan = hasArray(incoming.analysisPlan) ? incoming.analysisPlan : existing.analysisPlan;
  merged.selectedColumns = hasArray(incoming.selectedColumns) ? incoming.selectedColumns : existing.selectedColumns;
  merged.visualizations = hasArray(incoming.visualizations) ? incoming.visualizations : existing.visualizations;
  merged.raw = incoming.raw || existing.raw;
  return merged;
}

function setActiveInvestigation(investigation) {
  if (!investigation) return;
  const normalized = normalizeInvestigation(investigation);
  const existingIndex = state.investigations.findIndex((item) => item.id === normalized.id);
  if (existingIndex >= 0) {
    state.investigations.splice(existingIndex, 1, mergeInvestigationRecords(state.investigations[existingIndex], normalized));
  } else {
    state.investigations.unshift(normalized);
  }
  state.activeInvestigationId = normalized.id;
  state.selectedStage = normalized.progress.find((stage) => stage.status === 'current')?.key || 'answer';
  state.selectedReport = normalized.reports.analyst ? 'analyst' : normalized.reports.business ? 'business' : normalized.reports.executive ? 'executive' : 'analyst';
  setRoute('investigations', normalized.id);
}

function setDrawer(drawer) {
  state.drawer = drawer;
  render();
}

async function loadInvestigation(id) {
  if (!id) return null;
  const existing = state.investigations.find((item) => item.id === id) || null;
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 20000);
  try {
    const response = await getWorkspaceInvestigation(id, { signal: controller.signal });
    const investigation = normalizeInvestigation(response.investigation || response);
    const existingIndex = state.investigations.findIndex((item) => item.id === investigation.id);
    if (existingIndex >= 0) {
      state.investigations.splice(existingIndex, 1, mergeInvestigationRecords(state.investigations[existingIndex], investigation));
    } else {
      state.investigations.unshift(investigation);
    }
    state.activeInvestigationId = investigation.id;
    state.selectedStage = investigation.progress.find((stage) => stage.status === 'current')?.key || state.selectedStage || 'answer';
    state.selectedReport = investigation.reports.analyst ? 'analyst' : investigation.reports.business ? 'business' : investigation.reports.executive ? 'executive' : 'analyst';
    state.error = null;
    return investigation;
  } catch (error) {
    try {
      const response = await getInvestigation(id);
      const investigation = normalizeInvestigation(response.investigation || response);
      const existingIndex = state.investigations.findIndex((item) => item.id === investigation.id);
      if (existingIndex >= 0) {
        state.investigations.splice(existingIndex, 1, mergeInvestigationRecords(state.investigations[existingIndex], investigation));
      } else {
        state.investigations.unshift(investigation);
      }
      state.activeInvestigationId = investigation.id;
      state.selectedStage = investigation.progress.find((stage) => stage.status === 'current')?.key || state.selectedStage || 'answer';
      state.selectedReport = investigation.reports.analyst ? 'analyst' : investigation.reports.business ? 'business' : investigation.reports.executive ? 'executive' : 'analyst';
      state.error = null;
      return investigation;
    } catch (fallbackError) {
      state.error = null;
      const inflated = inflateSummaryInvestigation(existing);
      if (inflated) {
        const existingIndex = state.investigations.findIndex((item) => item.id === inflated.id);
        if (existingIndex >= 0) {
          state.investigations.splice(existingIndex, 1, mergeInvestigationRecords(state.investigations[existingIndex], inflated));
        } else {
          state.investigations.unshift(inflated);
        }
        state.activeInvestigationId = inflated.id;
        return inflated;
      }
      return existing;
    }
  } finally {
    window.clearTimeout(timeout);
  }
}

async function waitForInvestigationResult(clientRequestId, fallbackQuery = {}) {
  if (!clientRequestId) return null;
  const startedAt = Date.now();
  while (Date.now() - startedAt < RUN_POLL_TIMEOUT_MS) {
    try {
      const payload = await listInvestigations();
      const match = (payload.investigations || []).find((item) => item.client_request_id === clientRequestId);
      const status = text(match?.status).toLowerCase();
      if (match && (status === 'completed' || status === 'awaiting_user' || status === 'cancelled' || status === 'failed' || match.has_report)) {
        return await loadInvestigation(match.id);
      }
    } catch (error) {
      void error;
    }
    await sleep(RUN_POLL_INTERVAL_MS);
  }

  const fallback = state.investigations.find((item) => {
    if (!item) return false;
    return (
      text(item.question).toLowerCase() === text(fallbackQuery.question).toLowerCase() &&
      text(item.dataset?.path) === text(fallbackQuery.datasetPath) &&
      text(item.mode?.id || item.mode).toLowerCase() === text(fallbackQuery.mode || 'autonomous').toLowerCase()
    );
  });

  if (fallback) {
    return loadInvestigation(fallback.id);
  }

  return null;
}

function createPendingInvestigation(payload, id) {
  const dataset = state.bootstrap.datasets.find((item) => item.path === payload.datasetPath) || {
    name: payload.datasetPath || 'Selected dataset',
    path: payload.datasetPath || '',
    sourceType: 'csv',
    rowCount: null,
    columnCount: null,
  };
  return normalizeInvestigation({
    id: id || `pending-${Date.now()}`,
    question: payload.question,
    business_question: payload.question,
    mode: payload.mode || 'autonomous',
    status: 'running',
    dataset_path: payload.datasetPath,
    dataset,
    dataset_profile: {
      row_count: dataset.rowCount ?? dataset.row_count ?? null,
      column_count: dataset.columnCount ?? dataset.column_count ?? null,
    },
    analysis_evidence: {},
    analysis_plan: [],
    awaiting_user: false,
  });
}

function inflateSummaryInvestigation(summary) {
  if (!summary) return null;
  const confidenceLabel =
    typeof summary.confidence === 'number'
      ? summary.confidence >= 75
        ? 'High'
        : summary.confidence >= 45
          ? 'Moderate'
          : 'Low'
      : 'Unknown';
  return normalizeInvestigation({
    id: summary.id,
    question: summary.question,
    business_question: summary.question,
    mode: summary.mode,
    status: summary.status,
    dataset_path: summary.dataset?.path || summary.dataset_path,
    dataset: summary.dataset,
    dataset_profile: {
      row_count: summary.dataset?.row_count ?? null,
      column_count: summary.dataset?.column_count ?? null,
      source_type: summary.dataset?.source_type ?? null,
    },
    analysis_evidence: {
      answer_synthesis: {
        direct_answer: summary.answer || 'The investigation has not reached a direct answer yet.',
        business_interpretation: summary.answer || '',
        confidence: {
          overall: { label: confidenceLabel },
        },
      },
      judgment_summary: {
        summary: summary.answer || '',
        global_confidence: confidenceLabel,
      },
      top_stories: [],
      decision_recommendations: [],
      report_package: {
        analyst_report: summary.reports?.analyst ? 'Report available in backend history.' : '',
        business_report: summary.reports?.business ? 'Report available in backend history.' : '',
        executive_report: summary.reports?.executive ? 'Report available in backend history.' : '',
        master_report: '',
      },
    },
    final_report: summary.has_report ? summary.answer || 'Report available in backend history.' : '',
    analyst_report: summary.reports?.analyst ? 'Report available in backend history.' : '',
    business_report: summary.reports?.business ? 'Report available in backend history.' : '',
    executive_report: summary.reports?.executive ? 'Report available in backend history.' : '',
    master_report: '',
    selected_columns: [],
    visualizations: [],
  });
}

async function copyTextToClipboard(value) {
  const textValue = String(value ?? '');
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(textValue);
    return;
  }
  const textarea = document.createElement('textarea');
  textarea.value = textValue;
  textarea.setAttribute('readonly', 'readonly');
  textarea.style.position = 'absolute';
  textarea.style.left = '-9999px';
  document.body.appendChild(textarea);
  textarea.select();
  document.execCommand('copy');
  document.body.removeChild(textarea);
}

function downloadTextFile(filename, content, type = 'application/json') {
  const blob = new Blob([content], { type: `${type};charset=utf-8` });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function stageBlock(stage, active = false) {
  return `
    <button class="progress-step ${stage.status} ${active ? 'active' : ''}" data-action="stage" data-stage="${stage.key}" aria-pressed="${active ? 'true' : 'false'}">
      <span class="progress-step-label">${stage.label}</span>
      <span class="progress-step-state">${stage.status === 'complete' ? 'Done' : stage.status === 'current' ? 'Now' : 'Pending'}</span>
    </button>
  `;
}

function renderSidebar() {
  const active = state.view;
  return `
    <aside class="sidebar" aria-label="Application navigation">
      <div class="brand">
        <div class="brand-mark">DA</div>
        <div>
          <div class="brand-name">Data Analyst Agent</div>
          <div class="brand-tag">Investigation workspace</div>
        </div>
      </div>

      <button class="primary-action" data-action="go-new">New Investigation</button>

      <nav class="nav-list">
        ${[
          ['home', 'Home'],
          ['investigations', 'Investigations'],
          ['datasets', 'Datasets'],
          ['reports', 'Reports'],
          ['settings', 'Settings'],
        ]
          .map(
            ([view, label]) => `
              <button class="nav-item ${active === view ? 'active' : ''}" data-action="nav" data-view="${view}">
                ${label}
              </button>
            `,
          )
          .join('')}
      </nav>

      <div class="sidebar-panel">
        <div class="sidebar-label">Connection</div>
        <div class="connection-row">
          <span class="status-dot ${state.error ? 'error' : state.apiState === 'remote' ? 'good' : 'warn'}"></span>
          <div>
            <div class="connection-title">${state.error ? 'Backend unavailable' : state.apiState === 'remote' ? 'Connected to bridge' : 'Local preview mode'}</div>
            <div class="connection-subtitle">${state.error ? 'The UI is showing the shell while it waits for the bridge.' : state.apiState === 'remote' ? 'Live backend bridge is ready.' : 'Run the API bridge to enable live investigations.'}</div>
          </div>
        </div>
      </div>
    </aside>
  `;
}

function renderHome() {
  const bootstrap = state.bootstrap;
  const investigations = bootstrap.recentInvestigations || [];
  const completedCount = investigations.filter((item) => text(item.status).toLowerCase() === 'completed' || item.has_report).length;
  const runningCount = investigations.filter((item) => text(item.status).toLowerCase().includes('running') || text(item.status).toLowerCase().includes('awaiting')).length;
  return `
    <section class="page-stack">
      <header class="page-hero">
        <div class="page-hero-topline">
          <div>
            <div class="eyebrow">Home</div>
            <h1>Welcome back</h1>
            <p class="lede">Your AI data analyst workspace is ready. Start a new investigation or reopen an existing one to continue the evidence trail.</p>
          </div>
          <div class="hero-status">
            <span class="hero-status-dot"></span>
            <span>Backend connected</span>
          </div>
        </div>
        <div class="stats-grid">
          <div class="stat-card">
            <span class="stat-value">${investigations.length || 0}</span>
            <span class="stat-label">Investigations</span>
            <span class="stat-note">Tracked runs</span>
          </div>
          <div class="stat-card">
            <span class="stat-value">${bootstrap.datasets.length || 0}</span>
            <span class="stat-label">Datasets</span>
            <span class="stat-note">Available</span>
          </div>
          <div class="stat-card">
            <span class="stat-value">${completedCount}</span>
            <span class="stat-label">Reports</span>
            <span class="stat-note">Generated</span>
          </div>
          <div class="stat-card">
            <span class="stat-value">${runningCount}</span>
            <span class="stat-label">Pending</span>
            <span class="stat-note">Needs review</span>
          </div>
        </div>
      </header>

      <section class="home-grid">
        <article class="panel panel-deep">
          <div class="panel-heading">
            <div>
              <div class="section-label">Recent investigations</div>
              <h2>Recent work</h2>
            </div>
            <button class="link-button" data-action="nav" data-view="investigations">View all</button>
          </div>
          ${bootstrap.recentInvestigations.length ? renderInvestigationList(bootstrap.recentInvestigations.slice(0, 4)) : renderEmptyState('No investigations yet', 'Run your first question to populate the workspace with a real analysis trail.')}
        </article>

        <aside class="home-aside">
          <section class="panel quick-start-panel">
            <div class="section-label">Quick start</div>
            <p class="subtle">Start a new investigation with the existing backend workflow and review the outcome in a workspace built around question, answer, evidence, and confidence.</p>
            <button class="primary-action wide" data-action="go-new">New Investigation</button>
          </section>

          <section class="panel">
            <div class="section-label">Tips</div>
            <ul class="tip-list">
              <li>Be specific in your question.</li>
              <li>Select the most relevant dataset.</li>
              <li>Use Guided when you want checkpoints.</li>
              <li>Use Collaborative when the task needs follow-up.</li>
            </ul>
          </section>
        </aside>
      </section>

      <section class="split-grid">
        <article class="panel">
          <div class="panel-heading">
            <div>
              <div class="section-label">Suggested questions</div>
              <h2>What should we investigate?</h2>
            </div>
          </div>
          <div class="pill-list">
            ${bootstrap.suggestedQuestions
              .map(
                (item) => `
                  <button class="question-pill" data-action="fill-question" data-question="${escapeHtml(item)}">${escapeHtml(item)}</button>
                `,
              )
              .join('')}
          </div>
        </article>
        <article class="panel">
          <div class="panel-heading">
            <div>
              <div class="section-label">Dataset catalog</div>
              <h2>${bootstrap.datasets.length} datasets available</h2>
            </div>
          </div>
          <div class="pill-list">
            ${bootstrap.datasets.slice(0, 6).map((dataset) => `<span class="pill">${escapeHtml(dataset.name)}</span>`).join('')}
          </div>
        </article>
      </section>
    </section>
  `;
}

function renderNewInvestigation() {
  return `
    <section class="page-stack">
      <header class="page-hero compact">
        <div class="page-hero-topline">
          <div>
            <div class="eyebrow">New Investigation</div>
            <h1>New Investigation</h1>
            <p class="lede">Ask a business question and let the backend workflow take it from there.</p>
          </div>
          <button class="crumb" data-action="nav" data-view="home">Back</button>
        </div>
      </header>

      <div class="new-investigation-grid">
        <section class="hero-panel">
          <form class="investigation-form" id="investigation-form">
            <label class="field">
              <span class="step-label"><span class="step-number">1</span> Your Question</span>
              <span>Business question</span>
              <textarea name="question" rows="4" placeholder="Why did sales decline last quarter?">${escapeHtml(state.form.question)}</textarea>
            </label>
            <div class="form-row compact-form-row">
              <label class="field">
                <span class="step-label"><span class="step-number">2</span> Select Dataset</span>
              <span>Dataset</span>
              <select name="datasetPath">
                ${state.bootstrap.datasets
                  .map(
                    (dataset) => `
                      <option value="${escapeHtml(dataset.path)}" ${dataset.path === state.form.datasetPath ? 'selected' : ''}>
                        ${escapeHtml(dataset.name)}${dataset.row_count ? ` - ${Number(dataset.row_count).toLocaleString()} rows` : ''}
                      </option>
                    `,
                  )
                  .join('')}
              </select>
              </label>
              <div class="field">
                <span class="step-label"><span class="step-number">3</span> Choose Mode</span>
                <span>Mode</span>
                <div class="mode-strip mode-strip-grid">
                ${state.bootstrap.modes
                  .map(
                    (mode) => `
                      <button type="button" class="mode-chip ${state.form.mode === mode.id ? 'active' : ''}" data-action="mode" data-mode="${mode.id}">
                        <strong>${mode.label}</strong>
                        <span>${escapeHtml(mode.description || '')}</span>
                      </button>
                    `,
                  )
                  .join('')}
                </div>
              </div>
            </div>
            <div class="hero-actions">
              <button class="primary-action" type="submit">Start investigation</button>
              <button class="secondary-action" type="button" data-action="nav" data-view="home">Back to Home</button>
            </div>
          </form>
        </section>

        <aside class="context-column">
          <section class="context-card">
            <div class="section-label">What this does</div>
            <p class="subtle">The selected mode determines how much control the agent keeps during the investigation. Autonomous keeps moving, Guided pauses at review points, and Collaborative keeps the investigation desk open for follow-up questions and task management.</p>
          </section>
          <section class="context-card">
            <div class="section-label">Dataset preview</div>
            <div class="meta-stack">
              <div><dt>Selected</dt><dd>${escapeHtml((state.bootstrap.datasets.find((item) => item.path === state.form.datasetPath) || {}).name || 'Unknown dataset')}</dd></div>
              <div><dt>Question</dt><dd>${escapeHtml(state.form.question || 'No question entered yet.')}</dd></div>
            </div>
          </section>
        </aside>
      </div>
    </section>
  `;
}

function renderInvestigationList(items) {
  return `
    <div class="card-list">
      ${items
        .map(
          (item) => `
            <button class="investigation-card" data-action="open-investigation" data-id="${escapeHtml(item.id)}">
              <div class="card-topline">
                <span class="card-mode">${escapeHtml(text(item.mode?.label || item.mode))}</span>
                <span class="card-status ${escapeHtml(text(item.status))}">${escapeHtml(text(item.status))}</span>
              </div>
              <h3>${escapeHtml(item.question)}</h3>
              <p>${escapeHtml(text(item.answer?.direct || item.answer?.business || item.answer) || 'No direct answer yet.')}</p>
              <div class="card-meta">
                <span>${escapeHtml(item.dataset?.name || 'Unknown dataset')}</span>
                <span>${escapeHtml(text(item.confidence) || 'Unknown confidence')}</span>
              </div>
            </button>
          `,
        )
        .join('')}
    </div>
  `;
}

function renderEmptyState(title, detail) {
  return `
    <div class="empty-state">
      <h3>${escapeHtml(title)}</h3>
      <p>${escapeHtml(detail)}</p>
    </div>
  `;
}

function renderDatasets() {
  const datasets = state.bootstrap.datasets;
  const active = datasets.find((dataset) => dataset.path === state.form.datasetPath) || datasets[0];
  return `
    <section class="page-stack">
      <header class="page-hero compact">
        <div class="eyebrow">Datasets</div>
        <h1>Choose the dataset that frames the investigation</h1>
        <p class="lede">The catalog is populated from the existing backend data folder. The metadata is intentionally useful: source type, size, row count, and columns where available.</p>
      </header>
      <div class="split-grid datasets-layout">
        <article class="panel">
          <div class="section-label">Catalog</div>
          <div class="dataset-list">
            ${datasets
              .map(
                (dataset) => `
                  <button class="dataset-row ${dataset.path === active?.path ? 'active' : ''}" data-action="select-dataset" data-path="${escapeHtml(dataset.path)}">
                    <div>
                      <strong>${escapeHtml(dataset.name)}</strong>
                      <div class="subtle">${escapeHtml(dataset.source_type || 'dataset')}</div>
                    </div>
                    <div class="dataset-stats">
                      <span>${dataset.row_count ? `${Number(dataset.row_count).toLocaleString()} rows` : 'Row count loading'}</span>
                      <span>${dataset.column_count ? `${dataset.column_count} columns` : 'Column count loading'}</span>
                    </div>
                  </button>
                `,
              )
              .join('')}
          </div>
        </article>
        <article class="panel">
          <div class="section-label">Selected dataset</div>
          <h3>${escapeHtml(active?.name || 'Dataset')}</h3>
          <dl class="meta-grid">
            <div><dt>Source</dt><dd>${escapeHtml(active?.source_type || 'unknown')}</dd></div>
            <div><dt>Rows</dt><dd>${active?.row_count ? Number(active.row_count).toLocaleString() : 'Loading'}</dd></div>
            <div><dt>Columns</dt><dd>${active?.column_count || 'Loading'}</dd></div>
            <div><dt>Size</dt><dd>${escapeHtml(active?.size_label || 'Unknown')}</dd></div>
            <div><dt>Last modified</dt><dd>${escapeHtml(active?.last_modified ? new Date(active.last_modified).toLocaleString() : 'Unknown')}</dd></div>
            <div><dt>Path</dt><dd>${escapeHtml(active?.path || '')}</dd></div>
          </dl>
          <div class="section-label mt">Columns preview</div>
          <div class="pill-list">
            ${(active?.columns || []).length
              ? active.columns.map((column) => `<span class="pill">${escapeHtml(column)}</span>`).join('')
              : '<span class="subtle">Column metadata is not available yet.</span>'}
          </div>
        </article>
      </div>
    </section>
  `;
}

function renderReports() {
  const completed = state.investigations.filter((item) => text(item.status).toLowerCase() === 'completed' || item.raw?.final_report);
  const selected = completed.find((item) => item.id === state.activeInvestigationId) || completed[0];
  return `
    <section class="page-stack">
      <header class="page-hero compact">
        <div class="eyebrow">Reports</div>
        <h1>Completed investigations and report outputs</h1>
        <p class="lede">Reports are outputs of an investigation, not a replacement for it. Analyst, business, and executive views remain distinct because they serve different audiences.</p>
      </header>
      <div class="split-grid">
        <article class="panel">
          <div class="section-label">Completed investigations</div>
          ${completed.length ? renderInvestigationList(completed) : renderEmptyState('No completed reports yet', 'Run an investigation to generate the first report bundle.')}
        </article>
        <article class="panel">
          <div class="report-tabs">
            ${['analyst', 'business', 'executive']
              .map(
                (report) => `
                  <button class="report-tab ${state.selectedReport === report ? 'active' : ''}" data-action="select-report" data-report="${report}">
                    ${report.charAt(0).toUpperCase() + report.slice(1)} Report
                  </button>
                `,
              )
              .join('')}
          </div>
          ${selected ? renderReportBody(selected) : renderEmptyState('No report selected', 'Select a completed investigation to inspect its report.')}
        </article>
      </div>
    </section>
  `;
}

function renderInvestigations() {
  const investigations = state.investigations;
  const running = investigations.filter((item) => text(item.status).toLowerCase().includes('running') || text(item.status).toLowerCase().includes('awaiting'));
  const completed = investigations.filter((item) => text(item.status).toLowerCase() === 'completed' || item.has_report);
  return `
    <section class="page-stack">
      <header class="page-hero compact">
        <div class="eyebrow">Investigations</div>
        <h1>All investigations in one place</h1>
        <p class="lede">This view is the bridge between the backend records and the workspace. Open a record to load the full analysis bundle, evidence trail, and snapshots.</p>
      </header>
      <div class="split-grid investigations-grid">
        <article class="panel">
          <div class="panel-heading">
            <div>
              <div class="section-label">Active</div>
              <h2>${running.length} running or awaiting review</h2>
            </div>
          </div>
          ${running.length ? renderInvestigationList(running) : renderEmptyState('No active investigations', 'Create a new run to see it here.') }
        </article>
        <article class="panel">
          <div class="panel-heading">
            <div>
              <div class="section-label">Completed</div>
              <h2>${completed.length} finished investigations</h2>
            </div>
          </div>
          ${completed.length ? renderInvestigationList(completed) : renderEmptyState('No completed investigations', 'Finished investigations will appear here once the backend returns a final report.')}
        </article>
      </div>
    </section>
  `;
}

function renderSettings() {
  return `
    <section class="page-stack">
      <header class="page-hero compact">
        <div class="eyebrow">Settings</div>
        <h1>Workspace configuration</h1>
        <p class="lede">The frontend is intentionally lightweight. This page is for connection status, local preview behavior, and future configuration without exposing backend machinery.</p>
      </header>
      <div class="split-grid">
        <article class="panel">
          <div class="section-label">Backend bridge</div>
          <h3>${state.error ? 'Unavailable' : state.apiState === 'remote' ? 'Connected' : 'Local preview'}</h3>
          <p>${state.error ? escapeHtml(state.error.message || 'The bridge did not respond.') : state.apiState === 'remote' ? 'The browser is reading live data from the JSON bridge.' : 'Run the bridge to unlock live investigations.'}</p>
        </article>
        <article class="panel">
          <div class="section-label">Capabilities available in the backend</div>
          <div class="pill-list">
            ${['dataset profiling', 'quality checks', 'analysis planning', 'tool execution', 'evidence synthesis', 'guided checkpoints', 'collaborative tasks', 'reports']
              .map((item) => `<span class="pill">${escapeHtml(item)}</span>`)
              .join('')}
          </div>
        </article>
      </div>
    </section>
  `;
}

function renderWorkspace() {
  const investigation = getActiveInvestigation() || state.investigations[0] || emptyInvestigation();
  const stage = investigation.progress.find((item) => item.key === state.selectedStage) || investigation.progress.find((item) => item.status === 'current') || investigation.progress[0];
  const reportText = investigation.reports[state.selectedReport] || investigation.reports.analyst || investigation.reports.master;
  const status = text(investigation.status).toLowerCase();
  const mode = text(investigation.mode?.id || investigation.mode).toLowerCase();
  const isRunning = status.includes('running') || status.includes('queued') || status.includes('in_progress');
  const isAwaiting = status.includes('await');
  const isCompleted = status.includes('completed') || investigation.has_report || investigation.reports.analyst !== 'No report available yet.';

  if (mode === 'collaborative') {
    return renderCollaborativeWorkspace(investigation, stage, { completed: isCompleted });
  }

  if (isCompleted) {
    return renderCompletedWorkspace(investigation, stage, reportText);
  }

  if (mode === 'guided') {
    return renderGuidedWorkspace(investigation, stage, reportText, isAwaiting, isRunning);
  }

  if (isAwaiting) {
    return renderGuidedWorkspace(investigation, stage, reportText, true, isRunning);
  }

  if (isRunning) {
    return renderRunningWorkspace(investigation, stage, reportText);
  }

  return renderRunningWorkspace(investigation, stage, reportText);
}

function renderWorkspaceHeader(investigation, options = {}) {
  const status = options.status || investigation.status;
  const badgeClass = status.includes('await') ? 'warn' : status === 'completed' ? 'good' : 'active';
  return `
    <header class="workspace-hero">
      <div class="workspace-hero-topline">
        <button class="crumb" data-action="nav" data-view="investigations">Back to investigations</button>
        <div class="workspace-hero-actions">
          ${options.secondaryAction || ''}
          <span class="status-badge ${badgeClass}">${escapeHtml(status)}</span>
        </div>
      </div>
      <div class="workspace-hero-title">
        <div>
          <h1>${escapeHtml(investigation.question)}</h1>
          <div class="workspace-meta">
            <span>Dataset: ${escapeHtml(investigation.dataset?.name || 'Unknown dataset')}</span>
            <span>Mode: ${escapeHtml(investigation.mode.label)}</span>
            <span>${escapeHtml(options.subline || `Run ID: ${investigation.id}`)}</span>
          </div>
        </div>
        ${options.headerBadge ? `<div class="workspace-pill">${escapeHtml(options.headerBadge)}</div>` : ''}
      </div>
    </header>
  `;
}

function renderProgressRail(investigation, currentStage) {
  const stages = investigation.progress.map((step) => stageBlock(step, step.key === currentStage.key));
  const currentIndex = Math.max(0, investigation.progress.findIndex((item) => item.key === currentStage.key));
  const percent = Math.round(((currentIndex + 1) / Math.max(1, investigation.progress.length)) * 100);
  return `
    <section class="panel progress-panel">
      <div class="progress-strip" role="tablist" aria-label="Investigation progress">
        ${stages.join('')}
      </div>
      <div class="progress-bar-row">
        <div class="progress-bar"><span style="width: ${percent}%;"></span></div>
        <div class="progress-percent">${percent}%</div>
      </div>
    </section>
  `;
}

function renderCompletedWorkspace(investigation, stage, reportText) {
  const completedTabs = ['overview', 'findings', 'evidence', 'journey', 'reports'];
  const activeTab = completedTabs.includes(state.selectedStage) ? state.selectedStage : 'overview';
  return `
    <section class="workspace completed-workspace">
      ${renderWorkspaceHeader(investigation, {
        status: investigation.status,
        subline: `Completed ${investigation.updatedAt ? `on ${new Date(investigation.updatedAt).toLocaleString()}` : 'recently'}`,
        secondaryAction: `
          <button class="link-button" type="button" data-action="download-investigation">Download</button>
          <button class="link-button" type="button" data-action="share-investigation">Share</button>
        `,
        headerBadge: 'Completed',
      })}

      <nav class="workspace-tabs">
        ${[
          ['overview', 'Overview'],
          ['findings', 'Findings'],
          ['evidence', 'Evidence'],
          ['journey', 'Journey'],
          ['reports', 'Reports'],
        ]
          .map(
            ([key, label]) => `
              <button class="workspace-tab ${activeTab === key ? 'active' : ''}" data-action="stage" data-stage="${key}">
                ${label}
              </button>
            `,
          )
          .join('')}
      </nav>

      <div class="workspace-grid completed-grid">
        <main class="main-pane">
          ${activeTab === 'overview' ? `
            <section class="stage-panel answer-panel">
              <div class="section-head">
                <div>
                  <div class="section-label">Direct Answer</div>
                  <h2>${escapeHtml(investigation.answer.direct)}</h2>
                </div>
                <button class="link-button" type="button" data-action="open-answer">View more</button>
              </div>
              <div class="answer-card answer-hero">
                <p>${escapeHtml(investigation.answer.business || 'The answer is supported by the evidence trail below.')}</p>
                <div class="answer-meta">
                  <span class="status-badge ${investigation.confidence.label === 'High' ? 'good' : investigation.confidence.label === 'Moderate' ? 'warn' : 'active'}">Confidence ${escapeHtml(investigation.confidence.label)}</span>
                  <span>${escapeHtml(investigation.answer.position || 'unknown')}${investigation.selectedColumns.length ? ` | ${escapeHtml(investigation.selectedColumns.join(' | '))}` : ''}</span>
                </div>
              </div>
            </section>
            <section class="stage-panel">
              <div class="section-head">
                <div>
                  <div class="section-label">Key Recommendations</div>
                  <h2>What to do next</h2>
                </div>
                <button class="link-button" type="button" data-action="open-recommendation" data-index="0">View more</button>
              </div>
              ${investigation.recommendations.length ? investigation.recommendations.map((item, index) => `<button class="recommendation-row" type="button" data-action="open-recommendation" data-index="${index}"><span>${escapeHtml(item.label)}</span><strong>${escapeHtml(item.value)}</strong></button>`).join('') : '<p class="subtle">No recommendations were exposed.</p>'}
            </section>
          ` : ''}

          ${activeTab === 'findings' ? renderFindingsSection(investigation) : ''}
          ${activeTab === 'evidence' ? renderEvidenceSection(investigation) : ''}
          ${activeTab === 'journey' ? `
            ${renderJourneyPanel(investigation)}
          ` : ''}
          ${activeTab === 'reports' ? `
            <section class="stage-panel">
              <div class="section-head">
                <div>
                  <div class="section-label">Report</div>
                  <h2>Investigation output</h2>
                </div>
                <button class="link-button" type="button" data-action="open-report">View more</button>
              </div>
              <div class="report-tabs">
                ${['analyst', 'business', 'executive']
                  .map(
                    (report) => `
                      <button class="report-tab ${state.selectedReport === report ? 'active' : ''}" data-action="select-report" data-report="${report}">
                        ${report.charAt(0).toUpperCase() + report.slice(1)} Report
                      </button>
                    `,
                  )
                  .join('')}
              </div>
              <div class="report-body">
                <pre>${escapeHtml(reportText)}</pre>
              </div>
            </section>
          ` : ''}
        </main>

        <aside class="context-pane">
          ${renderConfidencePanel(investigation)}
          ${renderSummaryPanel(investigation)}
          ${renderSnapshotsPanel(investigation)}
        </aside>
      </div>

      ${state.drawer ? renderDrawer(investigation, state.drawer) : ''}
    </section>
  `;
}

function renderRunningWorkspace(investigation, stage) {
  const progressSteps = [
    'Planning',
    'Data understanding',
    'Analysis',
    'Findings',
    'Answer',
    'Complete',
  ];
  const currentIndex = Math.max(0, investigation.progress.findIndex((item) => item.key === stage.key));
  const activeLabel = progressSteps[Math.min(progressSteps.length - 1, currentIndex)];
  const workflowStatus = investigation.workflowStatus || {};
  const workflowMessage = workflowStatus.message || `Working on ${activeLabel.toLowerCase()} for ${investigation.question}`;
  const workflowPercent = Number.isFinite(Number(workflowStatus.progress)) ? Number(workflowStatus.progress) : Math.round(((currentIndex + 1) / Math.max(1, investigation.progress.length)) * 100);
  const activityLines = [
    workflowMessage,
    ...investigation.analysisPlan.slice(0, 5).map((item) => text(typeof item === 'string' ? item : item.description || item.purpose || item.tool || JSON.stringify(item))),
  ].filter(Boolean);
  const logLines = [
    `Loaded dataset: ${investigation.dataset.name}`,
    `Question captured and routed to the backend workflow`,
    `Current stage: ${stage.label}`,
    ...(investigation.journey.slice(0, 5).map((item) => `${item.label}: ${item.detail}`)),
  ];
  const percent = workflowPercent;
  return `
    <section class="workspace running-workspace">
      ${renderWorkspaceHeader(investigation, {
        status: investigation.status.includes('running') ? 'Running' : investigation.status,
        subline: `Run ID: ${investigation.id}`,
        secondaryAction: '<button class="link-button danger" type="button" data-action="cancel-run">Cancel run</button>',
        headerBadge: 'Running',
      })}

      ${renderProgressRail(investigation, stage)}

      <div class="workspace-grid running-grid">
        <section class="panel running-panel">
          <div class="panel-heading">
            <div>
              <div class="section-label">What is happening now</div>
              <h2>${escapeHtml(activityLines[0] || 'Analyzing your question')}</h2>
            </div>
          </div>
          <div class="timeline-list">
            ${activityLines.map((line, index) => `<div class="timeline-row ${index === 0 ? 'active' : ''}"><span class="timeline-dot"></span><span>${escapeHtml(line)}</span></div>`).join('')}
          </div>
          <div class="log-panel">
            ${logLines.map((line) => `<div class="log-row">${escapeHtml(line)}</div>`).join('')}
          </div>
        </section>

        <aside class="running-side">
          <section class="panel">
            <div class="section-label">Live progress</div>
            <div class="progress-ring">${percent}%</div>
            <div class="progress-bar-row">
              <div class="progress-bar"><span style="width: ${percent}%;"></span></div>
              <div class="progress-percent">${percent}%</div>
            </div>
            <p class="subtle">${escapeHtml(workflowMessage)}</p>
            <div class="metrics-grid">
              <div><span class="metric-label">Elapsed</span><strong>--:--</strong></div>
              <div><span class="metric-label">Remaining</span><strong>--:--</strong></div>
            </div>
          </section>

          <section class="panel">
            <div class="section-label">Current insight</div>
            <p class="subtle">${escapeHtml(investigation.answer.business || 'The investigation is still building evidence for a reliable answer.')}</p>
          </section>

          ${renderSnapshotsPanel(investigation)}
        </aside>
      </div>
    </section>
  `;
}

function renderGuidedWorkspace(investigation, stage, reportText, canInteract = false, isRunning = false) {
  const established = investigation.findings.slice(0, 3);
  const proposals = investigation.recommendations.slice(0, 3);
  const currentPhase = stage.label || 'Checkpoint';
  const workflowStatus = investigation.workflowStatus || {};
  const progressPercent = Number.isFinite(Number(workflowStatus.progress))
    ? Number(workflowStatus.progress)
    : Math.round((Math.max(1, investigation.progress.findIndex((item) => item.key === stage.key) + 1) / Math.max(1, investigation.progress.length)) * 100);
  const statusLabel = canInteract ? 'Awaiting your input' : isRunning ? 'Running guided workflow' : 'Guided';
  const bannerTitle = canInteract ? 'Review required' : isRunning ? 'Running guided workflow' : 'Guided workflow';
  return `
    <section class="workspace guided-workspace">
      ${renderWorkspaceHeader(investigation, {
        status: statusLabel,
        subline: `Checkpoint: ${currentPhase}`,
        headerBadge: canInteract ? 'Review required' : `Progress ${progressPercent}%`,
      })}

      ${renderProgressRail(investigation, stage)}

      <section class="review-banner">
        <div>
          <div class="review-label">${escapeHtml(bannerTitle)}</div>
          <p>${escapeHtml(canInteract ? (investigation.answer.business || 'The workflow has reached a checkpoint and is waiting for your decision before it proceeds.') : (workflowStatus.message || 'The guided workflow is still maturing. Checkpoints and decisions will appear as the backend advances.'))}</p>
          <div class="progress-bar-row">
            <div class="progress-bar"><span style="width: ${progressPercent}%;"></span></div>
            <div class="progress-percent">${progressPercent}%</div>
          </div>
        </div>
      </section>

      <div class="workspace-grid guided-grid">
        <main class="main-pane">
          <section class="panel">
            <div class="panel-heading">
              <div>
                <div class="section-label">What has been established</div>
                <h2>Current findings</h2>
              </div>
            </div>
            <div class="guided-cards">
              ${established.length ? established.map((finding) => `
                <article class="mini-card">
                  <strong>${escapeHtml(finding.title)}</strong>
                  <p>${escapeHtml(finding.summary)}</p>
                </article>
              `).join('') : '<div class="empty-state">No findings have been surfaced yet.</div>'}
            </div>
          </section>

          <section class="panel">
            <div class="panel-heading">
              <div>
                <div class="section-label">What we propose to do next</div>
                <h2>Recommended next step</h2>
              </div>
            </div>
            <div class="guided-cards">
              ${proposals.length ? proposals.map((item) => `
                <article class="mini-card">
                  <strong>${escapeHtml(item.label)}</strong>
                  <p>${escapeHtml(item.value)}</p>
                </article>
              `).join('') : '<div class="empty-state">No proposal is currently exposed.</div>'}
            </div>
          </section>

          <section class="panel">
            <div class="panel-heading">
              <div>
                <div class="section-label">Decision</div>
                <h2>Choose how to proceed</h2>
              </div>
            </div>
            <div class="decision-grid">
              <button class="decision-card continue" data-action="guided-continue" ${canInteract ? '' : 'disabled aria-disabled="true"'}>
                <strong>Continue</strong>
                <span>Proceed with this plan</span>
              </button>
              <button class="decision-card modify" data-action="guided-modify" ${canInteract ? '' : 'disabled aria-disabled="true"'}>
                <strong>Modify</strong>
                <span>Adjust the approach</span>
              </button>
              <button class="decision-card cancel" data-action="guided-stop" ${canInteract ? '' : 'disabled aria-disabled="true"'}>
                <strong>Cancel</strong>
                <span>Stop this investigation</span>
              </button>
            </div>
          </section>
        </main>

        <aside class="running-side">
          ${renderConfidencePanel(investigation)}
          ${renderSnapshotsPanel(investigation)}
          <section class="panel">
            <div class="section-label">What happens next</div>
            <p class="subtle">${escapeHtml(canInteract ? 'The workflow will pause until you make a decision.' : 'The backend is still progressing toward the checkpoint. The controls will unlock when input is needed.')}</p>
          </section>
        </aside>
      </div>

      ${state.drawer ? renderDrawer(investigation, state.drawer) : ''}
    </section>
  `;
}

function renderCollaborativeWorkspace(investigation, stage, options = {}) {
  const completed = Boolean(options.completed || text(investigation.status).toLowerCase().includes('completed'));
  return `
    <section class="workspace collaborative-workspace">
      ${renderWorkspaceHeader(investigation, {
        status: completed ? 'completed' : 'collaborative',
        subline: `Question: ${investigation.question}`,
        headerBadge: completed ? 'Completed' : 'Active',
      })}

      <div class="workspace-grid collaborative-grid">
        <aside class="collab-left">
          <section class="panel">
            <div class="section-label">Tasks</div>
            <div class="task-list">
              ${investigation.tasks.length ? investigation.tasks.map((task) => `
                <div class="task-row">
                  <div>
                    <strong>${escapeHtml(task.title)}</strong>
                    <p>${escapeHtml(task.summary || task.status)}</p>
                  </div>
                  <span class="task-badge">${escapeHtml(task.status)}</span>
                </div>
              `).join('') : '<div class="empty-state">No tasks are currently queued.</div>'}
            </div>
          </section>

          <section class="panel">
            <div class="section-label">Hypotheses</div>
            <div class="task-list">
              ${investigation.hypotheses.length ? investigation.hypotheses.map((hypothesis) => `
                <div class="task-row">
                  <div>
                    <strong>${escapeHtml(hypothesis.statement)}</strong>
                    <p>${escapeHtml(hypothesis.notes || hypothesis.status)}</p>
                  </div>
                  <span class="task-badge">${escapeHtml(hypothesis.confidence)}</span>
                </div>
              `).join('') : '<div class="empty-state">No hypotheses are currently tracked.</div>'}
            </div>
          </section>
        </aside>

        <main class="main-pane">
          <section class="panel">
            <div class="panel-heading">
              <div>
                <div class="section-label">Investigation space</div>
                <h2>${escapeHtml(stage.label || 'Current investigation')}</h2>
              </div>
            </div>
            <div class="workspace-tabs">
              ${['evidence', 'findings', 'answer'].map((tab) => `
                <button class="workspace-tab ${state.selectedStage === tab ? 'active' : ''}" data-action="stage" data-stage="${tab}">
                  ${tab.charAt(0).toUpperCase() + tab.slice(1)}
                </button>
              `).join('')}
            </div>
            ${renderMainStage(investigation, stage)}
          </section>

          ${renderFindingsSection(investigation)}
          ${renderEvidenceSection(investigation)}
        </main>

        <aside class="running-side">
          ${renderConfidencePanel(investigation)}
          ${renderSummaryPanel(investigation)}
          ${renderSnapshotsPanel(investigation)}
          <section class="panel">
            <div class="section-label">Next best actions</div>
            ${investigation.recommendations.length ? investigation.recommendations.slice(0, 4).map((item, index) => `<button class="recommendation-row" type="button" data-action="open-recommendation" data-index="${index}"><span>${escapeHtml(item.label)}</span><strong>${escapeHtml(item.value)}</strong></button>`).join('') : '<p class="subtle">No next action is currently exposed.</p>'}
          </section>
        </aside>
      </div>

      <footer class="command-bar">
        <div class="command-input">
          <label for="commandDraft">Command center</label>
          <input id="commandDraft" name="commandDraft" type="text" value="${escapeHtml(state.commandDraft)}" placeholder="Ask a question, request analysis, or assign a task..." />
        </div>
        <button class="primary-action" data-action="run-command">Send</button>
      </footer>

      ${state.drawer ? renderDrawer(investigation, state.drawer) : ''}
    </section>
  `;
}

function renderMainStage(investigation, stage) {
  if (stage.key === 'data') {
    return `
      <section class="stage-panel">
        <div class="section-label">Data</div>
        <div class="summary-grid">
          <div class="summary-card">
            <span class="summary-label">Rows</span>
            <strong>${investigation.dataQuality.rows ? Number(investigation.dataQuality.rows).toLocaleString() : 'Loading'}</strong>
          </div>
          <div class="summary-card">
            <span class="summary-label">Columns</span>
            <strong>${investigation.dataQuality.columns || 'Loading'}</strong>
          </div>
          <div class="summary-card">
            <span class="summary-label">Source</span>
            <strong>${escapeHtml(investigation.dataset.sourceType || 'csv')}</strong>
          </div>
        </div>
      </section>
    `;
  }

  if (stage.key === 'quality') {
    return `
      <section class="stage-panel">
        <div class="section-label">Data Quality</div>
        <div class="quality-header">
          <strong>${investigation.dataQuality.issues || 0} issue${investigation.dataQuality.issues === 1 ? '' : 's'} detected</strong>
          <span class="subtle">${investigation.dataQuality.rows ? `${Number(investigation.dataQuality.rows).toLocaleString()} rows` : 'Row count unavailable'} | ${investigation.dataQuality.columns || 'Unknown'} columns</span>
        </div>
        <div class="pill-list">
          ${investigation.dataQuality.anomalies.length ? investigation.dataQuality.anomalies.map((item) => `<span class="pill warning">${escapeHtml(typeof item === 'string' ? item : item.reason || JSON.stringify(item))}</span>`).join('') : '<span class="pill good">No major anomalies highlighted</span>'}
          ${investigation.dataQuality.warnings.length ? investigation.dataQuality.warnings.map((item) => `<span class="pill warning">${escapeHtml(typeof item === 'string' ? item : item.reason || JSON.stringify(item))}</span>`).join('') : ''}
        </div>
      </section>
    `;
  }

  if (stage.key === 'plan') {
    return `
      <section class="stage-panel">
        <div class="section-label">Investigation Plan</div>
        <ol class="plan-list">
          ${investigation.analysisPlan.length
            ? investigation.analysisPlan.map((step) => `<li>${escapeHtml(typeof step === 'string' ? step : step.description || step.purpose || step.tool || JSON.stringify(step))}</li>`).join('')
            : '<li>No explicit plan was exposed, but the backend still generated the analysis path.</li>'}
        </ol>
      </section>
    `;
  }

  if (stage.key === 'finding') {
    return `
      <section class="stage-panel">
        <div class="section-head">
          <div>
            <div class="section-label">Direct Answer</div>
            <h2>${escapeHtml(investigation.answer.direct)}</h2>
          </div>
          <button class="link-button" type="button" data-action="open-answer">View more</button>
        </div>
        <div class="answer-card">
          <p>${escapeHtml(investigation.answer.business || 'The answer is derived from the strongest evidence in the investigation.')}</p>
        </div>
      </section>
    `;
  }

  if (stage.key === 'answer') {
    const selectedColumns = investigation.selectedColumns.map((column) => text(column)).filter(Boolean);
    return `
      <section class="stage-panel">
        <div class="section-head">
          <div>
            <div class="section-label">Answer</div>
            <h2>${escapeHtml(investigation.answer.direct)}</h2>
          </div>
          <button class="link-button" type="button" data-action="open-answer">View more</button>
        </div>
        <div class="answer-card answer-hero">
          <div class="answer-meta">
            <span class="status-badge ${investigation.confidence.label === 'High' ? 'good' : investigation.confidence.label === 'Moderate' ? 'warn' : 'active'}">Confidence ${escapeHtml(investigation.confidence.label)}</span>
            <span>${escapeHtml(investigation.answer.position || 'unknown')}${selectedColumns.length ? ` | ${escapeHtml(selectedColumns.join(' | '))}` : ''}</span>
          </div>
          <p>${escapeHtml(investigation.answer.business || investigation.confidence.reason || 'The answer is supported by the evidence trail below.')}</p>
        </div>
      </section>
    `;
  }

  return `
    <section class="stage-panel">
      <div class="section-label">${escapeHtml(formatStageLabel(stage.key))}</div>
      <div class="summary-callout">
        <strong>${escapeHtml(stage.label)}</strong>
        <p>${escapeHtml(stage.detail)}</p>
      </div>
    </section>
  `;
}

function renderFindingsSection(investigation) {
  const findings = investigation.findings;
  return `
    <section class="stage-panel">
      <div class="section-head">
        <div>
          <div class="section-label">Findings</div>
          <h2>Evidence-backed observations</h2>
        </div>
        <button class="link-button" type="button" data-action="open-finding" data-index="0">View more</button>
        <div class="subtle">${findings.length ? `${findings.length} finding${findings.length === 1 ? '' : 's'}` : 'No findings yet'}</div>
      </div>
      <div class="finding-list">
        ${findings.length
          ? findings
              .map(
                (finding, index) => `
                  <button class="finding-card" data-action="open-finding" data-index="${index}">
                    <div class="finding-top">
                      <span class="finding-index">${String(index + 1).padStart(2, '0')}</span>
                      <span class="finding-confidence">${escapeHtml(finding.confidence)}</span>
                    </div>
                    <h3>${escapeHtml(finding.title)}</h3>
                    <p>${escapeHtml(finding.summary)}</p>
                    <div class="finding-footer">
                      <span>${escapeHtml(finding.method)}</span>
                      <span>${escapeHtml(finding.relationshipType)}</span>
                    </div>
                  </button>
                `,
              )
              .join('')
          : '<div class="empty-state">No findings are available yet. The investigation is still assembling evidence.</div>'}
      </div>
    </section>
  `;
}

function renderEvidenceSection(investigation) {
  const chartItems = toArray(investigation.visualizations).filter(Boolean);
  return `
    <section class="stage-panel">
      <div class="section-head">
        <div>
          <div class="section-label">Evidence</div>
          <h2>Supporting evidence and supporting details</h2>
        </div>
        <button class="link-button" type="button" data-action="open-evidence" data-index="0">View more</button>
      </div>
      <div class="evidence-grid">
        ${investigation.evidence.length
          ? investigation.evidence.slice(0, 6).map(
              (item, index) => `
                <button class="evidence-chip" data-action="open-evidence" data-index="${index}">
                  <span class="evidence-kind">${escapeHtml(item.label)}</span>
                  <span class="evidence-summary">${escapeHtml(item.summary)}</span>
                  <span class="evidence-meta">${escapeHtml(item.confidence)}</span>
                </button>
              `,
            ).join('')
          : '<div class="empty-state">The backend has not exposed supporting evidence yet.</div>'}
      </div>
      ${chartItems.length ? `
        <div class="section-label mt">Charts</div>
        <div class="evidence-chart-grid">
          ${chartItems.slice(0, 6).map((visual, index) => {
            const src = chartAssetUrl(visual);
            const title = text(visual.title || visual.caption || visual.type || `Chart ${index + 1}`);
            const summary = text(visual.caption || visual.summary || visual.file_path || visual.path || 'Generated chart');
            return `
              <button class="evidence-chart-card" type="button" data-action="open-visualization" data-index="${index}">
                ${src ? `<img class="evidence-chart-image" src="${escapeHtml(src)}" alt="${escapeHtml(title)}" loading="lazy" />` : ''}
                <div class="evidence-chart-body">
                  <strong>${escapeHtml(title)}</strong>
                  <span>${escapeHtml(summary)}</span>
                </div>
              </button>
            `;
          }).join('')}
        </div>
      ` : ''}
    </section>
  `;
}

function renderReportSection(investigation, reportText) {
  return `
    <section class="stage-panel">
      <div class="section-head">
        <div>
          <div class="section-label">Reports</div>
          <h2>Investigation outputs</h2>
        </div>
        <button class="link-button" type="button" data-action="open-report">View more</button>
        <div class="report-tabs compact">
          ${['analyst', 'business', 'executive']
            .map(
              (report) => `
                <button class="report-tab ${state.selectedReport === report ? 'active' : ''}" data-action="select-report" data-report="${report}">
                  ${report.charAt(0).toUpperCase() + report.slice(1)}
                </button>
              `,
            )
            .join('')}
        </div>
      </div>
      <div class="report-body">
        <pre>${escapeHtml(reportText)}</pre>
      </div>
    </section>
  `;
}

function renderConfidencePanel(investigation) {
  return `
    <section class="context-card">
      <div class="section-label">Confidence</div>
      <div class="confidence-value">${escapeHtml(investigation.confidence.label)}</div>
      <p>${escapeHtml(investigation.confidence.reason || 'Confidence is derived from the investigation evidence trail.')}</p>
      <div class="signal-list">
        ${[
          investigation.confidence.evidence,
          investigation.confidence.interpretation,
          investigation.confidence.business,
          investigation.confidence.recommendation,
        ]
          .filter(Boolean)
          .map((signal) => `<div class="signal-row"><span class="signal-check">+</span><span>${escapeHtml(text(signal))}</span></div>`)
          .join('')}
      </div>
    </section>
  `;
}

function renderSummaryPanel(investigation) {
  return `
    <section class="context-card">
      <div class="section-label">Summary</div>
      <dl class="meta-stack">
        <div><dt>Dataset</dt><dd>${escapeHtml(investigation.dataset.name)}</dd></div>
        <div><dt>Rows</dt><dd>${investigation.dataset.rowCount ? Number(investigation.dataset.rowCount).toLocaleString() : 'Unknown'}</dd></div>
        <div><dt>Columns</dt><dd>${investigation.dataset.columnCount || 'Unknown'}</dd></div>
        <div><dt>Mode</dt><dd>${escapeHtml(investigation.mode.label)}</dd></div>
      </dl>
      <div class="section-label mt">Recommended action</div>
      ${investigation.recommendations.length ? investigation.recommendations.map((item, index) => `<button class="recommendation-row" type="button" data-action="open-recommendation" data-index="${index}"><span>${escapeHtml(item.label)}</span><strong>${escapeHtml(item.value)}</strong></button>`).join('') : '<p class="subtle">No recommendation is currently exposed.</p>'}
    </section>
  `;
}

function renderJsonPanel(title, value) {
  return `
    <div class="drawer-block">
      <h3>${escapeHtml(title)}</h3>
      <pre>${escapeHtml(JSON.stringify(value ?? {}, null, 2))}</pre>
    </div>
  `;
}

function renderListPanel(title, items, formatter) {
  return `
    <div class="drawer-block">
      <h3>${escapeHtml(title)}</h3>
      ${items.length ? `<ul>${items.map((item) => `<li>${escapeHtml(formatter(item))}</li>`).join('')}</ul>` : '<p class="subtle">None available.</p>'}
    </div>
  `;
}

function renderJourneyPanel(investigation) {
  return `
    <section class="context-card">
      <div class="section-head compact-head">
        <div>
          <div class="section-label">Journey</div>
          <div class="subtle">${investigation.journey.length ? `${investigation.journey.length} steps available` : 'Journey available from the workflow trail'}</div>
        </div>
        <button class="link-button" type="button" data-action="open-journey">View more</button>
      </div>
      <div class="journey-list">
        ${investigation.journey.length
          ? investigation.journey.map(
              (step) => `
                <div class="journey-step ${step.status}">
                  <div class="journey-bullet"></div>
                  <div>
                    <strong>${escapeHtml(step.label)}</strong>
                    <p>${escapeHtml(step.detail)}</p>
                    ${step.timestamp ? `<span class="subtle">${escapeHtml(new Date(step.timestamp).toLocaleString())}</span>` : ''}
                  </div>
                </div>
              `,
            ).join('')
          : '<p class="subtle">The journey will appear once the investigation begins to move through the workflow.</p>'}
      </div>
    </section>
  `;
}

function renderSnapshotsPanel(investigation) {
  if (!investigation.snapshots.length) {
    return `
      <section class="context-card">
        <div class="section-label">Snapshots</div>
        <p class="subtle">Snapshot history will appear here once the guided workflow records checkpoint versions.</p>
      </section>
    `;
  }

  return `
    <section class="context-card">
      <div class="section-head compact-head">
        <div>
          <div class="section-label">Snapshots</div>
          <div class="subtle">${investigation.snapshots.length} checkpoint${investigation.snapshots.length === 1 ? '' : 's'} captured</div>
        </div>
        <button class="link-button" data-action="open-snapshots">View all</button>
      </div>
      <div class="snapshot-list">
        ${investigation.snapshots.slice(0, 3).map((snapshot, index) => `
          <button class="snapshot-card" data-action="open-snapshot" data-index="${index}">
            <strong>${escapeHtml(snapshot.label)}</strong>
            <span>${escapeHtml(snapshot.summary)}</span>
            <div class="snapshot-meta">
              <span>${escapeHtml(snapshot.kindLabel || snapshot.kind || 'Snapshot')}</span>
              <span>${escapeHtml(snapshot.stage || 'Stage')}</span>
            </div>
          </button>
        `).join('')}
      </div>
    </section>
  `;
}

function renderControlPanel(investigation) {
  if (investigation.mode.id === 'guided') {
    return `
      <section class="context-card">
        <div class="section-label">Guided checkpoint</div>
        <p class="subtle">This mode pauses for review at meaningful points. Use the controls in the command bar to continue, modify, or stop.</p>
      </section>
    `;
  }
  if (investigation.mode.id === 'collaborative') {
    return `
      <section class="context-card">
        <div class="section-label">Collaborative context</div>
        <div class="signal-list">
          ${investigation.tasks.length ? investigation.tasks.slice(0, 4).map((task) => `<div class="signal-row"><span class="signal-check">-</span><span>${escapeHtml(task.title)} <em>(${escapeHtml(task.status)})</em></span></div>`).join('') : '<div class="signal-row"><span class="signal-check">-</span><span>No tasks tracked yet.</span></div>'}
        </div>
      </section>
    `;
  }
  return `
    <section class="context-card">
      <div class="section-label">Next actions</div>
      <p class="subtle">Autonomous mode is observational. Open a new investigation or inspect the answer and evidence above.</p>
    </section>
  `;
}

function renderDrawer(investigation, drawer) {
  if (!drawer) return '';
  if (drawer.kind === 'finding') {
    const finding = investigation.findings[drawer.index];
    if (!finding) return '';
    return `
      <div class="drawer-backdrop" data-action="close-drawer">
        <div class="drawer" role="dialog" aria-modal="true" aria-label="Finding details">
          <div class="drawer-head">
            <div>
              <div class="section-label">Finding</div>
              <h2>${escapeHtml(finding.title)}</h2>
            </div>
            <button class="icon-button" data-action="close-drawer">Close</button>
          </div>
          <div class="drawer-body">
            <p>${escapeHtml(finding.summary)}</p>
            <div class="drawer-block">
              <h3>Why it matters</h3>
              <p>${escapeHtml(finding.whyItMatters)}</p>
            </div>
            <div class="drawer-block">
              <h3>Method</h3>
              <p>${escapeHtml(finding.method)}</p>
            </div>
            <div class="drawer-block">
              <h3>Limitations</h3>
              ${finding.limitations.length ? `<ul>${finding.limitations.map((item) => `<li>${escapeHtml(plainText(item))}</li>`).join('')}</ul>` : '<p class="subtle">No explicit limitations were surfaced.</p>'}
            </div>
            <div class="drawer-block">
              <h3>Supporting evidence</h3>
              ${finding.supportingEvidence.length ? `<ul>${finding.supportingEvidence.map((item) => `<li>${escapeHtml(plainText(item.statement || item.summary || item.insight || item))}</li>`).join('')}</ul>` : '<p class="subtle">The finding is supported by the broader evidence trail above.</p>'}
            </div>
          </div>
        </div>
      </div>
    `;
  }
  if (drawer.kind === 'evidence') {
    const item = investigation.evidence[drawer.index];
    if (!item) return '';
    return `
      <div class="drawer-backdrop" data-action="close-drawer">
        <div class="drawer" role="dialog" aria-modal="true" aria-label="Evidence details">
          <div class="drawer-head">
            <div>
              <div class="section-label">Evidence</div>
              <h2>${escapeHtml(item.label)}</h2>
            </div>
            <button class="icon-button" data-action="close-drawer">Close</button>
          </div>
          <div class="drawer-body">
            <p>${escapeHtml(item.summary)}</p>
            ${item.kind === 'visual' && chartAssetUrl(item.detail) ? `<img class="drawer-chart-image" src="${escapeHtml(chartAssetUrl(item.detail))}" alt="${escapeHtml(item.label)}" />` : ''}
            <pre>${escapeHtml(item.detail)}</pre>
          </div>
        </div>
      </div>
    `;
  }
  if (drawer.kind === 'visualization') {
    const visual = investigation.visualizations[drawer.index];
    if (!visual) return '';
    const src = chartAssetUrl(visual);
    const title = text(visual.title || visual.caption || visual.type || 'Chart');
    return `
      <div class="drawer-backdrop" data-action="close-drawer">
        <div class="drawer drawer-wide" role="dialog" aria-modal="true" aria-label="Visualization details">
          <div class="drawer-head">
            <div>
              <div class="section-label">Visualization</div>
              <h2>${escapeHtml(title)}</h2>
            </div>
            <button class="icon-button" data-action="close-drawer">Close</button>
          </div>
          <div class="drawer-body">
            ${src ? `<img class="drawer-chart-image" src="${escapeHtml(src)}" alt="${escapeHtml(title)}" />` : ''}
            <p>${escapeHtml(text(visual.caption || visual.summary || visual.file_path || visual.path || 'Generated chart'))}</p>
            ${renderJsonPanel('Raw visualization payload', visual)}
          </div>
        </div>
      </div>
    `;
  }
  if (drawer.kind === 'answer') {
    return `
      <div class="drawer-backdrop" data-action="close-drawer">
        <div class="drawer drawer-wide" role="dialog" aria-modal="true" aria-label="Answer details">
          <div class="drawer-head">
            <div>
              <div class="section-label">Direct Answer</div>
              <h2>${escapeHtml(investigation.answer.direct)}</h2>
            </div>
            <button class="icon-button" data-action="close-drawer">Close</button>
          </div>
          <div class="drawer-body">
            <div class="drawer-block">
              <h3>Business interpretation</h3>
              <p>${escapeHtml(investigation.answer.business || 'No business interpretation was exposed.')}</p>
            </div>
            <div class="snapshot-card-grid">
              <article class="snapshot-detail-card"><span class="snapshot-detail-label">Confidence</span><strong>${escapeHtml(investigation.confidence.label)}</strong></article>
              <article class="snapshot-detail-card"><span class="snapshot-detail-label">Position</span><strong>${escapeHtml(investigation.answer.position || 'unknown')}</strong></article>
              <article class="snapshot-detail-card"><span class="snapshot-detail-label">Mode</span><strong>${escapeHtml(investigation.mode.label)}</strong></article>
              <article class="snapshot-detail-card"><span class="snapshot-detail-label">Selected columns</span><strong>${escapeHtml(investigation.selectedColumns.length ? investigation.selectedColumns.join(' | ') : 'None')}</strong></article>
            </div>
            ${renderListPanel('Supporting evidence', investigation.answer.supportingEvidence, (item) => plainText(item.statement || item.summary || item.insight || item))}
            ${renderListPanel('Observed facts', investigation.answer.observedFacts, (item) => plainText(item))}
            ${renderListPanel('Analytical interpretation', investigation.answer.analyticalInterpretation, (item) => plainText(item))}
            ${renderListPanel('Assumptions', investigation.answer.assumptions, (item) => plainText(item))}
            ${renderListPanel('Uncertainty', investigation.answer.uncertainty, (item) => plainText(item))}
            ${renderListPanel('Next investigation', investigation.answer.nextInvestigation, (item) => plainText(typeof item === 'string' ? item : item.request || item.title || item.summary || item.value || JSON.stringify(item)))}
            ${renderJsonPanel('Raw answer payload', investigation.raw?.analysis_evidence?.answer_synthesis || investigation.raw?.answer_synthesis || {})}
          </div>
        </div>
      </div>
    `;
  }
  if (drawer.kind === 'report') {
    const reportKey = state.selectedReport || 'analyst';
    const reports = investigation.reports || {};
    const reportMap = {
      analyst: reports.analyst,
      business: reports.business,
      executive: reports.executive,
      master: reports.master,
      answer: reports.answerSynthesis,
      decision: reports.decision,
    };
    const selectedReport = reportMap[reportKey] || reports.analyst || reports.master || '';
    const reportPackage = investigation.raw?.analysis_evidence?.report_package || investigation.raw?.report_package || {};
    return `
      <div class="drawer-backdrop" data-action="close-drawer">
        <div class="drawer drawer-wide" role="dialog" aria-modal="true" aria-label="Report details">
          <div class="drawer-head">
            <div>
              <div class="section-label">Reports</div>
              <h2>${escapeHtml(investigation.question)}</h2>
            </div>
            <button class="icon-button" data-action="close-drawer">Close</button>
          </div>
          <div class="drawer-body">
            <div class="report-tabs">
              ${[
                ['analyst', 'Analyst'],
                ['business', 'Business'],
                ['executive', 'Executive'],
                ['master', 'Master'],
                ['answer', 'Answer synthesis'],
                ['decision', 'Decision'],
              ].map(([key, label]) => `
                <button class="report-tab ${reportKey === key ? 'active' : ''}" data-action="select-report" data-report="${key}">${label}</button>
              `).join('')}
            </div>
            <div class="drawer-block">
              <h3>${escapeHtml(reportKey.charAt(0).toUpperCase() + reportKey.slice(1))} report</h3>
              <pre>${escapeHtml(selectedReport || 'No report available yet.')}</pre>
            </div>
            ${renderJsonPanel('Report bundle', reportPackage)}
            ${renderJsonPanel('Traceability', reportPackage.traceability || {})}
            ${renderJsonPanel('Answer synthesis', investigation.raw?.analysis_evidence?.answer_synthesis || investigation.raw?.answer_synthesis || {})}
          </div>
        </div>
      </div>
    `;
  }
  if (drawer.kind === 'recommendation') {
    const recommendations = investigation.recommendations || [];
    const selected = recommendations[drawer.index] || recommendations[0] || null;
    return `
      <div class="drawer-backdrop" data-action="close-drawer">
        <div class="drawer drawer-wide" role="dialog" aria-modal="true" aria-label="Recommendation details">
          <div class="drawer-head">
            <div>
              <div class="section-label">Recommendations</div>
              <h2>What to do next</h2>
            </div>
            <button class="icon-button" data-action="close-drawer">Close</button>
          </div>
          <div class="drawer-body">
            ${selected ? `
              <div class="drawer-block">
                <h3>${escapeHtml(selected.label || 'Recommended action')}</h3>
                <p>${escapeHtml(selected.value || '')}</p>
              </div>
            ` : ''}
            ${recommendations.length ? recommendations.map((item, index) => `
              <div class="drawer-block">
                <h3>${escapeHtml(item.label || `Recommendation ${index + 1}`)}</h3>
                <p>${escapeHtml(item.value)}</p>
              </div>
            `).join('') : '<p class="subtle">No recommendations were exposed.</p>'}
            ${renderJsonPanel('Raw recommendation payload', investigation.raw?.analysis_evidence?.answer_synthesis || investigation.raw?.answer_synthesis || {})}
          </div>
        </div>
      </div>
    `;
  }
  if (drawer.kind === 'journey') {
    return `
      <div class="drawer-backdrop" data-action="close-drawer">
        <div class="drawer drawer-wide" role="dialog" aria-modal="true" aria-label="Journey details">
          <div class="drawer-head">
            <div>
              <div class="section-label">Journey</div>
              <h2>Workflow trail</h2>
            </div>
            <button class="icon-button" data-action="close-drawer">Close</button>
          </div>
          <div class="drawer-body">
            <div class="journey-list">
              ${investigation.journey.length
                ? investigation.journey.map((step, index) => `
                  <div class="journey-step ${step.status}">
                    <div class="journey-bullet"></div>
                    <div>
                      <strong>${escapeHtml(step.label)}</strong>
                      <p>${escapeHtml(step.detail)}</p>
                      <div class="subtle">Step ${index + 1}${step.timestamp ? ` | ${escapeHtml(new Date(step.timestamp).toLocaleString())}` : ''}</div>
                    </div>
                  </div>
                `).join('')
                : '<p class="subtle">No journey trail was exposed.</p>'}
            </div>
            ${renderJsonPanel('Raw journey payload', {
              guided_decision_log: investigation.raw?.guided_decision_log || investigation.raw?.analysis_evidence?.guided_decision_log || [],
              collaborative_decision_log: investigation.raw?.collaborative_decision_log || investigation.raw?.analysis_evidence?.collaborative_decision_log || [],
            })}
          </div>
        </div>
      </div>
    `;
  }
  if (drawer.kind === 'snapshot') {
    const item = investigation.snapshots[drawer.index];
    if (!item) return '';
    return `
      <div class="drawer-backdrop" data-action="close-drawer">
        <div class="drawer" role="dialog" aria-modal="true" aria-label="Snapshot details">
          <div class="drawer-head">
            <div>
              <div class="section-label">Snapshot</div>
              <h2>${escapeHtml(item.label)}</h2>
            </div>
            <button class="icon-button" data-action="close-drawer">Close</button>
          </div>
          <div class="drawer-body">
            <p>${escapeHtml(item.summary)}</p>
            <div class="snapshot-card-grid">
              ${(item.fields || []).map((field) => `
                <article class="snapshot-detail-card">
                  <span class="snapshot-detail-label">${escapeHtml(field.label)}</span>
                  <strong>${escapeHtml(field.value)}</strong>
                </article>
              `).join('')}
            </div>
            ${item.extra?.length ? `
              <div class="drawer-block">
                <h3>Highlights</h3>
                <ul>
                  ${item.extra.map((value) => `<li>${escapeHtml(plainText(value))}</li>`).join('')}
                </ul>
              </div>
            ` : ''}
            <div class="drawer-block">
              <h3>Raw snapshot</h3>
              <pre>${escapeHtml(JSON.stringify(item.detail, null, 2))}</pre>
            </div>
          </div>
        </div>
      </div>
    `;
  }
  if (drawer.kind === 'snapshots') {
    return `
      <div class="drawer-backdrop" data-action="close-drawer">
        <div class="drawer drawer-wide" role="dialog" aria-modal="true" aria-label="All snapshots">
          <div class="drawer-head">
            <div>
              <div class="section-label">Snapshots</div>
              <h2>Checkpoint history</h2>
            </div>
            <button class="icon-button" data-action="close-drawer">Close</button>
          </div>
          <div class="drawer-body">
            <div class="snapshot-history-grid">
              ${investigation.snapshots.map((snapshot, index) => `
                <button class="snapshot-card snapshot-history-card" data-action="open-snapshot" data-index="${index}">
                  <div class="snapshot-history-topline">
                    <strong>${escapeHtml(snapshot.label)}</strong>
                    <span>${escapeHtml(snapshot.kindLabel || snapshot.kind || 'Snapshot')}</span>
                  </div>
                  <span>${escapeHtml(snapshot.summary)}</span>
                  <div class="snapshot-meta">
                    <span>${escapeHtml(snapshot.stage || 'Stage')}</span>
                    <span>${escapeHtml(snapshot.version ? `v${snapshot.version}` : 'Summary')}</span>
                  </div>
                </button>
              `).join('')}
            </div>
          </div>
        </div>
      </div>
    `;
  }
  return '';
}

function renderTopbar() {
  const investigation = getActiveInvestigation();
  return `
    <header class="topbar">
      <div>
        <div class="topbar-kicker">Investigation workspace</div>
        <div class="topbar-title">${escapeHtml(investigation?.question || state.bootstrap.app.tagline)}</div>
      </div>
      <div class="topbar-actions">
        <span class="topbar-chip">${state.loading ? 'Executing task' : state.apiState === 'remote' ? 'Live bridge' : 'Local preview'}</span>
        <span class="topbar-chip">${investigation ? `${escapeHtml(investigation.mode.label)} mode` : 'No active investigation'}</span>
      </div>
    </header>
  `;
}

function renderApp() {
  const investigation = getActiveInvestigation();
  const content = state.view === 'home'
    ? renderHome()
    : state.view === 'investigations'
      ? state.requestedInvestigationId
        ? renderWorkspace()
        : renderInvestigations()
    : state.view === 'new'
      ? renderNewInvestigation()
      : state.view === 'datasets'
        ? renderDatasets()
        : state.view === 'reports'
          ? renderReports()
          : state.view === 'settings'
            ? renderSettings()
            : renderWorkspace();

  app.innerHTML = `
    <div class="app-shell">
      ${renderSidebar()}
      <div class="main-shell">
        ${renderTopbar()}
        <main class="content-shell">${content}</main>
      </div>
    </div>
  `;

  if (state.error) {
    const banner = document.createElement('div');
    banner.className = 'error-banner';
    banner.textContent = state.error.message || 'Backend unavailable';
    app.prepend(banner);
  }
}

function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function text(value) {
  if (value === null || value === undefined) return '';
  if (typeof value === 'string') return value.trim();
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  if (Array.isArray(value)) {
    return value
      .map((item) => text(item))
      .filter(Boolean)
      .join(', ')
      .trim();
  }
  if (typeof value === 'object') {
    const preferredKeys = [
      'label',
      'name',
      'title',
      'headline',
      'summary',
      'text',
      'value',
      'reason',
      'description',
      'insight',
      'message',
      'direct_answer',
      'business_interpretation',
      'statement',
      'request',
      'question',
      'answer',
      'detail',
      'explanation',
      'recommended_action',
      'decision_summary',
      'confidence',
    ];
    for (const key of preferredKeys) {
      const result = text(value[key]);
      if (result) return result;
    }
    const entries = Object.entries(value);
    if (entries.length === 1) {
      return text(entries[0][1]);
    }
    try {
      const result = JSON.stringify(value);
      if (result && result !== '{}') return result;
    } catch (error) {
      void error;
    }
    return '';
  }
  return String(value).trim();
}

function plainText(value) {
  return text(value);
}

function chartAssetUrl(value) {
  if (typeof value === 'object' && value?.data_url) {
    return text(value.data_url);
  }
  const rawPath = text(typeof value === 'string' ? value : value?.file_path || value?.path || value?.src || '');
  if (!rawPath) return '';
  const normalized = rawPath.replace(/\\/g, '/');
  const fileName = normalized.split('/').pop();
  if (!fileName) return '';
  return `/charts/${encodeURIComponent(fileName)}`;
}

function renderReportBody(investigation) {
  const reportText = investigation.reports[state.selectedReport] || investigation.reports.analyst || investigation.reports.master;
  return `
    <div class="report-body">
      <div class="report-summary">
        <div>
          <strong>${escapeHtml(investigation.question)}</strong>
          <div class="subtle">${escapeHtml(investigation.dataset?.name || 'Unknown dataset')}</div>
        </div>
        <div class="report-summary-actions">
          <div class="status-badge ${investigation.confidence.label === 'High' ? 'good' : investigation.confidence.label === 'Moderate' ? 'warn' : 'active'}">Confidence ${escapeHtml(investigation.confidence.label)}</div>
          <button class="link-button" type="button" data-action="open-report">View more</button>
        </div>
      </div>
      <pre>${escapeHtml(reportText)}</pre>
    </div>
  `;
}

async function refreshInvestigations() {
  try {
    const payload = await listInvestigations();
    const summaries = summarizeInvestigations(payload.investigations || []);
    const existingById = new Map(state.investigations.map((item) => [item.id, item]));
    state.investigations = summaries.map((item) => mergeInvestigationRecords(existingById.get(item.id), item));
    if (!state.activeInvestigationId && state.investigations[0]) {
      state.activeInvestigationId = state.investigations[0].id;
    }
  } catch (error) {
    state.error = error;
  }
}

async function refreshBootstrap() {
  try {
    const payload = await getBootstrap();
    state.bootstrap = normalizeBootstrap(payload);
    state.form.datasetPath = state.form.datasetPath || state.bootstrap.defaultDatasetPath;
    state.apiState = 'remote';
    state.error = null;
  } catch (error) {
    state.bootstrap = createFallbackBootstrap();
    state.apiState = 'local';
    state.error = error;
  }
}

async function runQuestion(payload) {
  state.loading = true;
  state.error = null;
  const clientRequestId = `req-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  const pendingInvestigation = createPendingInvestigation(payload, clientRequestId);
  state.investigations = [pendingInvestigation, ...state.investigations.filter((item) => item.id !== pendingInvestigation.id)];
  state.activeInvestigationId = pendingInvestigation.id;
  state.view = 'investigations';
  state.requestedInvestigationId = pendingInvestigation.id;
  state.selectedStage = pendingInvestigation.progress.find((stage) => stage.status === 'current')?.key || 'answer';
  state.selectedReport = 'analyst';
  render();

  const requestPayload = {
    ...payload,
    clientRequestId,
  };

  const syncCurrent = (investigation) => {
    if (!investigation) return;
    setActiveInvestigation(investigation);
    state.bootstrap.recentInvestigations = [
      normalizeInvestigation(investigation),
      ...(state.bootstrap.recentInvestigations || []).filter((item) => item.id !== (investigation.id || investigation.investigation_id)),
    ].slice(0, 8);
    state.error = null;
    render();
  };

  let finished = false;
  const finish = () => {
    if (finished) return;
    finished = true;
    window.clearTimeout(timeoutId);
    state.loading = false;
    render();
  };

  const timeoutId = window.setTimeout(finish, RUN_POLL_TIMEOUT_MS + 1000);

  void runInvestigation(requestPayload)
    .then((response) => {
      const investigation = response.investigation || response;
      syncCurrent(investigation);
    })
    .catch(() => undefined);

  void waitForInvestigationResult(clientRequestId, requestPayload)
    .then((investigation) => {
      if (investigation) {
        syncCurrent(investigation);
      }
      finish();
    })
    .catch(() => undefined);
}

function collectInvestigationForm() {
  const form = document.querySelector('#investigation-form');
  if (!form) return { ...state.form };
  const data = new FormData(form);
  return {
    question: text(data.get('question')) || state.form.question,
    datasetPath: text(data.get('datasetPath')) || state.form.datasetPath,
    mode: state.form.mode,
  };
}

async function handleAction(action, target) {
  if (action === 'nav') {
    setRoute(target.dataset.view || 'home');
    render();
    return;
  }
  if (action === 'go-new') {
    setRoute('new');
    render();
    return;
  }
  if (action === 'mode') {
    state.form.mode = target.dataset.mode || state.form.mode;
    render();
    return;
  }
  if (action === 'fill-question') {
    state.form.question = target.dataset.question || state.form.question;
    setRoute('new');
    render();
    return;
  }
  if (action === 'select-dataset') {
    state.form.datasetPath = target.dataset.path || state.form.datasetPath;
    render();
    return;
  }
  if (action === 'open-investigation') {
    const id = target.dataset.id;
    if (!id) return;
    setRoute('investigations', id);
    state.loading = true;
    render();
    loadInvestigation(id).finally(() => {
      state.loading = false;
      render();
    });
    return;
  }
  if (action === 'stage') {
    state.selectedStage = target.dataset.stage || state.selectedStage;
    render();
    return;
  }
  if (action === 'open-finding') {
    const current = getActiveInvestigation();
    if (!current?.findings?.length) return;
    state.drawer = { kind: 'finding', index: Number(target.dataset.index || 0) };
    render();
    return;
  }
  if (action === 'open-evidence') {
    const current = getActiveInvestigation();
    if (!current?.evidence?.length) return;
    state.drawer = { kind: 'evidence', index: Number(target.dataset.index || 0) };
    render();
    return;
  }
  if (action === 'open-visualization') {
    const current = getActiveInvestigation();
    if (!current?.visualizations?.length) return;
    state.drawer = { kind: 'visualization', index: Number(target.dataset.index || 0) };
    render();
    return;
  }
  if (action === 'open-answer') {
    state.drawer = { kind: 'answer' };
    render();
    return;
  }
  if (action === 'open-report') {
    state.drawer = { kind: 'report' };
    render();
    return;
  }
  if (action === 'open-recommendation') {
    const current = getActiveInvestigation();
    if (!current?.recommendations?.length) return;
    state.drawer = { kind: 'recommendation', index: Number(target.dataset.index || 0) };
    render();
    return;
  }
  if (action === 'open-journey') {
    state.drawer = { kind: 'journey' };
    render();
    return;
  }
  if (action === 'open-snapshots') {
    state.drawer = { kind: 'snapshots' };
    render();
    return;
  }
  if (action === 'open-snapshot') {
    state.drawer = { kind: 'snapshot', index: Number(target.dataset.index || 0) };
    render();
    return;
  }
  if (action === 'close-drawer' || action === 'toggle-evidence-summary') {
    state.drawer = null;
    render();
    return;
  }
  if (action === 'select-report') {
    state.selectedReport = target.dataset.report || state.selectedReport;
    render();
    return;
  }
  if (action === 'share-investigation') {
    const current = getActiveInvestigation();
    if (!current) return;
    try {
      await copyTextToClipboard(getInvestigationUrl(current.id));
    } catch (error) {
      state.error = error;
    }
    render();
    return;
  }
  if (action === 'download-investigation') {
    const current = getActiveInvestigation();
    if (!current) return;
    const payload = current.raw || current;
    downloadTextFile(`investigation-${current.id}.json`, JSON.stringify(payload, null, 2));
    return;
  }
  if (action === 'cancel-run') {
    const current = getActiveInvestigation();
    if (!current?.id) return;
    state.loading = true;
    render();
    try {
      const response = await cancelInvestigation(current.id);
      const investigation = response.investigation || response;
      await loadInvestigation(investigation.id);
      state.error = null;
    } catch (error) {
      state.error = error;
    } finally {
      state.loading = false;
      render();
    }
    return;
  }
  if (action === 'run-command') {
    const input = document.querySelector('#commandDraft');
    const command = text(input?.value || state.commandDraft);
    if (!command) return;
    state.commandDraft = command;
    const active = getActiveInvestigation();
    runQuestion({
      question: command,
      datasetPath: active?.dataset?.path || state.form.datasetPath,
      mode: 'collaborative',
      initialTasks: active?.tasks || [],
      collaborativeResponses: [command, 'continue', 'continue', 'continue'],
    });
    return;
  }
  if (action === 'guided-continue' || action === 'guided-modify' || action === 'guided-stop') {
    const current = getActiveInvestigation();
    const currentStatus = text(current?.status).toLowerCase();
    if (!currentStatus.includes('await')) {
      return;
    }
    const input = document.querySelector('#guideDraft');
    const detail = text(input?.value || state.guideDraft || 'refine the current plan');
    state.guideDraft = detail;
    const responses =
      action === 'guided-stop'
        ? ['continue', 'continue', 'continue', 'cancel']
        : action === 'guided-modify'
          ? ['modify', detail, 'continue', 'continue']
          : ['continue', 'continue', 'continue', 'continue'];
    runQuestion({
      question: getActiveInvestigation()?.question || state.form.question,
      datasetPath: getActiveInvestigation()?.dataset?.path || state.form.datasetPath,
      mode: 'guided',
      guidedResponses: responses,
    });
    return;
  }
}

app.addEventListener('click', (event) => {
  const target = event.target.closest('[data-action]');
  if (!target) return;
  event.preventDefault();
  handleAction(target.dataset.action, target);
});

app.addEventListener('submit', (event) => {
  if (!(event.target instanceof HTMLFormElement)) return;
  if (event.target.id === 'investigation-form') {
    event.preventDefault();
    const values = collectInvestigationForm();
    state.form = values;
    runQuestion(values);
  }
});

app.addEventListener('input', (event) => {
  const target = event.target;
  if (!(target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement)) return;
  if (target.name === 'guideDraft') state.guideDraft = target.value;
  if (target.name === 'commandDraft') state.commandDraft = target.value;
  if (target.name === 'question') state.form.question = target.value;
});

window.addEventListener('hashchange', () => {
  parseRoute();
  render();
});

async function boot() {
  parseRoute();
  render();
  await Promise.all([refreshBootstrap(), refreshInvestigations()]);
  if (state.requestedInvestigationId) {
    await loadInvestigation(state.requestedInvestigationId);
  }
  if (!state.activeInvestigationId && state.investigations[0]) {
    state.activeInvestigationId = state.investigations[0].id;
  }
  state.loading = false;
  render();
}

function render() {
  renderApp();
}

boot();
