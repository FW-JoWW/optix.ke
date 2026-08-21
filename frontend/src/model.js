const FALLBACK_DATASETS = [
  {
    name: 'olist_merged_dataset.csv',
    path: 'backend/data/olist_merged_dataset.csv',
    source_type: 'csv',
    size_label: '42.6 MB',
    row_count: null,
    column_count: null,
    last_modified: null,
  },
  {
    name: 'Car Dataset 1945-2020.csv',
    path: 'backend/data/Car Dataset 1945-2020.csv',
    source_type: 'csv',
    size_label: '23.7 MB',
    row_count: null,
    column_count: null,
    last_modified: null,
  },
  {
    name: 'data_sets.xlsx',
    path: 'backend/data/data_sets.xlsx',
    source_type: 'excel',
    size_label: '774 KB',
    row_count: null,
    column_count: null,
    last_modified: null,
  },
];

const STAGE_ORDER = [
  { key: 'question', label: 'Question' },
  { key: 'data', label: 'Data' },
  { key: 'quality', label: 'Quality' },
  { key: 'plan', label: 'Plan' },
  { key: 'investigation', label: 'Investigation' },
  { key: 'finding', label: 'Finding' },
  { key: 'answer', label: 'Answer' },
];

const EMPTY_REPORT = 'No report available yet.';

export function formatStageLabel(stage) {
  const labels = {
    question: 'Question',
    data: 'Data',
    quality: 'Quality',
    plan: 'Plan',
    investigation: 'Investigation',
    finding: 'Finding',
    answer: 'Answer',
    business_understanding: 'Business understanding',
    data_preparation: 'Data preparation',
    analysis_strategy: 'Analysis strategy',
    result_review: 'Result review',
  };
  return labels[stage] || capitalize(stage);
}

export function toArray(value) {
  if (!value) return [];
  return Array.isArray(value) ? value : [value];
}

function text(value, fallback = '') {
  if (value === null || value === undefined) return fallback;
  if (typeof value === 'string') {
    const result = value.trim();
    return result || fallback;
  }
  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value);
  }
  if (Array.isArray(value)) {
    const result = value
      .map((item) => text(item, ''))
      .filter(Boolean)
      .join(', ')
      .trim();
    return result || fallback;
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
    ];
    for (const key of preferredKeys) {
      const result = text(value[key], '');
      if (result) return result;
    }
    const entries = Object.entries(value);
    if (entries.length === 1) {
      const [, onlyValue] = entries[0];
      const result = text(onlyValue, '');
      if (result) return result;
    }
    try {
      const result = JSON.stringify(value);
      if (result && result !== '{}') return result;
    } catch (error) {
      void error;
    }
    return fallback;
  }
  const result = String(value).trim();
  return result || fallback;
}

function capitalize(value) {
  const result = text(value);
  if (!result) return '';
  return result.charAt(0).toUpperCase() + result.slice(1);
}

function confidenceLabel(value) {
  if (value === null || value === undefined || value === '') return 'Unknown';
  if (typeof value === 'number') {
    if (value >= 75) return 'High';
    if (value >= 45) return 'Moderate';
    return 'Low';
  }
  const normalized = text(value).toLowerCase();
  if (normalized.includes('high') || normalized.includes('strong')) return 'High';
  if (normalized.includes('moderate') || normalized.includes('medium')) return 'Moderate';
  if (normalized.includes('low') || normalized.includes('weak')) return 'Low';
  return capitalize(normalized);
}

function formatCount(value) {
  if (value === null || value === undefined || value === '') return 'Unknown';
  return Number(value).toLocaleString();
}

function asCompactReport(report) {
  const body = text(report);
  if (!body) return EMPTY_REPORT;
  return body;
}

function extractDataset(raw) {
  const dataset = raw?.dataset || {};
  const profile = raw?.dataset_profile || {};
  return {
    name: dataset.name || text(raw?.dataset_path).split(/[\\/]/).pop() || 'Dataset',
    path: raw?.dataset_path || dataset.path || '',
    sourceType: dataset.source_type || profile.source_type || 'csv',
    rowCount: profile.row_count ?? dataset.row_count ?? null,
    columnCount: profile.column_count ?? dataset.column_count ?? null,
    sizeLabel: dataset.size_label || null,
    lastModified: dataset.last_modified || null,
    columns: dataset.columns || profile.columns || [],
  };
}

function extractConfidence(raw) {
  const evidence = raw?.analysis_evidence || {};
  const answer = evidence.answer_synthesis || raw?.answer_synthesis || {};
  const judgment = evidence.judgment_summary || raw?.judgment_summary || {};
  const diagnostics = toArray(answer.confidence_diagnostics || answer.confidence?.diagnostics || []);
  const overall = answer.confidence?.overall || {};
  const label = confidenceLabel(overall.label || judgment.global_confidence || answer.confidence || overall.score);
  return {
    label,
    score: overall.score ?? judgment.global_confidence ?? null,
    reason: overall.reason || judgment.summary || '',
    evidence: answer.confidence?.evidence || null,
    interpretation: answer.confidence?.interpretation || null,
    business: answer.confidence?.business || null,
    recommendation: answer.confidence?.recommendation || null,
    diagnostics: diagnostics.slice(0, 4),
  };
}

function extractAnswer(raw) {
  const evidence = raw?.analysis_evidence || {};
  const answer = evidence.answer_synthesis || raw?.answer_synthesis || {};
  const direct = text(answer.direct_answer || answer.best_available_answer || raw?.answer || raw?.final_report || '');
  const recommendation = toArray(answer.recommended_next_investigation).slice(0, 3);
  return {
    direct: direct || 'The investigation has not reached a direct answer yet.',
    business: text(answer.business_interpretation || answer.semantic_reasoning || ''),
    supportingEvidence: toArray(answer.supporting_evidence_summary).slice(0, 4),
    observedFacts: toArray(answer.observed_facts).slice(0, 4),
    analyticalInterpretation: toArray(answer.analytical_interpretation).slice(0, 4),
    assumptions: toArray(answer.key_assumptions).slice(0, 4),
    uncertainty: toArray(answer.remaining_uncertainty).slice(0, 4),
    nextInvestigation: recommendation,
    position: answer.answer_position || 'unknown',
  };
}

function storyTitle(story) {
  return (
    text(story?.headline) ||
    text(story?.insight) ||
    text(story?.summary) ||
    text(story?.finding) ||
    'Finding'
  );
}

function storyEvidenceText(story) {
  return (
    text(story?.business_implication) ||
    text(story?.plain_english) ||
    text(story?.detail) ||
    text(story?.explanation) ||
    ''
  );
}

function extractFindings(raw) {
  const evidence = raw?.analysis_evidence || {};
  const stories = toArray(evidence.top_stories || raw?.top_stories);
  const decisions = toArray(evidence.decision_recommendations || raw?.decision_recommendations);
  const findings = stories.map((story, index) => ({
    id: story.story_signature || story.signature || `finding-${index + 1}`,
    title: storyTitle(story),
    summary: storyEvidenceText(story) || text(story.insight) || 'A supported analytical observation.',
    confidence: confidenceLabel(story.confidence ?? story.score),
    score: story.score ?? null,
    relationshipType: story.relationship_type || story.type || 'supporting',
    whyItMatters:
      text(story.business_implication) ||
      text(story.recommendation) ||
      text(story.interpretation) ||
      'This finding helps establish the answer.',
    supportingEvidence: toArray(story.supporting_evidence || story.evidence || []).slice(0, 4),
    limitations: toArray(story.limitations || story.warnings || []).slice(0, 3),
    method: story.method_used || story.method || story.analysis || story.tool || 'Analytical evidence',
  }));

  if (!findings.length && decisions.length) {
    return decisions.slice(0, 4).map((decision, index) => ({
      id: decision.story_signature || `decision-${index + 1}`,
      title: text(decision.recommended_action) || 'Recommendation',
      summary: text(decision.decision_summary) || text(decision.impact_assessment?.impact_summary) || 'A decision recommendation was generated.',
      confidence: confidenceLabel(decision.confidence_in_action),
      score: decision.priority?.priority_score ?? null,
      relationshipType: decision.action_type || 'recommendation',
      whyItMatters: text(decision.impact_assessment?.impact_summary) || 'It matters because it changes what the user can do next.',
      supportingEvidence: [],
      limitations: toArray(decision.recommendation_restrictions).slice(0, 3),
      method: 'Decision orchestration',
    }));
  }

  return findings;
}

function normalizeDataQuality(raw) {
  const evidence = raw?.analysis_evidence || {};
  const profile = raw?.dataset_profile || {};
  const validation = raw?.data_validation || evidence.data_validation || evidence.cleaning_validation || {};
  const issues = toArray(raw?.data_quality_issues || evidence.data_quality_issues);
  return {
    rows: profile.row_count ?? validation.row_count ?? null,
    columns: profile.column_count ?? validation.column_count ?? null,
    issues: issues.length || validation.anomalies?.length || validation.warnings?.length || 0,
    missingValues: profile.missing_values || validation.missing_values || {},
    duplicates: validation.duplicates ?? null,
    invalidValues: validation.invalid_values ?? null,
    anomalies: toArray(validation.anomalies),
    warnings: toArray(validation.warnings),
    raw: validation,
  };
}

function extractReports(raw) {
  const evidence = raw?.analysis_evidence || {};
  const reportPackage = evidence.report_package || raw?.report_package || {};
  const finalReport = text(raw?.final_report || evidence.final_report || '');
  const fallbackReport = finalReport || EMPTY_REPORT;
  return {
    analyst: text(raw?.analyst_report || reportPackage.analyst_report || fallbackReport),
    business: text(raw?.business_report || reportPackage.business_report || fallbackReport),
    executive: text(raw?.executive_report || reportPackage.executive_report || fallbackReport),
    master: text(raw?.master_report || reportPackage.master_report || fallbackReport),
    answerSynthesis: text(evidence.answer_synthesis_report || raw?.answer_synthesis_report || fallbackReport),
    decision: text(evidence.investigation_decision_report || raw?.investigation_decision_report || fallbackReport),
  };
}

function extractJourney(raw) {
  const evidence = raw?.analysis_evidence || {};
  const guided = toArray(raw?.guided_decision_log || evidence.guided_decision_log);
  const collaborative = toArray(raw?.collaborative_decision_log || evidence.collaborative_decision_log);
  const log = guided.length ? guided : collaborative;
  if (!log.length) {
    return STAGE_ORDER.map((stage, index) => ({
      id: stage.key,
      label: stage.label,
      timestamp: null,
      detail: index === 0 ? 'Question received' : 'Completed or available in the current evidence trail.',
      status: index < 4 ? 'complete' : index === 4 ? 'current' : 'pending',
    }));
  }
  return log.slice(-10).map((entry, index) => ({
    id: `${entry.stage || 'entry'}-${index}`,
    label: capitalize(entry.stage || 'checkpoint'),
    timestamp: entry.timestamp || null,
    detail:
      text(entry.user_decision) ||
      text(entry.ai_recommendation) ||
      text(entry.reason_for_modification) ||
      'Checkpoint recorded.',
    status: text(entry.user_decision).toLowerCase() === 'cancel' ? 'warning' : 'complete',
  }));
}

function extractSnapshots(raw) {
  const evidence = raw?.analysis_evidence || {};
  const snapshots = [];
  const guidedSnapshots = raw?.guided_version_snapshots || evidence.guided_version_snapshots || {};
  const guidedSummaries = raw?.guided_checkpoint_summaries || evidence.guided_checkpoint_summaries || {};

  function snapshotDetails(snapshot) {
    const rows = [
      ['Version', snapshot?.version ?? 'N/A'],
      ['Stage', snapshot?.stage ?? 'Unknown'],
      ['Reason', snapshot?.reason || snapshot?.note || snapshot?.description || ''],
      ['Summary', snapshot?.summary || ''],
    ];
    const selected = [];
    rows.forEach(([label, value]) => {
      const item = text(value);
      if (!item) return;
      selected.push({ label, value: item });
    });
    const extra = snapshot?.changes || snapshot?.highlights || snapshot?.updates || snapshot?.details || null;
    const extraList = toArray(extra)
      .flat()
      .map((entry) => text(typeof entry === 'string' ? entry : entry?.label || entry?.value || entry?.summary || JSON.stringify(entry)))
      .filter(Boolean)
      .slice(0, 4);
    return {
      fields: selected.slice(0, 4),
      extra: extraList,
    };
  }

  Object.entries(guidedSnapshots || {}).forEach(([stage, snapshot]) => {
    if (!snapshot) return;
    const meta = snapshotDetails(snapshot);
    snapshots.push({
      id: `${stage}-${snapshot.version || 'snapshot'}`,
      stage,
      version: snapshot.version ?? null,
      label: `${formatStageLabel(stage)} v${snapshot.version ?? '1'}`,
      summary: text(snapshot.summary || snapshot.note || snapshot.description || snapshot.reason || 'Checkpoint snapshot.'),
      detail: snapshot,
      kind: 'snapshot',
      kindLabel: 'Checkpoint snapshot',
      fields: meta.fields,
      extra: meta.extra,
    });
  });

  Object.entries(guidedSummaries || {}).forEach(([stage, summary]) => {
    if (!summary) return;
    const firstLine = Object.values(summary || {})
      .flat()
      .find((value) => text(value));
    const meta = snapshotDetails(summary);
    snapshots.push({
      id: `${stage}-summary`,
      stage,
      version: null,
      label: `${formatStageLabel(stage)} summary`,
      summary: text(firstLine || 'Checkpoint summary.'),
      detail: summary,
      kind: 'summary',
      kindLabel: 'Checkpoint summary',
      fields: meta.fields,
      extra: meta.extra,
    });
  });

  return snapshots.slice(0, 12);
}

function extractEvidence(raw) {
  const evidence = raw?.analysis_evidence || {};
  const items = [];
  const toolResults = evidence.tool_results || raw?.tool_results || {};
  Object.entries(toolResults || {}).forEach(([key, value]) => {
    if (!value) return;
    const tool = value.tool || value.type || key;
    const summary =
      text(value.summary) ||
      text(value.insight) ||
      text(value.message) ||
      text(value.direct_answer) ||
      text(value.business_interpretation) ||
      text(value.recommended_next_step) ||
      'Structured analytical output.';
    items.push({
      id: `tool-${key}`,
      kind: 'tool',
      label: capitalize(tool.replace(/_/g, ' ')),
      summary,
      detail: JSON.stringify(value, null, 2),
      confidence: confidenceLabel(value.confidence?.label || value.confidence?.score || value.confidence || ''),
      scope: text(value.evidence_scope || value.provenance?.scope || ''),
    });
  });

  const profile = raw?.dataset_profile || {};
  if (profile.row_count || profile.column_count) {
    items.push({
      id: 'dataset-profile',
      kind: 'profile',
      label: 'Dataset profile',
      summary: `${formatCount(profile.row_count)} rows | ${formatCount(profile.column_count)} columns`,
      detail: JSON.stringify(profile, null, 2),
      confidence: 'High',
      scope: text(profile.evidence_scope || profile.profiling_mode || profile.provenance?.scope || ''),
    });
  }

  if (raw?.data_validation || evidence.cleaning_validation) {
    const validation = raw.data_validation || evidence.cleaning_validation || {};
    items.push({
      id: 'data-validation',
      kind: 'quality',
      label: 'Data quality validation',
      summary: validation.anomalies?.length || validation.warnings?.length ? 'Issues were identified during validation.' : 'Validation completed without major issues.',
      detail: JSON.stringify(validation, null, 2),
      confidence: validation.anomalies?.length ? 'Moderate' : 'High',
      scope: text(validation.evidence_scope || validation.provenance?.scope || ''),
    });
  }

  toArray(evidence.visualizations || raw?.visualizations).forEach((visual, index) => {
    if (!visual) return;
    items.push({
      id: `visual-${index}`,
      kind: 'visual',
      label: text(visual.title || visual.type || `Visualization ${index + 1}`),
      summary: text(visual.caption || visual.file_path || visual.path || 'Generated chart'),
      detail: JSON.stringify(visual, null, 2),
      confidence: 'High',
      scope: text(visual.evidence_scope || visual.provenance?.scope || ''),
    });
  });

  toArray(evidence.guided_checkpoint_summaries || raw?.guided_checkpoint_summaries).forEach((summary, index) => {
    if (!summary) return;
    const firstValue = Object.values(summary || {})[0];
    items.push({
      id: `checkpoint-${index}`,
      kind: 'checkpoint',
      label: `Checkpoint ${index + 1}`,
      summary: text(Array.isArray(firstValue) ? firstValue[0] : firstValue) || 'Guided checkpoint summary.',
      detail: JSON.stringify(summary, null, 2),
      confidence: 'High',
    });
  });

  const session = evidence.collaborative_session || raw?.collaborative_session || {};
  const store = session.evidence_store || evidence.collaborative_evidence_store || raw?.collaborative_evidence_store || {};
  Object.entries(store || {}).forEach(([key, value]) => {
    if (!value) return;
    items.push({
      id: `evidence-${key}`,
      kind: 'collaborative',
      label: text(value.evidence_type || value.task_source || `Evidence ${key}`),
      summary: text(value.statement || value.summary || value.insight || 'Collaborative evidence item.'),
      detail: JSON.stringify(value, null, 2),
      confidence: confidenceLabel(value.confidence || ''),
      scope: text(value.evidence_scope || value.provenance?.scope || ''),
    });
  });

  return items.slice(0, 18);
}

function buildProgress(raw) {
  const evidence = raw?.analysis_evidence || {};
  const workflowStatus = raw?.workflow_status || evidence.workflow_status || {};
  const trace = toArray(evidence.execution_trace || raw?.execution_trace);
  const hasData = Boolean(raw?.dataset_profile || raw?.dataframe || raw?.dataset || raw?.dataset_path);
  const hasQuality = Boolean(raw?.data_validation || evidence.cleaning_validation || evidence.data_quality_issues);
  const hasPlan = Boolean((raw?.analysis_plan || evidence.analysis_plan || []).length);
  const hasInvestigation = Boolean((toArray(evidence.tool_results || raw?.tool_results).length || toArray(evidence.top_stories || raw?.top_stories).length));
  const hasFinding = Boolean(toArray(evidence.top_stories || raw?.top_stories).length);
  const hasAnswer = Boolean(evidence.answer_synthesis?.direct_answer || raw?.final_report || raw?.answer);
  const phase = text(workflowStatus.phase || '').toLowerCase();
  const phaseToStage = {
    question: 'question',
    loading: 'data',
    data: 'data',
    data_quality: 'quality',
    planning: 'plan',
    plan: 'plan',
    investigation: 'investigation',
    reasoning: 'finding',
    reporting: 'answer',
    synthesizing: 'answer',
    awaiting_user: 'answer',
    completed: 'answer',
  };
  const currentStageKey = phaseToStage[phase] || null;
  const flags = {
    question: true,
    data: hasData,
    quality: hasQuality,
    plan: hasPlan,
    investigation: hasInvestigation,
    finding: hasFinding,
    answer: hasAnswer,
  };
  let activeSeen = false;
  return STAGE_ORDER.map((stage) => {
    const done = flags[stage.key] || (currentStageKey ? STAGE_ORDER.findIndex((item) => item.key === stage.key) < STAGE_ORDER.findIndex((item) => item.key === currentStageKey) : false);
    let status = 'pending';
    if (phase === 'failed' && stage.key === 'answer') {
      status = 'current';
    } else if (done) {
      status = 'complete';
    } else if (currentStageKey === stage.key || (!currentStageKey && !activeSeen)) {
      status = 'current';
      activeSeen = true;
    }
    const traceMatch = trace.find((entry) => {
      const entryPhase = text(entry?.phase || '').toLowerCase();
      const entryOperation = text(entry?.operation || '').toLowerCase();
      return entryPhase === stage.key || entryOperation.includes(stage.key);
    });
    return {
      ...stage,
      status,
      done,
      detail: text(workflowStatus.current_operation || workflowStatus.message || traceMatch?.message) ||
        (stage.key === 'question'
          ? 'The business question was captured.'
          : stage.key === 'data'
            ? 'Dataset selected and loaded.'
            : stage.key === 'quality'
              ? 'Data quality was checked.'
              : stage.key === 'plan'
                ? 'An investigation plan was assembled.'
                : stage.key === 'investigation'
                  ? 'Analytical evidence is being gathered.'
                  : stage.key === 'finding'
                    ? 'Findings are being synthesized.'
                    : 'A direct answer is available.'),
    };
  });
}

function extractRecommendations(raw) {
  const evidence = raw?.analysis_evidence || {};
  const answer = evidence.answer_synthesis || {};
  const judgment = evidence.judgment_summary || {};
  const decision = evidence.decision_recommended_first || raw?.decision_recommended_first || {};
  const items = [];
  const primary = text(decision.recommended_action || judgment.recommended_first_action || answer.recommended_next_investigation?.[0] || '');
  if (primary) {
    items.push({
      label: 'Recommended action',
      value: primary,
    });
  }
  const next = toArray(answer.recommended_next_investigation || judgment.remaining_uncertainties || []).slice(0, 3);
  next.forEach((item) => {
    items.push({
      label: 'Next step',
      value: text(item),
    });
  });
  return items;
}

function extractHypotheses(raw) {
  const evidence = raw?.analysis_evidence || {};
  const session = evidence.collaborative_session || raw?.collaborative_session || {};
  return toArray(session.hypotheses || raw?.collaborative_hypotheses).map((hypothesis, index) => ({
    id: hypothesis.hypothesis_id || `hypothesis-${index + 1}`,
    statement: text(hypothesis.hypothesis || hypothesis.statement || 'Hypothesis'),
    status: text(hypothesis.status || 'open'),
    confidence: confidenceLabel(hypothesis.confidence || ''),
    evidence: toArray(hypothesis.supporting_evidence || []),
    notes: text(hypothesis.notes || ''),
  }));
}

function extractTasks(raw) {
  const evidence = raw?.analysis_evidence || {};
  const session = evidence.collaborative_session || raw?.collaborative_session || {};
  return toArray(session.tasks || raw?.collaborative_tasks).map((task, index) => ({
    id: task.task_id || `task-${index + 1}`,
    title: text(task.title || task.request || `Task ${index + 1}`),
    status: text(task.status || 'queued'),
    version: task.version || 1,
    summary: text(task.result_summary || task.description || task.request || ''),
  }));
}

function extractMode(raw) {
  const mode = text(raw?.mode || 'autonomous').toLowerCase();
  return {
    id: mode,
    label: capitalize(mode),
  };
}

export function normalizeBootstrap(payload) {
  return {
    app: payload?.app || {
      name: 'Data Analyst Agent',
      tagline: 'Investigation-centered analytics workspace',
    },
    datasets: toArray(payload?.datasets).length ? payload.datasets : FALLBACK_DATASETS,
    modes: toArray(payload?.modes).length ? payload.modes : [
      { id: 'autonomous', label: 'Autonomous' },
      { id: 'guided', label: 'Guided' },
      { id: 'collaborative', label: 'Collaborative' },
    ],
    recentInvestigations: toArray(payload?.recentInvestigations),
    suggestedQuestions: toArray(payload?.suggestedQuestions).length
      ? payload.suggestedQuestions
      : [
          'Why did sales decline last quarter?',
          'Which region has the strongest growth?',
          'What is driving delivery delays?',
        ],
    defaultDatasetPath: payload?.defaultDatasetPath || FALLBACK_DATASETS[0].path,
    serverTime: payload?.serverTime || null,
  };
}

export function createFallbackBootstrap() {
  return normalizeBootstrap({
    datasets: FALLBACK_DATASETS,
    modes: [
      { id: 'autonomous', label: 'Autonomous', description: 'Agent investigates independently.' },
      { id: 'guided', label: 'Guided', description: 'Agent pauses for checkpoints.' },
      { id: 'collaborative', label: 'Collaborative', description: 'Human and agent work together.' },
    ],
    recentInvestigations: [],
    suggestedQuestions: [
      'Why did sales decline last quarter?',
      'Which region has the strongest growth?',
      'What is driving delivery delays?',
    ],
    defaultDatasetPath: FALLBACK_DATASETS[0].path,
  });
}

export function normalizeInvestigation(raw) {
  const evidence = raw?.analysis_evidence || {};
  const id = text(raw?.id || raw?.investigation_id || raw?.session_id || `inv-${Date.now()}`);
  const dataset = extractDataset(raw);
  const answer = extractAnswer(raw);
  const confidence = extractConfidence(raw);
  const findings = extractFindings(raw);
  const dataQuality = normalizeDataQuality(raw);
  const reports = extractReports(raw);
  const progress = buildProgress(raw);
  const journey = extractJourney(raw);
  const recommendations = extractRecommendations(raw);
  const tasks = extractTasks(raw);
  const hypotheses = extractHypotheses(raw);

  return {
    id,
    question: text(raw?.question || raw?.business_question || 'Untitled investigation'),
    mode: extractMode(raw),
    status: text(raw?.status || (raw?.final_report ? 'completed' : raw?.awaiting_user ? 'awaiting review' : 'running')),
    dataset,
    createdAt: raw?.created_at || raw?.timestamp || null,
    updatedAt: raw?.updated_at || null,
    progress,
    dataQuality,
    answer,
    findings,
    evidence: extractEvidence(raw),
    confidence,
    recommendations,
    journey,
    snapshots: extractSnapshots(raw),
    tasks,
    hypotheses,
    reports,
    analysisPlan: toArray(raw?.analysis_plan || evidence.analysis_plan),
    computationPlan: raw?.analysis_evidence?.computation_plan || raw?.computation_plan || null,
    workflowStatus: raw?.workflow_status || evidence.workflow_status || null,
    selectedColumns: toArray(raw?.selected_columns)
      .map((item) => text(item))
      .filter(Boolean),
    visualizations: toArray(evidence.visualizations || raw?.visualizations),
    executionTrace: toArray(evidence.execution_trace || raw?.execution_trace),
    evidenceProvenance: evidence.evidence_provenance || raw?.evidence_provenance || {},
    raw,
  };
}

export function summarizeInvestigations(investigations) {
  return toArray(investigations).map((item) => normalizeInvestigation(item));
}

export function emptyInvestigation() {
  return {
    id: 'draft',
    question: 'Start by asking a business question',
    mode: { id: 'autonomous', label: 'Autonomous' },
    status: 'draft',
    dataset: FALLBACK_DATASETS[0],
    progress: STAGE_ORDER.map((stage, index) => ({
      ...stage,
      status: index === 0 ? 'current' : 'pending',
      done: false,
      detail: '',
    })),
    dataQuality: {
      rows: null,
      columns: null,
      issues: 0,
      missingValues: {},
      anomalies: [],
      warnings: [],
    },
    answer: {
      direct: 'No answer yet.',
      business: '',
      supportingEvidence: [],
      observedFacts: [],
      analyticalInterpretation: [],
      assumptions: [],
      uncertainty: [],
      nextInvestigation: [],
      position: 'unknown',
    },
    findings: [],
    evidence: [],
    confidence: {
      label: 'Unknown',
      score: null,
      reason: '',
      diagnostics: [],
    },
    recommendations: [],
    journey: [],
    snapshots: [],
    tasks: [],
    hypotheses: [],
    reports: {
      analyst: EMPTY_REPORT,
      business: EMPTY_REPORT,
      executive: EMPTY_REPORT,
      master: EMPTY_REPORT,
      answerSynthesis: EMPTY_REPORT,
      decision: EMPTY_REPORT,
    },
    analysisPlan: [],
    selectedColumns: [],
    visualizations: [],
    executionTrace: [],
  };
}
