import {
  CASE_BANK_VERSION,
  PROTOCOL_VERSION,
  QUALITATIVE,
  QUESTIONS_VERSION,
  RUBRIC,
  type LikertQuestionId,
  type QualitativeQuestionId,
} from "../data/rubric";
import {
  scrubReportSummary,
  visibleEvidence,
  visibleProfile,
  visibleXai,
} from "./stimulusView";
import type {
  CaseRatingState,
  CaseStimulus,
  QualitativeAnswers,
  RatingRecord,
  Scores,
  StimulusSnapshot,
} from "../types";

const STORAGE_KEY = "expert_eval_v2_store";

export interface SessionStore {
  session_id: string;
  expert_id: string;
  case_index: number;
  ratings: Record<string, CaseRatingState>;
  log: RatingRecord[];
}

function uid(): string {
  return crypto.randomUUID?.() ?? `s_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}

export function createEmptyScores(): Scores {
  return {};
}

export function createEmptyQualitative(): QualitativeAnswers {
  return {};
}

export function allScoresPresent(scores: Scores): boolean {
  return RUBRIC.every((q) => typeof scores[q.id] === "number");
}

export function loadStore(): SessionStore | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    return JSON.parse(raw) as SessionStore;
  } catch {
    return null;
  }
}

export function saveStore(store: SessionStore): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(store));
}

export function clearStore(): void {
  localStorage.removeItem(STORAGE_KEY);
}

export function startSession(expertId: string): SessionStore {
  const existing = loadStore();
  if (existing && existing.expert_id === expertId) {
    return existing;
  }
  const store: SessionStore = {
    session_id: uid(),
    expert_id: expertId.trim().toUpperCase(),
    case_index: 0,
    ratings: {},
    log: [],
  };
  saveStore(store);
  return store;
}

export function snapshotCase(c: CaseStimulus): StimulusSnapshot {
  return {
    case_bank_version: CASE_BANK_VERSION,
    case_id: c.id,
    source_case_id: c.source_case_id,
    stratum: c.stratum,
    test_index: c.test_index,
    probability: c.screening.probability,
    threshold: c.screening.threshold,
    label: c.screening.label,
    student_profile: visibleProfile(c.student_profile ?? []),
    xai: visibleXai(c.xai),
    evidence: visibleEvidence(c.evidence),
    report_summary: scrubReportSummary(c.report_summary),
    report_source: c.report_source,
  };
}

function fullScores(scores: Scores): Record<LikertQuestionId, number | null> {
  const out = {} as Record<LikertQuestionId, number | null>;
  for (const q of RUBRIC) {
    out[q.id] = typeof scores[q.id] === "number" ? (scores[q.id] as number) : null;
  }
  return out;
}

function fullQualitative(
  qualitative: QualitativeAnswers,
): Record<QualitativeQuestionId, string> {
  const out = {} as Record<QualitativeQuestionId, string>;
  for (const q of QUALITATIVE) {
    out[q.id] = (qualitative[q.id] ?? "").trim();
  }
  return out;
}

export function appendLog(
  store: SessionStore,
  caseItem: CaseStimulus,
  caseIndex: number,
  rating: CaseRatingState,
  event: RatingRecord["event"],
): SessionStore {
  const record: RatingRecord = {
    protocol_version: PROTOCOL_VERSION,
    questions_version: QUESTIONS_VERSION,
    case_bank_version: CASE_BANK_VERSION,
    session_id: store.session_id,
    expert_id: store.expert_id,
    case_id: caseItem.id,
    case_index: caseIndex,
    timestamp_iso: new Date().toISOString(),
    event,
    scores: fullScores(rating.scores),
    qualitative: fullQualitative(rating.qualitative ?? {}),
    stimulus_snapshot: snapshotCase(caseItem),
  };
  const next: SessionStore = {
    ...store,
    log: [...store.log, record],
    ratings: {
      ...store.ratings,
      [caseItem.id]: {
        ...rating,
        last_saved_iso: record.timestamp_iso,
        completed:
          event === "complete_case" || event === "finish_study"
            ? true
            : rating.completed,
      },
    },
  };
  saveStore(next);
  return next;
}

function downloadBlob(filename: string, content: string, mime: string): void {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export function exportJsonl(store: SessionStore): void {
  const lines = store.log.map((r) => JSON.stringify(r)).join("\n");
  downloadBlob(
    `expert_eval_${store.expert_id}_${store.session_id.slice(0, 8)}.jsonl`,
    lines + (lines ? "\n" : ""),
    "application/x-ndjson",
  );
}

function csvEscape(v: string | number | null | undefined): string {
  const s = v == null ? "" : String(v);
  if (/[",\n]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
}

/** Latest complete_case/finish_study (else latest save) per case for wide CSV. */
export function latestPerCase(store: SessionStore): RatingRecord[] {
  const byCase = new Map<string, RatingRecord>();
  for (const r of store.log) {
    const prev = byCase.get(r.case_id);
    if (!prev) {
      byCase.set(r.case_id, r);
      continue;
    }
    const prefer =
      (r.event === "complete_case" || r.event === "finish_study") &&
      prev.event === "save";
    if (prefer || r.timestamp_iso >= prev.timestamp_iso) {
      byCase.set(r.case_id, r);
    }
  }
  return [...byCase.values()].sort((a, b) => a.case_index - b.case_index);
}

export function exportCsv(store: SessionStore): void {
  const rows = latestPerCase(store);
  const qIds = RUBRIC.map((q) => q.id);
  const qualIds = QUALITATIVE.map((q) => q.id);
  const headers = [
    "protocol_version",
    "questions_version",
    "case_bank_version",
    "session_id",
    "expert_id",
    "case_id",
    "source_case_id",
    "case_index",
    "stratum",
    "test_index",
    "probability",
    "threshold",
    "label",
    "timestamp_iso",
    "event",
    ...qIds,
    ...qualIds,
  ];
  const lines = [headers.join(",")];
  for (const r of rows) {
    const vals = [
      r.protocol_version,
      r.questions_version,
      r.case_bank_version,
      r.session_id,
      r.expert_id,
      r.case_id,
      r.stimulus_snapshot.source_case_id,
      r.case_index,
      r.stimulus_snapshot.stratum,
      r.stimulus_snapshot.test_index ?? "",
      r.stimulus_snapshot.probability,
      r.stimulus_snapshot.threshold,
      r.stimulus_snapshot.label,
      r.timestamp_iso,
      r.event,
      ...qIds.map((id) => r.scores[id]),
      ...qualIds.map((id) => r.qualitative[id]),
    ];
    lines.push(vals.map(csvEscape).join(","));
  }
  downloadBlob(
    `expert_eval_${store.expert_id}_${store.session_id.slice(0, 8)}.csv`,
    lines.join("\n") + "\n",
    "text/csv",
  );
}

export function completedCount(store: SessionStore, caseIds: string[]): number {
  return caseIds.filter((id) => store.ratings[id]?.completed).length;
}
