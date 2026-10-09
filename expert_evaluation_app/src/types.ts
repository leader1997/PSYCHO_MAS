import type { LikertQuestionId, QualitativeQuestionId } from "./data/rubric";

export interface ProfileFeature {
  feature: string;
  display_name: string;
  /** Original HMS feature value (null if missing). */
  value?: number | null;
  /** Display string for the UI (e.g. "33" or "missing"). */
  value_display?: string;
  /** @deprecated tertile label — kept optional for older snapshots */
  level?: string;
}

export interface Screening {
  probability: number;
  threshold: number;
  label: string;
  risk_label?: string;
  disclaimer: string;
  model_name?: string;
}

export interface XaiItem {
  feature: string;
  display_name: string;
  shap: number;
}

export interface EvidenceItem {
  title: string;
  doc_id: string;
  score: number;
}

export interface CaseStimulus {
  id: string;
  source_case_id: string;
  stratum: "high" | "borderline" | "low";
  test_index?: number;
  student_profile: ProfileFeature[];
  screening: Screening;
  xai: XaiItem[];
  evidence: EvidenceItem[];
  report_summary: string;
  report_source?: string;
}

export type Scores = Partial<Record<LikertQuestionId, number>>;
export type QualitativeAnswers = Partial<Record<QualitativeQuestionId, string>>;

export interface CaseRatingState {
  scores: Scores;
  qualitative: QualitativeAnswers;
  completed: boolean;
  last_saved_iso?: string;
}

export interface StimulusSnapshot {
  case_bank_version: string;
  case_id: string;
  source_case_id: string;
  stratum: string;
  test_index?: number;
  probability: number;
  threshold: number;
  label: string;
  student_profile: ProfileFeature[];
  xai: XaiItem[];
  evidence: EvidenceItem[];
  report_summary: string;
  report_source?: string;
}

export interface RatingRecord {
  protocol_version: string;
  questions_version: string;
  case_bank_version: string;
  session_id: string;
  expert_id: string;
  case_id: string;
  case_index: number;
  timestamp_iso: string;
  event: "save" | "complete_case" | "finish_study";
  scores: Record<LikertQuestionId, number | null>;
  qualitative: Record<QualitativeQuestionId, string>;
  stimulus_snapshot: StimulusSnapshot;
}
