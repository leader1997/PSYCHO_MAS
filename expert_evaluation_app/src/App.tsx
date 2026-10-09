import { useMemo, useState } from "react";
import casesData from "./data/cases.json";
import { CASE_BANK_VERSION } from "./data/rubric";
import type { LikertQuestionId, QualitativeQuestionId } from "./data/rubric";
import { CasePanel } from "./components/CasePanel";
import { EvalForm } from "./components/EvalForm";
import { ExpertGate } from "./components/ExpertGate";
import { Header } from "./components/Header";
import {
  allScoresPresent,
  appendLog,
  completedCount,
  createEmptyQualitative,
  createEmptyScores,
  exportCsv,
  exportJsonl,
  loadStore,
  saveStore,
  startSession,
  type SessionStore,
} from "./lib/logging";
import type { CaseRatingState, CaseStimulus } from "./types";
import "./App.css";

const CASES = casesData as CaseStimulus[];

function ratingFor(store: SessionStore, caseId: string): CaseRatingState {
  const existing = store.ratings[caseId];
  return {
    scores: existing?.scores ?? createEmptyScores(),
    qualitative: existing?.qualitative ?? createEmptyQualitative(),
    completed: existing?.completed ?? false,
    last_saved_iso: existing?.last_saved_iso,
  };
}

export default function App() {
  const [store, setStore] = useState<SessionStore | null>(() => loadStore());
  const [toast, setToast] = useState<string | null>(null);

  const caseIndex = store?.case_index ?? 0;
  const caseItem = CASES[Math.min(caseIndex, CASES.length - 1)];
  const rating = store && caseItem ? ratingFor(store, caseItem.id) : null;
  const done = store ? completedCount(store, CASES.map((c) => c.id)) : 0;
  const canNext = rating ? allScoresPresent(rating.scores) : false;
  const isLast = caseIndex >= CASES.length - 1;
  const studyDone = done >= CASES.length;

  const flash = (msg: string) => {
    setToast(msg);
    window.setTimeout(() => setToast(null), 2500);
  };

  const updateRating = (patch: Partial<CaseRatingState>) => {
    if (!store || !caseItem) return;
    const prev = ratingFor(store, caseItem.id);
    const nextRating: CaseRatingState = { ...prev, ...patch };
    const next: SessionStore = {
      ...store,
      ratings: { ...store.ratings, [caseItem.id]: nextRating },
    };
    saveStore(next);
    setStore(next);
  };

  const setScore = (id: LikertQuestionId, value: number) => {
    if (!rating) return;
    updateRating({ scores: { ...rating.scores, [id]: value } });
  };

  const setQualitative = (id: QualitativeQuestionId, value: string) => {
    if (!rating) return;
    updateRating({ qualitative: { ...rating.qualitative, [id]: value } });
  };

  const go = (index: number) => {
    if (!store) return;
    const next = {
      ...store,
      case_index: Math.max(0, Math.min(CASES.length - 1, index)),
    };
    saveStore(next);
    setStore(next);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const onSave = () => {
    if (!store || !caseItem || !rating) return;
    const next = appendLog(store, caseItem, caseIndex, rating, "save");
    setStore(next);
    flash("Progress saved for this case.");
  };

  const onNext = () => {
    if (!store || !caseItem || !rating || !canNext) return;
    const event = isLast ? "finish_study" : "complete_case";
    const logged = appendLog(store, caseItem, caseIndex, rating, event);
    setStore(logged);
    if (isLast) {
      flash("Study complete. Please export CSV and JSONL for the manuscript.");
    } else {
      go(caseIndex + 1);
      flash("Case rating recorded.");
    }
  };

  const statusLine = useMemo(() => {
    if (!store) return null;
    return `Session ${store.session_id.slice(0, 8)} · bank ${CASE_BANK_VERSION}`;
  }, [store]);

  if (!store) {
    return <ExpertGate onStart={(id) => setStore(startSession(id))} />;
  }

  if (!caseItem || !rating) {
    return null;
  }

  return (
    <div className="app-shell">
      <Header
        expertId={store.expert_id}
        completed={done}
        total={CASES.length}
        onExportCsv={() => {
          exportCsv(store);
          flash("CSV downloaded.");
        }}
        onExportJsonl={() => {
          exportJsonl(store);
          flash("JSONL downloaded.");
        }}
      />

      {toast && <div className="toast">{toast}</div>}

      {studyDone && (
        <div className="banner success">
          All {CASES.length} cases marked complete for {store.expert_id}. Export CSV + JSONL
          before closing the browser.
        </div>
      )}

      <main className="main-grid">
        <CasePanel caseItem={caseItem} caseIndex={caseIndex} total={CASES.length} />
        <EvalForm
          scores={rating.scores}
          qualitative={rating.qualitative}
          onScore={setScore}
          onQualitative={setQualitative}
        />
      </main>

      <footer className="nav-footer">
        <button
          type="button"
          className="btn"
          disabled={caseIndex <= 0}
          onClick={() => go(caseIndex - 1)}
        >
          Previous Case
        </button>
        <div className="nav-center">
          <span className="status-line">{statusLine}</span>
          {!canNext && <span className="hint">Answer Q1–Q20 to enable Next</span>}
        </div>
        <div className="nav-right">
          <button type="button" className="btn" onClick={onSave}>
            Save Progress
          </button>
          <button
            type="button"
            className="btn primary"
            disabled={!canNext}
            onClick={onNext}
          >
            {isLast ? "Finish study" : "Next Case →"}
          </button>
        </div>
      </footer>
    </div>
  );
}
