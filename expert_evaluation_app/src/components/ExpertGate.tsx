interface GateProps {
  onStart: (expertId: string) => void;
}

export function ExpertGate({ onStart }: GateProps) {
  return (
    <div className="gate">
      <div className="gate-card">
        <h1>Expert Evaluation Study</h1>
        <p>
          You will review <strong>9 real de-identified HMS-derived screening cases</strong>{" "}
          (3 high / 3 borderline / 3 low) with prediction, XAI contributors, retrieved evidence,
          and generated report text, then complete Appendix items Q1–Q20 (Likert) and optional
          Q21–Q23 (free text).
        </p>
        <ul>
          <li>This is not a diagnostic tool and not clinical care.</li>
          <li>Case stimuli come from the evaluation case bank + frozen report-agent outputs.</li>
          <li>Ratings are logged for manuscript analysis (CSV / JSONL export).</li>
          <li>Use a stable Expert ID (e.g. E01) across sessions on this browser.</li>
        </ul>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            const fd = new FormData(e.currentTarget);
            const id = String(fd.get("expert_id") || "").trim();
            if (id.length >= 2) onStart(id);
          }}
        >
          <label>
            Expert ID
            <input name="expert_id" required minLength={2} maxLength={32} placeholder="E01" autoFocus />
          </label>
          <button type="submit" className="btn primary">
            Begin evaluation
          </button>
        </form>
      </div>
    </div>
  );
}
