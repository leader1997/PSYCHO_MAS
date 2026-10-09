import {
  CATEGORY_ORDER,
  LIKERT_LABELS,
  QUALITATIVE,
  RUBRIC,
  type LikertQuestionId,
  type QualitativeQuestionId,
} from "../data/rubric";
import type { QualitativeAnswers, Scores } from "../types";

interface EvalFormProps {
  scores: Scores;
  qualitative: QualitativeAnswers;
  onScore: (id: LikertQuestionId, value: number) => void;
  onQualitative: (id: QualitativeQuestionId, value: string) => void;
}

const CATEGORY_CLASS: Record<string, string> = {
  xai: "cat-blue",
  evidence: "cat-green",
  report: "cat-orange",
  safety: "cat-purple",
  overall: "cat-teal",
};

export function EvalForm({
  scores,
  qualitative,
  onScore,
  onQualitative,
}: EvalFormProps) {
  return (
    <section className="eval-panel">
      <h2>Expert Evaluation Questionnaire</h2>
      <p className="eval-intro">
        Appendix items Q1–Q20 use a 1–5 Likert scale (required before Next). Q21–Q23 are optional
        free-text feedback for the manuscript.
      </p>

      {CATEGORY_ORDER.map((cat) => {
        const items = RUBRIC.filter((q) => q.category === cat);
        const label = items[0]?.categoryLabel ?? cat;
        return (
          <div key={cat} className={`eval-category ${CATEGORY_CLASS[cat] ?? ""}`}>
            <h3>{label}</h3>
            {items.map((q) => (
              <fieldset key={q.id} className="eval-item">
                <legend>
                  <span className="q-num">Q{q.number}</span> {q.text}
                </legend>
                <div className="likert" role="radiogroup" aria-label={`Q${q.number}`}>
                  {[1, 2, 3, 4, 5].map((v) => (
                    <label
                      key={v}
                      className={`likert-opt ${scores[q.id] === v ? "selected" : ""}`}
                    >
                      <input
                        type="radio"
                        name={q.id}
                        value={v}
                        checked={scores[q.id] === v}
                        onChange={() => onScore(q.id, v)}
                      />
                      <span className="likert-val">{v}</span>
                      <span className="likert-hint">{LIKERT_LABELS[v]}</span>
                    </label>
                  ))}
                </div>
              </fieldset>
            ))}
          </div>
        );
      })}

      <div className="eval-category cat-gray">
        <h3>{QUALITATIVE[0]?.categoryLabel ?? "Section F — Qualitative Feedback"}</h3>
        {QUALITATIVE.map((q) => (
          <label key={q.id} className="qual-item">
            <span className="q-num">Q{q.number}</span> {q.prompt}
            <textarea
              value={qualitative[q.id] ?? ""}
              onChange={(e) => onQualitative(q.id, e.target.value)}
              rows={4}
              placeholder="Optional free-text response…"
            />
          </label>
        ))}
      </div>
    </section>
  );
}
