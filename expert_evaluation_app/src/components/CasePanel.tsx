import { useMemo } from "react";
import type { CaseStimulus } from "../types";
import { buildLabeledProfile } from "../lib/featureLabels";
import {
  scrubReportSummary,
  visibleEvidence,
  visibleProfile,
  visibleXai,
} from "../lib/stimulusView";

interface CasePanelProps {
  caseItem: CaseStimulus;
  caseIndex: number;
  total: number;
}

export function CasePanel({ caseItem, caseIndex, total }: CasePanelProps) {
  const p = caseItem.screening.probability;
  const above = p >= caseItem.screening.threshold;
  const xai = visibleXai(caseItem.xai);
  const maxAbs = Math.max(...xai.map((x) => Math.abs(x.shap)), 0.01);
  const evidence = visibleEvidence(caseItem.evidence);
  const profile = useMemo(
    () => buildLabeledProfile(visibleProfile(caseItem.student_profile ?? [])),
    [caseItem.student_profile],
  );
  const report = scrubReportSummary(caseItem.report_summary);

  return (
    <aside className="case-panel">
      <div className="case-chip">
        Case {caseIndex + 1} of {total} · stratum: <em>{caseItem.stratum}</em> ·{" "}
        <code>{caseItem.source_case_id}</code>
      </div>

      <section className="card">
        <h2>Student Case (dataset profile)</h2>
        <dl className="meta-grid">
          <div>
            <dt>Case ID</dt>
            <dd>{caseItem.id}</dd>
          </div>
          <div>
            <dt>Source</dt>
            <dd>{caseItem.source_case_id}</dd>
          </div>
          {caseItem.test_index != null && (
            <div>
              <dt>Test index</dt>
              <dd>{caseItem.test_index}</dd>
            </div>
          )}
        </dl>

        <details className="profile-dropdown">
          <summary>Student features</summary>
          <div className="profile-panel-inner">
            <ul className="profile-list profile-list-scroll">
              {profile.map((f) => (
                <li key={f.key}>
                  <span>{f.display_name}</span>
                  <strong className={`feat-val ${f.missing ? "missing" : ""}`}>
                    {f.label}
                  </strong>
                </li>
              ))}
            </ul>
          </div>
        </details>
        <p className="fine-print">
          Values are de-identified HMS features with interpretive labels where coded
          (gender, financial stress, belonging, etc.). Continuous scores show the raw number.
        </p>
      </section>

      <section className="card">
        <h2>AI Screening Output</h2>
        <div className="prob-row">
          <span>Probability of elevated risk</span>
          <strong className={above ? "warn" : "ok"}>{p.toFixed(4)}</strong>
        </div>
        <div className="prob-track">
          <div
            className={`prob-fill ${above ? "high" : "low"}`}
            style={{ width: `${Math.min(100, p * 100)}%` }}
          />
        </div>
        <div className="prob-meta">
          <span>Threshold: {caseItem.screening.threshold.toFixed(2)}</span>
          <span className={above ? "warn" : "ok"}>{caseItem.screening.label}</span>
        </div>
        <p className="disclaimer">{caseItem.screening.disclaimer}</p>
      </section>

      <section className="card">
        <h2>XAI Explanation (Top Contributors)</h2>
        <ul className="xai-list">
          {xai.map((x) => {
            const w = (Math.abs(x.shap) / maxAbs) * 100;
            const pos = x.shap >= 0;
            return (
              <li key={x.feature}>
                <div className="xai-label">
                  <span>{x.display_name}</span>
                  <span className={pos ? "pos" : "neg"}>
                    {pos ? "+" : ""}
                    {x.shap.toFixed(2)}
                  </span>
                </div>
                <div className="xai-track">
                  <div
                    className={`xai-bar ${pos ? "pos" : "neg"}`}
                    style={{ width: `${w}%` }}
                  />
                </div>
              </li>
            );
          })}
        </ul>
        <p className="fine-print">SHAP values are statistical contributors, not causes.</p>
      </section>

      <section className="card">
        <h2>Retrieved Evidence</h2>
        <ol className="evidence-list">
          {evidence.map((e) => (
            <li key={`${e.doc_id}-${e.title}`}>
              <div className="ev-title">{e.title}</div>
              <div className="ev-meta">
                <code>{e.doc_id}</code>
                <span>similarity {e.score.toFixed(2)}</span>
              </div>
            </li>
          ))}
        </ol>
      </section>

      <section className="card">
        <h2>Generated AI Report (Summary)</h2>
        {caseItem.report_source && (
          <p className="fine-print">Source: {caseItem.report_source}</p>
        )}
        <div className="report-box prewrap">{report}</div>
      </section>
    </aside>
  );
}
