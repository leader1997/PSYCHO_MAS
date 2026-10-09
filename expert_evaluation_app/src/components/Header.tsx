interface HeaderProps {
  expertId: string;
  completed: number;
  total: number;
  onExportCsv: () => void;
  onExportJsonl: () => void;
}

export function Header({
  expertId,
  completed,
  total,
  onExportCsv,
  onExportJsonl,
}: HeaderProps) {
  const pct = total ? Math.round((completed / total) * 100) : 0;
  return (
    <header className="app-header">
      <div className="brand">
        <span className="brand-icon" aria-hidden>
          ▣
        </span>
        <div>
          <h1>HMS Expert Assessment: AI-Assisted XAI + RAG Mental Health Screening Framework</h1>
          <p className="brand-sub">Non-diagnostic expert review interface — for research evaluation only</p>
        </div>
      </div>
      <div className="header-meta">
        <div className="expert-badge">
          Expert ID <strong>{expertId}</strong>
        </div>
        <div className="progress-block">
          <div className="progress-label">
            Progress <span>{completed} / {total} cases</span>
          </div>
          <div className="progress-track" role="progressbar" aria-valuenow={completed} aria-valuemin={0} aria-valuemax={total}>
            <div className="progress-fill" style={{ width: `${pct}%` }} />
          </div>
        </div>
        <div className="export-btns">
          <button type="button" className="btn ghost" onClick={onExportCsv}>
            Export CSV
          </button>
          <button type="button" className="btn ghost" onClick={onExportJsonl}>
            Export JSONL
          </button>
        </div>
      </div>
    </header>
  );
}
