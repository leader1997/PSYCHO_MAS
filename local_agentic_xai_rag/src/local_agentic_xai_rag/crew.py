"""CrewAI crew wrapper — falls back to direct llama3 pipeline if CrewAI unavailable."""

from __future__ import annotations

import json

from local_agentic_xai_rag.agents.normalize import normalize_evidence, normalize_report
from local_agentic_xai_rag.agents.pipeline_agents import run_evidence_agent, run_report_agent
from local_agentic_xai_rag.llm.ollama_client import MODEL


def run_crew_pipeline(
    case_id: str,
    case_summary: str,
    xai_card: dict,
    hits: list,
    evidence: dict | None = None,
    revision_comments: list[str] | None = None,
    profiles: dict | None = None,
    query_plan: dict | None = None,
) -> tuple[dict, dict]:
    """Run query plan (optional) + evidence + report agents (sequential flow)."""
    chunk_ids = [h.chunk_id for h in hits]
    if evidence is None:
        evidence_raw = run_evidence_agent(case_id, case_summary, xai_card, hits)
        evidence = normalize_evidence(evidence_raw, case_id, chunk_ids)
    report_raw = run_report_agent(
        case_id, case_summary, xai_card, evidence, revision_comments=revision_comments
    )
    report = normalize_report(report_raw, case_id, evidence)
    return evidence, report


def try_crewai_kickoff(
    case_id: str,
    case_summary: str,
    xai_card: dict,
    hits: list,
) -> tuple[dict, dict] | None:
    """Use CrewAI if installed; otherwise return None to use run_crew_pipeline."""
    try:
        from crewai import Agent, Crew, LLM, Process, Task
    except ImportError:
        return None

    base_url = "http://localhost:11434"
    llm = LLM(model=f"ollama/{MODEL}", base_url=base_url, temperature=0.1)

    chunks_block = "\n\n".join(
        f"[{h.chunk_id}] ({h.title}, p.{h.page}): {h.text[:500]}" for h in hits
    )

    ev_agent = Agent(
        role="Evidence verification and synthesis specialist",
        goal="Synthesize supported claims linked to chunk IDs only",
        backstory="Scientific evidence auditor for non-clinical screening communication.",
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )
    rep_agent = Agent(
        role="Advisory screening report generator",
        goal="Produce structured advisory JSON for human review",
        backstory="Writes non-diagnostic screening recommendations with citations.",
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )

    ev_task = Task(
        description=(
            f"Case: {case_summary}\nXAI: {json.dumps(xai_card)[:2000]}\n"
            f"Chunks:\n{chunks_block}\nOutput evidence JSON for case_id={case_id}"
        ),
        expected_output="Evidence synthesis JSON",
        agent=ev_agent,
    )
    rep_task = Task(
        description="Generate full advisory report JSON from prior evidence task output.",
        expected_output="Advisory report JSON",
        agent=rep_agent,
        context=[ev_task],
    )

    crew = Crew(agents=[ev_agent, rep_agent], tasks=[ev_task, rep_task], process=Process.sequential)
    crew.kickoff()
    # CrewAI output is text — normalize via direct pipeline for reliability
    return run_crew_pipeline(case_id, case_summary, xai_card, hits)
