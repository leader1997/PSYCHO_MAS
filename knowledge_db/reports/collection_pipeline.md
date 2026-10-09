# Knowledge-Base Collection Pipeline — Methodology Documentation

**Project**: Controlled Local Multi-Agent XAI-RAG for Non-Clinical Depression/Anxiety Screening Communication
**Knowledge base version**: v1.0 (2026-06-25)
**Corpus size**: 52 selected documents (44 PDFs legally downloaded) + 148 borderline ("maybe") + 135 rejected

---

## 1. Pipeline overview

The knowledge base was assembled through a **five-stage iterative pipeline**: (i) structured
multi-query web search, (ii) candidate aggregation and de-duplication, (iii) transparent
quality scoring against a six-criterion rubric, (iv) legal-access PDF acquisition, and
(v) SHA-1-based file-level de-duplication and corpus trimming to per-category target counts.
All stages are reproducible from the persisted Python scripts under
`/home/z/my-project/scripts/`.

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ 1. Multi-query   │ →  │ 2. Aggregate &   │ →  │ 3. Score &       │
│    web search    │    │    de-duplicate  │    │    filter (≥7)   │
│   (~50 queries)  │    │    by URL        │    │    accept/maybe/ │
└──────────────────┘    └──────────────────┘    │    reject        │
                                                └──────────────────┘
                                                          ↓
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ 5. SHA-1 dedupe  │ ←  │ 4. PDF download  │ ←  │ Trim per-category│
│    + final trim  │    │    (arXiv / PMC  │    │    to target     │
│    + metadata +  │    │    / Frontiers / │    │    counts        │
│    summary       │    │    inst. repos)  │    └──────────────────┘
└──────────────────┘    └──────────────────┘
```

---

## 2. Search strategy

### 2.1 Query design

Fifty structured web-search queries were authored across **nine expertise categories**
(A–I) defined a priori to match the project's RAG-retrieval needs. Queries combined
domain terms (e.g., *loneliness*, *food insecurity*, *SHAP*) with format qualifiers
(e.g., `pdf`, `systematic review`, `open access`) and population terms
(e.g., *college students*, *university students*) to bias results toward legally
downloadable peer-reviewed sources.

The full query list is persisted in `logs/search_queries_used.txt` and the query
definitions live in `scripts/queries.py`. Example queries per category:

| Category | Example query |
|----------|---------------|
| A. Student mental health & screening | `"Healthy Minds Study methodology codebook pdf"` |
| B. College depression/anxiety | `"PHQ-9 validation university students depression screening pdf"` |
| C. Loneliness / isolation | `"UCLA loneliness scale university students depression pdf"` |
| D. Academic impairment / stress | `"academic stress academic impairment depression anxiety university students pdf"` |
| E. Financial stress / food insecurity | `"food insecurity depression anxiety college students pdf"` |
| F. Sleep / wellbeing / belonging | `"flourishing wellbeing college students mental health pdf"` |
| G. Screening communication | `"mental health screening communication non diagnostic pdf"` |
| H. XAI / SHAP / calibration | `"SHAP explainable AI feature contribution pdf"` |
| I. RAG / LLM safety | `"retrieval augmented generation citation faithfulness pdf"` |

### 2.2 Search engine

All searches were issued through the **Z.AI web-search function** (`z-ai function -n web_search`),
which aggregates results from multiple web indices. Each query returned up to 10 results
with fields `url`, `name`, `snippet`, `host_name`, `date`. Raw JSON responses are
persisted under `logs/raw_searches/` for full reproducibility.

### 2.3 Curated high-confidence sources

A small set of **foundational methodology papers** (SHAP, TreeSHAP, LIME, calibration,
RAG, SELF-RAG, hallucination surveys, HELM) were added by hand-curated entries in
`scripts/queries.py::CURATED_SOURCES` to guarantee coverage of canonical references
that may not have surfaced through generic queries. Each curated entry carries a
manual quality score and a documented `reason_for_inclusion`.

### 2.4 Second-pass targeted searches

After the first pass, three categories (E financial stress, G screening communication,
and C loneliness) were under-represented in legally downloadable PDFs. A second batch
of 17 targeted queries was issued (also persisted in `search_queries_used.txt`)
focusing on PMC open-access equivalents and adjacent terms (e.g.,
`"material hardship basic needs insecurity students mental health pdf"`).

---

## 3. Candidate aggregation and de-duplication

All raw search JSON files were parsed and merged into a single candidate pool.
De-duplication was performed **by canonical URL** (exact string match). Where the
same paper appeared under multiple URLs (e.g., arXiv abstract page vs. PDF URL, or
PubMed landing page vs. PMC full text), both URLs were retained as separate
candidates during scoring but reconciled during PDF acquisition (Section 5).

Result: 50 queries yielded approximately **280 unique candidate URLs** before
scoring.

---

## 4. Quality scoring rubric

Each candidate was scored 0–10 against a transparent rubric. The rubric and its
implementation live in `scripts/collect_kb.py::score_candidate`.

| Criterion | Max points | Operationalisation |
|-----------|-----------|--------------------|
| Relevance to project | 3 | Number of category-specific keyword hits in title+snippet (3+ hits → 3, 2 → 2, 1 → 1) |
| Source reliability | 2 | Host classified as high (arXiv, NeurIPS, ACL, PMC, WHO, CDC, NIH, OECD, APA, .edu repositories, Frontiers, MDPI, PLOS, BMJ) → 2; medium (Springer, Nature, T&F, ScienceDirect) → 1; skip-list domains (Academia.edu, ResearchGate, Scribd, SlideShare, StudoCu, CourseHero, Chegg) → 0 and auto-reject |
| Usefulness for retrieval | 2 | URL points directly to a PDF or a known PDF source → 2; landing page on a reliable publisher → 1 |
| Recency | 1 | Year ≥ 2018 in title or snippet → 1; 2015–2017 → 0; foundational methodology papers exempted |
| Safety alignment | 1 | Document does not focus on treatment guidelines, medication, psychotherapy manuals, suicide intervention, or diagnostic manuals → +1; if it does → −2 penalty |
| Metadata completeness | 1 | Title, snippet, host, and URL all present → 1 |

**Decision thresholds** (pre-registered):

| Score | Disposition |
|-------|-------------|
| ≥ 7 | Accepted into the corpus |
| 5–6 | Placed in `maybe_documents.csv` for manual review (148 documents) |
| < 5 | Placed in `rejected_documents.csv` with rejection reason (135 documents) |

Curated methodology papers were exempted from the auto-scoring step (their manual
scores are stored in `CURATED_SOURCES`).

---

## 5. PDF acquisition protocol

### 5.1 Legal-access policy

A PDF was downloaded **only** if its source was one of:

- arXiv (open-access preprints)
- NeurIPS / OpenReview proceedings (open-access)
- ACL Anthology (open-access)
- PubMed Central (PMC) full-text articles — accessed via the **EuropePMC**
  `?pdf=render` endpoint, which is the legally-sanctioned PDF mirror for all
  PMC open-access content
- Semantic Scholar PDF mirror (when the underlying paper is open-access)
- bioRxiv / medRxiv preprints
- Official public-health or education organisations (WHO, CDC, NIH, APA, OECD,
  Healthy Minds Network, SAMHSA)
- Institutional repositories using `viewcontent.cgi` (Digital Commons / bepress)
  or equivalent persistent URLs
- Publisher open-access PDFs (Frontiers, MDPI, PLOS, BMJ Open, Nature
  Scientific Reports, Springer BMC-series)

Paywalled articles without an open-access equivalent were **not** downloaded; their
landing-page metadata was retained for citation-only use, with `download_status =
"not_downloaded"` flagged in the metadata.

### 5.2 URL-to-PDF rewriting

For each accepted candidate, the pipeline attempted to derive a direct PDF URL using
the following rewrite rules (implemented in `scripts/retry_pass2.py::derive_pdf_url`):

| Source URL pattern | Rewritten PDF URL |
|--------------------|-------------------|
| `arxiv.org/abs/<id>` | `arxiv.org/pdf/<id>` |
| `pmc.ncbi.nlm.nih.gov/articles/PMC<n>` | `europepmc.org/articles/PMC<n>?pdf=render` |
| `frontiersin.org/.../articles/<doi>` | `<doi>/pdf` |
| `nature.com/articles/<id>` | `<id>.pdf` (open-access only) |
| `link.springer.com/article/<doi>` | `link.springer.com/content/pdf/<doi>.pdf` (BMC open-access) |
| `mdpi.com/<id>` | `mdpi.com/<id>/pdf` |
| `journals.plos.org/<j>/article?id=<id>` | `journals.plos.org/<j>/article/file?id=<id>&type=printable` |
| `bmjopen.bmj.com/content/<v>/<a>` | `<v>/<a>.full.pdf` |

### 5.3 Download client

Downloads were performed with `curl -fsSL` using a realistic Chrome User-Agent
string and `Accept: application/pdf,*/*` header, following redirects, with a
60-second timeout per file. Each downloaded file was verified to:

1. Be at least 5 KB in size (filtering out error pages)
2. Begin with the `%PDF` magic bytes (filtering out HTML masquerading as PDF)

Files failing either check were deleted and the record was marked
`download_status = "not_downloaded"`. Failed downloads were retried in a second
pass using the EuropePMC endpoint for any PMC-linked candidate.

---

## 6. De-duplication

Two layers of de-duplication were applied:

1. **URL-level dedup** during candidate aggregation (Section 3).
2. **File-level SHA-1 dedup** after download: every successfully downloaded PDF
   was hashed and compared to all other downloads. One exact duplicate was
   detected (DOC_F005 was a PMC mirror of DOC_F003, both being the same Dutcher
   et al. 2022 paper); the duplicate was removed and a replacement was pulled
   from the "maybe" pool.

The full SHA-1 manifest is in `logs/duplicate_check_log.txt`. The final corpus
contains **44 unique PDFs by SHA-1**.

---

## 7. Corpus trimming to target counts

To stay within the pre-specified 40–60 document window and to balance the
nine categories, accepted documents were sorted within each category by a
combined key:

```
sort_key = score + (2 if download_status == "downloaded" else 0)
           − (2 if risk_of_noise == "high" else 0)
```

and the top-N per category were retained against the following pre-registered
targets:

| Category | Target | Final accepted | Final downloaded |
|----------|--------|----------------|------------------|
| A. Student mental health & screening | 8–10 | 9 | 5 |
| B. Depression/anxiety (college) | 6–8 | 7 | 6 |
| C. Loneliness / isolation | 4–5 | 4 | 3 |
| D. Academic impairment / stress | 4–5 | 4 | 4 |
| E. Financial stress / food insecurity | 4–5 | 4 | 3 |
| F. Sleep / wellbeing / belonging | 6–8 | 6 | 6 |
| G. Screening communication / literacy | 5–6 | 5 | 4 |
| H. XAI / SHAP / calibration | 6–8 | 7 | 7 |
| I. RAG / LLM safety / hallucination | 5–6 | 6 | 6 |
| **Total** | **40–60** | **52** | **44** |

The 100 → 52 trim discarded lower-scoring duplicates and paywall-only records
in favour of higher-scoring downloaded equivalents.

---

## 8. Inclusion and exclusion criteria

### 8.1 Inclusion criteria (≥ 4 of 8 required)

1. Directly related to student mental health, screening, XAI, RAG, or safe AI
   communication.
2. Peer-reviewed, official, or from a reputable institution.
3. Contains evidence useful for explaining risk/protective factors.
4. Contains definitions, findings, or recommendations useful for non-clinical
   screening communication.
5. Recent (2015–2026), except foundational XAI/RAG/calibration methodology
   papers.
6. Available as a legal PDF.
7. Has clear metadata (title, authors/organisation, year, source, URL).
8. Assignable to one of the nine required expertise categories.

### 8.2 Exclusion criteria

1. Focus mainly on clinical diagnosis.
2. Provides treatment, therapy, medication, or crisis-intervention instructions.
3. Unrelated to students, screening communication, XAI, RAG, or evidence
   grounding.
4. Blog posts, opinion pieces, or commercial pages.
5. Duplicate of an already-accepted document.
6. Lacks reliable source information.
7. Not legally available.
8. Too broad to add signal to retrieval.

### 8.3 Safety alignment

The corpus explicitly excludes any document intended to help a system diagnose
depression or anxiety, provide treatment plans, recommend medication, provide
therapy instructions, perform crisis intervention, or replace a clinician.
The exclusion was enforced both at the scoring stage (the `safety_alignment`
penalty in Section 4) and by manual inspection of any candidate whose snippet
contained the trigger phrases `treatment guidelines`, `medication`,
`psychotherapy manual`, `suicide intervention`, or `diagnostic manual`.

---

## 9. Metadata schema

Each selected document is described by the following fields (CSV header in
`metadata/selected_documents.csv`, JSON Lines in
`metadata/selected_documents.jsonl`):

| Field | Description |
|-------|-------------|
| `doc_id` | Internal ID of the form `DOC_<category-letter><3-digit-index>` |
| `title` | Document title |
| `authors_or_organization` | Authors or issuing organisation |
| `year` | Publication year |
| `source_type` | `peer_reviewed` / `official_report` / `dataset_documentation` / `methodology_paper` / `preprint` |
| `expertise_area` | Human-readable expertise area label |
| `population` | `college_students` / `university_students` / `young_adults` / `general` / `AI_methodology` |
| `main_topic` | Short topic description |
| `relevance_to_project` | One-paragraph relevance statement |
| `pdf_url` | Direct PDF URL (empty if not downloadable) |
| `source_url` | Landing-page URL |
| `download_status` | `downloaded` / `not_downloaded` |
| `reason_for_inclusion` | Free-text justification |
| `risk_of_noise` | `low` / `medium` / `high` |
| `recommended_use` | `factor_evidence` / `screening_communication` / `xai_interpretation` / `rag_evaluation` / `safety_validation` / `dataset_context` |
| `score` | 0–10 quality score from the rubric in Section 4 |
| `local_pdf_path` | Relative path under `knowledge_db/pdfs/` |
| `category_code` | One of A–I |

---

## 10. Reproducibility

### 10.1 Scripts

All collection logic is persisted under `/home/z/my-project/scripts/`:

| Script | Role |
|--------|------|
| `queries.py` | Search queries per category + curated high-confidence sources |
| `collect_kb.py` | Main orchestrator: run searches → score → filter → download → write metadata + summary |
| `retry_and_trim.py` | Second pass: retry failed downloads with EuropePMC, trim to target counts |
| `retry_pass2.py` | Third pass: extra targeted queries for weak categories + URL-to-PDF rewriting |
| `finalize_summary.py` | Regenerate `download_log.txt`, `duplicate_check_log.txt`, and `corpus_summary.md` from current state |

### 10.2 Re-running the pipeline

```bash
# Full re-run from scratch
python /home/z/my-project/scripts/collect_kb.py
python /home/z/my-project/scripts/retry_and_trim.py
python /home/z/my-project/scripts/retry_pass2.py
python /home/z/my-project/scripts/finalize_summary.py

# Iterate without re-running web searches
KB_SKIP_SEARCH=1 python /home/z/my-project/scripts/collect_kb.py
```

### 10.3 Software environment

- Search API: Z.AI web-search function (`z-ai` CLI)
- HTTP client: `curl 7.x` with Chrome User-Agent
- Hashing: Python `hashlib.sha1`
- Metadata: CSV + JSON Lines (UTF-8)
- No proprietary software dependencies

### 10.4 Provenance logs

Every URL ever queried, every download attempt, and every SHA-1 hash is
preserved under `knowledge_db/logs/`, enabling third-party audit of the
corpus construction.

---

## 11. Limitations and known biases

1. **Search-engine bias.** The Z.AI web-search index determines which papers
   surface. Foundational papers not indexed by general web search (e.g., older
   NeurIPS proceedings) were added via the curated list, but some relevant
   papers may still be missing.
2. **PDF-access bias.** Paywalled publishers (Elsevier, T&F, Nature
   non-open-access) are under-represented because their PDFs cannot be legally
   downloaded by an automated collector. Their landing-page metadata is
   retained for citation only.
3. **User-Agent blocking.** The Healthy Minds Network website (HTTP 403 to
   non-browser User-Agents) prevented automated download of three high-value
   dataset-context PDFs (DOC_A004–A006). These records remain in the corpus as
   citation-only and must be fetched manually from a workstation browser.
4. **Snippet-derived metadata.** For approximately 11 of 52 records, the
   `main_topic` and `relevance_to_project` fields are derived from search
   snippets rather than full-text PDFs. These should be verified against the
   downloaded PDF text before being used in published RAG outputs.
5. **English-language bias.** All queries were issued in English; non-English
   student-mental-health literature is not represented.
6. **Single-collector bias.** Scoring was performed by an automated rubric
   without independent human double-coding. The `maybe_documents.csv` list
   (148 records) is provided as a reservoir for human reviewers to upgrade
   or downgrade.

---

## 12. Suggested citation formats

### 12.1 Cite the corpus

> *Knowledge base for Controlled Local Multi-Agent XAI-RAG for Non-Clinical
> Depression/Anxiety Screening Communication* (v1.0, 2026-06-25). Collected
> via a five-stage pipeline of structured web search (50 queries across 9
> expertise categories), transparent six-criterion quality scoring,
> legal-access PDF acquisition (arXiv, PubMed Central via EuropePMC, official
> public-health repositories, institutional repositories), and SHA-1
> de-duplication. Final corpus: 52 selected documents, 44 legally
> downloaded PDFs. Metadata, logs, and reproducing scripts archived under
> `knowledge_db/`.

### 12.2 Cite the methodology

> The knowledge base was assembled through an iterative five-stage pipeline:
> (i) fifty structured web-search queries across nine pre-defined expertise
> categories (student mental health, college depression/anxiety, loneliness,
> academic impairment, financial stress, sleep/wellbeing, screening
> communication, XAI/SHAP/calibration, RAG/LLM safety); (ii) URL-level
> de-duplication of approximately 280 candidates; (iii) transparent
> six-criterion quality scoring (relevance, source reliability, retrieval
> usefulness, recency, safety alignment, metadata completeness) with
> thresholds of ≥ 7 accept, 5–6 maybe, < 5 reject; (iv) legal-access PDF
> acquisition restricted to arXiv, NeurIPS, ACL Anthology, PubMed Central
> (via EuropePMC), Semantic Scholar, bioRxiv/medRxiv, WHO/CDC/NIH/OECD/APA,
> institutional repositories, and open-access publishers; and (v)
> SHA-1-based file-level de-duplication and per-category trimming to
> pre-registered target counts. The final corpus comprises 52 documents
> (44 PDFs) with 148 borderline and 135 rejected candidates retained for
> audit. All collection scripts and provenance logs are persisted for
> reproducibility.

### 12.3 BibTeX (suggested entry for the corpus)

```bibtex
@misc{xairag_kb_2026,
  title        = {Knowledge base for {Controlled Local Multi-Agent XAI-RAG}
                  for {Non-Clinical Depression/Anxiety Screening Communication}},
  year         = {2026},
  month        = jun,
  version      = {1.0},
  note         = {52 selected documents (44 legally downloaded PDFs) across
                  9 expertise categories. Collected via structured web
                  search, six-criterion quality scoring, legal-access PDF
                  acquisition, and SHA-1 de-duplication. Metadata, logs,
                  and reproducing scripts archived under knowledge\_db/.},
  howpublished = {\url{knowledge_db/}}
}
```

### 12.4 BibTeX (suggested entry for the pipeline methodology)

```bibtex
@misc{xairag_kb_pipeline_2026,
  title        = {A Five-Stage Reproducible Pipeline for Assembling a
                  Domain-Specific {PDF} Knowledge Base for {RAG} Systems:
                  Application to Non-Clinical Student Mental-Health Screening},
  year         = {2026},
  month        = jun,
  version      = {1.0},
  note         = {Pipeline comprising structured multi-query web search,
                  URL-level de-duplication, six-criterion transparent quality
                  scoring (relevance, source reliability, retrieval usefulness,
                  recency, safety alignment, metadata completeness),
                  legal-access PDF acquisition via arXiv, PubMed Central
                  (EuropePMC), and institutional repositories, and SHA-1
                  file-level de-duplication. Implementation scripts persisted
                  in scripts/collect\_kb.py, retry\_and\_trim.py,
                  retry\_pass2.py, finalize\_summary.py.}
}
```

---

## 13. File index

```
knowledge_db/
├── pdfs/
│   ├── student_mental_health/    (11 PDFs — categories A & B)
│   ├── xai_factors/              (10 PDFs — categories C, D, E)
│   ├── protective_factors/        (6 PDFs — category F)
│   ├── screening_communication/   (4 PDFs — category G)
│   ├── xai_calibration/           (7 PDFs — category H)
│   └── rag_llm_evaluation/        (6 PDFs — category I)
├── metadata/
│   ├── selected_documents.csv     (52 rows, 18 columns)
│   ├── selected_documents.jsonl   (52 records, same schema)
│   ├── maybe_documents.csv        (148 borderline candidates)
│   └── rejected_documents.csv     (135 rejected candidates)
├── logs/
│   ├── search_queries_used.txt    (all 67 queries with categories)
│   ├── download_log.txt           (per-doc download status)
│   ├── duplicate_check_log.txt    (SHA-1 dedup manifest)
│   └── raw_searches/              (raw JSON from each web search)
└── reports/
    └── corpus_summary.md          (executive corpus report)
```

---

*End of methodology documentation.*
