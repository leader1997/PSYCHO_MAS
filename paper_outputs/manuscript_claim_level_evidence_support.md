# Manuscript guidance — Claim-level evidence support (query-strategy ablation)

Use this file when writing Methods / Results / Limitations / reviewer replies for **claim-level evidence grounding** in the end-to-end query-strategy ablation.

**What we actually did (do not overclaim):**
- Automatic **RAGAS faithfulness** (statement-level support vs pooled retrieved passages, B=5)
- Automatic **lexical claim-support rate** (sentence token overlap with evidence)
- **Not** citation-level checking (no claim → source ID verification)
- **Not** human claim annotation

Related numbers: [`end_to_end_query_strategy_ablation_v2_for_paper.md`](end_to_end_query_strategy_ablation_v2_for_paper.md) (Table X).  
Prompts: [`supplement_query_strategy_ablation_prompts.md`](supplement_query_strategy_ablation_prompts.md).

---

## 1. What this analysis means (author note)

Reviewers asked for finer-grained grounding than aggregate retrieval scores. We therefore checked whether **generated factual statements** are supported by the **retrieved evidence pool**, not by external knowledge or by explicit citations.

| Check | Granularity | How | Role in paper |
|---|---|---|---|
| RAGAS faithfulness | Statement / claim vs contexts | LLM judge (`gpt-oss:20b` protocol; temperature 0) | Primary claim-level grounding metric |
| Claim support rate | Answer sentence vs contexts | Lexical: sentence kept if ≥25% of tokens (length ≥4) appear in concatenated evidence; sentences &lt;20 chars ignored | Secondary / complementary; often near ceiling |
| Citation checking | Claim ↔ cite ID | — | **Not implemented** — state in Limitations |

---

## 2. Where to put text in the manuscript

| Section | Include? | Content |
|---|---|---|
| Methods (evaluation metrics) | **Yes** | Definition of both checks + what was *not* done |
| Results (near Table X) | **Yes** | Faithfulness + claim-support numbers; note claim support saturates |
| Limitations | **Yes** | Automatic only; no citation checking; no human adjudication |
| Figure caption | Optional | One line pointing to claim-level checks |
| Supplementary | Optional | Point to Appendix Y for prompts; claim-support definition |

---

## 3. Paste-ready Methods (recommended)

**Claim-level evidence support.** To further strengthen the evaluation beyond aggregate retrieval scores, we added automatic claim-level evidence-support analysis. Generated answers were checked against the pooled retrieved passages (final evidence budget B=5) in two ways: (i) **RAGAS faithfulness**, which decomposes each answer into statements and scores whether those statements are supported by the provided contexts using an LLM judge (`gpt-oss:20b`, temperature 0; embeddings `nomic-embed-text`); and (ii) a **lexical claim-support rate**, defined as the fraction of answer sentences (length &gt; 20 characters) for which at least 25% of content tokens (length ≥ 4) appear in the concatenated evidence text. These checks provide a more granular assessment of evidence grounding than retrieval Hit/Recall alone. We did **not** perform citation-level verification (linking each claim to an explicit source or chunk identifier) or human multi-rater claim adjudication.

---

## 4. Shorter Methods sentence (if space is tight)

> We additionally assessed claim-level evidence support via RAGAS faithfulness and a lexical claim-support rate against the pooled retrieved passages (B=5); citation-level checking was not performed.

---

## 5. Paste-ready Results

**Claim-level grounding.** Across 20 cases (360 question–arm answers), lexical claim support was near ceiling for all query strategies (Direct 0.990, Template 0.977, LLM planning 0.971; case-level means), indicating broad token overlap with retrieved evidence but limited discriminative power. RAGAS faithfulness was more informative: Direct 0.789 (0.740–0.831), Template 0.747 (0.705–0.791), LLM planning 0.735 (0.691–0.777). The paired difference Δ(LLM−Direct) was −0.054 (95% CI [−0.090, −0.016]); Δ(LLM−Template) was −0.012 (95% CI [−0.074, +0.047]). Thus LLM query planning did not improve statement-level grounding relative to the direct baseline under this automatic protocol.

**Judge disclosure (required until re-scored):** If Table X still uses the interim `llama3` RAGAS pass, add: *Faithfulness/relevancy values reported here were produced with judge `llama3`; the manuscript protocol specifies `gpt-oss:20b`.* Re-score before final submission if claiming gpt-oss parity with the rest of the paper.

---

## 6. Paste-ready Limitations

> Claim-level evidence support was assessed automatically. Lexical claim support saturates and may overestimate grounding. RAGAS faithfulness depends on a single LLM judge and is not equivalent to expert claim–evidence adjudication. Answers did not emit structured citations; therefore we did not evaluate citation correctness (claim–source linking).

---

## 7. Paste-ready reviewer response

Thank you for suggesting stronger grounding evaluation. We added automatic claim-level evidence-support analysis for all 360 end-to-end answers: RAGAS faithfulness (statement support against pooled retrieved contexts) and a lexical claim-support rate. Results are reported in Table X. We clarify that this is not citation-level checking or human claim annotation; those remain limitations / future work.

---

## 8. Table footnote (for Table X)

> Claim support rate = fraction of answer sentences with ≥25% lexical token overlap in the pooled evidence (B=5). RAGAS faithfulness = automatic statement-level support vs the same evidence. Neither metric is citation-level verification.

---

## 9. Checklist before you paste into the paper

- [ ] Name **both** metrics (faithfulness + claim support), not a vague “claim analysis”
- [ ] Say checks are against **retrieved passages (B=5)**, not the open web
- [ ] Say **automatic** (LLM / lexical), not expert-verified
- [ ] Explicitly exclude **citation checking** if you have not implemented it
- [ ] Report numbers from Table X with case-level CIs where available
- [ ] Align RAGAS judge name (`gpt-oss:20b` vs interim `llama3`) with the scored CSV
- [ ] Do **not** claim “citation faithfulness” or “human factuality audit”

---

## 10. Phrases to avoid vs prefer

| Avoid | Prefer |
|---|---|
| “We performed citation checking” | “We performed claim-level evidence-support checks against retrieved passages” |
| “Human-verified factuality” | “Automatic RAGAS faithfulness and lexical claim support” |
| “All claims were fact-checked” | “Statement support was scored automatically relative to the evidence pool” |
| “Granular assessment” alone (vague) | “Statement-level RAGAS faithfulness plus sentence-level claim-support rate” |
