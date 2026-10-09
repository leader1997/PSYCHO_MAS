# RAG Results (Draft)

## Retriever ablation
```
    retriever  precision@5  recall@5  recall@10    mrr  ndcg@10
         bm25         0.04    0.0667     0.2000 0.0722   0.1021
        dense         0.12    0.2000     0.3333 0.2986   0.2396
       hybrid         0.12    0.2000     0.3333 0.2000   0.2071
hybrid_rerank         0.20    0.3333     0.4667 0.5500   0.3887
```

## RAGAS summary
```
system  context_precision  faithfulness  answer_relevancy  context_recall  answer_correctness  semantic_similarity
    S1                0.5           0.5               0.6             0.4                 0.3                  0.5
    S2                0.5           0.5               0.6             0.4                 0.3                  0.5
    S3                0.5           0.5               0.6             0.4                 0.3                  0.5
    S4                0.5           0.5               0.6             0.4                 0.3                  0.5
    S5                0.5           0.5               0.6             0.4                 0.3                  0.5
    S6                0.5           0.5               0.6             0.4                 0.3                  0.5
```

## Safety pass rate: 0.0%

## Citation precision
```
 claim_support_rate  partial_support_rate  unsupported_claim_rate  hallucination_rate  citation_precision  n_claims
                0.0                   1.0                     0.0                 0.0                 1.0         3
```


## Limitations
- Screening communication only; not validated for clinical deployment
- XAI factors are model contributors, not causal explanations
- Retrieval precision remains a bottleneck for some benchmark categories
- Human expert review is required before any real-world use
