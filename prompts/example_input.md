# Shared smoke-test fixture

*One instantiation input reused by every prompt file. The "paper" is RankGPT (Sun et al., EMNLP 2023) with its future work masked — chosen because we know what actually happened after it, so ranking quality is eyeballable. Planted candidates test the failure modes: C4 is an SSLI distractor (topical but not implied), C6 is an anachronism (already done before t₀).*

## {as_of_date}
April 2023

## {horizon_months}
30

## {paper_summary}  (Agent-1 style, abridged)
```json
{
  "problem": "Can generative chat LLMs re-rank passages for search, zero-shot?",
  "methods": ["instructional permutation generation over 20-passage windows", "sliding window back-to-first", "permutation distillation into a 435M cross-encoder"],
  "datasets": ["TREC-DL19/20", "BEIR", "Mr.TyDi", "NovelEval (new, contamination-controlled)"],
  "limitations": [
    "L1: acute sensitivity to initial passage order (nDCG 65.8 -> 25.2 under shuffle)",
    "L2: proprietary API models; open-source LLMs fail at permutation generation",
    "L3: sliding window is sequential and latency-heavy",
    "L4: malformed outputs (repetition/missing/rejection) need post-hoc repair",
    "L5: effectiveness capped by first-stage BM25 recall"
  ],
  "unexplored_axes": ["training-time alignment for ranking", "efficiency", "robustness"]
}
```

## {candidates}
```
C1. Fine-tune an open-source 7B model with reinforcement learning to perform listwise reranking with explicit reasoning, matching GPT-4 quality at a fraction of inference cost.
C2. Replace generation-based permutation output with logit-based scoring over candidate labels to cut latency and remove malformed-output failure modes.
C3. Develop order-robust ranking schemes (e.g., setwise selection or tournament scheduling) that keep accuracy under adversarial initial orderings.
C4. Apply the sliding-window permutation prompt to rank job applicants' resumes for hiring pipelines.                     [planted SSLI distractor]
C5. Distill fused rankings from multiple heterogeneous LLM rankers into a single small cross-encoder for production use.
C6. Show that a cross-encoder (monoBERT-style) fine-tuned on MS MARCO outperforms BM25 for passage reranking.            [planted anachronism — done in 2019]
```

## {evidence}  (abridged; kNN future-work bank + KG stats as of 2023-04)
- "We plan to explore reinforcement learning from ranking feedback to align open models with reranking objectives." (neighbor paper, 2023-02)
- "Future work: reduce inference cost of LLM rankers via distillation." (neighbor paper, 2023-03)
- KG: concept pair (RL, reranking) degree-low + citation-velocity rising; (cross-encoder, MS MARCO) pair saturated since 2020.

## Known outcome (for eyeballing only — never shown to rankers)
C1 ≈ REARANK/Rank-R1 line (realized, high impact); C2 ≈ SCaLR/FIRST line (realized); C3 ≈ Setwise/PRP robustness line (realized); C5 ≈ permutation-distillation follow-ups (realized, moderate); C4 unrealized in-window; C6 pre-dates the paper. A good forecast ranking ≈ C1 > C3 ≈ C2 > C5 >> C4, C6-flagged.
