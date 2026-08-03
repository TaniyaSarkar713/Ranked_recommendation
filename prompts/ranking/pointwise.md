# Pointwise ranking prompts

**Sources:** Zhuang et al., *Setwise* (SIGIR 2024) baselines `pointwise.qlm` / `pointwise.yes_no` (reference impl: `github.com/ielab/llm-rankers`); PMI variant = ours (§L4).
**Role in ResearchHorizon:** NEVER the primary ranker (weakest paradigm; needs calibration; Si et al. found pointwise judging ≈ chance). Used only to produce **features** for fusion and **calibration anchors** (SCaLR-style, §L5).

## (a) yes/no relevance — logprob version [adapted from pointwise.yes_no]

```
It is {as_of_date}. PAPER: {paper_summary}
CANDIDATE FUTURE DIRECTION: {candidate_statement}

Will the research community pursue this direction and publish impactful work on it within
{horizon_months} months? Answer with exactly one word: Yes or No.
```

Score = P("Yes") from first-token logprobs (GPT/open models). Claude fallback: sample n=5 at temperature 1, score = fraction Yes.

## (b) Direct 1–10 score [use only as a feature, never a rank]

```
...same context...
Rate the probability this direction is realized impactfully within {horizon_months} months,
1–10 (1 = will not happen, 10 = certain). Output JSON: {"score": n, "why": "..."}
```

## (c) PMI / popularity-corrected likelihood [ours, §L4 — open-weights model required]

```
score(d) = log P(d | {paper_summary})  −  λ · log P(d)
```
Both terms from one local model (vLLM, e.g. Qwen/Llama): conditional and unconditional token-level log-likelihood of the direction statement. The −λ·log P(d) term cancels base-rate fluency — generic "apply LLMs to X" text scores high under ANY conditioning; PMI punishes exactly that (Hou popularity-bias defense). λ tuned on dev; validate via the popularity-per-rank audit curve (§E3).

## Warnings
- Zhuang: `qlm` *degrades* as models grow; `yes_no` needs score calibration to compare across papers — z-normalize within paper before fusing.
- Pointwise sees no cross-candidate context — it cannot demote near-duplicates or spot "already done" (anachronisms pass!). Comparative paradigms must carry the ranking.
- Cost: O(N) calls, embarrassingly parallel — the only virtue.
