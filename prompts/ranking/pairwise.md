# Pairwise ranking (PRP) prompts

**Sources:** Qin et al., *PRP* (NAACL-Findings 2024) §3 [verbatim template below]; reference impl `github.com/ielab/llm-rankers` (pairwise.allpair / heapsort / bubblesort).
**Role:** the comparison unit for tournaments (`swiss_scheduler.md`) and disagreement tie-breaks (§F2). Full round-robin only for N ≤ ~25.

## Original template [verbatim, passage ranking]

```
Given a query {query}, which of the following two passages is more relevant to the query?
Passage A: {doc1}
Passage B: {doc2}
Output Passage A or Passage B:
```

## Adapted template [adapted — forecast framing]

```
It is {as_of_date}. PAPER (future work removed): {paper_summary}
EVIDENCE (as of {as_of_date}): {evidence}

Which of the following two candidate future directions is the research community more
likely to pursue and publish impactfully within {horizon_months} months?
A direction already accomplished before {as_of_date} automatically loses.

Direction A: {candidate_A}
Direction B: {candidate_B}

Output exactly "Direction A" or "Direction B":
```

(For per-aspect reasoning votes use `generation/aiideabench_r_pairwise_judge.md` — costlier, richer.)

## Protocol (non-negotiable, from PRP)

1. **Both orders, every pair**: u(A,B) and u(B,A). Agreement → win; disagreement → **tie (0.5)**.
2. **Aggregation:** win counts (Copeland) for a quick rank; **Bradley–Terry fit** on the win/tie matrix for fusion-grade strengths + uncertainties (§F1; Shah & Wainwright win-ratio guarantees).
3. **Ties in aggregate:** break by the prior ranking (retrieval score / fused standings) — PRP's own fallback.
4. GPT-side soft votes: read the "Direction A"/"Direction B" first-token logprobs → fractional wins (§L2).

## Variants & cost (N candidates, top-k)

| Variant | Calls (×2 orders) | Notes |
|---|---|---|
| allpair (round-robin) | N(N−1) | most robust to initial order (PRP: 72.42→72.40 under inversion); parallel |
| heapsort | O(k·log₂N) | assumes transitivity — LLM comparators aren't (Qin's own caveat); use only with a good prior |
| sliding-K (backward bubble passes) | O(k·N) | backward (promotion) passes only — forward passes degrade (PRP App. A) |
| Swiss (see swiss_scheduler.md) | O(R·N/2) | the budgeted default |
