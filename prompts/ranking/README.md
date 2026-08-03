# prompts/ranking/ — paradigm cheat-sheet

One file per paradigm, each with the original [verbatim] template, our forecast-framed [adapted] version, and its protocol. Candidates = normalized direction statements; condition = Agent-1 summary (+ evidence). The anachronism rule ("already done before {as_of_date} ranks last") appears in every adapted prompt.

## Complexity / robustness / logprobs (N candidates, top-k; numbers from the source papers)

| Paradigm (file) | LLM calls | Order-robust? | Needs logprobs? | Works on Claude? | Key evidence & source impl |
|---|---|---|---|---|---|
| Pointwise yes/no, 1–10, PMI (`pointwise.md`) | O(N), parallel | trivially | yes for logprob/PMI variants | score-sampling fallback only | weakest; needs per-paper calibration; qlm degrades with scale (Zhuang). `ielab/llm-rankers` |
| Pairwise PRP (`pairwise.md`) | allpair N(N−1); heapsort O(k·log N); sliding O(k·N) | **allpair: best** (72.42→72.40 under inversion) | no (soft votes optional) | yes | both-orders + tie protocol; win matrix → Bradley–Terry. `ielab/llm-rankers` |
| Setwise (`setwise.md`) | heapsort O(k·log_c N); bubble O(k·N/(c−1)); Swiss O(R·N/c) | yes (unordered set) | logits mode optional (soft distribution/call) | generation mode yes | −62% cost vs pairwise-heapsort, −0.8% nDCG (Zhuang). `ielab/llm-rankers` |
| Listwise (`listwise.md`) | 1 call (N≤30); windows O(r·N/s) | **no** — 65.8→25.2 under shuffle (RankGPT) → **B=3 shuffle-bootstrap mandatory** (Hou) | no | yes | best quality/cost on frontier models; repair protocol for malformed output. `sunnweiwei/RankGPT`, `lezhang7/Rearank`, `RUCAIBox/LLMRank` |
| Swiss scheduler (`swiss_scheduler.md`) | O(R·N/c) + routed matches | yes (score-adjacent pairing) | soft points optional | yes (hard points) | Si N=5 non-monotone; Nova precedent; co-scientist proximity-Elo precedent — cite both | `NoviScl/AI-Researcher` |

## Assignment to agents (current plan)

- **GPT agent:** listwise + bootstrap (default) → Swiss-Setwise with soft logprob points (upgrade); pointwise yes/no logprobs as calibration anchors (§L5).
- **Claude agent:** listwise + bootstrap (default) → Swiss-Setwise with hard points; confidence proxy = bootstrap agreement rate (no logprobs on the API — §Q6/§L3).
- **Open-weights sidecar (optional 3rd ranker, §F4):** PMI scores (`pointwise.md` (c)) + listwise.likelihood — full logprobs available locally.

## Robustness suite every paradigm must pass before use (§E3)

original/reversed/random candidate order (ΔMRR), RBO across repeat runs, repetition/missing/rejection rates, fixture check: `example_input.md` C6 (anachronism) ranked last, C4 (SSLI) below all realized-line candidates.
