# Pairwise judge with bias controls (from AI Idea Bench 2025, Appendix R)

**Source:** Qiu et al., *AI Idea Bench 2025* (arXiv:2504.14191), Appendix R — the Ideas-Competition pairwise judge: per-aspect `thinking_process` + binary `choice`, with explicit anti-position-bias and anti-verbosity instructions.
**Our use:** the pairwise comparison unit inside tournaments (`ranking/pairwise.md`, `ranking/swiss_scheduler.md`) and for disagreement-routed tie-breaks (§F2).

## Original mechanism [summary]

Pairwise comparison across six aspects (innovativeness, importance, quality, feasibility, clarity, topic relevance); each aspect gets a reasoning trace + binary choice; explicit instructions to ignore presentation order and answer length; win = 3 points, loss = 0 (we replace 3/0 with Bradley–Terry — metrics.md C3 verdict).

## Adapted prompt [adapted — forecast aspects, §R3]

```
It is {as_of_date}. Given the paper below, compare TWO candidate future directions and
decide which the research community is MORE LIKELY to pursue and publish impactfully
within {horizon_months} months.

PAPER: {paper_summary}
EVIDENCE (as of {as_of_date}): {evidence}

DIRECTION 1: {candidate_A}
DIRECTION 2: {candidate_B}

Judge each aspect separately, thinking briefly first, then choosing 1 or 2:
- imminence: which follows more directly from the paper's limitations?
- enablement: for which did the required methods/data already exist as of {as_of_date}?
- momentum: which aligns better with the evidence of where the field was moving?
- specificity: which is concrete enough to be realized and recognized in a later paper?
- impact_potential: which, if realized, would matter more to this community?

IMPORTANT: the order in which the two directions are presented is random — do not favor
Direction 1 or Direction 2 by position. Do not favor the longer or more eloquent text.
A direction already accomplished before {as_of_date} loses every aspect automatically.

Output JSON:
{"aspects": {"imminence": {"thinking": "...", "choice": 1|2}, ... },
 "overall": {"thinking": "...", "choice": 1|2}}
```

## Notes
- Always run **both orders** (PRP protocol); disagreement between orders = tie (0.5). Per-aspect votes feed the tensor's feature set; `overall` feeds the tournament.
- Original aspect list (innovativeness/importance/quality/feasibility/clarity/relevance) is kept in comments for the badge-scoring variant — but the ranking objective uses ONLY the forecast aspects (HindSight: judged novelty is anti-predictive).
- GPT-side: request the choice token first in a constrained field to enable logprob soft votes (§L1/L2).
