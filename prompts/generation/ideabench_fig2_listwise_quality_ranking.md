# Listwise ranking by a defined quality indicator (from IdeaBench Fig. 2)

**Source:** Guo et al., *IdeaBench* (arXiv:2411.02429), Fig. 2 — the Insight-Score stage-1 listwise ranker with a user-configurable `{quality_indicator}`.
**Our use:** the single-call listwise ranking inside each agent (§R1), with the indicator instantiated to **expected realization** (forecast framing). Also reusable for badge dimensions when explicitly labeled.

## Original mechanism [summary of verbatim]

*"You are a reviewer tasked with ranking the quality of a set of research ideas based on {quality_indicator}. The idea with the highest {quality_indicator} should be ranked first…"* — returns a full ranked list `1. Hypothesis (number): (rationale)`. Provenance of items is hidden from the ranker (their gold idea is mixed in unlabeled) — keep that discipline: **never reveal candidate provenance** (generated vs retrieved vs citation-mined).

## Adapted prompt [adapted]

```
You are a research-strategy forecaster. It is {as_of_date}. Below is a paper (future-work
section removed) and a set of candidate future research directions from mixed, unlabeled sources.

PAPER: {paper_summary}
EVIDENCE (state of the field as of {as_of_date}): {evidence}

CANDIDATE DIRECTIONS:
{candidates}

Rank ALL candidates by: {quality_indicator}

Default indicator — EXPECTED REALIZATION: how likely is it that the research community
pursues this direction and publishes impactful work on it within {horizon_months} months
of {as_of_date}? Judge from: which limitation it addresses and how pressing that limitation
is; whether the enabling methods/data existed as of {as_of_date}; momentum in the evidence;
specificity (vague directions cannot be realized). A direction that was ALREADY accomplished
before {as_of_date} must be ranked last — it is not a future direction.

Rules: rank every candidate exactly once; no ties; do not reward eloquence or length.
Output only: a JSON list of candidate ids, best first, each with a one-sentence rationale:
[{"rank": 1, "id": "C3", "rationale": "..."}, ...]
```

## Notes
- Run B=3 times over shuffled candidate order, merge by within-agent RRF (Hou bootstrap, §R1).
- Malformed-output repair: append missing ids in prior order, drop duplicate ids, log repetition/missing/rejection counts (RankGPT protocol).
- The "already accomplished → rank last" line is the anachronism-distractor defense (§G5); smoke-test: fixture C6 must land last.
- For >30 candidates use the sliding window in `ranking/listwise.md` instead of one call.
