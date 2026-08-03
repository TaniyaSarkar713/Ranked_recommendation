# Direction atomizer / gold normalization (from IdeaBench Fig. 5)

**Source:** Guo et al., *IdeaBench* (arXiv:2411.02429), Fig. 5 — LLM extraction of a hypothesis-style paragraph from a target abstract, stripping result-leaking specifics so gold and generated ideas live in one representation space.
**Our use:** THE normalization step for ground truth (§G1/G2): split a masked future-work section (or a forward-citing paper's contribution statement) into **atomic, canonical direction statements + entity tuples**. Same prompt normalizes candidates before matching, so `match(d, g)` compares like with like.

## Original mechanism [summary of verbatim]

*"Write a concise paragraph summarizing the following biomedical paper abstract as if you are proposing your own research idea or hypothesis. Focus on the main research idea and a high-level summary of the findings without detailed results or specific numerical data. Please begin the paragraph with 'Hypothesis:' or 'Given that.'"*

## Adapted prompt [adapted — atomize + entity tuples, §G2 schema]

```
You will be given a FUTURE-WORK section (or a citing paper's statement of what it did
relative to a prior paper). Split it into ATOMIC research directions.

TEXT:
{future_work_section}

Rules:
- One direction = one pursuable research step (if a sentence contains two distinct steps,
  split it; if two sentences restate one step, merge them).
- Canonical form: one sentence, present tense, no results/numbers, no citations, no hedging.
- Anonymize bespoke method names to their method family (AI-Idea-Bench-style anonymization:
  "our XYZ-Loss" -> "a reprojection-based loss"), keep public method families as-is.
- For each direction extract the entity tuple.

Output JSON list:
[{
  "statement": "...",
  "direction_type": "new-method | new-dataset | domain-transfer | scale-up | theory | evaluation | deployment",
  "problem_entity": "...",
  "method_entity": "...",
  "data_domain_entity": "...",
  "verbatim_span": "the exact source words this direction came from"
}]
```

## Notes
- `verbatim_span` keeps the evidence link for Direction Cards and for auditing the atomizer (E2 human calibration set scores atomizer output too).
- Run with a **third-family model** when used for ground truth (judge-affinity control, §E2) — gold construction must not share a model family with the rankers.
- The same prompt with `{future_work_section}` = a candidate's raw text normalizes candidates; matching then proceeds type→entities→embedding→Matched-Score (metrics.md C6).
