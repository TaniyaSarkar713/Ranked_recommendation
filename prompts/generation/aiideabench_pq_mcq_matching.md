# MCQ matching with hard negatives (from AI Idea Bench 2025, Appendix P–Q)

**Source:** Qiu et al., *AI Idea Bench 2025* (arXiv:2504.14191), Appendix P (motivation MCQ) & Q (experiment-plan MCQ) — the IMCQ metric's judge.
**Our use:** a cheap, discrete tier of the matching function (§G2) and the engine of the **distractor confusion matrix** (§G5): can a judge tell which gold direction a candidate corresponds to, among hard negatives?

## Original mechanism [summary]

Correct answer = the target's (motivation, experiment plan); distractors = the two influential papers most conceptually aligned with the target + the paper most similar to all targets. Judge instruction: *choose the closest of A–D, match on shared intent not keywords*, return `closest_option` + explanation. Success if any generated idea selects the true target.

## Adapted prompt [adapted — candidate→gold assignment]

```
A research paper (summarized below) had several follow-up directions actually pursued by
later work. You are given ONE candidate direction and FOUR possible gold directions (A–D).
Exactly one may be the direction the candidate corresponds to — or none may match.

PAPER: {paper_summary}
CANDIDATE: {candidate_statement}

GOLD OPTIONS:
A) {gold_A}   B) {gold_B}   C) {gold_C}   D) {gold_D}

Rules: match on shared research INTENT (same problem being solved the same general way),
not on keyword overlap. Superficially similar wording with a different research object is
NOT a match. If no option shares the candidate's intent, answer "none".

Output JSON: {"closest_option": "A|B|C|D|none", "explanation": "..."}
```

## Distractor-set construction (per candidate)
- 1 true gold (if any) + hard negatives drawn from our G5 tiers: an **SSLI** direction (same subfield, different paper's future work), a **counterfactual** direction (contradicts the paper's findings), an **anachronism** (realized before t₀).
- Report per-tier confusion rates — that table IS the §G5 deliverable.

## Notes
- Use a third-family judge (not GPT/Claude — §E2 judge-affinity).
- "None" option is essential: candidates with no gold match are the majority; forcing a choice inflates false matches (AI Idea Bench's own IMCQ lacks "none" — our addition).
- Pairs with the graded Matched-Score tier (metrics.md C6): MCQ = cheap assignment; Matched-Score 0–5 = graded strength on the assigned pair.
