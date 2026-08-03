# Induced 5-level rubrics + criteria-induction meta-prompt (from ResearchAgent, Tables 12–15)

**Source:** Baek et al., *ResearchAgent* (arXiv:2404.07738), Table 12 (criteria) and Tables 13–15 (induced 5-level rubric texts). Their key method: **induce** detailed rubric anchors by showing an LLM ~10 human-scored idea pairs, so model scoring aligns with the team's actual preferences (their Fig. 5 shows the alignment shift).
**Our use:** (a) badge-dimension rubrics on Direction Cards; (b) the induction meta-prompt is our cheap judge-calibration tool (§E2) — run it on OUR ~10 dev-set human-scored direction pairs to get ResearchHorizon-specific rubric text.

## Original criteria (Table 12) [summary]

- **Problem:** Clarity, Relevance, Originality, Feasibility, Significance
- **Method:** Clarity, Validity, Rigorousness, Innovativeness, Generalizability
- **Experiment:** Clarity, Validity, Robustness, Feasibility, Reproducibility
Each with induced 5-level textual anchors (1 = worst … 5 = best).

## (a) Badge-scoring prompt [adapted — resource-conditioned feasibility per MLR-Copilot/Si anchors]

```
Score the direction below on the given dimension, 1–5, using the rubric.

PAPER: {paper_summary}
DIRECTION: {candidate_statement}
DIMENSION: {dimension_name} — {induced_rubric_text}

Anchor discipline: feasibility is judged for a typical PhD student with {resources}
(default: 1–2 months, ample API access, limited GPUs — Si et al. App. E convention).
Output JSON: {"score": 1-5, "evidence": "one sentence citing the paper/evidence"}
```

## (b) Criteria-induction meta-prompt [adapted from their method]

```
Below are {n} pairs of candidate future directions for papers, each pair scored by our
team's researchers on the dimension "{dimension_name}" (1–5, higher better), with the
scores shown.

{scored_pairs}

Induce a 5-level rubric for "{dimension_name}" that explains these judgments: for each
level 1–5, write 1–2 sentences describing what a direction at that level looks like,
using patterns you observe in the examples (not generic textbook definitions).
Output JSON: {"level_1": "...", "level_2": "...", "level_3": "...", "level_4": "...", "level_5": "..."}
```

## Notes
- Induction needs only ~10 scored pairs (ResearchAgent's own budget); refresh once after the E2 calibration set exists.
- Badge scores are display/feature-only — never the ranking objective (forecast framing rule).
- Agreement context when validating: ResearchAgent's human↔model pairwise κ reached 0.71 (problems) but only 0.52 (experiments) — expect the feasibility-ish dimensions to be noisiest.
