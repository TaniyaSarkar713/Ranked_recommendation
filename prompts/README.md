# prompts/ — ResearchHorizon prompt library

*Deliverable for `to-do.md` §2–3 (Taniya + Naz). Every prompt is adapted from a cited source to our task: given a paper (future-work masked), generate and RANK candidate future research directions. Section refs (G/R/F/E/L) → `ResearchHorizon_ideas_Q1-Q4.md`; formulas → `metrics.md`.*

## The one rule: forecast framing (§R3)

Every ranking/judging prompt asks the **forecast question**, never the aesthetic one:

> "Given this paper's limitations and the state of the field as of {as_of_date}, which of these directions is the research community **most likely to pursue next and publish impactfully within {horizon_months} months**?"

Why: HindSight found LLM-judged *novelty* anti-correlates with actual realization (ρ = −0.291, p=0.003) while judged *feasibility* is the only positively-correlated dimension (ρ = +0.252). "Which is most novel/interesting?" prompts are anti-predictive — banned outside explicitly-labeled badge scoring.

## Shared template slots (used across all files)

| Slot | Content |
|---|---|
| `{paper_summary}` | Agent-1 structured JSON (problem, methods, assumptions, limitations, unexplored axes) — plan §2 |
| `{limitations}` | the limitations/gaps list alone (highest-signal conditioning; keep short — Hou: less context = less popularity bias) |
| `{as_of_date}` | the paper's publication month/year t₀ — the ranker must reason *as of then* |
| `{horizon_months}` | realization window Δ (default 24–30) |
| `{candidates}` | numbered list of candidate direction statements (normalized by the atomizer prompt) |
| `{evidence}` | retrieved kNN future-work statements + KG stats as of t₀ (§R3 evidence conditioning) |
| `{quality_indicator}` | scoring dimension + its definition (default: expected realization) |

## Directory map

```
prompts/
  README.md                ← this file
  example_input.md         ← shared smoke-test fixture (1 paper summary + 6 candidates incl. planted distractors)
  generation/              ← task-side prompts (generate, normalize, match, judge)
    ideabench_fig1_grounded_generation.md
    ideabench_fig2_listwise_quality_ranking.md
    ideabench_fig5_direction_atomizer.md
    aiideabench_gl_input_reconstruction.md
    aiideabench_pq_mcq_matching.md
    aiideabench_r_pairwise_judge.md
    scimon_fewshot_generation.md
    scimon_novelty_boost.md
    coi_table24_pairwise_judge.md
    researchagent_rubric_criteria.md
  ranking/                 ← the four paradigms + Swiss scheduler
    pointwise.md  pairwise.md  setwise.md  listwise.md  swiss_scheduler.md
    README.md              ← complexity / order-robustness / logprobs table
```

## Status

- [x] All prompts extracted/adapted with source attribution (SciMON prompts read from the actual local code in `References/SciMON/models/GPTFS+NBR/`).
- [x] Template-instantiation check against `example_input.md` (slots resolve, output formats parseable).
- [ ] **Live API smoke-test pending** — needs OpenAI/Anthropic keys; run each prompt once on the fixture and paste outputs into the file's "smoke-test log" footer.

## Attribution note

Prompt texts marked **[verbatim]** are quoted from the cited paper/repo for research use; **[adapted]** marks our forecast-framed rewrites. Keep both blocks when editing so provenance survives.
