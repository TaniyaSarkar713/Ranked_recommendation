# Idea-Arena pairwise judge, per-criterion 0/1/2 (from Chain of Ideas, Tables 24 + 5)

**Source:** Li et al., *Chain of Ideas* (arXiv:2410.13185), Table 24 (judge prompt) + Table 5 (criteria definitions, adopted from ICML-2020 review guidelines + Si et al.).
**Our use:** system-level evaluation battles (comparing OUR pipeline variants' output lists — the Idea-Arena role), and human-study instrument wording (§E5). For per-candidate tournament ranking use `aiideabench_r_pairwise_judge.md` instead (forecast aspects).

## Original mechanism [summary of verbatim]

*"You are a judge… decide which idea is better… avoid position bias… DO NOT let the LENGTH of the responses influence you… output 0/1/2 (idea0 better / idea1 better / equal) for each criterion: Novelty, Significance, Clarity, Feasibility, Expected Effectiveness."* Each pair judged twice with order reversed; wins/ties → Elo (Chatbot-Arena update). Human↔GPT-4o agreement: 70.8% average (per-criterion 66.5–76.3%).

## Adapted prompt [adapted — for ranked-LIST battles between system variants]

```
You are comparing two RANKED LISTS of recommended future research directions for the same
paper, produced by two anonymous systems. Judge which list better serves a researcher
deciding what to work on next.

PAPER: {paper_summary}

LIST 0 (top-{k}): {list_0}
LIST 1 (top-{k}): {list_1}

For each criterion output 0 (List 0 better), 1 (List 1 better), or 2 (equal):
- head_quality: are the TOP-ranked directions the most pursuable, well-grounded ones?
- coverage: does the top-{k} span distinct limitation-anchored themes (vs near-duplicates)?
- specificity: are directions concrete enough to act on and to verify later?
- grounding: does each direction trace to the paper's limitations/evidence (no fabrication)?
- overall.

Avoid position bias (list order is random). Do NOT let list length or eloquence influence
you. Judge the ranking, not the prose.

Output JSON: {"head_quality": 0|1|2, "coverage": ..., "specificity": ..., "grounding": ..., "overall": ..., "rationale": "..."}
```

## Notes
- Keep CoI's original five idea-criteria (Novelty/Significance/Clarity/Feasibility/Effectiveness, Table 5 definitions) for the **badge/human-study** variant; the list-battle criteria above are ranking-appropriate.
- Both orders, always; Elo or BT aggregation across papers; report judge↔human agreement with the 70.8% CoI number as context (§E2).
- This is the ONLY place LLM judgment compares whole lists — the benchmark metric stays realization-grounded MRR/nDCG; arena battles are the secondary, human-anchored view.
