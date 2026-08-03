# Iterative novelty boosting / dedup-against-literature (from SciMON §3.3)

**Source:** Wang et al., *SciMON* (ACL 2024), §3.3 — retrieve-compare-update loop: retrieve the k=20 nearest prior-work ideas to the current draft; for every retrieved item with cosine ≥ μ=0.6, feed it back as a negative and rewrite; stop when all similarities < μ.
**Our use:** NOT as a novelty optimizer (forecast framing doesn't reward novelty per se) — repurposed as (a) **near-duplicate collapse** within the candidate pool and (b) **"already-done" detection** against the pre-t₀ literature, which is exactly the anachronism-distractor defense (§G5) applied at generation time.

## Original feedback line [paraphrase of verbatim]

*"Your idea has similarities with existing research… Make sure the idea you suggest is significantly different… Let's give it another try."*

## Adapted prompt [adapted — already-done check + differentiation]

```
Candidate future direction (proposed for a paper published {as_of_date}):
{candidate_statement}

Closest prior work found BEFORE {as_of_date} (cosine similarity in brackets):
1. [{sim_1}] {prior_title_1}: {prior_snippet_1}
2. [{sim_2}] {prior_title_2}: {prior_snippet_2}
3. [{sim_3}] {prior_title_3}: {prior_snippet_3}

Decide, for each prior item: does it ALREADY ACCOMPLISH this direction (not merely relate
to it)? Match on research intent, not keywords.

Output JSON:
{"verdicts": [{"prior": 1, "already_done": true|false, "why": "..."}, ...],
 "candidate_status": "novel_direction | already_done | needs_sharpening",
 "sharpened_statement": "only if needs_sharpening: rewrite to differ from the prior work
                         while still addressing the same limitation"}
```

## Notes
- Retrieval gate: SentenceBERT/SPECTER cosine ≥ 0.6 triggers the check (SciMON's μ; recalibrate on our dev set — metrics.md D2 warns μ is corpus/embedder-specific).
- `already_done` candidates are NOT discarded — they become labeled **anachronism distractors** for evaluation (§G5), free hard negatives.
- Cap at 2 sharpening iterations (SciMON's own study: gains flatten; Si et al.: overgeneration saturates ~5% unique anyway).
