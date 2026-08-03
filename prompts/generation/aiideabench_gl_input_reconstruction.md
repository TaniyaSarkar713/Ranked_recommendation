# Input reconstruction without leakage (from AI Idea Bench 2025, Appendix G–L)

**Source:** Qiu et al., *AI Idea Bench 2025* (arXiv:2504.14191), Appendix G (paper summary), I (clean topic), J (topic→ranked keywords), K (inspiration-paper preprocessing), L (sentence-level method-keyword split).
**Our use:** building the **masked task input** (§G3): what the system may see must not leak the masked future work; and building keyword/entity sets for matching (§G2).

## Original mechanism [summary]

Chain of prompts: summarize the target paper (G); strip method-specific detail to a clean topic (I) — e.g., *"novel Part Re-projection Distance Loss for 3D face reconstruction…" → "The topic of this paper is 3D face reconstruction"*; split topic into ranked keywords (J); keep only motivation + experimental procedures of inspiration papers (K); split motivation & experiment plan sentence-by-sentence into method keywords (L). Method names are **anonymized** — described by their steps, never named.

## Adapted prompts [adapted]

### (a) Leakage scrub of the paper body
```
The paper text below has had its Future Work section removed. Some future-work content may
still leak elsewhere (e.g., "in future work we will..." sentences in the conclusion, or
"ongoing work" footnotes). List every such sentence with its location, then output the text
with those sentences removed. Do not remove limitations — limitations stay.

TEXT: {paper_body}
Output JSON: {"leaked_sentences": [{"loc": "...", "text": "..."}], "clean_text": "..."}
```

### (b) Clean topic + keywords (Appendix I + J analog)
```
From the paper summary below, state its TOPIC in one sentence containing no method names
and no contribution claims — only the problem area. Then list up to 8 topic keywords,
ranked by importance, method names excluded.

SUMMARY: {paper_summary}
Output JSON: {"topic": "The topic of this paper is ...", "keywords": ["...", ...]}
```

### (c) Sentence-level keyword split for matching (Appendix L analog)
```
Split the following direction statement into its method keywords and concept keywords —
the minimal set of terms a later paper pursuing this direction would necessarily mention
or instantiate.

STATEMENT: {direction_statement}
Output JSON: {"method_keywords": [...], "concept_keywords": [...]}
```

## Notes
- (a) is our defense-in-depth on top of section masking — SciMON's trivial-overlap filter (§G3) then removes instances where gold is still surface-recoverable; report how many sentences (a) catches (leakage statistic for the paper).
- (c)'s output feeds entity soft/hard recall in the matching ladder (§G2, MOOSE-Chem2 style).
- Keyword sets are also the query for the KG features (SciMuse, metrics.md N3) — compute them **as of {as_of_date}** only.
