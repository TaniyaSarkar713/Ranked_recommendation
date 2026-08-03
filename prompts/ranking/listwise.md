# Listwise ranking prompts

**Sources:** Sun et al., *RankGPT* (EMNLP 2023) Appendix A.5 [verbatim core below], repo `github.com/sunnweiwei/RankGPT`; REARANK (arXiv:2505.20046, repo `github.com/lezhang7/Rearank`) reasoning variant; Hou et al., *LLMRank* (ECIR 2024, repo `github.com/RUCAIBox/LLMRank`) conditioning + bootstrap.
**Role:** the default single-call ranking inside each agent (§R1) when N ≤ ~30; sliding window beyond.

## RankGPT core [verbatim, condensed]

System: *"You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."* Passages injected as `[1] {passage}`, `[2] {passage}`, … Final instruction: *"Search Query: {query}. Rank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain."* Sliding window w=20, stride 10, applied back-to-first.

## Adapted prompt [adapted — forecast framing + REARANK reasoning]

```
SYSTEM: You are a research-strategy forecaster who ranks candidate future research
directions by how likely the community is to realize them.

USER:
It is {as_of_date}. PAPER (future work removed): {paper_summary}
EVIDENCE (as of {as_of_date}): {evidence}

CANDIDATES:
[1] {candidate_1}
[2] {candidate_2}
...
[{n}] {candidate_n}

Rank ALL {n} candidates by likelihood of impactful realization within {horizon_months}
months. Directions already accomplished before {as_of_date} go last. Do not reward
eloquence or length.

First reason briefly inside <think>...</think> (which limitations are most pressing; what
the evidence says; which candidates are vague, redundant, or already done).
Then output ONLY the ranking inside <answer>...</answer> as: [3] > [1] > ... > [7]
```

## Protocol

- **Bootstrap B=3** (Hou): rerun over shuffled candidate order, merge by within-agent RRF — non-negotiable; listwise collapses under adversarial order (RankGPT: 65.8 → 25.2 nDCG).
- **Windowing** for N > 30: window w=20, stride 10, back-to-first (RankGPT); or score-and-merge without windows if using calibrated scores (§L5).
- **Repair** (RankGPT App. F): drop duplicate ids, append missing ids in prior order, count repetition/missing/rejection per agent; report RBO between bootstrap runs as self-consistency.
- **Conditioning discipline** (Hou): keep {paper_summary} tight — their recency-focused trick translates to us as *"weigh the limitations list most heavily"*; more context ≠ better and popularity bias grows with fluffier context.
- <think> traces go to Direction-Card rationale candidates (shown, never trusted — plan §6).
