# Grounded candidate generation (from IdeaBench Fig. 1)

**Source:** Guo et al., *IdeaBench* (arXiv:2411.02429), Fig. 1 — researcher-profiled hypothesis generation grounded in reference abstracts.
**Our use:** candidate-pool source (a) in plan §3 — LLM brainstorm of future directions, grounded in the paper + retrieved evidence. Generation only — **never ranks** (candidate pool must stay ranking-free so rankers are the only ordering source).

## Original mechanism [summary of verbatim]

LLM is profiled as a domain researcher, given reference abstracts (Abstract 1…n): *"Using these abstracts, reason over them and come up with a novel hypothesis. Please avoid copying ideas directly, rather use the insights to inspire a novel hypothesis in the form of a brief and concise paragraph."* Low-resource = 5 filtered references; high-resource = all.

## Adapted prompt [adapted — forecast-framed candidate generation]

```
You are a senior researcher in {field}, reading in {as_of_date}. You have just finished
studying the paper summarized below, and you also have the following evidence from the
surrounding literature.

PAPER (future-work section removed):
{paper_summary}

EVIDENCE FROM THE FIELD (as of {as_of_date}):
{evidence}

Task: propose {n_candidates} candidate FUTURE RESEARCH DIRECTIONS that follow from this
paper — the concrete next steps a research group could pursue and publish.

Requirements:
- Each direction: ONE sentence, specific enough that a reader could tell whether a later
  paper realized it (name the method family, data/domain, or question — no vague "explore X further").
- Anchor each direction to a limitation or unexplored axis of the paper (cite its ID, e.g. L1).
- Cover different direction types: new-method, new-dataset/benchmark, domain-transfer,
  scale-up, theory/analysis, evaluation/robustness, deployment/efficiency.
- Do NOT rank, score, or order them. Do NOT copy the evidence sentences verbatim.

Output JSON: [{"id": "...", "direction": "...", "anchors": ["L1"], "type": "new-method"}, ...]
```

## Notes
- Run once per model family (GPT, Claude) — pool diversity, dedup at MiniLM cosine 0.8 (metrics.md D1).
- IdeaBench found reference *filtering* (≥5 citations, primary-only, background-cited) helps small models; apply the same filter when building `{evidence}`.
- Type list = collapsed heuristic taxonomy (§G2); one brainstorm pass per type is the coverage upgrade (§R3 note; ICLR-blogpost heuristics).
