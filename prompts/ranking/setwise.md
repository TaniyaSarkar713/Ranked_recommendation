# Setwise ranking prompts

**Sources:** Zhuang et al., *Setwise* (SIGIR 2024), Fig. 1d [verbatim below]; reference impl `github.com/ielab/llm-rankers` (setwise.heapsort / setwise.bubblesort).
**Role:** the per-call unit of the **Swiss-Setwise** tournament (§R2) and the budget option for top-k extraction (setwise.heapsort = −62% cost vs pairwise-heapsort at −0.8% nDCG).

## Original template [verbatim, passage ranking]

```
Given a query {query}, which of the following passages is the most relevant one to the query?
Passage A: {passage_A}
Passage B: {passage_B}
Passage C: {passage_C}
Output only the passage label of the most relevant passage:
```

## Adapted template [adapted — group ranking, forecast framing]

```
It is {as_of_date}. PAPER (future work removed): {paper_summary}
EVIDENCE (as of {as_of_date}): {evidence}

Below are {c} candidate future directions. Rank them by how likely the research community
is to pursue and publish impactfully on each within {horizon_months} months.
A direction already accomplished before {as_of_date} ranks last automatically.

A: {candidate_1}
B: {candidate_2}
C: {candidate_3}
D: {candidate_4}
E: {candidate_5}

Output only the labels, best first, e.g.: C > A > E > B > D
```

## Two output modes

- **Generation mode** (works on Claude + GPT): parse the label order; malformed → repair (append missing labels in prior order) and log.
- **Logits mode** (GPT/open only, §L1): ask for THE BEST label only ("Output only the single best label:"), read top-20 first-token logprobs over labels → a **soft distribution over the whole group in one call** (Zhuang: no effectiveness loss). Iterate masking the winner for a full soft order (PL-style). Feeds soft-Swiss points (§L2).

## Rules of thumb (from the paper + our task shape)

- Labels must be single tokens (capital letters; verify with tokenizer). Rotate candidate→label assignment across bootstrap rounds to cancel label priors (§L6).
- c: their sweet spot was 3–9 with truncation pressure; our direction statements are one sentence, so **c = 5–10 is safe** — tune c on dev MRR jointly with Swiss rounds R.
- Robust to within-set order by construction; still shuffle group membership order per round (hygiene).
- Early-stopping: heapsort variants can stop once top-k extracted — relevant if we only ever show top-10 Direction Cards.
