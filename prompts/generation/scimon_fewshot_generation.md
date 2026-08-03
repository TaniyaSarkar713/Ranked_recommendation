# Few-shot direction generation with retrieval (from SciMON, local code)

**Source [verbatim, extracted from `References/SciMON/models/GPTFS+NBR/fewshot3.py`]:** Wang et al., *SciMON* (ACL 2024). Their actual few-shot completion prompt, assembled as:

```
<5 few-shot examples sampled from train, each of the form:>
Consider the following context: {context} The retrieval results are: {retrieved_neighbors}
In that context, which {target_entity_type} can be used for {seed_entity}, and why?
{gold_relation_sentence}

<then the test instance:>
Consider the following context: {context} The retrieval results are: {retrieved_neighbors}
In that context, which {target_entity_type} can be used for {seed_entity}, and why?
```

Mechanics from the code: 5 exemplars randomly sampled per query (re-sampled until ≤1,600 tokens; fallback ≤1,024); forward template "which X can be used for Y" vs backward "which X do we use Y"; completion stopped at "."; `best_of=10`. Retrieval results = semantic-neighbor target terms (their §"inspiration retrieval").

## Our use

Candidate-generation source for **entity-anchored** directions (§G2): given a seed entity from the paper (a task, method, or metric) + retrieved future-work-bank neighbors, generate the `used-for` completion — a direction in problem→method form. Complements the free-form IdeaBench Fig-1 generator with structurally-typed candidates.

## Adapted prompt [adapted]

```
<3–5 exemplars from the future-work bank (train years only), each:>
Consider this paper's context: {exemplar_context}
Retrieved from the field: {exemplar_neighbors}
Q: In that context, which {entity_type} could future work apply to {seed_entity}, and why?
A: {exemplar_future_work_statement}

<test instance:>
Consider this paper's context: {limitations}
Retrieved from the field (as of {as_of_date}): {evidence}
Q: In that context, which {entity_type} could future work apply to {seed_entity}, and why?
A:
```

## Notes
- Seed entities come from the atomizer/entity extraction over the paper (PL-Marker in SciMON; we use the Fig-5 atomizer's tuples).
- Exemplars must predate {as_of_date} (temporal discipline, §G3).
- Output is one direction per (seed entity × entity type) pair — a wide, cheap sweep; dedup afterward (MiniLM 0.8).
