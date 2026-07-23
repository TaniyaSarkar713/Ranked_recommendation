# ResearchHorizon — Novel-but-Feasible Ideas for Q1–Q4

*Synthesized from all 24 papers in `References/` + `plan.pdf` (v2). Each idea is tagged **[Novel]** (not published in this combination — a claimable contribution), **[Adapt]** (published elsewhere, new to this task), or **[Adopt]** (established; use as-is and cite). Evidence citations refer to the reference papers.*

---

## 0. The white space (why this paper is publishable)

Reading the full reference set confirms the gap the plan targets, and sharpens it:

| Prior art | What it does | What it does NOT do |
|---|---|---|
| IdeaBench (Guo 2024) | Scores generated ideas vs. one gold idea (Insight Score = relative rank of the real paper's idea) | No ranked recommendation list, no MRR/nDCG, single LLM ranker, no fusion |
| AI Idea Bench 2025 (Qiu 2025) | Held-out anonymized (motivation, experiment plan) as gold; forward-citation novelty score; round-robin tournament across *systems* | Max-over-ideas matching only; never ranks *multiple directions per paper*; no rank-aware metric |
| HindSight (Jiang 2026) | Time-split, impact-grounded scoring of ideas (0.6·citations + 0.4·venue over matched post-cutoff papers) | One scalar per idea; topic-wide pool (not forward citations); no ranking, no multi-agent |
| FutureGen (cited in plan) | Generates future-work text | Free-form generation scored by overlap, not ranked recommendation |
| LLM ranking literature (PRP, Setwise, RankGPT, SCaLR, REARANK) | Mature ranking paradigms for passage retrieval | Never applied to future-direction recommendation; fusion across heterogeneous LLM rankers barely studied |

**The claimable intersection:** *ranked* recommendation of future directions, against a *realization-graded, multi-source* ground truth, produced by *heterogeneous LLM ranking agents* whose outputs are *fused with context-conditioned, MRR-calibrated* methods, and evaluated with *rank-aware + robustness + prospective* protocols. No reference paper occupies this.

Two findings from the literature make the framing defensible against the obvious reviewer attack ("why not just ask GPT-4 to rank?"):

1. **LLM judgment is anti-predictive of realization.** HindSight: LLM-judged novelty correlates *negatively* with whether an idea was actually pursued (ρ = −0.29, p < 0.01); LLM-judge overall couldn't distinguish a grounded system from a vanilla baseline (p = 0.584) while realization-grounded scoring showed a 2.5× gap (p < 0.001).
2. **Single LLM rankers are noisy annotators.** Si et al.: every LLM evaluator tested fell below the (already low) 56.1% human split-half agreement; their human-rerank condition beat their LLM-ranker condition on every metric.

→ This is the citation backbone for "ground truth must come from realization, and ranking must come from calibrated fusion of multiple noisy agents, not one LLM's opinion."

---

## Q1. Ground-truth ranked recommendations

### G1. Two-source, graded "realization ladder" labels — **[Novel]** (the core Q1 contribution)

For each test paper P (future-work section masked), build the gold set G(P) from **two label sources**, then grade every gold direction on an ordinal **realization ladder**:

```
Grade 0  — distractor / never realized (see G5)
Grade 1  — stated in P's masked future-work section, but no realizing paper found in window Δ
Grade 2  — realized: ≥1 post-P paper matches the direction (any provenance)
Grade 3  — realized with impact: matched paper(s) clear an impact bar
           (field- & age-normalized citation percentile, venue tier, or ≥k independent groups)
```

- **Source A (stated intent):** atomic directions split from the masked future-work text (plan §1).
- **Source B (revealed behavior):** directions extracted from *forward-citing* papers' intros/contributions ("Unlike [P], we…"). ILCiteR's evidence-bank construction is the template: store `direction-span → {realizing papers, support count}` where **support count = # independent papers that pursued it**.
- A direction can be Grade 2–3 *without ever being stated* in the future-work section (the community did it anyway) — this is exactly what makes the benchmark harder than FutureGen-style text overlap and is your cleanest novelty claim vs. IdeaBench/AI Idea Bench (both hold out a *single* gold idea).

**Impact score for Grade 3 (rank *within* grades):** adopt HindSight's recipe with two fixes flagged by its own limitations:
- `impact(g) = α·cit-percentile(matched papers) + β·top-venue + γ·log(1+support-count)`, with **citation percentiles normalized by field and paper age** (VirSci's year-mean normalization — otherwise older test papers get inflated impact) and support counts from ILCiteR-style distinct-group counting.
- Aggregate over matched papers with **max + count tie-break** rather than pure max (HindSight admits max is brittle to one lucky near-duplicate).
- **Time-to-realization** `lag(g)` = months from P's publication to the first realizing paper, as an *auxiliary* label: short-lag directions are "imminent next steps" (arguably what a recommender should surface first). Report MRR both unweighted and lag-discounted. No prior work uses realization lag — cheap and genuinely new.

**Why grades, not a total order:** MOOSE-Chem2's novelty-vs-effectiveness trade-off and ListNet's theory both argue against scalarizing early. Graded labels support nDCG directly, and can be converted to **soft Plackett–Luce target distributions** (ListNet top-one probabilities) when several directions are "equally realized" — no arbitrary tie-breaking. This also gives the tensor-fusion stage a principled training target (ListNet/ListMLE loss on PL targets, select by dev MRR — consistent with plan §5.4).

> #### Math box: from graded labels to metrics and training targets
>
> **The tie problem.** A paper's gold set is grades, not an ordering. Example: `d1=3, d2=2, d3=2, d4=1, d5=0, d6=0`. The data says d1 belongs above d2/d3, but contains **zero evidence** on d2 vs. d3 — both realized, comparably. Any method demanding a single gold permutation (ListMLE's target permutation, or pairwise losses over all pairs) forces an invented d2>d3 or d3>d2; the model is then penalized for disagreeing with a coin flip. That invented order is pure label noise.
>
> **(a) Evaluation: nDCG consumes grades natively.**
> `DCG@K = Σ_i gain(g_i) / log2(i+1)`, with `g_i` = grade of the item the *system* placed at rank i and `gain(g) = 2^g − 1`; `nDCG = DCG / IDCG`, where IDCG sorts by grade descending (any within-grade order gives the same IDCG — the metric is well-defined without breaking ties). The achieved score is also **invariant to swapping same-grade items** (equal gains at exchanged positions sum identically).
> Worked example — system output `[d2, d1, d3, d5, d4, d6]`, discounts `[1, .631, .5, .431, .387, .356]`:
> ```
> DCG  = 3·1 + 7·.631 + 3·.5 + 0 + 1·.387 + 0 ≈ 9.30
> IDCG = 7·1 + 3·.631 + 3·.5 + 1·.431 + 0 + 0 ≈ 10.82   →  nDCG ≈ 0.86
> ```
> Swapping d2↔d3 in the output leaves 0.86 unchanged. Unlike MRR (binary first-hit above a grade threshold), nDCG also rewards putting Grade 3 above Grade 2 — the reason to grade impact at all.
>
> **(b) Training: soft Plackett–Luce targets instead of a gold permutation.**
> The fusion layer (meta-ranker / tensor) needs a differentiable loss (plan §5.4). ListNet: map ground-truth scores y and model scores s each to a **top-one distribution** and minimize cross-entropy:
> ```
> P_y(j) = exp(y_j/τ) / Σ_k exp(y_k/τ)        # target, y_j = grade (or gain)
> P_s(j) = softmax over fusion scores s(d_j|p)  # model
> L(p)   = − Σ_j P_y(j) · log P_s(j)            # per-paper loss
> ```
> "Top-one probability" is the Plackett–Luce marginal P(item j ranked first) — computable in O(n) although the full PL distribution lives over n! permutations (ListNet Thm 6). With grades `[3,2,2,1,0,0]` and τ=1:
> ```
> P_y ≈ [0.51, 0.19, 0.19, 0.07, 0.025, 0.025]
> ```
> d2 and d3 get **identical target mass** — the tie is represented exactly, never resolved. The loss optimum scores them equally; asymmetry in either direction is penalized symmetrically. That is what "no arbitrary tie-breaking" means operationally.
>
> **Knobs.**
> - **Temperature τ:** τ→0 concentrates the target on the top grade (approximates MRR's first-hit emphasis); large τ flattens toward uniform. Tune on dev MRR — τ is "how much training cares about depth beyond rank 1."
> - **Coarse grades vs. continuous gains:** using the continuous impact score as y removes all ties but bakes impact-measurement noise (citations, venue proxies) into the target. Recommendation: **coarse grades for PL training targets** (robust), **continuous-gain nDCG as a secondary evaluation column** (MOOSE-Chem2: don't scalarize early).
>
> **Why listwise rather than pairwise for this loss:** ListNet's own TREC experiment shows pairwise loss diverging from list metrics — RankNet's loss kept decreasing while NDCG@5 *dropped* after ~60 epochs. When the selection criterion is a list metric (dev MRR), the surrogate should be listwise.
> **Family unification bonus:** Plackett–Luce is the list generalization of Bradley–Terry, so the tournament win matrices (F1/F2) and these graded soft targets live in one probabilistic family — pairwise match evidence and full-list targets can share a single likelihood.

### G2. Entity-anchored atomic directions + typed hierarchical matching — **[Adapt]** (answers "Entity-based")

Represent every direction (gold and candidate) as a normalized tuple, extracted by an LLM in one pass (IdeaBench Fig-5-style normalization prompt + SciMON's `used-for` relation schema):

```json
{"direction_type": "new-method | new-dataset | new-domain-transfer | scale-up | theory | evaluation | deployment",
 "problem_entity": "...", "method_entity": "...", "data/domain_entity": "...",
 "statement": "one-sentence canonical form"}
```

- `direction_type` comes from a fixed taxonomy — seed it with the ICLR-blogpost's 20 ideation heuristics collapsed to ~7 types (H9 assemble-dataset, H10 new-technique, H19 analogical-transfer, H13 challenge-assumption, …). Typing enables **stratified matching** (match within type first) and per-type realization base rates (datasets get built more often than assumptions get challenged — a debiasing statistic no prior benchmark reports).
- **Hierarchical match function** `match(d, g)` for MRR (upgrade of plan §1): (1) type agreement → (2) entity soft/hard overlap (MOOSE-Chem2's soft/hard component recall — report both strictness levels) → (3) embedding cosine ≥ τ (SPECTER2) → (4) entailment judge ("does d cover g?"). A match requires (1)+(2) plus (3 or 4).
- **Why this beats cosine-only matching:** SciRerankBench shows the dominant scientific-domain failure is *semantically-similar-but-logically-irrelevant* (SSLI) confusion; entity+type gates kill exactly those false positives. HindSight's cosine band (0.91–0.98 for all AI/ML text) shows how narrow the pure-embedding signal is.

### G3. Dual-clock temporal protocol with a contamination ledger — **[Adopt/Adapt]**

Two clocks, explicitly separated in the paper:
- **Corpus clock:** P's publication date t₀; realization window [t₀, t₀+Δ]. Pick Δ = 24–30 months (HindSight used 30). Report label prevalence vs. Δ.
- **Model clock:** every ranker/judge LLM's training cutoff. The **clean test slice** = papers with t₀ after all cutoffs (IdeaBench's design: all targets post-cutoff). For pre-cutoff papers, run VirSci's Appendix-A control: swap in a post-cutoff-only realization pool and verify conclusions are stable; report a per-paper contamination flag.
- **Trivial-overlap filter** (SciMON): drop instances where the masked future-work is recoverable from the paper body by surface similarity (they used ≤10th-percentile cosine as the *keep* criterion for the challenge set). Otherwise the benchmark degenerates into paraphrase retrieval. Report headline numbers on both the full and the filtered "challenge" slice.
- **Reference filtering** for the future-work bank (IdeaBench): ≥5 citations, primary research only — keeps the retrieval corpus clean.

### G4. Ground-truth novelty/feasibility as *secondary channels*, not the primary rank — **[Adapt]** (answers "use novelty/feasibility/insight scores to rank?")

Recommendation: **do not fold LLM-judged novelty/feasibility into the primary ground-truth ranking.** Evidence: HindSight's negative novelty↔realization correlation; Si et al.'s LLM-judge unreliability; IdeaBench's Insight Score is circular for our purpose (it measures whether generated ideas *out-rank* the gold idea under an LLM judge — the judge is the thing we distrust).

Instead:
- **Primary axis** = realization grade + impact (objective, time-grounded — G1).
- **Secondary channels**, reported per-dimension and shown on Direction Cards:
  - *Feasibility*: AI Idea Bench's citation-influence recipe (decompose direction into concepts, retrieve supporting methods, `Inf = 1 − e^(−citations/50)`) — non-LLM, cheap, and HindSight found feasibility is the *only* LLM dimension positively correlated with realization (+0.25), so it doubles as a sanity signal.
  - *Novelty*: knowledge-graph white-space signals (SciMuse): concept-pair degree/PageRank/citation-velocity computed *as of t₀* — low-degree, adjacent-but-unconnected concept pairs are what 100 research-group leaders found interesting (their supervised model on graph features alone: top-1 precision 70%). This is your "quantitative novelty" — auditable and time-stamped, no LLM opinion involved.
- If a single composite is demanded, calibrate the weights on the dev set against realized impact — but keep the benchmark's headline metric realization-based.

### G5. Distractor tiers with known-negative labels — **[Novel]** (evaluation hardening; SciRerankBench transplanted to ideation)

Inject controlled negatives into candidate pools so ranking metrics can't be gamed by topical fluency:

1. **SSLI distractors:** future-work statements harvested from *other* papers in the same subfield (topically fluent, wrong paper) — the future-work bank provides these for free.
2. **Counterfactual distractors:** directions that contradict P's stated findings/assumptions (generated + human-spot-checked).
3. **Anachronism distractors** — **[Novel]**: directions *already realized before t₀* (mined from P's *backward* citations). A competent recommender must rank "already done" below "genuinely next." No existing benchmark tests this, it's nearly free to build from the citation graph, and it directly measures whether rankers know the state of the art vs. merely sounding plausible.
4. **Random off-topic** (NC-style) as the easy floor.

Report a **distractor confusion matrix** (how much probability mass each ranker puts on each tier) alongside MRR. This is a compact, reviewer-friendly robustness table.

### G6. Label-efficiency multiplier — **[Adapt]** (REARANK's trick, for training the fusion layer)

From each labeled paper, **sample many candidate subsets/orderings** and reuse the graded labels → hundreds of fusion-training instances from ~179-paper-scale annotation (REARANK: 179 queries → 12k instances). Directly de-risks the "tensor is data-starved" concern in plan §12.

**Q1 pitfalls to state honestly in the paper:** (i) realization ≠ merit — the community pursuing a direction is popularity-tinged evidence (conformity bias, admitted by HindSight); (ii) window censoring — Grade 1 directions may be realized after Δ (report grade prevalence vs. Δ); (iii) forward-citation coverage varies by field. All three are measurable and reportable, which is itself a contribution.

---

## Q2. Ranking inside the LLM agents

### The literature's cost/quality/robustness map (what to baseline)

| Paradigm | Calls (N cands) | Order-robust? | Key numbers |
|---|---|---|---|
| Pointwise 0–1 scoring | O(N) | yes, but needs calibration | weakest; Si et al.: near-chance as judge |
| Pairwise all-pairs (PRP) | O(N²) | **best** (72.42→72.40 under inversion) | both-order swap + tie protocol |
| Setwise heapsort (c-way) | O(k·log_c N) | yes | −62% cost vs pairwise-heapsort, −0.8% nDCG |
| Listwise single-call / sliding window (RankGPT) | O(N/s) | **no** (65.8→25.2 under shuffle) | best quality on frontier models |
| Listwise + shuffled bootstrap (Hou) | O(B·N/s) | recovered | B=3 shuffles, merge |

With ~50–200 candidates (plan §3), a **single-call listwise ranking is one window** for ≤30 candidates and 2–3 windows otherwise — cheap enough that robustness, not cost, is the binding constraint.

### R1. Default per-agent recipe (the safe baseline) — **[Adopt]**

Each agent (GPT, Claude) runs: **listwise ranking over shuffled candidate order, B=3 bootstrap rounds, merged by within-agent RRF** (Hou's bootstrapping), with RankGPT's malformed-output handling (append missing IDs in prior order, drop duplicates, count repetition/missing/rejection rates) and RBO self-consistency reported per agent. Zero-shot; Si et al. found few-shot/CoT gave no gains for pairwise judging.

### R2. Swiss-Setwise tournament ranking — **[Novel]** (the Q2 method contribution)

Generalize Si et al.'s Swiss tournament (pairwise, +1 per win, N=5 rounds) by replacing pairs with **setwise groups of c**:

1. Seed scores from the retrieval prior (or round-1 listwise pass).
2. Each round: sort by accumulated score, partition into **score-adjacent groups of c**, one LLM call per group returns a within-group ranking (Setwise prompt); candidates earn points by within-group placement (e.g., c−rank).
3. Repeat R rounds with rematch avoidance; final ranking by total points; ties broken by prior.

- **Cost:** O(R·N/c) calls — with N=50, c=5, R=4: 40 calls/agent vs. 2,450 for all-pairs.
- **Interpolation property:** c=2 → Si et al.'s Swiss; c=N, R=1 → single-call listwise. R and c are tuned on dev MRR (Si et al. tuned rounds the same way and found non-monotonicity — gap peaked at N=5, fell at N=6 — so tuning is genuinely necessary, not decoration).
- **Why it should win:** it spends comparisons where the current standings are uncertain (adjacent scores) — the information-efficiency argument formalized in F2 below — while inheriting setwise's order-robustness and generation-API compatibility (no logits needed).
- Nobody in the reference set (or, to my knowledge, elsewhere) has combined Swiss scheduling with setwise group calls; it's a small, clean, ablatable algorithmic contribution.

### R3. Forecast-framed, evidence-conditioned ranking prompts — **[Novel]** (cheap, high-value ablation)

Reframe the ranking question inside every agent from *aesthetic judgment* to *prediction*:

> "Given this paper's limitations and the state of the field in {month, year}, which of these directions is the community **most likely to pursue next and publish impactfully**?"

versus the standard "which is most novel/promising?" framing. Grounding: HindSight found LLM-judged *novelty* anti-correlates with realization (ρ=−0.29) while *feasibility* positively correlates (+0.25); Hou found conditioning on less, more-focused context reduces popularity bias. Condition each call on Agent-1's structured summary (limitations, assumptions) + retrieved evidence (kNN future-work statements, KG stats as of t₀) — VirSci showed retrieval-grounded, memory-free judging beats parametric-only judging, and blind (no cross-agent chatter) voting avoids conformity.

**The ablation nobody has run:** {novelty-framed vs. feasibility-framed vs. forecast-framed} × {with vs. without evidence} on dev MRR. Four cells, one table, directly publishable as "prompt objective matters more than prompt paradigm."

### R4. Hierarchical theme→direction ranking — **[Adapt]** (MOOSE-Chem2's coarse-to-fine, as a ranking strategy)

Cluster candidates into ~5–8 themes (embedding clustering or the G2 type tags); rank themes in one listwise call; then rank within-theme candidates setwise; compose final list by calibrated interleaving. Justification: MOOSE-Chem2's hierarchy-as-reward-smoothing argument (coarse judgments are reliable where fine-grained long-list judgments are rugged: HHS beat greedy 76.5% by expert judgment), plus a free diversity benefit — top-K spans themes, which directly helps MRR when a paper has several true directions in different areas (plan improvement #3, without the MMR hack).

### R5. Do NOT fine-tune rankers for the hackathon paper — **[Adopt]** (scope control)

REARANK shows RL-trained 7B rankers can match GPT-4 (from 179 labeled queries), and SCaLR shows calibrated-listwise fine-tunes beat zero-shot — cite both as the scale-up path (plan §4 already lists losses), but the paper's claims should rest on prompt-only agents + fusion. If reviewers demand a learned component, the meta-ranker/tensor already is one, and R2's tournament scores give it clean features.

---

## Q3. Rank fusion — and where the Swiss tournament actually pays off

### The fusion ladder (keep plan's structure; sharpen the story)

`RRF (no training) → Bradley-Terry/Plackett-Luce pooled-evidence fusion → learned meta-ranker (LightGBM) → context-conditioned low-rank tensor (+CMTF)` — each rung ablated on identical candidates, selected by dev MRR (plan §10's decisive rows). The literature adds three upgrades:

### F1. Fuse *evidence*, not just *ranks* — **[Adapt]**

RRF throws away everything except rank positions. The agents' bootstrap rounds (R1) and tournament match outcomes (R2) form a **win/preference matrix pooled across both agents** — exactly the sufficient statistic for **Bradley–Terry/Plackett–Luce strength fitting** (PRP-Allpair's Copeland scores + Shah & Wainwright's win-ratio guarantees; ListNet's PL bridge from pairwise wins to full rankings). Weight each agent's votes by its **dev-set solo MRR** (quality-weighting — MOOSE-Chem2: a weak committee member actively drags down the optimum; SciMuse: GPT-4o ranks strictly better than GPT-3.5; Sun: 2023-era Claude trailed GPT-4 by ~10 nDCG as a listwise ranker, so uniform weights are provably wrong). BT/PL fusion slots between RRF and the meta-ranker in the ablation ladder, and its per-candidate strength ± uncertainty feeds Direction-Card confidence and abstention.

### F2. Disagreement-routed Swiss fusion — **[Novel]** (the direct answer to "Swiss tournament: are there advantages?")

**Where Swiss fits best is not as a replacement for RRF/tensor — it's as a *compute-adaptive refinement stage* that spends new LLM calls only where the fused ranking is uncertain:**

1. Compute the free fusion first (RRF or tensor) from the agents' existing lists.
2. Identify the **contested zone**: adjacent candidates whose cross-agent rank disagreement is high (rank-inversions between GPT and Claude lists) and whose fused scores are within ε — i.e., pairs the current fusion effectively decided by coin-flip.
3. Run **Swiss rounds only over the contested zone**, each match judged by *both* agents with evidence (R3 prompts), order-swapped (PRP protocol); update fused scores via BT.
4. Stop when the top-K stabilizes (anytime property) or the budget is exhausted.

**The advantages, stated precisely (this is the paragraph for the paper):**
- **Information efficiency.** A Bradley–Terry comparison's Fisher information is p(1−p), maximized when p≈0.5 — i.e., between candidates of similar strength. Swiss pairing (similar accumulated scores) approximately implements maximum-information pair selection without any learned model; round-robin wastes most of its O(N²) budget on foregone conclusions, and knockout eliminates strong candidates on a single noisy judgment. (Si et al.'s Swiss usage is precedent; the information-theoretic framing and the cross-agent disagreement routing are new.)
- **Anytime + budget-adaptive.** Every round strictly refines the standings; you can stop at any budget. This operationalizes plan improvement #4 ("disagreement as signal") and #5 (compute governor): route compute where MRR-per-dollar is highest.
- **Robustness to intransitivity/noise.** Score accumulation across rounds tolerates cyclic preferences that break sort-based aggregation (Qin et al. explicitly flag LLM comparator intransitivity as an open problem for their heapsort).
- **Top-heavy refinement matches MRR.** Contested-zone pairs near the list head are precisely what moves MRR; Swiss naturally concentrates rounds there once seeded.

**Honest caveats to report:** Swiss adds inference-time LLM calls where RRF/tensor are free post-hoc; Si et al. found round count non-monotone (tune on dev); and the hypothesis to test is that the lift concentrates on high-disagreement papers — ILCiteR's result (conditional, context-dependent ensembling beats unconditional) predicts exactly this, and it's also the empirical motivation for the tensor's context-conditioned weights. If Swiss refinement doesn't beat the tensor at matched compute, report MRR-per-dollar and let the governor pick per paper — that's still a contribution (adaptive fusion policy), and it's honest.

### F3. Non-LLM sources as first-class fusion citizens — **[Adapt]**

Feed the fusion (all rungs) with independent signals the reference set validates: **graph centrality** of each direction's concepts in the forward-citation/concept graph (MEDRank: PageRank over concept graphs with distance-decayed edges lifted major-heading recall 26–30% and matched human indexers in a blinded study); **KG horizon features** (SciMuse's degree/PageRank/citation-velocity — their feature-only model beat LLM ranking at top-N precision: 70% vs 51% top-1); **retrieval scores + provenance** (ILCiteR's support counts and recency, aggregated by best-rank/support/recency precedence). This strengthens the CMTF story: the horizon matrix H is not decoration — SciMuse proves graph features alone carry ranking signal at the head of the list, which is where MRR lives.

### F4. Same-family ensembles are near-worthless; heterogeneity is the axis — **[Adopt]** (design justification)

MOOSE-Chem2: same-LLM committees tie 82.35% of the time; VirSci: diversity boosts quality with an inverted-U. Cite both to justify exactly-two heterogeneous rankers (GPT + Claude) rather than N samples of one model, and to pre-empt "why not 5 agents?" Report 2-vs-3 rankers (add one open-weights model, e.g. Qwen/Llama) as an ablation if time permits.

### F5. Cascade + distillation as the efficiency epilogue — **[Adopt]**

Sun et al.'s cascade (cheap model ranks all, expensive model reranks top-30 = 1/5 cost) and permutation distillation (RankNet on fused dev rankings → 435M cross-encoder that *beat its teacher*) give you the "deployment" paragraph: distill the *fused* ranking into a small reranker for inference-time use; RankNet > LambdaLoss > listwise-CE on LLM-permutation labels is already established, so the recipe is one experiment, not a research program.

---

## Q4. Evaluating the agent for ranked recommendations

### E1. Metric suite (primary table) — **[Adopt/Adapt]**

1. **Candidate-recall ceiling first** (plan §8.2): Recall@K_cand per source and union — rankers can't fix a missing candidate.
2. **MRR** (headline, matched via G2's hierarchical function) + **nDCG@{5,10}** using the graded realization labels from G1 (grades = gain levels — this is where graded ground truth pays off) + **Recall@K** + **success@K** (≥1 realized direction in top-K — SciMuse's "top-N success probability", the most product-relevant number).
3. **Lag-discounted MRR** (G1): reciprocal rank × exp(−lag/τ_time) — rewards recommending *imminent* directions first. **[Novel]** as a metric for this task.
4. **MRR-per-dollar / per-call** (Zhuang's efficiency accounting: #calls, prompt+generated tokens, latency, $) for every system row — the MLRC-Bench compute-aware spirit, and the axis on which Swiss-vs-tensor is honestly compared.

### E2. Match-function validation (the part reviewers will attack) — **[Adopt]**

- **τ sensitivity sweep** (HindSight swept 0.93–0.97 and showed conclusions robust in *ratio* terms — replicate that figure for our τ and entailment thresholds).
- **Human calibration set**: ~150–200 (candidate, gold) pairs labeled by ≥2 team members + externals; report Cohen's κ for the entailment judge vs. humans (plan §1 promises this; ResearchAgent's κ range 0.41–0.71 and Si et al.'s 56.1% split-half set realistic expectations — quote them so our number has context).
- **Third-family judge**: the entailment judge must be neither GPT nor Claude (MOOSE-Chem2's judge-affinity effect: each judge favored committees containing itself; AI Idea Bench deliberately used Deepseek to judge GPT-4o systems). Use e.g. Gemini or an open-weights model.
- **Style normalization** of all candidates AND gold directions before any LLM judging (Si et al.'s anonymizer reduced human AI-detection to chance; length is a real confound — their AI ideas were 1,186 vs 902 words pre-normalization).

### E3. Robustness & bias audit table — **[Adapt]** (one compact table, huge credibility)

| Test | Protocol | Source |
|---|---|---|
| Candidate-order sensitivity | original / reversed / random order → ΔMRR | PRP, Sun, SCaLR |
| Self-consistency | RBO between repeated runs; repetition/missing/rejection rates | RankGPT |
| Paraphrase stability | Jaccard overlap of top-K under paraphrased paper summaries | HawkSight (0.71 benchmark) |
| Distractor confusion | mass on SSLI / counterfactual / anachronism / random tiers | G5; SciRerankBench |
| Popularity bias | mean concept-popularity per rank position; rank correlation with citation counts of nearest neighbors | Hou; SciMuse |
| Prestige withholding | rankers never see citation counts/venues of any evidence | HawkSight, Hou |
| Zero-fabrication | % recommended directions not traceable to the candidate pool (must be 0 by construction — verify) | HawkSight |
| Leakage control | post-cutoff-only slice + realization-pool swap stability | VirSci App. A; IdeaBench |

### E4. Statistics — **[Adopt]**

Bootstrap CIs on all headline metrics; paired Wilcoxon (+ Bonferroni across the ablation ladder) for system comparisons (MEDRank's protocol); **mixed-effects model** `metric ~ system + (1|paper) + (1|judge)` for any human-judged numbers — Si et al. found *reviewer* variance exceeds *idea* variance, so per-review t-tests alone are not defensible.

### E5. Small blinded human study (the trust anchor) — **[Adopt]**

MEDRank's design transplanted: for ~30–40 test papers, show experts the top-5 directions from {our system, best single agent, RRF baseline, the actual masked future work}, **source-blinded, vote-prior equalized** (sample so each source has equal prior probability of receiving a vote); experts vote for the most valuable directions. Two headline outcomes: (a) system vs. baselines preference; (b) **system vs. the real future-work section** — MEDRank matched human indexers (p=0.36 difference), and even *approaching* the real future-work section is a quotable result. Validate the automatic pipeline by reporting the correlation between expert votes and realization grades (VirSci's r=0.52 pattern gives the expected magnitude).

### E6. Quadrant diagnostics (the figure reviewers remember) — **[Adapt]**

HindSight's 2×2 transplanted per ranker: LLM-rank (high/low) × realization (high/low) → **True Positives / Hidden Gems (realized but ranked low) / Overhyped (ranked high, never realized) / True Negatives**. Their finding — "Overhyped" ideas are characteristically grand-but-vague — gives Direction Cards a concrete audit: does our fusion rescue Hidden Gems that individual rankers buried? That's the multi-agent story in one figure.

### E7. Prospective, pre-registered rolling evaluation — **[Novel]** (the headline move)

Everything above is retrospective (mask old papers, check realization in the archive). Add a **forward arm**:

1. Take papers published in the **last 1–3 months** (after every model cutoff, zero realization signal exists yet).
2. Run the frozen system; **publicly commit** the ranked recommendations (hash + timestamp in the repo, or an OSF pre-registration — Si et al. pre-registered their execution study the same way).
3. Score realization at t+6 and t+12 months with the frozen G1/G2 pipeline; publish the updates.

No idea-generation benchmark in the reference set does this. It is cheap (the pipeline already exists), *ungameable* (the answers don't exist at submission time), converts the benchmark into a **living leaderboard** (strong fit for a benchmark/resource track — plan §12's recommended venue), and it pre-answers the deepest reviewer objection: "your match function might be fit to the archive." Even with only ~50 prospective papers, the camera-ready can carry the 6-month scores.

### E8. Two-level evaluation: list quality + card utility — **[Adapt]**

SciRerankBench's warning: high ranking metrics did *not* translate into better downstream answers. Mirror it: besides list metrics, score the **Direction Cards** — grounding accuracy (% claims traceable to quoted evidence; HawkSight hit 98% with hard-grammar prompting), trade-off transparency (%), and rationale-faithfulness (plan §6's counterfactual probe: does removing the cited evidence actually drop the rank?). This keeps the explainability layer honest rather than decorative.

---

## Q5. Personalization (user research preferences) — added 2026-07-18

Design rule up front: **personalization is a re-weighting layer at fusion time, never a change to the benchmark objective.** The universal track (realization-calibrated MRR) stays the headline; personalization is a second head with its own ground truth and its own metric column. This keeps the benchmark ungameable and the Direction Cards auditable ("boosted because it matches your toolkit" is a visible, removable term).

### P1. Author-conditional realization: personalized ground truth for free — **[Novel]** (the strongest idea in this section)

Split every realized direction's label (G1) by **who realized it**, using Semantic Scholar author IDs on the forward-citing papers:

```
realized_by_authors  — ≥1 realizing paper shares an author with P (the authors' own follow-up work)
realized_by_field    — realized only by outside groups
```

Then define the **personalized task**: given P *and its authors' publication histories*, rank directions by what *these authors* will do next; evaluate **author-MRR** against `realized_by_authors`. This is a fully automatic personalization ground truth — zero annotation, same API calls we already make — and no benchmark in the reference set has it. Bonus analyses that fall out for free: (a) do authors disproportionately realize their *stated* future work while outsiders realize the unstated directions? (b) an explore/exploit split — are author-realized directions closer to their prior-work centroid (SciMuse's semantic-distance finding predicts yes)?

Report both columns: **field-MRR** (universal head) and **author-MRR** (personalized head). A system that wins both is a much stronger claim than either alone.

### P2. The Preference Adapter at the fusion layer — **[Adapt]** (HawkSight + SciMuse + the plan's own tensor)

Represent the user u three ways, cheapest first:
1. **Structured preference JSON** extracted from free text by a schema-constrained prompt (HawkSight's hard-grammar extraction hit 96% valid JSON): weights over our existing per-dimension scores (realization-likelihood, novelty, feasibility, impact horizon), hard constraints (compute budget → feasibility floor; time horizon → lag-discount τ_time; methods/domains to include/exclude).
2. **Publication-history embedding**: SPECTER2 centroid (or set) of the user's recent papers — Hou's conditional-ranking framing with the researcher's history as the condition; keep it to the *recent few* papers (Hou: fewer, more recent history items reduce popularity bias and overload).
3. **KG subgraph features** (SciMuse): the user's concept set with degree/PageRank/citation-velocity — their feature-only model predicted researcher interest at top-1 precision 70%, so these features demonstrably carry personal-relevance signal.

Where it enters the math — two small, ablatable hooks, both already anticipated by the plan's tensor:
- **Context-vector extension:** the tensor's context weights `w_m(p) = ⟨U_m, z(p)⟩` become `w_m(p,u) = ⟨U_m, z(p) ⊕ z(u)⟩` — per-source trust adapts to the *user* as well as the paper (e.g., a user who wants grounded incremental work up-weights the retrieval/citation sources; a risk-seeker up-weights the horizon/novelty source). Zero new machinery — z just gets longer.
- **Personal-fit source:** one extra column in ℳ scoring each candidate by user affinity, `fit(d,u) = cos(e_d, e_u)` plus a bilinear term `⟨e_d, W e_u⟩` learned on the author-realization labels (P1). RRF/meta-ranker/tensor all accept it as "one more source," so the ablation is a single row: ± personal-fit.

### P3. Preference elicitation without history (cold start) — **[Adapt]**

For users with no publication record (students, industry): ResearchAgent's criteria-induction trick — show the user ~10 direction pairs, ask which they'd pursue, induce a personal rubric/weight vector from those comparisons (they induced human-aligned rubrics from ~10 scored pairs). Ten pairwise clicks is a one-minute onboarding and reuses the exact pairwise machinery from R2/F2.

### P4. Pitfalls to state honestly

- **Filter bubble vs. discovery:** SciMuse found *closer-field* ideas rated more interesting, but ResearchAgent found *cross-domain* entity injection boosts originality. Expose the tension as an explicit explore/exploit slider (a scalar on the personal-fit weight), never hard-code it.
- **Leakage in the personalized head:** the authors' own future-work statements strongly predict their follow-ups — that's the *point* of P1, not a bug, but the personalized ranker must not see any post-t₀ author publications (same dual-clock discipline as G3).
- **Si et al. reframing (use in the paper):** their finding that reviewer variance exceeds idea variance is usually read as "human judgment is noisy" — the personalization head reads it constructively: a large share of that variance is *stable individual preference*, i.e., signal a personalized ranker can capture. Author-MRR quantifies exactly how much.
- **Scope control:** for the hackathon paper, P1 + the personal-fit source (one ablation row) is enough; the full Preference Adapter UI is the demo/extension, or a follow-up paper.

---

## Q6. Logit-level signals in ranking — added 2026-07-18

Practical constraint first (design around it, and say it in the paper): **OpenAI exposes token logprobs (top-20) on generated tokens; the Anthropic API does not.** So the GPT agent and any open-weights sidecar can produce distribution-level signals, while Claude stays generation-only. This asymmetry is not a problem for the fusion design — it's a *motivation* for it (heterogeneous sources with heterogeneous signal types is exactly what the tensor's per-source features handle), but it rules out "average the two models' logits" naively.

### L1. Setwise label-distribution scoring — **[Adopt]** (Zhuang's logits mode, our workhorse)

Wherever we make a setwise call (R2's tournament groups), ask for the best candidate as a **single-token label** (A…T) and read the top-k logprobs of that first token → a **soft distribution over the whole group from one call**, at zero extra cost. Zhuang found no effectiveness loss vs. generation mode. Iterating (mask the chosen label, re-read) yields a full within-group ranking with Plackett–Luce-style sequential probabilities. Requirement: labels must tokenize to single tokens — use capital letters, verify with the tokenizer.

### L2. Soft-Swiss: fractional wins from logprobs — **[Novel]** (upgrade to R2/F2)

Replace the Swiss tournament's binary +1 with **fractional points = the label distribution** (candidate i earns P(i) from each group call). Bradley–Terry/PL fitting accepts soft outcomes as weighted likelihood. Effect: more information extracted per LLM call → fewer rounds to converge, and the standings become real-valued earlier, which sharpens the score-adjacent pairing that makes Swiss information-efficient (F2's p(1−p) argument). Ablation: soft vs. hard Swiss at matched call budgets — a clean, novel comparison nobody has run.

### L3. Logit-margin confidence for routing, fusion features, and calibration — **[Adapt]**

The margin between top-1 and top-2 label logprobs is a per-decision confidence that is *relative* (unlike absolute scores, which Si et al. showed LLMs produce badly calibrated):
- **Routing:** low-margin groups get extra Swiss rounds; high-margin decisions are accepted — this composes with F2's cross-agent disagreement routing into a single uncertainty budget (route on `disagreement ∨ low-margin`).
- **Fusion features:** the plan's `gpt_confidence` feature (§5.2) becomes a measured logprob margin instead of verbalized confidence — verbalized confidence is the thing the literature distrusts; margins are at least monotone in the model's actual preference strength.
- **Calibration/abstention:** fit a temperature/isotonic map from margins to P(match ground truth) on dev; report ECE (plan §6 already promises ECE — logprobs make it meaningful). Claude's side uses bootstrap-agreement rate (fraction of B shuffled runs agreeing) as its confidence proxy — a fair generation-only analog, and the GPT-vs-Claude confidence-quality comparison is itself a reportable finding.

### L4. PMI scoring: popularity-corrected likelihood via an open-weights sidecar — **[Novel-ish]** (the most interesting logit idea)

With a local model (Llama/Qwen via vLLM — full logprobs on *input* tokens), score every candidate:

```
PMI(d; p) = log P(d | comprehension summary of p) − λ · log P(d)
```

`log P(d|·)` alone is query-likelihood pointwise ranking (Zhuang's `pointwise.qlm` — weak, and popularity-contaminated: generic "apply LLMs to X" text is high-probability under *any* conditioning). Subtracting the **unconditional** `log P(d)` cancels exactly that base-rate fluency, directly attacking the popularity bias Hou documented — the audit metric in E3 (popularity-per-rank curve) is the validation: the PMI source should show a visibly flatter curve than the LLM rankers. Cost: one forward pass per candidate on a small open model (no API calls); enters fusion as one more source column, and doubles as the third heterogeneous ranker for the 2-vs-3 ablation (F4).

### L5. Prompt-level self-calibration (SCaLR without fine-tuning) — **[Adapt]**

SCaLR's load-bearing idea (its ablation: −4.7 nDCG without calibration) is aligning list-context scores to context-free pointwise anchors. Zero-shot analog: for each paper, get pointwise `P("yes")` logprobs ("is this a likely next direction? yes/no") for all candidates in parallel, then **monotonically map each agent's listwise/tournament scores onto its own pointwise anchor scale** (isotonic fit on dev). Payoff: scores become comparable **across sublists and across agents**, which is precisely what rank-only RRF discards — the tensor can then consume calibrated continuous `φ_m` entries instead of reciprocal ranks (the CMTF formulation is agnostic to this; it just gets better-conditioned inputs).

### L6. Label-prior debiasing — **[Adopt]** (hygiene for all of the above)

Label logits carry positional/lexical priors ("A" is preferred a priori). Fix by **permutation marginalization**: rotate candidate→label assignments across the B bootstrap shuffles we already run (R1) and average the distributions — Hou's bootstrapping lifted into logit space. Report the residual label-prior (entropy of the average distribution under content-free candidates) as a diagnostic.

### Evaluation hooks for Q5/Q6 (add to the ablation ladder)

| Row | Question it answers |
|---|---|
| ± personal-fit source (P2) on field-MRR and author-MRR | does personalization help, and does it cost universal accuracy? |
| soft-Swiss vs hard-Swiss at matched calls (L2) | do logits buy convergence speed? |
| ± PMI source; popularity-per-rank curve (L4) | does likelihood correction debias, and does it lift MRR? |
| margin-routed vs disagreement-routed vs uniform Swiss budget (L3) | which uncertainty signal spends compute best? |
| ECE: logit-margin (GPT) vs bootstrap-agreement (Claude) vs verbalized confidence | which confidence is trustworthy enough for abstention? |

---

## Recommended paper story (what to lead with, in order)

1. **Task + benchmark (Q1/Q4):** masked future-work → *ranked* recommendation, with the realization ladder (G1), typed entity matching (G2), distractor tiers incl. anachronisms (G5), and the prospective arm (E7). This is the safest, most defensible novelty — both closest benchmarks (IdeaBench, AI Idea Bench) explicitly lack ranked ground truth and rank-aware metrics.
2. **Method (Q2/Q3):** heterogeneous agents with forecast-framed evidence-conditioned prompts (R3), Swiss-Setwise per-agent ranking (R2), and the fusion ladder with disagreement-routed Swiss refinement (F2) vs. the tensor — calibrated to dev MRR throughout, reported with MRR-per-dollar.
3. **Findings to hunt for (each is a paper-worthy sentence if it lands):** (a) forecast-framing beats novelty-framing; (b) fusion rescues Hidden Gems that both single agents bury; (c) context-conditioned fusion lift concentrates on high-disagreement papers (ILCiteR predicts this); (d) anachronism distractors expose rankers that merely "sound informed."
4. **Honesty clause (keep plan §12's):** primitives are known; the contribution is the task framing, the realization-graded ranked ground truth, the calibrated fusion recipe, and the prospective benchmark. If the tensor/Swiss don't beat RRF significantly, the benchmark + diagnosis still stands.

## Feasibility guardrails (hackathon → paper)

- **Corpus:** start with ACL Anthology / cs.CL+cs.LG slices of S2ORC/unarXive (section-parsed; SciMON precedent) — papers 2019–2023 with explicit future-work sections for the retrospective arm (realization window closes), 2025-Q1+ for the clean post-cutoff slice, 2026-Q2 for the prospective arm. ~1–2k dev/train papers, 300–500 test papers is plenty (Si et al. drew significant conclusions from 49/condition; we have automatic labels).
- **Budget order-of-magnitude:** 500 test papers × 2 agents × (3 bootstrap listwise calls + ~40 tournament calls) ≈ 45k LLM calls of ~2-4k tokens — tens of dollars on mini-tier models for ablations, low hundreds with frontier models for the headline rows. The cascade (F5) cuts the frontier bill 3–5×.
- **Reuse before building:** ILCiteR's public dataset (200k evidence spans) to prototype the evidence bank; HindSight's SPECTER2+FAISS matching code pattern; Semantic Scholar API for forward citations (plan already assumes it); Optuna for (k, w, R, c) tuning.
- **Exclude from the reference set:** `ivanitskiy-interp.pdf` (mechanistic-interpretability lecture slides — off-topic); note `MEDRank….pdf` is duplicated twice and `SciMON/scimon.pdf` duplicates Wang et al. 2024.

## Per-paper evidence index (for the related-work section)

| Paper | Load-bearing takeaway | Plugs into |
|---|---|---|
| Qin 2024 (PRP) | Both-order pairwise + tie protocol; all-pairs = round-robin Copeland; order-robustness table | R1, R2, F1, E3 |
| Zhuang 2024 (Setwise) | c-way selection cuts 62% of pairwise cost at −0.8% nDCG; efficiency accounting template | R2, E1 |
| Sun 2024 (RankGPT) | Listwise sliding window; malformed-output handling; RBO; cascade; distillation (RankNet best) | R1, F5, E3 |
| Ren 2024 (SCaLR) | Calibrate listwise scores against pointwise self-scores; position-bias-free merging | F1 (cross-agent calibration), E3 |
| Cao 2007 (ListNet) | Plackett–Luce soft targets; listwise loss aligns with list metrics | G1, F1, tensor loss |
| Hou 2024 (LLMRank) | Position + popularity bias; B=3 shuffled bootstrap; less context = less popularity bias | R1, R3, E3 |
| Zhang 2025 (REARANK) | GRPO + normalized-relative-nDCG reward; 179 queries → 12k instances | G6, R5 |
| Chen 2025 (SciRerankBench) | SSLI/counterfactual/noise distractor taxonomy; ranking metrics ≠ downstream utility | G5, E8 |
| Herskovic 2011 (MEDRank) | Concept-graph PageRank; blinded vote-prior-equalized human study; matched human indexers | F3, E5 |
| Roy & Han (ILCiteR) | Evidence bank span→(papers, support); conditional ensembling beats unconditional; leakage-free temporal split | G1, F2/F3, G3 |
| Amisu (HawkSight) | LLM never fabricates items (deterministic core); grounding/stability/zero-fabrication metrics | R1, E3, E8 |
| Guo 2024 (IdeaBench) | Post-cutoff targets; gold-idea normalization prompt; insight score's circularity | G2, G3, G4, §0 |
| Qiu 2025 (AI Idea Bench) | Anonymized (motivation, plan) gold; citation-based feasibility Inf=1−e^(−x/50); round-robin IC | G2, G4, §0 |
| Wang 2024 (SciMON) | Temporal splits; trivial-overlap filter; used-for entity tuples; gold ≫ generated (85%) | G2, G3 |
| Li 2024 (Chain of Ideas) | Idea Arena: round-robin→ELO, order-reversal, human↔GPT-4o 70.8%; forward-citation chains | F1, E2, G1 |
| Baek (ResearchAgent) | Criteria induction from ~10 human-scored pairs; agreement ceilings (κ≤0.71) | E2, E5 |
| Li 2025 (MLR-Copilot) | Resource-conditioned 1–10 feasibility anchors; OpenReview ratings as supervision | G4, E2 |
| Su 2025 (VirSci) | Contemporary-DB realization scoring; year-normalized citations; leakage-swap control; diversity inverted-U | G1, G3, F4 |
| MOOSE-Chem2 | Coarse-to-fine smooths rugged judgment; same-LLM committees tie 82%; judge affinity; soft/hard recall | R4, F4, E2, G2 |
| Gu & Krenn (SciMuse) | Time-stamped KG features predict expert interest (top-1 70%); Elo needs 10k+ matches | G4, F3, E1 |
| Si 2024 (Can LLMs…) | Swiss tournament N=5; pairwise 71.4% vs pointwise chance; LLM judges < human agreement; style normalization; duplication ceiling | R2, F2, E2, E4 |
| Jiang 2026 (HindSight) | Realization-grounded scoring; novelty anti-correlates with realization; θ-sweep; quadrants | G1, R3, E2, E6, §0 |
| ICLR blogpost (Heuristics) | 20-heuristic taxonomy → direction types; per-discipline validity | G2, G5 |
| ivanitskiy-interp | Off-topic (interp lecture slides) — exclude | — |
