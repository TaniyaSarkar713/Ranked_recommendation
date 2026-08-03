# metrics.md — Formula comparison: novelty, feasibility, impact & composite idea metrics

*Deliverable for to-do item "Taniya + Naz §1". Purpose: decide, per metric, whether it becomes a **[GT] ground-truth label** (Q1), a **[Badge] Direction-Card display score**, a **[Feat] fusion/meta-ranker feature**, or **[Avoid]** for our use. Sources: the 24 papers in `References/` + web survey of scientometrics and 2024–26 LLM-idea-eval literature. Section refs (G/R/F/E/P/L) point to `ResearchHorizon_ideas_Q1-Q4.md`.*

**Column key.** *Formula* = exact definition as published. *Inputs* = data/infra needed. *Valid.* = strongest known validation against humans or outcomes. *Role* = recommended use in ResearchHorizon.

---

## 1. Novelty metrics

### 1.1 From the reference folder

| # | Metric (source) | Formula / mechanism | Inputs | Valid. | Role |
|---|---|---|---|---|---|
| N1 | **Temporal-split novelty** — AI Idea Bench 2025 (Qiu et al., arXiv:2504.14191, Eq 7) | `S_N = (1 + cd_idea) · cc / (1 + hd_idea)`; `hd`/`cd` = embedding distance of the idea to *historical* (pre-cutoff) / *contemporary* (post-cutoff) papers; `cc` = citation count of matched contemporary papers. ⚠ Sign conventions of hd/cd read ambiguously in the paper — re-derive from their code before implementing. | embeddings, Semantic Scholar retrieval, cutoff date | none reported vs humans | [Feat]; the `cc` term is really *realization impact* → fold into our G1 grades instead |
| N2 | **Historical Dissimilarity HD** — VirSci (Su et al., arXiv:2410.09403) | `HD = mean Euclid. dist.` of idea embedding to its 5-NN in the *past* DB (higher = newer vs past); year-mean normalized | embedded past corpus (they: mxbai-embed-large + Faiss) | as part of ON (below), r = 0.52 vs human novelty | [Badge] [Feat] |
| N3 | **KG white-space features** — SciMuse (Gu & Krenn, arXiv:2405.17044) | Per-concept/pair, computed *as of t₀*: node degree, PageRank, total/annual concept citations, #papers mentioning concept (all **negatively** correlated with expert interest); semantic distance between concept subgraphs (closer = more interesting); dynamic features (new neighbors gained/yr). Supervised NN on 25 features → interest | time-stamped concept co-occurrence KG (OpenAlex-scale in the paper; we can approximate at subfield scale) | **AUC 0.645, top-1 precision 70%** vs 4,451 expert ratings — best-validated novelty signal we have | **[GT-secondary] [Badge] [Feat]** — our "quantitative novelty" (§G4) |
| N4 | **Similarity-to-prior-work penalty** — SciMON (Wang et al., ACL 2024) | novelty violated if `max cosine(idea, 20-NN of prior-work bank) ≥ μ = 0.6` (SentenceBERT); iterate rewriting until below | sentence-transformer + FAISS bank of prior ideas | human study: boosting ↑ novelty in ~56% of iterations | [Feat] (as continuous `max-sim`); also our dedup gate |
| N5 | **Similarity-to-original** — MLR-Copilot (arXiv:2408.14033) | `sim(generated idea, source paper) ∈ [0,1]`, lower = more novel | one embedding call | expert study only (n=5 raters) | [Feat] (same machinery as matching, opposite polarity) |
| N6 | **Relative-rank novelty ("Insight Score")** — IdeaBench (Guo et al., arXiv:2411.02429) | `I = mean_i (r_target_i − 1)/n` — how often generated ideas out-rank the real paper's idea under a GPT-4o listwise ranker with indicator q | LLM listwise ranker | none vs humans | **[Avoid]** as GT — circular for us (relies on the LLM judgment we distrust; §G4); fine as a system-level diagnostic |
| N7 | **Anchored human novelty rubric** — Si et al. (arXiv:2409.04109, App. E) | 1–10 with verbal anchors (1 trivial variant … 10 very novel); reviewers instructed to search prior work; papers online before cutoff count as prior art | expert raters | inter-reviewer split-half only 56.1% — noisy even for humans | [Badge] wording template; human-study instrument (E5) |
| — | **Warning label** — HindSight (arXiv:2603.15164) | LLM-judged novelty vs. actual realization: **ρ = −0.291 (p=0.003)** | — | — | Reason novelty is never our primary GT axis |

### 1.2 From the broader scientometrics literature (web survey)

| # | Metric (source) | Formula / mechanism | Inputs | Valid. | Role |
|---|---|---|---|---|---|
| N8 | **Atypical combinations** — Uzzi, Mukherjee, Stringer & Jones, *Science* 2013 | For each pair of cited *journals*: `z_ij = (obs_ij − μ_rand)/σ_rand`, null from ~1,000 Monte-Carlo citation-switched networks (degree- and age-preserving). Paper = (median z = *conventionality*, 10th-pctile z = *tail novelty*) | citation graph + journal IDs + heavy randomization (or precomputed per-year z-table) | hit rate 9.11/100 for high-conventionality + high-tail-novelty vs 5% baseline; modest convergent validity vs F1000 expert novelty (Bornmann 2019) | [Feat] at **concept level** (apply to a direction's entity pairs, not journals); expensive — prefer N9 |
| N9 | **Commonness (cheap analytical null)** — Lee, Walsh & Wang, *Research Policy* 2015 | `Commonness_ijt = obs_ijt / E_ijt`, `E_ijt ≈ N_it·N_jt / (2N_t)` (PMI-style); paper novelty = inverse of 10th-pctile commonness | co-occurrence counts only — closed form, no simulation | novelty predicts impact; inverted-U with team size | **[Feat] [Badge]** — the practical way to score "how rare is this direction's concept combination" (= quantitative version of §G5 anachronism logic) |
| N10 | **Distance-weighted new combinations** — Wang, Veugelers & Stephan, *Research Policy* 2017 | `Novelty = Σ_{first-ever journal pairs} (1 − cos(J_i, J_j))` over 3-yr co-citation profiles; require pair reuse ≥1× within 3 yrs; NOV_CAT top-1% = "highly novel" | first-appearance index since corpus start + rolling profiles | highly novel: +39.8% odds of top-1% hit, but **delayed recognition & lower-IF venues initially** | [Feat]; its "bias against novelty" finding justifies our lag-discounted MRR (I6) — novel directions realize *late* |
| N11 | **Embedding novelty from reference distances** — Shibayama, Yin & Matsumoto, *PLoS ONE* 2021 | refs → title-embedding vectors; `d_ij = 1 − cos(v_i,v_j)`; `Novel_q = percentile_q({d_ij})`, primary q=100 (max pair distance) | reference titles + any embedding model — **text-only, no network** | r ≈ 0.17 with author-reported novelty; strong tail-impact odds | **[Feat] [Badge]** — cheapest defensible novelty for a *candidate direction* (max distance among its evidence/concept embeddings); outlier-sensitive, use q=95 too |
| N12 | **Research-strategy typology + surprisal** — Foster, Rzhetsky & Evans, *ASR* 2015 | classify each contribution: {jump (new node), new-edge, repeat} × {consolidation (within cluster), bridge (between)}; reward = surprisal `I = −log P(strategy)`; mean citations: repeat 8.38 < new 11.00 < jump 12.90, higher variance | entity co-mention KG + community detection | strategy surprisal explains 29.2% of citation variance | [GT-secondary] [Feat] — the scientometric ancestor of our G2 direction-type base rates (§G5): per-type priors with a principled reward form |

---

## 2. Feasibility metrics

### 2.1 From the reference folder

| # | Metric (source) | Formula / mechanism | Inputs | Valid. | Role |
|---|---|---|---|---|---|
| F1 | **Citation-influence feasibility** — AI Idea Bench 2025 (Eq 8–10) | Recursively decompose idea (motivation + plan) into atomic concepts; retrieve supporting methods per concept; `Inf = 1 − e^(−x/λ)`, λ=50 (x = citations of supporting method; 35→0.5, 100→0.9); time-decayed variant (Eq 9); `S_F = mean over retrieved supports`; **FPS** = same per experiment step, averaged | concept decomposition (LLM) + Semantic Scholar citations | none vs humans, but non-LLM and auditable | **[Badge] [Feat]** — our default quantitative feasibility (§G4) |
| F2 | **Resource-conditioned rubric** — Si et al. App. E = MLR-Copilot App. B.2 | 1–10 anchored to "typical CS PhD student, 1–2 months, ample API access, limited GPU": 1 impossible, 5 moderately feasible, 8 highly feasible, 10 easy | LLM or human rater | HindSight: LLM feasibility is the *only* judge dimension positively correlated with realization (**ρ = +0.252**) | [Badge]; [Feat] as LLM-scored feature (forecast-relevant per HindSight) |
| F3 | **Induced 5-level rubrics** — ResearchAgent (arXiv:2404.07738, Tables 12–15) | Method: clarity/validity/rigor/innovativeness/generalizability; Experiment: +feasibility/reproducibility; rubric text *induced* from ~10 human-scored idea pairs | few human-scored pairs + LLM | human–model pairwise κ up to 0.71 | Template for calibrating any LLM-scored dimension to our team's taste (E2) |

### 2.2 From the broader literature (web survey)

**Honest headline: there is no established paper-level feasibility metric in scientometrics.** Feasibility is proxied indirectly via *method maturity*. Usable proxies:

| # | Metric (source) | Formula / mechanism | Inputs | Valid. | Role |
|---|---|---|---|---|---|
| F4 | **Price index (reference recency)** — de Solla Price 1970 | fraction of references younger than 5 years; plus mean/median reference age | reference-year distribution of the direction's *enabling-method* papers | long-standing scientometric standard | [Feat] — mature (older, consolidated) enabling methods ⇒ higher feasibility; fast-moving front ⇒ riskier |
| F5 | **Technology-maturity S-curves (BIMATEM)** — Lezama-Nicolás et al., *Scientometrics* 2018 | fit logistic growth to annual paper/patent counts for the enabling technology; life-cycle stage (emergence/growth/maturity/saturation) from curve position | topic-level publication time series (OpenAlex) | validated on additive manufacturing | [Feat] [Badge] — "how ready is the tooling this direction needs" |
| F6 | **TRL (Technology Readiness Levels)** — Mankins/NASA 1995 | ordinal 1–9, expert-assessed, no formula | expert panel | industry standard | [Avoid] for automation (manual); vocabulary for Direction-Card wording only |

⇒ Our default stays **F1 (AI Idea Bench citation-influence)** for quantitative feasibility, optionally enriched with F4/F5 method-maturity features; F2's LLM rubric remains the only *forecast-validated* judge dimension (HindSight ρ=+0.25).

---

## 3. Impact / realization metrics (the G1 ladder's raw material)

### 3.1 From the reference folder

| # | Metric (source) | Formula / mechanism | Inputs | Valid. | Role |
|---|---|---|---|---|---|
| I1 | **Realization impact** — HindSight | `h(p) = 0.6·ĉ(p) + 0.4·v(p)`; ĉ = min-max-normalized citations within pool; v = top-venue indicator; idea score = `max_{p∈M(i)} h(p)` over matches M(i) = top-20 SPECTER2 neighbors with cos ≥ θ=0.96 | SPECTER2 + FAISS + S2 citations/venues | grounded-vs-vanilla separation 2.5× (p<0.001) where LLM judge saw nothing (p=0.584) | **[GT]** core of Grade 3, with fixes: field/age-normalized percentile instead of pool min-max; max + count tie-break (§G1) |
| I2 | **Contemporary Impact CI** — VirSci | mean citation count of 5 most-similar post-cutoff papers, **year-mean normalized** | embedded contemporary corpus | part of ON, r=0.52 | [GT] ingredient — adopt the year normalization |
| I3 | **Support counts** — ILCiteR (arXiv:2403.08737) | evidence span → set of (paper, support s = # times cited for that span); aggregation precedence: best-rank → Σ support → recency | citation-context mining | conditional ensembling +MRR over lexical baselines | **[GT]** = "# independent realizing groups" in Grade 3; [Feat] |
| I4 | **Concept-graph centrality** — MEDRank (IJMI 2011) | PageRank/TextRank on concept graph: `R'(u) = c·Σ_{v∈B_u} R(v)/N_v + c·E(u)`; edges = co-occurrence with distance decay `1/c^d` (prune d≥5) or distributional (RRI) similarity; prior E(u) = extractor confidence | concept extraction + graph lib | +26–30% major-heading recall; matched human indexers (p=0.36, blinded) | [GT-tiebreak] [Feat] — centrality of a direction's concepts in the *forward-citation* neighborhood |
| I5 | **Horizon priors** — plan.pdf §7 | `popularity(d)=Σ_{p∈M_d} log(1+c_p)`; `recency(d)=Σ exp(−(Y−y_p)/τ)`; `novelty(d)=adjacency·1/(1+|M_d|)`; z-normalized within paper | related-papers graph (S2/OpenAlex) | — (ours, to ablate) | [Feat] — the CMTF side-info matrix H |
| I6 | **Time-to-realization lag** — ours (§G1) | `lag(g)` = months from t₀ to first realizing paper; lag-discounted MRR `RR·e^(−lag/τ_time)` | forward citations + dates | to validate | **[GT-auxiliary]**, [Badge] "imminence" |

### 3.2 From the broader literature (web survey)

| # | Metric (source) | Formula / mechanism | Inputs | Valid. | Role |
|---|---|---|---|---|---|
| I7 | **Field-normalized citation impact (MNCS / percentiles)** — Waltman et al. 2011; Bornmann & Leydesdorff | `MNCS = (1/n)·Σ c_i/e_i` with `e_i` = mean citations of same (field × year × doc-type); percentile rank `PR = 100·(i−0.5)/n` within field-year cohort; `PP_top10%` shares | citation counts + field labels + year (OpenAlex concepts suffice) | the standard in evaluative bibliometrics (Leiden Ranking); percentiles robust to skew | **[GT]** — this is the canonical form of the "field/age-normalized citation percentile" our I1 fix calls for; use Hazen percentiles within (field, year) bins |
| I8 | **CD / Disruption index** — Funk & Owen-Smith 2017; Wu, Wang & Evans, *Nature* 2019 | `CD_t = (1/n_t)·Σ_i (−2·f_it·b_it + f_it)`; equivalently `D = (N_i − N_j)/(N_i + N_j + N_k)` ∈ [−1,1]; +1 = disruptive (citers ignore predecessors), −1 = consolidating | citation graph only (focal → refs → citers) | face-valid on landmark papers; **near-zero correlation with raw citations (r≈0.03)** — a distinct axis. ⚠ 2024 critiques: citation-inflation bias (Petersen, QSS 2024), dataset artefacts (Holst, arXiv:2402.14583), degenerate when refs=0 | [Feat] [Badge] on the *realizing* paper — "was this direction realized disruptively or incrementally"; too contested for [GT] |
| I9 | **Sleeping Beauty coefficient** — Ke, Ferrara, Radicchi & Flammini, *PNAS* 2015 | `B = Σ_{t=0}^{t_m} (ℓ_t − c_t)/max{1, c_t}` with ℓ_t the line from (0,c₀) to the citation peak (t_m, c_{t_m}); awakening time `t_a = argmax d_t` (point-to-line distance) | yearly citation series (`counts_by_year` in OpenAlex) | heavy-tailed spectrum across 22M WoS papers | [Diagnostic] — retrospective only; flags "ahead-of-time" directions whose realization lag (I6) was long; pairs with N10's delayed-recognition finding |
| I10 | **KG impact forecasting (Impact4Cast)** — Gu & Krenn, *MLST* 2025 (arXiv:2402.08640); predecessor Krenn & Zeilinger, *PNAS* 2020 | link prediction on an evolving concept KG: will an *unconnected* concept pair become connected (PNAS'20: AUC 0.85, 17 features) and accrue **≥ IR citations within 3 yrs** (Impact4Cast: 141 features — degrees, PageRank, cosine/Simpson similarity, citation histories; NN, AUC>0.9 link-impact, ≈0.7 impact-beyond-link) | evolving concept KG from OpenAlex (code + data released) | top-predicted pairs accrue ~10³× average citations | **[Feat]** (predicted-impact of a direction's concept pair) — and **prior art we must cite**: the closest published analog to our "will it be realized, how impactfully" target, but at concept-pair level with no paper conditioning, no ranking, no LLM agents |

---

## 4. Composite / overall-quality metrics

### 4.1 From the reference folder

| # | Metric (source) | Formula / mechanism | Inputs | Valid. | Role |
|---|---|---|---|---|---|
| C1 | **Overall Novelty ON** — VirSci | `ON = (HD × CI) / CD`; CD = mean dist. to 5-NN contemporary papers (lower = better aligned with realized future) | two embedded corpora + citations | r = 0.52 (human), 0.57 (GPT-4o judge) | [Feat]; template for any composite — but keep channels separate per §G1/MOOSE-Chem2 |
| C2 | **Idea Arena Elo** — Chain of Ideas (arXiv:2410.13185) | round-robin pairwise (order-reversed ×2), per-criterion 0/1/2 verdicts → win/loss/tie → Elo (Chatbot-Arena update) | LLM judge, O(n²) matches | human↔GPT-4o agreement 70.8% (per-criterion 66.5–76.3%) | System-level eval only; per-candidate version = our F1/F2 BT fusion |
| C3 | **Ideas Competition** — AI Idea Bench | pairwise, 6 aspects, win=3/lose=0, summed over opponents → Total Rank/Score | LLM judge | none | [Avoid] — flat 3/0 loses information vs BT/Elo |
| C4 | **Swiss tournament score** — Si et al. | pair by similar accumulated score, +1/win, N rounds → score ∈ [0,N]; N=5 tuned on ICLR proxy (top-vs-bottom gap 1.73) | pairwise LLM judge, O(N·n/2) | ranker top-49 overlapped human's on only ~17/49 | Basis of our R2/F2 (with soft/BT upgrades) |
| C5 | **Realization ladder** — ours (§G1) | grades 0–3 (distractor / stated-only / realized / realized-with-impact) + PL soft targets | G2 matching + I1–I3 | to validate (E2/E5) | **[GT] primary** |

### 4.2 LLM-era systems beyond our folder (web survey, 2024–26)

| # | Metric (source, arXiv) | Formula / mechanism | Inputs | Valid. | Role |
|---|---|---|---|---|---|
| C6 | **Matched Score** — MOOSE-Chem (2410.07076, ICLR'25); reused by ResearchBench (2503.21248) | 0–5 LLM rubric of candidate vs. decomposed ground-truth hypothesis: 5 = covers all key points & uses them the same way … 2 = covers one … 0 = none; report Top-MS over ranked list | 1 judge call per (candidate, gold) pair | 2 chemistry-PhD spot-check; ResearchBench extraction 91.9% expert-ok | **[GT-matching]** — adopt as the rubric for our G2 entailment tier (graded, not binary "covers") |
| C7 | **FDR / HDR** — HypoBench (2504.11524) | `FDR = \|Ẑ∩Z\|/\|Z\|` (fraction of true latent features recovered); `HDR = FDR · RC`, RC = mean judge-rated correctness ∈[0,1] | matched sets | programmatic GT (synthetic) | **[Eval]** — FDR *is* our "recall of realized directions" with a correctness gate; use the name |
| C8 | **Hypothesis-ranking pairwise accuracy** — ResearchBench (2503.21248) | accuracy distinguishing the ground-truth hypothesis from **15 negatives**, each pair judged twice (order swap) — Claude-3.5: 81.6% | pairwise judge | expert-validated extraction | **[Eval]** — the published precedent for our G5 distractor-tier evaluation; ours generalizes to typed tiers + confusion matrix |
| C9 | **Claim-F1 + Future-Target Alignment** — ForeSci (2606.00644) | cutoff-aligned KB with post-cutoff papers *hidden*; Prediction Factuality = claim-level F1 vs hidden claim bank; FTA = ranking-alignment vs hidden preferred ranking; + evidence-traceability ∈[0,1] | temporal KB + LLM eval | LLM-evaluated; key finding: **evidence–decision decoupling** (right evidence, wrong research object) | **Prior art we must cite** — closest forward-looking benchmark; see alert below |
| C10 | **Elo with proximity-scheduled matches** — AI co-scientist (Google, 2502.18864) | hypotheses start Elo 1200, logistic update `E_A = 1/(1+10^((R_B−R_A)/400))`; a Proximity agent clusters similar hypotheses to **schedule informative matchups**; top ranks get multi-turn debates, low ranks short comparisons | pairwise debates | Elo concordant with GPQA accuracy; expert-preferred on 11/15 goals; 3 wet-lab validations | Prior art for F2's "informative pairing" idea (see alert); their debate-depth-by-rank = our compute-governor pattern |
| C11 | **Swiss tournament ranking** — Nova (2410.14255) | Claude-3.5 zero-shot pairwise, **Swiss system, 5 rounds, +1/win** (score 0–5); dedup MiniLM cosine >0.8, "similar-to-literature" gate >0.3 | O(5n/2) judge calls | 10 experts, no κ; 3.4× unique novel ideas | Prior art for R2 (see alert) — Swiss-for-ideas is now established practice, not our novelty |
| C12 | **Automated reviewer** — AI Scientist (2408.06292) | NeurIPS form (Soundness/Presentation/Contribution 1–4, Overall 1–10, accept@6), 5 self-reflections × 5-review ensemble + meta-aggregation | ~25 calls/idea (or 1-pass cheap variant) | balanced acc 0.65 vs 0.66 human; F1 0.57 vs 0.49; LLM↔human-mean r=0.18 vs human↔human 0.14 | 1-pass Overall+threshold as a candidate-pool accept filter; judge-validation numbers = context for our E2 |
| C13 | **Viewpoint-graph scoring** — GraphEval (2503.12600, ICLR'25) | decompose ideas into viewpoint-nodes; label propagation `d_i' = (1/Z)(d_i + Σ w_ij d_j)` or 2-layer GNN over BERT-cosine top-k graph; novelty via temporal features + synthetic-plagiarism negatives | 1 decomposition call/idea, then free inference | labels = ICLR outcomes; macro-F1 43.6% vs ~8% prompt baselines | [Feat] — near-free per-candidate scores once graph built; plagiarism-negative trick pairs with our G5 anachronisms |
| C14 | **Pre/post-execution score gap** — Ideation-Execution Gap (Si et al., 2506.20803) | same 1–10 rubric before and after 43 experts *executed* the ideas: AI ideas drop Novelty −1.05, Excitement −1.76, Effectiveness −1.88, Overall −1.98 (all sig.); human ideas ≈ flat | execution study | 58 reviewers, 181 reviews | **Motivation citation** — pre-execution judgment systematically overrates AI ideas; realization grounding (our premise) is the fix |
| C15 | **FutureGen eval stack** (2503.16561) | ROUGE/BLEU/BERTScore/cosine vs author future-work + GPT-4o-mini 1–5 on 6 criteria + novelty 0–10 + NLI hallucination check + binary feasibility | GT = author-stated future work | **κ = 0.30**, τ = 0.36 (weak) | Positioning target (plan §12): author-*promised* GT + overlap metrics ≈ everything our realization-graded ranking improves on; borrow the NLI hallucination check for Direction Cards |

> **⚠ PRIOR-ART ALERT (affects novelty claims in `ResearchHorizon_ideas_Q1-Q4.md`):**
> 1. **R2 (Swiss-Setwise):** Nova (Oct 2024) already runs Si-style Swiss pairwise for idea ranking. Our claim must narrow to the **setwise generalization (groups of c, soft logprob points)** and **tuned R/c**, not "Swiss for ideas."
> 2. **F2 (disagreement-routed Swiss fusion):** AI co-scientist (Feb 2025) already schedules matches by a **proximity graph** and spends debate depth by rank. Ours stays distinct — routing by **cross-agent disagreement** over *heterogeneous rankers*, BT posterior over a **fusion prior**, MRR-calibrated — but co-scientist must be cited and differentiated explicitly.
> 3. **E7 / benchmark framing:** ForeSci (Jun 2026) is a temporally-controlled forward-looking benchmark (hidden post-cutoff KB, claim-F1 + ranking alignment). Our distinctions: per-paper masked future-work grounding, realization-*graded* ranked ground truth, and the **prospective pre-registration arm** (predictions committed before the answers exist anywhere — ForeSci's future is hidden-but-extant). Cite ForeSci and ResearchBench in §0.
> 4. **G5 (distractor tiers):** ResearchBench's GT-vs-15-negatives pairwise accuracy is the precedent; our typed tiers (SSLI/counterfactual/anachronism) + confusion matrix remain the extension.

---

## 5. Diversity / duplication metrics (candidate-pool hygiene)

| # | Metric (source) | Definition | Role |
|---|---|---|---|
| D1 | Duplication rate — Si et al. | cosine (all-MiniLM-L6-v2) > **0.8** ⇒ duplicate; they found only ~5% of 4,000 seeds unique — overgeneration saturates | dedup gate for candidate pool (§3 of plan) |
| D2 | Novelty-boost threshold — SciMON | max cosine to retrieved prior work ≥ **0.6** ⇒ "too similar, rewrite" | candidate filtering; note μ is corpus/embedder-specific — recalibrate on dev |
| D3 | Theme coverage — ours (§R4) | # distinct themes (G2 types or clusters) in top-K | [Badge] for the list; diversity metric in E1 |

---

## 6. Recommended picks (the decision — draft, to finalize with Naz)

**Ground truth (Q1):** realization ladder (C5) built from I1 impact scoring with **I7's field×year percentile normalization** (Hazen percentiles — the canonical fix), I2's year-mean discipline, and I3 support counts; I6 lag as auxiliary. **No LLM-judged dimension enters the primary label** (HindSight ρ=−0.29; C14's post-execution drops; §G4).

**Matching function (G2):** entity/type gates + embedding τ + **C6 Matched-Score 0–5 rubric** as the graded LLM tier (replaces a bare "covers yes/no" entailment call).

**Evaluation metric names to adopt:** MRR/nDCG on graded labels (ours) + **C7 FDR** (recall of realized directions, with RC correctness gate) + **C8-style distractor pairwise accuracy** reported as our typed confusion matrix (G5).

**Direction-Card badges:** N3 KG white-space novelty (best-validated), F1 citation-influence feasibility (+FPS), I6 imminence, D3 theme tag; C15's **NLI hallucination check** (direction vs. paper premises) as a card-level sanity light.

**Fusion/meta-ranker features:** N2, N4 (max-sim), N5, N9 commonness, N11 embedding novelty (cheap), F2 (LLM feasibility — the one judge signal HindSight found forecast-relevant), F4/F5 method-maturity, I4 centrality, I5 horizon priors, I10 predicted concept-pair impact, provenance + support counts, C13 viewpoint-graph scores (near-free).

**Avoid:** N6 Insight Score (circular), C3 flat 3/0 tournament (use BT/Elo instead), any LLM novelty score as a label, N8's full Monte-Carlo machinery (use N9's analytical null instead), CD index as GT (I8 critiques).

---

## 7. Implementation notes

- **APIs:** Semantic Scholar Graph API (citations, venues, author IDs; rate limit ~1 rps unauthenticated — request a key) and/or OpenAlex (no key needed, concepts + citation counts; better for KG features at scale).
- **Embeddings:** SPECTER2 for paper/direction matching (HindSight precedent; their usable cosine band is narrow — 0.91–0.98 — so thresholds must be calibrated, not guessed); all-mpnet or MiniLM for cheap dedup.
- **Normalization discipline:** every citation-based quantity gets field + age normalization (percentile within (field, year) bins) before entering any formula — raw counts are never comparable across our temporal splits.
- **Thresholds to calibrate on dev, never copied:** match τ/θ, dedup cutoff, feasibility λ, lag τ_time.
- **Verify before coding:** AI Idea Bench Eq 7 sign conventions (N1) from their repo; MEDRank edge-decay constant c; SciMuse feature list (their Zenodo dump has the 144 features); MOOSE-Chem's exact Matched-Score rubric text from arXiv:2410.07076 appendix (C6); Nova's 0.3/0.8 thresholds are asserted, not tuned — recalibrate both on our dev set.
- **Must-read prior art before writing the paper's related-work section**: ForeSci (arXiv:2606.00644), ResearchBench (arXiv:2503.21248), Nova (arXiv:2410.14255), AI co-scientist (arXiv:2502.18864), Ideation-Execution Gap (arXiv:2506.20803), Impact4Cast (arXiv:2402.08640), MOOSE-Chem (arXiv:2410.07076), GraphEval (arXiv:2503.12600), FutureGen (arXiv:2503.16561) — **all nine PDFs added to `References/` on Jul 28**; highlighted with reading assignments at the top of `to-do.md`.

---

## 8. Prompt library — DONE (Jul 28)

The `to-do.md` §2–3 prompt tasks are complete: see **`prompts/`** — `generation/` (10 files: IdeaBench Figs 1/2/5, AI Idea Bench G–L/P–Q/R, SciMON few-shot + novelty-boost extracted from the local code, CoI Table 24, ResearchAgent rubric induction), `ranking/` (pointwise / pairwise / setwise / listwise / Swiss scheduler + README with the complexity/order-robustness/logprobs table), a top-level README (forecast-framing rule + shared template slots), and `example_input.md` (shared smoke-test fixture with planted SSLI + anachronism distractors). Every prompt keeps the original [verbatim] block beside our forecast-framed [adapted] block. Remaining: live API smoke-tests (need keys — see `prompts/README.md` status).
