# ResearchHorizon — Literature Survey

*Compiled July 2026. Covers the 24 papers in `References/` (read in full) plus a targeted web survey (scientometrics + 2024–26 LLM idea-evaluation) that surfaced nine additional must-read papers, now being added to `References/`. Companion documents: `metrics.md` (formula-level comparison) and `ResearchHorizon_ideas_Q1-Q4.md` (design proposals; §-references G/R/F/E point there).*

---

## 1. Scope and organization

ResearchHorizon sits at the intersection of four literatures: **(A)** LLM ranking paradigms from IR, **(B)** ranking/recommendation in scientific and structured domains, **(C)** LLM research-idea generation systems, and **(D)** benchmarks and evaluation of generated ideas. Sections 2–5 survey the reference-folder papers by theme; §6 covers the newly surfaced prior art including three alerts that affect our novelty claims; §7 states the remaining white space; §8 is the consolidated bibliography.

Housekeeping: `ivanitskiy-interp.pdf` is a mechanistic-interpretability lecture deck, unrelated to this project — excluded. `MEDRank` appears twice in the folder (identical content), and `SciMON/scimon.pdf` duplicates Wang et al. 2024. The file "Ren et al. - 2017" is actually a **November 2024** preprint (SCaLR, arXiv:2411.04602) — the "2017" comes from the ACM template placeholder; and "Listwise ranking loss.pdf" is the classic **ListNet** paper (Cao et al., ICML 2007).

---

## 2. LLM ranking paradigms (IR core)

**RankGPT — Sun et al., EMNLP 2023 (arXiv:2304.09542).** The canonical *listwise* method: instructional permutation generation over windows of 20 candidates, slid back-to-first. GPT-4 reached nDCG@10 75.59 on TREC-DL19, beating supervised monoT5-3B, and their contamination-controlled NovelEval set showed LLMs can rank content they have never seen. Two durable cautions: output malformation needs handling (repetition/missing/rejection), and listwise ranking is **acutely order-sensitive** — gpt-3.5 falls from 65.8 to 25.2 nDCG when the input order is randomized. Their permutation-distillation result (a 435M student trained on ChatGPT permutations with RankNet loss *beats its teacher*) underpins our planned meta-ranker distillation.

**Pairwise Ranking Prompting (PRP) — Qin et al., Findings of NAACL 2024 (arXiv:2306.17563).** Asks only "which of two candidates is better", queried in both orders with disagreement counted as a tie. PRP-Allpair is a round-robin tournament scored by win counts (Copeland-style) and is nearly **immune to initial order** (72.42 → 72.40 under inversion); moderate open models (FLAN-UL2 20B) match or beat GPT-4-class listwise. All-pairs costs O(N²); heapsort and sliding-window variants trade robustness for O(N log N)/O(kN). The win matrix it produces is the sufficient statistic for Bradley–Terry fitting — the bridge we use between per-agent tournaments and fusion.

**Setwise prompting — Zhuang et al., SIGIR 2024 (arXiv:2310.09497).** Generalizes pairwise to "pick the most relevant of c candidates", plugging c-way selections into heapsort/bubblesort. The first systematic efficiency comparison across all paradigms: setwise-heapsort cuts ~62% of pairwise-heapsort's cost at −0.8% nDCG and is robust across original/inverted/random orders. Their efficiency accounting template (#calls, tokens, latency, $) is what our evaluation adopts.

**SCaLR — Ren et al., arXiv:2411.04602 (Nov 2024).** Fine-tunes a listwise ranker that emits *globally comparable scores* in one forward pass, **self-calibrated** against point-view scores computed position-free in the same pass. Calibration is load-bearing (−4.7 nDCG without it); the model merges sublists without sliding windows and is position-bias-free by construction. The transferable idea for us is prompt-level: anchor each agent's listwise scores to its own pointwise scores so scores are comparable across sublists and across agents before fusion.

**ListNet — Cao et al., ICML 2007.** The theoretical ancestor: permutation probabilities under Plackett–Luce, top-one marginals computable in O(n), cross-entropy between target and model softmax distributions. Two things we use directly: **soft PL targets** encode graded/tied ground truth without arbitrary tie-breaking (see math box under §G1 of the ideas doc), and their classic plot of pairwise loss *diverging* from NDCG during training justifies listwise surrogates when the selection criterion is a list metric (dev MRR).

**Zero-shot rankers for recommendation — Hou et al., ECIR 2024 (arXiv:2305.08845).** Frames recommendation as conditional ranking (history = condition, candidates = list) and documents two biases we inherit: **position bias** (ground truth near the list tail is under-ranked; fixed by B=3 order-shuffled bootstrap rounds merged) and **popularity bias** (frequent items over-ranked; mitigated by conditioning on fewer, more recent history items). Both mitigations are in our per-agent recipe (§R1).

**REARANK — Zhang et al., arXiv:2505.20046.** A listwise re-ranking agent trained with GRPO to reason (`<think>`) before ranking (`<answer>`). Two transferable results: a **normalized relative-improvement NDCG reward** beats raw or difference rewards, and **multi-sampling augmentation** turns 179 labeled queries into 12k training instances — our recipe for training fusion on few labeled papers (§G6). REARANK-7B matches GPT-4 on TREC/BEIR and surpasses it on reasoning-intensive BRIGHT.

**SciRerankBench — Chen et al., arXiv:2508.08742.** Benchmarks 13 rerankers × 11 LLMs on scientific RAG with three diagnostic distractor types: random noise (NC), **semantically-similar-but-logically-irrelevant (SSLI)**, and counterfactual (CC). Cross-encoders (joint query–candidate encoding) win on SSLI; local-window listwise agents struggle to aggregate evidence. Sharpest finding for us: **high Recall@10 often fails to improve final answers** — ranking metrics are not downstream utility, motivating our two-level evaluation (§E8). Their distractor taxonomy is the blueprint for our typed distractor tiers (§G5).

---

## 3. Ranking and recommendation in scientific / structured domains

**MEDRank — Herskovic et al., Int. J. Medical Informatics 2011.** PageRank/TextRank over a document's concept graph (edges: distance-decayed co-occurrence `1/c^d` or distributional similarity) identifies *central* concepts; inserted into NLM's indexing pipeline it lifted major-heading recall 26–30% and **matched human indexers in a blinded preference study** (p=0.36) that used vote-prior equalization — our template for the human study (§E5). Establishes graph centrality as a strong non-LLM ranking signal (§F3).

**ILCiteR — Roy & Han, LREC-COLING 2024 (arXiv:2403.08737).** Evidence-grounded citation recommendation: rank (evidence span, paper) pairs, with a distantly-supervised evidence bank mapping spans to cited papers with **support counts**. Two results we build on: their temporal, leakage-free split (eval = newest papers excluded from the bank), and **conditional rank ensembling** — choosing which signal to trust based on query characteristics beats unconditional ensembling — the small-scale empirical precedent for our context-conditioned tensor fusion.

**HawkSight-LLM — Amisu (pilot study).** A controllability result: schema-constrained extraction (96% valid JSON with hard-grammar prompting), a **deterministic recommender core the LLM cannot override**, and grounded explanation with zero fabricated institutions across 75 lists. The architecture lesson — LLM interprets and explains, deterministic core selects, candidates can never be invented — is our Direction-Card contract, and its metrics (grounding accuracy, trade-off transparency, selection stability under paraphrase) transfer to §E8.

---

## 4. LLM research-idea generation systems

**SciMON — Wang et al., ACL 2024 (arXiv:2305.14259).** Retrieval-grounded idea generation with iterative novelty boosting (rewrite until max similarity to prior work < 0.6). Defines gold via IE (`used-for` relations), strict temporal splits (train <2021 / dev 2021 / test 2022), and a **trivial-overlap filter** so gold isn't recoverable from surface text — all adopted in §G2/§G3. Sobering calibration: in 85% of comparisons, human judges rated the *real* paper's idea above the best generated one.

**Chain of Ideas + Idea Arena — Li et al., arXiv:2410.13185.** Organizes literature into progressive chains (forward extension follows the citation graph — the same signal as our realization labels) and evaluates with **Idea Arena**: round-robin pairwise battles, order-reversed, per-criterion verdicts, Elo aggregation, with human↔GPT-4o agreement of 70.8%. CoI matches human ideas on novelty/significance but lags on feasibility — the recurring pattern.

**ResearchAgent — Baek et al., arXiv:2404.07738.** Idea = (problem, method, experiment) generated over a citation-graph survey plus an **entity co-occurrence knowledge store**, refined by reviewing agents whose rubrics are **induced from ~10 human-scored pairs** — the cheapest published judge-calibration method (§E2). Reports the agreement ceilings we benchmark against (human–human Spearman 0.67–0.83; human–model pairwise κ up to 0.71).

**MLR-Copilot — Li et al., arXiv:2408.14033.** Idea → experiment plan → executed implementation, with an IdeaAgent RL-tuned on OpenReview data against separate novelty/feasibility/effectiveness reward models. Source of the **resource-conditioned 1–10 feasibility anchors** (shared with Si et al.'s review form) that make feasibility judgments concrete.

**VirSci — Su et al., arXiv:2410.09403.** Multi-agent "virtual scientist teams" over real bibliometric ecosystems. Methodologically important to us for its evaluation: a **contemporary (post-cutoff) paper database as pseudo-ground-truth**, scored by contemporary dissimilarity (alignment with where the field actually went) and **contemporary impact (CI), year-normalized** — the direct precursor of our realization/impact labels, validated at r = 0.52 against human novelty ratings. Also: team diversity helps with an inverted-U (peak at 8 agents, 5 turns) — evidence for few heterogeneous rankers over large homogeneous committees.

**MOOSE-Chem2 — Yang et al., arXiv:2505.19209 (slide deck in folder — pull the full paper before citing details).** Hierarchical heuristic search for fine-grained hypothesis discovery: rank coarse structure first, refine within winners — the "smoothed reward landscape" argument behind our theme→direction ranking (§R4). Committee findings: **same-LLM ensembles tie 82% of the time** (heterogeneity is the productive axis) and a weak committee member *drags down* the optimum (quality-weighted fusion, §F1/F4). Their soft/hard component recall against decomposed gold is one of our matching-function tiers.

**SciMuse — Gu & Krenn, arXiv:2405.17044.** The largest expert evaluation of AI-generated ideas: 4,451 personalized suggestions rated 1–5 by ~110 Max Planck research-group leaders. **Time-stamped knowledge-graph features alone predict interest** (supervised NN AUC 0.645, top-1 precision 70% vs 23% base rate; zero-shot GPT-4o pairwise+Elo AUC 0.673) — less-popular concepts and closer fields rate as more interesting. This is our quantitative-novelty channel (§G4) and the strongest evidence that cheap graph features carry head-of-list ranking signal.

**Heuristic-Based Ideation — Liu, Liu & Tan, ICLR Blogposts 2026.** Twenty ideation heuristics in five families, mapped onto existing systems and rated by 13 researchers across disciplines. We reuse the taxonomy as a **direction-type system** for stratified matching, per-type realization base rates, and Direction-Card rationale tags (§G2/§G5).

---

## 5. Benchmarks and evaluation of generated ideas

**IdeaBench — Guo et al., arXiv:2411.02429.** 2,374 influential 2024 biomedical targets (post-cutoff) + filtered references; gold idea normalized from the abstract by an LLM prompt; **Insight Score** = the real idea's relative rank among generated ideas under a GPT-4o listwise ranker. Models exceed the real paper on judged novelty (GPT-4o 0.766) but collapse on feasibility (≤0.305). No ranked recommendation, no rank-aware metrics, single judge — and the Insight Score is circular for our purpose (it trusts exactly the judgment we're replacing).

**AI Idea Bench 2025 — Qiu et al., arXiv:2504.14191.** 3,495 post-cutoff AI-conference targets; gold = anonymized (motivation, experiment plan) — the closest published analog to our masking protocol; novelty scored with a **temporal split + contemporary-citation weighting** (Eq 7) and feasibility via **citation-influence of supporting methods** (Inf = 1 − e^(−x/50)). Evaluation matches only the *best* generated idea (max-over-set) against a *single* gold — no ranked list, no MRR/nDCG. Their pairwise judge prompt ships with explicit anti-position/anti-verbosity instructions we import.

**Can LLMs Generate Novel Research Ideas? — Si, Yang & Hashimoto, arXiv:2409.04109.** The landmark controlled human study (49 ideas/condition, 79 expert reviewers, style-normalized): AI ideas judged **more novel** (+0.76–0.9, p<0.05) with a non-significant feasibility deficit. Three engineering results we rely on: pairwise LLM judging beats pointwise (71.4% on ICLR accept/reject pairs vs ~chance pointwise); their **Swiss-system tournament** (pair by accumulated score, +1/win, N=5 rounds tuned, non-monotone in N); and every LLM judge tested fell below the already-low 56.1% human split-half agreement — single-judge ranking is not trustworthy, which is the core motivation for multi-ranker fusion. Also: overgeneration saturates (~5% of 4,000 seeds unique after MiniLM-0.8 dedup).

**HindSight — Jiang, arXiv:2603.15164.** The first time-split, **impact-grounded** idea evaluation: ideas matched (SPECTER2, θ=0.96) to actually-published later papers scored h(p) = 0.6·normalized-citations + 0.4·top-venue. A retrieval-grounded generator beats a vanilla baseline 2.5× on HindSight (p<0.001) while an LLM judge sees *no difference* (p=0.584) — and judged novelty **anti-correlates with realization** (ρ = −0.291). Single scalar per idea, topic-wide pool, no ranking, no multi-agent — but the ground-truth recipe our realization ladder extends (§G1).

---

## 6. Newly surfaced prior art (July 2026 web survey — not in the original folder)

### 6.1 Three prior-art alerts affecting our novelty claims

**Alert 1 — Nova (arXiv:2410.14255): Swiss tournaments for idea ranking already exist.** Nova's pipeline ranks generated ideas with a Claude-3.5 zero-shot pairwise ranker run as a **Swiss-system tournament, 5 rounds, +1 per win**, alongside MiniLM dedup at cosine 0.8 and a "similar-to-literature" gate at 0.3. Together with Si et al.'s ranker, this makes "Swiss for ideas" **established practice**. Consequence: the ideas doc's R2 claim narrows to what remains genuinely new — the **setwise generalization** (score-adjacent groups of c per call, soft points from label logprobs, R and c tuned on dev MRR) and its use *inside each ranking agent* of a fusion system.

**Alert 2 — AI co-scientist (Google, arXiv:2502.18864): informative match scheduling already exists.** Their ranking agent runs an **Elo tournament (init 1200, logistic update)** where a *proximity graph* over hypotheses schedules informative matchups, and compute is rank-dependent (multi-turn debates at the top, quick comparisons below) — validated by Elo–GPQA concordance, expert preference on 11/15 goals, and three wet-lab results. This is adjacent to F2's "pair near-equals for maximum Fisher information" argument. F2 stays distinct on three axes — routing by **cross-agent disagreement** between *heterogeneous rankers* (not self-similarity), Bradley–Terry updates over a **fusion prior** (RRF/tensor standings, not cold-start Elo), and calibration of the whole stage to **dev MRR** — but co-scientist must be cited and explicitly differentiated in the paper.

**Alert 3 — ForeSci (arXiv:2606.00644): a temporally-controlled forward-looking benchmark now exists.** ForeSci evaluates research-direction forecasting against a **cutoff-aligned knowledge base whose post-cutoff papers are hidden** and used only for validation: claim-level F1 against a hidden claim bank, Future-Target Alignment (ranking agreement), evidence traceability, and an LLM reviewer. Its headline diagnosis — **evidence–decision decoupling** (agents cite the right evidence but choose the wrong research object) — is a failure mode our evidence-gated matching should measure. Our remaining distinctions: **per-paper masked future-work grounding** (ForeSci is task/domain-level), a **realization-graded ranked ground truth** (grades + impact, not claim F1), and the **prospective pre-registration arm** (§E7) — our predictions are committed *before the answers exist anywhere*, whereas ForeSci's future is hidden-but-extant. ForeSci and ResearchBench go into §0 of the related work.

### 6.2 The other must-reads

**Ideation-Execution Gap — Si, Hashimoto & Yang, arXiv:2506.20803.** 43 experts *executed* AI or human ideas (>100h each); 58 reviewers re-scored pre- vs post-execution. **AI ideas dropped on every dimension** — novelty −1.05, excitement −1.76, effectiveness −1.88, overall −1.98 (all significant) — while human ideas stayed ≈flat. The strongest motivation citation for realization-grounded ranking: pre-execution judgment systematically overrates AI ideas.

**ResearchBench — arXiv:2503.21248.** 1,386 papers (2024-only, 12 disciplines) decomposed into inspiration retrieval (Hit Ratio), hypothesis composition (0–5 Matched Score), and **hypothesis ranking as pairwise accuracy of ground truth vs. 15 negatives** (Claude-3.5: 81.6%) — the published precedent our typed distractor tiers (§G5) extend into a confusion matrix.

**MOOSE-Chem — Yang et al., ICLR 2025 (arXiv:2410.07076).** Source of the **Matched Score 0–5 rubric** (graded key-point coverage of a candidate vs. decomposed gold hypothesis) that we adopt as the LLM tier of our matching function (§G2), replacing a binary "covers" call.

**Impact4Cast — Gu & Krenn, MLST 2025 (arXiv:2402.08640).** Forecasts whether an unconnected concept pair will be co-studied *and* accrue ≥IR citations within 3 years (141 KG features, NN, AUC up to >0.9; predecessor Krenn & Zeilinger PNAS 2020, AUC 0.85). The closest published analog to "will this direction be realized impactfully" — but concept-pair level, no paper conditioning, no ranked recommendation, no LLM agents. Cite in §0; reuse features in fusion (§F3).

**GraphEval — arXiv:2503.12600 (ICLR 2025).** Decomposes ideas into viewpoint-nodes and scores them with label propagation or a small GNN over a BERT-cosine graph — near-free per-candidate inference after one decomposition call (+14–42 F1 over prompt baselines on ICLR outcomes). Its synthetic-plagiarism negatives rhyme with our anachronism distractors; a cheap third scoring source for fusion.

**FutureGen — arXiv:2503.16561.** The direct task predecessor: generates (not ranks) future-work sections, evaluated by ROUGE/BERTScore overlap with **author-stated** future work plus an LLM judge (κ=0.30 vs humans) and an NLI hallucination check (worth borrowing for Direction Cards). Everything our framing improves on: author-*promised* rather than realized ground truth, overlap rather than rank-aware metrics, no candidates, no fusion.

*(Also catalogued in `metrics.md` §4.2 but secondary: LiveIdeaBench arXiv:2412.17596, HypoBench arXiv:2504.11524 — its FDR = recall-of-realized-directions naming we adopt, Scideator arXiv:2409.14634, Auto-Arena arXiv:2405.20267, execution-grounded search arXiv:2601.14525, AI Scientist arXiv:2408.06292 + v2 arXiv:2504.08066.)*

---

## 7. Synthesis: the white space after the full survey

| Capability | Closest prior art | What it lacks / where we differ |
|---|---|---|
| Masked-future-work task | FutureGen | generation + overlap metrics vs. our *ranked recommendation* against realized outcomes |
| Realization-grounded scoring | HindSight; VirSci; Impact4Cast | single scalar / corpus-level / concept-pair level — none produce a *per-paper ranked, graded* gold set from masked statements + forward citations |
| Forward-looking benchmark | ForeSci; ResearchBench; AI Idea Bench | hidden-but-extant futures; single-gold matching — vs. our realization ladder, typed distractors (incl. anachronisms), and truly prospective pre-registration |
| Idea ranking with tournaments | Si et al.; Nova; SciMuse; AI co-scientist | single-ranker Swiss/Elo — vs. our setwise-Swiss inside agents + disagreement-routed BT refinement *across heterogeneous rankers*, seeded by a fusion prior, calibrated to dev MRR |
| Rank fusion | RRF (Cormack 2009); LLM-Blender; ILCiteR | global weights or single-signal conditioning — vs. context-conditioned low-rank tensor + CMTF with scientometric side-info, selected by MRR |
| Judge reliability evidence | Si 2024/2025; HindSight; CoI | we *consume* these findings: no LLM judgment enters the primary label; judges are third-family and validated (κ, τ-sweeps) |

**Still claimable after this survey:** (1) the per-paper, realization-**graded ranked** ground truth (ladder + PL soft targets + typed distractors + anachronism negatives); (2) heterogeneous-agent ranking with **Swiss-setwise** scheduling and **disagreement-routed** fusion refinement over an MRR-calibrated fusion prior; (3) the tensor/CMTF fusion with scientometric side-information (unchallenged); (4) the **prospective pre-registration arm**; (5) the honest, compute-aware ablation ladder (MRR-per-dollar). Items 2's components each have near-neighbors now (Nova, co-scientist) — the specific combination and the fusion setting remain open, but the paper must cite and differentiate all of §6.1 explicitly.

---

## 8. Consolidated bibliography

**In `References/` (original 24, thematic order):** Sun et al. 2023 RankGPT (arXiv:2304.09542, EMNLP) · Qin et al. 2024 PRP (arXiv:2306.17563, NAACL-Findings) · Zhuang et al. 2024 Setwise (arXiv:2310.09497, SIGIR) · Ren et al. 2024 SCaLR (arXiv:2411.04602 — folder file mislabeled "2017") · Cao et al. 2007 ListNet (ICML — folder file "Listwise ranking loss.pdf") · Hou et al. 2024 LLMRank (arXiv:2305.08845, ECIR) · Zhang et al. 2025 REARANK (arXiv:2505.20046) · Chen et al. 2025 SciRerankBench (arXiv:2508.08742) · Herskovic et al. 2011 MEDRank (IJMI 80(6) — duplicated in folder) · Roy & Han 2024 ILCiteR (arXiv:2403.08737, LREC-COLING) · Amisu ~2025 HawkSight-LLM (pilot, no arXiv) · Wang et al. 2024 SciMON (arXiv:2305.14259, ACL) · Li et al. 2024 Chain of Ideas (arXiv:2410.13185) · Baek et al. 2024 ResearchAgent (arXiv:2404.07738) · Li et al. 2024 MLR-Copilot (arXiv:2408.14033) · Su et al. 2025 VirSci (arXiv:2410.09403) · Yang et al. 2025 MOOSE-Chem2 (arXiv:2505.19209 — slides only, fetch full paper) · Gu & Krenn 2025 SciMuse (arXiv:2405.17044) · Liu, Liu & Tan 2026 Heuristic-Based Ideation (ICLR Blogposts) · Guo et al. 2024 IdeaBench (arXiv:2411.02429) · Qiu et al. 2025 AI Idea Bench (arXiv:2504.14191) · Si, Yang & Hashimoto 2024 (arXiv:2409.04109) · Jiang 2026 HindSight (arXiv:2603.15164) · [excluded: ivanitskiy-interp.pdf].

**Being added to `References/` (the nine must-reads):** ForeSci (arXiv:2606.00644) · ResearchBench (arXiv:2503.21248) · Nova (arXiv:2410.14255) · Gottweis et al., AI co-scientist (arXiv:2502.18864) · Si et al., Ideation-Execution Gap (arXiv:2506.20803) · Gu & Krenn, Impact4Cast (arXiv:2402.08640, MLST 2025) · Yang et al., MOOSE-Chem (arXiv:2410.07076, ICLR 2025) · GraphEval (arXiv:2503.12600, ICLR 2025) · FutureGen (arXiv:2503.16561).

**Cited from the scientometrics survey (formulas in `metrics.md` §1.2/§3.2, PDFs not stored):** Uzzi et al. 2013 (Science) · Lee, Walsh & Wang 2015 (Research Policy) · Wang, Veugelers & Stephan 2017 (Research Policy) · Shibayama, Yin & Matsumoto 2021 (PLoS ONE) · Funk & Owen-Smith 2017 / Wu, Wang & Evans 2019 / Park, Leahey & Funk 2023 + 2024 critiques (CD index) · Foster, Rzhetsky & Evans 2015 (ASR) · Waltman et al. 2011 (MNCS) · Ke et al. 2015 (Sleeping Beauty, PNAS) · Krenn & Zeilinger 2020 (PNAS) · de Solla Price 1970 · Cormack et al. 2009 (RRF, SIGIR).
