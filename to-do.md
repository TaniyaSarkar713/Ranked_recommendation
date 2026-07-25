# To-Do — week of Jul 27 (due Fri, Jul 31, 2026)

## Nourah — ground truth (Q1) + human expert data

- [ ] Read ideas doc §G1–G6 (realization ladder, entity matching, temporal splits, distractors).
- [ ] Pick some **Ground-truth pilot papers (~20–50 papers):** papers with explicit Future-Work sections (ACL Anthology / cs.CL slice of S2ORC or unarXive, 2019–2023 so the realization window has closed). For each: mask the future-work section, split into atomic directions, pull forward citations via Semantic Scholar API.
  - [ ] Measure: how many directions per paper (avg), how often a direction is matchable to a forward-citing paper (Grade 2), how often the match clears an impact bar (Grade 3), % papers with ≥1 realized direction within 24–30 months.
  - [ ] Note failure cases: future work too vague to atomize, no forward citations, matching ambiguity.
  - [ ] To evaluate the ideas hands-on.
- [ ] **Human expert data scoping:**
  - [ ] How many people × papers?
  - [ ] Two uses: (a) small blinded preference ranking; (b) evaluation of our method.
  - [ ] Check whether we need ethics/IRB sign-off for the expert study (likely exempt, confirm).
- **Key references:** hindsight.pdf (SPECTER2 matching, θ sweep, 0.6·citations+0.4·venue impact); AI Idea Bench appendix G–L (input reconstruction / anonymization); IdeaBench (target-paper selection rules); ILCiteR (evidence bank + support counts, leakage-free temporal split).

---

## Taniya + Naz — metric formulas (novelty / feasibility / impact) + prompts

### 1. Collect & compare the formulas (→ decide: ground-truth label vs. Direction-Card badge vs. fusion feature)

Some ideas and references: 
| Metric | Source (paper, location) |
|---|---|
| Novelty (realization-aware) | AI Idea Bench 2025, Eq 7 |
| Novelty (vs past/future DBs) | VirSci (Su et al.), §metrics |
| Novelty (KG white-space) | SciMuse (Gu & Krenn) |
| Feasibility (citation influence) | AI Idea Bench 2025, Eq 8–10 |
| Feasibility (rubric anchors) | MLR-Copilot Appendix B.2; Si et al. Appendix E |
| Impact of realization | hindsight.pdf; ILCiteR; MEDRank |
| Insight score | IdeaBench, §Insight Score |

- [ ] For each: list required inputs (embeddings, citation counts, KG, LLM calls), implementation effort, formula, and ideas about where it plugs in (Q1 label / badge / fusion feature per §G4).

### 2. Prompts for *ranking the future directions*

- [ ] IdeaBench **Fig 1** (grounded idea generation from reference abstracts), **Fig 5** (gold-idea normalization — reuse for atomizing future-work statements), **Fig 2** (listwise ranking by a user-defined quality indicator — directly adaptable to "rank by expected realization").
- [ ] AI Idea Bench **Appendix G–R**: topic extraction + anonymization (G–L), MCQ matching (P–Q), pairwise competition judge with anti-position/length-bias instructions (R).
- [ ] SciMON (local copy in `References/SciMON/`): few-shot generation prompts in `models/GPTFS+NBR/`, novelty-boosting loop prompt (§3.3, μ=0.6).
- [ ] Chain of Ideas **Table 24** (pairwise judge, 0/1/2 per criterion) + **Table 5** criteria definitions; ResearchAgent **Tables 12–15** (induced 5-level rubrics).
- [ ] Adapt everything to **forecast framing** (§R3): "which direction is the community most likely to pursue and publish impactfully?" — not "which is most novel?"
- [ ] Output: `prompts/generation/` in the repo, one file per prompt, with source attribution.

### 3. Ranking-paradigm prompts (pointwise / pairwise / setwise / listwise)

- [ ] **Best single resource:** Zhuang et al.'s repo **github.com/ielab/llm-rankers** — reference implementations of pointwise (qlm, yes/no), pairwise (allpair, heapsort, bubblesort), setwise, listwise. Extract prompt templates + sorting logic.
- [ ] Pairwise: PRP paper §3 exact template ("which of the following two passages is more relevant…") + **both-orders + tie protocol**.
- [ ] Listwise: RankGPT prompt (Sun et al., Appendix A.5; repo **github.com/sunnweiwei/RankGPT**) — permutation output format `[i] > [j]`, sliding window w=20/s=10; REARANK repo **github.com/lezhang7/Rearank** for the reasoning-then-rank (`<think>/<answer>`) variant.
- [ ] Recsys-style listwise + bootstrapping: Hou et al. repo **github.com/RUCAIBox/LLMRank** (condition + candidate-list prompt, B=3 shuffle-and-merge).
- [ ] Swiss pairwise tournament: Si et al. repo **github.com/NoviScl/AI-Researcher** (pairing by accumulated score, N=5 rounds).
- [ ] Setwise: Zhuang §prompt (Fig 1d) — single-token labels; note which variants need logprobs (GPT/open models only, not Claude — ideas doc §Q6).
- [ ] Output: `prompts/ranking/` with one file per paradigm, adapted to direction-ranking (candidates = direction statements, condition = comprehension summary), + a README noting call complexity and order-robustness per paradigm (table in ideas doc §Q2).

**Done when:** `metrics.md` + `prompts/` are in the repo and each prompt has been smoke-tested once on one example paper.

---

## Next in the pipeline

- Swiss-Setwise implementation (§R2) and disagreement-routed fusion (§F2) — needs the prompts above first.
- Prospective pre-registration arm (§E7) — after the retrospective pipeline works end-to-end.
