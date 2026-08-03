# Swiss scheduler — Swiss-Setwise per-agent ranking + disagreement-routed fusion

**Sources:** Si et al. (arXiv:2409.04109) §ranker + repo `github.com/NoviScl/AI-Researcher` (Swiss pairwise: pair by accumulated score, +1/win, N=5 rounds tuned on ICLR proxy — gap non-monotone: 0.95 @4, **1.73 @5**, 1.30 @6). Same family: Nova (arXiv:2410.14255) 5-round Swiss; AI co-scientist (arXiv:2502.18864) proximity-scheduled Elo — **both must be cited; our additions are the setwise groups, soft points, and cross-agent disagreement routing** (see metrics.md §4.2 prior-art alert).
**Role:** (a) per-agent tournament ranking (§R2); (b) fusion refinement stage over contested candidates (§F2).

## (a) Swiss-Setwise per-agent ranking [ours, §R2]

```
seed:    s(d) ← retrieval prior (or one listwise bootstrap pass)
repeat R rounds:
    sort candidates by s; partition into score-adjacent groups of c (rotate to avoid rematches)
    for each group: one setwise call (ranking/setwise.md) → within-group order
    points:  hard = (c − rank_in_group);  soft = c · P(label) from logprobs   [§L2, GPT side]
    s(d) += points
final:   rank by s; ties → prior order.  Fit Bradley–Terry on all within-group outcomes
         for fusion-grade strengths + uncertainty.
```

- Cost: **O(R·N/c) calls** (N=50, c=5, R=4 → 40 calls vs 2,450 for all-pairs-both-orders).
- Interpolates: c=2 → Si/Nova Swiss; c=N, R=1 → single-call listwise.
- **Tune R and c on dev MRR** — Si's non-monotone table says more rounds ≠ better.
- Group-membership order shuffled per round; label assignment rotated (§L6).

## (b) Disagreement-routed fusion refinement [ours, §F2]

```
input:   GPT list r_G, Claude list r_C, free fused standings s0 (RRF or tensor)
contest: pairs (i,j) adjacent in s0 with |s0_i − s0_j| < ε  AND  agents disagree on order
         (sign(r_G(i)−r_G(j)) ≠ sign(r_C(i)−r_C(j)));  restrict to top-2K zone
         priority ∝ |1/rank_i − 1/rank_j|          # MRR sensitivity
repeat until budget/stability:
    Swiss-pair the contested set by current standings
    each match: BOTH agents judge (pairwise.md prompt), both orders; soft votes where logprobs exist
    update: Bradley–Terry with prior = s0 (pseudo-observations, weight w0);
            per-agent votes weighted by dev-set solo MRR                     [§F1 quality weighting]
stop:    top-K membership+order stable for a full round, or budget exhausted (anytime)
```

- **Why Swiss pairing:** BT Fisher information p(1−p) peaks at p≈0.5 — near-equal matches are maximally informative; round-robin wastes budget on blowouts, knockout kills strong candidates on one noisy loss (best judge ≈ 71% accurate — Si et al.).
- **Correlated-error guard:** audit a random sample of *settled* pairs vs dev ground truth (settled-pair error rate); persistent 1–1 splits get a third-family tie-break vote (PMI source, pointwise.md (c)).
- Report as **MRR-vs-calls curves** at matched budgets vs RRF/tensor alone (§E1 MRR-per-dollar) — the registered hypothesis is that lift concentrates on high-disagreement papers (ILCiteR conditional-ensembling precedent).
