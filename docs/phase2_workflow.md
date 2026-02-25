# Phase 2 Workflow: Structural Characterization of Expert Play

**Status**: COMPLETE (Feb 24, 2026). Post-audit corrections applied.

## Results Summary

Phase 2 ran four experiments to explain WHY bc_vs_bc ≈ V* despite TV(BC, Nash) = 0.99. Two clean findings, two confounded:

| Experiment | Hypothesis | Key Result | Verdict |
|-----------|-----------|-----------|---------|
| SVD | Low-rank structure | Eff rank 2.9, payoff-weighted TV 0.019 | **CONFIRMED** |
| QRE | BC ≈ QRE at moderate λ | Best TV = 0.68, 90% at λ=0.01 | **REJECTED** |
| Indifference | Ecological adaptation | Ratio 1.75 (confounded by Nash properties) | **UNINFORMATIVE** |
| Regret | Population-rationality | Regret ratio 1.32 (confounded by Nash properties) | **UNINFORMATIVE** |

**Post-audit corrections**:
- SVD: Replaced broken payoff-space TV projection (clip+renormalize) with payoff-weighted TV. Result is stronger: 0.019 vs original 0.99 (50× collapse).
- QRE: Added convergence tracking. 100% converged at λ≤1, 93% at λ=2, 28% at λ=5, <10% at λ≥10. High-λ data excluded.
- Indifference/Regret: Nash by definition makes opponents indifferent. This confounds both experiments — random strategies show similar ratios (~1.69, ~1.39). Demoted from "negative findings" to uninformative.
- SVD caveat: dynamics model d_action=32 bottleneck may amplify low-rank structure.

**Updated narrative**: "Expert play achieves Nash-level payoffs because VGC turn-1 games have low effective strategic dimension (~3 out of ~122 actions). Payoff-weighted TV between BC and Nash is just 0.019. Most strategy variation lies in the null space of the payoff matrix."

**Milestone writeup**: Feb 25. Poster: Mar 11. Report: Mar 17.

---

## Context

Phase 1 found experts are individually exploitable (~1.4) but collectively near-Nash (bc_vs_bc ≈ V*). The "symmetric error cancellation" explanation was deflated — opposite-sign cross-play gaps are trivially guaranteed by the minimax theorem. Phase 2 replaces the flawed narrative with four experiments grounded in established game theory, answering: **why does bc_vs_bc ≈ V* despite TV(BC, Nash) = 0.99?**

---

## Architecture

```
Phase 0 (GPU, ~3 min)          Phase 1 (CPU, parallel)
┌──────────────────────┐       ┌─────────────────┐
│ scripts/phase2_cache │       │ Exp 1: SVD      │──→ results/phase2/svd.json
│                      │──→────│ Exp 2: QRE      │──→ results/phase2/qre.json
│ Build 500 payoff     │ .pkl  │ Exp 3: Indiff   │──→ results/phase2/indifference.json
│ matrices + strategies│       │ Exp 4: Regret   │──→ results/phase2/regret.json
└──────────────────────┘       └─────────────────┘
                                        │
                                        ▼
                               Phase 2 (write up)
                               ┌─────────────────┐
                               │ Update docs:     │
                               │  PROJECT_BIBLE   │
                               │  why_pivot       │
                               │  phase2_workflow  │
                               └─────────────────┘
```

**Key insight**: Payoff matrices are not cached from Phase 1. We build them once in Phase 0, cache to `results/phase2/cache.pkl` (~25MB), then all four experiments read from cache (CPU-only, no GPU, fully parallel).

---

## Phase 0: Build and Cache (GPU)

**Script**: `scripts/phase2_cache.py`
**Time**: ~3 min (500 matchups on RTX 4080 Super)
**Reuse**: `_build_matchups()` pattern from `scripts/smoothed_br.py` (lines 42-141) — identical GPU phase.

```
Input:  bc_001/best.pt, dyn_001/best.pt, test.jsonl, seed=42, n=500
Output: results/phase2/cache.pkl
```

**Cache format**: pickle of `list[dict]`, each dict:
```python
{
    "idx": int,                    # dataset index (matches existing matchup_details.jsonl)
    "R": np.ndarray,               # (n1, n2) float64 payoff matrix
    "bc_p1": np.ndarray,           # (n1,) BC strategy for P1
    "bc_p2": np.ndarray,           # (n2,) BC strategy for P2
    "nash_p1": np.ndarray,         # (n1,) Nash strategy for P1
    "nash_p2": np.ndarray,         # (n2,) Nash strategy for P2
    "game_value": float,           # Nash game value
}
```

**Implementation**: Copy `_build_matchups()` from smoothed_br.py (already builds R + BC + Nash for both sides). Add `pickle.dump()` at the end. Use same seed=42 and same 500 indices as all other scripts.

---

## Experiment 1: SVD / Effective Rank

**Script**: `scripts/phase2_svd.py` (CPU-only, reads cache)
**Time**: ~10 seconds (SVD of 500 matrices ~100×100 is trivial)
**Output**: `results/phase2/svd.json`

### What to compute per matchup:
- `U, S, Vt = np.linalg.svd(R)` (full SVD)
- **Effective rank** at 90%, 95%, 99% cumulative energy: `k such that sum(S[:k]²) / sum(S²) >= threshold`
- **Top-k variance fractions**: fraction of Frobenius norm in top k=1,2,3,5,10 singular values
- **Spectral gap**: `S[0] / S[1]` (how dominant is the first component)
- **"Payoff-space TV"**: project BC and Nash into top-k SVD subspace, compute TV in that subspace

### What to compute in aggregate:
- Mean/median effective rank (with bootstrap CIs) at each threshold
- Correlations: effective_rank vs {exploitability, game_size, Nash support size}
- Distribution of top-1 variance fraction (how "rank-1" are these games?)

### Motivation (for the doc):
If effective rank is low (3-5 out of 100+), then the ~200 nominal actions collapse to ~5 strategic dimensions. The high TV distance (0.99) between BC and Nash is inflated by payoff-irrelevant dimensions. bc_vs_bc ≈ V* becomes structurally expected: most strategy pairs produce similar payoffs because R has a large null space.

**Connection**: Lipton, Markakis & Mehta (2003) — in rank-r games, Nash support ≤ r+1. If effective rank ≈ 3, this predicts Nash support ≈ 4, close to our observed 2.7.

---

## Experiment 2: QRE Path Analysis

**Script**: `scripts/phase2_qre.py` (CPU-only, reads cache)
**Time**: ~5-10 min (100 matchups × 12 λ values × convergence iterations)
**Output**: `results/phase2/qre.json`

### What to compute per matchup (use 200 matchups for speed):
For each λ ∈ {0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 25, 50, 100}:
- **QRE computation**: iterate simultaneous softmax BR with temperature 1/λ until convergence (TV < 1e-6 or max 2000 iterations)
  ```python
  x, y = uniform, uniform  # start from uniform (not BC)
  for t in range(max_iter):
      x_new = softmax(λ * R @ y)       # P1 quantal response
      y_new = softmax(-λ * x @ R)      # P2 quantal response (negate for minimizer)
      if TV(x_new, x) + TV(y_new, y) < 1e-6: break
      x, y = x_new, y_new
  ```
- **QRE game value**: `x_qre @ R @ y_qre`
- **Distance to BC**: `TV(x_qre, bc_p1)`, `TV(y_qre, bc_p2)`, `KL(bc_p1 || x_qre)` (where defined)
- **Distance to Nash**: `TV(x_qre, nash_p1)`, `TV(y_qre, nash_p2)`
- **QRE exploitability**: sum of both players' exploitability at the QRE strategies
- **Best-fit λ\***: λ that minimizes mean TV(QRE, BC) across both players

### What to compute in aggregate:
- **QRE path curve**: mean TV-to-BC, TV-to-Nash, QRE value, QRE exploitability as function of λ
- **Best-fit λ\* distribution**: histogram across matchups
- **Value preservation**: how close is QRE value to V* at the best-fit λ*?
- **QRE exploitability at λ\***: is the QRE at the "expert temperature" more or less exploitable than BC?

### Motivation (for the doc):
BC is literally a softmax policy — it IS a quantal response. If BC ≈ QRE at moderate λ, that precisely characterizes expert play as a bounded-rational equilibrium. GTO Wizard adopted QRE for poker in April 2025 (Sokota, Kroer, Brown et al. NeurIPS 2022 gave efficient QRE computation). QRE is THE hot solution concept.

**Important**: QRE converges to EXACT Nash as λ→∞. At finite λ, it trades off exploitation for entropy. Our BC has high entropy (support ~95) like a moderate-λ QRE, not like Nash (support ~2.7).

**Adaptation from existing code**: `smoothed_br.py` already does softmax BR (lines 209-218). The difference: QRE iterates to convergence at each λ (no learning rate η, no time-averaging), while smoothed BR uses a learning rate and tracks time-averages.

---

## Experiment 3: Indifference Analysis

**Script**: `scripts/phase2_indifference.py` (CPU-only, reads cache)
**Time**: ~5 seconds
**Output**: `results/phase2/indifference.json`

### What to compute per matchup:
For each (player_strategy, opponent_strategy) ∈ {BC, Nash} × {BC, Nash} (2×2 grid):
- **Action payoffs**: `payoffs = R @ opp_strategy` (P1's expected payoff per action against given opponent)
- **Weighted payoff variance**: `Var = Σ_i p_i * (payoffs_i - V_bar)²` where `V_bar = p @ payoffs`
- **Weighted payoff std**: `sqrt(Var)`
- **Max payoff gap within support**: `max(payoffs[support]) - min(payoffs[support])` (support = actions with prob > 1e-3)
- **Fraction of mass on ε-best-response**: for ε ∈ {0.1, 0.5, 1.0}, compute `Σ_i p_i * 1[payoffs_i >= max(payoffs) - ε]`

This produces a 2×2 table for each matchup:

|  | vs BC opponent | vs Nash opponent |
|--|---|---|
| **BC strategy** | Var_BC_vs_BC | Var_BC_vs_Nash |
| **Nash strategy** | Var_Nash_vs_BC | Var_Nash_vs_Nash |

Nash vs Nash should have Var ≈ 0 (indifference principle). The interesting cells: BC vs BC (is BC adapted to its niche?) and BC vs Nash (where does it break?).

### What to compute in aggregate:
- Mean/median weighted payoff std for each cell of the 2×2 table (with bootstrap CIs)
- **Indifference ratio**: `Var_BC_vs_BC / Var_BC_vs_Nash` — if <1, BC is MORE indifferent against BC than against Nash (ecological adaptation)
- Fraction of BC mass on ε-best-response against BC vs against Nash

### Motivation (for the doc):
The indifference principle (all support actions yield equal payoff) is the defining property of Nash equilibrium play. If BC approximately satisfies indifference against the BC opponent (low variance) but violates it against Nash (high variance), that's "ecological rationality" — expert play is adapted to the population it faces, not to worst-case opponents. This connects to self-confirming equilibrium (Fudenberg & Levine 1993): players can maintain suboptimal strategies when they only observe outcomes against the population they actually face.

---

## Experiment 4: Regret Decomposition

**Script**: `scripts/phase2_regret.py` (CPU-only, reads cache)
**Time**: ~30 seconds (swap regret is O(n²) per matchup, n~100-150)
**Output**: `results/phase2/regret.json`

### What to compute per matchup (P1 side; symmetric for P2):

**External regret** against various opponents:
- `ext_regret_vs_bc = max_i (R @ bc_p2)[i] - bc_p1 @ R @ bc_p2` (regret of BC against BC opponent)
- `ext_regret_vs_nash = max_i (R @ nash_p2)[i] - bc_p1 @ R @ nash_p2` (regret against Nash opponent)
- `ext_regret_vs_uniform = max_i (R @ u)[i] - bc_p1 @ R @ u` where u = uniform

**Swap regret** (checks if BC is an approximate correlated equilibrium):
- `swap_regret = max_{j} Σ_i bc_p1[i] * ((R @ bc_p2)[j] - (R @ bc_p2)[i])`
  - This is: "the best I could do by swapping ALL my mass from any action i to a single action j"
  - Actually, full swap regret is: `max_phi Σ_i bc_p1[i] * ((R @ bc_p2)[phi(i)] - (R @ bc_p2)[i])`
    where phi is an arbitrary mapping from actions to actions
  - For computational feasibility, compute the **max-swap** version (swap one action to one other): O(n²)
  - Also compute the **full internal regret**: for each action i, the best swap target j_i = argmax_j (R @ bc_p2)[j]. Then internal_regret = Σ_i bc_p1[i] * ((R @ bc_p2)[j_i] - (R @ bc_p2)[i])

**Best-response mass** (supplementary):
- Against BC opponent: fraction of BC mass on the BR action (the argmax of R @ bc_p2)
- Against Nash opponent: fraction of BC mass on the BR action (argmax of R @ nash_p2)

### What to compute in aggregate:
- Mean/median for each regret type (with bootstrap CIs)
- **Regret profile**: bar chart showing ext_regret_vs_bc, ext_regret_vs_nash, swap_regret, exploitability
- **Key ratio**: ext_regret_vs_bc / ext_regret_vs_nash — if <<1, BC is population-adapted
- Fraction of matchups where swap_regret < ε (for ε = 0.1, 0.5, 1.0)

### Motivation (for the doc):
External regret is the gap between the best fixed action in hindsight and actual play. If BC has near-zero external regret against BC (the population opponent), it's population-rational — no single action would improve outcomes against other experts. High regret against Nash means BC is vulnerable to a game-theoretically optimal opponent. The gap between these two regret measures is the "price of ecological specialization."

Swap regret is the stronger condition for correlated equilibrium (Hart & Mas-Colell 2000). Low swap regret would mean BC is an approximate correlated equilibrium, which by MacQueen (2023) implies approximate Nash in zero-sum games. This would connect bc_vs_bc ≈ V* to a precise solution concept.

---

## Parallelism Plan

```
Time    GPU                     CPU
────    ───                     ───
0:00    Phase 0: build cache    (idle)
        ~3 min
0:03    (idle)                  Exp 1 (SVD): ~10s     ← can all run
                                Exp 2 (QRE): ~5-10min    simultaneously
                                Exp 3 (Indiff): ~5s      on 8 cores
                                Exp 4 (Regret): ~30s
0:13    (idle)                  Aggregate + write results
0:15    Done
```

**Total wall time: ~15 minutes.** (Phase 0 is the bottleneck.)

Experiments 1-4 are fully independent and read the same cache file. They can be launched as 4 parallel processes or 4 background agents.

---

## Files to Create

| File | Type | Description |
|------|------|-------------|
| `scripts/phase2_cache.py` | Script | GPU: build payoff matrices + strategies, save to pickle |
| `scripts/phase2_svd.py` | Script | CPU: SVD analysis on cached matrices |
| `scripts/phase2_qre.py` | Script | CPU: QRE path computation |
| `scripts/phase2_indifference.py` | Script | CPU: indifference structure analysis |
| `scripts/phase2_regret.py` | Script | CPU: regret decomposition |

**Output directory**: `results/phase2/`

## Existing Code to Reuse

| What | Where | How |
|------|-------|-----|
| GPU phase (build matchups) | `scripts/smoothed_br.py:42-141` | Copy `_build_matchups()` verbatim |
| Nash LP solver | `turnone/game/nash.py:solve_nash_lp()` | Import directly |
| BC strategy from logits | `turnone/game/exploitability.py:bc_strategy_from_logits()` | Import directly |
| Exploitability computation | `turnone/game/exploitability.py:exploitability_from_nash()` | Import for QRE exploitability |
| Bootstrap CIs | `turnone/eval/bootstrap.py:bootstrap_all()` | Import for all experiments |
| Softmax utility | `scripts/smoothed_br.py:148-153` | Copy `_softmax()` |
| Dataset + Vocab loading | `turnone/data/dataset.py` | Import as in existing scripts |
| Model loading pattern | `scripts/evaluate.py:1-36` | Same argparse + model loading |

## Docs to Update After Experiments

- `docs/PROJECT_BIBLE.md` — add Section 6.9 (Phase 2 structural analysis)
- `docs/why_pivot.md` — update Finding 2 framing with structural evidence, update talk tracks
- `CLAUDE.md` — update Week 4 status with Phase 2 results

## Theoretical References for Writeup

| Concept | Paper | Relevance |
|---------|-------|-----------|
| CCE ⟹ Nash in 2p0s | MacQueen 2023 (arXiv:2304.07187) | bc_vs_bc ≈ V* ↔ approximate CCE |
| Population chaos → aggregate Nash | Bielawski et al. PNAS 2025 | Heterogeneous experts, aggregate convergence |
| BC is rate-optimal (non-interactive) | Freihaut et al. 2025 (arXiv:2505.17610) | Exploitability gap is fundamental |
| QRE as practical solution concept | Sokota, Kroer, Brown et al. NeurIPS 2022 | QRE path analysis |
| GTO Wizard adopts QRE | Blog, April 2025 | Timeliness |
| Low-rank games | Lipton, Markakis & Mehta 2003 | Support size ≤ rank + 1 |
| Self-confirming equilibrium | Fudenberg & Levine 1993 | Ecological adaptation |
| Ecological rationality | Gigerenzer & Gaissmaier | Heuristics adapted to niche |
| Swap regret → CE | Hart & Mas-Colell 2000 | Regret decomposition |
| VGC-Bench | Angliss et al. AAMAS 2025 | Closest comparable work |

## Verification (DONE)

All checks passed:
1. Cache has 500 matchups, same seed=42 indices. bc_vs_bc mean = 0.2023, game_value mean = 0.2213 (matches Phase 1 values: 0.20 and 0.22).
2. Nash vs Nash weighted std = 9.4e-16 ≈ 0 (indifference principle verified to machine precision).
3. QRE convergence: 100% at λ≤1, drops at λ≥5 (simultaneous iteration cycling — flagged, high-λ data excluded).
4. All external regret values ≥ 0 (by definition, confirmed).
5. SVD effective rank ≤ min(n1, n2) for all 500 matchups (confirmed).
6. Cache file: 31.6 MB, 500 matchups, matrix sizes P1 122±28, P2 122±26.

Post-audit corrections:
7. SVD payoff-space TV (clip+renormalize projection) was methodologically broken — replaced with payoff-weighted TV (0.019).
8. Indifference/regret ratios confounded by Nash definitional properties — demoted to supplementary.
9. QRE results at λ≥5 excluded due to convergence failure (28% at λ=5, <10% at λ≥10).
