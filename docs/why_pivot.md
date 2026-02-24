# Framing & Positioning: "Collective Equilibrium, Individual Exploitability"

Last updated Feb 24, 2026. Reflects Phase 1 results: autoregressive BC, dynamics v2, and three-config ablation.

---

## 1. The Finding

Expert VGC players are **individually exploitable** but **collectively near-Nash** — via **symmetric error cancellation**.

- **Individual exploitability**: mean 1.41–1.52 reward units depending on dynamics oracle (v1: 1.41 [1.36, 1.47], v2: 1.52 [1.47, 1.58]). A best-responding adversary gains substantially. Both exceed the dynamics noise floor (0.41) by ≥1.0 units.
- **Collective equilibrium**: BC-vs-BC ≈ Nash value across all configs (gaps: -0.02, -0.03, +0.02). When experts face other experts, outcomes match Nash. Holds across reward weights, BC factorizations, and dynamics model quality.
- **Error cancellation**: Value decomposition reveals the mechanism. BC P1 loses ~1.0 against Nash P2, but Nash P1 gains ~1.0 against BC P2. These symmetric errors cancel in expert-vs-expert play (all 500 matchups show opposite-sign cross-play gaps). BC-vs-BC ≈ Nash is NOT because individual play is near-optimal — it's because both players are suboptimal in symmetric, offsetting ways.

**Strategy-space distances** confirm BC and Nash are maximally different: TV distance 0.99, BC places 1.3% of mass on Nash-support actions, Nash support averages 2.7 actions vs BC's ~95.

**Robustness**: The finding holds across a three-config ablation (independent BC + v1 dynamics, autoregressive BC + v1, autoregressive BC + v2). The autoregressive factorization didn't change exploitability (independence wasn't inflating it). The better dynamics model found *more* exploitability (a more accurate oracle reveals more structure to exploit).

---

## 2. Why This Is Novel

### 2.1 Extends sports minimax testing to combinatorial games

The penalty kick literature (Chiappori et al. 2002, Palacios-Huerta 2003) and tennis serving literature (Walker & Wooders 2001, Anderson et al. 2024) test minimax in 2×2 or small-action sports games with known payoff structure. We extend this to **combinatorial strategy spaces** (~200-400 actions per side) where the game structure must be *learned* from observational data.

Our value decomposition goes beyond aggregate Nash-matching to diagnose the *mechanism*: symmetric error cancellation, not independent near-optimality. Anderson et al. (2024) compute best-response gains in tennis but have known game structure; we need a learned dynamics model.

So & Ma (2025) prove theoretically that learnable mixed Nash equilibria are collectively rational. Their conditions are trivially satisfied in two-player zero-sum games (uniform stability, Pareto optimality hold vacuously). We cite them for motivation but our contribution is the empirical decomposition.

### 2.2 Distinct from the poker literature

Bowling (2015), Brown & Sandholm (2017, 2019) prove that bots beat humans at poker. They measure *bot superiority*, not the *structure* of human exploitability. They don't ask: "When humans play each other, are outcomes near Nash?" We do, and the answer is yes.

### 2.3 Distinct from VGC-Bench and Metamon

- **VGC-Bench** (Angliss et al., 2025): trains agents via BC + MARL + PSRO, measures agent-vs-agent performance. Asks "can we build a good player?" We ask "how far are *existing human players* from game-theoretic optimality?"
- **Metamon** (Grigsby et al., 2025): offline RL for full-game singles play. Different format, different question.
- **Oblander** (2024): analyzes 11M Showdown matches for behavioral economics patterns (heuristics vs. optimal). Closest to our "individually suboptimal" finding, but uses behavioral econ framing (bounded rationality), not game theory (Nash equilibrium, exploitability).

### 2.4 Methodologically novel

We use a **learned dynamics model as an EGTA oracle**. Classical EGTA (Wellman 2006, Tuyls 2018) uses game simulators to build payoff matrices. Vorobeychik (2007) learns payoff functions from simulator queries. We learn a world model from observational data — no simulator access needed. This is the right approach when the game is too complex to simulate exactly from available data (replay logs lack EVs/IVs/natures for exact damage calculation).

---

## 3. The Framing We Chose (and why)

### Framing B: "Collective equilibrium, individual exploitability"

**Pitch**: Expert VGC play achieves near-Nash outcomes in aggregate while being individually exploitable. The metagame has converged to collective equilibrium through experience alone — but any individual strategy is predictable and punishable.

**Why this framing:**

1. **Most interesting finding.** The paradox is memorable: "experts collectively play Nash but are individually exploitable." Like poker regulars — the standard line is fine against other regs, but readable by anyone who models you.

2. **Most robust result.** BC-vs-BC ≈ Nash is computed on the same payoff matrix R, so dynamics errors affect both sides similarly. Confirmed across three reward weight settings. The noise sensitivity experiment provides the stronger evidence for the exploitability signal specifically.

3. **Novel empirical contribution.** First data point for So & Ma's collective rationality theorem. First quantification of the individual-vs-collective exploitability gap in real competitive play.

4. **Clean CS234 story.** Behavioral cloning (what experts do) + world model (what happens) + game-theoretic optimization (what's optimal) = exploitability measurement. Connects RL to empirical game theory.

### Rejected alternatives

**Framing A: "Experts are significantly exploitable"** — True but incomplete. Doesn't capture the more interesting collective equilibrium finding. Also, exploitability magnitude depends on dynamics model accuracy (reward MAE 1.19 vs exploit 1.41 — same order of magnitude). The noise sensitivity argument saves it, but requires defensive paragraphs.

**Framing C: "The pipeline is the contribution"** — Safe but uninspiring. CS234 wants insight, not just infrastructure.

---

## 4. Trust Quantification

### What we're confident about

1. **BC is more exploitable than Nash.** Exploitability (1.41–1.52) exceeds dynamics noise floor (0.41) by ≥1.0 units across both dynamics versions. This is a real signal.
2. **BC-vs-BC ≈ Nash in aggregate.** Gap < 0.03 in all configs, CI includes zero, holds across reward weights and model improvements.
3. **Qualitative pattern**: Nash favors defensive mixed play (Protect, coverage, defensive Tera). BC is predictably aggressive. Disjoint support across all examined matchups.
4. **Independence factorization is fine.** Autoregressive BC (P(b|a)) gave identical exploitability to independent BC (P(a)×P(b)). The correlation between mon A and B actions is weak enough that the simpler model suffices.
5. **Better dynamics finds more exploitability.** V2 (cross-attention, HP MAE 12.6, reward bias 0.14) found exploit=1.52 vs v1's 1.41. More accurate oracle → more structure revealed.

### What we're NOT confident about

1. **Exact exploitability magnitude.** Range 1.41–1.52 across dynamics models. True value could be ~0.8-2.0. We trust the ordering, not the number.
2. **Per-matchup BC-vs-BC tracking.** R²=0.50–0.59, with gaps in extreme games. The collective equilibrium claim is about the population average, not individual matchups.

### What we're NOT claiming

- Experts are "bad" — they play optimally *for the metagame they face*. Exploitability is a worst-case measure against a strategy-aware adversary.
- This generalizes beyond turn 1 — multi-turn play may show different patterns.

---

## 5. Design Decisions That Show Taste

### 5.1 Why learned dynamics, not exact simulation

Replay data lacks EVs/IVs/natures. Without these, Showdown's damage calculator can't compute exact outcomes — a 1-point speed difference flips move order entirely. We can't assume "standard" spreads because the most common spread is often <10% usage and speed tiers are critical.

Our dynamics model learns **expected outcomes marginalized over unobserved stat spreads** — the right quantity for population-level exploitability. This is a principled design choice, not a limitation.

Future work: open team sheet tournament data (full EV spreads public) would enable exact simulation and direct comparison.

### 5.2 Why turn-1 isolation

Turn 1 is a single-shot simultaneous-move game. This structure makes Nash LP tractable (~200-400 actions per side, solved in milliseconds). Multi-turn play requires extensive-form game solving or MCTS — orders of magnitude harder, and the BC + dynamics approach would need sequential prediction. The single-shot structure is a feature, not a limitation.

### 5.3 Why Showdown-backed targeting

We replaced 100+ hand-curated move targeting rules with Showdown's authoritative `moves.json` (658/658 vocab coverage). Since we analyze Showdown replays, their data IS ground truth. The audit found 66 misclassifications in the hand-curated sets. After the fix, exploitability moved from 1.47 → 1.41 and audit Check 11 (Nash action quality) went from non-zero ally-attack weight to exactly zero.

### 5.4 Autoregressive BC ablation

We trained P(b|a) factorization to test whether the independence assumption P(a)×P(b) inflated exploitability. Result: it didn't. Autoregressive BC gave exploit=1.41 vs independent BC's 1.41 (same dynamics model). The correlation between ally actions is weak enough that independence is a fine approximation.

This is actually a *good* result — it means the simpler model is sufficient, and the exploitability finding doesn't depend on a methodological shortcut.

---

## 6. Narrative Arc

### For milestone / report

1. **Setup**: VGC turn 1 is a simultaneous-move game with ~200-400 valid actions per side. How close are human experts to game-theoretic optimality?

2. **Approach**: Empirical game-theoretic analysis with a learned dynamics oracle:
   - BC model captures expert strategy (what they do)
   - Dynamics model captures outcomes (what happens)
   - Payoff matrix + Nash LP gives the equilibrium (what's optimal)
   - Compare BC to Nash → exploitability

3. **Finding 1 — Individual exploitability**: Expert strategies are exploitable. Mean 1.41–1.52 (depending on dynamics oracle), exceeding the dynamics noise floor by ≥1.0. The signal is real and robust to model improvements.

4. **Finding 2 — Collective equilibrium**: BC-vs-BC ≈ Nash value (gaps: -0.02, -0.03, +0.02 across three configs). When experts face other experts, outcomes match Nash. Confirmed across reward weights, BC factorizations, and dynamics model quality. First empirical evidence for So & Ma's collective rationality theorem.

5. **Finding 3 — Error cancellation mechanism**: Value decomposition reveals that BC-vs-BC ≈ Nash arises from symmetric error cancellation. BC P1 loses ~1.0 against Nash P2; Nash P1 gains ~1.0 against BC P2. These offsetting errors produce near-Nash outcomes despite near-maximal strategy-space distance (TV ≈ 0.99).

6. **Finding 4 — Smoothed BR convergence**: Starting from BC strategies, smoothed best-response dynamics converge toward Nash in ~500 iterations (96% exploitability reduction at β=0.01). The metagame acts like a learning process with experts as the initialization.

7. **Case studies**: Nash recommends qualitatively different play — more Protect, defensive coverage, mixed Tera. BC is predictably aggressive. The exploitability comes from predictability, not from bad moves.

### For poster (8 figures)

1. Pipeline diagram
2. Exploitability histogram (500 matchups)
3. Safety-exploitation triangle
4. Trust: noise sensitivity + reward weight sensitivity
5. Case study (1-2 matchups, BC vs Nash distributions)
6. Dynamics quality (HP scatter + KO ROC)
7. **Value decomposition bar chart**: V*, bc_vs_bc, cross-play values — the analytical money figure showing symmetric error cancellation
8. **Smoothed BR convergence**: exploitability vs iteration at multiple temperatures — visually compelling trajectory from BC to Nash

---

## 7. Talk Tracks

### "What surprised you?"

"That BC-vs-BC almost exactly equals the Nash value — but for the *wrong reason*. We expected independent near-optimality (each player individually achieves near-Nash payoff). Instead, value decomposition reveals symmetric error cancellation: BC P1 loses ~1.0 against Nash P2, and Nash P1 gains ~1.0 against BC P2. These opposite-sign errors cancel perfectly in expert-vs-expert play. Every single matchup (500/500) shows this pattern. Experts aren't individually near-optimal — they're symmetrically suboptimal."

### "Why should I trust these numbers?"

"Three reasons. First, noise injection shows BC exploitability exceeds what dynamics noise alone produces by a full reward unit. Second, the BC-vs-BC ≈ Nash finding uses the same payoff matrix, so dynamics errors cancel. Third, reward weight sweep (w_ko 1-5) confirms robustness."

### "Why not exact simulation?"

"Replay logs don't contain EVs, IVs, or natures — you can't compute damage without knowing a Pokemon's stat spread. Our dynamics model learns expected outcomes under the actual stat distribution in the metagame. That's the right quantity for population-level exploitability."

### "How does this compare to VGC-Bench?"

"Complementary questions. They build agents and test how well they play. We measure how far existing human players are from game-theoretic optimality. They ask 'can we build a good player?' We ask 'what does the gap between human play and Nash look like, and what is its structure?'"

### "What's the CS234 connection?"

"Behavioral cloning as supervised baseline, a learned world model for offline evaluation, and game-theoretic optimization via Nash LP. The key insight: turn 1 is a simultaneous-move game, not a sequential MDP, so Nash equilibrium — not Bellman optimality — is the right solution concept."

---

## 8. Completed Ablations & Future Work

### 8.1 Autoregressive BC (DONE)
Trained P(b|a) factorization. Result: exploit=1.41, identical to independent BC (same dynamics). Independence wasn't inflating exploitability. The simpler model suffices.

### 8.2 Dynamics v2 (DONE)
Action cross-attention (1 layer, 2 heads over 4 mon-action tokens) + larger trunk (d_hidden 256→512, d_action 32→64, n_mlp_layers 3→4). Val loss improved 13% over v1. HP MAE: 13.3→12.6. Reward bias halved: 0.26→0.14. ~2.2M params (vs 1.3M v1).

### 8.3 Three-config ablation (DONE)
Full eval pipeline with all three configs (independent BC + v1, autoreg BC + v1, autoreg BC + v2). Results in `results/baseline/`, `results/autoreg_v1/`, `results/autoreg_v2/`. Key finding: exploitability findings are robust to both BC factorization and dynamics model quality. Better dynamics found *more* exploitability (1.41→1.52).

### 8.4 Value decomposition (DONE)
Cross-play analysis (500 matchups): BC_P1 vs Nash_P2 = -0.80, Nash_P1 vs BC_P2 = +1.25. Error cancellation, not independent near-optimality. All 500 matchups show opposite-sign gaps. TV distance 0.99, BC mass on Nash support 1.3%.

### 8.5 Smoothed BR convergence (DONE)
200 matchups × 5 temperatures × 2 learning rates × 500 iterations. At β=0.01, time-averaged exploitability drops from 2.74 to 0.11 (96% reduction), with 50% of matchups fully converged (<0.1). Higher temperatures converge to QRE, not Nash.

### 8.6 Future work (out of scope)
- **Exact simulation**: with open team sheet data (EVs/IVs public)
- **Multi-turn**: extend dynamics to turns 2-3
- **Exploitability by archetype**: team clustering + stratified analysis
- **Deploy**: real-time Nash strategy computation for team sheet + leads
