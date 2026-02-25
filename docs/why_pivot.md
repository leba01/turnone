# Framing & Positioning: "Low-Rank Games, High-Entropy Experts"

Last updated Feb 24, 2026. Reflects Phase 2 structural analysis with post-audit corrections. Central finding: low effective rank (~3) explains collective equilibrium. Payoff-weighted TV = 0.019 (vs 0.99 unweighted). QRE doesn't fit. Indifference/regret experiments confounded.

---

## 1. The Finding

Expert VGC players are **individually exploitable** but **collectively near-Nash**.

- **Individual exploitability**: mean 1.41–1.52 reward units depending on dynamics oracle (v1: 1.41 [1.36, 1.47], v2: 1.52 [1.47, 1.58]). A best-responding adversary gains substantially. Both exceed the dynamics noise floor (0.41) by ≥1.0 units.
- **Collective equilibrium**: BC-vs-BC ≈ Nash value across all configs (gaps: -0.02, -0.03, +0.02). When experts face other experts, outcomes match Nash. Holds across reward weights, BC factorizations, and dynamics model quality.
- **Structural explanation (Phase 2)**: SVD analysis reveals payoff matrices have effective rank ~3 out of ~122 nominal actions. Top-1 singular value captures 76% of the Frobenius norm. Payoff-weighted TV is just **0.019** (vs 0.99 unweighted) — a 50× collapse confirming that strategy differences are concentrated in payoff-irrelevant dimensions. bc\_vs\_bc ≈ V\* is structurally inevitable: most strategy variation lies in the null space of R. Lipton et al. (2003) predict Nash support ≤ rank+1 ≈ 4; we observe 2.7.
- **QRE negative result**: QRE doesn't fit BC (best TV = 0.68). Expert play is not a bounded-rational equilibrium.
- **Caveat**: dynamics model bottleneck (d_action=32) may amplify low-rank structure, though VGC damage is genuinely additive.

**Strategy-space distances** confirm BC and Nash are maximally different in the full simplex: TV distance 0.99, BC places 1.3% of mass on Nash-support actions, Nash support averages 2.7 actions vs BC's ~95. But this distance is inflated by payoff-irrelevant dimensions.

**Robustness**: The finding holds across a three-config ablation (independent BC + v1 dynamics, autoregressive BC + v1, autoregressive BC + v2). The autoregressive factorization didn't change exploitability (independence wasn't inflating it). The better dynamics model found *more* exploitability (a more accurate oracle reveals more structure to exploit).

---

## 2. Why This Is Novel

### 2.1 Extends sports minimax testing to combinatorial games

The penalty kick literature (Chiappori et al. 2002, Palacios-Huerta 2003) and tennis serving literature (Walker & Wooders 2001, Anderson et al. 2024) test minimax in 2×2 or small-action sports games with known payoff structure. We extend this to **combinatorial strategy spaces** (~200-400 actions per side) where the game structure must be *learned* from observational data.

Our Phase 2 structural analysis goes beyond aggregate Nash-matching to explain *why*: SVD reveals payoff matrices have effective rank ~3, and payoff-weighted TV between BC and Nash is just 0.019 (vs 0.99 unweighted) — a 50× collapse when weighting by payoff relevance. Anderson et al. (2024) compute best-response gains in tennis but have known game structure; we need a learned dynamics model and the low-rank analysis to explain the payoff equivalence.

So & Ma (2025) prove theoretically that learnable mixed Nash equilibria are collectively rational. Their conditions are trivially satisfied in two-player zero-sum games (uniform stability, Pareto optimality hold vacuously). We cite them for motivation but our contribution is the empirical measurement: experts are collectively near-Nash despite individually far-from-Nash strategies, in a combinatorial strategy space with a learned game model.

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

### Framing B: "Low-rank games, high-entropy experts"

**Pitch**: Expert VGC play achieves near-Nash outcomes despite being individually exploitable and strategically distant from Nash. The explanation is structural: payoff matrices have effective rank ~3 out of ~122 actions. Payoff-weighted TV between BC and Nash is just 0.019 (vs 0.99 unweighted). Experts don't need to play Nash — in low-rank games, most strategies produce similar payoffs.

**Why this framing:**

1. **Most interesting finding.** The structural explanation is surprising: not behavioral adaptation, not bounded rationality, but matrix rank. The ~200 nominal actions collapse to ~3 strategic dimensions.

2. **Most robust result.** SVD is model-free (computed directly on payoff matrices). The low-rank structure is independent of BC model quality. Confirmed by correlation with Nash support size (Lipton et al. 2003).

3. **Novel empirical contribution.** First empirical application of low-rank game theory (Lipton et al. 2003) to explain human play patterns. First quantification showing that payoff-weighted TV (0.019) vs unweighted TV (0.99) demonstrates strategy distance is misleading when the game has low effective rank.

4. **Clean CS234 story.** Behavioral cloning (what experts do) + world model (what happens) + game-theoretic optimization (what's optimal) + structural analysis (why they match) = complete pipeline from data to insight.

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

### Why 44.8% BC accuracy is not a weakness (resolved)

The per-mon action space is 16 slots, but after masking the effective space is ~7 valid actions (ranges 4–12 depending on moveset). Context:

| Baseline | Accuracy |
|----------|----------|
| Uniform random (valid actions) | ~14% |
| Majority class | ~5-8% |
| **BC model** | **44.8%** |
| BC top-3 | **79.6%** |

BC is 3× better than random. NLL of 1.492 → perplexity 4.45, meaning the model concentrates uncertainty to ~4-5 plausible actions out of ~7 valid. This is near the ceiling of what top-1 accuracy can measure: expert play has genuine entropy (top-10 action pairs cover only 16.3% of data). Players are legitimately mixing over multiple reasonable options.

Critically, **BC accuracy doesn't drive exploitability**. The autoregressive ablation (fancier BC, P(b|a) instead of P(a)×P(b)) changed accuracy but not exploitability (1.41 → 1.41). The *distribution shape* matters for game theory, not the mode. Improving top-1 from 45% to 55% would not meaningfully change any downstream result.

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

### 5.3 Why Showdown-backed targeting & why two masks (resolved)

We replaced 100+ hand-curated move targeting rules with Showdown's authoritative `moves.json` (658/658 vocab coverage). Since we analyze Showdown replays, their data IS ground truth. The audit found 66 misclassifications in the hand-curated sets. After the fix, exploitability moved from 1.47 → 1.41 and audit Check 11 (Nash action quality) went from non-zero ally-attack weight to exactly zero.

**"Moves are fully known — why learn targeting?"** They are, and we don't learn move mechanics. The categorization is deterministic and hardcoded from Showdown (self: 123 moves, ally: 4, spread: 59, single: 472). What BC learns is *strategic targeting*: which opponent to Thunderbolt, when to Protect vs attack, whether to Pollen Puff your ally or the enemy. That's matchup-dependent and the whole point of behavioral cloning.

**Why two masks?** The BC mask (`compute_action_mask`) allows target=3 (self) for single-target moves because the Showdown protocol records failed/redirected moves as self-targeting — a data artifact we can't strip from training without losing real examples. The strategic mask (`compute_strategic_mask`) removes target=3 for single-target moves because no rational player intentionally Thunderbolts themselves, and restricts ally-targeting to 11 dual-purpose moves (Pollen Puff, Skill Swap, etc.). This is the mask used for payoff enumeration / Nash solving.

**Impact is negligible**: median BC mass stripped by the strategic mask is ~0.8% per mon, 97.1% retained at the joint level. Audit check 11 confirms Nash solutions put exactly 0% weight on the removed actions. The strategic mask is just cleaning protocol noise before game-theoretic computation.

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

4. **Finding 2 — Collective equilibrium**: BC-vs-BC ≈ Nash value (gaps: -0.02, -0.03, +0.02 across three configs). When experts face other experts, outcomes match Nash. Confirmed across reward weights, BC factorizations, and dynamics model quality.

5. **Finding 3 — Low-rank structure (the explanation)**: SVD reveals payoff matrices have effective rank ~3 out of ~122 actions. Top-1 captures 76% of Frobenius norm, top-3 captures 96%. Payoff-weighted TV collapses from 0.99 to 0.019 — a 50× reduction. bc\_vs\_bc ≈ V\* is structurally inevitable: most strategy variation is payoff-irrelevant. Lipton et al. (2003) predict Nash support ≤ rank+1; our support 2.7 matches. QRE doesn't fit (negative result).

6. **Finding 4 — Smoothed BR convergence**: Starting from BC strategies, smoothed best-response dynamics converge toward Nash in ~500 iterations (96% exploitability reduction at β=0.01). The metagame acts like a learning process.

7. **Case studies**: Nash recommends qualitatively different play — more Protect, defensive coverage, mixed Tera. BC is predictably aggressive. The exploitability comes from predictability, not from bad moves. But in a rank-3 game, this predictability costs little against other BC players.

### For poster (8-10 figures)

1. Pipeline diagram
2. Exploitability histogram (500 matchups)
3. Safety-exploitation triangle
4. Trust: noise sensitivity + reward weight sensitivity
5. Case study (1-2 matchups, BC vs Nash distributions)
6. Dynamics quality (HP scatter + KO ROC)
7. **SVD cumulative energy + payoff-weighted TV** — Phase 2 headline: rank ~3 captures 96%; payoff-weighted TV = 0.019 vs raw 0.99
8. **Value decomposition bar chart**: V*, bc_vs_bc, cross-play — collective equilibrium
9. **QRE path curve**: TV-to-BC and TV-to-Nash vs λ — shows QRE never fits BC
10. **Smoothed BR convergence**: exploitability vs iteration at multiple temperatures

---

## 7. Talk Tracks

### "What surprised you?"

"That the explanation is structural, not behavioral. We expected BC to be a bounded-rational equilibrium — a QRE at some temperature. Instead, SVD shows payoff matrices have effective rank ~3 out of ~122 actions. Payoff-weighted TV between BC and Nash is just 0.02 — fifty times lower than the unweighted 0.99. bc\_vs\_bc ≈ V\* isn't because experts are clever — it's because the games are so low-rank that most strategies produce similar payoffs. The ~95 extra actions in BC's support are payoff-irrelevant padding."

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
Cross-play analysis (500 matchups): BC_P1 vs Nash_P2 = -0.80, Nash_P1 vs BC_P2 = +1.25. Bilinear decomposition: bc\_vs\_bc − V\* = gap₁ + gap₂ + interaction, all three terms near-zero in aggregate. Opposite-sign gaps are guaranteed by minimax theorem (trivial); the non-trivial finding is collective near-Nash despite maximal individual distance (TV 0.99, BC mass on Nash support 1.3%).

### 8.5 Smoothed BR convergence (DONE)
200 matchups × 5 temperatures × 2 learning rates × 500 iterations. At β=0.01, time-averaged exploitability drops from 2.74 to 0.11 (96% reduction), with 50% of matchups fully converged (<0.1). Higher temperatures converge to QRE, not Nash.

### 8.6 Phase 2: Structural characterization (DONE, post-audit corrections applied)
Four experiments on cached payoff matrices (500 matchups):

1. **SVD / Effective Rank**: Rank ~3 at 95% energy. Top-1 captures 76%, top-3 captures 96%. Payoff-weighted TV = 0.019 (vs 0.99 unweighted). Corr(rank, Nash support) = 0.40. **Central finding**. Caveat: dynamics d_action=32 bottleneck may amplify.
2. **QRE Path**: Negative result. Best TV(QRE, BC) = 0.68 at λ=0.01 (uniform). BC is not a bounded-rational equilibrium. Convergence fails at λ≥5 (simultaneous iteration cycles); results at high λ excluded.
3. **Indifference**: Confounded. Indifference ratio 1.75 is a Nash property (Nash makes opponents indifferent by definition), not a BC property. Random strategies produce similar ratio. Demoted to supplementary.
4. **Regret**: Confounded. Regret ratio 1.32 arises because Nash compresses payoff vectors, lowering regret for ANY facing strategy. Random strategies produce similar ratio. Demoted to supplementary.

### 8.7 Future work (out of scope)
- **Exact simulation**: with open team sheet data (EVs/IVs public)
- **Multi-turn**: extend dynamics to turns 2-3
- **Exploitability by archetype**: team clustering + stratified analysis
- **Deploy**: real-time Nash strategy computation for team sheet + leads
