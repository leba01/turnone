# Framing & Positioning: "Collective Equilibrium, Individual Exploitability"

Last updated Feb 24, 2026. Reflects Showdown targeting fix, literature survey, and planned improvements.

---

## 1. The Finding

Expert VGC players are **individually exploitable** but **collectively near-Nash**.

- **Individual exploitability**: mean 1.41 [1.36, 1.46] reward units. A best-responding adversary gains substantially. This exceeds the dynamics noise floor (0.41) by 1.0 units.
- **Collective equilibrium**: BC-vs-BC ≈ Nash value (0.20 vs 0.22, gap = -0.02). When experts face other experts, outcomes match Nash. Holds across reward weight specifications.

The metagame has converged to approximate equilibrium through collective experience — without anyone solving for Nash. But any individual strategy is readable and punishable by an adversary who models it specifically.

---

## 2. Why This Is Novel

### 2.1 Theory exists, empirics don't

So & Ma (2025) prove theoretically that learnable mixed Nash equilibria are collectively rational. Evolutionary game theory establishes that populations can converge to Nash without individual rationality. But nobody has **measured** this in real competitive human play.

We provide the first empirical demonstration: 155K expert battles, learned dynamics model, exact Nash LP, quantified exploitability gap with bootstrap CIs.

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

1. **BC is more exploitable than Nash.** Exploitability (1.41) exceeds dynamics noise floor (0.41) by 1.0 units. This is a real signal.
2. **BC-vs-BC ≈ Nash in aggregate.** Gap = -0.02, CI includes zero, holds across reward weights.
3. **Qualitative pattern**: Nash favors defensive mixed play (Protect, coverage, defensive Tera). BC is predictably aggressive. Disjoint support across all examined matchups.

### What we're NOT confident about

1. **Exact exploitability magnitude.** 1.41 depends on dynamics model accuracy. True value could be ~0.8-2.0. We trust the ordering, not the number.
2. **Per-matchup BC-vs-BC tracking.** R²=0.59, with gaps reaching ±2.4 in extreme games. The collective equilibrium claim is about the population average, not individual matchups. (Autoregressive BC may tighten this.)

### What we're NOT claiming

- Experts are "bad" — they play optimally *for the metagame they face*. Exploitability is a worst-case measure against a strategy-aware adversary.
- This generalizes beyond turn 1 — multi-turn play may show different patterns.
- The independence factorization is accurate — it's an upper bound. Autoregressive BC will provide a tighter estimate.

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

### 5.4 Why autoregressive BC (planned)

The independence factorization P(a)×P(b)×P(tera) is the biggest known methodological weakness. It inflates exploitability and adds noise to BC-vs-BC. The autoregressive model P(a)×P(b|a)×P(tera) captures the real correlation structure in expert play — when mon A uses Fake Out, mon B is more likely to set up.

The code exists and is tested. This isn't speculative — it's a concrete improvement with a clear expected benefit.

---

## 6. Narrative Arc

### For milestone / report

1. **Setup**: VGC turn 1 is a simultaneous-move game with ~200-400 valid actions per side. How close are human experts to game-theoretic optimality?

2. **Approach**: Empirical game-theoretic analysis with a learned dynamics oracle:
   - BC model captures expert strategy (what they do)
   - Dynamics model captures outcomes (what happens)
   - Payoff matrix + Nash LP gives the equilibrium (what's optimal)
   - Compare BC to Nash → exploitability

3. **Finding 1 — Individual exploitability**: Expert strategies are exploitable. Mean 1.41, exceeding the dynamics noise floor by 1.0. The signal is real.

4. **Finding 2 — Collective equilibrium**: BC-vs-BC ≈ Nash value (gap = -0.02). When experts face other experts, outcomes match Nash. Confirmed across reward weights. First empirical evidence for So & Ma's collective rationality theorem.

5. **The paradox explained**: Individual strategies are *predictable* (BC imitates the population mean) but the population distribution IS the equilibrium. Game theory matters against a strategy-aware adversary — not against other humans playing "standard" lines. This mirrors poker: the GTO line is exploitable by a targeted counter, but against other regs it's fine.

6. **Case studies**: Nash recommends qualitatively different play — more Protect, defensive coverage, mixed Tera. BC is predictably aggressive. The exploitability comes from predictability, not from bad moves.

### For poster (6 figures)

1. Pipeline diagram
2. Exploitability histogram (500 matchups)
3. Safety-exploitation triangle
4. Trust: noise sensitivity + reward weight sensitivity
5. Case study (1-2 matchups, BC vs Nash distributions)
6. Dynamics quality (HP scatter + KO ROC)

---

## 7. Talk Tracks

### "What surprised you?"

"That BC-vs-BC almost exactly equals the Nash value — experts found the equilibrium through thousands of games of experience, without any game-theoretic reasoning. So & Ma recently proved this should happen theoretically; we're the first to measure it in real competitive play."

### "Why should I trust these numbers?"

"Three reasons. First, noise injection shows BC exploitability exceeds what dynamics noise alone produces by a full reward unit. Second, the BC-vs-BC ≈ Nash finding uses the same payoff matrix, so dynamics errors cancel. Third, reward weight sweep (w_ko 1-5) confirms robustness."

### "Why not exact simulation?"

"Replay logs don't contain EVs, IVs, or natures — you can't compute damage without knowing a Pokemon's stat spread. Our dynamics model learns expected outcomes under the actual stat distribution in the metagame. That's the right quantity for population-level exploitability."

### "How does this compare to VGC-Bench?"

"Complementary questions. They build agents and test how well they play. We measure how far existing human players are from game-theoretic optimality. They ask 'can we build a good player?' We ask 'what does the gap between human play and Nash look like, and what is its structure?'"

### "What's the CS234 connection?"

"Behavioral cloning as supervised baseline, a learned world model for offline evaluation, and game-theoretic optimization via Nash LP. The key insight: turn 1 is a simultaneous-move game, not a sequential MDP, so Nash equilibrium — not Bellman optimality — is the right solution concept."

---

## 8. Planned Improvements

### 8.1 Autoregressive BC (high confidence, code exists)
Train P(b|a) factorization. Expected to: tighten exploitability bounds (lower = more accurate), improve BC-vs-BC per-matchup tracking (Check 1), and strengthen the collective equilibrium claim.

### 8.2 Dynamics v2 (medium confidence)
Action cross-attention + larger trunk (d_hidden 256→512). The v1 architecture concatenates action embeddings with no interaction modeling — it can't represent move synergies. Training curves show capacity ceiling. Expected to improve reward-space R² from 0.63.

### 8.3 v1 vs v2 comparison (ablation)
Run the full eval pipeline with both dynamics versions. If v2 significantly improves reward-space error AND the exploitability findings hold → the results are robust to dynamics quality. If findings change → we learn which results depend on model fidelity.

### 8.4 Future work (out of scope)
- **Exact simulation**: with open team sheet data (EVs/IVs public)
- **Multi-turn**: extend dynamics to turns 2-3
- **Exploitability by archetype**: team clustering + stratified analysis
- **Deploy**: real-time Nash strategy computation for team sheet + leads
