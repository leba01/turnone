# TurnOne Paper Companion

A plain-English walkthrough of the paper, section by section. Covers every technical concept, why each decision was made, and what the numbers mean. Use this to prep for meetings.

---

## The One-Sentence Pitch

Experts in Pokemon VGC play nothing like game-theoretically optimal (Nash), but it doesn't matter because the game is secretly low-dimensional — most of the ~200 actions are payoff-irrelevant duplicates of each other.

---

## Key Concepts (Glossary)

### Total Variation (TV) Distance
How different two probability distributions are. Ranges from 0 (identical) to 1 (no overlap at all). If you and I each have a bag of 200 colored marbles and TV = 0.99, it means 99% of the marbles you'd need to swap to make the bags match. When we say TV(BC, Nash) = 0.99, it means experts pick *completely* different actions than Nash says to.

### Nash Equilibrium
The strategy pair where neither player can improve by changing their own strategy. In a zero-sum game, it's the minimax solution — the best you can guarantee regardless of what your opponent does. Computed via linear programming on the payoff matrix.

### Exploitability
How much you lose compared to Nash when an omniscient opponent best-responds to you. Formula: V* - (your worst-case payoff). If exploitability = 0, you're playing Nash. Our experts have exploitability 1.41, meaning a perfect counter-strategist gains ~1.41 reward units (roughly half a KO advantage).

### Cross-Play Gap
What happens when two conventional (BC) players face each other, compared to two Nash players. Gap = BC-vs-BC payoff - Nash value. Ours is -0.02, i.e., basically zero. This is the key puzzle: experts are individually exploitable but collectively fine.

### Behavioral Cloning (BC)
Supervised learning on expert data. "What would an expert do in this situation?" It's imitation learning — predict the action distribution from the game state. Our BC is a small Transformer (1.32M params) that achieves 44.8% top-1 accuracy (3x uniform baseline).

### Payoff Matrix
For a given matchup (team A vs team B), we enumerate every valid action for each side (~200 each), simulate every pair through the dynamics model, and get a ~200x200 matrix R where R[i,j] = reward if player 1 picks action i and player 2 picks action j.

### SVD (Singular Value Decomposition)
Factors a matrix into orthogonal components ranked by importance. R = U * Sigma * V^T. The singular values (diagonal of Sigma) tell you how much each component contributes. If the first 3 singular values capture 96% of the total, the matrix is "effectively rank 3" — it looks 200-dimensional but really only has 3 independent dimensions of variation.

### Effective Rank
How many singular values you need to capture 95% of the matrix's energy (sum of squared singular values). Our payoff matrices have effective rank ~3 out of ~122 nominal actions. This is the structural explanation for everything.

### Payoff Null Space
The dimensions of the action space that don't affect payoffs. If 93% of your deviation from Nash is in the null space, then 93% of how "wrong" you are doesn't matter — those dimensions are payoff-irrelevant. This is why convention is free.

### Spectral Bound
A formal theorem: if the game is well-approximated by rank k, then any strategy pair's payoff is close to Nash value. Specifically: |p'Rq - V*| <= |p'R_k q - V*_k| + 2*sigma_{k+1}. With sigma_4/sigma_1 ~ 0.04, the bound is tight.

### CQL (Conservative Q-Learning)
An offline RL algorithm (Kumar et al., 2020). Unlike BC which asks "what did experts do?", CQL asks "what action maximizes reward?" but with a penalty for going too far from the data distribution. It's the RL version of the same problem. Key result: CQL changes 60% of the action distribution but only gains 17% in exploitability — because most of the change is in payoff-irrelevant dimensions.

### Terastallization (Tera)
A one-time-use mechanic in VGC: you can change a Pokemon's type once per game, permanently. It's irreversible — once used, it's gone. Experts conserve it (use on turn 1 only 25% of the time); Nash says use it immediately (99%). This one binary decision accounts for 30% of all exploitability.

### QRE (Quantal Response Equilibrium)
A model of "noisy Nash" — players play Nash-like but with random trembles. We tested this and it fits terribly (lambda ~ 0.01 means essentially uniform random). Experts are NOT playing noisy Nash; they're playing something structured but totally different from Nash.

### EGTA (Empirical Game-Theoretic Analysis)
Building game-theoretic models from data rather than from known rules. We use a learned dynamics model as our "simulator" to construct payoff matrices, then do standard game theory (Nash, exploitability, etc.) on those matrices.

---

## Section-by-Section Walkthrough

### Abstract
States the puzzle and all four findings. The hook: experts look nothing like Nash (TV 0.99) yet get the same payoffs (gap 0.02). Explanation: game is low-rank (effective rank 3), so 93% of deviation is payoff-irrelevant. CQL confirms structure > objective. Exception: tera.

### Section 1: Introduction
Opens with the puzzle as a hook. Four numbered findings preview the whole paper. Related work positions us vs:
- **Minimax in sports**: penalty kicks/tennis have 2-3 actions, we have ~200
- **Price of anarchy**: measures cost of *equilibrium* vs social optimum; we measure cost of *convention* vs equilibrium (different thing)
- **Schelling focal points**: experts coordinate on defaults without strategic reasoning — that's exactly what we find, but quantified
- **Poker AI**: builds agents; we characterize human behavior structure
- **EGTA**: we follow this framework but with a learned dynamics model
- **Pokemon AI**: VGC-Bench and Metamon build agents; we measure existing human play

### Section 2: Approach

**Data**: 154K battles from Pokemon Showdown, 1500+ Elo, Gen 9 VGC Reg G. Each example = full team info + turn-1 actions. Actions encoded as 16 slots per Pokemon (4 moves x 4 targets) + 3-way tera flag. Excluded voluntary switches (~27%).

**Why exclude switches?** Switches change the action space entirely (different Pokemon = different moves). Keeping them would mix two different games.

**BC architecture**: Small Transformer (4 layers, 4 heads, d=128). Key design decision: **factored output** — P(a1, a2, tera) = P(a1) * P(a2) * P(tera). The two leads' actions are predicted independently. This is validated by mutual information = 0.018 bits (basically zero).

**Why factored?** Joint prediction over all action combinations would be ~200*200*3 = 120K outputs — intractable. The independence assumption is empirically justified AND makes the math clean.

**Dynamics model**: Same encoder, but predicts what happens (HP changes, KOs, field state) given a state and action pair. HP MAE = 13.3/100, KO AUC = 0.91.

**Why we need it**: Can't look up "what happens if player 1 uses Thunderbolt on slot 3 while player 2 uses Protect" from data alone — too sparse. The dynamics model lets us evaluate *any* action pair, which is required to build the full payoff matrix.

**Signal-to-noise check**: Our dynamics model has noise (MAE = 1.19 in reward space). If we inject that much noise into a perfect payoff matrix, the "fake" exploitability is 0.41. Our real measured exploitability is 1.41. So SNR = 1.41/0.41 = 3.4:1. The signal is real, not an artifact of model noise.

**Payoff matrix construction**: For each matchup, enumerate valid actions per side, run every pair through dynamics model, get R (m x n, where m,n ~ 200). Payoff = HP advantage + 3.0 * KO advantage + 0.5 * field advantage.

**Why those weights?** Domain-informed heuristic. KOs weighted high because eliminating a Pokemon is strategically dominant. But it doesn't matter much — we test w_ko in {1, 3, 5} and findings are invariant (Section 7).

**Nash via LP**: Standard linear programming for zero-sum games. Gives mixed strategies p*, q* and game value V*.

**Exploitability definition**: V* - min_j (p'R)_j. This is the gap between what Nash guarantees and your worst-case payoff when the opponent best-responds to you.

### Section 3: Experts Are Exploitable

**Key numbers**:
- Exploitability = 1.41 [1.36, 1.46] — about half a KO advantage
- TV(BC, Nash) = 0.99 — maximally different
- Nash uses 2.7 actions on average; BC spreads across ~95
- Only 1.3% of BC mass on Nash-support actions
- BUT: BC-vs-BC = +0.20, Nash value = +0.22, gap = -0.02

**What this means in English**: If an omniscient optimizer played against an expert, they'd gain about half a KO per turn. But experts never face omniscient optimizers — they face other experts. And expert-vs-expert outcomes are identical to Nash-vs-Nash. The puzzle: how can being "maximally wrong" produce the right answer?

### Section 4: Why Convention Is Free

**4.1 It's the Game, Not the Experts**: The killer experiment. If experts are secretly skilled at being near-Nash in payoff space, then non-experts should do worse. But: uniform random, shuffled BC, and Dirichlet-random strategies ALL get gap ~ -0.02. Even random play produces Nash payoffs. This proves it's a structural property of the *game*, not a property of expert *skill*.

**4.2 The Game Is 3-Dimensional**: SVD reveals effective rank ~3. The top 3 singular values capture 96% of the payoff matrix's energy. Out of ~122 actions, only 3 independent dimensions of strategic variation matter.

**Projection test**: Take the vector (BC strategy - Nash strategy) and project it onto R's singular subspaces. Only 7.1% of the deviation energy lands in the top-3 (payoff-relevant) dimensions. 93% is in the null space — dimensions where changing your strategy doesn't change your payoff.

**Analogy**: Imagine a 122-dimensional room, but gravity only works in 3 of those dimensions. You can flail wildly in the other 119 dimensions and nothing happens. That's what experts are doing — "wrong" in 119 dimensions, but those dimensions don't matter.

**4.3 Interaction Decomposition**: The payoff gap breaks into three terms:
- Delta_1 = +1.03 (cost of opponent deviating from Nash, while you play Nash)
- Delta_2 = -1.02 (cost of you deviating from Nash, while opponent plays Nash)
- Interaction = -0.03 (cross-term from both deviating simultaneously)

Delta_1 and Delta_2 are guaranteed to have opposite signs by the minimax theorem (one player gains, one loses when facing a deviant). Their near-equal magnitudes are empirical — they approximately cancel. The interaction term is tiny because the deviations are mostly orthogonal to R's row/column spaces (null space again).

**Rank-reduction validation**: If the game is truly rank-3, then keeping only the top 3 singular values of R should preserve game properties. It does: rank-3 preserves 97.2% of exploitability.

**Spectral bound**: Formalizes the intuition. If sigma_4 is small relative to sigma_1, then any strategy pair gets near-Nash payoffs. This is a known result from matrix perturbation theory (Eckart-Young theorem), applied to game theory.

**Robustness check**: "Maybe your dynamics model is too weak to produce a high-rank matrix." Tested with 3 different capacities (d=32, 64, 128). Effective rank goes from 2.9 to 4.3, but the highest-capacity model overfits. Low rank is real.

### Section 5: Where Convention Fails — Terastallization

The one exception. Tera is binary (use it or don't) and irreversible (gone for the whole game). Experts conserve it (25% turn-1 use), Nash says use it immediately (99%).

**Why it's different**: Tera sits outside the low-rank structure because it's not redundant — it's a single discrete choice with outsized consequences. Move selection has ~200 options that are mostly interchangeable (low-rank); tera is one binary switch that actually matters.

**30% of exploitability**: Restricting the best-response to match BC's tera choice drops exploitability by 0.41 out of 1.41. That's the largest single behavioral lever.

**Why experts conserve tera**: Rational from a multi-turn perspective — you might want it later when you have more information. Our analysis is turn-1-only, so it says "use it now." This is a known limitation.

**Aggression**: Best responses are also more aggressive (79% offensive targeting vs 52% for BC). The joint pattern shifts from balanced to heavily offensive.

### Section 6: Can Reward Optimization Escape Convention? (CQL)

**Why this section exists**: TA feedback said the project reads as game-theory + supervised learning, not RL. CQL is the RL component. But it also serves a scientific purpose: if BC is conventional because imitation never incentivizes breaking convention, maybe a reward-maximizing objective would escape it.

**Setup**: Same encoder architecture as BC, but trains a Q-function (expected reward per action) instead of imitating. Factored: Q(s,a) = Q_a + Q_b + Q_tera. CQL adds a conservative penalty so it doesn't go wild with out-of-distribution actions.

**Why horizon-1?** Our game is single-step (turn 1 only). No need for bootstrapping or target networks. Q(s,a) = E[reward | s, a].

**Results**:
- Exploitability: 1.40 -> 1.15 (17% reduction)
- TV(CQL, Nash) = 0.98 (still maximally far from Nash)
- TV(CQL, BC) = 0.60 (CQL and BC are very different from each other)
- Improvement is non-uniform: 26% reduction in most-exploitable quartile, ~0% in least

**The punchline**: CQL moves 60% of the distribution but gains only 17% in exploitability. Most of the movement is in payoff-irrelevant dimensions. The game's low-rank structure is the binding constraint, not the learning objective. If CQL = BC, that IS the result — convention is free even when you try to optimize your way out.

### Section 7: Ablations

**Autoregressive BC**: Let lead 2's action depend on lead 1's. No change in exploitability (1.41 -> 1.41). Confirms the independence assumption.

**Better dynamics model**: dyn_002 (d=64, lower val loss) finds MORE exploitability (1.41 -> 1.52). Our base number is a lower bound, not an overestimate.

**QRE rejection**: Tested whether experts play "noisy Nash." They don't — QRE fits terribly (lambda ~ 0.01 means basically uniform).

**Reward sensitivity**: Tested w_ko in {1, 3, 5}. Exploitability scales (0.58, 1.41, 2.25) but cross-play gap stays ~0 at all settings. Findings are robust to weight choice.

**Payoff-weighted TV**: Weight the TV distance by each action's contribution to payoff variance. TV drops from 0.99 to 0.019 — a 50x reduction. The BC-Nash divergence is almost entirely in payoff-irrelevant actions.

**Smoothed BR**: Start at BC, iteratively move toward best-response. Converges smoothly (no phase transition), consistent with 93% null-space energy.

### Section 8: Discussion

Summarizes the narrative, connects to TurnZero companion project (convention at every stage), highlights tera as the exception, and gives the "beyond Pokemon" prediction: low rank/action-ratio -> convention free; irreversible resources -> convention costly.

**Limitations to know for the meeting**:
1. Turn-1 only (tera finding is biased by myopic analysis)
2. Dynamics model noise (SNR 3.4:1 is fine but not enormous)
3. Cross-play gap cancellation is empirical, not guaranteed
4. Reg G only (but testable in other formats)
5. Later turns with switching may have higher effective rank

---

## Design Decisions Cheat Sheet

| Decision | Why |
|---|---|
| Turn 1 only | Cleanest simultaneous game; avoids sequential complications |
| 1500+ Elo cutoff | Expert play only; below this is too noisy |
| Exclude voluntary switches | Different action space; mixing would be comparing different games |
| Factored BC output | Joint space is ~120K; independence validated at 0.018 bits MI |
| Transformer encoder | Standard for set-structured input (12 Pokemon + field token) |
| Payoff weights (1/3/0.5) | Domain heuristic; KOs matter most; validated with sensitivity analysis |
| 500 matchups | CIs already tight; going higher has diminishing returns |
| Bootstrap CIs throughout | Non-parametric, no distributional assumptions |
| Factored CQL | Same justification as factored BC; plus makes logsumexp tractable |
| Horizon-1 CQL | Single-step game, no bootstrapping needed |

---

## Numbers to Have Ready

- **154,718** battles, **309,436** directed examples
- **Exploitability**: 1.41 [1.36, 1.46]
- **TV(BC, Nash)**: 0.99
- **Cross-play gap**: -0.02 [-0.07, 0.04]
- **Effective rank**: 2.9 [2.8, 2.9] at 95% energy
- **Null-space energy**: 93% [92.8, 93.2]
- **Tera contribution**: 30% of exploitability [29%, 32%]
- **CQL exploitability**: 1.15 [1.08, 1.23] (17% reduction)
- **TV(CQL, BC)**: 0.60 (policies are very different)
- **SNR**: 3.4:1
- **BC accuracy**: 44.8% top-1, 79.6% top-3

---

## What Was Cut (For TA Discussion)

The current paper is ~5 full pages. The previous version was longer. Here's what got cut and could be expanded back to reach 6 pages. Ask the TA: "Which of these would you prioritize adding back?"

### 1. Expanded Introduction (~0.3 pages)
The old intro opened with "What does it cost to play by habit instead of by calculation?" and had three bold-faced contribution bullets (Convention not strategy / Convention is free / Where convention fails), plus a paragraph connecting to TurnZero. The current intro is punchier but shorter. **Pro:** the contribution bullets made the structure very explicit. **Con:** the current puzzle-hook opening is stronger.

### 2. Separated Related Work: Price of Anarchy + Focal Points (~0.1 pages)
These were two separate subsections with more detail. Price of anarchy had its own paragraph tracing from Koutsoupias to Roughgarden. Focal points had a standalone paragraph on Schelling. Now merged into one. **Pro:** gives each concept proper treatment. **Con:** related work isn't what you're short on.

### 3. "Convention, Not Strategy" Paragraph (~0.1 pages)
In Section 3, there was a paragraph connecting to TurnZero's opponent ablation: "team selection contributes only -1.3pp to prediction accuracy — experts choose teams by convention, largely independent of the opponent. Turn-1 actions exhibit the same pattern." **Pro:** strengthens the cross-project narrative. **Con:** depends on whether citing TurnZero is appropriate.

### 4. Longer Game-Theoretic Framework (~0.1 pages)
The old version explicitly named poke-env, OpenAI Five, and fighting-game agents as precedent for weighted reward sums, with a sentence explaining why KOs are weighted higher. Currently just a citation cluster. **Pro:** makes the reward design less arbitrary-looking. **Con:** minor.

---

## Analyses Done But Not In Paper

These are results that exist in `results/` with completed scripts, but never made it into the current paper (or were in an older version and got cut). Any of these could be written up to fill space.

### 5. Mixture Interpolation (~0.3 pages) — BEST CANDIDATE
**Source:** `results/mixture_exploit/mixture_exploit.json`, `scripts/mixture_exploit.py`
**Was in an older version of the paper, then cut when CQL was added.**

Constructs blended strategies π_α = (1-α)*BC + α*Nash for α∈[0,1] and measures exploitability at each step. Result: exploitability decays approximately linearly (~1.41 × (1-α)) with **no phase transition**. This means there's no cliff — you don't suddenly need to be "close to Nash" to get Nash-like payoffs. The geometry between BC and Nash is flat, which is exactly what 93% null-space energy predicts.

**Why it's good:** Directly supports the low-rank story from yet another angle. Easy to explain ("we walked from BC to Nash and nothing interesting happened along the way"). Was already written up once. Could include a small table or even a figure.

### 6. Regret Dynamics — SKIP (Redundant)
**Source:** `results/phase2/regret.json` (487K)

External regret is 1.43 — just exploitability restated. Swap regret, BR mass, regret ratio are all different angles on "BC is exploitable," which Section 3 already covers. No new signal beyond what exploitability + SVD already tell us.

### 7. Indifference Curves — SKIP (Redundant)
**Source:** `results/phase2/indifference.json` (801K)

Tests whether BC satisfies the Nash indifference principle (all support actions yield equal payoff). It doesn't — payoff variance is 0.47. But of course it doesn't: BC isn't Nash. This is just "BC isn't Nash" in fancier language. Already covered by TV = 0.99.

### 8. Extended Case Studies (~0.3 pages) — USE THIS
**Source:** `results/exploitation/case_studies/extended_case_studies.json` (66K)
**Never in any version of the paper. 14 detailed matchups analyzed.**

This is real content that makes the abstract numbers tangible. Universal patterns across all 14 case studies:
- BR uses tera in **14/14** matchups; BC never does
- BC plays defensive (both_support) → BR shifts aggressive in 10/14 (71%)
- BR action at 0% Nash probability in 7/14 (exploiting from the null space)

**Best poster child — Matchup 23832** (Calyrex-Ice + Ogerpon-Hearthflame vs Kyurem-White + Maushold-Four):
- **BC convention**: Trick Room + Follow Me (defensive setup), payoff **-0.75**
- **Nash equilibrium**: High Horsepower + Horn Leech (aggressive targeting), payoff **+2.76**
- **BR exploit**: Copies Nash's targeting + adds tera (a resource BC ignores), payoff **+2.76**, gain of **3.51** (largest in dataset)
- **Story in one sentence**: Convention learned to play defensively when Nash demands aggression — and the 3.51 swing is the cost.

**Other notable matchups:**
- **Matchup 5711** (Tornadus + Chi-Yu vs Scream Tail + Gothitelle): Exploitability only 0.024 — BC is 97.6% optimal here. Yet BR still finds an exploit in the null space (0% Nash probability). Shows that even near-perfect convention gets exploited in dimensions Nash doesn't expect.
- **Matchup 31330** (Ogerpon-Cornerstone + Calyrex-Ice vs Whimsicott + Chi-Yu): Most exploitable at 3.80. BC plays Follow Me + Trick Room (defense); Nash spreads targets. Largest action space (192 actions for P1).

**Where to put it**: A case study paragraph in Section 5 (Where Convention Fails) or as a new subsection would give readers an "aha" moment. The Matchup 23832 example is paper-ready.

### 9. Model Capacity Robustness Table (~0.1 pages)
**Source:** `results/phase2/cache_d64.pkl`, `cache_d128.pkl`, + capacity consensus analysis
**Was a full table in an older version, currently compressed to one sentence (line 222).**

Three dynamics model capacities (d=32, 64, 128) with effective rank and payoff-weighted TV. Shows PW-TV stable at 0.019 across all capacities. Effective rank only goes 2.9→4.3 and the highest-capacity model overfits. Restoring the table makes the "low rank is real, not a model artifact" argument more convincing.

### 10. Unused Figures
Two generated figures exist in `figures/` but aren't referenced in the paper:

**`figures/fig2_deviation_projection.pdf`** — Deviation energy in SVD subspaces (corresponds to Table 3):

![Deviation Projection](../figures/fig2_deviation_projection.pdf)

Stacked bar chart showing fraction of BC-Nash deviation energy in top-k dimensions (red, payoff-relevant) vs null space (blue, payoff-irrelevant). The visual punch: the red bars are *tiny* at every k. At k=3, 93% of deviation energy is in the blue (null space). This makes the "most of how experts are 'wrong' doesn't matter" argument instantly visible — way more impactful than a table of numbers.

**`figures/fig3_rank_reduction.pdf`** — Rank-k approximation exploit ratio (corresponds to Table 4):

![Rank Reduction](../figures/fig3_rank_reduction.pdf)

Line plot of exploitability ratio (rank-k / full R) vs k, with error bars. Key feature: the curve hits ~97% at k=3 and is essentially flat after k=5. The red dotted line marks the 97.2% threshold. The wide error bars at k=1 and k=2 collapse by k=3, showing that 3 dimensions is the sweet spot — below that is noisy, above that is diminishing returns.

Either could replace a table with a figure, or supplement it. Figures are more skimmable than tables.

---

## Best Candidates for Expansion (Ranked)

| Priority | Content | Est. Pages | Status | Why |
|---|---|---|---|---|
| 1 | **Mixture interpolation** | ~0.3 | Was in paper, cut for CQL | Clean result, linear decay with no phase transition, directly supports low-rank story |
| 2 | **Case study example** (Matchup 23832) | ~0.2 | Never in paper, data ready | Makes abstract numbers tangible — "Trick Room vs aggression, 3.51 swing" |
| 3 | **Expanded intro** with contribution bullets | ~0.3 | Was in paper, cut for space | Makes structure explicit, already written |
| 4 | **Deviation projection figure** | ~0.1 | Figure exists, unused | Replaces/supplements Table 3, more visual |
| 5 | **Model capacity table** (restore) | ~0.1 | Was in paper, condensed to prose | Strengthens "not a model artifact" argument |
| 6 | **Convention not strategy** paragraph | ~0.1 | Was in paper, cut | TurnZero connection (if citing is okay) |
| 7 | **Longer game-theoretic framework** | ~0.1 | Was in paper, condensed | Makes reward design less arbitrary |
| — | ~~Regret dynamics~~ | — | SKIP | Redundant with exploitability |
| — | ~~Indifference curves~~ | — | SKIP | Redundant with TV distance |

**Easiest path to 6 pages:** #1 + #2 + #3 = ~0.8 pages. That's more than enough.

**Ask the TA:** "I have mixture interpolation (no phase transition between BC and Nash), a concrete case study (Matchup 23832: defensive convention vs aggressive Nash, 3.51 swing), and space for an expanded intro. Which would you prioritize? Or is there something else you'd rather see?"
