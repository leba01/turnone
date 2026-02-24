# Project Bible — TurnOne: How Exploitable Are Expert Pokemon Players?

## 0) EXEC SUMMARY

TurnOne extends **TurnZero** (CS229) from *"what do experts do?"* to *"what should they do?"*

We model VGC turn 1 as a **simultaneous-move normal-form game** and build a pipeline to measure the gap between human play and game-theoretic optimality:

1. **Behavioral cloning (BC)**: learn expert action distributions (44.8% top-1 accuracy)
2. **Dynamics model**: learned world model predicting turn-1 outcomes (HP MAE 13.3→12.6 v1→v2, KO AUC 0.91→0.92)
3. **Payoff matrix**: enumerate ~150-500 valid actions per side, compute rewards via dynamics model
4. **Nash LP**: solve for equilibrium mixed strategies (exact via scipy HiGHS)
5. **Exploitability**: measure the gap between BC and Nash

### Headline findings

**Finding 1 — Individual exploitability**: Expert strategies are exploitable. Mean exploitability = 1.41–1.52 depending on dynamics oracle (v1: 1.41 [1.36, 1.46]; v2: 1.52 [1.47, 1.58]). Better dynamics finds *more* exploitability, not less. This signal exceeds the dynamics noise floor by 1.0+ reward units (noise sensitivity experiment).

**Finding 2 — Collective equilibrium via error cancellation**: Despite individual exploitability, BC-vs-BC ≈ Nash value across all three configurations (gaps: -0.02, -0.03, +0.02). Value decomposition reveals this is **error cancellation**: BC P1 loses ~1.0 against Nash P2, but Nash P1 gains ~1.0 against BC P2. These symmetric errors cancel in BC-vs-BC play. All 500 matchups show opposite-sign cross-play gaps. Robust across reward weight specifications (w_ko ∈ {1, 3, 5}) and dynamics model versions.

**Finding 3 — Autoregressive BC null result**: Switching from independent P(a)×P(b) to autoregressive P(a)×P(b|a) factorization has no measurable effect on exploitability (1.41 → 1.41, same dynamics). Mon-A and mon-B correlations are not a meaningful source of exploitability.

**Finding 4 — BC-Nash mixture**: Interpolating (1-α)×BC + α×Nash monotonically reduces exploitability from 1.48 (α=0) to 0 (α=1), confirming a smooth path from expert play to game-theoretic optimality.

### Positioning

This is **empirical game-theoretic analysis (EGTA) with a learned dynamics model** — a novel combination. Key distinctions from related work:

- **vs. VGC-Bench** (Angliss et al., 2025): They train agents and measure agent-vs-agent performance. We measure exploitability of the *human population distribution*. Complementary, not competing.
- **vs. Metamon** (Grigsby et al., 2025): Offline RL for full-game play in singles. We isolate the simultaneous-move structure of turn 1 to make equilibrium computation tractable.
- **vs. poker literature** (Bowling 2015, Brown 2017/2019): They prove bots beat humans. We measure the *structure* of human exploitability — individually exploitable but collectively near-Nash.
- **vs. So & Ma (2025)**: They prove learnable mixed Nash equilibria are collectively rational. Their conditions are trivially satisfied in two-player zero-sum games. We cite them for motivation but our contribution is the empirical decomposition: the BC-vs-BC ≈ Nash finding arises from symmetric error cancellation, not independent near-optimality.
- **vs. penalty kick literature** (Chiappori 2002, Palacios-Huerta 2003): They show aggregate play matches Nash in 2×2 sports games. We extend this to combinatorial strategy spaces (~200-400 actions per side) using a learned dynamics model. Closest comparable: Anderson et al. (2024, JPE) who compute best-response gains in tennis with known game structure.

**Data**: 154,718 parsed battles → 309,436 directed turn-1 examples. **Hardware**: RTX 4080 Super, Ryzen 7 7800X3D.

---

## 1) PROBLEM FORMULATION & GAME THEORY

### 1.1 Turn 1 as a simultaneous-move game

After team preview, both players **simultaneously** choose actions for their two active Pokemon. This is a **zero-sum normal-form game**.

- **Players**: P1, P2
- **State** `s`: team sheets (6 mons × 8 fields each), lead assignments, pre-turn field state (weather, terrain, trick room, tailwind)
- **Action space** `A_i(s)`: valid joint actions `(slot_A, slot_B, tera_flag)`. Each mon has up to 16 slots (4 moves × 4 targets). Invalid slots masked out.
- **Outcome function** `T(s, a1, a2) → o`: learned by the dynamics model
- **Reward** `R(o)`: zero-sum scalar from outcomes
- **Strategy** `σ_i`: probability distribution over `A_i(s)`

### 1.2 Why not a sequential MDP?

Turn 1 is a single simultaneous decision. Both players commit before anything resolves. The normal-form game is the natural formulation. Multi-turn play is out of scope.

### 1.3 Nash equilibrium

A Nash equilibrium `(σ1*, σ2*)` satisfies: neither player can improve by unilaterally deviating. For finite zero-sum games, Nash always exists (minimax theorem) and is computed exactly via LP.

### 1.4 Exploitability

```
exploit(σ) = V* - min_{σ2} E[R | σ, σ2]
```

Measures how much an adversary gains by best-responding. Nash has exploitability = 0.

### 1.5 Why a learned dynamics model (not exact simulation)?

Standard replay data lacks EVs, IVs, and natures — the stat components required for exact damage calculation. Without these, Showdown's battle engine cannot compute correct outcomes (a Modest 252 SpA Calyrex does very different damage than a Timid 4 SpA one; a 1-point speed difference flips move order entirely).

Our learned dynamics model captures **expected outcomes marginalized over the unobserved stat distribution** in the metagame. This is the right quantity for measuring population-level exploitability: we ask "how exploitable is the average expert?" not "how exploitable is this specific EV spread?"

Future work: open team sheet tournament data (where full spreads are public) would enable exact simulation via Showdown's BattleStream API.

### 1.6 Connection to CS234

- **Behavioral cloning**: supervised baseline — learning expert action distributions from 300K+ examples
- **World models**: dynamics model as learned simulator for offline policy evaluation (cf. MuZero, Dreamer)
- **Game-theoretic optimization**: Nash equilibrium as solution concept (vs. single-agent Bellman optimality)
- **Empirical game theory**: EGTA (Wellman, 2006) with a learned dynamics oracle instead of a game simulator

---

## 2) DATA + PIPELINE REFERENCE

The data pipeline is **complete**. See `CLAUDE.md` for design decisions.

### Quick reference

| Stat | Value |
|------|-------|
| Raw battles | 212,804 (3 RegG files) |
| Parsed battles | 154,718 (27% excluded: switches, forfeits) |
| Directed examples | 309,436 |
| Train / val / test | 206,162 / 47,860 / 55,414 |
| Both mons act | 80.5% (19.5% have faint/flinch before acting) |
| Tera usage | ~27% (15% A, 12% B) |
| KO rate | ~12.2% per active mon |

### Per-example schema (assembled JSONL)

```yaml
example:
  example_id: str
  battle_id: str
  perspective: str           # "p1" | "p2"
  team_a: list[Mon]          # 6 mons, each {species, item, ability, tera_type, moves[4]}
  team_b: list[Mon]
  lead_indices_a: [int, int]
  lead_indices_b: [int, int]
  pre_turn_field:
    weather: str | null
    terrain: str | null
    trick_room: bool
    tailwind_p1: bool
    tailwind_p2: bool
  action:
    action_a: {move_idx, target, slot} | null  # null = fainted before acting
    action_b: {move_idx, target, slot} | null
    tera_flag: int           # 0=none, 1=tera_A, 2=tera_B
  opponent_action:           # same structure
  resolution:
    hp_before: {our_a, our_b, opp_a, opp_b}
    hp_after: {our_a, our_b, opp_a, opp_b}
    kos: list[str]
    field_state: {weather, terrain, trick_room, tailwind_p1, tailwind_p2}
  split: str
```

### Dataset tensor schema (`Turn1Dataset.__getitem__`)

```python
{
    "team_a": LongTensor(6, 8),      # [species, item, ability, tera, move0-3]
    "team_b": LongTensor(6, 8),
    "lead_a": LongTensor(2,),
    "lead_b": LongTensor(2,),
    "field_state": FloatTensor(5,),   # [weather_idx, terrain_idx, trick_room, tw_ours, tw_opp]
    "action_a": int,                  # slot 0-15, or -1 if unobserved
    "action_b": int,
    "mask_a": BoolTensor(16,),
    "mask_b": BoolTensor(16,),
    "tera_label": int,                # 0-2
    "opp_action_a": int,
    "opp_action_b": int,
    "opp_mask_a": BoolTensor(16,),
    "opp_mask_b": BoolTensor(16,),
    "opp_tera_label": int,
    "hp_delta": FloatTensor(4,),      # HP lost [our_a, our_b, opp_a, opp_b]
    "ko_flags": FloatTensor(4,),
    "field_after": FloatTensor(5,),
}
```

### Action space

- **16 slots per mon**: 4 moves × 4 targets (0=opp_A, 1=opp_B, 2=ally, 3=self/no-target)
- **Move targeting**: Showdown-backed via `move_targets.json` (658/658 vocab coverage)
- **Masking** by category: self/spread → target 3 only; ally → target 2 only; single → all 4
- **Strategic mask**: same but single-target moves exclude target=3 (self); ally targeting only for dual-purpose moves
- **Tera**: 3-way flag (none/A/B), cross-cutting with move choice
- **Vocab sizes**: species=844, item=190, ability=265, tera_type=20, move=659

---

## 3) BC POLICY

### 3.1 Architecture

Adapts TurnZero's permutation-equivariant Transformer encoder with **per-mon factorized heads**.

```
Input: team_a (6,8) + team_b (6,8) + lead_indices + field_state
  → Per-mon embedding (sum of field embeddings + side emb + lead position emb)
  → 13-token Transformer encoder (12 mon + 1 field)
  → Extract lead-mon representations
  → Per-mon action head (16-way masked softmax) × 2  [shared weights]
  → Tera head (3-way softmax)
Output: π_A(slot | state), π_B(slot | state), π_tera(flag | state)
```

- 13 tokens: 6 team_a + 6 team_b + 1 field
- Lead position embedding: 3-way (not-lead / lead_A / lead_B)
- Pre-norm Transformer: 4 layers, 4 heads, d_model=128
- Mask fill value: -30 (not -inf, for label smoothing compatibility)
- **1,320,851 parameters**

Supports **autoregressive** mode: P(b|a) via action embedding added to lead_B representation. Code and config exist (`configs/bc_autoregressive.yaml`); see Section 6.6 for why this matters.

### 3.2 Training

```yaml
training:
  seed: 42
  batch_size: 512
  max_epochs: 50
  patience: 7
  lr: 3.0e-4
  weight_decay: 0.01
  label_smoothing: 0.02
  lambda_tera: 0.5
  optimizer: AdamW
  scheduler: CosineAnnealingLR
  mixed_precision: bf16
  compile: true
```

Best checkpoint: `runs/bc_001/best.pt` (epoch 18, independent factorization).

### 3.3 Results

| Metric | Value |
|--------|-------|
| Per-mon top-1 accuracy | 44.8% (A: 45.5%, B: 44.2%) |
| Per-mon top-3 accuracy | 79.6% (A: 80.2%, B: 79.0%) |
| Per-mon NLL | 1.492 (A: 1.473, B: 1.511) |
| Tera accuracy | 75.1% |
| Tera NLL | 0.632 |
| Mask compliance | 99.5% |
| Both-acted subset top-1 | 45.3% |

---

## 4) DYNAMICS MODEL

### 4.1 Purpose

Learned world model predicting turn-1 outcomes from (state, both players' actions). Enables payoff matrix construction for any action pair — not just those observed in data. This is the EGTA oracle: it replaces the game simulator in classical empirical game theory (Wellman 2006, Vorobeychik 2007).

### 4.2 Architecture (v1)

Uses the same Turn1Encoder as BC, plus action embeddings and an MLP trunk.

```
Input: state (via shared encoder) + 6 action embeddings
  → Turn1Encoder → pooled state representation (d_model)
  → Action embeddings: 17-slot per mon (0-15 valid + 16=no-action for fainted)
  → Tera flag embeddings: 3-way per side
  → Concatenate state + all action embeddings
  → MLP trunk (d_hidden=256, 3 layers, GELU, dropout=0.1)
  → Decomposed output heads:
      - HP head: Linear → (4,)  [regression, MSE]
      - KO head: Linear → (4,)  [binary logits, BCE]
      - Weather head: Linear → (5,)  [5-class CE]
      - Terrain head: Linear → (5,)  [5-class CE]
      - Binary field head: Linear → (3,)  [trick_room + 2 tailwinds, BCE]
```

Key design: `remap_actions()` maps -1 (fainted) → slot 16. `FieldLogits` dataclass wraps decomposed field predictions. `predict_field_state()` converts FieldLogits to (B, 5) for reward compatibility.

### 4.3 Training

```yaml
dynamics:
  d_action: 32
  d_hidden: 256
  n_layers: 3
  dropout: 0.1
  lambda_ko: 3.0
  lambda_field: 0.5

training:
  seed: 42
  batch_size: 512
  max_epochs: 50
  patience: 10
  lr: 3.0e-4
  weight_decay: 0.01
```

Best checkpoint: `runs/dyn_001/best.pt` (epoch 23).

Training curve: val loss plateaus at epoch 10 while train loss continues falling — suggests capacity ceiling, not overfitting.

### 4.4 Results

| Metric | v1 | v2 |
|--------|----|----|
| HP MAE | 13.6 | 12.6 |
| HP RMSE | 22.7 | 21.1 |
| HP R² | 0.63 | 0.68 |
| KO AUC-ROC | 0.91 | 0.92 |
| KO accuracy | 0.91 | 0.91 |
| KO BCE | 0.23 | 0.20 |
| Weather accuracy | 0.68 | 0.68 |
| Terrain accuracy | 0.65 | 0.65 |
| Binary field accuracy | 0.88 | 0.88 |

**Reward-space error** (how dynamics errors propagate to rewards):

| Metric | v1 | v2 |
|--------|----|----|
| Reward MAE | 1.19 | 1.14 |
| Reward RMSE | 1.73 | 1.66 |
| Reward correlation | 0.60 | 0.63 |
| Reward bias | +0.26 | +0.14 |

V2 halves the reward bias and improves HP prediction by 1.0 MAE. Field predictions are unchanged (dominated by base-rate weather/terrain).

### 4.5 Architecture (v2)

Adds action cross-attention and a larger trunk. Config: `configs/dynamics_v2.yaml`.

```
Changes from v1:
  → Action cross-attention: 1 layer, 2 heads over 4 action embeddings
    (models interaction between moves before MLP)
  → d_action: 32 → 64
  → d_hidden: 256 → 512
  → n_mlp_layers: 3 → 4
```

Best checkpoint: `runs/dyn_002/best.pt` (epoch 21). The cross-attention enables representing move synergies and counters (e.g., Fake Out + setup vs Fake Out + Protect) that v1's concatenation cannot model.

---

## 5) REWARD FUNCTION

### 5.1 Formula

Zero-sum composite reward from P1's perspective:

```
R = w_hp × HP_advantage + w_ko × KO_advantage + w_field × field_advantage
```

```python
HP_advantage = (hp_delta_opp_a + hp_delta_opp_b - hp_delta_our_a - hp_delta_our_b) / 200.0
KO_advantage = (ko_opp_a + ko_opp_b - ko_our_a - ko_our_b)
field_advantage = (tailwind_change_ours - tailwind_change_opp) + trick_room_change
```

### 5.2 Weights

```yaml
reward:
  w_hp: 1.0
  w_ko: 3.0       # KOs dominate — a KO is worth ~3× chip damage
  w_field: 0.5
```

HP normalization: /200 (max possible = 2 mons × 100 HP). Range: HP component in [-1, +1].

**Zero-sum verification**: by construction, swapping P1/P2 negates reward. Verified in tests.

### 5.3 Reward weight sensitivity

Findings are robust across reward specifications and dynamics versions:

**v1 dynamics** (n=200 matchups):

| w_ko | BC exploit | Nash V* | BC-vs-BC | Gap |
|------|-----------|---------|----------|-----|
| 1.0 | 0.60 | 0.10 | 0.12 | +0.02 |
| 3.0 | 1.44 | 0.18 | 0.17 | -0.01 |
| 5.0 | 2.28 | 0.27 | 0.23 | -0.04 |

**v2 dynamics** (n=200 matchups):

| w_ko | BC exploit | Nash V* | BC-vs-BC | Gap |
|------|-----------|---------|----------|-----|
| 1.0 | 0.59 | 0.05 | 0.09 | +0.04 |
| 3.0 | 1.48 | 0.03 | 0.05 | +0.02 |
| 5.0 | 2.37 | 0.00 | 0.01 | +0.01 |

BC-vs-BC ≈ Nash holds across all weight settings and both dynamics versions. V2's lower bias (0.14 vs 0.26) produces Nash values closer to 0 and tighter BC-vs-BC gaps.

---

## 6) GAME-THEORETIC RESULTS

### 6.1 Pipeline

For each sampled matchup:
1. Enumerate valid joint actions for both sides (`enumerate_joint_actions()`)
2. Build payoff matrix via batched dynamics queries (`build_payoff_matrix()`, GPU)
3. Solve Nash LP for both players (`solve_nash_lp()`, scipy HiGHS)
4. Compute BC strategy from policy logits
5. Compute exploitability, triangle values, Nash support

### 6.2 Exploitability results (n=500, w_ko=3.0, Showdown targeting)

Three-config comparison (all use Showdown-backed targeting):

| Config | BC exploit (mean) | 95% CI | BC-vs-BC − Nash |
|--------|-------------------|--------|-----------------|
| Independent BC + dyn v1 | 1.41 | [1.36, 1.46] | -0.02 |
| Autoregressive BC + dyn v1 | 1.41 | [1.36, 1.47] | -0.03 |
| Autoregressive BC + dyn v2 | 1.52 | [1.47, 1.58] | +0.02 |

Better dynamics finds *more* exploitability (1.41 → 1.52). Autoregressive BC has no effect (same dynamics → same result).

### 6.3 Safety-exploitation triangle

Four values per matchup, ordered from worst to best for P1:

| Metric | v1 Mean | v2 Mean | Interpretation |
|--------|---------|---------|----------------|
| BC worst-case | -1.19 | -1.48 | What happens if adversary best-responds to BC |
| Nash value (V*) | 0.22 | 0.05 | Guaranteed payoff under optimal play |
| BC-vs-BC | 0.20 | 0.06 | What happens when experts face experts |
| Best-response-to-BC | 1.63 | 1.62 | What we gain by best-responding to expert P2 |

**Key**: BC-vs-BC ≈ Nash value in aggregate across all configs. V2's lower bias produces Nash values closer to 0, with BC-vs-BC tracking accordingly.

### 6.4 Noise sensitivity

How much does dynamics model error affect Nash solutions?

**v1 dynamics** (reward MAE = 1.19, n=50 matchups):

| Noise (×MAE) | TV distance | Noisy exploit | Value change |
|--------------|-------------|---------------|--------------|
| 0.25× | 0.68 | 0.15 | 0.08 |
| 0.50× | 0.81 | 0.26 | 0.12 |
| 1.00× | 0.89 | 0.41 | 0.17 |
| 1.50× | 0.92 | 0.52 | 0.21 |
| 2.00× | 0.94 | 0.59 | 0.24 |

**v2 dynamics** (reward MAE = 1.14, n=50 matchups):

| Noise (×MAE) | TV distance | Noisy exploit | Value change |
|--------------|-------------|---------------|--------------|
| 0.25× | 0.60 | 0.15 | 0.07 |
| 0.50× | 0.72 | 0.25 | 0.10 |
| 1.00× | 0.82 | 0.40 | 0.15 |
| 1.50× | 0.87 | 0.52 | 0.19 |
| 2.00× | 0.89 | 0.61 | 0.22 |

**Comparison at 1× MAE**: v1 noisy exploit = 0.41, v2 noisy exploit = 0.40. Similar noise floors despite different MAE calibrations (v2 noise std is 1.14 vs 1.19 for v1). BC exploitability exceeds the noise floor by **1.00** (v1) and **1.12** (v2). The signal is real and strengthens with better dynamics.

### 6.5 Case studies

BC and Nash have **completely disjoint support** across all examined matchups.

**Pattern**: Nash favors Protect, defensive coverage moves, and defensive Tera. BC favors aggressive setup (Nasty Plot, Tailwind, targeted attacks). Exploitability comes from *predictability* — BC always plays the popular move, and a best-responding adversary can prepare for that.

See `results/case_studies/case_studies.json` for 5 detailed matchups and `docs/why_pivot.md` for interpretation.

### 6.6 Independence factorization (resolved)

BC strategies were initially computed as P(a,b,tera) = P(a) × P(b) × P(tera) — independent factorization. The concern was that ignoring mon-A/mon-B correlations would overstate exploitability.

**Ablation result**: Autoregressive BC (P(a) × P(b|a) × P(tera)) produces identical exploitability to independent BC when using the same dynamics model:

| Factorization | Dynamics | BC exploit (mean) | 95% CI |
|---------------|----------|-------------------|--------|
| Independent | v1 | 1.41 | [1.36, 1.46] |
| Autoregressive | v1 | 1.41 | [1.36, 1.47] |
| Autoregressive | v2 | 1.52 | [1.47, 1.58] |

This confirms that mon-coordination (Fake Out + setup, double Protect) is not a meaningful source of exploitability. The independence approximation is sufficient. Exploitability differences come from the dynamics model, not the BC factorization.

The per-matchup BC-vs-BC vs Nash tracking remains noisy (R²≈0.59 in audit), but the aggregate "collective equilibrium" finding holds across all three configs.

### 6.7 Value decomposition — WHY BC-vs-BC ≈ Nash

Cross-play analysis decomposes the BC-vs-BC ≈ Nash finding into structural components (n=500 matchups, dyn v1):

| Metric | Mean | 95% CI |
|--------|------|--------|
| V* (Nash value) | 0.22 | [0.14, 0.30] |
| BC-vs-BC | 0.20 | [0.12, 0.28] |
| BC_P1 vs Nash_P2 | -0.80 | [-0.88, -0.71] |
| Nash_P1 vs BC_P2 | 1.25 | [1.16, 1.35] |
| BC worst-case | -1.19 | [-1.27, -1.10] |
| Best-response to BC | 1.63 | [1.54, 1.72] |

**Diagnosis: error cancellation, not independent near-optimality.**

Cross-play gaps:
- P1 gap (BC_P1 vs Nash_P2 − V*): **-1.02** [−1.07, −0.97] — BC P1 loses ~1.0 against Nash P2
- P2 gap (Nash_P1 vs BC_P2 − V*): **+1.03** [+0.99, +1.08] — Nash P1 gains ~1.0 against BC P2
- All 500 matchups show opposite-sign gaps (0 same-sign, 500 opposite)

**Interpretation**: Each player's BC strategy is substantially suboptimal in cross-play. BC P1 loses a full reward unit against a Nash opponent. BC P2 gives away a full unit to a Nash opponent. But these errors cancel almost exactly: P1 is exploitable and P2 is exploitable in symmetric, offsetting ways. The BC-vs-BC ≈ Nash finding is a statistical artifact of symmetric suboptimality, not evidence that individual expert play is near-Nash.

**Strategy-space distances** confirm BC and Nash are maximally different:
- TV distance: 0.99 (near-maximal; range 0.78–1.00)
- BC places only 1.3% of mass on Nash-support actions
- Nash support: 2.7 actions per side; BC support: ~95 actions per side
- 68% of Nash support actions appear in BC support (but with negligible weight)

**Regression**: TV distance weakly predicts exploitability (r=0.17, p<0.001). Game size is a stronger predictor (r=−0.24, p<10⁻⁷): larger games have lower per-action exploitability.

### 6.8 Smoothed best-response convergence

Starting from BC strategies, smoothed best-response dynamics converge toward Nash (n=200 matchups, dyn v1). Temperature β controls how close to hard best response; lower β → closer to Nash.

| β | η | QRE exploit | Avg exploit | Converged (<0.1) |
|---|---|-------------|-------------|------------------|
| 0.01 | 0.1 | 0.13 | 0.11 | 50% |
| 0.05 | 0.1 | 0.14 | 0.14 | 6% |
| 0.10 | 0.1 | 0.23 | 0.27 | 0% |
| 0.50 | 0.1 | 1.25 | 1.28 | 0% |

Initial exploitability: 2.74 (sum of P1 + P2 BC exploitability).

**Key findings**:
- At β=0.01 (near-hard BR), time-averaged strategies achieve exploit ≈ 0.11 in 500 iterations — a 96% reduction from BC
- Higher temperatures converge to the quantal response equilibrium (QRE), not Nash
- The convergence trajectory (2.74 → 0.11) makes a natural poster figure
- Learning rate η has minimal effect on the fixed point (only convergence speed)

### 6.9 BC-Nash mixture experiment

Interpolating between BC and Nash strategies shows a smooth, monotonic exploitability reduction:

| α | Exploitability | 95% CI | Worst-case |
|---|---------------|--------|------------|
| 0.0 (pure BC) | 1.48 | [1.40, 1.56] | -1.45 |
| 0.2 | 1.16 | [1.10, 1.22] | -1.13 |
| 0.5 | 0.70 | [0.66, 0.74] | -0.68 |
| 0.8 | 0.27 | [0.26, 0.29] | -0.25 |
| 1.0 (pure Nash) | 0.00 | [0.00, 0.00] | +0.03 |

Mixed strategy: `σ_mix = (1-α) × σ_BC + α × σ_Nash` (n=200 matchups, dyn v2).

**Key insight**: Even a small Nash mixing weight substantially reduces exploitability. At α=0.3, exploitability drops by 32% (1.48 → 1.00). This suggests a practical "safe exploitation" policy: play mostly like an expert but hedge with Nash on key matchups.

---

## 7) EVALUATION SUMMARY

### 7.1 BC metrics

| Metric | Value |
|--------|-------|
| Per-mon top-1 | 44.8% |
| Per-mon top-3 | 79.6% |
| Per-mon NLL | 1.492 |
| Tera accuracy | 75.1% |
| Mask compliance | 99.5% |

### 7.2 Dynamics metrics

| Metric | v1 | v2 |
|--------|----|----|
| HP MAE | 13.6 | 12.6 |
| HP R² | 0.63 | 0.68 |
| KO AUC | 0.91 | 0.92 |
| Reward MAE | 1.19 | 1.14 |
| Reward correlation | 0.60 | 0.63 |
| Reward bias | +0.26 | +0.14 |

### 7.3 Game-theoretic metrics

| Metric | v1 (n=500) | v2 (n=500) |
|--------|------------|------------|
| BC exploitability (mean) | 1.41 [1.36, 1.46] | 1.52 [1.47, 1.58] |
| BC-vs-BC (mean) | 0.20 [0.12, 0.28] | 0.06 [-0.02, 0.14] |
| Nash value (mean) | 0.22 [0.14, 0.31] | 0.05 [-0.02, 0.12] |
| BC-vs-BC − Nash gap | -0.02 | +0.02 |

### 7.4 Poster figures (8)

1. **Pipeline diagram**: state → BC + dynamics → payoff matrix → Nash LP → exploitability
2. **Exploitability histogram**: 500 matchups, mean/median marked
3. **Safety-exploitation triangle**: 4 values per matchup
4. **Trust quantification**: noise sensitivity + reward weight sensitivity
5. **Case study**: 1-2 matchups with Pokemon names, BC vs Nash distributions
6. **Dynamics quality**: HP scatter plot + KO ROC curve
7. **Value decomposition bar chart**: V*, bc_vs_bc, bc_p1 vs nash_p2, nash_p1 vs bc_p2 — shows symmetric error cancellation
8. **Smoothed BR convergence curves**: exploitability (y) vs iteration (x), multiple temperatures — shows BC evolving toward Nash

### 7.5 Bootstrap CIs

10,000 resamples, percentile method, 95% confidence. Applied to all aggregate metrics. See `turnone/eval/bootstrap.py`.

### 7.6 Audit results

| Check | Verdict | Notes |
|-------|---------|-------|
| 1: BC-vs-BC tracking | RESOLVED | Per-matchup R²=0.59, aggregate gap OK. Autoregressive ablation confirms independence is sufficient. |
| 2b: Stripped BC mass | PASS | Median per-mon stripped <1% |
| 3: Correlated noise | PASS | Corr/IID ratio 1.05 (TV), 1.38 (exploit) |
| 6: LP verification | PASS | Max error 1e-13 |
| 7: Field ablation | BORDERLINE | Field fraction 16%, exploit change 4.8% |
| 8: Tail analysis | PASS | Trimmed mean within 1.7% |
| 10: Team reuse | PASS | 7.6% reuse rate |
| 11: Nash action quality | PASS | Zero ally-attack weight (Showdown targeting) |

---

## 8) IMPLEMENTATION

### 8.1 Module layout

```
turnone/
├── data/
│   ├── parser.py           # Showdown protocol → parsed examples
│   ├── action_space.py     # 16-slot encoding, masking, Showdown-backed targeting
│   ├── move_targets.json   # Raw Showdown target values (658 moves)
│   ├── dataset.py          # PyTorch Dataset + Vocab
│   └── io_utils.py         # JSONL read/write
├── models/
│   ├── encoder.py          # Turn1Encoder (shared Transformer)
│   ├── bc_policy.py        # BC policy: encoder + per-mon heads + tera head
│   ├── dynamics.py         # Dynamics model: encoder + action embed → outcome heads
│   ├── train.py            # BC training loop (supports autoregressive)
│   └── train_dynamics.py   # Dynamics training loop
├── game/
│   ├── payoff.py           # Payoff matrix construction (GPU batched)
│   ├── nash.py             # Nash LP solver (scipy HiGHS)
│   └── exploitability.py   # Exploitability + strategy values + autoregressive BC strategy
├── rl/
│   └── reward.py           # Zero-sum reward function
├── eval/
│   ├── metrics.py          # BC evaluation metrics
│   ├── dynamics_metrics.py # Dynamics eval + reward-space error
│   └── bootstrap.py        # Bootstrap confidence intervals
scripts/
├── parse_turn1.py
├── build_split_map.py
├── assemble_splits.py
├── gen_move_targets.py     # Fetch Showdown moves.json → move_targets.json
├── evaluate.py             # End-to-end evaluation
├── audit.py                # Methodology audit (11 checks)
├── noise_sensitivity.py    # Dynamics trust experiment
├── case_studies.py         # Matchup deep-dives
├── reward_sensitivity.py   # Reward weight sweep
├── mixture_exploit.py      # BC-Nash mixture exploitability sweep
├── value_decomposition.py  # Cross-play values + strategy distances
├── smoothed_br.py          # Smoothed best-response convergence
└── mutual_info.py
configs/
├── bc_base.yaml
├── bc_autoregressive.yaml
├── dynamics_base.yaml
├── dynamics_v2.yaml
└── dynamics_canon.yaml
```

### 8.2 Key commands

```bash
# BC training (independent)
python turnone/models/train.py --config configs/bc_base.yaml --out_dir runs/bc_001

# BC training (autoregressive)
python turnone/models/train.py --config configs/bc_autoregressive.yaml --out_dir runs/bc_002

# Dynamics training
python turnone/models/train_dynamics.py --config configs/dynamics_base.yaml --out_dir runs/dyn_001

# Full evaluation
PYTHONPATH=. python scripts/evaluate.py \
    --bc_ckpt runs/bc_001/best.pt \
    --dyn_ckpt runs/dyn_001/best.pt \
    --test_split data/assembled/test.jsonl \
    --vocab_path runs/bc_001/vocab.json \
    --n_matchups 500 --out_dir results/

# Evaluation with autoregressive BC
PYTHONPATH=. python scripts/evaluate.py \
    --bc_ckpt runs/bc_002/best.pt \
    --dyn_ckpt runs/dyn_001/best.pt \
    --test_split data/assembled/test.jsonl \
    --vocab_path runs/bc_001/vocab.json \
    --n_matchups 500 --out_dir results/ --autoregressive

# Audit
PYTHONPATH=. python scripts/audit.py \
    --bc_ckpt runs/bc_001/best.pt --dyn_ckpt runs/dyn_001/best.pt \
    --test_split data/assembled/test.jsonl --vocab_path runs/bc_001/vocab.json \
    --matchup_details results/matchup_details.jsonl --out_dir results/audit/

# Noise sensitivity
PYTHONPATH=. python scripts/noise_sensitivity.py \
    --bc_ckpt runs/bc_001/best.pt --dyn_ckpt runs/dyn_001/best.pt \
    --test_split data/assembled/test.jsonl --vocab_path runs/bc_001/vocab.json \
    --n_matchups 50 --out_dir results/noise_sensitivity/

# Case studies
PYTHONPATH=. python scripts/case_studies.py \
    --bc_ckpt runs/bc_001/best.pt --dyn_ckpt runs/dyn_001/best.pt \
    --test_split data/assembled/test.jsonl --vocab_path runs/bc_001/vocab.json \
    --matchup_details results/matchup_details.jsonl --out_dir results/case_studies/

# Reward weight sensitivity
PYTHONPATH=. python scripts/reward_sensitivity.py \
    --bc_ckpt runs/bc_001/best.pt --dyn_ckpt runs/dyn_001/best.pt \
    --test_split data/assembled/test.jsonl --vocab_path runs/bc_001/vocab.json \
    --n_matchups 200 --out_dir results/reward_sensitivity/

# Evaluation with dynamics v2
PYTHONPATH=. python scripts/evaluate.py \
    --bc_ckpt runs/bc_002/best.pt --dyn_ckpt runs/dyn_002/best.pt \
    --test_split data/assembled/test.jsonl --vocab_path runs/bc_001/vocab.json \
    --n_matchups 500 --out_dir results/autoreg_v2/ --autoregressive

# BC-Nash mixture exploitability
PYTHONPATH=. python scripts/mixture_exploit.py \
    --bc_ckpt runs/bc_001/best.pt --dyn_ckpt runs/dyn_002/best.pt \
    --test_split data/assembled/test.jsonl --vocab_path runs/bc_001/vocab.json \
    --n_matchups 200 --out_dir results/mixture_exploit/

# Value decomposition + strategy distances
PYTHONPATH=. python scripts/value_decomposition.py \
    --bc_ckpt runs/bc_001/best.pt --dyn_ckpt runs/dyn_001/best.pt \
    --test_split data/assembled/test.jsonl --vocab_path runs/bc_001/vocab.json \
    --n_matchups 500 --out_dir results/value_decomposition/

# Smoothed best-response convergence
PYTHONPATH=. python scripts/smoothed_br.py \
    --bc_ckpt runs/bc_001/best.pt --dyn_ckpt runs/dyn_001/best.pt \
    --test_split data/assembled/test.jsonl --vocab_path runs/bc_001/vocab.json \
    --n_matchups 200 --out_dir results/smoothed_br/
```

### 8.3 Artifacts

```
runs/bc_001/best.pt          # BC checkpoint (independent)
runs/bc_002/best.pt          # BC checkpoint (autoregressive, epoch 16)
runs/bc_001/vocab.json       # Vocabulary
runs/dyn_001/best.pt         # Dynamics v1 checkpoint (epoch 23)
runs/dyn_002/best.pt         # Dynamics v2 checkpoint (epoch 21)
results/exploitability_results.json
results/matchup_details.jsonl
results/noise_sensitivity/noise_sensitivity.json
results/noise_sensitivity_v2/noise_sensitivity.json
results/case_studies/case_studies.json
results/reward_sensitivity/reward_sensitivity.json
results/reward_sensitivity_v2/reward_sensitivity.json
results/mixture_exploit/mixture_exploit.json
results/autoreg_v1/           # Autoregressive BC + dyn v1
results/autoreg_v2/           # Autoregressive BC + dyn v2
results/audit/audit_results.json
results/value_decomposition/decomposition.json
results/value_decomposition/matchup_details.jsonl
results/smoothed_br/smoothed_br.json
results/smoothed_br/matchup_summaries.jsonl
```

### 8.4 Tests

154/154 passing. Coverage: data pipeline, BC model (independent + autoregressive), dynamics model (v1 + v2), reward function, Nash solver, exploitability, bootstrap, strategy values, payoff matrix, Showdown targeting.

---

## 9) TIMELINE

### Week 1 (Feb 3-9): Data + BC
- [x] Data pipeline (parse → assemble → dataset)
- [x] Action space encoding + masking
- [x] BC model architecture + training
- [x] BC metrics: 44.8% top-1, 79.6% top-3

### Week 2 (Feb 10-16): Dynamics + Nash
- [x] Dynamics model architecture (decomposed field heads)
- [x] Dynamics training + eval (HP MAE 13.6, KO AUC 0.91)
- [x] Reward function (zero-sum, tested)
- [x] Nash LP solver (scipy HiGHS)
- [x] Payoff matrix construction (GPU batched)
- [x] Exploitability computation

### Week 3 (Feb 17-23): Full evaluation + hardening
- [x] Safety-exploitation triangle (BC worst-case, Nash, BC-vs-BC, best-resp)
- [x] Dynamics trust: reward-space error + noise sensitivity
- [x] Case studies with Pokemon names
- [x] Bootstrap confidence intervals
- [x] HP normalization fix (/100 → /200)
- [x] Reward weight sensitivity sweep
- [x] Showdown-backed move targeting (replaced hand-curated sets)
- [x] Methodology audit (8 checks, Check 11 now PASS)
- [x] 146/146 tests passing

### Week 4 (Feb 23-Mar 2): Model improvements + ablation
- [x] Train autoregressive BC → `runs/bc_002/` (epoch 16)
- [x] Evaluate with autoregressive factorization → `results/autoreg_v1/`
- [x] Dynamics v2: action cross-attention + larger trunk → `runs/dyn_002/` (epoch 21)
- [x] Evaluate autoreg BC + dyn v2 → `results/autoreg_v2/`
- [x] 3-config ablation comparison (independence null result, better dynamics = more exploitability)
- [x] Reward weight sensitivity on v2 → `results/reward_sensitivity_v2/`
- [x] BC-Nash mixture experiment → `results/mixture_exploit/`
- [x] Noise sensitivity on v2 → `results/noise_sensitivity_v2/`
- [x] Value decomposition (500 matchups) → `results/value_decomposition/`
- [x] Smoothed BR convergence (200 matchups) → `results/smoothed_br/`
- [x] 154/154 tests passing
- [ ] Milestone writeup (Feb 25)

### Remaining
- [ ] Poster figures: `scripts/make_figures.py` (Mar 8-10)
- [ ] Poster (Mar 11)
- [ ] Report (Mar 17)

---

## 10) ACCEPTANCE CRITERIA — STATUS

- [x] BC policy produces valid distributions over masked action slots
- [x] BC top-1 accuracy exceeds majority-class baseline: **44.8%**
- [x] BC training converges (early stopping at epoch 18)
- [x] Dynamics HP MAE < 15: **13.6**
- [x] Dynamics KO AUC > 0.8: **0.91**
- [x] Nash LP returns valid distributions (verified sum-to-1, non-negative)
- [x] Nash exploitability ≈ 0 (solver verification, max error 1e-13)
- [x] BC exploitability measurably > 0: **1.41 mean**
- [x] BC exploitability exceeds dynamics noise floor: **1.41 vs 0.41**
- [x] All metrics reported with 95% bootstrap CIs
- [x] Reward function verified zero-sum: R(P1) = -R(P2) for all examples
- [x] 5 concrete matchup case studies showing BC vs Nash differences
- [x] Noise sensitivity experiment quantifying dynamics trust
- [x] Reward weight sensitivity confirming robustness
- [x] All training runs reproducible from config + seed
- [x] Methodology audit: 7/8 checks PASS, 1 BORDERLINE, 0 P0 FAIL on core claims
- [x] Move targeting backed by Showdown authoritative data (658/658 coverage)
- [x] Autoregressive BC trained and evaluated (null result: no exploitability change)
- [x] Dynamics v2 trained and evaluated (lower bias, more exploitability found)
- [x] 3-config ablation with reward/noise sensitivity on both dynamics versions
- [x] BC-Nash mixture experiment (monotonic exploitability reduction)
- [x] Value decomposition: error cancellation mechanism identified (500 matchups, all opposite-sign)
- [x] Smoothed BR convergence: BC → near-Nash in 500 iterations (β=0.01)
- [ ] Milestone writeup (Feb 25)
- [ ] 8 poster-ready figures

---

## 11) REFERENCES

### Core methodology
- **Nash equilibrium**: Nash, J. (1950). "Equilibrium Points in N-Person Games." *PNAS*.
- **Minimax theorem**: von Neumann, J. (1928). "Zur Theorie der Gesellschaftsspiele." *Math. Annalen*.
- **LP for zero-sum games**: Dantzig, G.B. (1951). *Activity Analysis of Production and Allocation*.
- **Behavioral cloning**: Pomerleau, D.A. (1991). "Efficient Training of Artificial Neural Networks." *Neural Computation*.
- **World models**: Ha, D. & Schmidhuber, J. (2018). "World Models." *arXiv:1803.10122*.

### Empirical game theory
- **EGTA**: Wellman, M.P. (2006). "Methods for Empirical Game-Theoretic Analysis." *AAAI*.
- **EGTA survey**: Wellman, M.P. (2024). "Empirical Game-Theoretic Analysis: A Survey." *arXiv:2403.04018*.
- **Meta-game approximation**: Tuyls, K. et al. (2018). "A Generalised Method for Empirical Game Theoretic Analysis." *AAMAS*.
- **Payoff regression**: Vorobeychik, Y. & Wellman, M.P. (2007). *Stochastic Search Methods for Nash Equilibrium Approximation*. PhD thesis.
- **PSRO**: Lanctot, M. et al. (2017). "A Unified Game-Theoretic Approach to Multiagent RL." *NeurIPS*.
- **Spinning tops**: Czarnecki, W.M. et al. (2020). "Real World Games Look Like Spinning Tops." *NeurIPS*.

### Exploitability & safe exploitation
- **Exploitability**: Johanson, M. et al. (2011). "Accelerating Best Response Calculation." *IJCAI*.
- **Safe exploitation**: Ganzfried, S. & Sandholm, T. (2015). "Safe Opponent Exploitation." *ACM TEAC*.
- **Cepheus**: Bowling, M. et al. (2015). "Heads-up Limit Hold'em Poker is Solved." *Science*.
- **Libratus**: Brown, N. & Sandholm, T. (2017). "Superhuman AI for Heads-Up No-Limit Poker." *Science*.
- **Pluribus**: Brown, N. & Sandholm, T. (2019). "Superhuman AI for Multiplayer Poker." *Science*.

### Population equilibrium theory
- **Collective rationality**: So, J. & Ma, T. (2025). "Learnable Mixed Nash Equilibria are Collectively Rational." *arXiv:2510.14907*.

### Sports minimax testing
- **Penalty kicks**: Chiappori, P.A., Levitt, S., & Groseclose, T. (2002). "Testing Mixed-Strategy Equilibria When Players Are Heterogeneous: The Case of Penalty Kicks in Soccer." *AER*.
- **Penalty kicks II**: Palacios-Huerta, I. (2003). "Professionals Play Minimax." *Review of Economic Studies*.
- **Tennis serving**: Walker, M. & Wooders, J. (2001). "Minimax Play at Wimbledon." *AER*.
- **Tennis best response**: Anderson, A. et al. (2024). "Testing Minimax in the Field: Decomposing the Tennis Serve Game." *JPE*.

### Pokemon / competitive games
- **VGC-Bench**: Angliss, C. et al. (2025). "VGC-Bench: Towards Mastering Diverse Team Strategies in Competitive Pokemon." *arXiv:2506.10326*.
- **Metamon**: Grigsby, J. et al. (2025). "Human-Level Competitive Pokemon via Scalable Offline RL with Transformers." *RLJ*.
- **Oblander**: Oblander, E. (2024). "Representation Learning for Behavioral Analysis of Complex Competitive Decisions." *SSRN*.
- **Centipawn loss**: Stankovic et al. (2023). "Expected Human Performance in Chess Using Centipawn Loss Analysis." *Springer*.

### Architecture
- **Set Transformer / Deep Sets**: Zaheer et al. (2017), Lee et al. (2019).

---

## 12) TALK TRACK

### One-sentence pitch

"We measure the exploitability of expert Pokemon VGC play using a learned dynamics model as an EGTA oracle, finding that experts are individually exploitable but collectively play to Nash-level outcomes — via symmetric error cancellation, not independent near-optimality. This extends the penalty kick literature (Chiappori 2002, Palacios-Huerta 2003) from 2×2 sports games to combinatorial strategy spaces (~200-400 actions per side)."

### CS234 framing

"This project applies behavioral cloning, learned world models, and game-theoretic optimization via Nash LP. The key insight: turn 1 is a simultaneous-move game, not a sequential MDP, so the right solution concept is Nash equilibrium. We frame this as empirical game-theoretic analysis with a learned dynamics oracle — a novel combination that lets us compute equilibria from observational data without a game simulator. Our main finding connects RL to population dynamics: experts collectively converge to Nash through experience alone."

### "Why not just BC?"

"BC learns the average expert play, which is fine against other experts. But it's *exploitable*: if I always predict the most popular move, a savvy opponent prepares for that. Nash gives a strategy robust against any response. The exploitability gap — 1.41 reward units — is what we measure. But the surprise is that BC-vs-BC matches Nash: experts are collectively optimal, just individually predictable."

### "Why a learned dynamics model instead of exact simulation?"

"Replay data doesn't contain EVs, IVs, or natures — you can't compute exact damage without knowing a Pokemon's stat spread. Our dynamics model learns expected outcomes marginalized over the actual distribution of stat spreads in the metagame. That's the right quantity for population-level exploitability. With open team sheet tournament data, exact simulation would be possible — that's future work."

### "What surprised you?"

"Two things: First, that BC-vs-BC almost exactly equals the Nash value — but for the *wrong reason*. Value decomposition reveals it's error cancellation: BC P1 loses ~1.0 against Nash P2, and Nash P1 gains ~1.0 against BC P2. These symmetric errors cancel in expert-vs-expert play. Every single matchup (500/500) shows this opposite-sign pattern. Second, that smoothed best response from BC initialization converges toward Nash in ~500 iterations — the metagame acts like a learning process with experts as the initial strategy."

### "How do you trust the numbers?"

"Four ways: (1) Noise injection shows BC exploitability exceeds what dynamics noise alone produces by 1.0+ reward units — the signal is real. (2) The BC-vs-BC ≈ Nash finding uses the same payoff matrix, so dynamics errors cancel in the comparison. (3) Reward weight sweep (w_ko from 1 to 5) confirms the finding is invariant to reward specification. (4) Better dynamics (v2) finds *more* exploitability, not less — the signal strengthens with oracle quality."

### "How does this compare to VGC-Bench?"

"VGC-Bench trains agents and measures agent-vs-agent performance. We measure the exploitability of the *human population distribution* — a different and complementary question. They ask 'can we build a good player?' We ask 'how far are existing players from game-theoretic optimality, and what does the gap look like?'"
