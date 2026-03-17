# TurnOne

**The Cost of Convention in Low-Rank Games**

Lucas Brennan-Almaraz · Stanford CS234 · Winter 2025--26

[[Paper (PDF)]](paper/turnone.pdf) · [[Poster (PDF)]](paper/turnone_poster.pdf) · ![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)

---

Expert Pokémon VGC strategies look nothing like Nash equilibria (total variation 0.99), yet expert-vs-expert outcomes match Nash payoffs within 0.02. Random strategies achieve the same result. SVD of the ~200-action payoff matrices reveals effective rank 3, with 93% of strategic variation in the payoff null space. Switching from imitation to offline RL (Conservative Q-Learning) reshapes 60% of the distribution but gains only 17% in exploitability. The question is not whether experts are rational, but whether the game notices.

**See also:** [TurnZero](https://github.com/leba01/turnzero) studies the pre-battle decision (which Pokémon to lead) across best-of-three sets. Same dataset, same format, earlier decision point --- and the same conclusion: convention, not optimization.

### Key Findings

1. **Convention ≠ strategy.** Experts are exploitable by 1.41 reward units per turn. They spread probability across ~95 actions; Nash concentrates on ~3. Total variation distance: 0.99.

2. **Convention is free.** Expert-vs-expert payoffs match Nash within 0.02 [−0.07, 0.04]. But this isn't expert skill --- uniform random and shuffled-label strategies achieve the same zero gap. It's structural.

3. **The game is 3-dimensional.** SVD of ~120×120 payoff matrices reveals effective rank 2.9. The top 3 singular values capture 96% of spectral energy. 93% of BC--Nash deviation energy lies in the payoff null space. There aren't enough payoff-relevant directions for experts to go wrong in different ways.

4. **Reward optimization can't escape.** CQL (offline RL) changes 60% of the distribution vs. BC, but 94% of that shift lands in the null space. Exploitability drops only 17% (1.40 → 1.15). CQL finds a *different convention*, not a better strategy.

5. **The exception: terastallization.** Tera is an irreversible resource. Our turn-1 model says use it 99% of the time; experts conserve it (25%). This accounts for 30% of exploitability. But in full battles (n = 45,172), early tera yields only a 1.7pp win-rate edge --- the immediate payoff is largely offset by the option value of waiting.

### Results

500 matchups, 95% bootstrap CIs (B = 1000).

| Metric | BC | CQL | Nash |
|:---|:---:|:---:|:---:|
| Exploitability | 1.40 [1.36, 1.46] | 1.15 [1.08, 1.23] | 0.00 |
| TV(·, Nash) | 0.99 | 0.98 | 0.00 |
| Worst-case value | −1.23 | −0.99 | +0.16 |

| Structural metric | Value | 95% CI |
|:---|:---:|:---:|
| Effective rank (95% energy) | 2.9 | [2.8, 2.9] |
| BC--Nash deviation in null space | 93% | [92.5%, 93.2%] |
| Cross-play gap (BC-vs-BC − V*) | −0.02 | [−0.07, 0.04] |
| Tera fraction of exploitability | 30% | [29%, 32%] |

### Method

We formalize turn-1 play as an offline contextual bandit: context is the game state (~200 valid action combinations per side), reward comes from a learned dynamics model.

- **Behavioral cloning**: Transformer encoder (4 layers, 128d) on 155K expert battles. 44.8% top-1 accuracy (3× uniform), factored action heads
- **Dynamics model**: same encoder, predicts HP changes + KOs + field state. HP MAE 13.3/100, KO AUC 0.91, signal-to-noise ratio 3.4:1
- **Payoff matrices**: enumerate all valid joint actions per matchup (~120×120), score via dynamics model, solve Nash LP
- **CQL**: Conservative Q-Learning on the same dataset. Factored Q-function, horizon-1 (no bootstrapping). Tests whether reward optimization can escape convention
- **Battle outcomes**: 124K full battles validate the tera finding against actual win rates

### Limitations

Everything flows through the learned dynamics model --- if it systematically compresses payoff structure, low rank could be partially artifactual. The capacity ablation (effective rank 2.9 → 4.3 with a larger model that overfits) cuts both ways. The framework is turn-1 only: tera's 30% contribution is the framework identifying its own blind spot. Turn-1 reward predicts only 4% of win variance (r = 0.21). All results are Regulation G; generality to other formats is a testable prediction, not a finding.

### Citation

```bibtex
@misc{brennan2026turnone,
  title={TurnOne: The Cost of Convention in Low-Rank Games},
  author={Brennan-Almaraz, Lucas},
  year={2026},
  note={Stanford CS234 Final Project, Winter 2025--26},
  url={https://github.com/leba01/turnone}
}
```

<details>
<summary><strong>Reproducing the results</strong></summary>

#### Setup

```bash
git clone https://github.com/leba01/turnone.git
cd turnone
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

#### Data

This project uses the [VGC-Bench dataset](https://huggingface.co/datasets/cameronangliss/vgc-battle-logs) (Angliss et al., AAMAS 2026). TurnOne uses the Regulation G BO3 subset. Place the raw logs in `data/raw/` before running the pipeline.

#### Pipeline

Each stage reads the previous stage's output. Run in order:

```bash
# 1. Parse Showdown protocol → directed match examples
python -m turnone.data.parser \
    --raw_path data/raw/logs-gen9vgc2025reggbo3.json \
    --out_dir data/parsed

# 2. Train behavioral cloning policy
python -m turnone.models.train \
    --config configs/bc_base.yaml

# 3. Train dynamics model
python -m turnone.models.train_dynamics \
    --config configs/dynamics_base.yaml

# 4. Build payoff matrices (BC + dynamics → per-matchup payoff tensors)
python -m turnone.game.payoff

# 5. Solve Nash equilibria via LP
python -m turnone.game.nash

# 6. Train CQL (offline RL comparison)
python -m turnone.rl.train_cql \
    --config configs/cql_base.yaml

# 7. Run analyses
python scripts/counterfactual_analysis.py   # value decomposition + counterfactual
python scripts/smoothed_br.py               # best-response convergence
python scripts/phase2_qre.py                # SVD + rank analysis
python scripts/reward_sensitivity.py        # reward ablation
python scripts/cql_eval.py                  # CQL vs BC vs Nash

# 8. Generate figures
python scripts/generate_figures.py
```

#### Tests

```bash
pytest -q
```

</details>

<details>
<summary><strong>Repository structure</strong></summary>

```
turnone/
├── turnone/                        # Main package
│   ├── data/                       # Data pipeline
│   │   ├── action_space.py         #   Valid action enumeration per team
│   │   ├── dataset.py              #   PyTorch Dataset
│   │   ├── parser.py               #   Showdown protocol extraction
│   │   └── io_utils.py             #   JSONL streaming I/O
│   │
│   ├── models/                     # Neural networks
│   │   ├── bc_policy.py            #   Behavioral cloning policy head
│   │   ├── dynamics.py             #   Learned dynamics (world model)
│   │   ├── encoder.py              #   Shared Pokémon encoder
│   │   ├── train.py                #   BC training loop
│   │   └── train_dynamics.py       #   Dynamics training loop
│   │
│   ├── game/                       # Game theory
│   │   ├── payoff.py               #   Payoff matrix construction
│   │   ├── nash.py                 #   Nash equilibrium LP solver
│   │   └── exploitability.py       #   Exploitability measurement
│   │
│   ├── eval/                       # Evaluation
│   │   ├── bootstrap.py            #   Bootstrap CIs (B=1000)
│   │   ├── dynamics_metrics.py     #   Dynamics model evaluation
│   │   └── metrics.py              #   General metrics
│   │
│   └── rl/                         # Offline RL
│       ├── reward.py               #   Reward function
│       └── train_cql.py            #   CQL training loop
│
├── scripts/                        # Analysis scripts
│   ├── generate_figures.py         #   Paper figures (3 figures)
│   ├── counterfactual_analysis.py  #   Value decomposition + counterfactual
│   ├── smoothed_br.py              #   Best-response convergence
│   ├── phase2_qre.py               #   SVD + rank analysis
│   ├── reward_sensitivity.py       #   Reward ablation
│   └── ...                         #   Additional analyses
│
├── configs/                        # Model configs (YAML)
├── tests/                          # Test suite
├── results/                        # Analysis outputs (JSON)
├── figures/                        # Generated figures (PDF)
├── paper/                          # Paper + poster PDFs
└── pyproject.toml
```

</details>

---

MIT
