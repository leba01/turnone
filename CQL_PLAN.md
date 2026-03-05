# CQL Experiment Plan

## Context
TA feedback: project reads as game-theory + supervised learning, not RL.
Adding horizon-1 CQL gives a concrete RL comparison: BC (imitation) vs CQL (offline RL) vs Nash (game-theoretic optimum).

## Status

### DONE (laptop, commit 1d9247e)
- [x] Step 1: Added `reward` field to `Turn1Dataset.__getitem__` in `turnone/data/dataset.py`
- [x] Step 2: Created `turnone/models/cql_qvalue.py` — QLearner Q-network
- [x] Step 3: Created `turnone/rl/train_cql.py` — CQL training loop
- [x] Step 4: Created `configs/cql_base.yaml` — training config
- [x] Step 5: Created `scripts/cql_eval.py` — evaluation script
- [x] Step 6: Created `tests/test_cql.py` — 11 unit tests, all passing
- [x] All 162 existing tests still pass (no regressions)

### TODO (GPU machine)
- [ ] Step 7: Train CQL model
  ```bash
  python -m turnone.rl.train_cql --config configs/cql_base.yaml --out_dir runs/cql_001
  ```
  - Expected: ~50 epochs max, early stopping at patience=7
  - Watch: td_loss should decrease, cql_penalty should stabilize
  - If loss diverges: try alpha=0.5 or alpha=0.1 in config

- [ ] Step 8: Evaluate CQL vs BC vs Nash
  ```bash
  python scripts/cql_eval.py \
      --cql_ckpt runs/cql_001/best.pt \
      --bc_ckpt runs/bc_001/best.pt \
      --dyn_ckpt runs/dyn_001/best.pt \
      --test_split data/assembled/test.jsonl \
      --vocab_path runs/bc_001/vocab.json \
      --n_matchups 500 \
      --out_dir results/cql_eval/
  ```
  - Output: `results/cql_eval/cql_eval.json` with per-matchup + aggregate results
  - Prints comparison table: exploitability, TV distances, worst-case values

### TODO (laptop, after GPU results)
- [ ] Step 9: Update paper (`writeup/final.tex`)
  - Expand section 7 (Ablations) with CQL comparison subsection
  - Add table: BC vs CQL vs Nash (exploitability, TV distances, worst-case)
  - Add 1-2 paragraphs framing the comparison
  - Recompile PDF

## Key Design Decisions

**Factored Q-function:**
Q(s, slot_a, slot_b, tera) = Q_a(s, slot_a) + Q_b(s, slot_b) + Q_tera(s, tera)
- Justified by MI between leads = 0.018 bits (validated in paper)
- Same encoder backbone as BC (Turn1Encoder, 128-d, 4 layers)
- Shared Q-head for both leads (matching BC's shared action head)

**Horizon-1 (no bootstrapping):**
Q(s, a_me) = E[r | s, a_me] — averages over opponent distribution in data.
So CQL policy ~ best response to expert population.

**CQL loss:**
L = MSE(Q_data, reward) + alpha * [logsumexp_joint(Q) - Q(s, a_data)]
- Joint logsumexp = logsumexp_a + logsumexp_b + logsumexp_tera (exact for additive Q)
- Strategic masks exclude impossible self-targets

**Strategic honesty:**
If CQL converges to BC or doesn't beat it, that IS the result — it means
imitation and reward optimization are equivalent in low-rank games.

## Expected Results
- CQL exploitability: between BC (1.41) and Nash (0), likely 0.3-1.0
- If CQL ~ BC: "convention is free even under reward optimization"
- If CQL < BC: "reward signal recovers some Nash structure that imitation misses"

## File Map
| File | Purpose |
|------|---------|
| `turnone/models/cql_qvalue.py` | QLearner Q-network + extract_policy() |
| `turnone/rl/train_cql.py` | Training loop (CQL loss, early stopping) |
| `configs/cql_base.yaml` | Hyperparameters (alpha=1.0, same arch as BC) |
| `scripts/cql_eval.py` | Evaluation: exploitability, TV, cross-play |
| `tests/test_cql.py` | 11 unit tests |
| `turnone/data/dataset.py` | Modified: added reward to batch |
