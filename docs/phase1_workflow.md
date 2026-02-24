# Phase 1 Workflow: Autoregressive BC + Dynamics V2

Two independent model improvements that can run in parallel, followed by a joint evaluation.

---

## Overview

```
                    ┌─────────────────────┐
                    │    Current state     │
                    │  bc_001 (indep)      │
                    │  dyn_001 (concat)    │
                    │  results/ (baseline) │
                    └────────┬────────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
     ┌──────────▼──────────┐   ┌──────────▼──────────┐
     │  Track A: BC         │   │  Track B: Dynamics   │
     │                      │   │                      │
     │  1A. Train bc_002    │   │  1B. Write config    │
     │      (autoregressive)│   │      dynamics_v2.yaml│
     │      ~2h GPU         │   │                      │
     │                      │   │  2B. Add cross-attn  │
     │  2A. Quick sanity    │   │      to dynamics.py  │
     │      (val NLL, top1) │   │                      │
     │                      │   │  3B. Train dyn_002   │
     │  CPU-only after      │   │      ~2h GPU         │
     │  training completes  │   │                      │
     └──────────┬───────────┘   └──────────┬───────────┘
                │                          │
                └────────────┬─────────────┘
                             │
                  ┌──────────▼──────────┐
                  │  Track C: Evaluate   │
                  │                      │
                  │  1C. Full eval with  │
                  │      bc_002 + dyn_001│
                  │      (autoregressive)│
                  │                      │
                  │  2C. Full eval with  │
                  │      bc_002 + dyn_002│
                  │      (best combo)    │
                  │                      │
                  │  3C. Audit both      │
                  │                      │
                  │  4C. Compare v1/v2   │
                  │      dynamics ablation│
                  └──────────┬───────────┘
                             │
                  ┌──────────▼──────────┐
                  │  Track D: Write up   │
                  │  Update bible/pivot  │
                  │  with final numbers  │
                  └─────────────────────┘
```

---

## Track A: Autoregressive BC

**Goal**: Train BC with P(b|a) conditioning. Fix the independence factorization weakness.

**Dependencies**: None (code + config exist, just run it).

**GPU needed**: ~2 hours.

### Step 1A: Train

```bash
# Terminal 1
PYTHONPATH=. .venv/bin/python turnone/models/train.py \
    --config configs/bc_autoregressive.yaml \
    --out_dir runs/bc_002
```

This uses `autoregressive: true` in the config. Teacher-forcing during training: action_b conditioned on ground-truth action_a.

**Expected output**: `runs/bc_002/best.pt`, training curves, metadata.

**What to watch for**:
- Val loss should be similar or slightly better than bc_001 (NLL ~1.49)
- Top-1 accuracy should be similar (~44-46%)
- Mon B accuracy may improve (it now conditions on mon A's action)
- Convergence should be similar (~15-20 epochs)

### Step 2A: Quick sanity check

After training completes, verify the checkpoint is sane:

```bash
# Quick: just check the training metadata
python -c "
import torch, json
ckpt = torch.load('runs/bc_002/best.pt', map_location='cpu', weights_only=False)
meta = ckpt.get('metadata', {})
print(json.dumps(meta, indent=2))
print(f'Autoregressive: {ckpt.get(\"autoregressive\", False)}')
print(f'Best epoch: {ckpt.get(\"epoch\", \"?\")}')
"
```

**No GPU needed after this point for Track A.** The eval happens in Track C.

---

## Track B: Dynamics V2

**Goal**: Better dynamics model via (1) action cross-attention and (2) larger trunk.

**Dependencies**: Code changes to `dynamics.py` + new config. Can start coding immediately while Track A trains.

**GPU needed**: ~2 hours (after code changes).

### Step 1B: Create `configs/dynamics_v2.yaml`

```yaml
model:
  d_model: 128
  n_layers: 4
  n_heads: 4
  d_ff: 512
  dropout: 0.1
  d_action: 64          # was 32
  d_hidden: 512          # was 256
  n_mlp_layers: 4        # was 3
  action_cross_attn: true # NEW: cross-attention between action embeddings

training:
  seed: 42
  batch_size: 512
  max_epochs: 50
  patience: 10
  lr: 3.0e-4
  weight_decay: 0.01
  lambda_ko: 3.0
  lambda_field: 0.5
  require_both_actions: true
  num_workers: 4

data:
  split_dir: data/assembled
  vocab_path: null
```

### Step 2B: Add action cross-attention to `dynamics.py`

The current architecture concatenates 6 action embeddings (4 mon actions + 2 tera flags) with the pooled state. No interaction between actions.

**What to add**: A small Transformer (1-2 layers, 4 heads) over the 4 mon-action tokens before concatenating with state. This lets the model learn:
- "Fake Out + Tailwind" vs "Fake Out + Protect" (ally synergy)
- "our Fake Out vs their Protect" (opponent interaction)
- Speed-dependent sequencing effects

**Architecture change**:

```
Before (v1):
  pooled_state ⊕ [emb(a1), emb(a2), emb(a3), emb(a4), emb(tera1), emb(tera2)]
  → MLP trunk → heads

After (v2, action_cross_attn=true):
  action_tokens = [emb(a1), emb(a2), emb(a3), emb(a4)]
  action_tokens = TransformerEncoder(action_tokens, n_layers=1, n_heads=4)
  action_pooled = mean(action_tokens)
  pooled_state ⊕ action_pooled ⊕ [emb(tera1), emb(tera2)]
  → MLP trunk → heads
```

**Implementation notes**:
- Guard with `if action_cross_attn:` — v1 behavior preserved when flag is false
- `from_checkpoint()` must handle missing flag gracefully (default false)
- Trunk input dim changes: `d_model + d_action + 2*d_action` (state + pooled_actions + 2 tera) instead of `d_model + 6*d_action`
- Actually: keep it simple. Concatenate 4 action embeddings → 1-layer Transformer → pool to single vector. Then concat [state, action_pool, tera1, tera2] → MLP. Trunk input = `d_model + d_action + 2*d_action`.

**Tests to add** (in `tests/test_dynamics.py`):
- `test_cross_attn_output_shapes`: v2 model produces same output shapes as v1
- `test_cross_attn_gradients`: gradients flow through cross-attention
- `test_v1_v2_different_outputs`: with cross-attention, different action orderings produce different results (vs v1 where permuting actions should give different results due to position in concat but this isn't meaningful)

### Step 3B: Train

```bash
# Terminal 2 (after Track A finishes, or on a second GPU)
PYTHONPATH=. .venv/bin/python turnone/models/train_dynamics.py \
    --config configs/dynamics_v2.yaml \
    --out_dir runs/dyn_002
```

**What to watch for**:
- Val loss should be lower than dyn_001 (511.06)
- HP MAE should improve from 13.6 (target: <12)
- Train/val gap should be smaller (v1 showed capacity bottleneck)
- If val loss doesn't improve: the bottleneck is data/features, not capacity. Still useful to know.

---

## Track C: Evaluation

**Goal**: Full eval pipeline with improved models. Compare v1 vs v2.

**Dependencies**: Tracks A and B must complete first.

**GPU needed**: ~30 min per eval run (500 matchups).

### Step 1C: Eval with autoregressive BC + dynamics v1

This isolates the effect of fixing the independence factorization:

```bash
PYTHONPATH=. .venv/bin/python scripts/evaluate.py \
    --bc_ckpt runs/bc_002/best.pt \
    --dyn_ckpt runs/dyn_001/best.pt \
    --test_split data/assembled/test.jsonl \
    --vocab_path runs/bc_001/vocab.json \
    --out_dir results/autoreg_v1/ \
    --n_matchups 500 --seed 42 --autoregressive
```

**Key comparisons to baseline** (`results/`):
- Exploitability: should decrease (tighter bound, was 1.41)
- BC-vs-BC tracking: should improve (R² > 0.59, smaller gaps)
- If exploitability drops a lot → independence was inflating it significantly
- If exploitability barely changes → the correlation between mon A and B actions is weak

### Step 2C: Eval with autoregressive BC + dynamics v2

This is the best-case scenario:

```bash
PYTHONPATH=. .venv/bin/python scripts/evaluate.py \
    --bc_ckpt runs/bc_002/best.pt \
    --dyn_ckpt runs/dyn_002/best.pt \
    --test_split data/assembled/test.jsonl \
    --vocab_path runs/bc_001/vocab.json \
    --out_dir results/autoreg_v2/ \
    --n_matchups 500 --seed 42 --autoregressive
```

### Step 3C: Audit both

```bash
# Audit autoreg + dyn_v1
PYTHONPATH=. .venv/bin/python scripts/audit.py \
    --bc_ckpt runs/bc_002/best.pt \
    --dyn_ckpt runs/dyn_001/best.pt \
    --test_split data/assembled/test.jsonl \
    --vocab_path runs/bc_001/vocab.json \
    --matchup_details results/autoreg_v1/matchup_details.jsonl \
    --out_dir results/autoreg_v1/audit/

# Audit autoreg + dyn_v2
PYTHONPATH=. .venv/bin/python scripts/audit.py \
    --bc_ckpt runs/bc_002/best.pt \
    --dyn_ckpt runs/dyn_002/best.pt \
    --test_split data/assembled/test.jsonl \
    --vocab_path runs/bc_001/vocab.json \
    --matchup_details results/autoreg_v2/matchup_details.jsonl \
    --out_dir results/autoreg_v2/audit/
```

**Key audit checks to watch**:
- Check 1 (BC-vs-BC tracking): does autoregressive fix the FAIL?
- Check 11 (Nash action quality): should remain PASS
- All P0 checks should PASS

### Step 4C: Comparison table

Build a comparison table across all configurations:

| Config | BC | Dyn | Exploit | BC-vs-BC | Gap | Check 1 | Reward R² |
|--------|-----|-----|---------|----------|-----|---------|-----------|
| Baseline | indep | v1 | 1.41 | 0.20 | -0.02 | FAIL | 0.63 |
| Autoreg + v1 | autoreg | v1 | ? | ? | ? | ? | 0.63 |
| Autoreg + v2 | autoreg | v2 | ? | ? | ? | ? | ? |

This tells the story: "fixing the BC factorization does X, improving the dynamics does Y."

---

## Parallelization Strategy

### With 1 GPU (your setup)

```
Time    GPU                          CPU
────    ───                          ───
0h      Track A: train bc_002       Track B: write config + code changes
        (~2h)                        (~1-2h, no GPU needed)
                                     Run tests for dynamics v2
2h      Track B: train dyn_002      (wait)
        (~2h)
4h      Track C: eval autoreg+v1    (wait)
        (~30min)
4.5h    Track C: eval autoreg+v2    (wait)
        (~30min)
5h      Track C: audit both         Track D: start writeup
        (~5min each)
5.5h    Done                        Update bible with final numbers
```

**Total wall time: ~5.5 hours** (mostly GPU training).

### With Claude subagents

You can have me do this:

1. **Agent 1** (background): Train bc_002. Just runs the training command, reports when done.
2. **You/me** (foreground): Write dynamics v2 code changes + tests while bc_002 trains.
3. **Agent 2** (background): Train dyn_002 after code changes are done.
4. **Agent 3** (foreground): Run eval + audit after both training runs complete.

The key constraint is **GPU serialization** — only one training job at a time on a single GPU. The code changes for Track B are the only thing that benefits from interactive development.

### What can truly run in parallel

- **Track A training** ∥ **Track B code changes** (GPU + CPU)
- **Step 1C eval** and **Step 2C eval** are sequential (both need GPU)
- **Audits** can run in parallel (CPU-only after payoff matrices are built)
- **Writeup** can start as soon as Step 1C results are in

### What CANNOT run in parallel

- bc_002 training and dyn_002 training (both need GPU, will OOM if simultaneous)
- Eval runs (each needs ~4GB VRAM for dynamics inference)
- Track C depends on both Track A and Track B completing

---

## Decision Points

### After Step 1A (bc_002 training)

If bc_002 has **worse** val loss than bc_001:
- Check if autoregressive embedding is learning (inspect weights)
- May need to tune lr or label_smoothing
- Could be that the action correlation is genuinely weak

### After Step 3B (dyn_002 training)

If val loss doesn't improve over dyn_001:
- The bottleneck is data/features, not capacity
- Still run eval — same findings with v2 = robustness evidence
- Report as "dynamics quality is bounded by available features (no EVs/IVs)"

### After Step 1C (autoreg eval)

If exploitability barely changes:
- Independence factorization wasn't the main issue
- The exploitability number is already tight
- Simplifies the story: "even with correlated BC strategies, experts are exploitable"

If exploitability drops a lot (e.g., 1.41 → 0.8):
- Independence was inflating it significantly
- The tighter number is the real one
- Check 1 should improve dramatically
- Strengthens the "collective equilibrium" finding

### After Step 4C (comparison)

Pick the best model combination for final numbers. Update:
- `docs/PROJECT_BIBLE.md` Section 6.2, 6.3, 7.3
- `docs/why_pivot.md` Section 1
- `CLAUDE.md` Week 4 status

---

## Files Modified

### Track A (no code changes)
- **New**: `runs/bc_002/` (checkpoint, curves, metadata)

### Track B (code changes)
- **Modified**: `turnone/models/dynamics.py` (add cross-attention option)
- **Modified**: `turnone/models/train_dynamics.py` (pass new config flag)
- **New**: `configs/dynamics_v2.yaml`
- **Modified**: `tests/test_dynamics.py` (add v2 tests)

### Track C (no code changes)
- **New**: `results/autoreg_v1/`, `results/autoreg_v2/`

### Track D
- **Modified**: `docs/PROJECT_BIBLE.md`, `docs/why_pivot.md`, `CLAUDE.md`

---

## Success Criteria

1. bc_002 trains successfully with autoregressive flag, val metrics comparable to bc_001
2. dyn_002 trains with cross-attention, val loss ≤ dyn_001 (ideally lower)
3. Autoregressive eval produces tighter exploitability bound
4. Check 1 (BC-vs-BC tracking) improves with autoregressive factorization
5. All audit P0 checks PASS
6. 146+ tests passing (new dynamics v2 tests added)
7. Comparison table populated with all three configurations
