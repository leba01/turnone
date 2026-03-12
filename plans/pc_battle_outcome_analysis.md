# PC Task: Battle Outcome Analysis

## Context

Read this plan fully, then execute it. This is a self-contained task for Claude on the PC (which has GPU + raw battle logs). The laptop will pull results afterward for paper integration.

Two post-presentation additions for the paper:

1. **Tera-first win rate**: Our Nash policy terastalizes turn 1 ~99% of the time. We suspect this is myopic (horizon-1 can't value "saving" tera). To probe this: compute win rates conditioned on whether a player terastalizes on turn 1.
2. **Turn 1 ≠ battle outcome**: Show that a strong turn 1 doesn't predict winning the battle. This justifies our reward model using turn-1 HP/KO/field signals rather than battle outcome.

## Step 0: Find the raw data

The raw battle logs are a JSON file somewhere on this machine. It was used by `turnone/data/parser.py:run_parse()` which expects:
```python
with open(raw_path) as f:
    data = json.load(f)  # {battle_id: [timestamp, log_text], ...}
```

Search for it. Likely patterns:
- `logs-*.json` or `*battles*.json`
- Could be in `~/data/`, `~/Downloads/`, `~/Desktop/`, or anywhere in the project tree
- File will be large (tens of thousands of battles)
- The format is VGC 2025 Regulation H (`gen9vgc2025regh` or similar)

```bash
find ~ -maxdepth 4 -name "logs-*.json" -size +10M 2>/dev/null
find ~ -maxdepth 4 -name "*battle*" -name "*.json" -size +10M 2>/dev/null
```

If you can't find it, check git log for any references to the raw path, or search shell history (`~/.zsh_history` or `~/.bash_history`) for `run_parse` or the filename.

## Step 1: Extract battle outcomes

Create `scripts/battle_outcomes.py`. This script reads the raw JSON and extracts per-battle:

### What to parse from each battle log

The log is a newline-delimited Showdown protocol log. Key tags:

- `|win|<player_name>` — who won (appears near end of log)
- `|-terastallize|<position>|<type>` — tera events. Position is like `p1a: Nickname` or `p2b: Nickname`. Extract the `p1`/`p2` prefix.
- `|turn|<N>` — turn markers. Track which turn tera happens on.
- `|player|p1|<name>|...` and `|player|p2|<name>|...` — map player names to p1/p2

### Output schema

Write to `results/battle_outcomes/outcomes.jsonl`, one line per battle:

```json
{
  "battle_id": "battle-gen9vgc2025regh-12345",
  "winner": "p1",           // or "p2" or null (forfeit/tie)
  "p1_tera_turn": 3,        // turn p1 first terastallized, null if never
  "p2_tera_turn": 1,        // turn p2 first terastallized, null if never
  "total_turns": 8,
  "p1_name": "PlayerOne",
  "p2_name": "PlayerTwo"
}
```

### Implementation notes

- Parse ALL battles, not just the ones that pass the turn-1 filter. We want the broadest sample.
- Skip battles with no `|win|` tag (these are forfeits/disconnects — log them but don't include).
- A player might never terastallize — that's fine, record `null`.
- Track tera turn by counting `|turn|` tags. Turn 1 actions happen before the first `|turn|2` marker, so tera before `|turn|2` = turn 1.
- Print summary stats: total battles, battles with winner, tera usage rates.

## Step 2: Compute turn-1 reward per battle

We need to join battle outcomes with turn-1 quality. The existing parser (`turnone/data/parser.py:parse_turn1`) already computes HP deltas, KO flags, and field changes. The reward function is in `turnone/rl/reward.py:compute_reward()`.

Create `scripts/turn1_vs_outcome.py` that:

1. Loads the raw JSON (same file as Step 1)
2. For each battle that has a winner:
   - Try `parse_turn1()` to get turn-1 resolution data
   - Compute turn-1 reward using `compute_reward()`
   - Join with the battle outcome from Step 1's output
3. Save to `results/battle_outcomes/turn1_vs_outcome.jsonl`:

```json
{
  "battle_id": "...",
  "winner": "p1",
  "turn1_reward_p1": 0.35,
  "p1_tera_turn1": true,
  "p2_tera_turn1": false,
  "total_turns": 8
}
```

### Computing turn-1 reward

The parser returns a dict with a `resolution` key containing HP deltas, KO flags, and field state. Use the existing code path:

```python
from turnone.data.parser import parse_turn1, SkipBattle
from turnone.rl.reward import compute_reward
import numpy as np

parsed = parse_turn1(log_text)
res = parsed["resolution"]
# resolution has: hp_delta (4,), ko_flags (4,), field_before (5,), field_after (5,)
# These are from P1's perspective
reward_p1 = compute_reward(
    hp_delta=np.array([res["hp_delta"]]),
    ko_flags=np.array([res["ko_flags"]]),
    field_before=np.array([res["field_before"]]),
    field_after=np.array([res["field_after"]]),
)
```

Check the actual dict keys — they may differ slightly. Read `parser.py`'s resolution construction to confirm. The reward is from P1's perspective; P2's reward = -P1's.

## Step 3: Compute summary statistics

Create `scripts/outcome_analysis.py` that loads both JSONL files and computes:

### Analysis A: Tera-first win rate

From `outcomes.jsonl`:
- **Overall tera turn-1 rate**: What fraction of players terastallize on turn 1?
- **Win rate by tera timing**:
  - Player teras turn 1 → win rate
  - Player teras later (turn 2+) → win rate
  - Player never teras → win rate
- **Head-to-head**: When one player teras turn 1 and the other doesn't, who wins more?
- **Both tera turn 1**: Win rate when both players tera turn 1

Use 95% confidence intervals (Wilson score or normal approximation — doesn't matter, sample is large).

### Analysis B: Turn 1 ≠ battle outcome

From `turn1_vs_outcome.jsonl`:
- **Point-biserial correlation** between turn-1 reward and winning (for P1)
- **Binned analysis**: Bucket turn-1 reward into quintiles, compute win rate per bucket
- **Key stat to report**: "A player in the top quintile of turn-1 reward wins only X% of the time" (we expect this to be well below 100%, probably 55-65%)

### Output

Save all summary stats to `results/battle_outcomes/summary.json`:

```json
{
  "n_battles_total": ...,
  "n_battles_with_winner": ...,
  "n_battles_with_turn1_data": ...,

  "tera_analysis": {
    "overall_tera_turn1_rate": ...,
    "win_rate_tera_turn1": {"mean": ..., "ci_low": ..., "ci_high": ..., "n": ...},
    "win_rate_tera_later": {"mean": ..., "ci_low": ..., "ci_high": ..., "n": ...},
    "win_rate_no_tera": {"mean": ..., "ci_low": ..., "ci_high": ..., "n": ...},
    "head_to_head_tera_first_vs_not": {"tera_first_wins": ..., "n": ...},
    "both_tera_turn1_p1_win_rate": ...
  },

  "turn1_vs_outcome": {
    "point_biserial_r": ...,
    "point_biserial_p": ...,
    "quintile_win_rates": [
      {"quintile": 1, "reward_range": [-2.0, -0.5], "win_rate": ..., "n": ...},
      ...
    ],
    "top_quintile_win_rate": ...,
    "bottom_quintile_win_rate": ...
  }
}
```

Print a human-readable summary to stdout as well.

## Step 4: Commit and push

After all scripts run successfully and results are saved:

```bash
git add scripts/battle_outcomes.py scripts/turn1_vs_outcome.py scripts/outcome_analysis.py
git add results/battle_outcomes/
git commit -m "feat: add battle outcome analysis (tera win rate + turn1 vs outcome)"
git push
```

## Verification

Before committing, sanity-check:
- Tera turn-1 rate should be substantial (VGC players do tera a lot) — probably 30-60%
- Point-biserial correlation should be positive but modest (0.1-0.3 range)
- Top quintile win rate should be notably less than 100% — we expect ~55-65%
- If any numbers look crazy, investigate before committing

## What NOT to do

- Don't modify the paper (`.tex` files) — that happens on the laptop
- Don't generate figures — that also happens on the laptop
- Don't run GPU training — this is pure data analysis, CPU only
- Don't modify existing scripts or parser code
