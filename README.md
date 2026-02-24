# TurnOne

**TL;DR**: We figured out how exploitable top Pokemon players are on turn 1 — and discovered that expert play only *looks* optimal because everyone is bad in the same way.

## What is this?

In competitive Pokemon VGC (doubles), both players pick moves simultaneously on turn 1. This is a game theory problem: there's a mathematically optimal mixed strategy (Nash equilibrium), and we can measure how far real experts are from it.

## What did we find?

1. **Experts are exploitable.** If you knew exactly how the average top player picks moves, you could punish them for ~1.4 reward units per turn. They're predictable — they always go for the popular aggressive play.

2. **But expert-vs-expert looks optimal.** When two experts play each other, the outcome matches what Nash equilibrium predicts. So are experts secretly geniuses?

3. **No — it's a coincidence of symmetric errors.** We decomposed the expert-vs-expert value and found that Player 1 loses against a Nash opponent by ~1.0, and Player 2 gives away ~1.0 to a Nash opponent. These errors cancel perfectly. Every single matchup (500/500) shows this pattern. Experts aren't individually near-optimal — they're symmetrically suboptimal.

4. **Nash plays completely differently.** The optimal strategy uses ~3 actions with heavy Protect and defensive plays. Experts spread weight across ~95 actions and favor aggression. Total variation distance: 0.99 out of 1.0 (basically zero overlap).

5. **You can learn Nash from expert play.** Starting from expert strategies and iteratively best-responding, you converge toward Nash in ~500 steps — a 96% reduction in exploitability.

## How?

- **Behavioral cloning**: trained a neural net on 155K expert battles to learn "what do experts do?"
- **Dynamics model**: trained a world model to predict what happens when any two moves collide
- **Payoff matrices**: enumerated all ~200-400 valid action combos per side, scored each pair
- **Nash LP**: solved for the exact equilibrium via linear programming
- **Exploitability**: measured the gap

This is called **empirical game-theoretic analysis (EGTA)** — but instead of a game simulator, we use a learned dynamics model. That's new.

## Why is this cool?

There's a classic sports economics result: penalty kick shooters in soccer play Nash equilibrium (Chiappori 2002, Palacios-Huerta 2003). But that's a 2x2 game (left/right). We show the same phenomenon in a game with hundreds of actions per side — and we can decompose *why* it happens (error cancellation, not individual optimality). Nobody's done that before.

## Tech

Python 3.12, PyTorch, scipy (Nash LP), numpy. Runs on a single RTX 4080 Super. CS234 (Stanford RL) project, Winter 2026.
