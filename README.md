# SUPAMINES

An NES Tetris agent: a C++ environment with exact frame-level move search, a constrained
PPO trainer, and a live-play bridge into TetrisGYM running in FCEUX.

## What this environment is for

Not to produce the strongest bot, but to **measure the pace-survival frontier**: how much
pre-killscreen score you have to trade for each unit of top-out risk.  The reward function
is deliberately minimal, because every additional shaping term biases the frontier being
measured.  [`NEW_DESIGN_SPEC.md`](NEW_DESIGN_SPEC.md) is the specification; this README says
where each part of it lives.

An **episode** is a level-18 start -- empty board, line counter 0 -- played to exactly one
of two terminals:

| terminal | trigger | cost |
| --- | --- | --- |
| `TOP_OUT` | the spawned piece collides at spawn | 1 |
| `MILESTONE` | the line counter reaches 230 (level-29 entry) | 0 |

Two channels come out of it, and nothing else:

```
r_t    = base(n_lines) * (level + 1) / 22800      # the NES score of the clear, over one
                                                  # level-18 tetris.  0 if nothing cleared.
cost_t = 1 on the transition into TOP_OUT
```

with `gamma_R = gamma_C = 1`.  The undiscounted return is therefore *exactly* the quantity
being measured, and variance is traded off with GAE `lambda` (0.95), which changes the
estimator rather than the objective.

The policy optimizes `A_t = A_R,t - beta * A_C,t` under a constraint
`P(top out before line 230) <= c`, with `beta` a Lagrange multiplier driven by dual ascent
(RL/train.py: `DualState`).  Sweeping `c` traces the frontier, and `beta` is its slope --
points per unit of top-out probability -- so the exchange rate comes free with each run:

```sh
python tools/frontier.py plan --name frontier      # the commands, one per c
python tools/frontier.py run  --name frontier      # ... or run them
python tools/frontier.py collect --name frontier   # frontier.csv + the monotonicity check
```

Everything that used to shape this reward -- step rewards, aggression levels, burn
penalties, phantom top-out probabilities, bottom-row bonuses -- was removed on purpose;
section 6 of the spec lists each one and why.  Pace, tetris rate and burn counts are
**reported, never rewarded**.

## Layout

| Directory | What it is |
| --- | --- |
| [`training_env/`](training_env) | The environment. C++ core + CPython extension + the Python rollout layer. The only installable package. |
| [`distillation/`](distillation) | **Step 1 — implemented.** DAgger distillation of the BetaTetris teacher into the four segment experts, with value targets from Monte Carlo rollouts. |
| [`RL/`](RL) | **Step 2 — implemented.** Constrained PPO self-play: the two-channel reward, the cost critic, and the Lagrangian dual. Also holds the policy network, the segment router, and the teacher's (legacy) architecture. |
| [`demo/`](demo) | Live play: serves a model to the FCEUX Lua script driving TetrisGYM. |
| [`tools/`](tools) | Headless evaluation, the frontier sweep driver, and ONNX export. |
| [`models/`](models) | Checkpoints — what each one is, and the frontier point measured so far. Weights themselves are not in the tree. `models/experts/segment_<18\|19\|29\|39>/` once step 1 has run. |
| [`slurm/`](slurm) | The batch scripts the runs were submitted with. Not required to use the repo. |

Only `training_env` is a Python package. `RL/`, `distillation/`, `demo/` and `tools/`
are script directories — run them **by path** (`python RL/train.py`) so that Python puts
the script's own directory on `sys.path` and their flat sibling imports resolve.
Because `model.py` lives in `RL/`, the three files outside `RL/` that need it
(`demo/fceux.py`, `tools/evaluate.py`, `tools/export_onnx.py`) each insert `RL/` onto
`sys.path` explicitly at the top of the file.

## Setup

```sh
pip install -r requirements.txt
make ext        # build the tetris extension in place
make install    # pip install -e .  -- required; see below
```

The editable install is not optional. Python 3.11+ does not put the working directory on
`sys.path` when you run a script by path, so `python RL/train.py` cannot find
`training_env` without it.

### Build-time variants

The environment's rules are compile-time constants, so a variant is a rebuild:

```sh
make ext TETRIS_DEFINES="LINE_CAP=290 TETRIS_ONLY"   # defaults to LINE_CAP=430
make ext TETRIS_ARCH=native                          # defaults to x86-64-v3
```

`x86-64-v3` (AVX2 + BMI1/2 + LZCNT + FMA) is the floor — the board is 4×`uint64` and the
move search is `pdep`/`pext`-heavy. Query the active variant at runtime with
`tetris.Tetris.LineCap()`, `.IsNoro()`, `.IsTetrisOnly()`; several modules branch on these.

`zstd` is optional at build time. Without it the supervised data reader is compiled out
(`setup.py` warns), which only matters for [distillation](distillation).

## Four segment experts

The agent is not one network. NES Tetris changes gravity at 130, 230 and 330 lines, which
splits a game into four regimes — level 18, 19-28, 29-38, 39-48 — and this repo trains one
**5.17M-parameter** transformer for each of them. An expert only ever sees its own segment,
at 30hz taps with an 18-frame adjustment reaction; every other tap speed, reaction time and
line range is *binned*, never sampled. Playing a whole game hands each state to the expert
that owns it, which takes no bookkeeping: the environment already writes the segment into
every observation (`move_meta[0:4]`), so [`RL/experts.py`](RL/experts.py) reads it off and
routes.

```sh
python distillation/download_teacher.py
python distillation/distill.py --out-dir models/experts     # all four
python RL/train.py expert_29 --segment 29 \
    --init-model-file models/experts/segment_29/distilled_checkpoint.pt
python tools/evaluate.py models/experts -n 100              # all four, routed
```

See [`training_env/README.md`](training_env/README.md) for the segment definitions and how
an episode is kept inside one, and [`distillation/README.md`](distillation/README.md) for
why the split is worth its cost.

## Pipeline

```
distillation/  ──►  RL/  ──►  tools/export_onnx.py  ──►  demo/
  (step 1)        (step 2)         (deploy)            (live play)
   4 experts      4 experts       one at a time        routed
                     │
                     ├──► tools/evaluate.py  (score a checkpoint on true level-18 starts)
                     └──► tools/frontier.py  (the sweep over c, and its readout)
```

Step 2 picks up step 1's checkpoint with `--init-model-file`, and can hold the fine-tuned
policy near it with a KL penalty (`--kl-distill-file`, annealed to 0 over the second half
of the run). The two are not strictly sequential either: `RL/train.py` can mix a supervised
cross-entropy term into the PPO objective via `--supervised-file` and
`--supervised-weight`, so distillation can also run concurrently with self-play.

Note that `--kl-distill-file` is **not** potential-based, so while it is on it biases the
frontier being measured. `--phi-model-file` is the shaping that does not: it adds
`F_t = gamma * Phi(s_{t+1}) - Phi(s_t)` with `Phi(terminal) = 0`, which is policy-invariant
for any `Phi`, and anneals it to zero. Prefer it, and report any run that used the KL
penalty as such.

Score any of the policies against each other with
`tools/evaluate.py --compare a.pth b.pth c.pth -n 100`; a directory of segment experts is
accepted anywhere a checkpoint is, except in `export_onnx.py`, where routing cannot be
traced into the graph.

## Where the implementation and the spec differ

One place, and it is worth knowing about because the spec asks you to check it.

**The level a clear is scored at.** Section 2 says line clears are scored at the level *in
effect before* the advance they cause. This environment scores them at the level *after* —
`core/game.h: GameScore(base_lines, lines) = ScoreFromLevel(GetLevelByLines(base_lines +
lines), lines)` — so the tetris that crosses 130 lines is worth `1200 * 20 = 24000` rather
than `1200 * 19 = 22800`. That is what the NES does and what a CTWC score readout shows,
and it is why `r_t` is taken from the environment's own score rather than recomputed from
the table: the undiscounted return then equals the reported pace *exactly*, which is the
whole reason for `gamma = 1`. The two conventions disagree by one tetris at each of the two
transitions inside an episode.

If you want the pre-advance convention, change `GameScore` — not the reward — so that the
return and the reported score stay the same number.

Everything else in the spec is implemented as written: the two terminals and their costs
(§1), `r_t` and `K = 22800` (§2), the sigmoid cost head with BCE (§3), the dual update with
its EMA and optional proportional term (§4), potential-based shaping with `Phi(terminal) =
0` and annealing (§5), the exclusions (§6), lines-so-far in the observation (§7), terminal
vs. truncation in the buffer (§8), and the sweep with its monotonicity check (§9). Pace is
total score at the milestone, the default of §10.

## History

This repo began as a fork of a tablebase-based agent. The tablebase pipeline
(`src/main.cpp`, `board_set`, `evaluate`, `sample_svd`, `prune`, `server`, `simulate`,
`inspect`, its CMake build and its gtest suite) was removed; only the ~19 core files the
environment actually compiles against were kept, under
[`training_env/core/`](training_env/core). Everything removed is recoverable from git
history at commit `f68460d`.
