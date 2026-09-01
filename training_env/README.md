# training_env

The NES Tetris environment used for both rollouts and live play. This is the only
installable package in the repo (`pip install -e .` from the repo root).

```
core/     C++ core: board representation, move search, frame sequencing
tetris/   CPython extension wrapping core/ (module name: training_env.tetris)
tools/    standalone C++ utilities (not built by setup.py)
game.py       BatchedGames -- steps every env in lockstep
game_param.py per-episode parameter sampling, the tap/adj-delay tables, level segments
```

## `training_env.tetris`

The compiled extension. Built by the repo-root `setup.py` as
`training_env.tetris.tetris`; the last component must stay `tetris` because the C init
symbol is `PyInit_tetris`. `training_env/tetris/__init__.py` re-exports it, so
`from training_env import tetris` gives you the flat namespace.

Exposed types:

- `Tetris` — a single game. `Reset(...)`, `InputPlacement(...)`, `GetState()`,
  `GetAdjStates(...)`. Class methods `StateShapes()`, `StateTypes()`, `LineCap()`,
  `IsNoro()`, `IsTetrisOnly()` report the compile-time variant.

  `InputPlacement` returns `(reward, cost)` — the two channels of NEW_DESIGN_SPEC.md
  §2 and §3, and nothing else. `IsEpisodeOver()` / `IsTopOut()` / `ReachedMilestone()`
  ask which of the two terminals an episode hit; `GetRunTetrises()` is there for the
  tetris rate and burn count the protocol reports.
- `Batch` — many games stepped by an internal C++ thread pool, writing observations
  directly into caller-owned numpy buffers.
- `Board` — the board representation, constructible from a `.`/`x` string.
- `SupervisedDataReader` — reads zstd `SupervisedData` datasets. **Compiled out if
  `zstd.h` is absent at build time.** Used by [distillation](../distillation) and by
  `RL/train.py`'s supervised loss term.

Observation is a 5-tuple: `(board, board_meta, moves, move_meta, meta_int)`. For the
default `LINE_CAP=430` variant the shapes are
`((6,20,10), (32,), (18,20,10), (28,), (2,))` with the two plane tensors as `uint8`;
consumers widen them to float on device. Do not hardcode these — read
`Tetris.StateShapes()` / `StateTypes()`, since they change with the variant.

## `BatchedGames` (game.py)

Owns no observation memory. The caller allocates a
`(worker_steps + 1, envs, *shape)` buffer per observation component and hands it over:
step `t` reads slot `t` and writes its result into slot `t + 1`, with slot `worker_steps`
holding the bootstrap observation. So a rollout costs no per-env Python work, no
allocation and no copies. `RL/generator.py` allocates these as shared memory, which is
why passing a rollout to the trainer is a mapping rather than a copy.

```python
import numpy as np
from training_env import tetris, BatchedGames

shapes, types = tetris.Tetris.StateShapes(), tetris.Tetris.StateTypes()
obs = [np.zeros((steps + 1, envs, *s), dtype=t) for s, t in zip(shapes, types)]

games = BatchedGames(envs=envs, worker_steps=steps, obs=obs, num_threads=0, seed=1234,
                     milestone=230)                     # the episode the spec defines
games.set_params([mid_ratio, board_ratio, short_ratio])
games.reset_all()
for t in range(steps):
    games.step(t, actions, info)   # resets finished episodes in place
```

Two per-step channels come back beside the observations:

```
games.rewards[env, t] = (r_t, cost_t)       # NEW_DESIGN_SPEC.md sections 2 and 3
games.is_over[env, t] = (done, truncated)   # section 8
```

`r_t` is the NES score of the clear over `K = 22800`, so the undiscounted sum over an
episode is exactly `RunScore() / 22800`. `cost_t` is 1 on the transition into `TOP_OUT` and
0 everywhere else, including at the milestone. `done` says the episode ended and the env
was reset; `truncated` distinguishes a *cut* — where both value heads must bootstrap off
the state that follows — from the two real terminals, where they must not.

`step()` returns `(true_start_finished, true_start_topouts)` for that step: the denominator
and numerator of the batch estimate of `p_hat = P(top out before the milestone)`, which the
dual update in `RL/train.py` consumes. Only true level-18 starts are counted, so `c` stays a
constraint on the episode the spec defines even when the training reset distribution is
something else. Everything folded into `info` is measured on those episodes too.

`eval_ratio` reserves the first envs for true level-18 starts whatever the training reset
distribution is doing. Raise it alongside `mid_ratio` / `board_ratio`.

## `GameParamManager` (game_param.py)

Samples the per-episode parameters — tap speed, adjustment delay, starting lines, and
optionally a starting board drawn from a `--board-file`. It keeps a running count over the
(tap, adj-delay, line-bucket) grid and samples inversely to it, so training coverage stays
even across the parameter space rather than collapsing onto whatever is easiest.

The default is the spec's episode and nothing else: `mid_ratio` and `board_ratio` are both
0, so every draw is a **true start** — level 18, empty board, line counter 0. Randomized
mid-game resets and curriculum boards are allowed as training aids, but they change the
state distribution and therefore what `p_hat` means, so every draw is stamped with
`is_true_start` and the constraint is measured on those alone.

`segment`, `tap_ids` and `adj_delay_ids` restrict what may be sampled at all. Anything
outside them is **binned** — given infinite count, so the sampler never returns it — rather
than down-weighted, which is what makes a segment expert's training distribution exactly
the slice of the game it will be asked to play.

There is no aggression level any more. It was a knob on the old shaped reward, which the
spec removed (§6); `meta[28]` keeps its slot in the observation, pinned to 1, so that every
checkpoint distilled against the old layout still loads.

## The episode

`MILESTONE_LINES` (230, level-29 entry) ends an episode at no cost; a top-out ends it at
cost 1. Those are the only two terminals. The milestone is a *runtime* setting on each env
(`PythonTetris::SetMilestone`, `ResetEnv(..., milestone=N)`) rather than a build constant,
so a different milestone is a flag and not a rebuild — and so live play, which runs whole
games to the build's line cap, can turn it off with `milestone=0`.

`Tetris::IsOver()` keeps its old meaning: a genuine game over, or the build's `kLineCap`.
The episode's end is `IsEpisodeOver()`, and the milestone wins ties — a placement that
clears to exactly 230 *and* leaves a board the next piece cannot spawn into is a milestone,
because the spawn that would have collided is past the end of the episode and never
happens.

## Level segments

Gravity changes at 130 / 230 / 330 lines (`core/game.h`: `kLevelSpeedLines`,
`GetLevelSpeed`), splitting a game into four regimes:

| segment | levels | lines | frames per row |
| --- | --- | --- | --- |
| `18` | 18 | 0-129 | 3 |
| `19` | 19-28 | 130-229 | 2 |
| `29` | 29-38 | 230-329 | 1 |
| `39` | 39-48 | 330-429 | 1, with a faster entry delay |

They are the unit this repo trains on: one ~5.2M-parameter expert per segment
(see [`../distillation`](../distillation)). Named by their starting level, and normalized
by `segment_id`, which accepts either the name or the index. `segment_lines`,
`segment_of_lines`, `segment_buckets` and `segment_label` are the rest of the API; a build
with a shorter `LINE_CAP` simply has fewer segments, and `NUM_SEGMENTS` is the truth.

Nothing has to carry a segment id next to an observation: `move_meta[0:4]` is the
environment's own `LevelSpeed` one-hot, so `argmax` over it *is* the segment. That is how
`RL/experts.py` routes a batch of states to the right expert.

### Keeping an episode inside its segment

`ResetEnv(..., line_cap=N)` cuts an episode when its absolute line count reaches `N`.
`GameParamManager` sets it to the end of the pinned segment, so an expert's games start
inside its line range and end when they clear past it. (A segment that runs into the
milestone needs no cut — the milestone ends it first — so `line_cap` is dropped there.)

The cut is a **truncation**, not a terminal, and the same mechanism short curriculum games
use: it flags the *next* step, so the trainer has a real successor state to bootstrap both
value heads from before the env is reset, and then drops that step (`generator.py`'s
`skip_mask`) rather than charging the policy for a top-out that did not happen. A truncated
episode is reported as neither `is_topout` nor `is_milestone`, carries no cost, and is
excluded from `p_hat`. `BatchedGames.total_truncated` counts them.

## `tools/random_boards.cpp`

Kept from the deleted tablebase tooling because it is an environment utility, not a
tablebase one: it reads a flat `CompactBoard` file and samples a cell-count-stratified
subset, producing the board files that `GameParamManager` consumes via `--board-file`.
It is not part of the extension build — see `make random_boards`.
