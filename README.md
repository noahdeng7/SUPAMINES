# SUPAMINES

SUPAMINES is a NES Tetris agent trained in a C++ environment. It first distills a policy from BetaTetris then fine-tunes it with constrained PPO.

## Install

```sh
python supamines.py setup
```

This installs the Python dependencies, builds the environment, and installs it

## Use

```sh
python supamines.py teacher
python supamines.py distill
python supamines.py train my_run
python supamines.py evaluate models/distilled_checkpoint.pt
python supamines.py evaluate models/distilled_checkpoint.pt 1000
python supamines.py frontier plan
python supamines.py frontier run
python supamines.py frontier collect
python supamines.py play models/distilled_checkpoint.pt
```

`teacher` downloads the BetaTetris checkpoint. `distill` creates
`models/distilled_checkpoint.pt`. `train` starts PPO from that checkpoint when it exists.
`evaluate` plays 2000 games unless another count is supplied. `frontier` plans, runs, or
collects the pace-survival sweep. `play` starts the FCEUX model server.

## Repository

| Path | Purpose |
| --- | --- |
| `training_env/` | C++ Tetris environment and Python bindings |
| `distillation/` | BetaTetris teacher and DAgger distillation |
| `RL/` | models and constrained PPO training |
| `tools/` | evaluation, frontier sweeps, and ONNX export |
| `demo/` | FCEUX live-play server |
| `models/` | local checkpoints |
| `slurm/` | optional cluster jobs |

The training objective is score before 230 lines subject to a maximum topout probability.
