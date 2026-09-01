# Reinforcement learning

This directory contains the policy and constrained PPO trainer.

## Train

Run training from this directory so labml stores runs in the expected location:

```sh
cd RL
python train.py ppo --init-model-file ../models/distilled_checkpoint.pt
```

From the repository root, the simpler equivalent is:

```sh
python supamines.py train ppo
```

The top-level command uses `models/distilled_checkpoint.pt` automatically when it exists.
Training logs and checkpoints are written under `RL/logs/<name>/`.

## Resume

```sh
cd RL
python train.py ppo last
```

Use `python RL/train.py --help` to list advanced training overrides.

## Files

| File | Purpose |
| --- | --- |
| `model.py` | policy and value network |
| `train.py` | PPO updates and dual optimization |
| `generator.py` | environment rollouts |
| `config.py` | training defaults and CLI overrides |
| `saver.py` | model, optimizer, and dual checkpoints |
| `legacy_model.py` | BetaTetris teacher architecture |

The policy maximizes score before 230 lines while constraining top-out probability. The
full objective and evaluation protocol are defined in `NEW_DESIGN_SPEC.md`.
