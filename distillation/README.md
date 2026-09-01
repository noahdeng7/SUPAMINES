# Distillation

This stage trains one policy from the BetaTetris teacher with DAgger. The student learns
the teacher's action distribution and value estimates before PPO fine-tuning.

## Run

```sh
python distillation/download_teacher.py
python distillation/distill.py
```

The run writes intermediate checkpoints as `models/round_NN.pt`, the final model as
`models/distilled_checkpoint.pt`, and metrics as `models/metrics.jsonl`.

The defaults are 10 rounds, 50 games per round, 15% random-action noise, 4096 retained
states per round, and 4 training epochs per round. Run
`python distillation/distill.py --help` for advanced tuning options.

## Fine-tune

```sh
python RL/train.py ppo --init-model-file models/distilled_checkpoint.pt
```

The top-level command selects that checkpoint automatically:

```sh
python supamines.py train ppo
```

## Files

| File | Purpose |
| --- | --- |
| `download_teacher.py` | downloads the BetaTetris checkpoint |
| `teacher.py` | loads the teacher and estimates values |
| `envs.py` | runs games and snapshots states |
| `buffer.py` | stores DAgger samples |
| `distill.py` | trains and saves the student |
