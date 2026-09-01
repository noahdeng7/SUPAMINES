#!/usr/bin/env python3

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'RL'))

import numpy as np
import torch
from torch import optim
from torch.amp import GradScaler
from torch.distributions import Categorical
from torch.nn import functional as F

import teacher as teacher_mod
from buffer import DaggerBuffer
from envs import ParamSampler, TorchPolicy, collect_rollout_states, stack_obs
from model import MODEL_ARG_NAMES, Model, infer_model_args, load_model, save_model

def build_student(args, device):
    model_args = {name: getattr(args, name) for name in MODEL_ARG_NAMES}
    if args.init:
        loaded = infer_model_args(torch.load(args.init, weights_only=True, map_location='cpu'))
        if loaded != model_args:
            print('--init overrides the architecture flags: {}'.format(loaded))
        model = load_model(args.init, device=device, eval_mode=False)
    else:
        model = Model(**model_args).to(device)
    return model


def distillation_losses(student_logits, student_value, teacher_logits, teacher_value,
                        has_value, tau, value_weight, entropy_weight):

    mask = torch.isfinite(student_logits)
    zero = torch.zeros((), device=student_logits.device, dtype=student_logits.dtype)

    teacher_logp = F.log_softmax(torch.where(mask, teacher_logits, -torch.inf) / tau, dim=1)
    teacher_prob = torch.where(mask, teacher_logp.exp(), zero)
    student_logp = F.log_softmax(student_logits / tau, dim=1)
    kl = (teacher_prob * (torch.where(mask, teacher_logp, zero)
                          - torch.where(mask, student_logp, zero))).sum(dim=1).mean()

    loss_distill = kl * tau * tau

    if bool(has_value.any()):
        loss_value = F.mse_loss(student_value[has_value], teacher_value[has_value])
    else:
        loss_value = torch.zeros((), device=student_logits.device)

    entropy = Categorical(logits=student_logits).entropy().mean()
    loss = loss_distill + value_weight * loss_value - entropy_weight * entropy

    with torch.no_grad():
        accuracy = (student_logits.argmax(dim=1) == teacher_logits.argmax(dim=1)).float().mean()
    metrics = {
        'loss': loss.item(),
        'loss_distill': loss_distill.item(),
        'value_mse': loss_value.item(),
        'entropy': entropy.item(),
        'action_accuracy': accuracy.item(),
    }
    return loss, metrics


def label_round(teacher, states, args, rng):
    obs_batch = stack_obs([obs for obs, _ in states])
    start = time.time()
    teacher_logits = teacher.get_action_logits(obs_batch)
    policy_time = time.time() - start

    n = len(states)
    values = np.zeros(n, dtype='float32')
    has_value = np.zeros(n, dtype='bool')
    n_value = min(args.value_states, n) if args.value_states else n
    start = time.time()
    if n_value:
        picked = rng.choice(n, size=n_value, replace=False)
        snapshots = [states[i][1] for i in picked]
        values[picked] = teacher.get_value(
            snapshots, n_rollouts=args.value_rollouts, horizon=args.value_horizon,
            gamma=args.gamma, bootstrap=args.value_bootstrap, max_envs=args.value_max_envs,
            progress=args.progress)
        has_value[picked] = True
    return obs_batch, teacher_logits, values, has_value, policy_time, time.time() - start


def train_on_buffer(student, optimizer, scaler, buffer, args, rng, device):
    student.train()
    epoch_metrics = {}
    for epoch in range(args.epochs):
        totals, batches = {}, 0
        for batch in buffer.sample_batches(args.batch_size, rng, device=device):
            student_logits, value = student(batch['obs'])
            loss, metrics = distillation_losses(
                student_logits, value[0], batch['teacher_logits'], batch['value'],
                batch['has_value'], args.tau, args.value_weight, args.entropy_weight)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            for k, v in metrics.items(): totals[k] = totals.get(k, 0.) + v
            batches += 1
        epoch_metrics = {k: v / max(1, batches) for k, v in totals.items()}
        print('    epoch {}/{}: loss {:.4f} (distill {:.4f}, value mse {:.4f}), '
              'accuracy {:.3f}, entropy {:.3f}'.format(
                  epoch + 1, args.epochs, epoch_metrics['loss'], epoch_metrics['loss_distill'],
                  epoch_metrics['value_mse'], epoch_metrics['action_accuracy'],
                  epoch_metrics['entropy']), flush=True)
    student.eval()
    return epoch_metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    teacher_mod.add_arguments(parser)
    ParamSampler.add_arguments(parser)

    group = parser.add_argument_group('student architecture (model.Model, ~5.2M parameters)')
    group.add_argument('--d-model', type=int, default=256)
    group.add_argument('--layers', type=int, default=6)
    group.add_argument('--heads', type=int, default=8)
    group.add_argument('--mlp-ratio', type=float, default=4.)
    group.add_argument('--init', type=str, help='warm-start from a checkpoint')

    group = parser.add_argument_group('DAgger')
    group.add_argument('--dagger-rounds', type=int, default=10)
    group.add_argument('--games-per-round', type=int, default=50)
    group.add_argument('--noise-prob', type=float, default=0.15,
                       help='fraction of rollout steps that take a random legal placement')
    group.add_argument('--max-pieces', type=int, default=300,
                       help='cut a rollout game off after this many pieces')
    group.add_argument('--states-per-round', type=int, default=4096,
                       help='states kept per round (uniform sample); 0 keeps every one')
    group.add_argument('--buffer-capacity', type=int, default=0,
                       help='cap on buffered states, dropping whole old rounds; 0 = unlimited')

    group = parser.add_argument_group('teacher value targets')
    group.add_argument('--value-states', type=int, default=512,
                       help='states per round that get a Monte Carlo value target; 0 = all of them')
    group.add_argument('--value-rollouts', type=int, default=5, help='rollouts averaged per state')
    group.add_argument('--value-horizon', type=int, default=50, help='pieces per rollout')
    group.add_argument('--value-max-envs', type=int, default=2048,
                       help='games stepped at once while estimating values')
    group.add_argument('--value-bootstrap', action=argparse.BooleanOptionalAction, default=True,
                       help="add the teacher's own value estimate at the rollout horizon "
                            'instead of truncating the return there.  Required in practice '
                            'now that gamma is 1: an undiscounted 50-piece return is "score '
                            'in the next 50 pieces", not the value')
    group.add_argument('--gamma', type=float, default=1.0,
                       help='discount; 1.0 matches RL/config.py -- see NEW_DESIGN_SPEC.md '
                            'section 2 -- so the value head transfers to PPO')

    group = parser.add_argument_group('optimisation')
    group.add_argument('--epochs', type=int, default=4, help='passes over the buffer per round')
    group.add_argument('--batch-size', type=int, default=256)
    group.add_argument('--lr', type=float, default=3e-4)
    group.add_argument('--tau', type=float, default=3.0, help='distillation temperature')
    group.add_argument('--value-weight', type=float, default=0.5)
    group.add_argument('--entropy-weight', type=float, default=0.01)
    group.add_argument('--max-grad-norm', type=float, default=1.0, help='0 disables clipping')
    group.add_argument('--rollout-batch-size', type=int, default=512,
                       help='observations per forward pass during rollouts')

    parser.add_argument('--out-dir', type=str, default='models')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--progress', action='store_true', help='print progress within a round')
    return parser.parse_args(argv)


def distill(args, teacher, device, rng):
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'metrics.jsonl'

    student = build_student(args, device)
    student.eval()
    student_policy = TorchPolicy(student, device=device, batch_size=args.rollout_batch_size)
    param_sampler = ParamSampler.from_args(args)
    optimizer = optim.Adam(student.parameters(), lr=args.lr)
    scaler = GradScaler('cuda', enabled=(device.type == 'cuda'))
    buffer = DaggerBuffer(capacity=args.buffer_capacity)

    print('\n' + '=' * 78)
    print('student       : {:.2f}M parameters -- {}'.format(
        student.num_params() / 1e6,
        ', '.join('{}={}'.format(k, getattr(args, k)) for k in MODEL_ARG_NAMES)))
    print('rollouts      : {}'.format(param_sampler.describe()))
    print('output        : {}'.format(out_dir))
    print('=' * 78, flush=True)

    for dagger_round in range(args.dagger_rounds):
        started = time.time()
        # round 0 has nothing to roll out but a random student, so the teacher drives it
        from_teacher = dagger_round == 0 and not args.init
        policy = teacher.policy if from_teacher else student_policy
        print('\n=== DAgger round {}/{} (rollouts from the {}) ==='.format(
            dagger_round + 1, args.dagger_rounds, 'teacher' if from_teacher else 'student'),
            flush=True)

        states, stats = collect_rollout_states(
            policy, args.games_per_round, param_sampler, rng, noise_prob=args.noise_prob,
            max_pieces=args.max_pieces, max_states=args.states_per_round or None,
            progress=args.progress)
        print('  {} games: mean length {:.1f} pieces (median {:.0f}, max {}), '
              'mean {:.1f} lines, {} topped out, {} reached the milestone, '
              '{} hit the piece cap'.format(
                  stats['games'], stats['mean_game_length'], stats['median_game_length'],
                  stats['max_game_length'], stats['mean_lines'], stats['topped_out'],
                  stats['reached_milestone'], stats['truncated']),
              flush=True)
        print('  labelling {} of {} states seen'.format(
            len(states), stats['states_seen']), flush=True)

        obs_batch, teacher_logits, values, has_value, policy_time, value_time = label_round(
            teacher, states, args, rng)
        buffer.add(obs_batch, teacher_logits, values, has_value)
        print('  teacher: {} distributions in {:.1f}s, {} values in {:.1f}s '
              '(mean value {:.4f})'.format(len(states), policy_time, int(has_value.sum()),
                                           value_time, float(values[has_value].mean())
                                           if has_value.any() else 0.), flush=True)
        print('  buffer: {} states over {} rounds ({:.2f} GB)'.format(
            len(buffer), buffer.rounds, buffer.nbytes() / 1e9), flush=True)

        metrics = train_on_buffer(student, optimizer, scaler, buffer, args, rng, device)

        checkpoint = save_model(student, out_dir / 'round_{:02d}.pt'.format(dagger_round))
        record = {'round': dagger_round, 'seconds': time.time() - started,
                  'buffer_states': len(buffer), 'checkpoint': str(checkpoint),
                  'rollout_policy': 'teacher' if from_teacher else 'student',
                  **{'rollout_' + k: v for k, v in stats.items()}, **metrics}
        with open(log_path, 'a') as f:
            f.write(json.dumps(record) + '\n')
        print('  saved {} ({:.0f}s for the round)'.format(checkpoint, record['seconds']),
              flush=True)

    final = save_model(student, out_dir / 'distilled_checkpoint.pt')
    print('\nwrote {}'.format(final))
    print('metrics log: {}'.format(log_path))
    return final


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device if args.device else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.distributions.Distribution.set_default_validate_args(False)

    teacher = teacher_mod.Teacher(args.teacher, device=device,
                                  temperature=args.teacher_temperature,
                                  batch_size=args.rollout_batch_size)

    print('device        : {}'.format(device))
    print('teacher       : {} ({:.2f}M parameters)'.format(teacher.checkpoint.name,
                                                           teacher.model.num_params() / 1e6))
    print('rounds        : {} x {} games, {} noise, keeping {} states/round'.format(
        args.dagger_rounds, args.games_per_round, args.noise_prob,
        args.states_per_round or 'all'))
    print('value targets : {} states/round, {} rollouts x {} pieces{}'.format(
        args.value_states or 'all', args.value_rollouts, args.value_horizon,
        ', bootstrapped' if args.value_bootstrap else ''))

    checkpoint = distill(args, teacher, device, rng)

    flags = ' '.join('--{} {}'.format(k.replace('_', '-'), getattr(args, k))
                     for k in MODEL_ARG_NAMES)
    print('\n' + '=' * 78)
    print('checkpoint: {}'.format(checkpoint))
    print('\nPPO fine-tuning:')
    print('  python RL/train.py <name> {} --init-model-file {}'.format(flags, checkpoint))
    return checkpoint


if __name__ == '__main__':
    main()
