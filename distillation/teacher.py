"""The teacher: BetaTetris' released network, queried for action distributions and values.

The teacher is the checkpoint published at

    https://github.com/BetaTetris/betatetris-tablebase

which this repo is a fork of.  It sees the same observation and emits the same 800
placements as the student, but it is *not* the same network: the teacher is the residual
tower this repo used to train, so it is built from RL/legacy_model.py rather than the
transformer in RL/model.py -- see download_teacher.py.  Two things are asked of it that a
plain forward pass does not give:

  * get_action_distribution -- the full softmax over the legal placements, at a temperature,
    rather than the argmax.  A single label per state throws away most of what the teacher
    knows; the relative weight it puts on the second- and third-best placements is the part
    that survives distillation into a network that will never search.
  * get_value -- a Monte Carlo estimate of the teacher's own return from a state, used to
    pre-train the value head so PPO does not start fine-tuning against a value function that
    predicts noise.
"""

import pathlib
import sys

# model.py lives in RL/ and only training_env is an installed package, so put that
# directory on sys.path before importing from it (same as tools/ and demo/ do)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'RL'))

import numpy as np
import torch
from torch.nn import functional as F

from legacy_model import load_model
from envs import Snapshot, TorchPolicy, rollout_returns

DEFAULT_CHECKPOINT = pathlib.Path(__file__).resolve().parent.parent / 'models' / 'model-v1.0.0-normal.pth'


class Teacher:
    """A frozen policy that can be asked for distributions and for values.

    Everything takes and returns batches: a single observation (the 5-tuple GetState gives)
    is accepted too and comes back without the batch axis.
    """

    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, device=None, temperature=1.0,
                 batch_size=512):
        checkpoint = pathlib.Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                '{} not found -- fetch it with:\n'
                '    python distillation/download_teacher.py'.format(checkpoint))
        self.model = load_model(checkpoint, device=device)
        self.device = next(self.model.parameters()).device
        self.temperature = temperature
        self.policy = TorchPolicy(self.model, device=self.device, batch_size=batch_size,
                                  temperature=temperature)
        self.checkpoint = checkpoint

    @staticmethod
    def _as_batch(obs):
        """(batched_obs, was_single) -- GetState returns unbatched (planes, h, w) arrays."""
        if obs[0].ndim == 3:
            return [i[None] for i in obs], True
        return obs, False

    def get_action_logits(self, obs):
        """(batch, 800) log-probabilities over placements, -inf where illegal.

        Normalized rather than raw so the numbers are bounded (they get stored as float16),
        which changes nothing downstream: softmax is invariant to a constant shift, so
        temperature scaling these gives the same distribution as scaling the raw logits.
        """
        obs, single = self._as_batch(obs)
        logits = self.policy.action_logits(obs)
        logits = F.log_softmax(logits, dim=1)
        out = logits.numpy()
        return out[0] if single else out

    def get_action_distribution(self, obs, temperature=None):
        """(batch, 800) probabilities over the legal placements; 0 on the illegal ones.

        Lower temperature sharpens toward the argmax, higher flattens toward uniform over
        what the move search allows.
        """
        obs, single = self._as_batch(obs)
        tau = self.temperature if temperature is None else temperature
        logits = self.policy.action_logits(obs) / tau
        probs = F.softmax(logits, dim=1).numpy()
        return probs[0] if single else probs

    def get_action(self, obs, deterministic=False):
        """Sampled (or argmax) placements, as flat action indices."""
        obs, single = self._as_batch(obs)
        actions = self.policy.act(obs, deterministic=deterministic)
        return int(actions[0]) if single else actions

    def get_value(self, snapshots, n_rollouts=5, horizon=50, gamma=0.999 ** 0.5,
                  deterministic=False, bootstrap=False, max_envs=2048, progress=False):
        """Mean discounted return of the teacher's own policy from each snapshot.

        n_rollouts games are played out from every snapshot, each for horizon pieces or until
        the game ends, and their returns are averaged.  The rollouts run in lockstep, so the
        work per snapshot is horizon batched forward passes spread over max_envs games at a
        time rather than one call per piece.
        """
        single = isinstance(snapshots, Snapshot)
        if single: snapshots = [snapshots]
        per_chunk = max(1, max_envs // max(1, n_rollouts))
        out = []
        for start in range(0, len(snapshots), per_chunk):
            chunk = snapshots[start:start + per_chunk]
            out.append(rollout_returns(self.policy, chunk, n_rollouts=n_rollouts,
                                       horizon=horizon, gamma=gamma,
                                       deterministic=deterministic, bootstrap=bootstrap))
            if progress:
                print('    values: {}/{} states'.format(min(start + per_chunk, len(snapshots)),
                                                        len(snapshots)), flush=True)
        values = np.concatenate(out) if out else np.zeros(0, dtype='float32')
        return float(values[0]) if single else values


def add_arguments(parser):
    parser.add_argument('--teacher', type=str, default=str(DEFAULT_CHECKPOINT),
                        help='BetaTetris checkpoint to distil from')
    parser.add_argument('--teacher-temperature', type=float, default=1.0,
                        help='temperature of the distribution the teacher is queried for')


if __name__ == '__main__':
    # sanity check: play one game with the teacher and print what it did
    import argparse
    from envs import Game, ParamSampler, stack_obs

    parser = argparse.ArgumentParser(description='Play one game with the teacher.')
    add_arguments(parser)
    ParamSampler.add_arguments(parser)
    parser.add_argument('--pieces', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    teacher = Teacher(args.teacher, temperature=args.teacher_temperature)
    print('teacher: {} ({:.2f}M parameters) on {}'.format(
        teacher.checkpoint.name, teacher.model.num_params() / 1e6, teacher.device))
    game = Game(seed=args.seed)
    game.reset(ParamSampler.from_args(args).sample(rng))
    while not game.over and game.run_pieces < args.pieces:
        game.step(teacher.get_action(game.obs(), deterministic=True))
    snap = game.snapshot()
    print('after {} pieces: {} lines, over={}'.format(game.run_pieces, game.run_lines, game.over))
    value = teacher.get_value(snap, n_rollouts=2, horizon=10, bootstrap=True)
    print('value from here: {:.4f} (2 rollouts x 10 pieces, bootstrapped), '
          'value head says {:.4f}'.format(value, teacher.policy.values(stack_obs([game.obs()]))[0]))
