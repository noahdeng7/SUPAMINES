"""The DAgger replay buffer: every round's labelled states, kept for the whole run.

DAgger's correctness argument rests on training against the union of the rounds rather than
the newest one -- a student trained only on the states its current policy visits will forget
the recoveries it learned from the messier boards of two rounds ago and oscillate.  So this
buffer only ever grows, and a round is stored as its own block of contiguous numpy arrays
(no per-state Python objects, no re-allocation of what is already there).

What is stored per state:
  obs             the five observation arrays, at the dtypes the environment produces them
  teacher_logits  the teacher's log-probabilities over the 800 placements, float16, -inf
                  where the placement is illegal
  value           the teacher's Monte Carlo value estimate, float32
  has_value       whether that value was actually computed -- a value target costs a few
                  hundred forward passes against one for a policy target, so only a subset
                  of each round's states gets one, and the value loss skips the rest
"""

import numpy as np
import torch

from training_env import tetris
from envs import NUM_ACTIONS

STATE_SHAPES = tetris.Tetris.StateShapes()
STATE_TYPES = tetris.Tetris.StateTypes()


class DaggerBuffer:
    """Append-only store of (obs, teacher distribution, teacher value) triples."""

    def __init__(self, capacity=0):
        self.capacity = capacity          # 0 = keep everything
        self.blocks = []                  # one dict per DAgger round
        self.index = np.zeros((0, 2), dtype='int64')   # (block, offset) per state

    def __len__(self):
        return len(self.index)

    @property
    def rounds(self):
        return len(self.blocks)

    def add(self, obs_batch, teacher_logits, values, has_value):
        """Append one round.  obs_batch is a list of stacked arrays, as stack_obs returns."""
        n = len(obs_batch[0])
        assert teacher_logits.shape == (n, NUM_ACTIONS)
        block = {
            'obs': [np.ascontiguousarray(i) for i in obs_batch],
            'teacher_logits': np.ascontiguousarray(teacher_logits, dtype='float16'),
            'value': np.ascontiguousarray(values, dtype='float32'),
            'has_value': np.ascontiguousarray(has_value, dtype='bool'),
        }
        self.blocks.append(block)
        new = np.stack([np.full(n, len(self.blocks) - 1, dtype='int64'),
                        np.arange(n, dtype='int64')], axis=1)
        self.index = np.concatenate([self.index, new])
        self._enforce_capacity()

    def _enforce_capacity(self):
        # Dropping whole rounds is against the spirit of DAgger, so this only fires if the
        # caller asked for a bound; oldest first, since those are the states the student has
        # already been trained on the most times.
        if not self.capacity: return
        while len(self.index) > self.capacity and len(self.blocks) > 1:
            dropped = len(self.blocks.pop(0)['value'])
            self.index = self.index[dropped:]
            self.index[:, 0] -= 1
            print('  buffer over capacity: dropped the oldest round ({} states)'.format(dropped))

    def nbytes(self):
        total = 0
        for block in self.blocks:
            total += sum(i.nbytes for i in block['obs'])
            total += block['teacher_logits'].nbytes + block['value'].nbytes + block['has_value'].nbytes
        return total

    def gather(self, idx, device=None):
        """Materialize the states at flat positions idx as torch tensors."""
        rows = self.index[idx]
        n = len(idx)
        obs = [np.empty((n, *shape), dtype=typ) for shape, typ in zip(STATE_SHAPES, STATE_TYPES)]
        teacher_logits = np.empty((n, NUM_ACTIONS), dtype='float16')
        value = np.empty(n, dtype='float32')
        has_value = np.empty(n, dtype='bool')
        for block_id in np.unique(rows[:, 0]):
            where = np.nonzero(rows[:, 0] == block_id)[0]
            offsets = rows[where, 1]
            block = self.blocks[block_id]
            for dst, src in zip(obs, block['obs']):
                dst[where] = src[offsets]
            teacher_logits[where] = block['teacher_logits'][offsets]
            value[where] = block['value'][offsets]
            has_value[where] = block['has_value'][offsets]
        to = (lambda x: torch.from_numpy(x).to(device, non_blocking=True)) if device is not None \
            else torch.from_numpy
        return {
            'obs': [to(i) for i in obs],
            # float16 keeps a round's labels at 1.6 kB per state; the loss casts to float32
            'teacher_logits': to(teacher_logits).float(),
            'value': to(value),
            'has_value': to(has_value),
        }

    def sample_batches(self, batch_size, rng, device=None, drop_last=False):
        """One shuffled pass over everything in the buffer, batch_size states at a time."""
        order = rng.permutation(len(self.index))
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            if drop_last and len(batch) < batch_size: break
            yield self.gather(batch, device)
