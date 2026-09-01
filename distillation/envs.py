"""Whole-game rollout plumbing for the distillation stage.

RL/generator.py drives the C++ Batch for PPO: it fills a fixed-size rollout buffer and
resets finished episodes in place, which is exactly wrong for DAgger.  DAgger wants whole
games, a snapshot of every state it labels so a value rollout can restart from there, and a
policy that changes between rounds.  So this module steps plain tetris.Tetris objects in
lockstep instead, batching their observations so the network still sees one call per step
rather than one per game.

The action encoding is the one the rest of the repo uses -- action = r * 200 + x * 10 + y
over the (rotation, row, column) placements -- with legality read off the move planes.
"""

import argparse
import pathlib
import secrets
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

# model.py lives in RL/ and only training_env is an installed package, so put that
# directory on sys.path before importing from it (same as tools/ and demo/ do)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'RL'))

import numpy as np
import torch

from training_env import tetris
from training_env import ADJ_DELAYS, MILESTONE_LINES, TAP_SEQUENCE_MAP
from model import kH, kW, kR, kMoveStart, obs_to_torch

NUM_ACTIONS = kR * kH * kW
# board_meta[14] is the is-adjustment flag and board planes 2 .. 2+kR hold the one-hot
# pre-adjustment placement; see GetState in training_env/tetris/state.cpp
META_IS_ADJ = 14
BOARD_PREMOVE_START = 2
LINE_CAP = tetris.Tetris.LineCap()


def action_to_position(action):
    return int(action) // 200, int(action) // 10 % 20, int(action) % 10


def position_to_action(r, x, y):
    return int(r) * 200 + int(x) * 10 + int(y)


def legal_mask(obs):
    """Bool (NUM_ACTIONS,) mask of the placements the move search found.

    These are the same planes Model.forward masks its logits with, so student and teacher
    agree on the support without either being told.
    """
    return obs[2][kMoveStart:kMoveStart + kR].reshape(-1) != 0


def stack_obs(obs_list):
    """Stack per-game observation tuples into one batched observation."""
    return [np.stack(part) for part in zip(*obs_list)]


@dataclass(frozen=True)
class EnvParams:
    """The per-episode knobs tetris.Tetris.Reset takes."""
    tap_sequence: Tuple[int, ...]
    adj_delay: int
    start_lines: int
    milestone: int = MILESTONE_LINES


@dataclass(frozen=True)
class Snapshot:
    """Enough of a game to rebuild it exactly.

    Tetris::Reset derives the piece counter from the line count and the board's cell count,
    so a restored game agrees with the original on every observation feature rather than
    only on the board.  An adjustment-phase state also needs the placement that got it
    there: reset lands in the pre-adjustment phase unless the initial placement was forced,
    in which case the C++ side replays it on its own (skip_unique_initial).
    """
    board: bytes
    lines: int
    now_piece: int
    next_piece: int
    is_adj: bool
    premove: Optional[Tuple[int, int, int]]
    params: EnvParams


class ParamSampler:
    """Per-episode parameters, each axis either pinned or drawn from the whole grid.

    Tap speed, adjustment delay, and starting lines can be fixed or sampled. The default
    matches PPO: 30hz taps, an 18-frame adjustment reaction, and a level-18 start.
    """

    def __init__(self, tap_speed='30hz', adj_delay=18, start_lines=0,
                 milestone=MILESTONE_LINES):
        self.tap_speed = tap_speed
        self.adj_delay = adj_delay
        self.milestone = milestone
        self.start_lines = start_lines
        if start_lines != 'all' and int(start_lines) % 2 != 0:
            # Reset requires (lines * 10 + cells) % 4 == 0 and starts from an empty board
            raise ValueError('start-lines must be even or "all"')

    def sample(self, rng):
        tap = self.tap_speed
        if tap == 'all': tap = rng.choice(sorted(TAP_SEQUENCE_MAP))
        adj = self.adj_delay
        if adj == 'all': adj = int(rng.choice(ADJ_DELAYS))
        lines = self.start_lines
        if lines == 'all':
            lines = 2 * int(rng.integers((LINE_CAP - 40) // 2))
        return EnvParams(tap_sequence=tuple(int(i) for i in TAP_SEQUENCE_MAP[tap]),
                         adj_delay=int(adj), start_lines=int(lines),
                         milestone=self.milestone)

    def describe(self):
        return '{} taps, {}f adjustment, milestone {}, start lines {}'.format(
            self.tap_speed, self.adj_delay, self.milestone, self.start_lines)

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument('--tap-speed', default='30hz',
                            choices=sorted(TAP_SEQUENCE_MAP) + ['all'])
        parser.add_argument('--adj-delay', default='18',
                            choices=[str(i) for i in ADJ_DELAYS] + ['all'])
        parser.add_argument('--milestone', type=int, default=MILESTONE_LINES,
                            help='absolute line count that ends an episode at no cost')
        parser.add_argument('--start-lines', default='0',
                            help='even line count to start every game at, or "all" to sample')

    @classmethod
    def from_args(cls, args):
        start_lines = args.start_lines
        return cls(tap_speed=args.tap_speed,
                   adj_delay=args.adj_delay if args.adj_delay == 'all' else int(args.adj_delay),
                   start_lines=start_lines if start_lines == 'all' else int(start_lines),
                   milestone=getattr(args, 'milestone', MILESTONE_LINES))


class Game:
    """One game, plus the snapshotting the value rollouts need."""

    def __init__(self, seed=None):
        self.env = tetris.Tetris(seed=secrets.randbelow(2 ** 40) if seed is None else int(seed))
        self.params = None

    def reset(self, params: EnvParams, now_piece=None, next_piece=None, board=None, lines=None):
        if lines is None: lines = params.start_lines
        kwargs = {'lines': int(lines),
                  'tap_sequence': list(params.tap_sequence),
                  'adj_delay': params.adj_delay,
                  'milestone': params.milestone,
                  'skip_unique_initial': True}
        # Reset reads a passed None as a bad piece id rather than "pick one", so the
        # arguments have to be left out entirely to get random pieces
        if now_piece is not None: kwargs['now_piece'] = int(now_piece)
        if next_piece is not None: kwargs['next_piece'] = int(next_piece)
        if board is not None:
            kwargs['board'] = board
            # Reset derives the piece count as (lines * 10 + cells) / 4 and throws a C++
            # exception -- which aborts the interpreter -- if that does not divide
            if (int(lines) * 10 + board.Count()) % 4 != 0:
                raise ValueError('board with {} cells is not reachable at {} lines'.format(
                    board.Count(), lines))
        self.env.Reset(**kwargs)
        self.params = params
        return self.obs()

    def obs(self):
        return self.env.GetState()

    @property
    def over(self):
        """Whether the *episode* ended -- a top-out or the milestone (spec section 1)."""
        return self.env.IsEpisodeOver()

    @property
    def lines(self):
        """Absolute line count."""
        return self.env.GetLines()

    @property
    def run_pieces(self):
        return self.env.GetRunPieces()

    @property
    def run_lines(self):
        return self.env.GetRunLines()

    def step(self, action):
        """Place a piece; returns the env's (reward, cost).  See training_env/tetris/tetris.h."""
        return self.env.InputPlacement(*action_to_position(action))

    def snapshot(self, obs=None) -> Snapshot:
        if obs is None: obs = self.obs()
        is_adj = bool(obs[1][META_IS_ADJ])
        premove = None
        if is_adj:
            r, x, y = np.argwhere(obs[0][BOARD_PREMOVE_START:BOARD_PREMOVE_START + kR])[0]
            premove = (int(r), int(x), int(y))
        return Snapshot(board=self.env.GetBoard().GetBytes(), lines=self.env.GetLines(),
                        now_piece=self.env.GetNowPiece(), next_piece=self.env.GetNextPiece(),
                        is_adj=is_adj, premove=premove, params=self.params)

    @classmethod
    def restore(cls, snap: Snapshot, seed=None):
        """Rebuild the game a snapshot came from, with a fresh piece-generator stream."""
        game = cls(seed)
        game.reset(snap.params, now_piece=snap.now_piece, next_piece=snap.next_piece,
                   board=tetris.Board(snap.board), lines=snap.lines)
        if snap.is_adj and not game.over and not bool(game.obs()[1][META_IS_ADJ]):
            game.step(position_to_action(*snap.premove))
        return game


class TorchPolicy:
    """A Model behind the small interface the rollout helpers use."""

    def __init__(self, model, device=None, batch_size=512, temperature=1.0):
        self.model = model
        self.device = torch.device(device if device is not None else
                                   ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.batch_size = batch_size
        self.temperature = temperature

    def _chunks(self, obs_batch):
        n = len(obs_batch[0])
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            yield obs_to_torch([i[start:end] for i in obs_batch], self.device)

    @torch.no_grad()
    def action_logits(self, obs_batch):
        """(batch, NUM_ACTIONS) float32 logits, already -inf at every illegal placement."""
        return torch.cat([self.model(chunk, pi_only=True)[0].float().cpu()
                          for chunk in self._chunks(obs_batch)])

    @torch.no_grad()
    def values(self, obs_batch):
        """(batch,) state values -- row 0 of the model's three-row value output."""
        return torch.cat([self.model(chunk, pi_only=True)[1][0].float().cpu()
                          for chunk in self._chunks(obs_batch)]).numpy()

    @torch.no_grad()
    def act(self, obs_batch, deterministic=False):
        """Actions for a batched observation, through Model.get_action."""
        actions = []
        for chunk in self._chunks(obs_batch):
            if deterministic or self.temperature == 1.0:
                action = self.model.get_action(chunk, deterministic=deterministic)[0]
            else:
                # temperature is not part of get_action's contract, so sharpen the already
                # masked logits and sample from those
                logits = self.model(chunk, pi_only=True)[0].float() / self.temperature
                action = torch.distributions.Categorical(logits=logits).sample()
            actions.append(action.cpu())
        return torch.cat(actions).numpy()


class Reservoir:
    """Uniform sample of a stream whose length is not known up front."""

    def __init__(self, capacity, rng):
        self.capacity = capacity
        self.rng = rng
        self.seen = 0
        self.items = []

    def add(self, item):
        self.seen += 1
        if self.capacity is None or len(self.items) < self.capacity:
            self.items.append(item)
            return
        j = int(self.rng.integers(self.seen))
        if j < self.capacity: self.items[j] = item


def collect_rollout_states(policy, num_games, param_sampler, rng, noise_prob=0.15,
                           max_pieces=300, max_states=None, deterministic=False,
                           progress=False):
    """Play num_games games in lockstep and return the decision points they passed through.

    With probability noise_prob a step takes a uniformly random *legal* placement instead of
    the policy's.  That is the point of injecting noise into DAgger: it drags the rollout
    into the messy boards a competent policy never reaches on its own, and what gets stored
    is the state, not the action taken there -- the label comes from the teacher afterwards.

    max_states keeps a uniform random sample of the states, so a policy that survives for a
    thousand pieces cannot swamp the round with one game's worth of them.  Returns
    (states, stats), where each state is an (obs, snapshot) pair.
    """
    games = [Game() for _ in range(num_games)]
    for game in games:
        game.reset(param_sampler.sample(rng))
    states = Reservoir(max_states, rng)
    lengths, truncated, topped_out = [], 0, 0
    reached_milestone = 0
    alive = [i for i, g in enumerate(games) if not g.over]
    step = 0
    while alive:
        obs_list = [games[i].obs() for i in alive]
        actions = policy.act(stack_obs(obs_list), deterministic=deterministic)
        noise = rng.random(len(alive)) < noise_prob
        for k, i in enumerate(alive):
            obs = obs_list[k]
            states.add((obs, games[i].snapshot(obs)))
            action = int(actions[k])
            if noise[k]:
                legal = np.nonzero(legal_mask(obs))[0]
                if len(legal): action = int(rng.choice(legal))
            games[i].step(action)
        still = []
        for i in alive:
            if games[i].over:
                # `over` is IsEpisodeOver: a top-out, or the milestone reached at no cost
                if games[i].env.ReachedMilestone():
                    reached_milestone += 1
                else:
                    topped_out += 1
                lengths.append(games[i].run_pieces)
            elif games[i].run_pieces >= max_pieces:
                truncated += 1
                lengths.append(games[i].run_pieces)
            else:
                still.append(i)
        alive = still
        step += 1
        if progress and step % 25 == 0:
            print('    step {}: {}/{} games alive, {} states seen'.format(
                step, len(alive), num_games, states.seen), flush=True)
    stats = {
        'games': num_games,
        'states_seen': states.seen,
        'mean_game_length': float(np.mean(lengths)) if lengths else 0.,
        'median_game_length': float(np.median(lengths)) if lengths else 0.,
        'max_game_length': int(np.max(lengths)) if lengths else 0,
        'mean_lines': float(np.mean([g.run_lines for g in games])),
        'truncated': truncated,
        'topped_out': topped_out,
        'reached_milestone': reached_milestone,
    }
    return states.items, stats


def rollout_returns(policy, snapshots, n_rollouts=5, horizon=50, gamma=1.0,
                    deterministic=False, bootstrap=False):
    """Mean return of a policy played out from each snapshot.

    Every snapshot is restarted n_rollouts times and all the copies are stepped in lockstep,
    so the cost is one batched forward pass per step rather than one per game.  The return
    uses the same recursion generator.py folds its score-channel advantages with, which is
    what makes the result a target for the same value head:

        G_t = r_t + gamma * G_{t+1},   G = 0 at either terminal

    gamma is 1 to match the reward spec: the return is the score still to come, in units of
    K = 22800, undiscounted.  A rollout that reaches horizon pieces contributes nothing past
    that point unless bootstrap is set, in which case the policy's own value estimate stands
    in for the tail -- and it has to be, or the target is "score in the next `horizon`
    pieces" rather than the value.
    """
    n = len(snapshots)
    if n == 0: return np.zeros(0, dtype='float32')
    games = [Game.restore(snap) for snap in snapshots for _ in range(n_rollouts)]
    starts = [g.run_pieces for g in games]
    trajectories = [[] for _ in games]
    finished = [g.over for g in games]
    alive = [i for i, g in enumerate(games) if not g.over]
    while alive:
        obs_list = [games[i].obs() for i in alive]
        actions = policy.act(stack_obs(obs_list), deterministic=deterministic)
        for k, i in enumerate(alive):
            reward, _cost = games[i].step(int(actions[k]))
            trajectories[i].append(reward)
        still = []
        for i in alive:
            if games[i].over:
                finished[i] = True
            elif games[i].run_pieces - starts[i] < horizon:
                still.append(i)
        alive = still
    tails = np.zeros(len(games), dtype='float64')
    if bootstrap:
        pending = [i for i, done in enumerate(finished) if not done]
        if pending:
            tails[pending] = policy.values(stack_obs([games[i].obs() for i in pending]))
    returns = np.empty(len(games), dtype='float64')
    for i, steps in enumerate(trajectories):
        g = tails[i]
        for reward in reversed(steps):
            g = reward + gamma * g
        returns[i] = g
    return returns.reshape(n, n_rollouts).mean(axis=1).astype('float32')
