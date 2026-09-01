import secrets
from typing import Optional

import numpy as np

from . import tetris
from .game_param import GameParamManager, MILESTONE_LINES, segment_label

# Keys of a sampled parameter dict that are bookkeeping only and not accepted by ResetEnv.
# `line_cap`, `milestone` and `is_true_start` are not among them: the C++ batch takes all
# three.
_NON_RESET_KEYS = ('tap_id', 'adj_delay_id', 'is_board')


class BatchedGames:
    """All environments of a rollout, stepped in lockstep by the C++ thread pool.

    Observations are written by the C++ side straight into `obs`, a (worker_steps + 1, envs, ...)
    rollout buffer owned by the caller: step t reads slot t and stores its result in slot t + 1,
    so a rollout involves no per-env Python work, no allocation and no copies.

    Two per-step channels come back beside them (NEW_DESIGN_SPEC.md sections 2 and 3):

        rewards[env, t] = (r_t, cost_t)     r_t = score of the clear / 22800; cost_t = 1
                                            exactly on the transition into TOP_OUT
        is_over[env, t] = (done, truncated) done: the episode ended here and the env was
                                            reset.  truncated: it was *cut* rather than
                                            ended, so both value heads must bootstrap off
                                            the state that follows instead of taking 0.

    `eval_ratio` reserves the first envs for true level-18 starts whatever the training
    reset distribution is doing, so that p_hat and the reported numbers always have a
    sample from the distribution `c` is a constraint on.
    """

    def __init__(self, envs: int, worker_steps: int, obs: list, num_threads: int = 0,
                 seed: Optional[int] = None, board_file: Optional[str] = None,
                 segment=None, tap_ids=None, adj_delay_ids=None,
                 milestone: int = MILESTONE_LINES, eval_ratio: float = 0.0):
        self.envs = envs
        self.worker_steps = worker_steps
        self.rewards = np.zeros((envs, worker_steps, 2), dtype='float32')
        self.is_over = np.zeros((envs, worker_steps, 2), dtype='bool')
        if seed is None: seed = secrets.randbelow(2**40)
        self.batch = tetris.Batch(envs, seed=seed, num_threads=num_threads)
        self.batch.SetBuffers(obs, self.rewards, self.is_over)
        self.manager = GameParamManager(board_file, segment=segment, tap_ids=tap_ids,
                                        adj_delay_ids=adj_delay_ids, milestone=milestone)
        self.milestone = self.manager.milestone
        self.n_eval = min(envs, int(round(envs * max(0.0, eval_ratio))))
        if self.n_eval and self.manager.true_start_lines is None:
            # A segment that does not contain line 0 has no true start to evaluate on.
            self.n_eval = 0
        self.params = [None] * envs
        self.total_games = 0
        self.total_topouts = 0
        self.total_milestones = 0
        # Episodes cut rather than ended -- a short curriculum game or a segment line_cap.
        # Neither terminal, and never counted as one.
        self.total_truncated = 0
        if segment is not None:
            print('segment {}: {} envs at {}'.format(
                segment_label(segment), envs, 'the pinned tap/adj settings'
                if tap_ids is not None or adj_delay_ids is not None else 'every tap/adj setting'))

    def num_threads(self) -> int:
        return self.batch.NumThreads()

    def set_params(self, params):
        self.manager.UpdateParams(params)

    def save_params(self):
        self.manager.SaveParams()

    def _reset(self, idx: int):
        while True:
            params = self.manager.GetNewParam(force_true_start=idx < self.n_eval)
            self.params[idx] = params
            kwargs = {k: v for k, v in params.items() if k not in _NON_RESET_KEYS}
            if not self.batch.ResetEnv(idx, skip_unique_initial=True, **kwargs):
                return

    def reset_all(self):
        self.batch.SetObsStep(0)
        for i in range(self.envs): self._reset(i)

    def step(self, step: int, actions: np.ndarray, ret_info: Optional[dict] = None):
        """Steps every env once. Finished episodes are reset and their stats folded into ret_info.

        Returns `(true_start_finished, true_start_topouts)` for that step: the numerator and
        denominator of the batch estimate of p_hat = P(top out before the milestone), which
        the dual update in RL/train.py consumes.  Only true level-18 starts are counted, so
        `c` stays a constraint on the distribution the spec defines even when the training
        reset distribution is something else.
        """
        finished = self.batch.Step(step, actions)
        if not finished: return 0, 0
        n_true = n_topout = 0
        self.total_games += len(finished)
        for (idx, is_short, is_topout, is_milestone, is_true_start, reward,
             score, lines, pieces, tetrises) in finished:
            truncated = not is_topout and not is_milestone
            if is_topout: self.total_topouts += 1
            if is_milestone: self.total_milestones += 1
            if truncated: self.total_truncated += 1
            if is_true_start and not truncated:
                n_true += 1
                n_topout += is_topout
            if ret_info is not None:
                if is_short:
                    ret_info['short_finish'].append(0.0 if is_topout else 1.0)
                elif truncated:
                    ret_info['truncated'].append(1.0)
                if is_true_start and not truncated:
                    # Everything reported is measured here, on the spec's episode.
                    ret_info['topout'].append(1.0 if is_topout else 0.0)
                    ret_info['ret'].append(reward)
                    ret_info['scorek'].append(score * 1e-3)
                    ret_info['lns'].append(lines)
                    ret_info['pcs'].append(pieces)
                    if lines > 0:
                        ret_info['tetris_rate'].append(4.0 * tetrises / lines)
                    ret_info['burn'].append(lines - 4 * tetrises)
                    if is_milestone:
                        # Pace: total score at the milestone.  Only defined for an episode
                        # that got there (spec section 10).
                        ret_info['pace'].append(score * 1e-3)
            self.manager.UpdateState(self.params[idx], pieces, lines)
            self._reset(idx)
        return n_true, n_topout
