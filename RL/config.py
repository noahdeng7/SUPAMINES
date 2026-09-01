import os, argparse
from typing import Optional

from labml import experiment
from labml.configs import BaseConfigs, FloatDynamicHyperParam, IntDynamicHyperParam

from training_env import MILESTONE_LINES

class Configs(BaseConfigs):
    # #### Configurations
    ## NN
    # The spatial transformer in model.py: 5.17M parameters at these defaults, which is the
    # size a segment expert is trained at.  `heads` leaves no trace in a checkpoint's tensor
    # shapes, so a checkpoint written by saver.py can only be read back as d_model / 32
    # heads -- change it here and you have to remember it.  (distillation writes the
    # architecture into its checkpoints; see model.save_model.)
    d_model: int = 256
    layers: int = 6
    heads: int = 8
    mlp_ratio: float = 4.

    def model_args(self):
        return {'d_model': self.d_model, 'layers': self.layers,
                'heads': self.heads, 'mlp_ratio': self.mlp_ratio}

    ## episode -- NEW_DESIGN_SPEC.md section 1
    # Absolute line count that ends an episode at no cost.  230 is level-29 entry from a
    # level-18 start, the milestone the pace-survival frontier is measured at.
    milestone: int = MILESTONE_LINES
    # The level segment: '18', '19', '29' or '39' (see training_env/game_param.py), or None
    # to play the whole episode.  Leave it out for the spec's episode: pinning a segment
    # restricts where episodes start and cuts them at the segment boundary, which is a
    # training aid and not the thing being measured.
    segment: Optional[str] = None
    # Tap speed and adjustment reaction time.  These default to the setting the experts are
    # trained for; every other combination is binned rather than down-weighted.  Pass 'all'
    # for the full sweep the generalist checkpoints were trained under.
    tap_speed: str = '30hz'
    adj_delay: str = '18'
    # Fraction of envs reserved for true level-18 starts, whatever the training reset
    # distribution is.  Randomized resets (`mid_ratio`, `board_ratio`) are a strong training
    # aid but they change what p_hat means, so raise this alongside them or `c` ends up
    # constraining a distribution nobody plays (spec section 9).  0 is right only while the
    # training distribution *is* true starts, which is the default.
    eval_ratio: float = 0.

    ## training
    lr: float = FloatDynamicHyperParam(1e-4, range_ = (0, 1e-3))
    # $\gamma$ and $\lambda$ for advantage calculation.
    #
    # gamma is 1.0 on purpose and applies to both channels (spec section 2).  The episode is
    # guaranteed finite -- it ends at the milestone or at a top-out -- so the undiscounted
    # return is well defined and equals exactly the quantity being measured.  Discounting
    # would make the objective "discounted pace", which is not a thing anyone wants to know,
    # and at ~600-800 placements per episode any gamma < 0.999 makes the milestone invisible
    # to the critic anyway.  Control return variance with lamda, which trades bias for
    # variance in the advantage estimate without changing the objective.
    gamma: float = FloatDynamicHyperParam(1.0, range_ = (0.95, 1))
    lamda: float = FloatDynamicHyperParam(0.95, range_ = (0.7, 1))
    # number of updates
    updates: int = 400000
    # number of epochs to train the model with sampled data
    epochs: int = 1
    # total environments is n_workers * env_per_worker; they are all stepped by one
    # C++ thread pool inside the generator process
    n_workers: int = 2
    env_per_worker: int = 100
    # env stepping threads; 0 = one per hardware thread
    env_threads: int = 0
    # number of steps to run on each process for a single update
    worker_steps: int = 128
    # size of mini batches
    n_update_per_epoch: int = 32
    # calculate loss in batches of mini_batch_size
    mini_batch_size: int = 800
    supervised_batch_size: int = IntDynamicHyperParam(400, range_ = (0, 4096))
    weight_sync_per_epoch: int = 2

    ## constraint -- spec sections 3 and 4
    # c: the allowed probability of topping out before the milestone.  The sweep in section
    # 9 is over exactly this number; everything else is held identical across runs.
    cost_limit: float = FloatDynamicHyperParam(0.05, range_ = (0, 1))
    # Dual ascent on beta, once per PPO iteration (never per minibatch):
    #     p_hat <- ema(fraction of true-start episodes ending in TOP_OUT, decay=beta_ema)
    #     beta  <- max(0, beta + lr_dual * (p_hat - c))     [+ kp_dual * (p_hat - c)]
    #
    # beta starts at 0 so the agent learns to score first and the constraint engages only
    # once p_hat rises past c.  Starting it high gives a policy that satisfies the
    # constraint by never doing anything interesting, and it never escapes.
    #
    # lr_dual is the integral gain, and the spec asks for it an order of magnitude or more
    # below the policy LR -- beta chases a moving target, and matched speeds produce the
    # standard oscillation (beta overshoots, policy turtles, p_hat crashes, beta collapses,
    # policy goes reckless, repeat).  Note the units: a dual step is lr_dual * (p_hat - c)
    # with |p_hat - c| <= 1, so at 1e-5 beta moves by at most 1e-5 per iteration and needs
    # tens of thousands of iterations to reach O(1).  That is deliberate for the integral
    # term.  If the constraint has to engage faster, reach for kp_dual first: plain dual
    # ascent is pure integral control, which is why it rings, and a proportional term
    # (Stooke et al., PID Lagrangian Methods) responds immediately without the wind-up.
    lr_dual: float = FloatDynamicHyperParam(1e-5, range_ = (0, 1e-2))
    # The range_ is the web UI's slider bound, not a clamp on the flag.  It goes to 200
    # because the useful value is set by the ratio of the two advantage scales, and at the
    # measured 1.37 / 0.036 that is ~150 -- far outside anything a (0, 10) slider suggests.
    kp_dual: float = FloatDynamicHyperParam(0., range_ = (0, 200))
    beta_init: float = 0.
    beta_ema: float = 0.9
    # Optional smooth nonnegativity floor: beta = softplus(theta) instead of max(0, beta).
    dual_softplus: bool = False

    ## loss calculation
    use_kl: bool = False
    clipping_range: float = 0.2
    beta: float = 5.0
    policy_weight: float = FloatDynamicHyperParam(1, range_ = (0, 5))
    vf_weight: float = FloatDynamicHyperParam(1, range_ = (0, 5))
    # BCE on the cost critic.  Not MSE: V_C predicts a Bernoulli outcome and the policy
    # operates near the boundary at low c, where a regression head is badly calibrated.
    cost_vf_weight: float = FloatDynamicHyperParam(1, range_ = (0, 5))
    low_prob_threshold: float = FloatDynamicHyperParam(5e-4, range_ = (0, 1e-2))
    low_prob_weight: float = FloatDynamicHyperParam(1e-2, range_ = (0, 1))
    entropy_weight: float = FloatDynamicHyperParam(1.5e-2, range_ = (0, 5e-2))
    supervised_weight: float = FloatDynamicHyperParam(0, range_ = (0, 10))
    supervised_smooth: float = FloatDynamicHyperParam(1e-7, range_ = (0, 1e-2))
    reg_l2: float = FloatDynamicHyperParam(0., range_ = (0, 5e-5))

    ## reset distribution (a training aid; see eval_ratio)
    # Fraction of episodes that start at a random line count inside the run's range instead
    # of at a true level-18 start, and the curriculum-board ratios that go with --board-file.
    mid_ratio = FloatDynamicHyperParam(0., range_ = (-1, 1))
    board_ratio = FloatDynamicHyperParam(0., range_ = (-1, 1))
    short_ratio = FloatDynamicHyperParam(0., range_ = (-1, 1))

    ## potential-based shaping -- spec section 5
    # A checkpoint whose value head is used as Phi.  The shaping added to r_t is
    #   F_t = gamma * Phi(s_{t+1}) - Phi(s_t),  Phi(terminal) = 0
    # which is policy-invariant for any Phi (Ng, Harada & Russell 1999), so it cannot move
    # the frontier being measured.  It is annealed linearly to 0 by `phi_anneal_end` of the
    # run so that the final policy optimizes the true objective exactly -- non-negotiable
    # given this environment is a measuring instrument.  Off unless a file is given.
    phi_model_file: Optional[str] = None
    phi_scale: float = 1.
    phi_anneal_end: float = 0.5

    ## distillation hand-off (see ../distillation)
    # a plain state_dict to start from, e.g. distillation's distilled_checkpoint.pt
    init_model_file: Optional[str] = None
    # the frozen distilled policy to hold the fine-tuned one near; enables the KL penalty
    #   reward_total = shaped_reward - beta_kl * KL(current || distilled)
    # applied per step in generator.py before GAE.  Costs a second forward pass per step.
    # This one is *not* potential-based, so it biases the frontier while it is on; prefer
    # --phi-model-file, and report any run that used it as such.
    kl_distill_file: Optional[str] = None
    beta_kl: float = 0.1
    # fraction of the run at which beta_kl starts annealing linearly to 0 at the end
    kl_anneal_start: float = 0.5

    # Seed for the env batch and for torch.  Section 9 asks for identical seeds across the
    # `c` sweep -- only `c` may vary, or the sweep measures architecture noise.  0 means
    # "draw one", which is fine for a single run and wrong for a sweep.
    seed: int = 0

    time_limit: int = -1
    save_interval: int = 250
    warmup_epochs: int = 16
    board_file: Optional[str] = None
    supervised_file: Optional[str] = None


# Config keys whose value is a path or a name rather than a number.
STR_KEYS = ('board_file', 'supervised_file', 'init_model_file', 'kl_distill_file',
            'phi_model_file', 'segment', 'tap_speed', 'adj_delay')


def MaxUUID(name):
    mx_uuid = 0
    for i in os.listdir('logs/{}'.format(name)):
        if i[:len(name)+1] == '{}-'.format(name):
            try:
                mx_uuid = max(mx_uuid, int(i[len(name)+1:]))
            except ValueError: pass
    return mx_uuid


def LoadConfig(with_experiment = True):
    parser = argparse.ArgumentParser()
    if with_experiment:
        parser.add_argument('name')
        parser.add_argument('uuid', nargs = '?', default = '')
        parser.add_argument('checkpoint', nargs = '?', type = int, default = None)
        parser.add_argument('--ignore-optimizer', action = 'store_true')
    conf = Configs()
    keys = conf._to_json()
    dynamic_keys = set()
    for key in keys:
        ptype = type(conf.__getattribute__(key))
        flag = '--' + key.replace('_', '-')
        if ptype == FloatDynamicHyperParam:
            ptype = float
            dynamic_keys.add(key)
        elif ptype == IntDynamicHyperParam:
            ptype = int
            dynamic_keys.add(key)
        elif ptype == bool:
            # `type=bool` would make `--use-kl False` set it to True, since bool('False')
            # is True.  BooleanOptionalAction gives `--use-kl` / `--no-use-kl` instead, and
            # leaves the value None when neither is passed, which is what the override
            # logic below expects.
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=None)
            continue
        elif key in STR_KEYS:
            ptype = str
        parser.add_argument(flag, type = ptype)

    args, others = parser.parse_known_args()
    args = vars(args)
    override_dict = {}
    for key in keys:
        if key not in dynamic_keys and args[key] is not None: override_dict[key] = args[key]
    conf = Configs()
    for key in dynamic_keys:
        if args[key] is not None:
            conf.__getattribute__(key).set_value(args[key])
    if with_experiment:
        name = args['name']
        os.makedirs('logs/{}'.format(name), exist_ok = True)
        uuid = MaxUUID(name) + 1
        experiment.create(name = args['name'], uuid = '{0}-{1:03d}'.format(args['name'], uuid))
        experiment.configs(conf, override_dict)
    else:
        for key, val in override_dict.items():
            conf.__setattr__(key, val)
    return conf, args, others
