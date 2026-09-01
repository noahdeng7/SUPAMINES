#!/usr/bin/env python3

# Modified from https://github.com/vpj/rl_samples

import traceback, time, random
from typing import Dict

import torch
from torch import optim
from torch.nn import functional as F
from torch.distributions import Categorical, kl_divergence
from torch.amp import GradScaler

import labml.lab
from labml import monit, tracker, logger, experiment

from generator import GeneratorProcess
from model import Model, adapt_state_dict, infer_model_args, model_state_dict, obs_to_torch
from config import Configs, LoadConfig, MaxUUID
from saver import TorchSaver
from training_env import tetris

start_time = time.time()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
generator_device = torch.device('cuda:1') if torch.cuda.device_count() > 1 else device


class DualState:
    """The Lagrange multiplier and the EMA it is driven by (NEW_DESIGN_SPEC.md section 4).

    Kept as its own object so that it goes into the checkpoint next to the model: resuming
    a run with beta reset to 0 restarts the dual from scratch and throws away everything
    the constraint had learned.

    `theta` is the raw variable; beta is `max(0, theta)`, or `softplus(theta)` when
    `dual_softplus` is on, which gives the same nonnegativity floor smoothly.
    """

    def __init__(self, c: Configs):
        self.softplus = c.dual_softplus
        self.decay = c.beta_ema
        self.theta = float(c.beta_init)
        self.p_hat = 0.
        self.started = False

    @property
    def beta(self):
        if self.softplus:
            return float(F.softplus(torch.tensor(self.theta)))
        return max(0., self.theta)

    def update(self, p_batch, cost_limit, lr_dual, kp_dual):
        """One dual step, per PPO iteration -- never per minibatch.

        Plain dual ascent is pure integral control, which is why it rings; kp_dual adds the
        proportional term from Stooke et al.'s PID Lagrangian methods, which responds to the
        current violation instead of the accumulated one.  It is applied on top of the
        integral state rather than into it, so it cannot wind up.
        """
        if p_batch is not None:
            self.p_hat = p_batch if not self.started else (
                self.decay * self.p_hat + (1. - self.decay) * p_batch)
            self.started = True
        violation = self.p_hat - cost_limit
        self.theta = self.theta + lr_dual * violation
        if not self.softplus:
            # Keep the integral state itself nonnegative, or beta sits at 0 while theta digs
            # an arbitrarily deep hole that has to be climbed back out of before the
            # constraint can ever re-engage.
            self.theta = max(0., self.theta)
        proportional = kp_dual * max(0., violation)
        return self.beta + proportional

    def state_dict(self):
        return {'theta': torch.tensor(self.theta), 'p_hat': torch.tensor(self.p_hat),
                'started': torch.tensor(float(self.started))}

    def load_state_dict(self, state):
        self.theta = float(state['theta'])
        self.p_hat = float(state['p_hat'])
        self.started = bool(float(state['started']))


class Main:
    def __init__(self, c: Configs, name: str):
        self.name = name
        self.c = c
        # total number of samples for a single update
        self.envs = self.c.n_workers * self.c.env_per_worker
        self.batch_size = self.envs * self.c.worker_steps
        assert (self.batch_size % (self.c.n_update_per_epoch * self.c.mini_batch_size) == 0)
        self.update_batch_size = self.batch_size // self.c.n_update_per_epoch

        assert self.c.n_update_per_epoch % self.c.weight_sync_per_epoch == 0
        assert self.c.worker_steps % self.c.weight_sync_per_epoch == 0
        self.send_weight_interval = self.c.epochs * self.c.n_update_per_epoch // self.c.weight_sync_per_epoch

        # #### Initialize
        # model for sampling
        self.model = Model(**c.model_args()).to(device, memory_format=torch.channels_last)
        if self.c.init_model_file: self._load_init_model(self.c.init_model_file)
        self.model_opt = torch.compile(self.model)

        # dynamic hyperparams
        self.cur_lr = self.c.lr()
        self.cur_reg_l2 = self.c.reg_l2()
        self.cur_game_params = (0., 0., 0., 0., 0.)
        self.cur_beta_kl = 0.
        self.cur_phi_coef = 0.
        self.set_weight_params()

        # the constrained side: beta and the p_hat it chases
        self.dual = DualState(c)
        self.cur_beta = self.dual.beta

        # optimizer
        self.scaler = GradScaler('cuda')
        self.optimizer = optim.Adam(self.model_opt.parameters(),
                lr=self.cur_lr, weight_decay=self.cur_reg_l2)

        # generator
        cur_params = self.get_game_params()
        self.generator = GeneratorProcess(self.model_opt, self.name, self.c, cur_params, generator_device)
        self.set_game_params(cur_params)
        self.supervised = None
        if self.c.supervised_file:
            self.supervised = tetris.SupervisedDataReader(self.c.supervised_file, random.randint(0, 2 ** 32 - 1))

    def _load_init_model(self, path):
        """Start from distillation's checkpoint, not a labml one.

        Distillation records the architecture in the file and older checkpoints do not, so
        `infer_model_args` gives the best answer available either way; compare it against
        what the configs asked for and say which flags would have matched.  `adapt_state_dict`
        then carries a pre-cost-critic value head across.
        """
        checkpoint = torch.load(path, weights_only=True, map_location=device)
        want = infer_model_args(checkpoint)
        have = self.c.model_args()
        if want != have:
            flags = ' '.join('--{} {}'.format(k.replace('_', '-'), v) for k, v in want.items())
            raise RuntimeError('{} has a different architecture ({} vs {}); rerun with: {}'.format(
                path, want, have, flags))
        self.model.load_state_dict(adapt_state_dict(model_state_dict(checkpoint), self.model))

    def get_beta_kl(self, epoch):
        """Weight of the KL penalty against the distilled policy: held flat for the first
        `kl_anneal_start` of the run, then annealed linearly to 0 at the last update."""
        if not self.c.kl_distill_file: return 0.
        frac = epoch / max(1, self.c.updates)
        span = max(1e-9, 1. - self.c.kl_anneal_start)
        return self.c.beta_kl * min(1., max(0., (1. - frac) / span))

    def get_phi_coef(self, epoch):
        """Weight of the potential-based shaping: 1 at the start, linearly to 0 by
        `phi_anneal_end` of the run, and 0 for the rest of it.

        The annealing is not optional.  Potential-based shaping cannot move the optimum, but
        it does change what a partially-trained policy optimizes, and this environment is a
        measuring instrument: the policy whose pace gets reported has to have been optimizing
        the true objective, exactly.
        """
        if not self.c.phi_model_file: return 0.
        frac = epoch / max(1, self.c.updates)
        return min(1., max(0., 1. - frac / max(1e-9, self.c.phi_anneal_end)))

    def set_beta_kl(self, beta_kl):
        if beta_kl == self.cur_beta_kl: return
        self.generator.SetBetaKL(beta_kl)
        self.cur_beta_kl = beta_kl

    def set_phi_coef(self, phi_coef):
        if phi_coef == self.cur_phi_coef: return
        self.generator.SetPhiCoef(phi_coef)
        self.cur_phi_coef = phi_coef

    def get_game_params(self):
        return (self.c.gamma(), self.c.lamda(), self.c.mid_ratio(), self.c.board_ratio(),
                self.c.short_ratio())

    def set_optim(self, lr, reg_l2):
        if lr == self.cur_lr and reg_l2 == self.cur_reg_l2: return
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            param_group['weight_decay'] = reg_l2
        self.cur_lr = lr
        self.cur_reg_l2 = reg_l2

    def set_game_params(self, game_params):
        if game_params == self.cur_game_params: return
        self.generator.SetParams(game_params)
        self.cur_game_params = game_params

    def set_weight_params(self):
        self.cur_policy_weight = self.c.policy_weight()
        self.cur_entropy_weight = self.c.entropy_weight()
        self.cur_vf_weight = self.c.vf_weight()
        self.cur_cost_vf_weight = self.c.cost_vf_weight()
        self.cur_low_prob_weight = self.c.low_prob_weight()
        self.cur_low_prob_threshold = self.c.low_prob_threshold()
        self.cur_supervised_weight = self.c.supervised_weight()
        self.cur_supervised_smooth = self.c.supervised_smooth()
        self.cur_supervised_batch_size = self.c.supervised_batch_size()

    def update_dual(self, info):
        """Advance beta from this batch's top-out rate, once per PPO iteration.

        `p_hat` is measured on true level-18 starts only -- see BatchedGames.step -- so `c`
        stays a constraint on the episode the spec defines even when the training reset
        distribution is something else.  An iteration in which no true-start episode finished
        contributes no new observation and leaves the EMA where it was.
        """
        cost_limit = self.c.cost_limit()
        self.cur_beta = self.dual.update(info.get('p_hat'), cost_limit,
                                         self.c.lr_dual(), self.c.kp_dual())
        tracker.add({'beta': self.cur_beta, 'p_hat_ema': self.dual.p_hat,
                     'cost_limit': cost_limit,
                     'violation': self.dual.p_hat - cost_limit})

    def destroy(self):
        self.generator.Close()
        self.generator = None

    def train(self, samples: Dict[str, torch.Tensor]):
        """### Train the model based on samples"""
        self._preprocess_samples(samples)
        n_updates = 0
        for _ in range(self.c.epochs):
            # shuffle for each epoch (on device: the indices only ever index device tensors)
            indexes = torch.randperm(self.batch_size, device=device)
            for start in range(0, self.batch_size, self.update_batch_size):
                # get mini batch
                end = start + self.update_batch_size
                # train
                self.optimizer.zero_grad(set_to_none=True)
                loss_mul = self.update_batch_size // self.c.mini_batch_size
                for t_start in range(start, end, self.c.mini_batch_size):
                    t_end = t_start + self.c.mini_batch_size
                    mini_batch_indexes = indexes[t_start:t_end]
                    mini_batch = {}
                    with torch.no_grad():
                        for k, v in samples.items():
                            if k == 'obs':
                                if self.supervised and self.cur_supervised_batch_size:
                                    # somehow if we call the model multiple times the results are messed up
                                    # so we concat supervised & self-play data into the same tensor
                                    x, y = self.supervised.ReadBatch(self.cur_supervised_batch_size)
                                    x = obs_to_torch(x, device)
                                    mini_batch['supervised_y'] = torch.tensor(y, device=device)
                                    mini_batch[k] = [torch.cat([i[mini_batch_indexes], j]) for i, j in zip(v, x)]
                                else:
                                    mini_batch[k] = [i[mini_batch_indexes] for i in v]
                            else:
                                mini_batch[k] = v[mini_batch_indexes]
                    loss = self._calc_loss(samples=mini_batch) / loss_mul
                    self._backward(loss)
                self._grad_update()
                n_updates += 1
                if (n_updates + 1) % self.send_weight_interval == 0:
                    self.generator.SendModel(self.model_opt)

    def _backward(self, loss):
        self.scaler.scale(loss).backward()

    def _grad_update(self):
        # compute gradients
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model_opt.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_value_(self.model_opt.parameters(), 16)
        self.scaler.step(self.optimizer)
        self.scaler.update()

    @staticmethod
    def _normalize(adv: torch.Tensor, keep: torch.Tensor):
        """#### Normalize advantage function

        Statistics are taken over the steps that survive `skip_mask`.  A truncated step's
        advantage is never used, and it is left holding whatever the backward pass had in
        flight, so letting it into the mean and the standard deviation would move every
        other step's advantage for no reason.
        """
        kept = adv[keep]
        if kept.numel() == 0: return torch.zeros_like(adv)
        std = kept.std()
        tracker.add({'advantage_std': std})
        return (adv - kept.mean()) / (std + 1e-8)

    def _preprocess_samples(self, samples: Dict[str, torch.Tensor]):
        # $R_t$ returns sampled from $\pi_{\theta_{OLD}}$
        samples['returns'] = (samples['values'] + samples['advantages']).float()
        # V_C's target: the cost-to-go under GAE, a soft 0/1 label.  lambda = 1 makes it the
        # episode's actual outcome; below that it is a mixture with the critic's own
        # estimate, which is still a valid BCE target because it stays in [0, 1].
        samples['cost_returns'] = (samples['cost_values'] +
                                   samples['cost_advantages']).float().clamp(0., 1.)
        # A_t = A_R,t - beta * A_C,t, and it is normalized *after* combining.  Normalizing
        # the two separately would destroy the relative scaling beta exists to set, which is
        # the whole mechanism of the Lagrangian.
        keep = ~samples['skip_mask']
        # The two channels' scales, before they are combined.  beta is an exchange rate
        # between them, so what counts as a large beta is exactly adv_r_std / adv_c_std --
        # tracked because the combined `advantage_std` below cannot tell you either one.
        tracker.add({'adv_r_std': samples['advantages'][keep].std(),
                     'adv_c_std': samples['cost_advantages'][keep].std()})
        combined = samples['advantages'] - self.cur_beta * samples['cost_advantages']
        samples['advantages'] = Main._normalize(combined, keep)

    def _calc_loss(self, samples: Dict[str, torch.Tensor]) -> torch.Tensor:
        clip_range = self.c.clipping_range
        """## PPO Loss"""
        # Sampled observations are fed into the model to get $\pi_\theta(a_t|s_t)$ and $V^{\pi_\theta}(s_t)$;
        pi_logits, value = self.model_opt(samples['obs'])
        if self.supervised and self.cur_supervised_batch_size:
            sup_pi_logits = pi_logits[-self.cur_supervised_batch_size:]
            pi_logits = pi_logits[:-self.cur_supervised_batch_size]
            value = value[:,:-self.cur_supervised_batch_size]
        pi = Categorical(logits=pi_logits)
        cost_logit = value[1]
        value = value[0]

        with torch.no_grad():
            finite_mask = torch.isfinite(pi_logits)
            high_mask = torch.logical_and(finite_mask, pi.probs >= self.cur_low_prob_threshold)
            row_avg = torch.where(finite_mask, pi_logits, 0).sum(axis=1) / finite_mask.sum(axis=1)
            row_sign = torch.where(row_avg > 0, 1, -1).unsqueeze(1)
        skip_mask = samples['skip_mask']

        # #### Policy
        log_pi = pi.log_prob(samples['actions'])
        # *this is different from rewards* $r_t$.
        ratio = torch.exp(log_pi - samples['log_pis'])
        if self.c.use_kl:
            # reverse KL objective
            # Revisiting Design Choices in Proximal Policy Optimization
            beta = self.c.beta
            # reverse objective
            kl_div = kl_divergence(pi, Categorical(logits=samples['pi_logits'])).clamp(max=500)
            policy_reward = ratio * samples['advantages'] - beta * kl_div
        else:
            # CLIP objective
            # The ratio is clipped to be close to 1.
            # Using the normalized advantage
            #  $\bar{A_t} = \frac{\hat{A_t} - \mu(\hat{A_t})}{\sigma(\hat{A_t})}$
            #  introduces a bias to the policy gradient estimator,
            #  but it reduces variance a lot.
            clipped_ratio = ratio.clamp(min = 1.0 - clip_range,
                                        max = 1.0 + clip_range)
            # advantages are normalized
            policy_reward = torch.min(ratio * samples['advantages'],
                                      clipped_ratio * samples['advantages'])
            kl_div = .5 * ((samples['log_pis'] - log_pi) ** 2) # approximation

        policy_reward[skip_mask] = 0
        policy_reward = policy_reward.mean()

        # #### Entropy Bonus
        entropy_bonus = pi.entropy()
        entropy_bonus[skip_mask] = 0
        entropy_bonus = entropy_bonus.mean()

        # #### Revive low-probability actions
        high_prob_penalty = pi_logits.clone()
        high_prob_penalty[~high_mask] = 0
        high_prob_penalty = high_prob_penalty.mean()
        logit_value_penalty = pi_logits * row_sign
        logit_value_penalty[~finite_mask] = 0
        logit_value_penalty = logit_value_penalty.mean()

        # #### Value -- the score channel
        # Clipping makes sure the value function $V_\theta$ doesn't deviate
        #  significantly from $V_{\theta_{OLD}}$.
        clipped_value = samples['values']
        clipped_value += (value - samples['values']).clamp(min=-clip_range, max=clip_range)
        vf_loss = torch.max((value - samples['returns']) ** 2,
                            (clipped_value - samples['returns']) ** 2)
        vf_loss[skip_mask] = 0
        vf_loss = 0.5 * vf_loss.mean()

        # #### Value -- the constraint channel
        # V_C predicts P(top out before the milestone), a Bernoulli outcome, so it is a
        # sigmoid head fitted with BCE rather than an MSE regression.  At low c the policy
        # operates right at the boundary, and that is exactly where a squared-error head is
        # worst calibrated.
        cost_vf_loss = F.binary_cross_entropy_with_logits(
            cost_logit, samples['cost_returns'], reduction='none')
        cost_vf_loss[skip_mask] = 0
        cost_vf_loss = cost_vf_loss.mean()

        # #### supervised
        # F.cross_entropy does not deal with -inf properly
        def cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            log_probs = torch.log_softmax(logits, dim=1)
            log_probs = torch.where(torch.isinf(logits), torch.zeros_like(log_probs), log_probs)
            loss_per_sample = -((target + self.cur_supervised_smooth) * log_probs).sum(dim=1)
            return loss_per_sample.mean()
        if self.supervised and self.cur_supervised_batch_size:
            supervised_loss = cross_entropy(sup_pi_logits, samples['supervised_y'])
        else:
            supervised_loss = 0

        # we want to maximize $\mathcal{L}^{CLIP+VF+EB}(\theta)$
        # so we take the negative of it as the loss
        loss = (
            -self.cur_policy_weight * policy_reward
            -self.cur_entropy_weight * entropy_bonus
            +self.cur_low_prob_weight * high_prob_penalty
            +self.cur_vf_weight * vf_loss
            +self.cur_cost_vf_weight * cost_vf_loss
            +self.cur_supervised_weight * supervised_loss
        )

        # for monitoring
        clip_fraction = (abs((ratio - 1.0)) > clip_range).to(torch.float).mean()
        tracker.add({'policy_reward': policy_reward,
                     'vf_loss': vf_loss ** 0.5,
                     'cost_vf_loss': cost_vf_loss,
                     'v_cost': torch.sigmoid(cost_logit).mean(),
                     'supervised_loss': supervised_loss,
                     'entropy_bonus': entropy_bonus,
                     'kl_div': kl_div.mean(),
                     'clip_fraction': clip_fraction,
                     'median_prob': torch.median(pi.probs[finite_mask]),
                     'logits_mean': logit_value_penalty})
        return loss

    def run_training_loop(self):
        """### Run training loop"""
        offset = tracker.get_global_step()
        self.set_beta_kl(self.get_beta_kl(offset))
        self.set_phi_coef(self.get_phi_coef(offset))
        if offset > 1:
            # If resumed, sample several iterations first to reduce sampling bias
            self.generator.SendModel(self.model_opt, offset)
            for i in range(self.c.warmup_epochs): self.generator.StartGenerate(offset)
            tracker.save() # increment step
        else:
            self.generator.StartGenerate(offset)
        try:
            for _ in monit.loop(self.c.updates - offset):
                if self.c.time_limit > 0 and time.time() - start_time >= self.c.time_limit: break
                epoch = tracker.get_global_step()
                # sample with current policy
                samples, info = self.generator.GetData(device)
                self.generator.StartGenerate(epoch, update=True)
                tracker.add(info)
                # one dual step per iteration, before the policy update that uses beta
                self.update_dual(info)
                # train the model
                self.train(samples)
                # write summary info to the writer, and log to the screen
                tracker.save()
                # update hyperparams
                self.set_optim(self.c.lr(), self.c.reg_l2())
                self.set_game_params(self.get_game_params())
                # safe to send only between rollouts: mid-rollout the generator is blocking
                # on the weight-sync recv and would take this for an update_model message
                self.set_beta_kl(self.get_beta_kl(epoch + 1))
                self.set_phi_coef(self.get_phi_coef(epoch + 1))
                self.set_weight_params()
                if (epoch + 1) % 25 == 0: logger.log()
                if (epoch + 1) % self.c.save_interval == 0: experiment.save_checkpoint()
        except KeyboardInterrupt:
            pass
        finally:
            experiment.save_checkpoint()


def claim_experiment(uuid: str):
    """Claim the run on the labml web UI, which is where the dynamic hyperparameters are
    steered from.  Best effort: a run without network access is still a valid run, and
    losing the web UI is not a reason to lose the training."""
    from urllib.request import Request, urlopen
    from urllib.parse import urlsplit, urlunsplit
    try:
        time.sleep(2)
        url = labml.lab.get_info()['configs']['web_api']
        scheme, host, _, _, _ = urlsplit(url)
        url = urlunsplit((scheme, host, f'/api/v1/run/{uuid}/claim', '', ''))
        urlopen(Request(url, method='PUT'), timeout=10)
    except Exception as e:
        print('could not claim the run on the labml web UI ({}); '
              'dynamic hyperparameters stay at their command-line values'.format(e))


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    # distribution argument validation defaults to on and costs a full pass over the logits
    # (plus a support check on every sample) in the loss, for no benefit here
    torch.distributions.Distribution.set_default_validate_args(False)
    conf, args, _ = LoadConfig()
    # Section 9's controls: with a seed given, the weight init, the action sampling and the
    # env's piece streams are all pinned, so two runs that differ only in `c` really do
    # differ only in `c`.
    if conf.seed:
        torch.manual_seed(conf.seed)
        random.seed(conf.seed)
    m = Main(conf, args['name'])
    experiment.add_model_savers({
            'model': TorchSaver('model', m.model),
            'scaler': TorchSaver('scaler', m.scaler),
            'optimizer': TorchSaver('optimizer', m.optimizer, not args['ignore_optimizer']),
            # beta and its EMA: resuming with the dual reset to 0 throws away everything the
            # constraint had learned
            'dual': TorchSaver('dual', m.dual),
        })
    if len(args['uuid']):
        try:
            uuid = '{}-{:03d}'.format(args['name'], int(args['uuid']))
        except ValueError:
            uuid = args['uuid']
            if uuid == 'last':
                nd = MaxUUID(args['name'])
                if nd >= 1:
                    uuid = '{}-{:03d}'.format(args['name'], nd)
        experiment.load(uuid, args['checkpoint'])
    with experiment.start():
        claim_experiment(experiment.get_uuid())
        try: m.run_training_loop()
        except Exception as e: print(traceback.format_exc())
        finally: m.destroy()
