import traceback
import numpy as np, torch
from torch.distributions import Categorical, kl_divergence
from torch.multiprocessing import Pipe
from model import Model, adapt_state_dict, infer_model_args, model_state_dict, kH, kW, kR

from training_env import tetris
from training_env import BatchedGames, adj_delay_ids_for, tap_ids_for

TORCH_DTYPE = {'uint8': torch.uint8, 'int32': torch.int32, 'float32': torch.float32}


class DataGenerator:
    """Rollout worker: steps the env, runs the policy, and computes both GAE channels.

    The trainer gets `A_R` and `A_C` separately and combines them itself as
    `A = A_R - beta * A_C` with whatever the dual variable is at that moment; combining
    here would freeze beta at the value it had when the rollout started.
    """

    def __init__(self, name, model, c, game_params):
        self.model = model
        self.envs = c.n_workers * c.env_per_worker
        self.worker_steps = c.worker_steps
        self.gamma, self.lamda = game_params[-2:]
        self.device = next(self.model.parameters()).device
        self.game_params = None
        self.recv_weight_interval = c.worker_steps // c.weight_sync_per_epoch
        self.use_kl = c.use_kl

        # Frozen distilled policy for the KL penalty (Configs.kl_distill_file): the shaped
        # reward of every step is reduced by beta_kl * KL(current || distilled) before GAE,
        # which keeps fine-tuning from walking away from what distillation taught.  beta_kl
        # is pushed from the trainer once per epoch and annealed to 0, at which point the
        # extra forward pass per step stops.
        #
        # This is *not* potential-based, so while it is on it biases the frontier being
        # measured (spec section 5 admits only policy-invariant shaping).  It is off by
        # default and annealed to zero; use --phi-model-file for shaping that is safe.
        self.ref_model = None
        self.beta_kl = 0.
        if c.kl_distill_file:
            self.ref_model = self._load_frozen(c.kl_distill_file)

        # Potential-based shaping (spec section 5).  F_t = gamma * Phi(s_{t+1}) - Phi(s_t)
        # with Phi(terminal) = 0, which is policy-invariant for any Phi, so it cannot move
        # the frontier -- provided the terminal condition holds and it is annealed away.
        # Phi is the frozen distilled value head: a learned function of the raw board, never
        # a hand-written feature combination.
        self.phi_model = None
        self.phi_coef = 0.
        self.phi_scale = c.phi_scale
        if c.phi_model_file:
            self.phi_model = self._load_frozen(c.phi_model_file)

        shapes = tetris.Tetris.StateShapes()
        types = tetris.Tetris.StateTypes()
        # Rollout observations live on the host and are filled in place by the C++ batch:
        # slot t holds the observation the policy acts on at step t, and slot worker_steps
        # holds the bootstrap observation.  Shared memory so handing them to the trainer is
        # a mapping, not a copy.
        self.obs_t = [
            torch.empty((self.worker_steps + 1, self.envs, *shape), dtype=TORCH_DTYPE[typ]).share_memory_()
            for shape, typ in zip(shapes, types)
        ]
        self.obs_np = [i.numpy() for i in self.obs_t]
        # device-side copy of the step currently being acted on
        self.cur_obs = [
            torch.empty((self.envs, *shape), dtype=TORCH_DTYPE[typ], device=self.device)
            for shape, typ in zip(shapes, types)
        ]
        self.actions_np = np.zeros(self.envs, dtype='int32')
        self.actions_cpu = torch.from_numpy(self.actions_np)

        # Level-18 starts to the milestone at 230 lines.  `eval_ratio` reserves envs that
        # always play that episode even when the training reset distribution is randomized,
        # so p_hat is always measured on the distribution `c` constrains.
        self.games = BatchedGames(self.envs, self.worker_steps, self.obs_np,
                                  num_threads=c.env_threads, board_file=c.board_file,
                                  segment=c.segment, tap_ids=tap_ids_for(c.tap_speed),
                                  adj_delay_ids=adj_delay_ids_for(c.adj_delay),
                                  milestone=c.milestone, eval_ratio=c.eval_ratio,
                                  seed=c.seed or None)
        self.set_params(game_params)
        self.games.reset_all()

        # rollout tensors that the trainer consumes; allocated once and reused
        self.out = {
            'actions': torch.empty((self.worker_steps, self.envs), dtype=torch.int32).share_memory_(),
            'log_pis': torch.empty((self.worker_steps, self.envs)).share_memory_(),
            'skip_mask': torch.empty((self.worker_steps, self.envs), dtype=torch.bool).share_memory_(),
            'values': torch.empty((self.worker_steps, self.envs)).share_memory_(),
            'advantages': torch.empty((self.worker_steps, self.envs)).share_memory_(),
            # constraint channel: V_C as a probability, and its GAE
            'cost_values': torch.empty((self.worker_steps, self.envs)).share_memory_(),
            'cost_advantages': torch.empty((self.worker_steps, self.envs)).share_memory_(),
        }
        if self.use_kl:
            self.out['pi_logits'] = torch.empty(
                (self.worker_steps, self.envs, kR * kH * kW)).share_memory_()
        self.flat_obs = [i[:self.worker_steps].flatten(0, 1) for i in self.obs_t]

    def _load_frozen(self, path):
        ckpt = torch.load(path, weights_only=True, map_location=self.device)
        model = Model(**infer_model_args(ckpt)).to(self.device, memory_format=torch.channels_last)
        model.load_state_dict(adapt_state_dict(model_state_dict(ckpt), model))
        model.eval()
        for p in model.parameters(): p.requires_grad_(False)
        return model

    def update_model(self, state_dict, epoch=-1):
        target_device = next(self.model.parameters()).device
        for i in state_dict:
            if state_dict[i].device != target_device:
                state_dict[i] = state_dict[i].to(target_device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def set_beta_kl(self, beta_kl):
        self.beta_kl = beta_kl

    def set_phi_coef(self, phi_coef):
        self.phi_coef = phi_coef

    def set_params(self, game_params):
        if self.game_params != game_params[2:]:
            self.games.set_params(game_params[2:])
            self.game_params = game_params[2:]
        self.gamma, self.lamda = game_params[:2]

    def _load_obs(self, step):
        for dst, src in zip(self.cur_obs, self.obs_t):
            dst.copy_(src[step])
        return self.cur_obs

    def sample(self, epoch=0, remote=None):
        """### Sample data with current policy"""
        device = self.device
        T, E = self.worker_steps, self.envs
        log_pis = torch.empty((T, E), dtype=torch.float32, device=device)
        # (t, 2, env): row 0 is V_R, row 1 is V_C read as a probability
        values = torch.empty((T, 2, E), dtype=torch.float32, device=device)
        actions = torch.empty((T, E), dtype=torch.int32, device=device)
        pi_logits = torch.empty((T, E, kR * kH * kW), dtype=torch.float32,
                                device=device) if self.use_kl else None
        penalize_ref = self.ref_model is not None and self.beta_kl > 0
        ref_kl = torch.empty((T, E), dtype=torch.float32, device=device) if penalize_ref else None
        shape_phi = self.phi_model is not None and self.phi_coef > 0
        # Phi at every step *and* at the bootstrap slot, so F_t has both ends.
        phi = torch.empty((T + 1, E), dtype=torch.float32, device=device) if shape_phi else None

        ret_info = {i: [] for i in ('ret', 'scorek', 'pace', 'lns', 'pcs', 'topout',
                                    'tetris_rate', 'burn', 'short_finish', 'truncated',
                                    'mil_games', 'ref_kl', 'phi_shaping')}

        n_true = n_topout = 0
        for t in range(T):
            with torch.inference_mode():
                pi, v = self.model(self._load_obs(t), categorical=True)
                values[t, 0] = v[0]
                values[t, 1] = torch.sigmoid(v[1])
                a = pi.sample()
                if self.use_kl: pi_logits[t] = pi.logits
                if ref_kl is not None:
                    # same observation, so the two policies carry the same -inf mask and the
                    # divergence is over the legal placements only
                    ref_logits = self.ref_model(self.cur_obs, pi_only=True)[0]
                    ref_kl[t] = kl_divergence(pi, Categorical(logits=ref_logits)).clamp(max=500)
                if phi is not None:
                    phi[t] = self.phi_model(self.cur_obs, pi_only=True)[1][0] * self.phi_scale
                actions[t] = a
                log_pis[t] = pi.log_prob(a)
                self.actions_cpu.copy_(a)

            step_true, step_topout = self.games.step(t, self.actions_np, ret_info)
            n_true += step_true
            n_topout += step_topout

            if remote and (t + 1) % self.recv_weight_interval == 0:
                cmd, data = remote.recv()
                assert cmd == 'update_model'
                self.update_model(data[0], epoch=epoch)

        rewards, is_over = self.games.rewards, self.games.is_over
        if ref_kl is not None: ret_info['ref_kl'].append(ref_kl.mean().item())
        ret_info['mil_games'].append(self.games.total_games * 1e-6)
        # p_hat for the dual update, and the sample size behind it.  Only true level-18
        # starts count; an iteration in which none finished leaves the EMA alone.
        if n_true > 0:
            ret_info['p_hat'] = [n_topout / n_true]
        ret_info['n_eps'] = [n_true]

        advantages, skip_mask = self._calc_advantages(is_over, rewards, values, phi, ref_kl,
                                                      ret_info)
        values_t = values.transpose(0, 1)
        advantages_t = advantages.transpose(0, 1)
        self.out['actions'].copy_(actions)
        self.out['log_pis'].copy_(log_pis)
        self.out['skip_mask'].copy_(skip_mask)
        self.out['values'].copy_(values_t[0])
        self.out['advantages'].copy_(advantages_t[0])
        self.out['cost_values'].copy_(values_t[1])
        self.out['cost_advantages'].copy_(advantages_t[1])
        if self.use_kl: self.out['pi_logits'].copy_(pi_logits)

        # The envs keep running across rollouts, so their current observation - the one in the
        # bootstrap slot - becomes step 0 of the next rollout.  (_calc_advantages has already
        # read the bootstrap slot by now.)
        for buf in self.obs_t: buf[0].copy_(buf[self.worker_steps])

        samples = {'obs': self.flat_obs}
        for k, v in self.out.items():
            samples[k] = v.flatten(0, 1) if v.dim() > 1 else v
        for i in list(ret_info):
            if ret_info[i]:
                ret_info[i] = np.mean(ret_info[i])
            else:
                del ret_info[i]
        return samples, ret_info

    def _calc_advantages(self, over: np.ndarray, rewards: np.ndarray, pred_values: torch.Tensor,
                         phi: torch.Tensor = None, ref_kl: torch.Tensor = None,
                         ret_info: dict = None):
        """GAE on both channels at once (spec sections 2, 3 and 8).

        Row 0 is the score channel, row 1 the cost channel.  Both run at gamma = 1: the
        episode is guaranteed finite, so the undiscounted return is well defined and equals
        exactly the quantity being measured.  Variance is traded off with lambda, which
        changes the estimator rather than the objective.

        Terminal handling is the part that has to be exactly right:

            TOP_OUT / MILESTONE   no bootstrap.  V_R target 0, V_C target 0 or 1 as the
                                  cost on that transition says.
            truncation            bootstrap both heads off the successor state.  The batch
                                  plays one extra step after the cut precisely so that
                                  successor exists; that step is then dropped from the loss.
        """
        with torch.no_grad():
            # (env, t, 2) -> (t, 2, env)
            rewards = torch.permute(torch.from_numpy(rewards).to(self.device), (1, 2, 0)).clone()
            # (env, t, 2) -> (2, t, env)
            over_torch = torch.permute(torch.from_numpy(over).to(self.device), (2, 1, 0))
            not_done = ~over_torch[0]
            truncated = over_torch[1]

            if ref_kl is not None: rewards[:, 0] -= self.beta_kl * ref_kl

            # $V(s_{t+1})$ for the last step: bootstrap off the observation in the extra
            # slot.  Phi of that same state closes the shaping telescope.
            bootstrap_obs = self._load_obs(self.worker_steps)
            last = self.model(bootstrap_obs)[1]
            last_value = torch.stack((last[0], torch.sigmoid(last[1]))).float()
            if phi is not None:
                phi[self.worker_steps] = (
                    self.phi_model(bootstrap_obs, pi_only=True)[1][0] * self.phi_scale)
                # F_t = gamma * Phi(s_{t+1}) - Phi(s_t), with Phi(terminal) = 0.  `not_done`
                # zeroes the successor term at every episode end, which is the condition the
                # whole invariance rests on -- get it wrong and the shaping stops being
                # policy-invariant, which is the single most common way people break PBRS.
                # Only the score channel is shaped; the constraint channel is a probability
                # and shaping it would change what V_C means.
                shaping = self.phi_coef * (self.gamma * phi[1:] * not_done - phi[:-1])
                rewards[:, 0] += shaping
                if ret_info is not None:
                    ret_info['phi_shaping'].append(shaping.abs().mean().item())

            advantages = torch.zeros((self.worker_steps, 2, self.envs), dtype=torch.float32,
                                     device=self.device)
            last_advantage = torch.zeros((2, self.envs), dtype=torch.float32, device=self.device)
            gammas = torch.full((2, 1), float(self.gamma), device=self.device)

            for t in reversed(range(self.worker_steps)):
                alive = not_done[t]
                # delta_t + gamma * lambda * A_{t+1}, folded into one expression
                ground = rewards[t] + alive * gammas * (last_value + self.lamda * last_advantage)
                last_advantage = ground - pred_values[t]
                advantages[t] = last_advantage
                # A cut is not a transition: break the chain here, and hand the predecessor
                # V(s_t) to bootstrap from.  The step itself is dropped by skip_mask.
                last_advantage = torch.where(truncated[t], torch.zeros_like(last_advantage),
                                             last_advantage)
                last_value = pred_values[t]
            return advantages, truncated

    def destroy(self):
        self.games.save_params()


def generator_process(remote, name, c, game_params, device):
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.distributions.Distribution.set_default_validate_args(False)
    if c.seed: torch.manual_seed(c.seed + 1)
    generator = None
    try:
        model = Model(**c.model_args()).to(device, memory_format=torch.channels_last)
        model = torch.compile(model)
        generator = DataGenerator(name, model, c, game_params)
        samples = None
        while True:
            cmd, data = remote.recv()
            if cmd == "update_model":
                generator.update_model(data[0], epoch=data[1])
            elif cmd == "set_param":
                generator.set_params(data)
            elif cmd == "set_beta_kl":
                generator.set_beta_kl(data)
            elif cmd == "set_phi_coef":
                generator.set_phi_coef(data)
            elif cmd == "start_generate":
                samples = generator.sample(remote=remote if data[0] else None, epoch=data[1])
            elif cmd == "get_data":
                remote.send(samples)
                samples = None
            elif cmd == "close":
                return
            else:
                raise NotImplementedError
    except:
        print(traceback.format_exc())
        raise
    finally:
        if generator is not None: generator.destroy()
        remote.close()


class GeneratorProcess:
    def __init__(self, model, *args):
        # args: name, c, game_params, device
        self.device = args[-1]
        self.child, parent = Pipe()
        ctx = torch.multiprocessing.get_context('spawn')
        self.process = ctx.Process(target=generator_process, args=(parent, *args))
        self.process.start()
        self.SendModel(model)

    def SendModel(self, model, epoch=-1):
        state_dict = model.state_dict()
        for i in state_dict:
            state_dict[i] = state_dict[i].cpu()
        self.child.send(('update_model', (state_dict, epoch)))

    def StartGenerate(self, epoch, update=False):
        self.child.send(('start_generate', (update, epoch)))

    def SetParams(self, game_params):
        self.child.send(('set_param', game_params))

    def SetBetaKL(self, beta_kl):
        self.child.send(('set_beta_kl', beta_kl))

    def SetPhiCoef(self, phi_coef):
        self.child.send(('set_phi_coef', phi_coef))

    def GetData(self, device):
        self.child.send(('get_data', None))
        data, info = self.child.recv()
        for i in data:
            if i == 'obs':
                data[i] = [j.to(device, non_blocking=True) for j in data[i]]
            else:
                data[i] = data[i].to(device, non_blocking=True)
        if device.type == 'cuda': torch.cuda.synchronize(device)
        return data, info

    def Close(self):
        self.child.send(('close', None))
