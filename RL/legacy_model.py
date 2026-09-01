"""The pre-transformer residual tower -- kept because the BetaTetris teacher *is* one.

`model.py` is a transformer now, and the published BetaTetris checkpoints
(`model-v1.0.0-*.pth`, 8 + 8 blocks of 256 channels, 20.8M parameters) are not loadable
into it: different modules, different tensor names, different shapes.  Distillation needs
the teacher at full strength, so the architecture it was trained as lives on here, used for
exactly one thing -- `distillation/teacher.py` loading a downloaded checkpoint.

Nothing else should import this.  New models are `model.Model`; this file is frozen at what
the released weights were written with, and the observation constants come from `model` so
the two cannot drift apart.
"""

import re

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from model import (AUTOCAST, ONNX_EXPORT, device, kBoardShape, kH, kMetaShape, kMovesShape,
                   kMoveMetaShape, kMoveStart, kR, kW)
from torch import autocast


class BatchNorm2dCast(nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if ONNX_EXPORT:
            self.num_batches_tracked = None

    def forward(self, x):
        if ONNX_EXPORT:
            return super().forward(x.float())
        else:
            return super().forward(x)


class ConvBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.main = nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                BatchNorm2dCast(ch),
                nn.ReLU(True),
                nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                BatchNorm2dCast(ch),
                )
        self.final = nn.ReLU(True)
    def forward(self, x):
        return self.final(self.main(x) + x)


class InitialEmbed(nn.Module):
    def __init__(self, feats, meta_feats, channels):
        super().__init__()
        self.embed_1 = nn.Conv2d(feats, channels, 5, padding=2)
        self.embed_2 = nn.Conv2d(feats, channels, (kH, 1))
        self.embed_3 = nn.Conv2d(feats, channels, (1, kW))
        self.meta = nn.Linear(meta_feats, channels)
        self.finish = nn.Sequential(
            BatchNorm2dCast(channels),
            nn.ReLU(True),
        )

    @AUTOCAST
    def forward(self, obs, meta):
        x_meta = self.meta(meta)
        x = self.embed_1(obs) + self.embed_2(obs) + self.embed_3(obs) + x_meta.view(*x_meta.shape, 1, 1)
        return self.finish(x)


class EvDev(nn.Module):
    def __init__(self, in_feat, ev_rank, dev_rank):
        super().__init__()
        self.linear_ev = nn.Linear(in_feat, ev_rank)
        self.linear_dev = nn.Linear(in_feat, dev_rank)
        self.ev_mat = torch.nn.Parameter(nn.Linear(ev_rank, 215).weight)
        self.dev_mat = torch.nn.Parameter(nn.Linear(dev_rank, 215).weight)
        self.ev_mat.requires_grad = True
        self.dev_mat.requires_grad = True

    @autocast(device_type=device, enabled=False)
    def evdev_coeff(self, obs):
        obs = obs.float()
        return self.linear_ev(obs), self.linear_dev(obs)

    @autocast(device_type=device, enabled=False)
    def forward(self, obs, idx):
        obs = obs.float()
        idx = torch.clamp(idx, max=214)
        ev = (self.linear_ev(obs) * self.ev_mat[idx]).sum(axis=1)
        dev = (self.linear_dev(obs) * self.dev_mat[idx]).sum(axis=1)
        return torch.stack([ev, dev])


class PiValueHead(nn.Module):
    def __init__(self, in_feat):
        super().__init__()
        self.linear = nn.Linear(in_feat, 1)

    @autocast(device_type=device, enabled=False)
    def forward(self, pi, value, invalid, multiplier):
        pi = torch.where(invalid, -float('inf'), pi.float())
        v = self.linear(value.float())
        v = v.transpose(0, 1) * multiplier
        return pi, v

class Model(nn.Module):
    """Two-stage residual conv tower with the policy and value heads sharing the trunk.

    Depth is `start_blocks + end_blocks`; everything else is a width.  The three head
    channel counts and the two head hidden sizes default to what every checkpoint written
    before they were configurable was trained with, so those still load unchanged --
    `infer_model_args` reads all of them back off a state dict, which is how the tools
    rebuild a model whose architecture they were not told.

    Widening the heads is the cheap way to add capacity: they are 1x1 convolutions over a
    20x10 grid followed by one linear, so their cost is negligible next to the trunk's
    3x3 convolutions, and the 8/2/1-channel squeezes are otherwise the tightest
    bottleneck in the network.
    """

    def __init__(self, start_blocks, end_blocks, channels,
                 pi_head_channels=8, evdev_head_channels=2, value_head_channels=1,
                 evdev_hidden=512, value_hidden=256):
        super().__init__()
        self.board_embed = InitialEmbed(kBoardShape[0], kMetaShape[0], channels)
        self.moves_embed = InitialEmbed(kMovesShape[0], kMoveMetaShape[0], channels)
        self.main_start = nn.Sequential(*[ConvBlock(channels) for i in range(start_blocks)])
        self.main_end = nn.Sequential(*[ConvBlock(channels) for i in range(end_blocks)])
        self.pi_logits_head = nn.Sequential(
            nn.Conv2d(channels, pi_head_channels, 1, bias=False),
            BatchNorm2dCast(pi_head_channels),
            nn.Flatten(),
            nn.ReLU(True),
            nn.Linear(pi_head_channels * kH * kW, kR * kH * kW)
        )
        self.evdev_head = nn.Sequential(
            nn.Conv2d(channels, evdev_head_channels, 1, bias=False),
            BatchNorm2dCast(evdev_head_channels),
            nn.Flatten(),
            nn.ReLU(True),
            nn.Linear(evdev_head_channels * kH * kW, evdev_hidden),
            nn.ReLU(True),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, value_head_channels, 1, bias=False),
            BatchNorm2dCast(value_head_channels),
            nn.Flatten(),
            nn.ReLU(True),
            nn.Linear(value_head_channels * kH * kW, value_hidden),
            nn.ReLU(True),
        )
        self.evdev_final = EvDev(evdev_hidden, 48, 32)
        self.pi_value_final = PiValueHead(value_hidden)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def get_action(self, obs, action_mask=None, deterministic=False):
        """Run the policy on `obs` and pick an action.

        `forward` already masks every placement the move search rejected, so `action_mask`
        (True = allowed, shape (batch, kR * kH * kW)) is only for narrowing the choice
        further; pass None for the plain masked policy.  Returns
        `(action, log_prob, value, entropy)`, where `value` is the scalar state value --
        row 0 of the three-row value tensor `forward` returns, the other two rows being the
        raw-score Normal the PPO trainer fits.
        """
        # pi_only skips the evdev head, which nothing here reads; the value row still comes out
        pi_logits, value = self(obs, pi_only=True)
        pi_logits = pi_logits.float()
        if action_mask is not None:
            action_mask = torch.as_tensor(action_mask, device=pi_logits.device).bool()
            pi_logits = torch.where(action_mask, pi_logits, -float('inf'))
        # a row with nothing left is a game the environment is about to end anyway; sampling
        # from it would give NaN probabilities, so let it pick uniformly and be rejected
        dead = ~torch.isfinite(pi_logits).any(dim=1, keepdim=True)
        pi_logits = torch.where(dead, torch.zeros_like(pi_logits), pi_logits)
        pi = Categorical(logits=pi_logits)
        action = torch.argmax(pi_logits, dim=1) if deterministic else pi.sample()
        return action, pi.log_prob(action), value[0], pi.entropy()

    @AUTOCAST
    def evdev_coeff(self, board, board_meta):
        batch = board.shape[0]
        x = self.board_embed(board, board_meta)
        x = self.main_start(x)
        x = self.evdev_head(x)
        return self.evdev_final.evdev_coeff(x)

    @AUTOCAST
    def forward(self, obs, categorical=False, pi_only=False, evdev_only=False, onnx=False):
        assert not (pi_only and evdev_only)
        board, board_meta, moves, moves_meta, meta_int = obs
        batch = board.shape[0]
        pi = None
        v = torch.zeros((1, batch), dtype=torch.float32, device=board.device)
        evdev = torch.zeros((2, batch), dtype=torch.float32, device=board.device)

        # the mask is cheapest to take before widening the uint8 planes
        invalid = moves[:,kMoveStart:kMoveStart+kR].reshape(batch, -1) == 0
        # board/move planes arrive as uint8 (see tetris/state.h); widen them on device
        if board.dtype != torch.float32: board = board.to(torch.float32)
        if moves.dtype != torch.float32: moves = moves.to(torch.float32)
        entry = meta_int[:,0].long()

        x = self.board_embed(board, board_meta)
        x = self.main_start(x)
        if kR == 1:
            x = x + self.moves_embed(moves, moves_meta)
            x = self.main_end(x)
            if not pi_only:
                evdev = self.evdev_final(self.evdev_head(x), entry)
        else:
            if not pi_only:
                evdev = self.evdev_final(self.evdev_head(x), entry)
            if not evdev_only:
                x = x + self.moves_embed(moves, moves_meta)
                x = self.main_end(x)
        if not evdev_only:
            pi, v = self.pi_value_final(
                self.pi_logits_head(x),
                self.value_head(x),
                invalid,
                torch.exp(moves_meta[:,-1]) if kR == 1 else 1
            )
            if categorical: pi = Categorical(logits=pi)
        v = torch.concat([v, evdev])
        if onnx:
            pi_rank = torch.argsort(pi, dim=1, descending=True)
            pi = F.softmax(pi, dim=1)
            return pi, pi_rank, v
        return pi, v

def infer_model_args(state_dict):
    """Recover the `Model` constructor arguments a checkpoint was written with.

    Checkpoints carry no architecture metadata, so every consumer that loads one it did not
    train (tools/, demo/, distillation/) has to read the shapes back out of the tensors.
    """
    return {
        'start_blocks': len([0 for i in state_dict if re.fullmatch(r'main_start.*main\.0\.weight', i)]),
        'end_blocks': len([0 for i in state_dict if re.fullmatch(r'main_end.*main\.0\.weight', i)]),
        'channels': state_dict['main_start.0.main.0.weight'].shape[0],
        'pi_head_channels': state_dict['pi_logits_head.0.weight'].shape[0],
        'evdev_head_channels': state_dict['evdev_head.0.weight'].shape[0],
        'value_head_channels': state_dict['value_head.0.weight'].shape[0],
        'evdev_hidden': state_dict['evdev_head.4.weight'].shape[0],
        'value_hidden': state_dict['value_head.4.weight'].shape[0],
    }


def load_model(path, device=None, eval_mode=True):
    """Build the `Model` a checkpoint describes and load it."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state_dict = torch.load(path, weights_only=True, map_location=device)
    model = Model(**infer_model_args(state_dict)).to(device)
    model.load_state_dict(state_dict)
    if eval_mode: model.eval()
    return model
