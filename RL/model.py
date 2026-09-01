import os

import torch
from torch import autocast, nn
from torch.distributions import Categorical
from torch.nn import functional as F

from training_env import tetris

ONNX_EXPORT = os.environ.get('ONNX_EXPORT') == '1'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
AUTOCAST = (lambda x: x) if ONNX_EXPORT else autocast(device_type=device)
kBoardShape, kMetaShape, kMovesShape, kMoveMetaShape, _ = tetris.Tetris.StateShapes()
kR = 1 if tetris.Tetris.IsNoro() else 4
kMoveStart = 2 if tetris.Tetris.IsNoro() else 14
kCurrentPieceStart = 5 if tetris.Tetris.IsNoro() else 0
kNextPieceStart = 12 if tetris.Tetris.IsNoro() else 7
kH, kW = kBoardShape[1:]
kActionSize = kR * kH * kW

kPlanes = kBoardShape[0] + kMovesShape[0]


class TransformerBlock(nn.Module):


    def __init__(self, d_model, heads, mlp_ratio):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.attn_norm = nn.RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.mlp_norm = nn.RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model)
        )

    def forward(self, x):
        normed = self.attn_norm(x)
        x = x + self.attn(normed, normed, normed, need_weights=False)[0]
        return x + self.mlp(self.mlp_norm(x))


class Model(nn.Module):

    def __init__(self, d_model=256, layers=6, heads=8, mlp_ratio=4):
        super().__init__()
        if d_model % heads:
            raise ValueError('d_model must be divisible by heads')
        self.d_model, self.layers, self.heads = d_model, layers, heads
        self.mlp_ratio = mlp_ratio
        self.column_projection = nn.Linear(kPlanes * kH, d_model)
        self.row_projection = nn.Linear(kPlanes * kW, d_model)
        self.column_embedding = nn.Parameter(torch.empty(kW, d_model))
        self.row_embedding = nn.Parameter(torch.empty(kH, d_model))
        self.piece_embedding = nn.Embedding(8, d_model)
        self.context_projection = nn.Linear(kMetaShape[0] + kMoveMetaShape[0], d_model)
        self.blocks = nn.ModuleList(
            TransformerBlock(d_model, heads, mlp_ratio) for _ in range(layers)
        )
        self.final_norm = nn.RMSNorm(d_model)

        self.policy_column = nn.Linear(d_model, kR * kH)
        self.policy_global = nn.Linear(d_model, kActionSize)
        self.value_head = nn.Linear(d_model, 1)

        self.cost_head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.cost_head.weight)
        nn.init.zeros_(self.cost_head.bias)
        nn.init.normal_(self.column_embedding, std=0.02)
        nn.init.normal_(self.row_embedding, std=0.02)
        nn.init.normal_(self.piece_embedding.weight, std=0.02)
        nn.init.zeros_(self.policy_global.weight)
        nn.init.zeros_(self.policy_global.bias)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def get_action(self, obs, action_mask=None, deterministic=False):

        pi_logits, value = self(obs, pi_only=True)
        pi_logits = pi_logits.float()
        if action_mask is not None:
            action_mask = torch.as_tensor(action_mask, device=pi_logits.device).bool()
            pi_logits = torch.where(action_mask, pi_logits, -float('inf'))

        dead = ~torch.isfinite(pi_logits).any(dim=1, keepdim=True)
        pi_logits = torch.where(dead, torch.zeros_like(pi_logits), pi_logits)
        pi = Categorical(logits=pi_logits)
        action = torch.argmax(pi_logits, dim=1) if deterministic else pi.sample()
        return action, pi.log_prob(action), value[0], pi.entropy()

    @staticmethod
    def _piece_id(one_hot):
        piece = one_hot.argmax(dim=1)
        return torch.where(one_hot.sum(dim=1) > 0, piece, torch.full_like(piece, 7))

    def _tokens(self, board, meta, moves, move_meta):
        batch = board.shape[0]
        planes = torch.cat((board, moves), dim=1).to(self.column_projection.weight.dtype)
        columns = planes.permute(0, 3, 1, 2).reshape(batch, kW, -1)
        rows = planes.permute(0, 2, 1, 3).reshape(batch, kH, -1)
        columns = self.column_projection(columns) + self.column_embedding
        rows = self.row_projection(rows) + self.row_embedding
        current_one_hot = meta[:, kCurrentPieceStart:kCurrentPieceStart + 7]
        next_one_hot = meta[:, kNextPieceStart:kNextPieceStart + 7]
        current = self.piece_embedding(self._piece_id(current_one_hot)).unsqueeze(1)
        next_piece = self.piece_embedding(self._piece_id(next_one_hot)).unsqueeze(1)
        context = self.context_projection(torch.cat((meta, move_meta), dim=1)).unsqueeze(1)
        return torch.cat((columns, rows, current, next_piece, context), dim=1)

    @AUTOCAST
    def forward(self, obs, categorical=False, pi_only=False, evdev_only=False, onnx=False):
        if pi_only and evdev_only:
            raise ValueError('pi_only and evdev_only are mutually exclusive')
        board, meta, moves, move_meta, _ = obs
        batch = board.shape[0]
        x = self._tokens(board, meta, moves, move_meta)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        pooled = x[:, -1]

        pi = None
        if not evdev_only:

            logits = self.policy_column(x[:, :kW]).view(batch, kW, kR, kH)
            logits = logits.permute(0, 2, 3, 1).reshape(batch, kActionSize)
            logits = logits + self.policy_global(pooled)
            invalid = moves[:, kMoveStart:kMoveStart + kR].reshape(batch, -1) == 0
            pi = logits.float().masked_fill(invalid, -float('inf'))
            if categorical:
                pi = Categorical(logits=pi)

        value = torch.cat((self.value_head(pooled), self.cost_head(pooled)),
                          dim=1).float().transpose(0, 1)
        if onnx:
            pi_rank = torch.argsort(pi, dim=1, descending=True)
            return F.softmax(pi, dim=1), pi_rank, value
        return pi, value


kDefaultHeadDim = 32
MODEL_ARG_NAMES = ('d_model', 'layers', 'heads', 'mlp_ratio')


def model_state_dict(obj):

    if isinstance(obj, dict) and 'model' in obj and isinstance(obj['model'], dict):
        return obj['model']
    return obj


def infer_model_args(obj, heads=None):

    if isinstance(obj, dict) and isinstance(obj.get('model_args'), dict):
        args = {k: obj['model_args'][k] for k in MODEL_ARG_NAMES}
        if heads is not None: args['heads'] = heads
        return args
    state_dict = model_state_dict(obj)
    d_model = state_dict['column_embedding'].shape[1]
    return {
        'd_model': d_model,
        'layers': len({key.split('.')[1] for key in state_dict if key.startswith('blocks.')}),
        'heads': max(1, d_model // kDefaultHeadDim) if heads is None else heads,
        'mlp_ratio': state_dict['blocks.0.mlp.0.weight'].shape[0] / d_model,
    }


def adapt_state_dict(state_dict, model):

    state_dict = dict(state_dict)
    own = model.state_dict()
    changed = []
    for key in ('value_head.weight', 'value_head.bias'):
        if key in state_dict and state_dict[key].shape != own[key].shape:
            if state_dict[key].shape[0] < own[key].shape[0]:
                raise RuntimeError('{} has {} value rows, need {}'.format(
                    key, state_dict[key].shape[0], own[key].shape[0]))
            state_dict[key] = state_dict[key][:own[key].shape[0]].clone()
            changed.append(key)
    for key in ('cost_head.weight', 'cost_head.bias'):
        if key not in state_dict:
            state_dict[key] = own[key].clone()
            changed.append(key)
    if changed:
        print('checkpoint predates the cost critic; adapted {} (V_C starts at 0.5)'.format(
            ', '.join(changed)))
    return state_dict


def model_from_state_dict(state_dict, heads=None):
    return Model(**infer_model_args(state_dict, heads=heads))


def load_model(path, device=None, eval_mode=True, heads=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    obj = torch.load(path, weights_only=True, map_location=device)
    model = model_from_state_dict(obj, heads=heads).to(device)
    model.load_state_dict(adapt_state_dict(model_state_dict(obj), model))
    if eval_mode: model.eval()
    return model


def save_model(model, path, **extra):

    model = getattr(model, '_orig_mod', model)  # unwrap torch.compile
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    args = {'d_model': model.d_model, 'layers': model.layers, 'heads': model.heads,
            'mlp_ratio': model.mlp_ratio}
    torch.save({'model': state_dict, 'model_args': args, **extra}, str(path))
    return path


def cost_prob(value):
    return torch.sigmoid(value[1])


def obs_to_torch(obs, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if len(obs[0].shape) == 3:
        return [torch.as_tensor(i, device=device).unsqueeze(0) for i in obs]
    return [torch.as_tensor(i, device=device) for i in obs]
