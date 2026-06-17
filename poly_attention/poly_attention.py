# Chakrabarti et al. https://arxiv.org/abs/2602.02422

from functools import partial

import torch
from torch import nn, einsum, stack
from torch.nn import Module, RMSNorm
import torch.nn.functional as F

import einx
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# constants

LinearNoBias = partial(nn.Linear, bias = False)

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(num, den):
    return (num % den) == 0

def softclamp(x, c):
    return c * torch.tanh(x / c)

# poly attention

class PolyAttention(Module):
    def __init__(
        self,
        dim,
        heads = 8,
        kv_heads = None,
        dim_head = 64,
        causal = False,
        softclamp_value = 20.,
        eps = 1e-9
    ):
        super().__init__()
        self.eps = eps
        self.scale = dim_head ** -0.5

        kv_heads = default(kv_heads, heads)
        assert divisible_by(heads, kv_heads), 'heads must be divisible by kv_heads'

        self.heads = heads
        self.kv_heads = kv_heads

        dim_inner = dim_head * heads
        dim_inner_kv = dim_head * kv_heads

        self.causal = causal
        self.softclamp_value = softclamp_value

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.split_kv_heads = Rearrange('b n (h d) -> b h n d', h = kv_heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.is_gqa = heads != kv_heads

        if self.is_gqa:
            self.num_rep = heads // kv_heads

        self.to_q_gates = LinearNoBias(dim, dim_inner * 2)
        self.to_kv = LinearNoBias(dim, dim_inner_kv * 4)

        self.q1_norm = RMSNorm(dim_head)
        self.q2_norm = RMSNorm(dim_head)
        self.q3_norm = RMSNorm(dim_head)

        self.to_out = nn.Linear(dim_inner, dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # small initialization is crucial to prevent exp() overflow
        nn.init.normal_(self.to_q_gates.weight, mean = 0.0, std = 0.01)
        nn.init.normal_(self.to_kv.weight, mean = 0.0, std = 0.01)
        nn.init.normal_(self.to_out.weight, mean = 0.0, std = 0.01)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, x, mask = None):
        device = x.device

        q1, gates = map(self.split_heads, self.to_q_gates(x).chunk(2, dim = -1))
        q2, q3, v2, v3 = map(self.split_kv_heads, self.to_kv(x).chunk(4, dim = -1))

        q1 = self.q1_norm(q1)
        q2 = self.q2_norm(q2)
        q3 = self.q3_norm(q3)

        if self.is_gqa:
            q2, q3, v2, v3 = (repeat(t, 'b g n d -> b (g r) n d', r = self.num_rep) for t in (q2, q3, v2, v3))

        q_left = stack((q1, q2))
        q_right = stack((q2, q3))

        # unscaled exp values

        scores = einsum('... i d, ... j d -> ... i j', q_left, q_right) * self.scale
        scores = softclamp(scores, self.softclamp_value)

        exp_scores = scores.exp()

        # causal masking

        if self.causal:
            i, j = exp_scores.shape[-2:]
            causal_mask = torch.ones((i, j), device = device, dtype = torch.bool).triu(1)
            exp_scores = exp_scores.masked_fill(causal_mask, 0.)

        # padding masking

        if exists(mask):
            exp_scores = einx.where('b j, c b h i j, -> c b h i j', mask, exp_scores, 0.)

        exp_scores12, exp_scores23 = exp_scores

        # aggregate

        exp_scores23_v3 = einsum('b h i j, b h j d -> b h i d', exp_scores23, v3)
        unnormalized_out = einsum('b h i j, b h j d -> b h i d', exp_scores12, exp_scores23_v3)

        # elementwise multiply the root values (v2) with the aggregated messages from the rest of the tree

        out = v2 * unnormalized_out

        # normalize

        exp_scores23_sum = exp_scores23.sum(dim = -1, keepdim = True)

        denominator = einsum('b h i j, b h j d -> b h i d', exp_scores12, exp_scores23_sum)

        out = out / denominator.clamp_min(self.eps)

        # gate

        out = out * gates.sigmoid()

        # combine heads

        return self.to_out(self.merge_heads(out))
