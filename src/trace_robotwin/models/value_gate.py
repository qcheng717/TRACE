"""Parameter-compatible MHA with gating after the learned Value projection."""
import torch
from torch import nn
from torch.nn import functional as F


def router_gate(weights, beta, gamma):
    w = weights.float()
    gate = torch.exp(beta * torch.tanh(gamma * (w.shape[-1] * w - 1)))
    return gate / gate.mean(-1, keepdim=True).clamp_min(1e-6)


class SkillAttention(nn.MultiheadAttention):
    def forward(self, query, key, value, *, attn_mask=None, need_weights=False, value_gate=None):
        if value_gate is None:
            return super().forward(query, key, value, attn_mask=attn_mask, need_weights=need_weights)
        b, a, d = query.shape
        k = key.shape[1]; h = self.num_heads; dh = d // h
        wq, wk, wv = self.in_proj_weight.chunk(3)
        bq, bk, bv = self.in_proj_bias.chunk(3)
        def heads(x, n):
            return x.reshape(b, n, h, dh).transpose(1, 2)
        q = heads(F.linear(query, wq, bq), a)
        kk = heads(F.linear(key, wk, bk), k)
        v = heads(F.linear(value, wv, bv), k)
        # Gate the projected V, including its bias; do not modify K or normalize alpha*g.
        v = v * value_gate[:, None, :, None].to(v.dtype)
        mask = None if attn_mask is None else attn_mask.reshape(b, h, a, k)
        out = F.scaled_dot_product_attention(q, kk, v, attn_mask=mask, dropout_p=0.)
        out = self.out_proj(out.transpose(1, 2).reshape(b, a, d))
        return out, None
