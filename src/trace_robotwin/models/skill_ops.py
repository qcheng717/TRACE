
import math

import torch

from torch import nn

from torch.nn import functional as F



class SkillEncoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        d = c['skill_encoder_dim']
        self.query = nn.Linear(c['semantic_dim'], d, bias=False)
        self.key = nn.Linear(c['dictionary_dim'], d, bias=False)
        self.scale = math.sqrt(d)*c['skill_encoder_temperature']

    def forward(self, semantic, dictionary, rescue=False):
        keys = self.key(dictionary)
        if rescue: keys = keys.detach()  # rescue cannot reshape dictionary or key projection
        return (self.query(semantic) @ keys.T / self.scale).float().softmax(-1)

def masked_mean(error, mask, denominator=None):
    """error has one scalar per timestep. DDP/accumulation may supply global denominator."""
    if denominator is None: denominator = mask.sum()
    return torch.where(mask, error, torch.zeros_like(error)).sum()/torch.as_tensor(denominator, device=error.device).clamp_min(1)

def orthogonal_loss(d):
    gram = d @ d.T
    mask = ~torch.eye(len(d), dtype=torch.bool, device=d.device)
    return gram[mask].square().mean() if len(d)>1 else gram.sum()*0

def rescue_loss(probabilities, usage, min_ratio, reference=None):
    # Global per-microbatch differentiable mean; usage itself is full-update EMA.
    p = probabilities.reshape(-1, probabilities.shape[-1])
    total = p.sum(0); count = p.new_tensor(len(p))
    if torch.distributed.is_initialized():
        from torch.distributed.nn.functional import all_reduce
        total = all_reduce(total); torch.distributed.all_reduce(count)
    deficit = (min_ratio/len(usage)-usage.detach()).clamp_min(0)
    weights = deficit/deficit.sum().clamp_min(1e-12)
    mean = total/count
    if reference is not None:
        # Two-pass accumulation: full-update value, microbatch derivative.
        mean = reference.detach()+(mean-mean.detach())
    return -(weights*mean.clamp_min(1e-12).log()).sum()

@torch.no_grad()
def geometry(d):
    d = F.normalize(d.float(), dim=-1)
    g = d @ d.transpose(-1, -2); k = d.shape[-2]
    mask = ~torch.eye(k, dtype=torch.bool, device=d.device)
    singular = torch.linalg.svdvals(d); p = singular/singular.sum(-1, keepdim=True).clamp_min(1e-12)
    return dict(effective_rank=(-(p*p.clamp_min(1e-12).log()).sum(-1)).exp().mean(),
                mean_abs_offdiag=g[..., mask].abs().mean(), max_abs_offdiag=g[..., mask].abs().max())

