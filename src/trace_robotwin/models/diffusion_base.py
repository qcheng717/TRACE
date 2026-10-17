"""Token-conditioned action diffusion. Observations are causal; action denoising is bidirectional."""
import math
import torch
from torch import nn
from torch import Tensor

class DiffusionTransformer(nn.Module):
    def __init__(self, semantic_dim, dynamic_dim, language_dim, history_dim, proprio_dim,
                 skill_dim, queries, action_dim=7, action_horizon=8, diffusion_steps=100,
                 inference_steps=10, hidden_dim=256, layers=4, heads=8):
        nn.Module.__init__(self)
        if min(action_horizon,inference_steps,layers,heads)<1 or not 2<=diffusion_steps or inference_steps>diffusion_steps or hidden_dim%heads or hidden_dim%2:
            raise ValueError('invalid diffusion transformer dimensions/steps')
        self.action_dim=action_dim;self.action_horizon=action_horizon;self.steps=diffusion_steps
        self.inference_steps=inference_steps;self.queries=queries;self.proprio_dim=proprio_dim
        self.semantic_dim=semantic_dim;self.dynamic_dim=dynamic_dim
        dims=dict(semantic=semantic_dim,dynamic=dynamic_dim,skill=skill_dim,basis=skill_dim,summary=dynamic_dim,language=language_dim,history=history_dim,proprio=proprio_dim)
        self.projections=nn.ModuleDict({k:nn.Linear(v,hidden_dim) for k,v in dims.items()})
        self.types=nn.Parameter(torch.randn(8,hidden_dim)*.02)
        self.query_position=nn.Parameter(torch.randn(queries,hidden_dim)*.02)
        self.action_position=nn.Parameter(torch.randn(action_horizon,hidden_dim)*.02)
        self.action_input=nn.Linear(action_dim,hidden_dim)
        self.time=nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,hidden_dim))
        self.hidden_dim=hidden_dim
        block=nn.TransformerDecoderLayer(hidden_dim,heads,4*hidden_dim,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
        self.decoder=nn.TransformerDecoder(block,layers,norm=nn.LayerNorm(hidden_dim))
        # Avoid identical initialization of cloned blocks.
        for layer in self.decoder.layers:
            for p in layer.parameters():
                if p.ndim>1:nn.init.xavier_uniform_(p)
        self.head=nn.Linear(hidden_dim,action_dim)
        t=torch.linspace(0,1,diffusion_steps+1,dtype=torch.float64)
        cumulative=torch.cos((t+.008)/1.008*math.pi/2).square();cumulative=cumulative/cumulative[0]
        betas=(1-cumulative[1:]/cumulative[:-1]).clamp(max=.999).float()
        self.register_buffer('betas',betas);self.register_buffer('alpha_bar',(1-betas).cumprod(0))


    def add_noise(self, clean: Tensor, step: Tensor, noise: Tensor) -> Tensor:
        a=self.alpha_bar[step][...,None,None]
        return a.sqrt()*clean+(1-a).sqrt()*noise
