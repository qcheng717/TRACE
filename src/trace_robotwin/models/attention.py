from dataclasses import dataclass

import math

import torch

from torch import nn



from trace_robotwin.models.diffusion_base import DiffusionTransformer

class CrossBlock(nn.Module):
    def __init__(self,d,heads):
        super().__init__();self.qnorm=nn.LayerNorm(d);self.knorm=nn.LayerNorm(d)
        self.attn=nn.MultiheadAttention(d,heads,dropout=0,batch_first=True)
        self.norm=nn.LayerNorm(d);self.ff=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def forward(self,q,kv,mask=None):
        a,w=self.attn(self.qnorm(q),self.knorm(kv),self.knorm(kv),key_padding_mask=mask,need_weights=True)
        q=q+a;return q+self.ff(self.norm(q)),w

def encoder(d,heads):
    return nn.TransformerEncoderLayer(d,heads,4*d,dropout=0,activation='gelu',batch_first=True,norm_first=True)

@dataclass
class SkillCondition:
    skill: torch.Tensor
    language: torch.Tensor
    proprio: torch.Tensor
    valid: torch.Tensor

class SkillDiT(DiffusionTransformer):
    def __init__(self,skill_dim,language_dim,proprio_dim,h,c):
        super().__init__(1,1,language_dim,1,proprio_dim,skill_dim,1,c['action_dim'],c['action_horizon'],c['diffusion_steps'],c['inference_steps'],c['hidden_dim'],c['layers'],c['heads'])
        del self.projections,self.types,self.query_position
        d=c['hidden_dim'];self.skill_input=nn.Linear(skill_dim,d);self.language_input=nn.Linear(language_dim,d);self.proprio_input=nn.Linear(proprio_dim,d)
        self.condition_type=nn.Parameter(torch.randn(3,d)*.02);self.proprio_position=nn.Parameter(torch.randn(h,d)*.02)
    def condition_tokens(self,c):
        return torch.cat([self.language_input(c.language)+self.condition_type[0],self.skill_input(c.skill)+self.condition_type[1],self.proprio_input(c.proprio)+self.proprio_position+self.condition_type[2]],1)
    def forward(self,condition,noisy_action=None,diffusion_step=None):
        if noisy_action is None or diffusion_step is None:raise ValueError('Noisy action and diffusion timestep required')
        b=noisy_action.shape[0]
        freq=torch.exp(-math.log(10000)*torch.arange(self.hidden_dim//2,device=noisy_action.device)/(self.hidden_dim//2))
        phase=diffusion_step.reshape(b,1).float()*freq
        time=self.time(torch.cat([phase.sin(),phase.cos()],-1))[:,None]
        memory=self.condition_tokens(condition)+time
        mask=torch.cat([torch.zeros(b,2,dtype=torch.bool,device=condition.valid.device),~condition.valid],1)
        x=self.action_input(noisy_action[:,0])+self.action_position+time
        pred=self.head(self.decoder(x,memory,memory_key_padding_mask=mask))
        return {'noise_pred':pred[:,None]}

