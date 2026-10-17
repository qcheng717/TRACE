from dataclasses import dataclass

import math

import torch

from torch import nn

from trace_robotwin.models.value_gate import SkillAttention, router_gate

from trace_robotwin.models.attention import SkillCondition, SkillDiT, encoder

@dataclass
class PolicyCondition(SkillCondition):
    visual: torch.Tensor
    dictionary: torch.Tensor
    weights: torch.Tensor

class PolicyBlock(nn.Module):
    def __init__(self,d,heads,prior_scale,router_mode="prior",gate_gamma=1.):
        super().__init__();self.heads=heads;self.prior_scale=prior_scale
        self.router_mode=router_mode;self.gate_gamma=gate_gamma;self.gate_beta=0.
        self.norms=nn.ModuleList([nn.LayerNorm(d) for _ in range(5)])
        self.self_attention=nn.MultiheadAttention(d,heads,dropout=0,batch_first=True)
        self.state_attention=nn.MultiheadAttention(d,heads,dropout=0,batch_first=True)
        self.visual_attention=nn.MultiheadAttention(d,heads,dropout=0,batch_first=True)
        self.skill_attention=SkillAttention(d,heads,dropout=0,batch_first=True)
        self.ff=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def forward(self,x,state,visual,skill,weights,action_valid_mask=None):
        # nn.MultiheadAttention: True in key_padding_mask means ignore that K/V.
        # Leave query rows intact; cross-attention and FFN are query-local.
        key_padding_mask=None if action_valid_mask is None else ~action_valid_mask
        q=self.norms[0](x);x=x+self.self_attention(q,q,q,key_padding_mask=key_padding_mask,need_weights=False)[0]
        x=x+self.state_attention(self.norms[1](x),state,state,need_weights=False)[0]
        x=x+self.visual_attention(self.norms[2](x),visual,visual,need_weights=False)[0]
        b,t,_=x.shape;k=weights.shape[-1]
        # Additive logit bias, not a value multiplier or a log(w) prior.
        bias=self.prior_scale*(weights-1./k)
        bias=bias[:,None,None,:].expand(b,self.heads,t,k).reshape(b*self.heads,t,k).to(x.dtype)
        mask=bias if self.router_mode not in ('value_gate','local_sparse') else None
        gate=None if self.router_mode in ('prior','local_sparse') else router_gate(weights,self.gate_beta,self.gate_gamma)
        if self.router_mode=='local_sparse':
            pi=weights.float()
            neff=(-(pi*pi.clamp_min(1e-30).log()).sum(-1)).exp().detach()
            gate=neff[:,None]*pi
            mask=torch.zeros_like(bias).reshape(b,self.heads,t,k).masked_fill(~(pi>0)[:,None,None,:],float('-inf')).reshape(b*self.heads,t,k)
        x=x+self.skill_attention(self.norms[3](x),skill,skill,attn_mask=mask,need_weights=False,value_gate=gate)[0]
        return x+self.ff(self.norms[4](x))

class ConditionedDiT(SkillDiT):
    def __init__(self,config):
        c=config['policy'];v=config['v7'];extra=config.get('v8',{});d=c['hidden_dim']
        super().__init__(v['skill_dim'],config['language_dim'],config['proprio_dim'],v['history_length'],c)
        del self.decoder,self.condition_type
        self.proprio_mode=extra.get('proprio_mode','temporal')
        if self.proprio_mode not in ('temporal','current'):
            raise ValueError('v8.proprio_mode must be temporal or current')
        self.condition_type=nn.Parameter(torch.randn(3 if self.proprio_mode=='current' else 4,d)*.02)
        if self.proprio_mode=='current':
            del self.proprio_position
        else:
            self.proprio_temporal=nn.ModuleList([encoder(d,c['heads']) for _ in range(extra.get('proprio_layers',1))])
        self.visual_input=nn.Identity() if v['hidden_dim']==d else nn.Linear(v['hidden_dim'],d)
        self.basis_input=nn.Linear(v['skill_dim'],d)
        self.router_mode=extra.get('skill_router_mode','prior')
        if self.router_mode not in ('prior','value_gate','prior_plus_gate','local_sparse'):raise ValueError('Invalid skill_router_mode')
        self.gate_beta_max=float(extra.get('skill_router_gate_beta_max',.5))
        self.gate_gamma=float(extra.get('skill_router_gate_gamma',1.))
        self.gate_start=int(extra.get('skill_router_gate_start_step',5000))
        self.gate_full=int(extra.get('skill_router_gate_full_step',10000))
        if not (0<=self.gate_beta_max<=10 and 0<self.gate_gamma and 0<=self.gate_start<self.gate_full):raise ValueError('Invalid value gate schedule')
        self.blocks=nn.ModuleList([PolicyBlock(d,c['heads'],extra.get('skill_router_prior_lambda',extra.get('skill_prior_scale',1.)),self.router_mode,self.gate_gamma) for _ in range(c['layers'])])
        self.set_policy_step(0)
        self.final_norm=nn.LayerNorm(d)
    def set_policy_step(self,step):
        """Completed global policy updates; reconstructed from checkpoint step on resume/eval."""
        self.policy_step=int(step)
        beta=self.gate_beta_max*min(1.,max(0.,(step-self.gate_start)/(self.gate_full-self.gate_start)))
        for block in self.blocks:block.gate_beta=beta if self.router_mode in ('value_gate','prior_plus_gate') else 0.

    def skill_tokens(self,dictionary,batch):
        return self.basis_input(dictionary) if dictionary.ndim==3 else self.basis_input(dictionary.T)[None].expand(batch,-1,-1)

    def condition_tokens(self,c):
        if self.proprio_mode=='current':
            cur=self.proprio_input(c.proprio[:,-1:])
            state=torch.cat([self.language_input(c.language)+self.condition_type[0],self.skill_input(c.skill)+self.condition_type[1],cur+self.condition_type[2]],1)
            return state,self.visual_input(c.visual),self.skill_tokens(c.dictionary,cur.shape[0])
        q=self.proprio_input(c.proprio)+self.proprio_position
        for block in self.proprio_temporal:q=block(q,src_key_padding_mask=~c.valid)
        hist=(q*c.valid[:,:,None]).sum(1,keepdim=True)/c.valid.sum(1)[:,None,None].clamp_min(1)
        cur=q[:,-1:]
        state=torch.cat([self.language_input(c.language)+self.condition_type[0],self.skill_input(c.skill)+self.condition_type[1],hist+self.condition_type[2],cur+self.condition_type[3]],1)
        return state,self.visual_input(c.visual),self.skill_tokens(c.dictionary,q.shape[0])
    def forward(self,condition,noisy_action=None,diffusion_step=None,action_valid_mask=None):
        if noisy_action is None or diffusion_step is None:raise ValueError('Noisy action and timestep required')
        if action_valid_mask is not None:
            if action_valid_mask.dtype!=torch.bool or action_valid_mask.shape!=(noisy_action.shape[0],self.action_horizon):
                raise ValueError('action_valid_mask must be BoolTensor[B, action_horizon]')
            if action_valid_mask.device!=noisy_action.device or not action_valid_mask.any(dim=1).all():
                raise ValueError('Each sample needs at least one valid action on the input device')
        # Sampling has no padded actions: None retains the original inference path.
        b=noisy_action.shape[0];d=self.hidden_dim
        freq=torch.exp(-math.log(10000)*torch.arange(d//2,device=noisy_action.device)/(d//2))
        phase=diffusion_step.reshape(b,1).float()*freq
        time=self.time(torch.cat([phase.sin(),phase.cos()],-1))[:,None]
        state,visual,skill=self.condition_tokens(condition)
        x=self.action_input(noisy_action[:,0])+self.action_position+time
        for block in self.blocks:x=block(x,state+time,visual+time,skill+time,condition.weights,action_valid_mask)
        return {'noise_pred':self.head(self.final_norm(x))[:,None]}

