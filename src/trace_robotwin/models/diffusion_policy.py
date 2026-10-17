"""Progressive DiT with disjoint continuous-noise and binary-gripper heads."""
import math
import torch
from torch import nn
from trace_robotwin.models.policy_blocks import ConditionedDiT


class DiffusionPolicy(ConditionedDiT):
    def __init__(self, config):
        c = config['v9']; p = dict(config['policy'])
        self.gripper_mode = c.get('gripper_mode', 'binary_ce')
        if self.gripper_mode not in ('binary_ce', 'continuous_diffusion'):raise ValueError('Invalid gripper mode')
        if self.gripper_mode == 'binary_ce':p['action_dim'] -= 1
        super().__init__(dict(policy=p, language_dim=config['language_dim'], proprio_dim=config['proprio_dim'],
            v7=dict(skill_dim=c['dictionary_dim'], hidden_dim=c['semantic_dim'], history_length=1),
            v8=dict(proprio_mode='current', skill_router_mode='local_sparse', skill_router_prior_lambda=0.)))
        self.gripper_only = c.get('gripper_only', False)
        if self.gripper_only:
            if self.gripper_mode != 'continuous_diffusion':raise ValueError('gripper_only requires continuous diffusion')
            self.gripper_branch = nn.Sequential(nn.Linear(p['hidden_dim']+1,p['hidden_dim']),nn.SiLU(),nn.Linear(p['hidden_dim'],1))
        if self.gripper_mode == 'binary_ce':self.gripper_head = nn.Linear(p['hidden_dim'], 2)

    def forward(self, condition, noisy_action=None, diffusion_step=None, action_valid_mask=None):
        if noisy_action is None or noisy_action.shape[-1] != self.action_dim:
            raise ValueError('V9 diffusion input must contain continuous coordinates only')
        b = len(noisy_action); d = self.hidden_dim
        if action_valid_mask is not None and (action_valid_mask.dtype != torch.bool or
                action_valid_mask.shape != (b,self.action_horizon) or not action_valid_mask.any(-1).all()):
            raise ValueError('Invalid action padding mask')
        freq = torch.exp(-math.log(10000)*torch.arange(d//2,device=noisy_action.device)/(d//2))
        phase = diffusion_step.reshape(b,1).float()*freq
        time = self.time(torch.cat([phase.sin(), phase.cos()], -1))[:,None]
        state, visual, skill = self.condition_tokens(condition)
        action = noisy_action[:,0]
        if self.gripper_only:
            action = torch.cat([action[...,:-1],torch.zeros_like(action[...,-1:])],-1)
        x = self.action_input(action)+self.action_position+time
        for block in self.blocks:
            x = block(x, state+time, visual+time, skill+time, condition.weights, action_valid_mask)
        hidden = self.final_norm(x)
        result = dict(noise_pred=self.head(hidden)[:,None], hidden=hidden)
        if self.gripper_only:
            grip = self.gripper_branch(torch.cat([hidden.detach(),noisy_action[:,0,...,-1:]],-1))
            result['noise_pred'] = torch.cat([result['noise_pred'][...,:-1],grip[:,None]],-1)
        if self.gripper_mode == 'binary_ce':result['gripper_logits']=self.gripper_head(hidden)
        return result

    @torch.no_grad()
    def sample(self, condition, generator=None):
        shape = (len(condition.skill),1,self.action_horizon,self.action_dim)
        x = torch.randn(shape,device=condition.skill.device,dtype=condition.skill.dtype,generator=generator)
        schedule = torch.linspace(self.steps-1,0,self.inference_steps,device=x.device).round().long()
        for i,n in enumerate(schedule):
            result = self(condition,x,torch.full(shape[:2],int(n),device=x.device,dtype=torch.long))
            a = self.alpha_bar[n]; eps = result['noise_pred']
            clean = ((x-(1-a).sqrt()*eps)/a.sqrt()).clamp(-1,1)
            prev = self.alpha_bar[schedule[i+1]] if i+1<len(schedule) else x.new_tensor(1.)
            eps = (x-a.sqrt()*clean)/(1-a).sqrt()
            x = prev.sqrt()*clean+(1-prev).sqrt()*eps
        if self.gripper_mode == 'continuous_diffusion':return dict(actions=x[:,0])
        # Last denoising forward (t=0), not an extra unrelated/noisy prediction.
        self.last_gripper_logits = result['gripper_logits']
        self.last_gripper_probability = result['gripper_logits'].float().softmax(-1)[...,1]
        grip = result['gripper_logits'].argmax(-1)
        return dict(continuous=x[:,0], gripper=grip, open_probability=self.last_gripper_probability)
