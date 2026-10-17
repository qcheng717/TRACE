"""TRACE model: predictive spatial skills and a conditioned diffusion policy."""
import torch
from torch import nn
from trace_robotwin.models.backbone.factory import make_backbone
from trace_robotwin.models.diffusion_policy import DiffusionPolicy
from trace_robotwin.models.skill_ops import masked_mean,orthogonal_loss,rescue_loss
from trace_robotwin.models.policy_blocks import PolicyCondition
from trace_robotwin.models.components import SpatialSkillPredictor, SpatialEvidenceRouter
from trace_robotwin.models.diagnostics import prediction_metrics


class TRACE(nn.Module):
    def __init__(self,config,backbone=None,stage='stage1'):
        nn.Module.__init__(self);self.config=config;self.step=0;c=config['v10']
        if config.get('model_version')!='v10' or config.get('stage1_arch')!='v10_spatial_skill':raise ValueError('Explicit V10 required')
        if c.get('language_conditioning')!='film':raise ValueError('V10 now requires explicit language_conditioning=film; old attention checkpoints are incompatible')
        if c.get('final_normalization')!='layer_norm_no_affine':raise ValueError('V10 requires explicit final_normalization=layer_norm_no_affine')
        if c['future_target_offset']<1 or c['future_horizon']!=1:raise ValueError('V10 predicts one positive-offset endpoint')
        if c['gripper_mode']!='continuous_diffusion':raise ValueError('V10 requires continuous gripper diffusion')
        if c['dictionary_size']!=15 or c['dictionary_dim']!=100:raise ValueError('V10 dictionary must be 15x100')
        if c['future_decoder_depth']!=2 or c.get('contextualizer_depth',2)!=2:raise ValueError('V10 uses exactly two spatial blocks')
        if c['skill_encoder_dim']!=c['semantic_dim'] or c['future_decoder_dim']!=c['semantic_dim']:raise ValueError('Spatial widths must agree')
        if c.get('gripper_only',False):raise ValueError('V10 trains full continuous actions')
        if c['router_heads']!=8 or c['router_dim']%8:raise ValueError('Eight router heads required')
        if not 0<c['ema_target_decay']<1:raise ValueError('Invalid target EMA')
        if not 0<=c['entmax_start_step']<c['entmax_full_step']:raise ValueError('Invalid sparse schedule')
        if min(c['skill_encoder_temperature'],c['router_temperature'],c['router_norm_eps'],c['orth_warmup_steps'])<=0:raise ValueError('Invalid scale/warmup')
        if not 1<=config['policy']['execution_horizon']<=config['policy']['action_horizon']:raise ValueError('Execution horizon must fit the configured action chunk')
        self.backbone=make_backbone(config['backbone']) if backbone is None else backbone
        self.predictor=SpatialSkillPredictor(self.backbone.feature_dims,config['language_dim'],c)
        self.router=SpatialEvidenceRouter(c)
        self.policy=DiffusionPolicy(dict(config,v9=c)) # Private adapter to stable continuous DiT API.
        self.register_buffer('usage_ema',torch.full((15,),1/15))
        self.register_buffer('usage_updates',torch.zeros((),dtype=torch.long))
        self.register_buffer('previous_top4',torch.zeros(15,dtype=torch.bool))
        self.register_buffer('batch_top4_jaccard',torch.zeros(()))
        self.rescue_reference=None;self.selector_health=None;self.monitor_dynamics=False;self.selector_frozen=False;self.set_stage(stage)

    def set_stage(self,stage):
        if stage not in ('stage1','policy'):raise ValueError(stage)
        self.stage=stage;self.requires_grad_(False)
        if stage=='stage1':
            self.predictor.requires_grad_(True);self.predictor.target_selector.requires_grad_(False)
            if self.selector_frozen:self.predictor.selector.requires_grad_(False)
        else:self.router.requires_grad_(True);self.policy.requires_grad_(True)
        self.train(self.training)

    def freeze_stage1_selector(self):
        """Keep both online input representation and EMA supervision fixed."""
        if self.stage!='stage1':raise ValueError('Selector freezing requires Stage-I')
        self.selector_frozen=True
        self.predictor.selector.requires_grad_(False).eval()
        self.predictor.target_selector.requires_grad_(False).eval()

    @torch.no_grad()
    def encode(self,images,wrist=None):
        if self.config['v10'].get('camera_views',2)==2:
            raise ValueError("This release supports the single head camera protocol")
        if self.config['v10'].get('camera_views')!=1 or images.ndim!=5:
            raise ValueError('Expected one head camera with B,T,3,H,W images')
        if wrist is not None and wrist.numel():raise ValueError('Single-camera model must not receive wrist images')
        b,t=images.shape[:2]
        features=self.backbone(images.flatten(0,1))
        return {k:(v.flatten(-2).transpose(-1,-2) if v.ndim==4 else v).reshape(b,t,1,-1,self.backbone.feature_dims[k]) for k,v in features.items()}

    def train(self,mode=True):
        nn.Module.train(self,mode);self.backbone.eval();self.predictor.target_selector.eval()
        if self.selector_frozen:self.predictor.selector.eval()
        if self.stage=='policy':self.predictor.eval()
        return self

    def stage1_loss(self,batch):
        r=self.stage1_outputs(batch);c=self.config['v10'];mask=r['future_valid_mask']
        pred,target=r['pred_transition'],r['target_transition']
        trans=masked_mean((pred-target).float().abs().mean((-1,-2)),mask,batch.get('_future_denominator'))
        orth=orthogonal_loss(r['D_normalized'])
        surrogate=self.predictor.encoder(r['H'],r['D_normalized'],rescue=True)
        rescue=rescue_loss(surrogate,self.usage_ema,c['dictionary_usage_min_ratio'],self.rescue_reference)
        weight=c['lambda_orth_max']*min(1.,self.step/c['orth_warmup_steps'])
        loss=trans+weight*orth+c['lambda_dict_rescue']*rescue
        self.diagnostic_weights=r['A'].detach().flatten(0,1)
        if self.selector_health is not None:self.selector_health.add(r,self.predictor.selector.last_layer_weights)
        metrics=prediction_metrics(pred[:,0],target[:,0],mask[:,0])
        metrics['future_supervision_coverage']=mask.float().mean()
        if 'future_clamped' in batch:
            metrics['future_clamped_fraction']=batch['future_clamped'].float().mean()
            metrics['future_actual_offset_mean']=masked_mean(batch['future_actual_offset'].float(),mask)
            metrics['future_zero_offset_fraction']=((batch['future_actual_offset']==0)&mask).float().mean()
        metrics.update(orth_loss=orth.detach(),orth_weight=loss.new_tensor(weight),dict_rescue_loss=rescue.detach(),
            starving_basis_count=(self.usage_ema<c['dictionary_usage_min_ratio']/15).sum().float())
        return loss,metrics

    def condition(self,features,proprio,language,valid=None):
        with torch.no_grad():r=self.prediction(features,language)
        d=r['D_normalized'];route=self.router(d,r['pred_transition'],r['A'],self.step)
        bases=d[None].expand(len(language),-1,-1)
        mixture=route['pi']@d
        mask=torch.ones(len(language),1,dtype=torch.bool,device=language.device)
        cond=PolicyCondition(mixture[:,None],language[:,None],proprio[:,-1:],mask,r['H'],bases,route['pi'])
        self.diagnostic_weights=route['pi'].detach()
        return cond,dict(**route,local=bases,transition=r['pred_transition'],A=r['A'])

    def policy_loss(self,batch):
        cond,r=self.condition(self.encode(batch['images'],batch['wrist']),batch['proprio'],batch['language'])
        clean=batch['actions'][:,None];mask=batch['action_mask'];c=self.config['v10']
        noise=torch.randn_like(clean);step=torch.randint(self.policy.steps,clean.shape[:2],device=clean.device)
        result=self.policy(cond,self.policy.add_noise(clean,step,noise),step,mask)
        errors=(result['noise_pred']-noise).float().square()[:,0]
        diffusion=masked_mean(errors.mean(-1),mask,batch.get('_action_denominator'))
        rescue=rescue_loss(r['soft'],self.usage_ema,c['router_usage_min_ratio'],self.rescue_reference)
        pi=r['pi'];metrics=dict(diffusion=diffusion.detach(),router_rescue_loss=rescue.detach(),
            gripper_diffusion=masked_mean(errors[...,-1],mask).detach(),pose_diffusion=masked_mean(errors[...,:-1].mean(-1),mask).detach(),
            router_support_size=(pi>0).sum(-1).float().mean(),sample_neff=(-(pi*pi.clamp_min(1e-12).log()).sum(-1)).exp().mean().detach(),
            router_sparse_eta=diffusion.new_tensor(r['eta']),starving_basis_count=(self.usage_ema<c['router_usage_min_ratio']/15).sum().float())
        if self.config.get('benchmark_version')=='robotwin_v4':metrics['joint_diffusion']=metrics['pose_diffusion']
        if self.monitor_dynamics:
            with torch.no_grad():
                d=self.predictor.normalized_dictionary()
                zero=self.router(d,torch.zeros_like(r['transition']),r['A'],self.step)['pi']
                shuffled=self.router(d,r['transition'].roll(1,0),r['A'],self.step)['pi'];mid=(pi+shuffled)/2
                js=.5*((pi*(pi.clamp_min(1e-12).log()-mid.clamp_min(1e-12).log())).sum(-1)+(shuffled*(shuffled.clamp_min(1e-12).log()-mid.clamp_min(1e-12).log())).sum(-1))
                metrics.update(router_zero_transition_deviation=(zero-1/15).norm(dim=-1).mean(),router_future_shuffle_JS=js.clamp_min(0).mean())
        return diffusion+c['lambda_router_rescue']*rescue,metrics

    @torch.no_grad()
    def completed_update(self,weights):
        c=self.config['v10'];total=weights.float().sum(0);count=total.new_tensor(len(weights))
        if torch.distributed.is_initialized():torch.distributed.all_reduce(total);torch.distributed.all_reduce(count)
        top=torch.zeros_like(self.previous_top4);top[(total/count).topk(4).indices]=True
        if self.previous_top4.any():self.batch_top4_jaccard.copy_((top&self.previous_top4).sum()/(top|self.previous_top4).sum())
        self.previous_top4.copy_(top)
        beta=c['dictionary_usage_ema_beta'] if self.stage=='stage1' else c['router_usage_ema_beta']
        self.usage_ema.lerp_(total/count,1-beta);self.usage_updates.add_(1)
        if self.stage=='stage1' and not self.selector_frozen:self.predictor.update_target()
        self.step+=1;self.policy.set_policy_step(self.step)

    def optimizer_groups(self):
        groups=[]
        for name,module in ([('predictor',self.predictor)] if self.stage=='stage1' else [('router',self.router),('policy',self.policy)]):
            groups.append(dict(name=name,params=[p for p in module.parameters() if p.requires_grad]))
        return groups

    def prediction(self, features, language):
        if any(v.shape[1]!=1 for v in features.values()):raise ValueError('Predictor input is current-only')
        return self.predictor({k:v[:,-1] for k,v in features.items()}, language)


    def stage1_outputs(self, batch):
        features=self.encode(batch['images'],batch['wrist'])
        current={k:v[:,0] for k,v in features.items()}
        result=self.predictor(current,batch['language'])
        with torch.no_grad():
            anchor=self.predictor.target_selector(current,batch['language'])
            future=self.encode(batch['future_images'],batch['future_wrist'])
            b,h=batch['future_valid_mask'].shape
            target=self.predictor.target_selector({k:v.flatten(0,1) for k,v in future.items()},
                batch['language'][:,None].expand(-1,h,-1).flatten(0,1)).unflatten(0,(b,h))
        result.update(S_tar_current=anchor,target_future=target,target_transition=target-anchor[:,None],
            pred_future=anchor[:,None]+result['pred_transition'],future_valid_mask=batch['future_valid_mask'])
        return result


    def forward(self,batch):
        if batch['images'].shape[1]!=1:raise ValueError('V9 accepts exactly one current frame, no history')
        return self.stage1_loss(batch) if self.stage=='stage1' else self.policy_loss(batch)

