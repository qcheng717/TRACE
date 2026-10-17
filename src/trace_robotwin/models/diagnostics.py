"""Detached prediction and dictionary diagnostics."""
import math
import torch
from torch.nn import functional as F
from trace_robotwin.models.skill_ops import geometry


def entropy(p):return -(p*p.clamp_min(1e-12).log()).sum(-1)


@torch.no_grad()
def prediction_metrics(pred,target,valid):
    p=pred.float()[valid];t=target.float()[valid]
    zero=pred.new_zeros((),dtype=torch.float32)
    keys=['pred_l1','copy_l1','prediction_gain','transition_cosine','transition_magnitude_ratio']
    for suffix in ('active','static'):
        keys.extend([f'pred_l1_{suffix}',f'transition_cosine_{suffix}',f'target_residual_norm_{suffix}'])
    if not len(p):return {k:zero for k in keys}
    error=(p-t).abs().mean(-1);pn=p.norm(dim=-1);tn=t.norm(dim=-1)
    cos=F.cosine_similarity(p,t,dim=-1,eps=1e-8)
    copy=t.abs().mean(-1).mean();loss=error.mean()
    result=dict(pred_l1=loss,copy_l1=copy,prediction_gain=(copy-loss)/(copy+1e-8),
                transition_cosine=cos.mean(),transition_magnitude_ratio=pn.mean()/(tn.mean()+1e-8))
    active=torch.zeros_like(tn,dtype=torch.bool)
    active.scatter_(-1,tn.topk(math.ceil(.2*tn.shape[-1]),dim=-1).indices,True)
    for name,mask in [('active',active),('static',~active)]:
        result.update({f'pred_l1_{name}':error[mask].mean(),f'transition_cosine_{name}':cos[mask].mean(),
                       f'target_residual_norm_{name}':tn[mask].mean()})
    return result


@torch.no_grad()
def representation_metrics(h,a,alpha):
    h=h.float();a=a.float();alpha=alpha.float();ent=entropy(a);sample=a.mean(1)
    # Deterministic sampled distinct patch pairs: diagnostics do not consume training RNG.
    p=torch.linspace(0,a.shape[1]-1,min(128,a.shape[1]),device=a.device).long()
    q=(p*73+19)%a.shape[1];q=torch.where(q==p,(q+1)%a.shape[1],q)
    pair=F.cosine_similarity(a[:,p],a[:,q],dim=-1).mean()
    centered=h.flatten(0,1);centered=centered-centered.mean(0)
    # Covariance eigenspectrum equals singular values squared; report singular-value entropy rank.
    singular=torch.linalg.eigvalsh(centered.T@centered).clamp_min(0).sqrt()
    prob=singular/singular.sum().clamp_min(1e-12)
    rank=torch.where(singular.sum()>1e-8,entropy(prob).exp(),singular.new_zeros(()))
    result=dict(semantic_batch_std=h.std(0,unbiased=False).mean(),semantic_token_std=h.std(1,unbiased=False).mean(),
        semantic_effective_rank=rank,skill_coeff_entropy_mean=ent.mean(),skill_coeff_entropy_std=ent.std(unbiased=False),
        skill_coeff_neff_mean=ent.exp().mean(),skill_coeff_neff_min=ent.exp().min(),skill_coeff_neff_max=ent.exp().max(),
        skill_coeff_spatial_var=a.var(1,unbiased=False).mean(),skill_coeff_spatial_cosine=pair,
        skill_coeff_spatial_diversity=1-pair,sample_skill_usage_neff=entropy(sample).exp().mean(),
        sample_skill_usage_top3=sample.topk(3,-1).values.sum(-1).mean(),batch_sample_usage_std=sample.std(0,unbiased=False).mean(),
        sample_usage_cosine=F.cosine_similarity(sample,sample.roll(1,0),dim=-1).mean() if len(sample)>1 else sample.new_zeros(()),
        layer_weight_entropy=entropy(alpha).mean(),layer_weight_max=alpha.max(-1).values.mean(),
        layer_weight_batch_std=alpha.std(0,unbiased=False).mean())
    for i,l in enumerate((3,6,9,12)):
        result[f'layer_weight_{l}_mean']=alpha[:,i].mean();result[f'layer_weight_{l}_std']=alpha[:,i].std(unbiased=False)
    return result


class Stage1Monitor:
    """One effective update including all microbatches/ranks; only used on log updates."""
    def __init__(self):self.rows=[]
    @torch.no_grad()
    def add(self,r,alpha):
        self.rows.append(tuple(x.detach().float() for x in (r['H'],r['A'],alpha,r['pred_transition'][:,0],r['target_transition'][:,0],r['future_valid_mask'][:,0],r['H_raw'],r['S_tar_current'])))
    @torch.no_grad()
    def finish(self):
        values=[torch.cat([r[i] for r in self.rows]) for i in range(8)]
        if torch.distributed.is_initialized():
            gathered=[]
            for v in values:
                parts=[torch.empty_like(v) for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather(parts,v.contiguous());gathered.append(torch.cat(parts))
            values=gathered
        h,a,alpha,p,t,mask,raw,anchor=values
        return dict(**representation_metrics(h,a,alpha),**prediction_metrics(p,t,mask.bool()),
            semantic_raw_batch_std=raw.std(0,unbiased=False).mean(),
            semantic_normalized_batch_std=h.std(0,unbiased=False).mean(),
            semantic_normalized_channel_std=h.std(-1,unbiased=False).mean(),
            online_target_cosine=F.cosine_similarity(h,anchor,dim=-1).mean(),
            online_target_l1=(h-anchor).abs().mean())


@torch.no_grad()
def dictionary_metrics(model):
    raw=model.predictor.dictionary.float();init=model.predictor.D_init.float();d=model.predictor.normalized_dictionary()
    # FP64 chord distance avoids cancellation in 1-cos for very small rotations.
    direction=F.normalize(raw.double(),dim=-1);initial=F.normalize(init.double(),dim=-1)
    chord=(direction-initial).norm(dim=-1)
    angles=2*torch.asin((chord/2).clamp(0,1))*180/math.pi
    metrics={'dictionary_'+k:v for k,v in geometry(d).items()}
    metrics.update(dictionary_norm_mean=raw.norm(dim=-1).mean(),dictionary_norm_std=raw.norm(dim=-1).std(unbiased=False),
        dictionary_cosine_displacement=(chord.square()/2).mean().float(),
        dictionary_normalized_l2_displacement=chord.mean().float(),
        dictionary_angle_degrees_mean=angles.mean().float(),dictionary_angle_degrees_max=angles.max().float(),
        dictionary_l2_displacement=(raw-init).norm(dim=-1).mean())
    sets={
        'raw':(direction,initial),
        'key_effective':(direction@model.predictor.encoder.key.weight.double().T,
                         initial@model.predictor.key_weight_init.double().T),
        'decoder_effective':(direction@model.predictor.decoder.projection.weight.double().T,
                             initial@model.predictor.decoder_weight_init.double().T)}
    for label,(current,origin) in sets.items():
        current=F.normalize(current,dim=-1);origin=F.normalize(origin,dim=-1)
        chord=(current-origin).norm(dim=-1)
        degrees=2*torch.asin((chord/2).clamp(0,1))*180/math.pi
        prefix='dictionary_'+label
        metrics[prefix+'_angle_mean']=degrees.mean().float()
        metrics[prefix+'_angle_max']=degrees.max().float()
        geo=geometry(current.float())
        metrics[prefix+'_rank' if label!='raw' else prefix+'_effective_rank']=geo['effective_rank']
        metrics[prefix+'_offdiag']=geo['mean_abs_offdiag']
        metrics[prefix+'_max_abs_offdiag']=geo['max_abs_offdiag']
    return metrics


def gradient_metrics(model):
    groups={'layer_projection':('predictor.selector.project.',),
        'layer_selector':('predictor.selector.language.','predictor.selector.read.','predictor.selector.score.'),
        'spatial_contextualizer':('predictor.selector.contextualizer.','predictor.selector.position.'),
        'skill_query':('predictor.encoder.query.',),'dictionary_key':('predictor.encoder.key.',),
        'dictionary':('predictor.dictionary',),'future_decoder':('predictor.decoder.',),'router':('router.',),
        'state_cross_attn':('.state_attention.',),'visual_cross_attn':('.visual_attention.',),
        'skill_cross_attn':('.skill_attention.',),'proprio_encoder':('policy.proprio_input.',)}
    result={}
    for key,prefixes in groups.items():
        grads=[p.grad.detach().float().square().sum() for n,p in model.named_parameters() if any(s in n for s in prefixes) and p.grad is not None]
        result['grad_norm_'+key]=torch.stack(grads).sum().sqrt() if grads else model.usage_ema.new_zeros(())
    return result
