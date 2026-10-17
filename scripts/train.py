"""TRACE RoboTwin two-stage trainer. No formal run starts on import."""
import argparse
from contextlib import nullcontext
import copy
import fcntl
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from trace_robotwin.config import load_config
from trace_robotwin.models.model import TRACE
from trace_robotwin.models.checkpoint import metadata,validate,transfer_stage1
from trace_robotwin.models.skill_ops import geometry
from trace_robotwin.models.diagnostics import Stage1Monitor, dictionary_metrics, gradient_metrics

from trace_robotwin.train.ema import ModelEMA
from trace_robotwin.train.schedule import lr_multiplier
from trace_robotwin.train.selector_freeze import auto_freeze_step, maybe_freeze_selector


def resolve_config(path,overrides):
    c=load_config(path,overrides)
    if c['data']['adapter'] in ('robotwin_motif20','robotwin_motif20_v4'):
        os.environ.setdefault('ROBOTWIN_ROOT',str(ROOT/'third_party/RoboTwin'))
        stats=json.loads(Path(c['data']['norm_stats']).read_text())
        c['data']['proprio_stats']=stats['state'];c['data']['action_stats']=stats['action']
    if c['data']['batch_size']!=c['train']['batch_size']:raise ValueError('Set data.batch_size and train.batch_size together')
    return c


def snapshot(model,out,step,batch=None):
    folder=out/'dictionary';folder.mkdir(exist_ok=True)
    snapshot_path=folder/f'dictionary_snapshot_step{step:06d}.pt'
    torch.save(dict(D_raw=model.predictor.dictionary.detach().cpu(),D_normalized=model.predictor.normalized_dictionary().detach().cpu(),
        usage_ema=model.usage_ema.cpu(),step=step),snapshot_path)
    sys.path.insert(0,str(ROOT/'tools'))
    from visualize_dictionary import draw
    draw('snapshot',[snapshot_path],folder)
    if model.stage=='policy' and hasattr(model,'local_basis_diagnostic'):
        torch.save(dict(D_star=model.predictor.normalized_dictionary().cpu(),D_local=model.local_basis_diagnostic[:4].cpu(),step=step,
            metadata={k:(v[-len(model.local_basis_diagnostic):][:4].tolist() if torch.is_tensor(v) else v[-len(model.local_basis_diagnostic):][:4]) for k,v in batch.get('metadata',{}).items()} if batch else None,segment=batch['segment'][-len(model.local_basis_diagnostic):][:4] if batch else None,
            anchor=batch['anchor'][-len(model.local_basis_diagnostic):][:4] if batch else None),folder/f'global_local_step{step:06d}.pt')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);p.add_argument('--stage',choices=['stage1','policy'],required=True)
    p.add_argument('--set',action='append',default=[])
    p.add_argument('--stop-after-step',type=int,help='Save and exit at this absolute step without changing the LR schedule')
    a=p.parse_args();os.chdir(ROOT)
    c=resolve_config(a.config,a.set);t=c['train'];rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    device=torch.device('cuda',int(os.environ.get('LOCAL_RANK',0))) if t['device']!='cpu' and torch.cuda.is_available() else torch.device('cpu')
    torch.set_num_threads(t.get('cpu_threads',2))
    if device.type=='cuda':torch.cuda.set_device(device)
    if world>1:torch.distributed.init_process_group('nccl' if device.type=='cuda' else 'gloo')
    accum=t['gradient_accumulation_steps'];micro=t['batch_size']
    freeze_at=t.get('stage1_freeze_selector_at_step')
    auto_freeze_at=auto_freeze_step(c,a.stage)
    if freeze_at is not None and (a.stage!='stage1' or not isinstance(freeze_at,int) or freeze_at<0):
        raise ValueError('stage1_freeze_selector_at_step requires Stage-I and a nonnegative integer')
    end_step=t['max_steps'] if a.stop_after_step is None else a.stop_after_step
    if not 1<=end_step<=t['max_steps']:raise ValueError('stop-after-step must be within the configured training range')
    if min(accum,micro,t['max_steps'],t['log_every'],t['checkpoint_every_steps'])<1:raise ValueError('Positive training counts required')
    random.seed(t['seed']);np.random.seed(t['seed']);torch.manual_seed(t['seed'])
    out=Path(t['output_dir']);out.mkdir(parents=True,exist_ok=True);lock=None
    try:
        allowed=torch.ones((),device=device,dtype=torch.int)
        if rank==0:
            lock=(out/'.train.lock').open('a')
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:allowed.zero_()
            if not t.get('resume') and (any(out.glob('step_*.pt')) or (out/'metrics.jsonl').exists()):allowed.zero_()
        if world>1:torch.distributed.broadcast(allowed,0)
        if not allowed.item():raise RuntimeError('Output is occupied; choose a new directory or explicit resume')
        model=TRACE(c['model'],stage=a.stage);resume=None;provenance={}
        if t.get('resume'):
            resume=torch.load(t['resume'],map_location='cpu',weights_only=False);validate(resume,c,a.stage)
            if resume['world_size']!=world or resume['config']['data']!=c['data']:raise ValueError('Resume world/data differs')
            previous_freeze=resume['config']['train'].get('stage1_freeze_selector_at_step')
            if previous_freeze!=freeze_at:
                if previous_freeze is not None or freeze_at is None or resume['step']!=freeze_at:
                    raise ValueError('Selector freeze may only be introduced exactly at the resumed step; frozen resumes must preserve it')
            for key,value in t.items():
                if key not in ('output_dir','resume','num_workers','log_every','checkpoint_every_steps','stage1_checkpoint','selector_health_interval','selector_probe_size','selector_probe_interval','selector_probe_batch_size','dino_frame_batch_size','stage1_freeze_selector_at_step') and resume['config']['train'].get(key)!=value:
                    raise ValueError('Resume training parameter changed: '+key)
            model.load_state_dict(resume['model'],strict=True);model.step=resume['step'];provenance=resume['provenance']
        elif t.get('finetune_checkpoint'):
            raise ValueError('V10 requires its own Stage-I transfer, not legacy fine-tuning')
        elif a.stage=='policy':
            if not t.get('stage1_checkpoint'):raise ValueError('Policy requires new V10 Stage-I checkpoint')
            provenance=transfer_stage1(model,t['stage1_checkpoint'],c)
        model.backbone.frame_batch_size=t.get('dino_frame_batch_size',model.backbone.frame_batch_size)
        if model.backbone.frame_batch_size<1:raise ValueError('Positive DINO frame batch required')
        model.to(device);model.policy.set_policy_step(model.step)
        # Retain the original optimizer group membership so exact resume keeps
        # Adam moments for encoder/dictionary/decoder. Frozen parameters have no
        # gradients and AdamW skips them, including weight decay.
        opt=torch.optim.AdamW(model.optimizer_groups(),lr=t['learning_rate'],weight_decay=t['weight_decay'])
        ema=ModelEMA(model,t['ema_decay'])
        if resume:opt.load_state_dict(resume['optimizer']);ema.load_state_dict(resume['ema'])
        if freeze_at is not None:
            if model.step<freeze_at:
                raise ValueError('Freeze continuation must start at or after its configured step')
            model.freeze_stage1_selector()
            # The full-model inference EMA must use the same fixed online
            # selector as raw training, rather than slowly catching up to it.
            with torch.no_grad():
                for name,param in model.named_parameters():
                    if name.startswith('predictor.selector.') and name in ema.shadow:
                        ema.shadow[name].copy_(param)
            if 'selector_freeze' not in provenance:
                provenance=dict(provenance,selector_freeze=dict(step=freeze_at,
                    source_checkpoint=str(Path(t['resume']).resolve()) if t.get('resume') else None,
                    online_selector='frozen_raw',target_selector='frozen_raw_no_target_ema',
                    inference_ema_selector='fixed_raw',optimizer='continued'))
        # V6 resumes rebuild original Adam groups before restoring freeze state.
        maybe_freeze_selector(model,ema,auto_freeze_at,provenance)
        params=[p for p in model.parameters() if p.requires_grad]
        if end_step<=model.step:raise ValueError('Stopping step must be after the initialized/resumed step')
        wrapped=DDP(model,device_ids=[device.index] if device.type=='cuda' else None,broadcast_buffers=False) if world>1 else model
        random.seed(t['seed']+rank);np.random.seed(t['seed']+rank);torch.manual_seed(t['seed']+rank)
        if resume:
            state=resume['rng'][rank];random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
            if device.type=='cuda':torch.cuda.set_rng_state(state['cuda'])
        if c['data']['adapter']=='robotwin_motif20_v4':
            from trace_robotwin.data.dataset import RoboTwinDataset
            dataset=RoboTwinDataset(c,a.stage,start_step=model.step,end_step=end_step)
        else:raise ValueError('Only RoboTwin V4 is supported')
        loader=DataLoader(dataset,batch_size=micro*accum,num_workers=t['num_workers'],pin_memory=device.type=='cuda',
            generator=torch.Generator().manual_seed(t['seed']),
            **({'multiprocessing_context':'spawn','prefetch_factor':1} if t['num_workers'] else {}))
        if rank==0:
            (out/'config.json').write_text(json.dumps(c,indent=2));snapshot(model,out,model.step)
            print(f'{c["model"]["benchmark_version"]} {a.stage} micro={micro} accumulation={accum} global={micro*accum*world}',flush=True)
        if model.selector_frozen:print(f'Online and target selectors frozen at step {freeze_at if freeze_at is not None else auto_freeze_at}; target EMA disabled; optimizer/RNG/LR continued',flush=True)
        spatial_low_streak=0
        start=time.monotonic();initial=model.step
        for full in loader:
            if model.step>=end_step:break
            wrapped.train();opt.zero_grad(set_to_none=True);metrics={};weights=[];total_loss=0.
            for group in opt.param_groups:group['lr']=t['learning_rate']*lr_multiplier(model.step,t)
            normalizers={}
            for key,mask in [('action',full['action_mask'])]+([('future',full['future_valid_mask']),('velocity',full['future_valid_mask'][:,1:]&full['future_valid_mask'][:,:-1])] if a.stage=='stage1' else []):
                count=mask.sum().to(device).float()
                if world>1:torch.distributed.all_reduce(count)
                normalizers['_'+key+'_denominator']=count/(world*accum)
            health_interval=t.get('selector_health_interval',100)
            if health_interval<1:raise ValueError('selector_health_interval must be positive')
            health_expensive=(model.step+1)%health_interval==0 or model.step==initial
            log_update=(model.step+1)%t['log_every']==0 or model.step==initial or (a.stage=='stage1' and health_expensive)
            model.selector_health=Stage1Monitor() if a.stage=='stage1' and log_update else None
            model.monitor_dynamics=(model.step+1)%t['log_every']==0 or model.step==0
            # Exact full-effective-batch rescue mean. Only do the extra pass when
            # EMA identifies starving slots; no predictor target images are needed.
            model.rescue_reference=None
            ratio=c['model']['v10']['dictionary_usage_min_ratio' if a.stage=='stage1' else 'router_usage_min_ratio']
            if (model.usage_ema<ratio/len(model.usage_ema)).any():
                rescue_sum=torch.zeros_like(model.usage_ema);rescue_count=0
                with torch.no_grad(),torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=bool(t['amp'] and device.type=='cuda')):
                    for i in range(accum):
                        mb={k:v[i*micro:(i+1)*micro].to(device) for k,v in full.items() if torch.is_tensor(v) and k in ('images','wrist','language','proprio')}
                        features=model.encode(mb['images'],mb['wrist'])
                        if a.stage=='stage1':
                            r=model.prediction(features,mb['language'])
                            probs=model.predictor.encoder(r['S_on'],r['D_normalized'],rescue=True).flatten(0,1)
                        else:probs=model.condition(features,mb['proprio'],mb['language'])[1]['soft']
                        rescue_sum+=probs.float().sum(0);rescue_count+=len(probs)
                count=rescue_sum.new_tensor(rescue_count)
                if world>1:torch.distributed.all_reduce(rescue_sum);torch.distributed.all_reduce(count)
                model.rescue_reference=rescue_sum/count
            for i in range(accum):
                batch={k:v[i*micro:(i+1)*micro].to(device,non_blocking=True) for k,v in full.items() if torch.is_tensor(v)}
                batch.update(normalizers)
                sync=wrapped.no_sync() if world>1 and i<accum-1 else nullcontext()
                diagnostic=None
                if a.stage=='policy' and model.monitor_dynamics:
                    from trace_robotwin.train.policy_diagnostics import PolicyDiagnostics
                    diagnostic=PolicyDiagnostics(model)
                try:
                    with sync,torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=bool(t['amp'] and device.type=='cuda')):
                        loss,values=wrapped(batch)
                        (loss/accum).backward()
                    if diagnostic is not None:
                        values.update(diagnostic.finish())
                        grads=[p.grad.detach().float().square().sum() for p in model.router.parameters() if p.grad is not None]
                        values['grad_norm_router']=torch.stack(grads).sum().sqrt() if grads else loss.new_zeros(())
                finally:
                    if diagnostic is not None:
                        for handle in diagnostic.handles:handle.remove()
                for k,v in values.items():metrics[k]=metrics.get(k,0)+v.detach().float()/accum
                total_loss+=loss.detach().float()/accum;weights.append(model.diagnostic_weights)
            health_metrics=model.selector_health.finish() if model.selector_health is not None else {}
            model.selector_health=None
            metrics.update(gradient_metrics(model))
            grad=torch.nn.utils.clip_grad_norm_(params,t['grad_clip'])
            finite=torch.tensor(int(torch.isfinite(total_loss) and torch.isfinite(grad)),device=device)
            if world>1:torch.distributed.all_reduce(finite,op=torch.distributed.ReduceOp.MIN)
            if not finite.item():raise FloatingPointError('Nonfinite update rejected; no optimizer/EMA/usage advancement')
            opt.step();model.completed_update(torch.cat(weights));ema.update(model)
            step=model.step
            if maybe_freeze_selector(model,ema,auto_freeze_at,provenance):
                # DDP reducers must be rebuilt when the trainable parameter set changes.
                # Adam groups, moments, RNG and LR progress stay intact.
                if world>1:
                    torch.distributed.barrier()
                    del wrapped
                    wrapped=DDP(model,device_ids=[device.index] if device.type=='cuda' else None,broadcast_buffers=False)
                if rank==0:print(f'Automatically froze online/target selectors at step {step}; optimizer and cosine schedule continued',flush=True)
            if log_update:
                metrics.update(loss=total_loss,grad_norm=grad)
                keys=list(metrics);vals=torch.stack([metrics[k] for k in keys])
                if world>1:torch.distributed.all_reduce(vals);vals/=world
                row=dict(zip(keys,vals.cpu().tolist()));row.update(step=step,stage=a.stage,learning_rate=opt.param_groups[0]['lr'],steps_per_second=(step-initial)/(time.monotonic()-start))
                row.update(cuda_peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                               cuda_peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30)
                row.update({k:float(v) for k,v in health_metrics.items()})
                row['batch_top4_jaccard']=float(model.batch_top4_jaccard)
                row.update({k:float(v) for k,v in dictionary_metrics(model).items()})
                usage=model.usage_ema.float();row.update(usage_ema=usage.cpu().tolist(),usage_min=float(usage.min()),usage_max=float(usage.max()),
                    usage_neff=float((-(usage*usage.clamp_min(1e-12).log()).sum()).exp()),usage_top3=float(usage.topk(min(3,len(usage))).values.sum()))
                if a.stage=='stage1':
                    row['selector_frozen']=model.selector_frozen
                    spatial_low_streak=spatial_low_streak+1 if row.get('skill_coeff_spatial_var',1)<1e-8 else 0
                    row['skill_coeff_spatial_collapse_warning']=spatial_low_streak>=10
                if rank==0:
                    with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                    print(json.dumps(row),flush=True)
            if rank==0 and (step%c['model']['v10']['dictionary_snapshot_interval']==0 or step==end_step):snapshot(model,out,step,full)
            if step%t['checkpoint_every_steps']==0 or step==end_step:
                rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device) if device.type=='cuda' else None)
                states=[None]*world
                if world>1:torch.distributed.all_gather_object(states,rng)
                else:states=[rng]
                if rank==0:
                    payload=dict(metadata=metadata(c,a.stage),stage=a.stage,step=step,config=c,model=model.state_dict(),optimizer=opt.state_dict(),
                        
                        scheduler=dict(step=step,kind=t['lr_schedule']),ema=ema.state_dict(),rng=states,world_size=world,provenance=provenance)
                    tmp=out/'checkpoint.partial';torch.save(payload,tmp);tmp.replace(out/f'step_{step:06d}.pt')
                if world>1:torch.distributed.barrier()
    finally:
        if lock is not None:lock.close()
        if world>1 and torch.distributed.is_initialized():torch.distributed.destroy_process_group()

if __name__=='__main__':main()
