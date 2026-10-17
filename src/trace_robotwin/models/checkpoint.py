"""Strict version boundary and exact Stage-I transfer, with no permissive loading."""
import hashlib
from pathlib import Path
import torch
from trace_robotwin.train.ema import inference_state
from trace_robotwin.models.model import TRACE


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def metadata(config,stage):
    m=config['model'];v=m['v10']
    fields=('dictionary_size','dictionary_dim','semantic_dim','future_horizon',
            'future_decoder_depth','future_decoder_dim','router_dim','router_heads','language_conditioning','final_normalization')
    return dict(model_version=m['model_version'],benchmark_version=m['benchmark_version'],stage1_arch=m['stage1_arch'],stage=stage,
                gripper_mode=v.get('gripper_mode','binary_ce'),**{k:v[k] for k in fields})


def validate(payload,config=None,stage=None):
    if payload.get('config',{}).get('model',{}).get('model_version')!='v10' or 'metadata' not in payload:
        raise ValueError('Requires an explicit V10 checkpoint; legacy V8.4/V2 is incompatible')
    if payload['config']['model']['v10'].get('language_conditioning')!='film':
        raise ValueError('Requires FiLM V10 checkpoint; pre-FiLM V10 is incompatible')
    if payload['config']['model']['v10'].get('final_normalization')!='layer_norm_no_affine':
        raise ValueError('Requires target-stabilized V10 checkpoint with final normalization')
    c=payload['config'] if config is None else config
    stage=payload['stage'] if stage is None else stage
    if payload.get('stage')!=stage:raise ValueError('Checkpoint stage differs')
    if payload.get('metadata')!=metadata(c,stage):raise ValueError('Incompatible checkpoint version/stage/dimensions')
    if payload['config']['model']!=c['model']:raise ValueError('Checkpoint model configuration differs')


def transfer_stage1(model,path,config):
    from copy import deepcopy
    p=torch.load(path,map_location='cpu',weights_only=False);validate(p,stage='stage1')
    source=deepcopy(p['config']['model']);target=deepcopy(config['model'])
    # Policy gripper heads are not transferred from Stage-I. Everything else stays strict.
    for m in (source,target):m['v10'].pop('gripper_mode',None)
    if source!=target:raise ValueError('Stage-I transfer model differs beyond policy gripper mode')
    def semantic_data(data):return {k:v for k,v in data.items() if k not in ('batch_size','epoch_steps')}
    if semantic_data(p['config']['data'])!=semantic_data(config['data']):raise ValueError('Stage-I/II data or normalization differs')
    # Use coherent RAW online/target/predictor weights; no mixing of selector EMA and predictor states.
    state=p['model']; model.predictor.load_state_dict({k.removeprefix('predictor.'):v for k,v in state.items() if k.startswith('predictor.')},strict=True)
    model.backbone.load_state_dict({k.removeprefix('backbone.'):v for k,v in state.items() if k.startswith('backbone.')},strict=True)
    return dict(stage1_checkpoint=str(Path(path).resolve()),stage1_sha256=sha(path),stage1_weights='raw')


def load_policy(path,device='cuda',use_ema=True):
    p=torch.load(path,map_location='cpu',weights_only=False);validate(p,stage='policy')
    if use_ema and 'ema' not in p:raise ValueError('Policy EMA missing')
    model=TRACE(p['config']['model'],stage='policy')
    model.load_state_dict(inference_state(p,use_ema),strict=True)
    model.step=p['step'];model.policy.set_policy_step(p['step'])
    return model.to(device).eval(),p

