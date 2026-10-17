"""RoboTwin V4: native head RGB, single-arm joint targets, compacted timeline.

Reference: LiLa-WAM 339860f, clean_dataset_stationary.py / model_runner.py.
Raw HDF5 is immutable; frame_indices applies the same deletion to every field.
The verified XPolicyLab action[t] already targets the next capture: no new shift.
"""
import hashlib
import json
import os
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
from trace_robotwin.data.schema import ARMS, canonical, decoder, rgb_frame


def joint_vector(f,arm,kind,selection=slice(None)):
    if arm not in ARMS or kind not in ('state','action'):raise ValueError('Invalid joint selection')
    joints=np.asarray(f[f'{kind}/{arm}_arm_joint_states'][selection],dtype=np.float32)
    grip=np.asarray(f[f'{kind}/{arm}_ee_joint_states'][selection],dtype=np.float32)
    x=np.concatenate((joints,grip),axis=-1)
    if x.shape[-1]!=7 or not np.isfinite(x).all():raise ValueError('Expected finite 6 joints + gripper')
    return x


def stationary_keep_indices(actions,threshold=1e-6,margin=1,min_duration=1):
    """Exact half-open deletion intervals from the reference CLI (margin=1)."""
    if actions.ndim!=2 or not np.isfinite(actions).all():raise ValueError('Invalid actions')
    if threshold<=0 or margin<1 or min_duration<1:raise ValueError('Invalid stationary settings')
    flags=np.abs(np.diff(actions,axis=0)).max(-1)<threshold
    keep=np.ones(len(actions),dtype=bool);i=0
    while i<len(flags):
        if not flags[i]:i+=1;continue
        j=i
        while j<len(flags) and flags[j]:j+=1
        if j-i+1>=min_duration and i+margin<j+1-margin:keep[i+margin:j+1-margin]=False
        i=j
    return np.flatnonzero(keep)


def joint_outlier(actions):
    """Reference filters entire episodes using both arms, excluding grippers."""
    if actions.ndim!=2 or actions.shape[-1]!=14:raise ValueError('Expected dual-arm joint vector')
    return bool((np.abs(actions[:,[0,1,2,3,4,5,7,8,9,10,11,12]])>np.pi).any())


def normalization_bounds(x,stats):
    low=x.new_tensor(stats['min']);high=x.new_tensor(stats['max'])
    if low.shape!=(x.shape[-1],) or high.shape!=low.shape:raise ValueError('Min/max dimension mismatch')
    span=high-low
    if not torch.isfinite(low).all() or not torch.isfinite(high).all() or (span<0).any():raise ValueError('Invalid min/max')
    return low,torch.where(span<1e-6,torch.ones_like(span),span)


def normalize(x,stats):
    low,span=normalization_bounds(x,stats)
    return 2*(x-low)/span-1  # No percentile clipping, including gripper.


def denormalize(x,stats):
    low,span=normalization_bounds(x,stats)
    return (x+1)/2*span+low


def head_tensor(rgb):
    if rgb.shape!=(240,320,3) or rgb.dtype!=np.uint8:raise ValueError('V4 requires native RGB 320x240')
    return torch.from_numpy(np.array(rgb,copy=True)).permute(2,0,1).float()/255


class RoboTwinDataset(IterableDataset):
    def __init__(self,config,stage,start_step=0,end_step=None):
        super().__init__();self.config=config;self.stage=stage
        c=config['data'];m=config['model'];t=config['train']
        if stage not in ('stage1','policy') or m['benchmark_version']!='robotwin_v4':raise ValueError('Explicit V4 stage required')
        if c['augment'] or c['image_size']!=[320,240] or c['cameras']!=['head'] or c['anchor_weights'] is not None:raise ValueError('V4 requires native head, no augmentation or pause weights')
        if m['policy']['action_dim']!=7 or m['proprio_dim']!=8 or m['v10']['camera_views']!=1:raise ValueError('V4 requires 7D joints, 8D EE state, one camera')
        if c['action_tail_mode']!='repeat_last' or m['v10']['future_tail_mode']!='last_frame':raise ValueError('V4 tails repeat last frame/action')
        self.root=Path(os.path.expandvars(c['data_root'])).expanduser().resolve()
        raw=Path(c['manifest']).read_bytes();self.entries=[json.loads(s) for s in raw.splitlines() if s]
        self.stats=json.loads(Path(c['norm_stats']).read_text())
        if not self.entries or self.stats['manifest_sha256']!=hashlib.sha256(raw).hexdigest():raise ValueError('Empty or mismatched V4 manifest/stats')
        if self.stats['action']['dim']!=7 or self.stats['state']['dim']!=8:raise ValueError('Wrong V4 statistics')
        for e in self.entries:
            indices=np.asarray(e['frame_indices'])
            if len(indices)!=e['num_steps'] or indices[0]!=0 or indices[-1]!=e['raw_num_steps']-1 or (np.diff(indices)<=0).any():raise ValueError('Invalid compacted timeline')
        self.cdf=np.cumsum([e['num_steps'] for e in self.entries]);self.total_frames=int(self.cdf[-1])
        with np.load(c['language_cache']) as cache:self.language=dict(zip(cache['phrases'].tolist(),cache['embeddings']))
        for e in self.entries:
            if not e['instructions'] or any(p not in self.language for p in e['instructions']):raise ValueError('Missing real instruction embedding')
        self.seed=t['seed'];self.batch_size=t['batch_size']*t['gradient_accumulation_steps']
        self.start_step=start_step;self.end_step=t['max_steps'] if end_step is None else end_step
        self.offset=m['v10']['future_target_offset'];self.horizon=m['policy']['action_horizon'];self.decode=None

    def sample_anchor(self,rng):
        flat=int(rng.integers(self.total_frames));index=int(np.searchsorted(self.cdf,flat,side='right'))
        return index,flat-(int(self.cdf[index-1]) if index else 0)

    def get_window(self,index,anchor,step=0):
        e=self.entries[index];n=e['num_steps'];indices=np.asarray(e['frame_indices']);arm=e['active_arm']
        if not 0<=anchor<n:raise IndexError('Anchor outside episode')
        if self.decode is None:self.decode=decoder()
        positions=np.minimum(np.arange(anchor,anchor+self.horizon),n-1)
        future=min(anchor+self.offset,n-1);phrase=e['instructions'][(anchor+step)%len(e['instructions'])]
        with h5py.File(self.root/e['hdf5_path'],'r') as f:
            if len(f['state/left_arm_joint_states'])!=e['raw_num_steps']:raise ValueError('Source episode length changed')
            image=head_tensor(rgb_frame(f,'cam_head',int(indices[anchor]),self.decode))[None]
            prop=torch.from_numpy(canonical(f,arm,'state',slice(int(indices[anchor]),int(indices[anchor])+1)))
            # h5py requires unique sorted selection; repeat endpoint only after reading.
            unique,inverse=np.unique(indices[positions],return_inverse=True)
            actions=torch.from_numpy(joint_vector(f,arm,'action',unique)[inverse])
            sample=dict(images=image,wrist=torch.empty(0,3,240,320),proprio=normalize(prop,self.stats['state']),
                language=torch.tensor(self.language[phrase],dtype=torch.float32),valid=torch.tensor([True]),
                actions=normalize(actions,self.stats['action']),action_mask=torch.ones(self.horizon,dtype=torch.bool),
                action_valid_mask=torch.arange(self.horizon)<n-anchor,segment=torch.tensor(index),anchor=torch.tensor(anchor),
                raw_anchor=torch.tensor(int(indices[anchor])),task=torch.tensor(e['task_id']),
                metadata=dict(task_name=e['task_name'],episode_id=e['episode_id'],active_arm=arm,instruction=phrase))
            if self.stage=='stage1':
                sample.update(future_images=head_tensor(rgb_frame(f,'cam_head',int(indices[future]),self.decode))[None],
                    future_wrist=torch.empty(0,3,240,320),future_valid_mask=torch.tensor([True]),
                    future_clamped=torch.tensor([anchor+self.offset>=n]),future_actual_offset=torch.tensor([future-anchor]),
                    raw_future_anchor=torch.tensor(int(indices[future])))
        return sample

    def __iter__(self):
        worker=get_worker_info();wid,workers=(worker.id,worker.num_workers) if worker else (0,1)
        rank=int(os.environ.get('RANK',0))
        for step in range(self.start_step+wid,self.end_step,workers):
            for slot in range(self.batch_size):
                rng=np.random.default_rng(np.random.SeedSequence([self.seed,step,rank,slot]))
                yield self.get_window(*self.sample_anchor(rng),step=step)
