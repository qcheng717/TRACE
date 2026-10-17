"""Create immutable V4 frame-index manifest and global min/max from OUR episodes."""
import argparse
import os
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import h5py
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trace_robotwin.data.dataset import joint_vector,joint_outlier,stationary_keep_indices
from trace_robotwin.data.schema import canonical,validate_episode


def prepare(source,data_root,output,threshold=1e-6,margin=1):
    source=Path(source);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    raw=source.read_bytes();entries=[json.loads(s) for s in raw.splitlines() if s]
    accepted=[];excluded=[];counts=Counter();removed=0;states=[];actions=[];warnings=[]
    for e in entries:
        with h5py.File(Path(data_root)/e['hdf5_path'],'r') as f:
            n=validate_episode(f)
            if n!=e['num_steps']:raise ValueError('Source manifest length mismatch')
            both=np.concatenate([joint_vector(f,a,'action') for a in ('left','right')],-1)
            if joint_outlier(both):
                excluded.append(dict(hdf5_path=e['hdf5_path'],reason='joint_action_outside_plus_minus_pi'));continue
            keep=stationary_keep_indices(both,threshold,margin)
            state=canonical(f,e['active_arm'],'state',keep);action=joint_vector(f,e['active_arm'],'action',keep)
            if (np.abs(state)>5).any():warnings.append(e['hdf5_path'])
        accepted.append(dict(e,raw_num_steps=n,num_steps=len(keep),frame_indices=keep.tolist()))
        states.append(state);actions.append(action);counts[e['task_name']]+=1;removed+=n-len(keep)
    if not accepted:raise ValueError('All episodes excluded')
    manifest=''.join(json.dumps(e,ensure_ascii=False)+'\n' for e in accepted).encode()
    def bounds(xs):
        x=np.concatenate(xs)
        return dict(min=x.min(0).tolist(),max=x.max(0).tolist(),dim=x.shape[-1])
    stats=dict(format='robotwin_v4_global_minmax_v1',manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        source_manifest_sha256=hashlib.sha256(raw).hexdigest(),state=bounds(states),action=bounds(actions),
        normalization='2*(x-min)/where(max-min<1e-6,1,max-min)-1; no clipping',
        action_representation='active_arm_absolute_6_joint_angles_gripper',state_representation='active_world_ee_xyz_wxyz_gripper',
        reference_commit='339860f65d5cde9e229c037b672889bc4f70497c',preprocessing=dict(
            stationary_threshold=threshold,stationary_margin=margin,min_duration=1,stationary_vector='both_arms_14d_joint_action',
            joint_outlier_limit=float(np.pi),timeline='compacted, all fields share frame_indices',raw_files_modified=False),
        source_episodes=len(entries),accepted_episodes=len(accepted),excluded_episodes=excluded,
        source_frames=sum(e['num_steps'] for e in entries),accepted_frames=sum(e['num_steps'] for e in accepted),
        removed_stationary_frames=removed,episodes_per_task=dict(counts),ee_abs_over_5_warnings=warnings)
    (output/'manifest.jsonl').write_bytes(manifest)
    (output/'norm_stats.json').write_text(json.dumps(stats,indent=2)+'\n')
    return stats


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--source',default=str(ROOT/'metadata/robotwin/source_manifest.jsonl'))
    p.add_argument('--data-root',default=os.environ.get('ROBOTWIN_DATA_ROOT',str(ROOT/'third_party/RoboTwin/data')))
    p.add_argument('--output',default=str(ROOT/'metadata/robotwin'))
    p.add_argument('--threshold',type=float,default=1e-6);p.add_argument('--margin',type=int,default=1)
    a=p.parse_args();s=prepare(a.source,a.data_root,a.output,a.threshold,a.margin)
    print(json.dumps({k:s[k] for k in ('source_episodes','accepted_episodes','source_frames','accepted_frames','removed_stationary_frames','episodes_per_task')},indent=2))
