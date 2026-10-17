"""V4 subprocess: native head RGB/state -> 7D joint chunk -> smoothed dual qpos."""
import os
import pickle
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));os.chdir(ROOT)
wire=sys.stdout.buffer;sys.stdout=sys.stderr
import numpy as np
import torch
from trace_robotwin.models.checkpoint import load_policy
from trace_robotwin.data.dataset import normalize,denormalize,head_tensor
from trace_robotwin.eval.actions import bspline_smooth,joint_observation,dual_arm_joint_action


def reply(value):pickle.dump(value,wire,protocol=4);wire.flush()


def main():
    torch.set_num_threads(2);device=sys.argv[2] if len(sys.argv)>2 else 'cuda'
    model,payload=load_policy(sys.argv[1],device=device);c=payload['config'];del payload
    if c['model']['benchmark_version']!='robotwin_v4' or c['data']['adapter']!='robotwin_motif20_v4':raise ValueError('Requires V4 checkpoint')
    if c['environment']['action_type']!='qpos':raise ValueError('V4 requires joint execution')
    cache={};encoder=None;current=None
    reply(dict(ready=True,horizon=c['model']['policy']['action_horizon'],history=1,model_version='robotwin_v4',action_type='qpos',action_dim=14))
    while True:
        try:req=pickle.load(sys.stdin.buffer)
        except EOFError:return
        if req['op']=='reset':
            current=None;torch.manual_seed(req['seed']);phrase=req['instruction'];arm=req['arm']
            if arm not in ('left','right'):raise ValueError('Explicit single active arm required')
            if phrase not in cache:
                if encoder is None:
                    from sentence_transformers import SentenceTransformer
                    encoder=SentenceTransformer(str(ROOT/'models/all-MiniLM-L6-v2'),local_files_only=True,device='cpu').eval()
                cache[phrase]=encoder.encode([phrase],convert_to_tensor=True,show_progress_bar=False).detach().to(device)
            language=cache[phrase];reply(True)
        elif req['op']=='obs':
            obs=req['obs'];ep=obs['endpose']
            image=head_tensor(np.asarray(obs['observation']['head_camera']['rgb']))[None,None].to(device)
            prop=torch.tensor(np.r_[ep[arm+'_endpose'],ep[arm+'_gripper']],dtype=torch.float32)
            current=(image,normalize(prop,c['data']['proprio_stats'])[None,None].to(device),joint_observation(obs));reply(True)
        elif req['op']=='action':
            if current is None:raise ValueError('Observation required after reset')
            image,prop,joints=current
            with torch.inference_mode():
                condition,_=model.condition(model.encode(image),prop,language)
                predicted=model.policy.sample(condition)['actions'][0]
                active=denormalize(predicted.float(),c['data']['action_stats']).cpu().numpy()
            smooth=c['environment']['action_smoothing']
            if smooth['type']!='bspline':raise ValueError('V4 postprocessing requires B-spline')
            active=bspline_smooth(active,smooth['degree'],smooth['num_ctrl_pts'])
            reply(dual_arm_joint_action(active,joints,arm))
        else:raise ValueError('Unknown operation')


if __name__=='__main__':main()
