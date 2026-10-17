import copy
import os
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
import h5py
from trace_robotwin.config import load_config
from trace_robotwin.data.dataset import (normalize,denormalize,stationary_keep_indices,
    joint_outlier,joint_vector,head_tensor,RoboTwinDataset)
from trace_robotwin.models.model import TRACE
from trace_robotwin.models.components import SpatialBlock,partition,unpartition
from trace_robotwin.eval.actions import bspline_smooth,joint_observation,dual_arm_joint_action

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def test_minmax_roundtrip_no_clip_constant_dimensions():
    stats=dict(min=[-2,4,0],max=[2,4,1])
    x=torch.tensor([[-2,4,0],[2,4,1],[3,4.5,.3]])
    norm=normalize(x,stats)
    assert torch.equal(norm[:2],torch.tensor([[-1.,-1,-1],[1,-1,1]]))
    assert norm[-1,0]>1
    assert torch.allclose(denormalize(norm,stats),x,atol=1e-6)


def test_stationary_deletion_preserves_gripper_changes_both_arms():
    actions=np.zeros((12,14));actions[6:,13]=1
    assert stationary_keep_indices(actions).tolist()==[0,5,6,11]
    assert stationary_keep_indices(actions,margin=3).tolist()==list(range(12))
    assert not joint_outlier(actions)
    actions[0,6]=100  # gripper is not a joint outlier dimension
    assert not joint_outlier(actions)
    actions[0,8]=np.pi+.01
    assert joint_outlier(actions)


def test_native_head_no_resize_or_wrist():
    rgb=np.arange(240*320*3,dtype=np.uint8).reshape(240,320,3)
    image=head_tensor(rgb)
    assert image.shape==(3,240,320)
    assert torch.equal(image,torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255)
    with pytest.raises(ValueError):head_tensor(rgb[:224,:224])


def test_rectangular_windows_camera_isolation_and_gradients():
    x=torch.arange(2*8*12*3).reshape(2,8,12,3)
    assert torch.equal(unpartition(partition(x,4),4,(8,12),2),x)
    for shift in (0,2):
        block=SpatialBlock(8,2,(6,9),shift=shift)
        x=torch.randn(2,2,6,9,8,requires_grad=True)
        y=block(x);assert torch.isfinite(y).all()
        y.square().mean().backward();assert torch.isfinite(x.grad).all()
        other=x.detach().clone();other[:,1]+=100
        assert torch.allclose(y[:,0],block(other)[:,0],atol=1e-6)


class RectangularBackbone(nn.Module):
    feature_dims={str(i):16 for i in (2,5,8,11)}
    frame_batch_size=16
    def __init__(self):super().__init__();self.conv=nn.Conv2d(3,16,1);self.seen=[]
    def forward(self,x):
        self.seen.append(tuple(x.shape))
        x=self.conv(torch.nn.functional.adaptive_avg_pool2d(x,(6,9)))
        return {k:x*(i+1) for i,k in enumerate(self.feature_dims)}


def test_v4_two_stage_single_camera_7d_actions():
    c=load_config(ROOT/'configs/stage1.yaml')
    c['model']['v10'].update(grid_shape=[6,9],semantic_dim=32,skill_encoder_dim=32,future_decoder_dim=32,
        future_decoder_heads=4,future_decoder_ffn_dim=64,selector_heads=4,router_dim=32)
    c['model']['policy'].update(hidden_dim=32,layers=2,heads=4,diffusion_steps=10,inference_steps=2)
    backbone=RectangularBackbone();m=TRACE(c['model'],backbone)
    b=dict(images=torch.rand(2,1,3,24,32),wrist=torch.empty(2,0,3,24,32),future_images=torch.rand(2,1,3,24,32),
        future_wrist=torch.empty(2,0,3,24,32),future_valid_mask=torch.ones(2,1,dtype=torch.bool),
        proprio=torch.rand(2,1,8),language=torch.rand(2,384),actions=torch.rand(2,32,7),action_mask=torch.ones(2,32,dtype=torch.bool))
    loss,_=m(b);loss.backward();assert torch.isfinite(loss)
    assert m.stage1_outputs(b)['H'].shape==(2,54,32)
    assert all(x==(2,3,24,32) for x in backbone.seen)
    m.zero_grad(set_to_none=True);m.set_stage('policy');loss,metrics=m(b);loss.backward()
    assert torch.isfinite(loss) and 'joint_diffusion' in metrics
    assert all(p.grad is None and not p.requires_grad for p in m.predictor.parameters())
    cond,_=m.condition(m.encode(b['images']),b['proprio'],b['language'])
    assert m.policy.sample(cond)['actions'].shape==(2,32,7)


@pytest.mark.skipif(not (ROOT/"metadata/robotwin/manifest.jsonl").exists(), reason="Requires prepared RoboTwin data")
def test_real_data_cleaned_t32_tail_and_next_capture_joint_labels(monkeypatch):
    monkeypatch.setenv('ROBOTWIN_ROOT',os.environ.get('ROBOTWIN_ROOT',str(ROOT/'third_party/RoboTwin')))
    monkeypatch.setenv('ROBOTWIN_DATA_ROOT',os.environ.get('ROBOTWIN_DATA_ROOT',str(Path(os.environ['ROBOTWIN_ROOT'])/'data')))
    c=load_config(ROOT/'configs/stage1.yaml');ds=RoboTwinDataset(c,'stage1')
    e=ds.entries[0];indices=np.asarray(e['frame_indices']);n=e['num_steps']
    with h5py.File(ds.root/e['hdf5_path'],'r') as f:
        for anchor in (0,n-33,n-2,n-1):
            sample=ds.get_window(0,anchor)
            assert sample['future_images'].shape==(1,3,240,320)
            assert sample['wrist'].numel()==0 and sample['future_wrist'].numel()==0
            assert sample['future_valid_mask'].item()
            assert sample['raw_future_anchor'].item()==indices[min(anchor+32,n-1)]
            positions=indices[np.minimum(np.arange(anchor,anchor+32),n-1)]
            expected=joint_vector(f,e['active_arm'],'action')[positions]
            assert np.allclose(denormalize(sample['actions'],ds.stats['action']),expected,atol=4e-7)
            assert sample['action_mask'].all()
            assert sample['action_valid_mask'].sum()==min(32,n-anchor)
            if anchor==n-1:
                assert torch.equal(sample['images'],sample['future_images'])
                assert torch.equal(sample['actions'],sample['actions'][0].expand_as(sample['actions']))
    counts=np.zeros(len(ds.entries),dtype=int);rng=np.random.default_rng(4)
    for _ in range(5000):i,a=ds.sample_anchor(rng);counts[i]+=1;assert a<ds.entries[i]['num_steps']
    assert counts.sum()==5000


def test_bspline_all_dimensions_and_inactive_joint_bridge():
    time=np.arange(32,dtype=np.float64);x=np.stack([time**2,3*time-2,np.ones(32),time**3,time/31,-time,time/31],-1)
    assert np.allclose(bspline_smooth(x),x,atol=1e-9)  # cubic reproduction incl gripper
    noisy=np.random.default_rng(5).normal(size=(32,7))
    smooth=bspline_smooth(noisy)
    assert smooth.shape==noisy.shape and np.isfinite(smooth).all()
    assert not np.array_equal(smooth[:,-1],noisy[:,-1])
    assert np.array_equal(bspline_smooth(noisy[:8]),noisy[:8])
    obs=dict(joint_action=dict(left_arm=list(range(6)),left_gripper=.4,right_arm=list(range(6,12)),right_gripper=.8))
    current=joint_observation(obs)
    for arm,start in [('left',0),('right',7)]:
        native=dual_arm_joint_action(smooth,current,arm)
        assert np.array_equal(native[:,start:start+7],smooth)
        other=7-start;assert np.array_equal(native[:,other:other+7],np.broadcast_to(current[other:other+7],(32,7)))
