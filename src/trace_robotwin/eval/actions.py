"""V4 native joint commands and LiLa-WAM B-spline postprocessing."""
import numpy as np
from scipy.interpolate import make_lsq_spline


def bspline_smooth(action_seq,degree=3,num_ctrl_pts=8):
    """Reference robotwin_infer.py algorithm, applied to ALL 7 dimensions."""
    x=np.asarray(action_seq)
    if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('Expected finite action chunk')
    n,_=x.shape
    if n<=num_ctrl_pts:return x
    times=np.arange(n)
    internal=np.linspace(0,n-1,num_ctrl_pts-degree+2)[1:-1]
    knots=np.concatenate(([0]*(degree+1),internal,[n-1]*(degree+1)))
    return make_lsq_spline(times,x,knots,k=degree)(times)


def joint_observation(obs):
    """Use explicit arm fields; never infer joint actions from end-effector pose."""
    joint=obs['joint_action']
    result=np.concatenate([np.r_[joint[a+'_arm'],joint[a+'_gripper']] for a in ('left','right')])
    if result.shape!=(14,) or not np.isfinite(result).all():raise ValueError('Invalid 14D native joint observation')
    return result


def dual_arm_joint_action(active,current,arm):
    active=np.asarray(active);current=np.asarray(current)
    if arm not in ('left','right') or active.shape[-1]!=7 or current.shape!=(14,):raise ValueError('Invalid single-arm qpos bridge')
    result=np.broadcast_to(current,(*active.shape[:-1],14)).copy()
    start=0 if arm=='left' else 7
    result[...,start:start+7]=active
    if not np.isfinite(result).all():raise ValueError('Nonfinite native qpos command')
    return result
