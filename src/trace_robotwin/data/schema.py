"""Validated XPolicyLab v1.0 fields, not positional guesses on a 14D vector.

Sources: RoboTwin envs/utils/pkl2hdf5.py:create_xpolicylab_hdf5 and
robot/robot.py:_trans_endpose; XPolicyLab utils/robot/_robot_info.json.
state[t]=captured pose t, action[t]=captured pose t+1, images[t]=capture t.
EE poses are world-frame xyz + transforms3d wxyz, already robot-frame corrected.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
import numpy as np

ARMS = ('left', 'right')
JOINT_DIM = 6  # aloha_agilex arm_dim=[6,6], ee_dim=[1,1] in official robot info
GRIPPER_DIM = 1
POSE_DIM = 7  # xyz + quaternion wxyz, not Euler
CANONICAL_DIM = POSE_DIM + GRIPPER_DIM
CAMERAS = dict(head='cam_head', left='cam_left_wrist', right='cam_right_wrist')


def roots():
    robot = Path(os.environ['ROBOTWIN_ROOT']).expanduser().resolve()
    return robot, Path(os.environ.get('ROBOTWIN_DATA_ROOT', robot/'data')).expanduser().resolve()


def decoder():
    robot, _ = roots()
    if str(robot) not in sys.path:
        sys.path.insert(0, str(robot))
    try:
        from XPolicyLab.utils.process_data import decode_image_bit
        return decode_image_bit
    except ModuleNotFoundError as e:
        if not (e.name or '').startswith('XPolicyLab'):
            raise  # Missing dependency inside installed XPolicyLab is a real error.
        path = robot/'data/decode_image_bit.py'
        spec = importlib.util.spec_from_file_location('_robotwin_official_codec', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.decode_image_bit


def rgb_frame(f, camera, index, decode):
    rgb = np.asarray(decode(f[f'vision/{camera}/colors'][index]))
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f'Expected RGB uint8 HWC, got {rgb.shape}/{rgb.dtype}')
    return rgb  # official helper returns RGB for both legacy and marked JPEG


def numeric_episode(path):
    return int(path.stem.split('_')[-1])


def episodes(root, task, limit=50):
    return sorted((root/'demo_clean'/task/'aloha_agilex/data').glob('episode_*.hdf5'), key=numeric_episode)[:limit]


def instructions(f):
    if 'instructions' not in f:
        return []
    raw = f['instructions'][()]
    if isinstance(raw, bytes): raw = raw.decode('utf-8')
    values = json.loads(raw) if isinstance(raw, str) else raw.tolist()
    if isinstance(values, str): values = [values]
    if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
        raise ValueError('Unsupported instruction schema')
    return list(dict.fromkeys(v for v in values if v.strip()))


def episode_instructions(f, path, task, robot_root):
    """Prefer recorded language, then official per-episode sidecar/task text.

    Unresolved {object}/{arm} templates are never invented or substituted.
    """
    values=instructions(f)
    if values: return values,'hdf5/instructions'
    sidecar=Path(path).parent.parent/'instruction'/(Path(path).stem+'.json')
    if sidecar.exists():
        payload=json.loads(sidecar.read_text())
        if isinstance(payload,dict):
            values=[]
            for key in ('instructions','seen','unseen'):
                v=payload.get(key,[]);values.extend([v] if isinstance(v,str) else v)
        else: values=[payload] if isinstance(payload,str) else payload
        values=[v for v in values if isinstance(v,str) and v.strip() and '{' not in v]
        if values:return list(dict.fromkeys(values)),'official episode instruction sidecar'
    definition=Path(robot_root)/'description/task_instruction'/f'{task}.json'
    if definition.exists():
        payload=json.loads(definition.read_text());values=[]
        for key in ('seen','unseen'):
            values.extend(v for v in payload.get(key,[]) if isinstance(v,str) and v.strip() and '{' not in v)
        if not values:
            text=payload.get('full_description','')
            if isinstance(text,str) and text.strip() and '{' not in text:values=[text]
        if values:return list(dict.fromkeys(values)),'official task definition (no episode language available)'
    raise ValueError('No resolved official instruction available')


def validate_episode(f):
    version=f['data_format_version'][()]
    if isinstance(version,bytes): version=version.decode()
    if version!='v1.0': raise ValueError('Unsupported HDF5 format version')
    n = len(f['state/left_arm_joint_states'])
    if n < 2: raise ValueError('Episode too short to audit motion')
    for arm in ARMS:
        for field, dim in ((f'{arm}_arm_joint_states',JOINT_DIM),
                           (f'{arm}_ee_joint_states',GRIPPER_DIM), (f'{arm}_ee_poses',POSE_DIM)):
            s, a = (np.asarray(f[f'{kind}/{field}']) for kind in ('state','action'))
            if s.shape != (n,dim) or a.shape != s.shape:
                raise ValueError(f'{field}: unexpected state/action dimensions')
            if not np.isfinite(s).all() or not np.isfinite(a).all():
                raise ValueError(f'{field}: nonfinite values')
            if field.endswith('ee_joint_states') and (s.min()<0 or s.max()>1 or a.min()<0 or a.max()>1):
                raise ValueError('Expected normalized RoboTwin gripper in [0,1]')
            if not np.allclose(s[1:],a[:-1],rtol=0,atol=1e-6):
                raise ValueError(f'{field}: not verified next-capture temporal alignment')
        for kind in ('state','action'):
            q = f[f'{kind}/{arm}_ee_poses'][:,3:POSE_DIM]
            if not np.allclose(np.linalg.norm(q,axis=-1),1,atol=1e-3):
                raise ValueError('Invalid wxyz quaternion')
    for camera in CAMERAS.values():
        if len(f[f'vision/{camera}/colors']) != n: raise ValueError('Camera length mismatch')
    return n


def canonical(f, arm, kind, selection=slice(None)):
    if arm not in ARMS or kind not in ('state','action'): raise ValueError('Invalid selection')
    pose = np.asarray(f[f'{kind}/{arm}_ee_poses'][selection], dtype=np.float32)
    grip = np.asarray(f[f'{kind}/{arm}_ee_joint_states'][selection],dtype=np.float32)
    result = np.concatenate((pose,grip),axis=-1)
    if result.shape[1] != CANONICAL_DIM or not np.isfinite(result).all():
        raise ValueError('Invalid canonical pose')
    return result
