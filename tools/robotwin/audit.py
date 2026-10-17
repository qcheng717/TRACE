"""Build the single-arm source manifest from the first 50 episodes per task."""
import argparse,hashlib,json,sys
from pathlib import Path
import h5py
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trace_robotwin.data.schema import ARMS,episodes,episode_instructions,roots,validate_episode
from trace_robotwin.data.tasks import ROBOTWIN_TASKS,TASK_IDS,TASK_TO_GROUP
from trace_robotwin.data.audit import calibrate,classify

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',default='metadata/robotwin');a=p.parse_args()
 robot,root=roots();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);rows=[]
 for task in ROBOTWIN_TASKS:
  files=episodes(root,task)
  if not files:raise FileNotFoundError(f'No episodes for {task} under {root}')
  for path in files:
   row=dict(task=task,episode=path.stem,hdf5_path=str(path.relative_to(root)))
   with h5py.File(path) as f:
    row['num_steps']=validate_episode(f)
    for arm in ARMS:
     for field,suffix in [('arm_joint_states','motion'),('ee_joint_states','gripper_activity')]:
      s=f[f'state/{arm}_{field}'][:];last=f[f'action/{arm}_{field}'][-1:]
      row[f'{arm}_{suffix}']=float(np.abs(np.diff(np.concatenate((s,last)),axis=0)).mean())
    row['instructions'],row['instruction_source']=episode_instructions(f,path,task,robot)
    if not row['instructions']:raise ValueError(f'Missing instruction: {path}')
   rows.append(row)
 calibration={metric:calibrate([r[f'{arm}_{suffix}'] for r in rows for arm in ARMS]) for metric,suffix in [('joint','motion'),('gripper','gripper_activity')]}
 accepted=[]
 for row in rows:
  arm,reason=classify(row,calibration);row.update(active_arm=arm,reason=reason)
  if arm not in ARMS:continue
  task=row['task'];path=root/row['hdf5_path']
  accepted.append(dict(task_name=task,task_id=TASK_IDS[task],task_group=TASK_TO_GROUP[task],episode_id=row['episode'],hdf5_path=row['hdf5_path'],num_steps=row['num_steps'],active_arm=arm,active_arm_id=ARMS.index(arm),instruction=row['instructions'][0],instructions=row['instructions'],instruction_source=row['instruction_source'],is_single_arm=True,raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
 if not accepted:raise ValueError('No unambiguous single-arm episodes')
 (out/'source_manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in accepted))
 (out/'audit.json').write_text(json.dumps(dict(calibration=calibration,episodes=rows,accepted=len(accepted)),indent=2))
 print(f'{len(accepted)}/{len(rows)} single-arm episodes accepted')
if __name__=='__main__':main()
