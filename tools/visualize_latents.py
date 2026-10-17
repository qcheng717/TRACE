"""Real-sample CAVE feature visualization; defaults to checkpoint's dictionary folder."""
import argparse,copy,json,sys,os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import default_collate
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from trace_robotwin.data.dataset import RoboTwinDataset
from trace_robotwin.models.model import TRACE
from trace_robotwin.models.checkpoint import validate


def visualization_data(config,stage='policy'):
    data=RoboTwinDataset(config,stage,end_step=1)
    grouped={}
    for index,entry in enumerate(data.entries):grouped.setdefault(entry['task_name'],[]).append(index)
    return data,[(name,np.asarray(indices)) for name,indices in grouped.items()],[e['num_steps'] for e in data.entries]


def pool_spatial_tokens(x,selector,pool_grid):
    if not pool_grid:return x.mean(1)
    b,n,d=x.shape
    h,w=selector.g if isinstance(selector.g,(tuple,list)) else (selector.g,selector.g)
    cameras=selector.cameras
    if n!=cameras*h*w:raise ValueError('Unexpected camera count or spatial shape')
    grid=x.reshape(b,cameras,h,w,d).permute(0,1,4,2,3).reshape(b*cameras,d,h,w)
    pooled=torch.nn.functional.adaptive_avg_pool2d(grid,(pool_grid,pool_grid))
    return pooled.reshape(b,cameras,d,pool_grid,pool_grid).permute(0,1,3,4,2).flatten(1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--samples-per-task',type=int,default=20);p.add_argument('--batch-size',type=int,default=4);p.add_argument('--device',default='cpu');p.add_argument('--output');group=p.add_mutually_exclusive_group();group.add_argument('--num-tasks',type=int);group.add_argument('--tasks',nargs='+');p.add_argument('--pool-grid',type=int,default=0,help='0: global mean; 1/2/4/8/16: separate-camera spatial pooling');p.add_argument('--extract-only',action='store_true');a=p.parse_args()
    if a.pool_grid not in (0,1,2,4,8,16):p.error('Invalid pool grid')
    os.chdir(ROOT);os.environ.setdefault('ROBOTWIN_ROOT',str(ROOT/'third_party/RoboTwin'))
    torch.set_num_threads(2);torch.manual_seed(42)
    path=Path(a.checkpoint).resolve();ck=torch.load(path,map_location='cpu',weights_only=False,mmap=True);validate(ck)
    if ck['stage'] not in ('stage1','policy'):raise ValueError('Requires Stage-I or policy checkpoint')
    out=Path(a.output) if a.output else path.parent/'dictionary'/f'cave_latents_step{ck["step"]:06d}'
    out.mkdir(parents=True,exist_ok=True)
    c=copy.deepcopy(ck['config']);c['data']['augment']=False
    data,named,lengths=visualization_data(c)
    model=TRACE(c['model'],stage=ck['stage']).to(a.device).eval();model.load_state_dict(ck['model'],strict=True)
    if a.tasks:
        lookup=dict(named)
        if len(set(a.tasks))!=len(a.tasks) or any(n not in lookup for n in a.tasks):
            p.error('Duplicate or unknown task. Available: '+', '.join(lookup))
        named=[(n,lookup[n]) for n in a.tasks]
    if a.num_tasks is not None:
        if not 1<=a.num_tasks<=len(named):p.error('--num-tasks outside available range')
        named=named[:a.num_tasks]
    if a.samples_per_task<1 or a.batch_size<1 or len(named)*a.samples_per_task<4:
        p.error('Positive counts and at least four total samples required')
    pools=[pool for _,pool in named]
    print('Selected tasks: '+', '.join(n for n,_ in named),flush=True)
    rng=np.random.default_rng(42);samples=[];meta=[]
    for pool in pools:
        order=rng.permutation(pool)
        for j in range(a.samples_per_task):
            idx=int(order[j%len(order)]);n=lengths[idx]
            anchor=int(rng.integers(n));samples.append((idx,anchor))
    captured={}
    def hook(module,args,value):captured['hidden']=value.detach()
    handle=model.predictor.decoder.norm.register_forward_hook(hook)
    def pool_tokens(x):
        return pool_spatial_tokens(x,model.predictor.selector,a.pool_grid)
    zs=[];hs=[];ats=[]
    with torch.inference_mode():
        for start in range(0,len(samples),a.batch_size):
            items=[data.get_window(i,t,step=0) for i,t in samples[start:start+a.batch_size]]
            for (i,t),item in zip(samples[start:start+a.batch_size],items):
                record=dict(index=i,anchor=t,metadata=item['metadata'])
                if 'raw_anchor' in item:record['raw_anchor']=int(item['raw_anchor'])
                meta.append(record)
            b=default_collate(items)
            r=model.prediction(model.encode(b['images'].to(a.device),b['wrist'].to(a.device)),b['language'].to(a.device))
            zs.append(pool_tokens(r['Z_skill']).cpu().numpy());hs.append(pool_tokens(captured['hidden'].flatten(1,3)).cpu().numpy());ats.append(pool_tokens(r['A']).cpu().numpy())
            if start%40==0:print(f'Extracted {min(start+a.batch_size,len(samples))}/{len(samples)}',flush=True)
    handle.remove()
    np.savez_compressed(out/'features.npz',z_skill=np.concatenate(zs),decoder_hidden=np.concatenate(hs),assignment=np.concatenate(ats))
    (out/'samples.json').write_text(json.dumps(meta,indent=2,default=str))
    labels=np.repeat(np.arange(len(pools)),a.samples_per_task)
    tasks=[]
    for k in range(len(pools)):
        m=meta[k*a.samples_per_task]['metadata'];tasks.append(str(m.get('task_name',m.get('task',f'task_{k}'))))
    selector=model.predictor.selector
    shape=selector.g if isinstance(selector.g,(tuple,list)) else (selector.g,selector.g)
    tokens=selector.cameras*shape[0]*shape[1]
    pooling=f'per-camera {a.pool_grid}x{a.pool_grid} spatial pooling' if a.pool_grid else f'mean over all {tokens} spatial tokens; sample-level task visualization'
    info=dict(checkpoint=str(path),step=ck['step'],pooling=pooling,tasks=tasks,labels=labels.tolist(),
              benchmark_version=c['model']['benchmark_version'],adapter=c['data']['adapter'],
              camera_views=selector.cameras,grid_shape=list(shape),seed=42,weights='raw',
              samples_per_task=a.samples_per_task,stage=ck['stage'],
              split_unit='episode')
    if c['data']['adapter'] in ('robotwin_motif20_v4','real_robot_v1'):info['manifest_sha256']=data.stats['manifest_sha256']
    (out/'feature_metadata.json').write_text(json.dumps(info,indent=2))
    if a.extract_only:
        print(out/'features.npz',flush=True);return
    from sklearn.manifold import TSNE
    projections={}
    for name,values in [('z_skill',zs),('decoder_hidden',hs)]:
        x=np.concatenate(values).reshape(len(samples),-1).astype(np.float32)
        print('Fitting 2D t-SNE: '+name,flush=True)
        xy=TSNE(n_components=2,perplexity=min(30,(len(x)-1)/3),init='pca',learning_rate='auto',random_state=42,max_iter=1500,n_jobs=2).fit_transform(x)
        projections[name]=xy.tolist()
    (out/'projection.json').write_text(json.dumps(dict(**info,projections=projections,dimensions=2,method='t-SNE',metric='euclidean',perplexity=min(30,(len(samples)-1)/3),iterations=1500,init='pca',learning_rate='auto'),indent=2))
    from render_latents import render
    render(out)
    print(out/'index.html',flush=True)
if __name__=='__main__':main()
