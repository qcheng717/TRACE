"""PCA/Gram, shared-coordinate evolution, and global-to-local arrows."""
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def normalized(x):
    x=x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def pca(x):
    mean=x.mean(0);_,_,v=np.linalg.svd(x-mean,full_matrices=False)
    return lambda a:(a-mean)@v[:2].T


def draw(mode,paths,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    records=[torch.load(p,map_location='cpu',weights_only=False) for p in paths]
    if mode=='snapshot':
        for record in records:
            d=normalized(record['D_normalized']);xy=pca(d)(d);step=record['step'];labels=[f'd{i+1}' for i in range(len(d))]
            fig,ax=plt.subplots(figsize=(8,6));u=record['usage_ema'].numpy();ax.scatter(*xy.T,s=80+700*u/u.max())
            for label,(x,y) in zip(labels,xy):ax.annotate(label,(x,y),xytext=(4,4),textcoords='offset points')
            ax.set(title=f'Dictionary PCA: step {step} (size = usage)',xlabel='PC1',ylabel='PC2');fig.tight_layout()
            fig.savefig(out/f'dictionary_pca_step{step:06d}.png',dpi=150);plt.close(fig)
            fig,ax=plt.subplots(figsize=(8,7));im=ax.imshow(d@d.T,vmin=-1,vmax=1,cmap='coolwarm')
            ax.set(xticks=range(len(d)),yticks=range(len(d)),xticklabels=labels,yticklabels=labels,title=f'Normalized dictionary Gram: step {step}')
            fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(out/f'dictionary_gram_step{step:06d}.png',dpi=150);plt.close(fig)
    elif mode=='evolution':
        records.sort(key=lambda r:r['step']);ds=[normalized(r['D_normalized']) for r in records]
        if len({d.shape for d in ds})!=1:raise ValueError('Basis identities/shapes must agree across snapshots')
        transform=pca(np.concatenate(ds));xy=np.stack([transform(d) for d in ds]);fig,ax=plt.subplots(figsize=(10,8))
        for k in range(xy.shape[1]):
            points=xy[:,k];ax.plot(*points.T,'o-',label=f'd{k+1}',markersize=3);ax.annotate(f'd{k+1}',points[-1])
        ax.set(title=f'One shared PCA: steps {records[0]["step"]} to {records[-1]["step"]}',xlabel='PC1',ylabel='PC2')
        fig.tight_layout();fig.savefig(out/'dictionary_evolution.png',dpi=150);plt.close(fig)
    else:
        for record in records:
            d=normalized(record['D_star']);local=normalized(record['D_local']);transform=pca(np.concatenate([d,local.reshape(-1,d.shape[-1])]))
            xy=transform(d);fig,axes=plt.subplots(1,len(local),figsize=(6*len(local),5),squeeze=False)
            for i,(ax,loc) in enumerate(zip(axes[0],local)):
                target=transform(loc);ax.scatter(*xy.T,c='black',label='Global');ax.scatter(*target.T,c='tab:orange',label='Local')
                for k,(a,b) in enumerate(zip(xy,target)):
                    ax.annotate('',b,xytext=a,arrowprops=dict(arrowstyle='->',color='tab:blue'));ax.annotate(f'd{k+1}',a)
                ax.set(title=f'Sample {i}, step {record["step"]}',xlabel='PC1',ylabel='PC2');ax.legend()
            fig.tight_layout();fig.savefig(out/f'global_to_local_step{record["step"]:06d}.png',dpi=150);plt.close(fig)
    return str(out)


def main(default=None):
    p=argparse.ArgumentParser(__doc__);p.add_argument('--mode',choices=['snapshot','evolution','local'],default=default or 'snapshot');p.add_argument('--input',nargs='+',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    print(draw(a.mode,a.input,a.output))

if __name__=='__main__':main()
