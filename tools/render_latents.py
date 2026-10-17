"""Redraw saved CAVE projections without extracting features or fitting t-SNE."""
import json
import html
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(out):
    out=Path(out);data=json.loads((out/'projection.json').read_text())
    tasks=data['tasks'];labels=np.asarray(data['labels']);n=len(tasks)
    # Greedily separate colors in perceptual Lab space; exclude near-white colors.
    import colorsys
    rgb=np.array([colorsys.hsv_to_rgb(h,s,v) for h in np.linspace(0,1,180,endpoint=False)
                  for s in (.55,.75,.95) for v in (.55,.72,.9)])
    linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
    xyz=linear @ np.array([[.4124564,.3575761,.1804375],
                          [.2126729,.7151522,.0721750],
                          [.0193339,.1191920,.9503041]]).T
    xyz/=np.array([.95047,1.,1.08883])
    f=np.where(xyz>(6/29)**3,np.cbrt(xyz),xyz/(3*(6/29)**2)+4/29)
    lab=np.stack([116*f[:,1]-16,500*(f[:,0]-f[:,1]),200*(f[:,1]-f[:,2])],axis=1)
    keep=(lab[:,0]>32)&(lab[:,0]<76)
    rgb,lab=rgb[keep],lab[keep]
    selected=[int(np.argmin(((rgb-np.array([.1,.4,.8]))**2).sum(1)))]
    distance=np.full(len(rgb),np.inf)
    for _ in range(n-1):
        distance=np.minimum(distance,((lab-lab[selected[-1]])**2).sum(1))
        distance[selected]=-1
        selected.append(int(distance.argmax()))
    colors=plt.get_cmap("tab20")(np.arange(n)%20)
    panels=[]
    for name,title in [('z_skill','CAVE input · Z_skill'),('decoder_hidden','CAVE decoder · pre-head features')]:
        xy=np.asarray(data['projections'][name])
        fig,ax=plt.subplots(figsize=(15,max(8,n*.27+1.8)),facecolor='white')
        fig.subplots_adjust(left=.04,right=.64,bottom=.10,top=.92)
        for k,task in enumerate(tasks):
            points=xy[labels==k]
            ax.scatter(points[:,0],points[:,1],s=32,color=colors[k],marker='o',alpha=.9,edgecolors='white',linewidths=.3,label=f'{k+1:02d}  '+task.replace('_',' '))
        ax.axis('off');ax.set_title(title,loc='left',fontsize=20,fontweight='bold',pad=18)
        ax.legend(loc='center left',bbox_to_anchor=(1.02,.5),frameon=False,fontsize=9,labelspacing=.7,handletextpad=.5,markerscale=1.5)
        fig.text(.04,.035,f'Step {data["step"]:,} / {len(labels)} real samples / {n} tasks\nt-SNE · seed 42 · raw checkpoint · {data.get("pooling", "fixed slots flattened")} · no augmentation',fontsize=10,color='#667085')
        fig.savefig(out/(name+'.png'),dpi=200);fig.savefig(out/(name+'.svg'));plt.close(fig)
        panels.append(f'<section><h2>{html.escape(title)}</h2><img src="{name}.svg" style="width:100%"></section>')
    (out/'index.html').write_text('<meta charset="utf-8"><title>CAVE latent space</title><body style="background:#f5f6fa;color:#182230;font:16px sans-serif;margin:40px auto;max-width:1500px"><h1>CAVE · 潜在特征空间</h1><p>每点是一个真实训练样本。所有任务均为圆点，使用区分度更高的配色；两图的任务配色一致。FP32 原始 checkpoint 参数。</p>'+''.join(panels)+'<p>沿用原始投影坐标。两个投影独立拟合，不能直接比较坐标；t-SNE 不证明技能解耦或 rollout 成功。</p></body>')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True)
    render(p.parse_args().output)
