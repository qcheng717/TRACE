"""Fixed RoboTwin Stage-I feature prediction examples; no optimization."""
import argparse,copy,hashlib,html,json,os,sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import default_collate
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from trace_robotwin.models.model import TRACE
from trace_robotwin.models.checkpoint import validate
from visualize_latents import visualization_data
STYLE="""body{font:16px/1.7 system-ui;background:#f2f5fa;color:#182638;max-width:1280px;margin:32px auto;padding:0 20px}section,article,.note{background:white;border-radius:16px;padding:22px;margin:18px 0;box-shadow:0 4px 20px #18263808}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(245px,1fr));gap:18px}img{max-width:100%}a{color:#1765ce;text-decoration:none}.card img{width:100%}h1{font-size:32px}small{color:#65758b}button,select{padding:10px;border-radius:8px}"""
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--device',default='cpu');p.add_argument('--output',required=True);p.add_argument('--examples',help='Fixed RoboTwin episodes/anchors JSON');a=p.parse_args()
 os.chdir(ROOT);os.environ.setdefault('ROBOTWIN_ROOT',str(ROOT/'third_party/RoboTwin'));torch.set_num_threads(2);torch.manual_seed(42)
 ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False,mmap=True);validate(ck,stage='stage1')
 c=copy.deepcopy(ck['config']);c['data']['augment']=False
 offset=c['model']['v10'].get('future_target_offset')
 if not offset or c['model']['v10']['future_horizon']!=1:raise ValueError('Requires a single endpoint target')
 ds,named,lengths=visualization_data(c,'stage1');m=TRACE(c['model']).to(a.device).eval();m.load_state_dict(ck['model'],strict=True)
 real=c['data']['adapter']=='real_robot_v1'
 v4=c['data']['adapter'] in ('robotwin_motif20_v4','real_robot_v1')
 libero=c['data']['adapter'] in ('libero_long_v6','libero_long_v7')
 episode_data=v4 or libero
 robot=v4 or (not libero and ds.robot)
 entries=ds.entries if episode_data else (ds.base.entries if robot else ds.base.segments)
 cameras=c['model']['v10'].get('camera_views',2)
 grid=m.predictor.selector.g
 height,width=grid if isinstance(grid,(tuple,list)) else (grid,grid)
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True);cards=[];reports=[]
 windows=[]
 if a.examples:
  fixed=json.loads(Path(a.examples).read_text())
  if v4 and fixed.get('manifest_sha256')!=ds.stats['manifest_sha256']:raise ValueError('Fixed V4 examples require matching manifest SHA256')
  if v4 and fixed.get('future_target_offset')!=offset:raise ValueError('Fixed V4 examples future offset differs')
  if libero and (fixed.get('index_sha256')!=c['data']['index_sha256'] or fixed.get('future_target_offset')!=offset):raise ValueError('Fixed V6 examples require matching index and future offset')
  if robot and not v4:raise ValueError('Fixed episode examples currently support RoboTwin V4 only')
  for row in fixed['rows']:
   idx,anchor=row['index' if episode_data else 'segment'],row['anchor']
   if entries[idx]['task_name' if robot or libero else 'task']!=row['task']:raise ValueError('Fixed task/episode mismatch')
   if episode_data and entries[idx]['episode_id']!=row['episode_id']:raise ValueError('Fixed episode ID mismatch')
   if not 0<=anchor<lengths[idx]:raise ValueError('Fixed anchor outside episode')
   if not episode_data and anchor+offset>=lengths[idx]:raise ValueError('Fixed endpoint outside segment')
   windows.append((idx,anchor))
  selection='Previously fixed task/episode/anchor; no selection by prediction error' if robot or libero else 'Previously fixed CALVIN task/segment/anchor; no selection by prediction error'
 else:
  for _,pool in named:
   idx=next((int(i) for i in pool if lengths[int(i)]>offset),None)
   if idx is not None:windows.append((idx,(lengths[idx]-offset-1)//2))
  selection='First eligible episode per task; midpoint of valid anchor range'
 label=('LIBERO Long V7' if c['data']['adapter']=='libero_long_v7' else 'LIBERO Long V6') if libero else ('RoboTwin V4' if v4 else ('RoboTwin V3' if robot else 'CALVIN V10'))
 if real:label='Real Robot V1'
 digest=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()
 for idx,anchor in windows:
  s=ds.get_window(idx,anchor);b={k:v.to(a.device) for k,v in default_collate([s]).items() if torch.is_tensor(v)}
  with torch.inference_mode():r=m.stage1_outputs(b)
  current=r['S_tar_current'][0].float().cpu().numpy();target=r['target_future'][0,0].float().cpu().numpy()
  pred=r['pred_future'][0,0].float().cpu().numpy();delta=r['pred_transition'][0,0].float().cpu().numpy()
  true=target-current;error=pred-target
  task=s['metadata'].get('task_name',s['metadata'].get('task'));dest=out/task;dest.mkdir(exist_ok=True)
  np.savez_compressed(dest/'features.npz',current=current,target=target,prediction=pred,pred_transition=delta,z_skill=r['Z_skill'][0].cpu().numpy())
  rms=lambda x:float(np.sqrt(np.mean(x*x)))
  err=float(np.abs(error).mean());base=float(np.abs(true).mean())
  actual=int(s['future_actual_offset'][0])
  report=dict(task=task,step=ck['step'],checkpoint=a.checkpoint,weights='raw',index=idx,anchor=anchor,target_frame=anchor+actual,delta=offset,actual_offset=actual,future_clamped=bool(s['future_clamped'][0]),checkpoint_sha256=digest,segment=idx,metadata=s['metadata'],future_l1=err,copy_l1=base,gain=1-err/base if base>1e-8 else None,transition_l1=float(np.abs(delta-true).mean()),selection=selection,target_rms=rms(target),camera_views=cameras,grid_shape=[height,width])
  if v4:report.update(raw_anchor=int(s['raw_anchor']),raw_target_frame=int(s['raw_future_anchor']),manifest_sha256=ds.stats['manifest_sha256'])
  reports.append(report);(dest/'metrics.json').write_text(json.dumps(report,indent=2))
  rgbcards=[]
  views=[('head_current','Head · t',s['images'][0]),('head_future',f'Head · true t+{actual}',s['future_images'][0])]
  if cameras==2:views.extend([('wrist_current','Wrist · t',s['wrist'][0]),('wrist_future',f'Wrist · true t+{actual}',s['future_wrist'][0])])
  for name,image_label,x in views:
   Image.fromarray((x.permute(1,2,0).numpy().clip(0,1)*255).astype('uint8')).save(dest/(name+'.png'))
   rgbcards.append(f'<article><h3>{image_label}</h3><img src="{name}.png"></article>')
  fig,axes=plt.subplots(2,3,figsize=(15,6),layout='constrained')
  lim=max(float(np.abs(np.stack([current,pred,target])).max()),1e-8)
  mag=[np.abs(true),np.abs(delta),np.abs(error)];vmax=max(float(np.stack(mag).max()),1e-8)
  for ax,x,title in zip(axes[0],[current,pred,target],['Current target patches','Predicted future patches','True future patches']):
   im=ax.imshow(x,aspect='auto',cmap='coolwarm',vmin=-lim,vmax=lim);ax.set_title(title);ax.set_ylabel('Spatial patch');ax.set_xlabel('Feature dimension');fig.colorbar(im,ax=ax,shrink=.7)
  for ax,x,title in zip(axes[1],mag,['True transition magnitude','Predicted transition magnitude','Future prediction error']):
   im=ax.imshow(x,aspect='auto',cmap='magma',vmin=0,vmax=vmax);ax.set_title(title);ax.set_xlabel('Feature dimension');fig.colorbar(im,ax=ax,shrink=.7)
  fig.savefig(dest/'semantic_comparison.png',dpi=160);plt.close(fig)
  fig,axes=plt.subplots(cameras,3,figsize=(12,4*cameras),layout='constrained',squeeze=False)
  maps=[np.abs(x).mean(-1).reshape(cameras,height,width) for x in (true,delta,error)]
  vmax=max(float(x.max()) for x in maps)
  for camera in range(cameras):
   for j,title in enumerate(['True residual','Predicted residual','Prediction error']):
    im=axes[camera,j].imshow(maps[j][camera],cmap='magma',vmin=0,vmax=vmax)
    axes[camera,j].set_title(('Head' if camera==0 else 'Wrist')+' '+title)
    fig.colorbar(im,ax=axes[camera,j],shrink=.7)
  fig.savefig(dest/'slot_errors.png',dpi=160);plt.close(fig)
  gain='N/A' if report['gain'] is None else f"{report['gain']:.1%}"
  if real:rgbcards=[card.replace('Head ·','Realsense ·') for card in rgbcards]
  camera_options='<option value="head">头部相机</option>'+('<option value="wrist">腕部相机</option>' if cameras==2 else '')
  if real:camera_options=camera_options.replace('头部相机','Realsense 第三视角')
  timeline_note=f'处理后帧编号；原始帧 {report["raw_anchor"]} → {report["raw_target_frame"]}。' if v4 else ''
  (dest/'index.html').write_text(f'<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>{STYLE}</style><a href="../index.html">← 全部任务</a><h1>{html.escape(task)} · 预测 t+{offset}</h1><p>{html.escape(s["metadata"]["instruction"])}</p><div class="note">Stage-I {ck["step"]:,} / raw / episode {html.escape(str(s["metadata"]["episode_id"]))} / t={anchor} → {anchor+actual}；末帧截断={report["future_clamped"]}<br>{timeline_note}<br>预测L1 <b>{err:.5f}</b> · 复制L1 <b>{base:.5f}</b> · 相对收益 <b>{gain}</b></div><div class="grid">{"".join(rgbcards)}</div><section><h2>真实RGB对照</h2><p>以下滑块用于比较真实当前与真实未来图像，不是生成的预测RGB。</p><select id="camera" onchange="sync()">{camera_options}</select><input type="range" id="blend" min="0" max="100" value="50" oninput="sync()"><div style="position:relative;width:448px"><img id="now" src="head_current.png" style="width:100%;display:block"><img id="later" src="head_future.png" style="position:absolute;inset:0;width:100%;opacity:.5"></div></section><section><h2>Spatial patch预测对照</h2><p>{cameras*height*width}行对应{cameras}个相机的patch，{current.shape[-1]}列是特征维度；另展示每个相机{height}×{width}的通道平均绝对残差。上排共用对称色阶；下排共用幅度色阶。</p><img src="semantic_comparison.png"><img src="slot_errors.png"></section><p>单个训练集示例不代表整体泛化或rollout成功率；这里没有动作执行。目标来自同一checkpoint的EMA target selector，模型权重采用raw，不混用全模型EMA。</p><a href="metrics.json">指标和来源</a> · <a href="features.npz">原始特征</a><script>function sync(){{let c=document.getElementById("camera").value;document.getElementById("now").src=c+"_current.png";document.getElementById("later").src=c+"_future.png";document.getElementById("later").style.opacity=document.getElementById("blend").value/100;}}</script>')
  cards.append(f'<a class="card" href="{task}/index.html"><article><img src="{task}/head_future.png"><h3>{html.escape(task)}</h3><p>相对复制收益：{gain}</p><small>预测 {err:.4f} / 复制 {base:.4f}</small></article></a>')
  print(task,gain,flush=True)
 (out/'gallery_metrics.json').write_text(json.dumps(reports,indent=2))
 (out/'index.html').write_text(f'<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>{STYLE}</style><h1>{label} · {ck["step"]:,} · 未来特征预测</h1><div class="note">{len(reports)}任务固定示例 / 单目标t+{offset} / 原始checkpoint权重 / 关闭增强。{html.escape(selection)}。点击卡片查看当前／真实未来{cameras}个相机视角、预测patch特征和全维误差。图像均为真实RGB，预测器不生成RGB。本报告预测融合并归一化后的空间patch特征；不同特征空间与跨度的L1不可直接比较。正收益表示优于复制当前target特征。</div><div class="grid">{"".join(cards)}</div>')
 print(out/'index.html')
if __name__=='__main__':main()
