"""Spatial skill dictionary and task-conditioned visual abstraction."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from trace_robotwin.models.skill_ops import SkillEncoder
from trace_robotwin.models.routing import sparse_route


def partition(x, w):
    b,h,s,d=x.shape
    return x.reshape(b,h//w,w,s//w,w,d).permute(0,1,3,2,4,5).reshape(-1,w*w,d)


def unpartition(x, w, g, b):
    h,s=(g,g) if isinstance(g,int) else g
    return x.reshape(b,h//w,s//w,w,w,-1).permute(0,1,3,2,4,5).reshape(b,h,s,-1)


def grid_shape(c):
    shape=tuple(c.get('grid_shape',(c['grid_size'],c['grid_size'])))
    if len(shape)!=2 or min(shape)<1:raise ValueError('Invalid spatial grid')
    return shape


class GridEmbedding(nn.Module):
    def __init__(self,g,d,cameras=2):
        super().__init__()
        h,w=(g,g) if isinstance(g,int) else g
        self.camera=nn.Parameter(torch.randn(1,cameras,1,1,d)*.02)
        self.row=nn.Parameter(torch.randn(1,1,h,1,d)*.02)
        self.col=nn.Parameter(torch.randn(1,1,1,w,d)*.02)
    def forward(self,x):
        return x+self.camera+self.row+self.col


class SpatialBlock(nn.Module):
    def __init__(self,d,heads,g,window=4,shift=0,language=False,ffn=1024):
        super().__init__();self.g=g;self.w=window;self.shift=shift;self.heads=heads
        h,w=(g,g) if isinstance(g,int) else g
        if min(h,w,window)<1 or not 0<=shift<window:raise ValueError('Invalid window/grid/shift')
        self.shape=(h,w);self.padded_shape=(h+(-h)%window,w+(-w)%window)
        ph,pw=self.padded_shape
        self.norm1=nn.LayerNorm(d);self.attention=nn.MultiheadAttention(d,heads,batch_first=True)
        self.norm2=nn.LayerNorm(d)
        self.ff=nn.Sequential(nn.Linear(d,ffn),nn.GELU(),nn.Linear(ffn,d))
        self.language_film=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,2*d)) if language else None
        if language:
            # Initially identity modulation of the pre-FFN LayerNorm.
            nn.init.zeros_(self.language_film[-1].weight)
            nn.init.zeros_(self.language_film[-1].bias)
        mask=torch.zeros((ph//window)*(pw//window),window**2,window**2,dtype=torch.bool)
        if shift:
            labels=torch.zeros(1,ph,pw,1)
            slices=(slice(0,-window),slice(-window,-shift),slice(-shift,None))
            for i,a in enumerate(slices):
                for j,b in enumerate(slices):labels[:,a,b]=i*3+j
            regions=partition(labels,window).squeeze(-1)
            mask=regions[:,:,None]!=regions[:,None,:]
        if (h,w)!=(ph,pw):
            valid=torch.zeros(1,ph,pw,1,dtype=torch.bool);valid[:,:h,:w]=True
            if shift:valid=torch.roll(valid,(-shift,-shift),(1,2))
            valid=partition(valid,window).squeeze(-1)
            mask=mask|~valid[:,None,:]
            # Cropped padding queries must still have a finite attention row.
            rows,queries=mask.all(-1).nonzero(as_tuple=True)
            mask[rows,queries,queries]=False
        self.needs_mask=bool(mask.any())
        self.register_buffer('window_mask',mask,persistent=False)

    def forward(self,x,language=None):
        b,c,h,w,d=x.shape
        if (h,w)!=self.shape:raise ValueError('Spatial input grid differs')
        ph,pw=self.padded_shape
        y=self.norm1(x).reshape(b*c,h,w,d)
        y=F.pad(y,(0,0,0,pw-w,0,ph-h))
        if self.shift:y=torch.roll(y,(-self.shift,-self.shift),(1,2))
        windows=partition(y,self.w)
        mask=self.window_mask.repeat(b*c,1,1).repeat_interleave(self.heads,0) if self.needs_mask else None
        windows=self.attention(windows,windows,windows,attn_mask=mask,need_weights=False)[0]
        y=unpartition(windows,self.w,self.padded_shape,b*c)
        if self.shift:y=torch.roll(y,(self.shift,self.shift),(1,2))
        x=x+y[:,:h,:w].reshape_as(x)
        conditioned=self.norm2(x)
        if self.language_film is not None:
            if language is None or language.shape!=(b,1,d):raise ValueError('FiLM requires B,1,D language')
            gamma,beta=self.language_film(language[:,0]).chunk(2,-1)
            conditioned=(1+gamma[:,None,None,None,:])*conditioned+beta[:,None,None,None,:]
        return x+self.ff(conditioned)


class SpatialEncoder(nn.Module):
    def __init__(self,dims,language_dim,c):
        super().__init__();self.keys=list(dims);self.g=grid_shape(c);self.cameras=c.get('camera_views',2);d=c['semantic_dim'];heads=c['selector_heads']
        self.project=nn.ModuleDict({k:nn.Sequential(nn.LayerNorm(v),nn.Linear(v,d)) for k,v in dims.items()})
        self.language=nn.Linear(language_dim,d)
        self.read=nn.ModuleList([nn.MultiheadAttention(d,heads,batch_first=True) for _ in dims])
        self.score=nn.ModuleList([nn.Linear(d,1) for _ in dims])
        self.position=GridEmbedding(self.g,d,self.cameras)
        self.contextualizer=nn.ModuleList([SpatialBlock(d,heads,self.g,4,s,True,c['future_decoder_ffn_dim']) for s in (0,2)])

    def forward(self,features,language,return_grid=False):
        l=self.language(language)[:,None];xs=[];scores=[]
        for i,k in enumerate(self.keys):
            raw=features[k]
            if raw.shape[1:3]!=(self.cameras,self.g[0]*self.g[1]):raise ValueError('Unexpected camera count or patch grid')
            x=self.project[k](raw).flatten(1,2);xs.append(x)
            u=self.read[i](l,x,x,need_weights=False)[0]
            scores.append(self.score[i](u[:,0]))
        alpha=torch.cat(scores,-1).softmax(-1)
        self.last_layer_weights=alpha.detach()
        s=(torch.stack(xs,1)*alpha[:,:,None,None]).sum(1).reshape(len(l),self.cameras,*self.g,-1)
        h=self.position(s)
        for block in self.contextualizer:h=block(h,l)
        self.last_raw_h=h.detach().flatten(1,3)
        h=F.layer_norm(h, (h.shape[-1],), weight=None, bias=None).flatten(1,3)
        return (h,s) if return_grid else h


class SpatialFutureDecoder(nn.Module):
    """Only the spatial latent field crosses this API. No language/visual skip."""
    def __init__(self,c):
        super().__init__();self.g=grid_shape(c);self.cameras=c.get('camera_views',2);d=c['semantic_dim']
        self.projection=nn.Linear(c['dictionary_dim'],d);self.position=GridEmbedding(self.g,d,self.cameras)
        self.blocks=nn.ModuleList([SpatialBlock(d,c['future_decoder_heads'],self.g,4,s,False,c['future_decoder_ffn_dim']) for s in (0,2)])
        self.norm=nn.LayerNorm(d);self.transition_head=nn.Linear(d,d)
        nn.init.zeros_(self.transition_head.weight);nn.init.zeros_(self.transition_head.bias)
    def forward(self,z_skill):
        if z_skill.ndim!=3 or z_skill.shape[1]!=self.cameras*self.g[0]*self.g[1]:raise ValueError('Unexpected spatial latent field')
        x=self.position(self.projection(z_skill).reshape(len(z_skill),self.cameras,*self.g,-1))
        for block in self.blocks:x=block(x)
        return self.transition_head(self.norm(x)).flatten(1,3)


class SpatialSkillPredictor(nn.Module):
    def __init__(self,dims,language_dim,c):
        super().__init__();self.c=c
        self.selector=SpatialEncoder(dims,language_dim,c)
        self.target_selector=copy.deepcopy(self.selector).requires_grad_(False).eval()
        self.dictionary=nn.Parameter(torch.randn(c['dictionary_size'],c['dictionary_dim']))
        self.register_buffer('D_init',self.dictionary.detach().clone())
        self.encoder=SkillEncoder(c);self.decoder=SpatialFutureDecoder(c)
        self.register_buffer('key_weight_init',self.encoder.key.weight.detach().clone())
        self.register_buffer('decoder_weight_init',self.decoder.projection.weight.detach().clone())
    def train(self,mode=True):
        super().train(mode);self.target_selector.eval();return self
    def normalized_dictionary(self):return F.normalize(self.dictionary.float(),dim=-1)
    def forward(self,features,language):
        h,s=self.selector(features,language,return_grid=True);d=self.normalized_dictionary()
        a=self.encoder(h,d);z=a.to(d.dtype)@d;delta=self.decoder(z)
        # Singleton endpoint axis retained solely for the existing data/loss interface.
        return dict(S_grid=s,S_on=h,H=h,H_raw=self.selector.last_raw_h,D=self.dictionary,D_normalized=d,A=a,Z_skill=z,
                    delta_pred=delta,pred_transition=delta[:,None])
    @torch.no_grad()
    def update_target(self):
        for t,s in zip(self.target_selector.parameters(),self.selector.parameters()):t.lerp_(s,1-self.c['ema_target_decay'])
        for t,s in zip(self.target_selector.buffers(),self.selector.buffers()):t.copy_(s)


class SpatialEvidenceRouter(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c
        self.query=nn.Linear(c['dictionary_dim'],c['router_dim'],bias=False)
        self.evidence=nn.Linear(c['semantic_dim'],c['router_dim'],bias=False)
    def forward(self,dictionary,transition,coefficients,step):
        with torch.autocast(device_type=transition.device.type,enabled=False):
            delta=transition[:,0] if transition.ndim==4 else transition
            a=coefficients.detach().float();d=dictionary.detach().float()
            rp=F.linear(delta.float(),self.evidence.weight.float())
            rb=torch.einsum('bpk,bpd->bkd',a,rp)/(a.sum(1)[...,None]+1e-8)
            q=F.linear(d,self.query.weight.float())
            if q.ndim==2:q=q[None].expand(len(a),-1,-1)
            heads=self.c['router_heads'];eps=self.c['router_norm_eps']
            q=q.reshape(len(a),-1,heads,q.shape[-1]//heads)
            r=rb.reshape_as(q)
            def unit(x):
                norm=x.norm(dim=-1,keepdim=True)
                return torch.where(norm>eps,x/norm.clamp_min(eps),torch.zeros_like(x))
            cosine=(unit(q)*unit(r)).sum(-1)
            logits=cosine.mean(-1)/self.c['router_temperature']
            pi,soft,eta=sparse_route(logits,step,self.c['entmax_start_step'],self.c['entmax_full_step'])
            return dict(pi=pi,soft=soft,eta=eta,logits=logits,cosine=cosine,R_basis=rb)
