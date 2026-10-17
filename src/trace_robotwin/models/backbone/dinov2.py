"""Pretrained DINOv2 patch-token extraction, shared over all sequence frames."""
from contextlib import nullcontext
from pathlib import Path
import json
import torch
from torch import Tensor, nn


class DINOv2Backbone(nn.Module):
    """[N,3,H,W] in [0,1] -> ordered raw block tokens [N,patches,C].

    candidate_layers are zero-based transformer block indices. CLS is excluded.
    Raw post-block tokens are used consistently for semantic and temporal paths;
    do not use the final LayerNorm for just the last candidate. No time mixing.
    """
    def __init__(self, pretrained_path: str, candidate_layers: list[int],
                 mode: str = 'frozen', unfreeze_last: int = 2,
                 frame_batch_size: int = 8, image_size: int | list[int] = 224,
                 pad_to_patch: bool = False):
        super().__init__()
        from transformers import Dinov2Model
        root=Path(pretrained_path)
        if not (root/'model.safetensors').is_file():
            raise FileNotFoundError(f'DINOv2 local pretrained weights missing: {root}')
        self.encoder=Dinov2Model.from_pretrained(str(root),local_files_only=True)
        if not candidate_layers or len(set(candidate_layers))!=len(candidate_layers):
            raise ValueError('DINO candidate layers must be nonempty and unique')
        if any(i<0 or i>=self.encoder.config.num_hidden_layers for i in candidate_layers):
            raise ValueError('DINO candidate layer outside transformer blocks')
        shape=(image_size,image_size) if isinstance(image_size,int) else tuple(image_size)
        if len(shape)!=2 or min(shape)<1:raise ValueError('image_size must be H,W or a positive square size')
        if frame_batch_size<1 or (not pad_to_patch and any(s % self.encoder.config.patch_size for s in shape)):
            raise ValueError('frame_batch_size must be positive and image_size divisible by patch size')
        self.candidate_layers=candidate_layers
        self.feature_dims={str(i):self.encoder.config.hidden_size for i in candidate_layers}
        self.frame_batch_size,self.image_size=frame_batch_size,image_size
        self.image_shape=shape;self.pad_to_patch=pad_to_patch
        processor=json.loads((root/'preprocessor_config.json').read_text())
        self.register_buffer('mean',torch.tensor(processor['image_mean']).view(1,3,1,1))
        self.register_buffer('std',torch.tensor(processor['image_std']).view(1,3,1,1))
        self.set_mode(mode,unfreeze_last)

    def set_mode(self, mode: str, unfreeze_last: int = 2) -> None:
        if mode not in ('frozen','partial','trainable'): raise ValueError('invalid DINO freeze mode')
        if mode=='partial' and not 1<=unfreeze_last<=len(self.encoder.encoder.layer):
            raise ValueError('invalid DINO unfreeze_last')
        self.mode=mode; self.encoder.requires_grad_(mode=='trainable')
        if mode=='partial':
            for layer in self.encoder.encoder.layer[-unfreeze_last:]: layer.requires_grad_(True)
        # Deterministic feature extraction also during fine-tuning: residuals
        # should not include dropout/stochastic-depth noise between frames.
        self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, images: Tensor) -> dict[str,Tensor]:
        if images.ndim!=4 or images.shape[1:]!=(3,*self.image_shape):
            raise ValueError(f'DINO expects [N,3,{self.image_shape[0]},{self.image_shape[1]}]')
        if not images.is_floating_point() or images.min() < -1e-6 or images.max() > 1+1e-6:
            raise ValueError('DINO input must be floating RGB in [0,1] before normalization')
        # Bilinear resizing can overshoot endpoints by float32 roundoff.
        images=images.clamp(0,1)
        collected={k:[] for k in self.feature_dims}
        for frames in images.split(self.frame_batch_size):
            with torch.no_grad() if self.mode=='frozen' else nullcontext():
                pixels=(frames-self.mean)/self.std
                if self.pad_to_patch:
                    patch=self.encoder.config.patch_size
                    pixels=torch.nn.functional.pad(pixels,(0,(-pixels.shape[-1])%patch,0,(-pixels.shape[-2])%patch))
                out=self.encoder(pixel_values=pixels,output_hidden_states=True,return_dict=True)
                # hidden_states[0] is patch embedding; block i is index i+1.
                for i in self.candidate_layers: collected[str(i)].append(out.hidden_states[i+1][:,1:])
        return {k:torch.cat(v,0) for k,v in collected.items()}
