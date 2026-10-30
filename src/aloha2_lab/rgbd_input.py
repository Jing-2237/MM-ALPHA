"""Aligned RGB-D model inputs: RGB, metric optical depth and validity mask."""
import numpy as np

DEPTH_MIN=.05
DEPTH_MAX=2.0
DEPTH_SCALE=1.0 # metres; fixed physical scale, never fitted to held-out data


def metric_depth(depth,encoding):
    a=np.asarray(depth)
    if a.ndim!=2:raise ValueError('Depth must be HxW')
    if encoding=='16UC1':
        if a.dtype!=np.uint16:raise ValueError('16UC1 input requires uint16 millimetres')
        a=a.astype(np.float32)*.001
    elif encoding=='32FC1':a=a.astype(np.float32)
    else:raise ValueError('Depth encoding must be 16UC1 or 32FC1')
    valid=np.isfinite(a)&(a>=DEPTH_MIN)&(a<=DEPTH_MAX)
    return np.where(valid,a,0).astype(np.float32),valid


def model_input(rgb,depth_m,crop=None):
    rgb=np.asarray(rgb);depth=np.asarray(depth_m)
    if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or depth.shape!=rgb.shape[:2]:
        raise ValueError('RGB uint8 and aligned metric depth of matching dimensions required')
    if crop is not None:
        x0,y0,x1,y1=crop
        if not (0<=x0<x1<=rgb.shape[1] and 0<=y0<y1<=rgb.shape[0]):raise ValueError('Invalid crop')
        rgb=rgb[y0:y1,x0:x1];depth=depth[y0:y1,x0:x1]
    d,valid=metric_depth(depth,'32FC1')
    # Invalid depth is explicitly represented rather than confused with a zero-distance point.
    return np.concatenate([rgb.astype(np.float32)/255.,(d/DEPTH_SCALE)[...,None],valid.astype(np.float32)[...,None]],axis=2).transpose(2,0,1)


def make_rgbd_encoder():
    import torch
    from torch import nn
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv=nn.Sequential(nn.Conv2d(5,24,3,padding=1),nn.ReLU(),nn.Conv2d(24,24,3,padding=1),nn.ReLU(),nn.Conv2d(24,1,1))
        def forward(self,x):
            if x.ndim!=4 or x.shape[1]!=5:raise ValueError('Expected Bx5xHxW RGB-D+validity tensor')
            heat=self.conv(x).flatten(1)
            # Surface attention uses valid depth; empty-depth batches remain finite.
            valid=x[:,4].flatten(1)>0
            logits=heat.masked_fill(~valid,-1e4)
            weights=logits.softmax(1)*valid
            weights=weights/weights.sum(1,keepdim=True).clamp_min(1e-8)
            yy,xx=torch.meshgrid(torch.linspace(-1,1,x.shape[2],device=x.device),torch.linspace(-1,1,x.shape[3],device=x.device),indexing='ij')
            latent=torch.stack([(weights*xx.flatten()).sum(1),(weights*yy.flatten()).sum(1),(weights*x[:,3].flatten(1)).sum(1)],dim=1)
            return latent,heat
    return Encoder()
