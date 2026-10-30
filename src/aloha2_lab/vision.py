"""RGB-D localization for an explicitly registered blue cup; no simulator object IDs."""
from dataclasses import dataclass
import numpy as np

@dataclass
class RGBDFrame:
    rgb: np.ndarray
    depth: np.ndarray  # optical Z in metres
    K: np.ndarray
    T_world_optical: np.ndarray
    timestamp: float


def locate_cup(frame, now, max_age=.2):
    if not 0 <= now-frame.timestamp <= max_age:
        raise ValueError('RGB-D frame is stale or from the future')
    rgb=np.asarray(frame.rgb,dtype=float)
    depth=np.asarray(frame.depth)
    if rgb.ndim!=3 or depth.ndim!=2 or rgb.shape[:2]!=depth.shape or rgb.shape[-1]!=3:
        raise ValueError('Aligned RGB/depth dimensions required')
    # Registered object appearance, NOT a general semantic cup classifier.
    mask=(rgb[:,:,2]>70)&(rgb[:,:,2]>rgb[:,:,0]*1.7)&(rgb[:,:,2]>rgb[:,:,1]*2.2)
    mask &= np.isfinite(depth)&(depth>.05)&(depth<2.)
    ys,xs=np.where(mask)
    if len(xs)<12: raise ValueError('Registered blue cup not visible with valid depth')
    if np.ptp(xs)>100 or np.ptp(ys)>100: raise ValueError('Ambiguous/oversized blue region')
    K=np.asarray(frame.K);T=np.asarray(frame.T_world_optical)
    if K.shape!=(3,3) or T.shape!=(4,4) or not np.isfinite(K).all() or not np.isfinite(T).all():
        raise ValueError('Invalid camera calibration')
    if K[0,0]<=0 or K[1,1]<=0 or not np.allclose(T[3],[0,0,0,1]) or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(T[:3,:3]),1):raise ValueError('Invalid rigid transform or focal length')
    z=depth[ys,xs]
    optical=np.stack([(xs-K[0,2])*z/K[0,0],(ys-K[1,2])*z/K[1,1],z],axis=1)
    world=optical@T[:3,:3].T+T[:3,3]
    # Robust centre of observed surface; known registered cup sits on z=0 table.
    wall=world[(world[:,2]>.01)&(world[:,2]<.07),:2]
    if len(wall)<12:raise ValueError('Insufficient cup wall depth')
    design=np.c_[2*wall,np.ones(len(wall))]
    fit,_,rank,_=np.linalg.lstsq(design,(wall*wall).sum(1),rcond=None)
    if rank!=3 or np.linalg.cond(design)>1e5:raise ValueError('Cup depth geometry is degenerate')
    xy=fit[:2]
    radius=np.sqrt(max(0,fit[2]+xy@xy))
    residual=np.median(np.abs(np.linalg.norm(wall-xy,axis=1)-radius))
    if not .015<radius<.035 or residual>.006:raise ValueError('Depth shape does not match registered cup')
    if not (-.3<xy[0]<.1 and -.2<xy[1]<.25):raise ValueError('Cup outside left-arm fixture workspace')
    return {'position_world':[float(xy[0]),float(xy[1]),.045],
            'bbox_xyxy':[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())],
            'pixels':len(xs),'timestamp':frame.timestamp,'method':'registered_blue_cup_rgbd',
            'surface_z':float(np.median(world[:,2]))}


def locate_paper_cup(frame,now,profile):
    """Fit registered tapered cup to observed depth above a calibrated table.

    No colour mask or simulator segmentation/object pose is used.
    """
    if not 0<=now-frame.timestamp<=.2:raise ValueError('Stale RGB-D frame')
    depth=np.asarray(frame.depth);K=np.asarray(frame.K);T=np.asarray(frame.T_world_optical)
    ys,xs=np.where(np.isfinite(depth)&(depth>.05)&(depth<2.))
    z=depth[ys,xs]
    optical=np.stack([(xs-K[0,2])*z/K[0,0],(ys-K[1,2])*z/K[1,1],z],1)
    world=optical@T[:3,:3].T+T[:3,3]
    rb=profile['bottom_radius_m'];rt=profile['top_radius_m'];h=profile['height_m']
    keep=(world[:,2]>.02)&(world[:,2]<h-.014)&(world[:,0]>-.16)&(world[:,0]<.10)&(world[:,1]>-.09)&(world[:,1]<.22)
    points=world[keep]
    if len(points)<30:raise ValueError('Insufficient paper-cup depth points')
    radii=rb+(rt-rb)*points[:,2]/h
    xy=np.median(points[:,:2],axis=0)
    for _ in range(40):
        delta=points[:,:2]-xy;distance=np.linalg.norm(delta,axis=1).clip(1e-5)
        error=distance-radii;J=-delta/distance[:,None]
        weight=1/np.maximum(1,np.abs(error)/.004)
        step=np.linalg.solve(J.T@(J*weight[:,None])+np.eye(2)*1e-6,-J.T@(error*weight))
        xy+=np.clip(step,-.02,.02)
        if np.linalg.norm(step)<1e-7:break
    residual=float(np.median(np.abs(np.linalg.norm(points[:,:2]-xy,axis=1)-radii)))
    if residual>.008:raise ValueError('Depth does not fit registered paper-cup taper')
    if not(-.09<xy[0]<.05 and -.02<xy[1]<.15):raise ValueError('Paper cup outside side-grasp fixture workspace')
    return dict(position_world=[float(xy[0]),float(xy[1]),.04],method='registered_frustum_rgbd',depth_fit_residual_m=residual,points=len(points),timestamp=frame.timestamp)
