"""RGB / aligned RGB-D conditioned open-loop trajectory imitation learning.

Learned inference has no detector, IK or object pose input. New paper-cup
execution uses a separate collision rejection monitor and versioned contact model.
"""
import argparse,json
from pathlib import Path
import h5py
import numpy as np
import mujoco
from PIL import Image,ImageDraw
from .sim import AlohaSim
from .grasp import ik
from .vision import locate_cup,locate_paper_cup
from .dataset import EpisodeWriter

ROOT=Path(__file__).resolve().parents[2]
RGBD_CROP=(96,80,208,192)
CROP=(100,96,164,176) # fixed table ROI; identical for all positions/splits


def measure(sim,hold):
    # Evaluator only: privileged simulator state never reaches the learned policy.
    b=mujoco.mj_name2id(sim.model,mujoco.mjtObj.mjOBJ_BODY,'cup')
    touching=set()
    for c in sim.data.contact:
        pair=[int(c.geom1),int(c.geom2)]
        if any(sim.model.geom_bodyid[g]==b for g in pair):
            for g in pair:
                name=mujoco.mj_id2name(sim.model,mujoco.mjtObj.mjOBJ_BODY,int(sim.model.geom_bodyid[g]))
                if name in ('left/left_finger_link','left/right_finger_link'):touching.add(name)
    height=float(sim.data.body('cup').xpos[2])
    return dict(success=height>.12 and min(hold)>.12 and len(touching)==2,
                final_height_m=height,min_hold_height_m=float(min(hold)),finger_contacts=sorted(touching))


def plan(root,sensor="rgb",profile=None,seed=1927):
    if profile is not None and sensor!="rgbd":raise ValueError("Paper-cup training requires --sensor rgbd")
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    train=[[float(x),float(y)] for x in (np.linspace(-.06,.02,5) if profile else np.linspace(-.21,-.14,5)) for y in np.linspace(.02,.105,5)]
    rng=np.random.default_rng(seed)
    def sample(n,others,separation):
        points=[]
        while len(points)<n:
            p=rng.uniform([-.055,.025],[.015,.100]) if profile else rng.uniform([-.205,.025],[-.145,.100])
            if all(np.linalg.norm(p-q)>separation for q in others+points):points.append(p.tolist())
        return points
    val=sample(6,train,.006)
    test=sample(10,train+val,.007)
    manifest=dict(cup_profile=profile,sensor=sensor,seed=seed,object='coffee_mug_with_physical_handle',train=train,validation=val,test=test,
                  input='initial overhead RGB-D' if sensor=='rgbd' else 'initial overhead RGB',policy='learned visual encoder plus learned trajectory decoder',
                  note='fixed-camera stationary single-mug interpolation benchmark; no generalization claim outside domain')
    if profile is not None:
        if not profile.get('geometry_confirmed'):raise ValueError('Cup dimensions must be confirmed')
        manifest['object']='mcdonalds_paper_cup';manifest['input']='initial overhead RGB-D'
    (root/'split.json').write_text(json.dumps(manifest,indent=2));return manifest



def make_sim(root,xy):
    manifest=json.loads((Path(root)/'split.json').read_text())
    return AlohaSim(cup_xy=xy,cup_profile=manifest.get('cup_profile'))


def collect(root):
    root=Path(root);manifest=json.loads((root/'split.json').read_text());results=[]
    for split in ('train','validation'):
        for index,xy in enumerate(manifest[split]):
            out=root/split/f'episode_{index:06d}.hdf5';out.parent.mkdir(parents=True,exist_ok=True)
            if out.exists():
                with h5py.File(out) as f:
                    results.append(json.loads(f.attrs['evaluation']))
                continue
            sim=make_sim(root,xy);writer=None;complete=False
            row=dict(split=split,index=index,spawn_xy=xy)
            try:
                frame=sim.rgbd()
                paper=manifest.get('cup_profile') is not None
                found=locate_paper_cup(frame,sim.data.time,manifest['cup_profile']) if paper else locate_cup(frame,sim.data.time)
                centre=np.array(found['position_world'])
                R=np.eye(3) if paper else np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
                if paper:centre[0]+=.014
                writer=EpisodeWriter(out,'vision_ik_teacher_paper_cup_side' if paper else 'vision_ik_teacher_coffee_mug',dt=sim.dt)
                writer.file.create_dataset('initial_rgb',data=frame.rgb,compression='lzf')
                writer.file.create_dataset('initial_depth_m',data=frame.depth.astype(np.float32),compression='lzf')
                writer.file.create_dataset('initial_depth_valid',data=(np.isfinite(frame.depth)&(frame.depth>=.05)&(frame.depth<=2.)).astype(np.uint8),compression='lzf')
                writer.file.create_dataset('camera_K',data=frame.K)
                writer.file.create_dataset('T_world_optical',data=frame.T_world_optical)
                writer.file.attrs['image_timestamp']=frame.timestamp
                writer.file.attrs['depth_units']='metres_optical_z'
                writer.file.attrs['depth_aligned_to']='initial_rgb'
                writer.file.attrs['scene_cup_xy']=xy
                writer.file.attrs['visual_conditioning']='initial_rgb + initial_depth_m; no per-step image claim'
                # Auxiliary image-space teacher label from RGB-D localization, not spawn metadata.
                opt=np.linalg.inv(frame.T_world_optical)@np.r_[centre,1.]
                uv=(frame.K@opt[:3]);uv=uv[:2]/uv[2]
                writer.file.create_dataset('teacher_pixel',data=uv)
                if paper and manifest['cup_profile'].get('contact_revision',0)>=3:
                    from .side_grasp_teacher import execute
                    def record(a):
                        q,v=sim.observe();writer.append(q,v,a,sim.data.time)
                    result=execute(sim,found['position_world'],manifest['cup_profile'],record)
                    hold=result.pop('hold');row.update(result)
                else:
                    def step(a):
                        q,v=sim.observe();writer.append(q,v,a,sim.data.time);sim.step(a)
                    def move(target,gripper,n):
                        q=ik(sim,target,R);start=sim.data.ctrl.copy();end=start.copy();end[:6]=q;end[6]=gripper
                        for i in range(n):
                            u=(i+1)/n;u=u*u*(3-2*u);step(start+(end-start)*u)
                    move(centre+[-.08,0,.05] if paper else centre+[0,0,.16],.037,180)
                    move(centre,.037,180)
                    for _ in range(120):
                        a=sim.data.ctrl.copy();a[6]=max(.020 if paper else .016,a[6]-.0003);step(a)
                    move(centre+[0,0,.15],.020 if paper else .016,240)
                    hold=[]
                    for _ in range(100):
                        step(sim.data.ctrl.copy());hold.append(float(sim.data.body('cup').xpos[2]))
                row.update(measure(sim,hold))
                row['success']=row['success'] and row.get('collision_pass',True) and row.get('stable_hold',True) and row.get('side_wall_pinch',True)
                writer.file.attrs['task_success']=row['success']
                writer.file.attrs['evaluation']=json.dumps(row)
                complete=True
                Image.fromarray(frame.rgb).save(out.with_suffix('.png'))
            except Exception as exc:row.update(success=False,error=f'{type(exc).__name__}: {exc}')
            finally:
                if writer:writer.close(complete=complete)
                sim.close()
            results.append(row)
            (root/'collection.json').write_text(json.dumps(results,indent=2))
            print(json.dumps(row),flush=True)
    return results


def image_tensor(rgb):
    x0,y0,x1,y1=CROP
    return np.asarray(rgb[y0:y1,x0:x1],dtype=np.float32).transpose(2,0,1)/255.


def make_encoder():
    import torch
    from torch import nn
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.ReLU(),nn.Conv2d(16,16,3,padding=1),nn.ReLU(),nn.Conv2d(16,1,1))
        def forward(self,x):
            heat=self.conv(x).flatten(1)
            prob=heat.softmax(1)
            yy,xx=torch.meshgrid(torch.linspace(-1,1,x.shape[2],device=x.device),torch.linspace(-1,1,x.shape[3],device=x.device),indexing='ij')
            uv=torch.stack([(prob*xx.flatten()).sum(1),(prob*yy.flatten()).sum(1)],dim=1)
            return uv,heat
    return Encoder()


def features(uv):
    if uv.shape[1]==3:
        x,y,z=uv.T
        return np.stack([x**i*y**j*z**k for i in range(4) for j in range(4-i) for k in range(4-i-j)],axis=1)
    x,y=uv.T
    return np.stack([x**i*y**j for i in range(4) for j in range(4-i)],axis=1)


def train(root,epochs=1200):
    import torch
    root=Path(root);out=root/'model'
    sensor=json.loads((root/'split.json').read_text()).get('sensor','rgb')
    profile=json.loads((root/'split.json').read_text()).get('cup_profile') or {}
    if profile.get('teacher_revision',0)>=4 and not (root/'saved-action-replay.json').exists():
        raise ValueError('Run scripts/verify_side_dataset.py before training contact-aware revision4')
    rgbd=sensor=='rgbd'
    crop=RGBD_CROP if rgbd else CROP
    from .rgbd_input import model_input,make_rgbd_encoder
    torch.manual_seed(17);torch.set_num_threads(2)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    def load(split):
        paths=[];images=[];uv=[];actions=[]
        for p in sorted((root/split).glob('episode_*.hdf5')):
            with h5py.File(p) as f:
                if not f.attrs.get('task_success',False):continue
                if profile.get('teacher_revision',0)>=4 and 'replay_pass' not in f.attrs:
                    raise ValueError('Unverified saved-action episode in revision4 training data')
                if not f.attrs.get('replay_pass',True):continue
                paths.append(p.relative_to(root).as_posix())
                if rgbd:
                    if 'initial_depth_m' not in f:raise ValueError('RGB-D training requires recorded aligned metric depth; no RGB-only fallback')
                    images.append(model_input(f['initial_rgb'][:],f['initial_depth_m'][:],crop))
                else:images.append(image_tensor(f['initial_rgb'][:]))
                px=f['teacher_pixel'][:]
                uv.append([(px[0]-crop[0])/(crop[2]-crop[0]-1)*2-1,(px[1]-crop[1])/(crop[3]-crop[1]-1)*2-1])
                if rgbd:
                    depth=f['initial_depth_m'][:]
                    u=int(np.clip(round(px[0]),0,depth.shape[1]-1));v=int(np.clip(round(px[1]),0,depth.shape[0]-1))
                    z=float(depth[v,u])
                    if not np.isfinite(z) or not .05<=z<=2:raise ValueError('Invalid auxiliary depth target')
                    uv[-1].append(z)
                actions.append(f['action'][:])
        if len(paths)<3:raise ValueError('Need at least3 successful episodes in each training/validation split')
        return paths,np.stack(images),np.stack(uv),np.stack(actions)
    paths,x,y,actions=load('train');vpaths,vx,vy,va=load('validation')
    out.mkdir(exist_ok=False)
    tx=torch.tensor(x,device=device);ty=torch.tensor(y,dtype=torch.float32,device=device)
    vtx=torch.tensor(vx,device=device);vty=torch.tensor(vy,dtype=torch.float32,device=device)
    model=(make_rgbd_encoder() if rgbd else make_encoder()).to(device);opt=torch.optim.Adam(model.parameters(),lr=.003)
    # Soft pixel targets teach the visual encoder to attend to the mug.
    yy,xx=torch.meshgrid(torch.linspace(-1,1,tx.shape[2],device=device),torch.linspace(-1,1,tx.shape[3],device=device),indexing='ij')
    grid=torch.stack([xx.flatten(),yy.flatten()],1)
    target=torch.softmax(-((grid[None]-ty[:,None,:2])**2).sum(2)/(2*.035**2),dim=1)
    history=[];best=float('inf')
    for epoch in range(epochs):
        model.train();opt.zero_grad()
        aug=tx.clone()
        aug[:,:3]=(tx[:,:3]*torch.empty((len(tx),1,1,1),device=device).uniform_(.85,1.15)).clamp(0,1)
        uv,heat=model(aug)
        loss=-(target*heat.log_softmax(1)).sum(1).mean()+((uv-ty)**2).mean()*20
        loss.backward();opt.step()
        if epoch%10==0 or epoch==epochs-1:
            model.eval()
            with torch.no_grad():pred,_=model(vtx);value=((pred-vty)**2).mean().item()
            history.append(dict(epoch=epoch,loss=loss.item(),validation_uv_mse=value))
            if value<best:
                best=value;torch.save(model.state_dict(),out/'encoder.pt')
            if epoch%200==0:print(json.dumps(history[-1]),flush=True)
    model.load_state_dict(torch.load(out/'encoder.pt',weights_only=True,map_location=device));model.eval()
    with torch.no_grad():
        z=model(tx)[0].cpu().numpy();vz=model(vtx)[0].cpu().numpy()
    # Whole-action-trajectory BC: the decoder and its basis are fitted ONLY to successful train actions.
    flat=actions.reshape(len(actions),-1).astype(float);mean=flat.mean(0)
    _,singular,basis=np.linalg.svd(flat-mean,full_matrices=False)
    basis=basis[:min(10,len(paths)-1)]
    scores=(flat-mean)@basis.T
    latent_mean=z.mean(0) if rgbd else np.zeros(2)
    latent_std=z.std(0).clip(.001) if rgbd else np.ones(2)
    F=features((z-latent_mean)/latent_std);ridge=1e-3 if rgbd else 1e-7
    candidates=[]
    for candidate in ([1e-5,1e-4,1e-3,1e-2,.1,1.] if rgbd else [ridge]):
        fitted=np.linalg.solve(F.T@F+np.eye(F.shape[1])*candidate,F.T@scores)
        prediction=features((vz-latent_mean)/latent_std)@fitted@basis+mean
        error=float(np.mean((prediction-va.reshape(len(va),-1))**2))
        candidates.append((error,candidate,fitted))
    action_mse,ridge,coef=min(candidates,key=lambda item:item[0])
    np.savez_compressed(out/'decoder.npz',mean=mean,basis=basis,coef=coef,shape=np.array(actions.shape[1:]),latent_mean=latent_mean,latent_std=latent_std)
    report=dict(sensor=sensor,device=device,epochs=epochs,train_episodes=paths,validation_episodes=vpaths,
                decoder_ridge=ridge,decoder_validation_candidates=[dict(mse=e,ridge=r) for e,r,_ in candidates],test_episodes_used=0,encoder_validation_uv_mse=best,validation_action_mse=action_mse,
                history=history,architecture='learned RGB-D spatial attention + metric depth latent + trajectory BC' if rgbd else 'learned RGB spatial-attention CNN + learned trajectory BC decoder',
                inference_inputs=['initial overhead RGB','initial aligned depth metres','depth validity'] if rgbd else ['initial overhead RGB'],runtime_detector=False,runtime_ik=False,
                conditioning='episode-start aligned RGB-D' if rgbd else 'episode-start RGB',crop=crop)
    (out/'training.json').write_text(json.dumps(report,indent=2));return report


def load_policy(model_dir):
    import torch
    torch.set_num_threads(2)
    model_dir=Path(model_dir)
    metadata=json.loads((model_dir/'training.json').read_text())
    sensor=metadata.get('sensor','rgb');crop=tuple(metadata.get('crop',CROP))
    from .rgbd_input import model_input,make_rgbd_encoder
    model=make_rgbd_encoder() if sensor=='rgbd' else make_encoder()
    model.load_state_dict(torch.load(model_dir/'encoder.pt',weights_only=True,map_location='cpu'));model.eval()
    data=np.load(model_dir/'decoder.npz',allow_pickle=False)
    def infer(rgb,depth=None):
        if sensor=='rgbd':
            if depth is None:raise ValueError('This policy requires aligned metric depth')
            x=model_input(rgb,depth,crop)
            if np.mean(x[4])<.1:raise ValueError('Insufficient valid depth; refusing to infer actions')
        else:x=image_tensor(rgb)
        with torch.no_grad():uv=model(torch.tensor(x[None]))[0].numpy()
        lm=data['latent_mean'] if 'latent_mean' in data else np.zeros(2)
        ls=data['latent_std'] if 'latent_std' in data else np.ones(2)
        actions=(features((uv-lm)/ls)@data['coef']@data['basis']+data['mean']).reshape(tuple(data['shape']))
        return actions,uv[0]
    infer.sensor=sensor
    return infer



def execution_audits(root,sim):
    profile=json.loads((Path(root)/'split.json').read_text()).get('cup_profile') or {}
    if profile.get('contact_revision',0)<3:return None,None
    from .contact_audit import ContactAudit,HoldAudit
    audit=ContactAudit(sim,profile['max_penetration_m']);audit.close_end=600+profile.get('extra_squeeze_steps',0);audit.minimum_side_force=profile.get('minimum_side_force_n',.05)
    return audit,HoldAudit()


def audited_step(sim,a,t,total,audit,hold_audit):
    if audit:
        audit.phase='open' if t<60 else ('approach' if t<480 else ('close' if t<audit.close_end else ('lift' if t<total-100 else 'hold')))
        audit.step(sim,a)
        if t>=total-100:hold_audit.update(sim)
        if audit.max_penetration>audit.limit or audit.max_hand_table_penetration>audit.limit or audit.nonfinger_contacts:
            raise RuntimeError('Execution rejected: penetration/collision budget exceeded')
    else:sim.step(a)


def evaluate(root):
    root=Path(root);out=root/'test';out.mkdir(exist_ok=False)
    manifest=json.loads((root/'split.json').read_text());policy=load_policy(root/'model');rows=[]
    train_positions=np.array(manifest['train']+manifest['validation'])
    for index,xy in enumerate(manifest['test']):
        assert np.linalg.norm(train_positions-np.array(xy),axis=1).min()>.0069
        # Same unseen physical scene for actual RGB and blank-RGB causal ablation.
        for condition in (('normal','blank_rgb','missing_depth','fixed_depth') if policy.sensor=='rgbd' else ('normal','blank_rgb')):
            ablation=condition!='normal'
            sim=make_sim(root,xy);audit,hold_audit=execution_audits(root,sim);row=dict(index=index,spawn_xy=xy,condition=condition,blank_image=condition=='blank_rgb')
            try:
                frame=sim.rgbd() if policy.sensor=='rgbd' else None
                rgb=frame.rgb if frame else sim.image();input_rgb=np.zeros_like(rgb) if condition=='blank_rgb' else rgb
                input_depth=frame.depth if frame else None
                if condition=='missing_depth':input_depth=np.zeros_like(frame.depth)
                if condition=='fixed_depth':
                    with h5py.File(sorted((root/'train').glob('*.hdf5'))[0]) as ref:input_depth=ref['initial_depth_m'][:]
                actions,uv=policy(input_rgb,input_depth)
                hold=[];clipped=0
                for t,a in enumerate(actions):
                    lo,hi=sim.model.actuator_ctrlrange.T
                    clipped+=int(np.any((a<lo)|(a>hi)));audited_step(sim,np.clip(a,lo,hi),t,len(actions),audit,hold_audit)
                    if t>=len(actions)-100:hold.append(float(sim.data.body('cup').xpos[2]))
                row.update(measure(sim,hold))
                if audit:
                    from .contact_audit import side_pinch_report
                    row.update(audit.report());row.update(hold_audit.report());row.update(side_pinch_report(sim,audit.minimum_side_force));row['success']=row['success'] and row['collision_pass'] and row['stable_hold'] and row['side_wall_pinch']
                row.update(clip_steps=clipped,visual_latent=uv.tolist(),runtime_ik=False,runtime_detector=False)
                if not ablation:
                    Image.fromarray(rgb).save(out/f'{index:02d}_input.png')
                    Image.fromarray(sim.image('teleoperator_pov')).save(out/f'{index:02d}_result.png')
                    np.save(out/f'{index:02d}_actions.npy',actions)
            except Exception as exc:
                row.update(success=False,error=f'{type(exc).__name__}: {exc}')
                if audit:row.update(audit.report())
            finally:sim.close()
            rows.append(row);(out/'trials.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
    normal=[r for r in rows if r['condition']=='normal'];blank=[r for r in rows if r['condition']=='blank_rgb']
    report=dict(successes=sum(r['success'] for r in normal),trials=len(normal),
                success_rate=sum(r['success'] for r in normal)/len(normal),depth_ablations={c:sum(r['success'] for r in rows if r['condition']==c) for c in ('missing_depth','fixed_depth') if any(r['condition']==c for r in rows)},blank_image_successes=sum(r['success'] for r in blank),
                held_out_positions=True,train_test_min_separation_m=float(min(np.linalg.norm(train_positions-np.array(p),axis=1).min() for p in manifest['test'])),
                policy_inputs=['initial_RGB','initial_depth_m','validity'] if policy.sensor=='rgbd' else ['initial_RGB'],runtime_detector=False,runtime_ik=False,hardware_motion_commanded=False,
                note='Static single coffee mug, fixed camera and geometry. Test set evaluated after training; all failures retained.')
    (out/'report.json').write_text(json.dumps(report,indent=2));return report



def rollout(root,xy,output,viewer=False):
    from contextlib import nullcontext
    import time
    import subprocess
    root=Path(root);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    sim=make_sim(root,xy);policy=load_policy(root/'model');gui=None;video=None;close_video=None;close_camera=None
    audit,hold_audit=execution_audits(root,sim)
    try:
        if viewer:
            from mujoco import viewer as viewer_module
            gui=viewer_module.launch_passive(sim.model,sim.data)
        frame=sim.rgbd() if policy.sensor=='rgbd' else None
        rgb=frame.rgb if frame else sim.image();Image.fromarray(rgb).save(out/'input.png')
        if frame:np.save(out/'depth_m.npy',frame.depth)
        actions,latent=policy(rgb,frame.depth if frame else None);hold=[]
        video=subprocess.Popen(['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s','320x240','-r','5','-i','pipe:0','-an','-c:v','libx264','-threads','2','-pix_fmt','yuv420p',str(out/'rollout.mp4')],stdin=subprocess.PIPE)
        if audit:
            close_camera=mujoco.MjvCamera();close_camera.lookat[:]=[xy[0],xy[1],.13]
            close_camera.distance=.50;close_camera.azimuth=90;close_camera.elevation=-20
            close_video=subprocess.Popen(['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s','320x240','-r','5','-i','pipe:0','-an','-c:v','libx264','-threads','2','-pix_fmt','yuv420p',str(out/'closeup.mp4')],stdin=subprocess.PIPE)
        for t,a in enumerate(actions):
            if gui and not gui.is_running():raise KeyboardInterrupt('Viewer closed')
            with gui.lock() if gui else nullcontext():
                lo,hi=sim.model.actuator_ctrlrange.T;audited_step(sim,np.clip(a,lo,hi),t,len(actions),audit,hold_audit)
                if t>=len(actions)-100:hold.append(float(sim.data.body('cup').xpos[2]))
                if t%10==0:
                    video.stdin.write(sim.image('teleoperator_pov').tobytes())
                    if close_video:
                        # Observer-only tracking keeps the raised cup in view;
                        # this camera is never supplied to the learned policy.
                        close_camera.lookat[:]=sim.data.body('cup').xpos+[0,0,.015]
                        observed=Image.fromarray(sim.image(close_camera))
                        draw=ImageDraw.Draw(observed)
                        gap=np.linalg.norm(sim.data.geom('left/left_g2').xpos-sim.data.geom('left/right_g2').xpos)-.0012
                        draw.rectangle((0,0,320,36),fill=(20,20,20))
                        draw.text((4,2),f'{audit.phase} | pad gap {gap*1000:.1f} mm',fill='white')
                        draw.text((4,18),f'max cup overlap {audit.max_penetration*1000:.3f} mm',fill='white')
                        close_video.stdin.write(np.asarray(observed).tobytes())
            if gui:gui.sync();time.sleep(sim.dt)
        video.stdin.close()
        if video.wait()!=0:raise RuntimeError('Video encoding failed')
        if close_video:
            close_video.stdin.close()
            if close_video.wait()!=0:raise RuntimeError('Closeup encoding failed')
        report=dict(spawn_xy=xy,visual_latent=latent.tolist(),**measure(sim,hold),policy_inputs=['initial_RGB','initial_depth_m','validity'] if policy.sensor=='rgbd' else ['initial_RGB'],runtime_ik=False,runtime_detector=False,hardware_motion_commanded=False)
        if audit:
            from .contact_audit import side_pinch_report
            report.update(audit.report());report.update(hold_audit.report());report.update(side_pinch_report(sim,audit.minimum_side_force));report['success']=report['success'] and report['collision_pass'] and report['stable_hold'] and report['side_wall_pinch']
        (out/'report.json').write_text(json.dumps(report,indent=2));np.save(out/'actions.npy',actions)
        return report
    finally:
        if video and video.poll() is None:video.terminate();video.wait(timeout=10)
        if close_video and close_video.poll() is None:close_video.terminate();close_video.wait(timeout=10)
        if gui:gui.close()
        sim.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['plan','collect','train','test','rollout']);p.add_argument('--root',required=True);p.add_argument('--epochs',type=int,default=1200)
    p.add_argument('--seed',type=int,default=1927);p.add_argument('--cup-profile');p.add_argument('--sensor',choices=['rgb','rgbd'],default='rgb');p.add_argument('--cup',nargs=2,type=float,default=None);p.add_argument('--output');p.add_argument('--viewer',action='store_true')
    a=p.parse_args()
    if a.phase=='plan':r=plan(a.root,a.sensor,json.loads(Path(a.cup_profile).read_text()) if a.cup_profile else None,a.seed)
    elif a.phase=='collect':r=collect(a.root)
    elif a.phase=='train':r=train(a.root,a.epochs)
    elif a.phase=='test':r=evaluate(a.root)
    else:
        if not a.output:p.error('rollout requires --output')
        r=rollout(a.root,a.cup or json.loads((Path(a.root)/'split.json').read_text())['test'][0],a.output,a.viewer)
    print(json.dumps(r,indent=2) if isinstance(r,dict) else 'collection complete')

if __name__=='__main__':main()
