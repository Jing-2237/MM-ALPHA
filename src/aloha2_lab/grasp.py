"""Simulation-only, vision-guided IK baseline. No object pose read for control."""
import json
from contextlib import nullcontext
from .dataset import EpisodeWriter
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image,ImageDraw
from .sim import AlohaSim
from .vision import locate_cup


def ik(sim, target, rotation):
    m=sim.model;d=mujoco.MjData(m)
    d.qpos[sim.qadr]=sim.data.qpos[sim.qadr]
    sid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'left/gripper')
    dofs=sim.vadr[:6];qadr=sim.qadr[:6]
    for _ in range(500):
        mujoco.mj_forward(m,d)
        R=d.site_xmat[sid].reshape(3,3)
        er=.5*sum((np.cross(R[:,i],rotation[:,i]) for i in range(3)))
        ep=np.asarray(target)-d.site_xpos[sid]
        if np.linalg.norm(ep)<.001 and np.linalg.norm(er)<.015:return d.qpos[qadr].copy()
        jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jacSite(m,d,jp,jr,sid)
        J=np.vstack([jp[:,dofs],jr[:,dofs]*.3]);e=np.r_[ep,er*.3]
        delta=J.T@np.linalg.solve(J@J.T+np.eye(6)*1e-4,e)
        d.qpos[qadr]+=np.clip(delta,-.08,.08)
        d.qpos[qadr]=np.clip(d.qpos[qadr],m.actuator_ctrlrange[:6,0],m.actuator_ctrlrange[:6,1])
    raise RuntimeError(f'IK unreachable: residual {np.linalg.norm(ep):.4f}m')


def run(output,xy=(-.16,.05),viewer=False,record_demo=False):
    sim=AlohaSim(cup_xy=xy);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    gui=None
    writer=EpisodeWriter(out/"episode_000000.hdf5","vision_ik_teacher_not_human",dt=sim.dt) if record_demo else None
    finished=False
    try:
        if viewer:
            from mujoco import viewer as viewer_module
            gui=viewer_module.launch_passive(sim.model,sim.data)
        def step(action):
            if gui and not gui.is_running():
                raise KeyboardInterrupt("Viewer closed; simulation stopped")
            with gui.lock() if gui else nullcontext():
                if writer:
                    q,v=sim.observe()
                    writer.append(q,v,action,sim.data.time,{'overhead_cam':sim.image()})
                sim.step(action)
            if gui:
                gui.sync()
                import time
                time.sleep(sim.dt)
        frame=sim.rgbd();found=locate_cup(frame,sim.data.time)
        img=Image.fromarray(frame.rgb);ImageDraw.Draw(img).rectangle(found['bbox_xyxy'],outline='lime',width=2);img.save(out/'detection.png')
        np.savez_compressed(out/'observation.npz',rgb=frame.rgb,depth=frame.depth,K=frame.K,T_world_optical=frame.T_world_optical,timestamp=frame.timestamp)
        # Tool +X downward, +Y is closing axis.
        R=np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
        centre=np.array(found['position_world']);trajectory=[]
        def move(target,gripper,steps=180):
            qgoal=ik(sim,target,R);start=sim.data.ctrl.copy();end=start.copy();end[:6]=qgoal;end[6]=gripper
            for t in range(steps):
                u=(t+1)/steps;u=u*u*(3-2*u)
                action=start+(end-start)*u
                step(action);trajectory.append(action.copy())
        move(centre+[0,0,.16],.037)
        move(centre,.037)
        for _ in range(120):
            a=sim.data.ctrl.copy();a[6]=max(.016,a[6]-.0003);step(a);trajectory.append(a.copy())
        move(centre+[0,0,.15],.016,240)
        for _ in range(100):step(sim.data.ctrl.copy())
        Image.fromarray(sim.image('teleoperator_pov')).save(out/'lift.png')
        # Ground truth is read ONLY here for independent evaluation after execution.
        height=float(sim.data.body('cup').xpos[2])
        cupid=mujoco.mj_name2id(sim.model,mujoco.mjtObj.mjOBJ_BODY,'cup')
        touching=set()
        for contact in sim.data.contact:
            pair=[int(contact.geom1),int(contact.geom2)]
            if any(sim.model.geom_bodyid[g]==cupid for g in pair):
                for g in pair:
                    name=mujoco.mj_id2name(sim.model,mujoco.mjtObj.mjOBJ_BODY,int(sim.model.geom_bodyid[g]))
                    if name in ('left/left_finger_link','left/right_finger_link'):touching.add(name)
        report={'detection':found,'final_cup_height_m':height,'lift_success':height>.12 and len(touching)==2,'finger_contacts':sorted(touching),
                'control_source':'RGB-D pixels + calibrated camera + known fixture cup geometry',
                'ground_truth_used_for_control':False,'hardware_motion_commanded':False}
        (out/'report.json').write_text(json.dumps(report,indent=2));np.save(out/'actions.npy',trajectory)
        if writer:writer.file.attrs["task_success"]=report["lift_success"]
        finished=True
        return report
    finally:
        if writer:writer.close(complete=finished)
        if gui:gui.close()
        sim.close()
