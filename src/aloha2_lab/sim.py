from pathlib import Path
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[2]
JOINTS = [f"{side}/{joint}" for side in ("left", "right") for joint in
          ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate", "gripper")]
CAMERAS = ["overhead_cam", "worms_eye_cam", "wrist_cam_left", "wrist_cam_right"]
HOME = np.array([0., -.6, .6, 0., .4, 0., .025] * 2)


class AlohaSim:
    """14 controls: left 6 joints + finger displacement (m), then right.

    A gripper control is ONE finger displacement, not jaw width, motor radians,
    PWM or the normalized gripper value of the real ALOHA data format.
    """
    def __init__(self, dt=.02, cup_xy=None, cup_profile=None):
        scene = ROOT / "third_party/mujoco_menagerie/aloha/scene.xml"
        if cup_xy is None:
            self.model = mujoco.MjModel.from_xml_path(str(scene))
        elif cup_profile is not None:
            from .paper_cup import build_model
            self.model = build_model(scene,cup_xy,cup_profile)
        else:
            from .cup_scene import cup_model
            self.model = cup_model(scene, cup_xy)
        self.data = mujoco.MjData(self.model)
        names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.model.nu)]
        if names != JOINTS:
            raise ValueError(f"Unexpected actuator order: {names}")
        joints = self.model.actuator_trnid[:, 0]
        self.qadr = self.model.jnt_qposadr[joints]
        self.vadr = self.model.jnt_dofadr[joints]
        self.substeps = round(dt / self.model.opt.timestep)
        if self.substeps < 1 or not np.isclose(self.substeps * self.model.opt.timestep, dt):
            raise ValueError("Control dt must be an integer multiple of simulation timestep")
        self.dt = dt
        self.renderer = None
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qadr] = HOME
        for side in ("left", "right"):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}/right_finger")
            self.data.qpos[self.model.jnt_qposadr[jid]] = .025
        self.data.ctrl[:] = HOME
        mujoco.mj_forward(self.model, self.data)
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)
        # Episode time starts after settling; velocities are real simulated values.
        self.data.time = 0.
        return self.observe()

    def observe(self):
        return self.data.qpos[self.qadr].copy(), self.data.qvel[self.vadr].copy()

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (14,) or not np.isfinite(action).all():
            raise ValueError("Action must contain 14 finite values")
        lo, hi = self.model.actuator_ctrlrange.T
        if np.any(action < lo) or np.any(action > hi):
            raise ValueError("Action exceeds model control limits")
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data, nstep=self.substeps)
        q, v = self.observe()
        if not np.isfinite(q).all() or not np.isfinite(v).all():
            raise RuntimeError("Simulation diverged")
        return q, v

    def image(self, camera="overhead_cam"):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=240, width=320)
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def close(self):
        if self.renderer:
            self.renderer.close()
            self.renderer = None

    def rgbd(self, camera="overhead_cam"):
        from .vision import RGBDFrame
        rgb=self.image(camera)
        self.renderer.enable_depth_rendering()
        try:
            self.renderer.update_scene(self.data,camera=camera)
            depth=self.renderer.render().copy()
        finally:
            self.renderer.disable_depth_rendering()
        cid=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_CAMERA,camera)
        height,width=depth.shape
        fy=height/(2*np.tan(np.deg2rad(self.model.cam_fovy[cid])/2))
        fx=fy
        if np.all(self.model.cam_sensorsize[cid]>0):
            fx=self.model.cam_intrinsic[cid,0]/self.model.cam_sensorsize[cid,0]*width
            fy=self.model.cam_intrinsic[cid,1]/self.model.cam_sensorsize[cid,1]*height
        K=np.array([[fx,0,width/2],[0,fy,height/2],[0,0,1.]])
        T=np.eye(4)
        T[:3,:3]=self.data.cam_xmat[cid].reshape(3,3)@np.diag([1,-1,-1])
        T[:3,3]=self.data.cam_xpos[cid]
        return RGBDFrame(rgb,depth,K,T,float(self.data.time))
