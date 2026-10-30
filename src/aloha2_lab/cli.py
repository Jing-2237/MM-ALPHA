import argparse
import json
from pathlib import Path
import time
import numpy as np
from .sim import AlohaSim, CAMERAS, HOME
from .dataset import EpisodeWriter, validate


def collect(args):
    sim = AlohaSim()
    rng = np.random.default_rng(args.seed)
    try:
        for episode in range(args.episodes):
            sim.reset()
            writer = EpisodeWriter(Path(args.output)/f"episode_{episode:06d}.hdf5", "scripted_smoke_not_human")
            # Bounded smooth joint-space motion to exercise the data path.
            amp = rng.uniform(.02,.09,14)
            amp[[6,13]] = .003
            frequency = rng.uniform(.3,.7,14)
            completed = False
            try:
                for step in range(args.steps):
                    q, v = sim.observe()
                    action = HOME + amp*np.sin(frequency*step*sim.dt)
                    images = {c:sim.image(c) for c in args.cameras}
                    writer.append(q,v,action,sim.data.time,images)
                    sim.step(action)
                completed = True
            finally:
                writer.close(complete=completed)
            print(json.dumps(validate(writer.path)), flush=True)
    finally:
        sim.close()


def manual(args):
    import mujoco.viewer
    sim = AlohaSim(cup_xy=tuple(args.cup) if args.cup else None)
    writer = EpisodeWriter(args.output, "human_mujoco_control_sliders") if args.output else None
    steps = 0
    finished = False
    try:
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            print("Use the viewer's Control sliders to demonstrate joint targets; close the window to save.", flush=True)
            while viewer.is_running():
                start = time.monotonic()
                with viewer.lock():
                    q, v = sim.observe()
                    action = sim.data.ctrl.copy()
                    images = {c:sim.image(c) for c in args.cameras} if writer else {}
                    if writer:
                        writer.append(q,v,action,sim.data.time,images)
                    sim.step(action)
                viewer.sync()
                steps += 1
                time.sleep(max(0,sim.dt-(time.monotonic()-start)))
            finished = True
    except KeyboardInterrupt:
        finished = True
    finally:
        if writer:
            writer.close(complete=finished)
        sim.close()
    return {"frames":steps,"output":args.output}


def evaluate(args):
    from .learning import load_policy
    policy = load_policy(args.checkpoint)
    sim = AlohaSim(dt=policy.control_dt)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    writer = EpisodeWriter(output/"rollout.hdf5", "learned_policy_simulation", dt=sim.dt)
    clamps, contacts = 0, 0
    finished = False
    try:
        for i in range(args.steps):
            q,v = sim.observe()
            proposed = policy(q,v)[0]  # Replan every step; chunk prediction is still trained.
            lo,hi = sim.model.actuator_ctrlrange.T
            rate = np.array([.4]*6+[.02]+[.4]*6+[.02])*sim.dt
            action = np.clip(proposed, np.maximum(lo,sim.data.ctrl-rate), np.minimum(hi,sim.data.ctrl+rate))
            clamps += int(not np.allclose(action,proposed))
            writer.append(q,v,action,sim.data.time)
            sim.step(action)
            contacts = max(contacts,sim.data.ncon)
        from PIL import Image
        Image.fromarray(sim.image("teleoperator_pov")).save(output/"final.png")
        finished = True
    finally:
        writer.close(complete=finished)
        sim.close()
    report = dict(steps=args.steps, finite=True, clipped_steps=clamps, max_contacts=contacts,
                  task_success_evaluated=False, hardware_motion_commanded=False)
    (output/"report.json").write_text(json.dumps(report,indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description="ALOHA 2 simulation and imitation learning lab")
    sub = parser.add_subparsers(dest="command",required=True)
    p = sub.add_parser("collect",help="Generate labelled scripted smoke episodes, not expert demonstrations")
    p.add_argument("--output",required=True)
    p.add_argument("--episodes",type=int,default=8)
    p.add_argument("--steps",type=int,default=180)
    p.add_argument("--seed",type=int,default=7)
    p.add_argument("--cameras",nargs="*",choices=CAMERAS,default=[])
    p = sub.add_parser("view",help="Interactive MuJoCo; optionally record human control sliders")
    p.add_argument("--output")
    p.add_argument("--cup",nargs=2,type=float,metavar=("X","Y"),help="Add registered blue cup for visual demonstrations")
    p.add_argument("--cameras",nargs="*",choices=CAMERAS,default=["overhead_cam"])
    p = sub.add_parser("train")
    p.add_argument("--data",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--epochs",type=int,default=80)
    p.add_argument("--chunk",type=int,default=10)
    p.add_argument("--seed",type=int,default=7)
    p.add_argument("--device",default="auto",choices=["auto","cpu","cuda"])
    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--steps",type=int,default=300)
    p = sub.add_parser("replay",help="Export recorded camera frames to MP4; sends no robot commands")
    p.add_argument("--episode",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--camera",default="overhead_cam")
    p = sub.add_parser("validate")
    p.add_argument("path")
    p = sub.add_parser("cup-grasp",help="RGB-D localization and physical grasp of registered blue cup in simulation")
    p.add_argument("--output",required=True)
    p.add_argument("--cup",nargs=2,type=float,default=[-.16,.05])
    p.add_argument("--viewer",action="store_true")
    p.add_argument("--record-demo",action="store_true",help="Save RGB+state+action HDF5 labelled IK teacher, not human")
    sub.add_parser("ros-sim",help="Publish simulated JointState and clock; no hardware topics")
    args = parser.parse_args()
    for name in ("steps", "episodes"):
        if hasattr(args,name) and getattr(args,name) < 1:
            parser.error(f"{name} must be positive")
    if args.command == "collect":
        result = collect(args)
    elif args.command == "view":
        result = manual(args)
    elif args.command == "train":
        from .learning import train
        result = train(args.data,args.output,args.epochs,args.chunk,args.seed,args.device)
    elif args.command == "evaluate":
        result = evaluate(args)
    elif args.command == "cup-grasp":
        from .grasp import run
        result = run(args.output,tuple(args.cup),args.viewer,args.record_demo)
    elif args.command == "replay":
        from .replay import export_video
        result = export_video(args.episode,args.output,args.camera)
    elif args.command == "validate":
        result = validate(args.path)
    else:
        from .ros_sim import main as ros_main
        result = ros_main()
    if result is not None:
        print(json.dumps(result,indent=2))
        if args.command == "cup-grasp" and not result["lift_success"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
