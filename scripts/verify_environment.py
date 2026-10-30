#!/usr/bin/env python3
"""Verify imports, CUDA computation, MuJoCo stepping and all camera renders."""
import argparse
import json
import platform
from pathlib import Path
import shutil
import subprocess
import numpy as np
import mujoco
import torch
from PIL import Image
from aloha2_lab.sim import AlohaSim, CAMERAS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; GPU verification did not pass')
    torch.manual_seed(3)
    x = torch.randn(128,128,device='cuda',requires_grad=True)
    loss = (x@x.T).square().mean()
    loss.backward()
    torch.cuda.synchronize()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
    sim = AlohaSim()
    try:
        for _ in range(200):
            q,v = sim.step(sim.data.ctrl.copy())
        for name in CAMERAS+['teleoperator_pov']:
            rgb = sim.image(name)
            assert rgb.shape == (240,320,3) and rgb.std() > 1
            Image.fromarray(rgb).save(args.output/f'{name}.png')
        report = dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
                      gpu=torch.cuda.get_device_name(0),capability=torch.cuda.get_device_capability(0),
                      cuda_forward_backward=True,mujoco=mujoco.__version__,simulation_steps=200,
                      control_dt=sim.dt,actuators=sim.model.nu,cameras=CAMERAS,
                      finite_state=bool(np.isfinite(q).all() and np.isfinite(v).all()),
                      free_disk_gib=round(shutil.disk_usage(args.output).free/2**30,2),
                      hardware_motion_commanded=False,manipulation_success_evaluated=False)
    finally:
        sim.close()
    try:
        import rclpy
        report['rclpy_import'] = True
        result = subprocess.run(['ros2','pkg','prefix','aloha'],capture_output=True,text=True,timeout=10)
        report['aloha_package_prefix'] = result.stdout.strip() if result.returncode == 0 else None
    except ImportError:
        report['rclpy_import'] = False
    (args.output/'environment.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
