#!/usr/bin/env bash
# Recreate the workstation baseline. This script never starts robot drivers.
# Source commits and core Python versions are pinned; Ubuntu/ROS apt packages
# intentionally follow their signed Humble/Jammy repositories.
set -euo pipefail
lab_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$lab_root"
phase="${1:-all}"
case "$phase" in all|system|sources|python|hardware-python|build) ;; *) echo 'Usage: bash scripts/bootstrap.sh [all|system|sources|python|hardware-python|build]' >&2; exit 2;; esac
source /etc/os-release
if [[ "$ID" != ubuntu || "$VERSION_ID" != 22.04 || "$(uname -m)" != x86_64 ]]; then
  echo 'This bootstrap is reviewed for Ubuntu 22.04 x86_64 only.' >&2
  exit 1
fi
mkdir -p .cache/bootstrap

if [[ "$phase" == all || "$phase" == system ]]; then
  sudo apt-get -o Acquire::ForceIPv4=true -o Acquire::https::Timeout=20 -o Acquire::Retries=2 update
  sudo apt-get install --no-remove -y ca-certificates curl gnupg git python3 python3-venv python3-pip
  # Reuse an existing ROS source if apt can already find Humble. Otherwise
  # choose an available official/mirror endpoint, still checking ROS signatures.
  if ! apt-cache show ros-humble-ros-base >/dev/null 2>&1; then
    python3 - <<'PY'
import json
from pathlib import Path
s=json.loads(Path('config/sources.lock.json').read_text())['ros']
Path('.cache/bootstrap/ros-key-url').write_text(s['key_url'])
Path('.cache/bootstrap/ros-key-sha256').write_text(s['key_sha256'])
Path('.cache/bootstrap/ros-repositories').write_text('\n'.join(s['repositories'])+'\n')
PY
    curl -fL --retry 2 --connect-timeout 10 --max-time 90 "$(cat .cache/bootstrap/ros-key-url)" -o .cache/bootstrap/ros.key
    printf '%s  %s\n' "$(cat .cache/bootstrap/ros-key-sha256)" '.cache/bootstrap/ros.key' | sha256sum --check -
    sudo install -m 644 .cache/bootstrap/ros.key /usr/share/keyrings/aloha2-ros-archive-keyring.gpg
    ros_repository=''
    while IFS= read -r candidate; do
      if curl -fsL --connect-timeout 10 --max-time 25 "$candidate/dists/jammy/InRelease" -o .cache/bootstrap/ros-inrelease; then
        if gpgv --keyring "$lab_root/.cache/bootstrap/ros.key" .cache/bootstrap/ros-inrelease; then
          ros_repository="$candidate"
          break
        fi
      fi
    done < .cache/bootstrap/ros-repositories
    if [[ -z "$ros_repository" ]]; then
      echo 'No reachable ROS repository with a valid ROS signature.' >&2
      exit 1
    fi
    printf 'deb [arch=amd64 signed-by=/usr/share/keyrings/aloha2-ros-archive-keyring.gpg] %s jammy main\n' "$ros_repository" > .cache/bootstrap/aloha2-ros2.list
    if [[ -f /etc/apt/sources.list.d/aloha2-ros2.list ]] && ! cmp -s .cache/bootstrap/aloha2-ros2.list /etc/apt/sources.list.d/aloha2-ros2.list; then
      echo 'Existing aloha2-ros2.list differs; review it instead of overwriting it.' >&2
      exit 1
    fi
    sudo install -m 644 .cache/bootstrap/aloha2-ros2.list /etc/apt/sources.list.d/aloha2-ros2.list
    sudo apt-get -o Acquire::ForceIPv4=true -o Acquire::https::Timeout=20 -o Acquire::Retries=2 update
  fi
  sudo apt-get -o Acquire::ForceIPv4=true -o Acquire::Retries=3 install --no-remove -y build-essential cmake python3-colcon-common-extensions python3-rosdep python3-vcstool \
    ros-humble-desktop ros-humble-ros2-control ros-humble-ros2-controllers ros-humble-xacro ros-humble-cv-bridge \
    ros-humble-realsense2-camera ros-humble-realsense2-description ros-humble-message-filters ros-humble-dynamixel-sdk \
    ros-humble-nav2-msgs ros-humble-tf-transformations ros-humble-usb-cam ros-humble-joy ros-humble-teleop-twist-joy \
    ros-humble-kdl-parser ros-humble-orocos-kdl-vendor ros-humble-joint-state-publisher-gui libyaml-cpp-dev \
    python3-transforms3d python3-opencv python3-matplotlib python3-numpy \
    libglfw3 libglew-dev libegl1-mesa-dev ffmpeg
fi

if [[ "$phase" == all || "$phase" == sources ]]; then
  python3 - <<'PY'
import json, subprocess
from pathlib import Path
root=Path.cwd()
lock=json.loads((root/'config/sources.lock.json').read_text())
run=lambda *a:subprocess.run(a,check=True)
for name,spec in lock['repositories'].items():
    path=root/'third_party'/name
    if path.exists():
        revision=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
        if revision != spec['commit']:
            raise RuntimeError(f'{name}: existing checkout differs; no reset performed')
        if subprocess.check_output(['git','-C',str(path),'status','--porcelain'],text=True).strip():
            raise RuntimeError(f'{name}: local modifications require review')
        continue
    path.mkdir(parents=True)
    run('git','-C',str(path),'init')
    run('git','-C',str(path),'remote','add','origin',spec['url'])
    if spec.get('sparse_paths'):
        run('git','-C',str(path),'sparse-checkout','init','--cone')
        run('git','-C',str(path),'sparse-checkout','set',*spec['sparse_paths'])
    run('git','-C',str(path),'fetch','--depth','1','--filter=blob:none','origin',spec['commit'])
    run('git','-C',str(path),'checkout','--detach',spec['commit'])
PY
  python3 scripts/setup_ros_workspace.py
fi

if [[ "$phase" == all || "$phase" == python ]]; then
  /usr/bin/python3 -m venv .venv
  .venv/bin/python -m pip install --no-cache-dir 'pip>=24' 'setuptools>=68' wheel
  .venv/bin/python -m pip install --no-cache-dir -e .
  .venv/bin/python -m pip install --no-cache-dir 'torch==2.7.1' --index-url https://download.pytorch.org/whl/cu128
  .venv/bin/python -m pip check
fi

if [[ "$phase" == all || "$phase" == hardware-python ]]; then
  # Keep ROS/OpenCV ABI compatibility with Ubuntu's NumPy and compiled modules.
  /usr/bin/python3 -m venv --system-site-packages .venv-hardware
  .venv-hardware/bin/python -m pip install --no-cache-dir \
    'modern_robotics==1.1.1' 'dm-env==1.6' 'pyquaternion==0.9.9' 'tqdm==4.67.1' 'h5py==3.14.0' \
    'cffi==1.17.1' 'pycparser==2.22'
  .venv-hardware/bin/python -c 'import cv2, matplotlib, numpy, modern_robotics, dm_env, pyquaternion, tqdm, h5py'
fi

if [[ "$phase" == all || "$phase" == build ]]; then
  # Use system Python for ROS packaging, separate from the learning venv.
  set +u
  source /opt/ros/humble/setup.bash
  set -u
  /usr/bin/python3 scripts/setup_ros_workspace.py
  cd ros2_ws
  PATH="/usr/bin:$PATH" /usr/bin/colcon build --symlink-install --executor sequential --packages-up-to aloha aloha2_lab_bringup
fi
printf 'Bootstrap stage %s finished. Open a fresh shell and source %s/scripts/env.sh\n' "$phase" "$lab_root"
