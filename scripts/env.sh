#!/usr/bin/env bash
# Source this file, do not execute it. No global shell configuration is changed.
export ALOHA_LAB_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f /opt/ros/humble/setup.bash ]]; then
  source /opt/ros/humble/setup.bash
fi
if [[ -f "$ALOHA_LAB_ROOT/ros2_ws/install/setup.bash" ]]; then
  source "$ALOHA_LAB_ROOT/ros2_ws/install/setup.bash"
fi
source "$ALOHA_LAB_ROOT/.venv/bin/activate"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS=2
export WANDB_MODE=disabled

export ROS_LOG_DIR="$ALOHA_LAB_ROOT/runs/ros-logs"
mkdir -p "$ROS_LOG_DIR"

export MPLCONFIGDIR="$ALOHA_LAB_ROOT/.cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"
