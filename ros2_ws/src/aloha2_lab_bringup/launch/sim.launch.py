import os
from pathlib import Path
from launch import LaunchDescription
from launch.actions import ExecuteProcess


def generate_launch_description():
    root = os.environ.get("ALOHA_LAB_ROOT")
    if not root or not (Path(root)/".venv/bin/python").exists():
        raise RuntimeError("First source the project's scripts/env.sh")
    return LaunchDescription([ExecuteProcess(
        cmd=[str(Path(root)/".venv/bin/python"),"-m","aloha2_lab.cli","ros-sim"],output="screen")])
