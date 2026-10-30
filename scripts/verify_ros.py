#!/usr/bin/env python3
"""Launch only MuJoCo and verify actual DDS delivery, then stop it."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import rclpy
from sensor_msgs.msg import JointState
from rosgraph_msgs.msg import Clock
from aloha2_lab.sim import JOINTS

root=Path(__file__).resolve().parents[1]
out=root/'runs/ros-verification'
out.mkdir(parents=True,exist_ok=True)
os.environ['ROS_DOMAIN_ID']='142'
os.environ['ROS_LOCALHOST_ONLY']='1'
with (out/'sim.log').open('w') as log:
    process=subprocess.Popen(['ros2','launch','aloha2_lab_bringup','sim.launch.py'],stdout=log,stderr=log,start_new_session=True)
    rclpy.init()
    node=rclpy.create_node('aloha2_verifier')
    joints=[]
    clocks=[]
    node.create_subscription(JointState,'/aloha2_sim/joint_states',lambda msg:joints.append(msg),10)
    node.create_subscription(Clock,'/clock',lambda msg:clocks.append(msg),10)
    try:
        deadline=time.monotonic()+20
        while (len(joints)<10 or len(clocks)<10) and time.monotonic()<deadline:
            rclpy.spin_once(node,timeout_sec=.1)
            if process.poll() is not None:
                raise RuntimeError('Simulation launch exited; inspect sim.log')
        assert len(joints)>=10 and len(clocks)>=10,'DDS message timeout'
        assert all(msg.name==JOINTS and len(msg.position)==14 for msg in joints)
        ts=[msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9 for msg in joints]
        assert all(b>a for a,b in zip(ts,ts[1:]))
        report=dict(joint_messages=len(joints),clock_messages=len(clocks),actuators=14,
                    monotonic_time=True,ros_domain_id=142,hardware_motion_commanded=False)
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report))
    finally:
        node.destroy_node()
        rclpy.shutdown()
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                process.wait(timeout=5)
