#!/usr/bin/env python3
"""Read-only inventory. Never instantiate Interbotix robots or call services."""
import argparse
import glob
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def read_topics(seconds=2.):
    import rclpy
    rclpy.init(args=[])
    node = rclpy.create_node('aloha2_read_only_audit')
    try:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=.1)
        return dict(node.get_topic_names_and_types())
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topics', action='store_true', help='Read local ROS graph; no hardware commands')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    packages = ['aloha', 'interbotix_xs_sdk', 'interbotix_xsarm_control', 'interbotix_gravity_compensation',
                'realsense2_camera', 'rosbag2_transport']
    found = {}
    try:
        from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
        for name in packages:
            try:
                found[name] = get_package_share_directory(name)
            except PackageNotFoundError:
                found[name] = None
    except ImportError:
        found = {name: None for name in packages}
    devices = sorted(set(glob.glob('/dev/ttyDXL*') + glob.glob('/dev/serial/by-id/*') + glob.glob('/dev/video*')))
    report = {'motion_commanded': False, 'torque_enabled': False, 'packages': found,
              'devices': [{'path': d, 'readable': os.access(d, os.R_OK), 'writable': os.access(d, os.W_OK)} for d in devices],
              'disk_free_gib': round(shutil.disk_usage(ROOT).free/2**30, 2),
              'ros_domain_id': os.getenv('ROS_DOMAIN_ID', '0'),
              'note': 'Package/device presence does not prove calibration, connectivity or safe operation.'}
    if shutil.which('lsusb'):
        report['usb_inventory'] = subprocess.run(['lsusb'], capture_output=True, text=True, timeout=10).stdout.splitlines()
    if args.topics:
        report['topics'] = read_topics()
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text+'\n')
    print(text)


if __name__ == '__main__':
    main()
