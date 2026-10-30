#!/usr/bin/env python3
"""Import-only check of the pinned upstream runtime; robot creation is prohibited."""
import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def prohibited(*args, **kwargs):
    raise RuntimeError('Hardware import check forbids ROS initialization and robot construction')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    results = {}
    modules = ['rclpy', 'cv2', 'cv_bridge', 'numpy', 'h5py', 'yaml', 'matplotlib',
               'modern_robotics', 'dm_env', 'pyquaternion', 'tqdm', 'transforms3d',
               'interbotix_xs_msgs.msg', 'interbotix_common_modules.common_robot.robot',
               'interbotix_xs_modules.xs_robot.arm', 'aloha.robot_utils', 'aloha.real_env']
    for name in modules:
        try:
            module = importlib.import_module(name)
            results[name] = {'ok': True, 'path': getattr(module, '__file__', None),
                             'version': getattr(module, '__version__', None)}
            if name == 'rclpy':
                module.init = prohibited
                module.create_node = prohibited
                from rclpy.node import Node
                Node.__init__ = prohibited
            if name == 'interbotix_xs_modules.xs_robot.arm':
                module.InterbotixManipulatorXS.__init__ = prohibited
        except Exception as exc:
            results[name] = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
    for script in ('teleop', 'record_episodes', 'replay_episodes', 'sleep'):
        name = f'upstream_script:{script}'
        try:
            # The non-__main__ module name leaves the CLI and control functions uncalled.
            path = ROOT / 'third_party/interbotix_aloha/scripts' / f'{script}.py'
            spec = importlib.util.spec_from_file_location(f'aloha_import_check_{script}', path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            results[name] = {'ok': True, 'path': str(path)}
        except Exception as exc:
            results[name] = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
    try:
        import numpy as np
        from cv_bridge import CvBridge
        bridge = CvBridge()
        source = np.zeros((8, 8, 3), dtype=np.uint8)
        msg = bridge.cv2_to_imgmsg(source, encoding='rgb8')
        restored = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        assert np.array_equal(source, restored)
        results['cv_bridge_roundtrip'] = {'ok': True}
    except Exception as exc:
        results['cv_bridge_roundtrip'] = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
    report = {'python': sys.executable, 'all_passed': all(x['ok'] for x in results.values()),
              'motion_commanded': False, 'ros_initialized': False, 'results': results}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n')
    print(text)
    return 0 if report['all_passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
