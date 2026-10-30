#!/usr/bin/env python3
"""Verify raw recording using synthetic test topics only, never robot topics.

Uses isolated ROS_DOMAIN_ID=143, localhost only. No hardware SDK is imported,
no robot is constructed, and no control message uses a hardware namespace.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

os.environ['ROS_DOMAIN_ID'] = '143'
os.environ['ROS_LOCALHOST_ONLY'] = '1'

import rclpy
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import JointState, Image, CameraInfo
from interbotix_xs_msgs.msg import JointGroupCommand, JointSingleCommand
import rosbag2_py
import yaml

ROOT = Path(__file__).resolve().parents[1]
ARM_JOINTS = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate', 'gripper']
TYPES = {
    'sensor_msgs/msg/JointState': JointState,
    'sensor_msgs/msg/Image': Image,
    'sensor_msgs/msg/CameraInfo': CameraInfo,
    'interbotix_xs_msgs/msg/JointGroupCommand': JointGroupCommand,
    'interbotix_xs_msgs/msg/JointSingleCommand': JointSingleCommand,
}


def inspect_bag(path, expected):
    metadata = yaml.safe_load((path/'metadata.yaml').read_text())['rosbag2_bagfile_information']
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(path),storage_id='sqlite3'),
                rosbag2_py.ConverterOptions(input_serialization_format='cdr',output_serialization_format='cdr'))
    actual = {x.name:x.type for x in reader.get_all_topics_and_types()}
    assert actual == expected, f'Wrong bag topics: {actual}'
    counts = Counter()
    previous = {}
    while reader.has_next():
        topic,raw,received = reader.read_next()
        message = deserialize_message(raw,TYPES[expected[topic]])
        assert received >= previous.get(topic,0), f'Bag time reversed: {topic}'
        previous[topic] = received
        counts[topic] += 1
        if isinstance(message,JointState):
            assert message.name == ARM_JOINTS and len(message.position) == 7
            assert message.header.frame_id == 'synthetic_fixture'
        elif isinstance(message,Image):
            if topic.endswith('/depth'):
                assert (message.width,message.height,message.encoding,message.step)==(8,8,'16UC1',16)
                assert bytes(message.data)==b'\xe8\x03'*64
            else:
                assert (message.width,message.height,message.encoding,message.step) == (8,8,'rgb8',24)
                assert len(message.data)==192
            assert message.header.frame_id=='synthetic_fixture'
        elif isinstance(message,CameraInfo):
            assert message.width==8 and message.height==8 and list(message.k)==[8.,0.,4.,0.,8.,4.,0.,0.,1.]
        elif isinstance(message,JointGroupCommand):
            assert message.name == 'arm' and len(message.cmd) == 6
        else:
            assert message.name == 'gripper' and abs(message.cmd-.1)<1e-6
    assert all(counts[t]>=5 for t in expected), f'Too few messages: {counts}'
    assert metadata['message_count'] == sum(counts.values())
    for item in metadata['topics_with_message_count']:
        assert item['message_count'] == counts[item['topic_metadata']['name']]
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'runs/recording-verification')
    args = parser.parse_args()
    run_name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    out = args.output/run_name
    out.mkdir(parents=True,exist_ok=False)
    prefix = '/aloha2_test/run_'+run_name.replace('.','_')
    expected = {}
    for side in ('left','right'):
        for role in ('leader','follower'):
            expected[f'{prefix}/{role}_{side}/joint_states'] = 'sensor_msgs/msg/JointState'
        expected[f'{prefix}/follower_{side}/commands/joint_group'] = 'interbotix_xs_msgs/msg/JointGroupCommand'
        expected[f'{prefix}/follower_{side}/commands/joint_single'] = 'interbotix_xs_msgs/msg/JointSingleCommand'
    for camera in ('high','low','wrist_left','wrist_right'):
        expected[f'{prefix}/camera_{camera}/image'] = 'sensor_msgs/msg/Image'
    expected[f'{prefix}/camera_high/depth']='sensor_msgs/msg/Image'
    expected[f'{prefix}/camera_high/info']='sensor_msgs/msg/CameraInfo'
    config = {'robot':'aloha_stationary','source':'synthetic_ros_verification',
              'required_topics':expected,'optional_topics':{}}
    config_path = out/'synthetic.yaml'
    config_path.write_text(yaml.safe_dump(config,sort_keys=False))
    rclpy.init(args=[])
    node = rclpy.create_node('aloha2_synthetic_recorder_verification')
    publishers = {t:node.create_publisher(TYPES[k],t,10) for t,k in expected.items()}
    sequence = 0
    def publish():
        nonlocal sequence
        sequence += 1
        stamp = node.get_clock().now().to_msg()
        for topic,pub in publishers.items():
            msg = TYPES[expected[topic]]()
            if isinstance(msg,JointState):
                msg.header.stamp,msg.header.frame_id = stamp,'synthetic_fixture'
                msg.name = ARM_JOINTS
                msg.position = [sequence*.001]*7
                msg.velocity = [.05]*7
            elif isinstance(msg,Image):
                msg.header.stamp,msg.header.frame_id = stamp,'synthetic_fixture'
                if topic.endswith('/depth'):
                    msg.width,msg.height,msg.step,msg.encoding=8,8,16,'16UC1';msg.data=b'\xe8\x03'*64
                else:
                    msg.width,msg.height,msg.step,msg.encoding = 8,8,24,'rgb8'
                    msg.data = bytes([sequence%256]*192)
            elif isinstance(msg,CameraInfo):
                msg.header.stamp,msg.header.frame_id=stamp,'synthetic_fixture'
                msg.width,msg.height=8,8;msg.k=[8.,0.,4.,0.,8.,4.,0.,0.,1.]
            elif isinstance(msg,JointGroupCommand):
                msg.name,msg.cmd = 'arm',[sequence*.001]*6
            else:
                msg.name,msg.cmd = 'gripper',.1
            pub.publish(msg)
    node.create_timer(.02,publish)
    process = None
    try:
        with (out/'recorder.log').open('w') as log:
            process = subprocess.Popen([sys.executable,str(ROOT/'scripts/hardware_record.py'),
                                        '--config',str(config_path),'--output',str(out/'session')],
                                       stdout=log,stderr=log,start_new_session=True)
            deadline = time.monotonic()+25
            subscribed_since = None
            while time.monotonic()<deadline:
                rclpy.spin_once(node,timeout_sec=.05)
                if process.poll() is not None:
                    raise RuntimeError(f'Recorder exited early: {process.returncode}; inspect {out}/recorder.log')
                if all(p.get_subscription_count()>0 for p in publishers.values()):
                    subscribed_since = subscribed_since or time.monotonic()
                    if time.monotonic()-subscribed_since >= 1.:
                        break
            else:
                raise RuntimeError('Recorder did not subscribe to every fixture topic')
            # Signal only wrapper; it forwards one SIGINT to rosbag for clean close.
            process.send_signal(signal.SIGINT)
            result = process.wait(timeout=15)
            assert result == 0, f'Recorder failed to finalize: {result}'
        manifest = json.loads((out/'session/manifest.json').read_text())
        assert manifest['complete'] and manifest['source'] == 'synthetic_ros_verification'
        assert manifest['training_ready'] is False and manifest['motion_commanded'] is False
        assert manifest['stop_reason'] == 'operator_interrupt' and manifest['demonstration_success'] is None
        assert manifest['shutdown_mode'] == 'sigint'
        counts = inspect_bag(out/'session/bag',expected)
        report = {'all_passed':True,'source':'synthetic_ros_verification','ros_domain_id':143,
                  'ros_localhost_only':True,'hardware_motion_commanded':False,
                  'robot_constructed':False,'topics':expected,'message_counts':counts,
                  'messages_deserialized':sum(counts.values()),'metadata_counts_match':True,
                  'depth_encoding_verified':'16UC1 millimetres','camera_info_verified':True,'manifest_complete':True,'stop_reason':manifest['stop_reason'],
                  'shutdown_mode':manifest['shutdown_mode'],'output':str(out)}
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            process.wait(timeout=5)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
