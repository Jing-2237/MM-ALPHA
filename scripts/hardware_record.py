#!/usr/bin/env python3
"""Record existing raw ROS streams. This script never starts or drives a robot."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from datetime import datetime, timezone
import yaml
from hardware_audit import read_topics


def storage_policy(config):
    storage = config.get('storage', {})
    if not isinstance(storage, dict):
        raise ValueError('storage must be a mapping')
    reserve = storage.get('min_free_gib', 5.)
    interval = storage.get('check_interval_s', 1.)
    for name, value, low, high in [('min_free_gib',reserve,1.,1024.),
                                    ('check_interval_s',interval,.1,2.)]:
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not low<=value<=high:
            raise ValueError(f'{name} must be finite and within {low}..{high}')
    return {'min_free_bytes':int(reserve*2**30),'check_interval_s':float(interval)}


def stop_owned_recorder(process):
    """Stop only our rosbag child, never the process group or robot drivers."""
    if process.poll() is not None:
        return process.wait(), 'already_exited'
    try:
        process.send_signal(signal.SIGINT)
    except ProcessLookupError:
        return process.wait(), 'already_exited'
    try:
        return process.wait(timeout=15), 'sigint'
    except subprocess.TimeoutExpired:
        # A wedged recorder must not continue filling the system volume.
        # Non-graceful fallback is reported; it does not claim a finalized bag.
        process.terminate()
        try:
            return process.wait(timeout=5), 'sigterm_fallback'
        except subprocess.TimeoutExpired:
            process.kill()
            return process.wait(timeout=5), 'sigkill_fallback'


def monitor_recorder(process, directory, policy, disk_usage=shutil.disk_usage):
    """Bounded space checks; injected process/disk readers enable offline tests."""
    free = None
    try:
        while process.poll() is None:
            free = int(disk_usage(directory).free)
            if free < policy['min_free_bytes']:
                code, mode = stop_owned_recorder(process)
                return {'recorder_exit_code':code,'stop_reason':'disk_reserve_reached',
                        'shutdown_mode':mode,'last_free_bytes':free}
            try:
                code = process.wait(timeout=policy['check_interval_s'])
                return {'recorder_exit_code':code,'stop_reason':'recorder_exited',
                        'shutdown_mode':'natural','last_free_bytes':free}
            except subprocess.TimeoutExpired:
                continue
        return {'recorder_exit_code':process.wait(),'stop_reason':'recorder_exited',
                'shutdown_mode':'natural','last_free_bytes':free}
    except KeyboardInterrupt:
        code, mode = stop_owned_recorder(process)
        return {'recorder_exit_code':code,'stop_reason':'operator_interrupt',
                'shutdown_mode':mode,'last_free_bytes':free}
    except OSError as exc:
        code, mode = stop_owned_recorder(process)
        return {'recorder_exit_code':code,'stop_reason':'disk_monitor_error',
                'shutdown_mode':mode,'last_free_bytes':free,'monitor_error':str(exc)}


def recording_source(config):
    source = config.get('source', 'real_hardware_raw')
    if source not in {'real_hardware_raw', 'synthetic_ros_verification'}:
        raise ValueError('Unknown recording source')
    configured = [t for category in ('required_topics', 'optional_topics')
                  for t in config.get(category, {})]
    if source == 'synthetic_ros_verification':
        if not configured or not all(t.startswith('/aloha2_test/') for t in configured):
            raise ValueError('Synthetic verification may only record /aloha2_test/ topics')
    elif any(t.startswith('/aloha2_test/') for t in configured):
        raise ValueError('Synthetic test topics must not be labelled real_hardware_raw')
    return source


def select_topics(config, available):
    recording_source(config)
    selected = []
    for category in ('required_topics', 'optional_topics'):
        for topic, expected_type in config.get(category, {}).items():
            actual = available.get(topic, [])
            if expected_type not in actual:
                if category == 'required_topics':
                    raise ValueError(f'Missing or wrong type: {topic}, expected {expected_type}, observed {actual}')
                continue
            selected.append(topic)
    if not config.get('required_topics'):
        raise ValueError('Define required streams explicitly in the local config')
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config.get('robot') != 'aloha_stationary':
        raise ValueError('This recorder is configured for stationary ALOHA 2')
    source = recording_source(config)
    policy = storage_policy(config)
    if not shutil.which('ros2'):
        raise RuntimeError('Source scripts/hardware_env.sh first')
    topics = select_topics(config, read_topics())
    if args.check_only:
        print(json.dumps({'topics': topics, 'source': source, 'motion_commanded': False}, indent=2))
        return
    # Reserve a parent directory exclusively so repeated commands cannot overwrite data.
    args.output.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(args.output).free < policy['min_free_bytes']:
        raise RuntimeError('Free space is below the configured reserve; choose a data volume with more space')
    manifest = {'schema': 'aloha2-lab/raw-rosbag-v1', 'complete': False,
                'source': source, 'motion_commanded': False,
                'started_utc': datetime.now(timezone.utc).isoformat(),
                'ros_domain_id': os.getenv('ROS_DOMAIN_ID', '0'), 'topics': topics,
                'config': config, 'action_units': 'raw controller messages; no normalization applied',
                'timing': 'bag receive timestamps plus original message headers; commands may have no header',
                'training_ready': False, 'demonstration_success': None,
                'storage_policy': policy, 'stop_reason': 'recording'}
    manifest_path = args.output/'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    # rosbag2 adapts subscriber QoS to discovered publishers. It also retains original
    # headers and message definitions. No bag playback is offered here.
    command = ['ros2', 'bag', 'record', '--storage', 'sqlite3', '--output', str(args.output/'bag'), *topics]
    process = subprocess.Popen(command)
    outcome = monitor_recorder(process,args.output,policy)
    manifest.update(outcome)
    manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
    manifest['complete'] = (outcome['recorder_exit_code'] == 0
                            and outcome['shutdown_mode'] in {'natural','sigint','already_exited'}
                            and (args.output/'bag/metadata.yaml').is_file())
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    if not manifest['complete']:
        sys.exit('Recorder did not finalize normally; preserve this directory for inspection')
    if outcome['stop_reason'] in {'disk_reserve_reached','disk_monitor_error'}:
        print(f"Recording stopped: {outcome['stop_reason']}. Finalized bag preserved; demonstration success is unknown.",file=sys.stderr)
        sys.exit(2)


if __name__ == '__main__':
    main()
