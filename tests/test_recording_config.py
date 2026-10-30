from pathlib import Path
import sys
import signal
import subprocess
from types import SimpleNamespace
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
from hardware_record import recording_source, select_topics, storage_policy, monitor_recorder


def test_synthetic_topics_require_explicit_source():
    config = {'required_topics':{'/aloha2_test/state':'sensor_msgs/msg/JointState'}}
    with pytest.raises(ValueError,match='must not be labelled'):
        recording_source(config)
    config['source'] = 'synthetic_ros_verification'
    assert recording_source(config) == 'synthetic_ros_verification'


def test_synthetic_source_rejects_hardware_names():
    config = {'source':'synthetic_ros_verification',
              'required_topics':{'/follower_left/joint_states':'sensor_msgs/msg/JointState'}}
    with pytest.raises(ValueError,match='only record'):
        recording_source(config)


def test_required_topic_types_are_checked():
    config = {'source':'synthetic_ros_verification',
              'required_topics':{'/aloha2_test/state':'sensor_msgs/msg/JointState'},
              'optional_topics':{'/aloha2_test/image':'sensor_msgs/msg/Image'}}
    with pytest.raises(ValueError,match='wrong type'):
        select_topics(config,{'/aloha2_test/state':['std_msgs/msg/String']})
    assert select_topics(config,{'/aloha2_test/state':['sensor_msgs/msg/JointState']}) == ['/aloha2_test/state']


class FakeRecorder:
    def __init__(self,outcomes):
        self.outcomes = iter(outcomes)
        self.returncode = None
        self.signals = []
        self.timeouts = []

    def poll(self):
        return self.returncode

    def wait(self,timeout=None):
        self.timeouts.append(timeout)
        if self.returncode is not None:
            return self.returncode
        outcome = next(self.outcomes)
        if isinstance(outcome,BaseException):
            raise outcome
        self.returncode = outcome
        return outcome

    def send_signal(self,signum):
        self.signals.append(signum)

    def terminate(self):
        self.signals.append(signal.SIGTERM)

    def kill(self):
        self.signals.append(signal.SIGKILL)


def test_disk_monitor_stops_only_owned_recorder_when_reserve_crossed(tmp_path):
    process = FakeRecorder([subprocess.TimeoutExpired('rosbag',.5),0])
    free = iter([6*2**30,4*2**30])
    result = monitor_recorder(process,tmp_path,storage_policy({'storage':{'check_interval_s':.5}}),
                              disk_usage=lambda path:SimpleNamespace(free=next(free)))
    assert result['stop_reason'] == 'disk_reserve_reached'
    assert result['last_free_bytes'] == 4*2**30
    assert result['shutdown_mode'] == 'sigint' and result['recorder_exit_code'] == 0
    assert process.signals == [signal.SIGINT]
    assert process.timeouts == [.5,15]


def test_disk_monitor_natural_exit_does_not_signal_process(tmp_path):
    process = FakeRecorder([0])
    result = monitor_recorder(process,tmp_path,storage_policy({}),
                              disk_usage=lambda path:SimpleNamespace(free=8*2**30))
    assert result['stop_reason'] == 'recorder_exited' and result['recorder_exit_code'] == 0
    assert process.signals == []


def test_operator_interrupt_finalizes_recorder(tmp_path):
    process = FakeRecorder([KeyboardInterrupt(),0])
    result = monitor_recorder(process,tmp_path,storage_policy({}),
                              disk_usage=lambda path:SimpleNamespace(free=8*2**30))
    assert result['stop_reason'] == 'operator_interrupt'
    assert process.signals == [signal.SIGINT]


def test_unresponsive_recorder_has_bounded_child_only_shutdown(tmp_path):
    process = FakeRecorder([subprocess.TimeoutExpired('rosbag',15),
                            subprocess.TimeoutExpired('rosbag',5),-9])
    result = monitor_recorder(process,tmp_path,storage_policy({}),
                              disk_usage=lambda path:SimpleNamespace(free=4*2**30))
    assert result['shutdown_mode'] == 'sigkill_fallback' and result['recorder_exit_code'] == -9
    assert process.signals == [signal.SIGINT,signal.SIGTERM,signal.SIGKILL]
    assert process.timeouts == [15,5,5]


@pytest.mark.parametrize('storage',[
    {'min_free_gib':0},{'min_free_gib':float('nan')},{'min_free_gib':True},
    {'min_free_gib':'5'},{'check_interval_s':60},{'check_interval_s':0},None,
])
def test_invalid_storage_policy_is_rejected(storage):
    with pytest.raises(ValueError):
        storage_policy({'storage':storage})
