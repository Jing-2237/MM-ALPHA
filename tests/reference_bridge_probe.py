"""Offline probe of the reference bridge's execute body; never imports ROS.

The fake transport resolves synchronously. This checks local return/cancel
logic only, not callback scheduling, ROS transport, or physical behavior.
"""
import ast
import json
from pathlib import Path
from types import SimpleNamespace as NS
import threading

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/"third_party/RobotAgent/ros2_ws/src/arirobot_ros/scripts/arirobot_aloha_gripper_bridge.py"


class Future:
    def __init__(self,value=None,done=True):
        self.value,self.finished = value,done
    def done(self):
        return self.finished
    def result(self):
        return self.value
    def add_done_callback(self,callback):
        if self.finished:
            callback(self)


class Event:
    def set(self):
        pass
    def wait(self,timeout):
        return True


def run():
    tree = ast.parse(SOURCE.read_text())
    cls = next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name == "AlohaGripperBridge")
    execute = next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name == "execute")
    namespace = dict(
        GripperCommand=NS(Result=lambda:NS()),
        JointTrajectoryPoint=lambda:NS(time_from_start=NS()),
        FollowJointTrajectory=NS(Goal=lambda:NS(trajectory=NS())),
        GoalStatus=NS(STATUS_SUCCEEDED=4),
        gap_to_finger=lambda gap:(gap+.027)/2,
        threading=NS(Event=Event),
    )
    exec(compile(ast.Module(body=[execute],type_ignores=[]),str(SOURCE),"exec"),namespace)
    reports = {}
    for mode in ("stale_feedback","result_timeout","late_acceptance"):
        called = []
        child = NS(accepted=True,cancel_goal_async=lambda:called.append("cancel_child"))
        child.get_result_async = lambda:Future(NS(status=4),done=mode != "result_timeout")
        acceptance = Future(child,done=mode != "late_acceptance")
        handle = NS(request=NS(command=NS(position=.04,max_effort=.01)),is_cancel_requested=False,
                    abort=lambda:called.append("abort"),succeed=lambda:called.append("succeed"),
                    canceled=lambda:called.append("canceled"))
        bridge = NS(joint="left_finger",lock=threading.Lock(),state=(.04,.5,NS(sec=0,nanosec=0)),
                    trajectory=NS(wait_for_server=lambda **kw:True,send_goal_async=lambda goal:acceptance))
        result = namespace["execute"](bridge,handle)
        reports[mode] = dict(calls=called,returned_position=getattr(result,"position",None),
                             observation_stamp_s=0,requested_max_effort=.01,
                             returned_effort=getattr(result,"effort",None))
    return reports


if __name__ == "__main__":
    print(json.dumps(run(),indent=2))
