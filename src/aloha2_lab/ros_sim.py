"""ROS monitoring bridge for simulation only; does not subscribe to hardware commands."""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from rosgraph_msgs.msg import Clock
from builtin_interfaces.msg import Time
from .sim import AlohaSim, JOINTS


class SimNode(Node):
    def __init__(self):
        super().__init__("aloha2_mujoco")
        self.sim = AlohaSim()
        self.joints = self.create_publisher(JointState,"/aloha2_sim/joint_states",10)
        self.clock = self.create_publisher(Clock,"/clock",10)
        self.create_timer(self.sim.dt,self.tick)
        self.get_logger().info("ALOHA 2 MuJoCo running: simulation state only, no hardware publisher")

    def tick(self):
        q,v = self.sim.step(self.sim.data.ctrl.copy())
        ns = round(self.sim.data.time*1e9)
        stamp = Time(sec=ns//10**9,nanosec=ns%10**9)
        msg = JointState()
        msg.header.stamp = stamp
        msg.name = JOINTS
        msg.position = q.tolist()
        msg.velocity = v.tolist()
        self.joints.publish(msg)
        self.clock.publish(Clock(clock=stamp))


def main():
    rclpy.init()
    node = SimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.sim.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
