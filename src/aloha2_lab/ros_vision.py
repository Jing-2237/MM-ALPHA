"""Read-only calibrated RGB-D perception node; no motor command interfaces."""
import argparse,json
from pathlib import Path
import numpy as np
import yaml
from .vision import RGBDFrame,locate_cup


def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image,CameraInfo
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import String
    from cv_bridge import CvBridge
    from message_filters import Subscriber,ApproximateTimeSynchronizer
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    args=parser.parse_args();cfg=yaml.safe_load(Path(args.config).read_text())
    if cfg.get('calibration_validated') is not True:raise ValueError('Measured optical-to-table calibration must be validated first')
    T=np.array(cfg['T_table_optical'],float)
    if T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1]) or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(T[:3,:3]),1):raise ValueError('Invalid rigid optical-to-table transform')
    rclpy.init()
    class VisionNode(Node):
        def __init__(self):
            super().__init__('aloha2_cup_vision');self.bridge=CvBridge()
            self.pose=self.create_publisher(PoseStamped,'/aloha2_vision/cup_pose',10)
            self.status=self.create_publisher(String,'/aloha2_vision/status',10)
            self.annotated=self.create_publisher(Image,'/aloha2_vision/annotated',10)
            self.subs=[Subscriber(self,typ,cfg[key],qos_profile=qos_profile_sensor_data) for typ,key in [(Image,'rgb_topic'),(Image,'aligned_depth_topic'),(CameraInfo,'camera_info_topic')]]
            self.sync=ApproximateTimeSynchronizer(self.subs,queue_size=10,slop=.03)
            self.sync.registerCallback(self.frame)
        def frame(self,rgb,depth,info):
            try:
                if rgb.header.frame_id!=cfg['optical_frame'] or info.header.frame_id!=cfg['optical_frame'] or depth.header.frame_id!=cfg['optical_frame']:raise ValueError('Frame mismatch: aligned depth and RGB optical frame required')
                if depth.encoding not in ('16UC1','32FC1'):raise ValueError('Unsupported depth units')
                d=self.bridge.imgmsg_to_cv2(depth,desired_encoding='passthrough').astype(float)
                if depth.encoding=='16UC1':d*=.001
                stamp=rgb.header.stamp.sec+rgb.header.stamp.nanosec*1e-9
                f=RGBDFrame(self.bridge.imgmsg_to_cv2(rgb,'rgb8'),d,np.array(info.k).reshape(3,3),T,stamp)
                found=locate_cup(f,self.get_clock().now().nanoseconds*1e-9)
                msg=PoseStamped();msg.header.stamp=rgb.header.stamp;msg.header.frame_id=cfg['table_frame']
                msg.pose.position.x,msg.pose.position.y,msg.pose.position.z=found['position_world'];msg.pose.orientation.w=1.
                self.pose.publish(msg)
                import cv2
                image=f.rgb.copy();x1,y1,x2,y2=found['bbox_xyxy'];cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
                annotated=self.bridge.cv2_to_imgmsg(image,'rgb8');annotated.header=rgb.header;self.annotated.publish(annotated)
                self.status.publish(String(data=json.dumps(dict(valid=True,**found))))
            except (ValueError,TypeError) as exc:self.status.publish(String(data=json.dumps(dict(valid=False,reason=str(exc)))))
    node=VisionNode()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()

if __name__=='__main__':main()
