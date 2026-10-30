"""Feed rendered RGB-D to the real ROS perception adapter in a private DDS domain."""
import json,os,subprocess,sys,time,signal
from pathlib import Path
import numpy as np
import yaml
import rclpy
from sensor_msgs.msg import Image,CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data

root=Path(__file__).resolve().parents[1]
out=root/'runs/vision-ros-verification';out.mkdir(exist_ok=True)
f=np.load(root/'runs/vision-final/observation.npz',allow_pickle=False)
cfg=dict(calibration_validated=True,T_table_optical=f['T_world_optical'].tolist(),table_frame='test_table',optical_frame='test_camera_optical',rgb_topic='/vision_test/rgb',aligned_depth_topic='/vision_test/depth',camera_info_topic='/vision_test/info')
config=out/'synthetic.yaml';config.write_text(yaml.safe_dump(cfg))
os.environ.update(ROS_DOMAIN_ID='144',ROS_LOCALHOST_ONLY='1')
rclpy.init();node=rclpy.create_node('vision_test_source');bridge=CvBridge();received=[]
node.create_subscription(PoseStamped,'/aloha2_vision/cup_pose',lambda msg:received.append(msg),10)
pubs=[node.create_publisher(t,cfg[k],qos_profile_sensor_data) for t,k in [(Image,'rgb_topic'),(Image,'aligned_depth_topic'),(CameraInfo,'camera_info_topic')]]
with (out/'node.log').open('w') as log:
 p=subprocess.Popen([sys.executable,'-m','aloha2_lab.ros_vision','--config',str(config)],stdout=log,stderr=log)
 try:
  deadline=time.monotonic()+20
  while not received and time.monotonic()<deadline:
   stamp=node.get_clock().now().to_msg()
   rgb=bridge.cv2_to_imgmsg(f['rgb'],'rgb8');depth=bridge.cv2_to_imgmsg(f['depth'].astype(np.float32),'32FC1');info=CameraInfo();info.k=f['K'].ravel().tolist();info.height=240;info.width=320
   for msg,pub in zip([rgb,depth,info],pubs):
    msg.header.stamp=stamp;msg.header.frame_id=cfg['optical_frame'];pub.publish(msg)
   rclpy.spin_once(node,timeout_sec=.1)
   if p.poll() is not None:raise RuntimeError('Perception node exited; see node.log')
  assert received,'No valid detected pose'
  pose=received[-1].pose.position
  assert np.linalg.norm(np.array([pose.x,pose.y])-[-.18,.07])<.006
  report=dict(valid_pose_received=True,position=[pose.x,pose.y,pose.z],source='synthetic_rendered_rgbd',hardware_motion_commanded=False)
  (out/'report.json').write_text(json.dumps(report,indent=2));print(report)
 finally:
  if p.poll() is None:
   p.send_signal(signal.SIGINT);p.wait(timeout=10)
  node.destroy_node();rclpy.shutdown()
