# 杯子视觉定位与抓取（IK教师）

**已新增咖啡杯视觉模仿学习实验，见 [训练与独立测试报告](coffee-visual-il.md)。本文说明旧的RGB-D定位和IK教师入口，不代表新学习策略的推理实现。**

现有模块已能在 MuJoCo 中通过 RGB-D 图像定位杯子，再执行接近、闭爪、抬升。控制不读取杯子仿真 body 位姿或分割 ID；物体真实高度和双指接触只用于执行结束后的独立验收。相机外参来自标定/仿真相机，机器人运动学来自关节状态。

## 运行

```bash
source scripts/env.sh
MUJOCO_GL=glfw aloha-lab cup-grasp --output "$ALOHA_LAB_ROOT/runs/cup-demo-1" --cup -.18 .07 --viewer
```

`--cup X Y` 只设置测试场景中的杯子初始位置；抓取目标由 RGB-D 定位重新计算。输出目录需为新目录。无窗口执行使用 `MUJOCO_GL=egl` 并省略 `--viewer`。输出包括检测框图、原始 RGB-D/相机标定、抬升图和报告；失败时不报告成功。

要保存自动控制器的带图像示范，添加 `--record-demo`。生成的 `episode_000000.hdf5` 包含每步观测前的关节状态、RGB 和动作，来源明确为 `vision_ik_teacher_not_human`，`task_success` 记录任务结果。只有确认成功的示范才应进入任务训练集。

人工示范可在有杯子的场景录制：

```bash
MUJOCO_GL=glfw aloha-lab view --cup -.18 .07 \
  --output "$ALOHA_LAB_ROOT/data/my_first_demo/episode_000000.hdf5" \
  --cameras overhead_cam wrist_cam_left wrist_cam_right
```

## 当前适用范围

当前检测器识别**已登记的蓝色圆柱杯**：约直径48mm、高80mm、不透明，放在标定的 z=0 桌面，左臂工作区内，单杯静止场景。先由 RGB 外观找候选，再用米制深度反投影，拟合杯壁圆截面得到杯心。需要有效深度和相机内外参；无杯、过期帧、无效深度或形状不匹配会拒绝定位。

这是一条可执行的任务专用视觉基线，不是任意杯子的语义识别器。透明杯、不同尺寸/外观、多个目标、移动目标或杂乱遮挡场景不在已验证范围。杯子几何来自登记信息，抓取高度使用登记高度；不能把该高度或几何直接套到其他实物。IK 执行器为无额外障碍物仿真夹具服务，未提供通用避障规划。

当前自动抓取使用视觉定位加 IK 控制器。原来的学习器仍是状态行为克隆，新增相机数据可用于后续训练视觉策略；本次没有把 IK 成功说成已训练出视觉抓取模型。

## 真机相机输入

ROS 节点仅做视觉定位和发布，不发布电机命令：

```bash
source scripts/hardware_env.sh
export PYTHONPATH="$ALOHA_LAB_ROOT/src:$PYTHONPATH"
cp config/vision.example.yaml config/vision.local.yaml
# 在本机填写真实 aligned RGB/depth/CameraInfo 话题、光学帧和实测光学帧到桌面坐标变换。
# 只有实测并验收后才能设置 calibration_validated: true。
python -m aloha2_lab.ros_vision --config config/vision.local.yaml
```

输出 `/aloha2_vision/cup_pose`、`/aloha2_vision/annotated`、`/aloha2_vision/status`。消费者必须检查 status 和 pose 时间戳，不能继续使用检测失败前的旧目标。输入是对齐到 RGB 的深度（16UC1 毫米或32FC1米）、RGB 相机内参，30ms 内同步；默认只接受200ms内帧。`T_table_optical` 必须是实际光学帧到桌面坐标系的刚体变换，不能用 camera_link 外参替代。桌面坐标定义、杯子登记尺寸与工作区限制必须现场确认。

## 实际证据

完整测试35项通过。`runs/vision-final/verification.json` 汇总验证结果。`runs/vision-recorded/episode_000000.hdf5` 已实际录下820帧240×320 RGB、关节状态和动作，完成校验且 task_success=true；该数据来源为 IK 教师，不是人工示范。


- `runs/vision-final/`：视觉定位后真实物理夹持，杯心最终约0.182m，两个实体手指都与杯子保持接触，抬升后继续保持2秒。
- `runs/vision-varied-0..2/`：三组位置的定位与抬升记录（早期报告采用高度指标，最终报告另验双指接触）。
- `runs/vision-ros-verification/report.json`：合成相机输入经实际 ROS 视觉节点发布三维位置；未接真实相机。
- `tests/test_vision_grasp.py`：定位精度、无杯/旧帧/无效深度拒绝和实体双指抓取测试。

这些是仿真与合成 ROS 输入证据；真机视觉和抓取仍需实物标定验收。MuJoCo 渲染/相机接口参见官方文档：https://mujoco.readthedocs.io/en/stable/python.html 。

## 坐标、保存和回放速查

视觉定位器的严格边界为 `-0.30 < x < 0.10`、`-0.20 < y < 0.25` 米；这是检测允许的工作区，不是整个矩形都已验证可抓取。实际还取决于 IK 可达性、相机可见性、手臂/桌面净空。已测试位置包括 `(-.19,.02)`、`(-.14,.08)`、`(-.18,.10)`、`(-.18,.07)`。当前是开始时观测一次的静态单杯任务，不跟踪执行中移动的杯子。

所有文件写到命令的 `--output`：`view` 指向具体 HDF5 文件，`cup-grasp` 指向独立目录。`cup-grasp --record-demo` 才会在该目录增加 `episode_000000.hdf5`；不加时只有检测/报告等运行结果。重复采集需换目录/文件名，已有文件不会覆盖。

查看已保存的相机画面，无需训练：

```bash
aloha-lab replay \
  --episode "$ALOHA_LAB_ROOT/runs/cup-demo-1/episode_000000.hdf5" \
  --camera overhead_cam \
  --output "$ALOHA_LAB_ROOT/runs/cup-demo-1/replay.mp4"
```

这是保存图像的 MP4 回放，不会重新执行控制命令。已录820帧示范的导出视频经 ffprobe 检查为320×240、50fps、820帧。`evaluate --checkpoint ...` 则是让模型重新预测动作的仿真运行，不是示范文件回放。

当前视觉定位加 IK 抓杯无需训练；如需模仿学习，则需多位置成功示范及使用图像/目标位置的策略输入。原有状态学习器不读取图像，不能由其训练完成推断已学会不同位置抓杯。
