> 主相机RGB-D训练录制现要求 aligned depth 与 CameraInfo。启动深度与像素对齐的配置见 [纸杯RGB-D说明](paper-cup-rgbd.md#真机相机采集)。

# ALOHA 2 真机接入与原始示范采集

本项目按 **Stationary ALOHA 2：两台 ViperX follower、两台 WidowX leader、四路相机**准备。这里的驱动源码来自 Interbotix ALOHA `2.0` 分支及 Humble/ROS 2 配套仓库，当前提交固定在 `config/hardware-sources.lock.json`。这些软件已准备不等于真机接线、标定或动作验收已完成。

## 工作区与依赖

本机 ROS Humble 桌面、控制和 RealSense 软件包已安装，包含本项目仿真 launch 的 Stationary 工作区共 11 个包已构建成功。源码构建成功证明软件包可编译，不证明真机已经接通。

除 ROS 基础环境外，当前驱动工作区使用这些系统依赖；`bootstrap.sh system` 已包含它们：

```bash
sudo apt-get install --no-remove -y \
  ros-humble-dynamixel-sdk ros-humble-nav2-msgs ros-humble-tf-transformations \
  ros-humble-usb-cam ros-humble-joy ros-humble-teleop-twist-joy \
  ros-humble-kdl-parser ros-humble-orocos-kdl-vendor \
  ros-humble-joint-state-publisher-gui libyaml-cpp-dev python3-transforms3d \
  python3-opencv python3-matplotlib python3-numpy
```

真机脚本采用独立的 `.venv-hardware`，保留 Ubuntu 提供的 NumPy、OpenCV 和 ROS ABI 兼容性。重建命令：

```bash
/usr/bin/python3 -m venv --system-site-packages .venv-hardware
.venv-hardware/bin/python -m pip install \
  modern_robotics==1.1.1 dm-env==1.6 pyquaternion==0.9.9 tqdm==4.67.1 h5py==3.14.0
```

这部分也可单独运行 `bash scripts/bootstrap.sh hardware-python`。模拟和训练环境 `.venv` 仍独立保留；不要把上游宽泛的 `requirements.txt` 全部装入真机环境，其中还包括另一套 OpenCV、Torch 等依赖。

在项目根目录执行：

```bash
python3 scripts/setup_ros_workspace.py
source /opt/ros/humble/setup.bash
cd ros2_ws
colcon build --symlink-install --executor sequential --packages-up-to aloha aloha2_lab_bringup
cd ..
source scripts/hardware_env.sh
```

链接脚本只验证本地仓库版本并创建链接，不下载源码、不重置已有改动、不连接机器人。应在未激活学习虚拟环境的新终端中使用系统 Python 构建。安装脚本负责下载 ROS 和依赖；缺包时检查构建输出，不要跳过编译失败。

本工作区刻意不包含移动底盘 `interbotix_ros_slate`：上游 `aloha/package.xml` 声明了它，但 `aloha_stationary.yaml` 的 `base: false` 分支不会加载底盘。不要在本工作区使用 `aloha_mobile`。如使用 rosdep 做维护，应明确排除这个移动底盘键，并为 `python-transforms3d-pip` 使用已安装的 `python3-transforms3d`。

## 第一次只读检查

```bash
source scripts/hardware_env.sh
python scripts/hardware_audit.py --output runs/hardware-inventory.json
python scripts/verify_hardware_imports.py --output runs/hardware-imports.json
```

`verify_hardware_imports.py` 仅导入固定版本的官方 teleop、record、replay、sleep 模块，禁止 ROS 初始化和机器人构造，并做一次 `cv_bridge` RGB 转换往返检查。通过它表示 Python 运行时依赖完整，不表示设备已接通。

该程序只读取 ROS 包索引、设备路径和 USB 列表。`--topics` 可读取已经启动的 ROS 图；它不会创建机器人对象、启用扭矩、调用服务或发送动作。输出中的 `torque_enabled: false` 指本程序没有启用扭矩，不能证明外部控制器当前已断电。接线后应分别确认四个串口别名：

```text
/dev/ttyDXL_leader_left
/dev/ttyDXL_leader_right
/dev/ttyDXL_follower_left
/dev/ttyDXL_follower_right
```

上游 udev 规则位于 `third_party/interbotix_aloha/resource/99-fixed-interbotix-udev.rules`，含设备对应关系示例。必须先根据本机 USB 序列号逐一映射；本搭建没有盲目复制规则、修改串口权限或改变电机参数。现场完成后再安装规则并复插设备。检查操作者是否具备串口组权限；新增用户组后需要重新登录。

复制记录配置后填写实际相机、关节和标定信息：

```bash
cp config/hardware.example.yaml config/hardware.local.yaml
```

这份 YAML 只供本项目审计/原始记录使用，**不会覆盖上游启动配置**。真机启动时还要编辑 `third_party/interbotix_aloha/config/robot/aloha_stationary.yaml` 中四路相机的 `serial_no`、分辨率/频率和实际安装变换。官方 launch 只通过 `robot:=aloha_stationary` 选择包内配置，不接受本项目 `--config` 参数。每台 D405 的序列号需要分别核对，四台留空会导致设备选择不确定。相机外参必须实测；模型自带的变换不能作为手眼标定结果。

## 现场启动、遥操作与关闭

以下命令是供现场操作者使用的入口，搭建过程没有执行。先确认机器人支撑、供电、急停、关节/夹爪方向、活动空间和上游安全停机流程，再启动硬件。`ros2 launch` 会启动驱动，不能把它当作只读检查。

```bash
source scripts/hardware_env.sh
# 与现场控制电脑保持相同 DDS 域。单机 USB 连接可保持默认 localhost 配置。
ros2 launch aloha aloha_bringup.launch.py robot:=aloha_stationary
```

遥操作入口位于 `third_party/interbotix_aloha/scripts/teleop.py`。先使用真机环境查看该固定版本参数，再按官方现场流程执行：

```bash
source scripts/hardware_env.sh
python third_party/interbotix_aloha/scripts/teleop.py --help
```

`--help` 仅显示参数，不启动遥操作。不要直接运行 `record_episodes.py` 来“检测连接”：它会重启夹爪电机、设置工作模式、启用扭矩并运动到起始位姿。

正常停机时保持 bringup 进程运行，在另一终端按官方流程先送回 sleep 位姿；确认稳定后再终止 bringup：

```bash
source scripts/hardware_env.sh
python third_party/interbotix_aloha/scripts/sleep.py -r aloha_stationary --all
```

`sleep.py` 本身会运动并设置扭矩，必须由现场人员执行。异常状态应遵循机器人的急停/支撑处置流程；不要将直接杀进程视作可靠的机械停机。官方文档指出，机械臂未到 sleep 位姿时结束 bringup 可能导致机械臂下坠。

## 记录已有遥操作流

本项目的记录器只订阅 **已经在运行** 的驱动和遥操作话题，不启动遥操作、不产生指令。先验证话题名称和类型：

```bash
source scripts/hardware_env.sh
python scripts/hardware_audit.py --topics
python scripts/hardware_record.py --config config/hardware.local.yaml \
  --output data/real/session-001 --check-only
```

相机 topic 以本机实际 ROS 图为准。模板默认使用官方配置中的 `/<camera_name>/camera/color/image_rect_raw`。模板要求双臂 follower/leader 状态、follower 关节组/夹爪指令以及四路 RGB；缺少任一必需流会拒绝开始。确有不同相机布局时，要在本地 YAML 明确调整，不要用空白图像填充。

正式采集并按 Ctrl+C 结束：

```bash
python scripts/hardware_record.py --config config/hardware.local.yaml \
  --output data/real/session-001
ros2 bag info data/real/session-001/bag
```

每次使用新目录，记录器不会覆盖旧采集。生成 `manifest.json` 和 sqlite3 rosbag。只有录制进程正常结束且 `metadata.yaml` 存在时，manifest 的 `complete` 才为 true；该字段只表示文件正常收尾，不表示示范成功或时序质量合格，`demonstration_success` 始终保留为 null。

记录器默认要求并持续保留 5GiB 空闲，每秒检查一次目标目录所在分区。低于保留量时，只向自己启动的 rosbag 子进程发送 SIGINT，使其写出元数据并关闭文件；manifest 保存 `stop_reason: disk_reserve_reached`，进程返回码为 2，已正常收尾的 bag 会保留。停止录制不会停止遥操作或机械臂，操作者仍按现场流程处理机器人。如果 rosbag 15 秒未退出，则仅对该子进程使用 TERM、再等待 5 秒仍不退出才使用 KILL，并记录非正常关闭方式；不会把强制关闭当作完整数据。

本地 YAML 可以设置：

```yaml
storage:
  min_free_gib: 5.0       # 允许 1–1024 GiB，不能关闭保留空间保护
  check_interval_s: 1.0  # 允许 0.1–2 秒，防止检查间隔过长
```

四路原始 RGB 的体积较大，应按任务分段采集、核验后将需要长期保留的批次迁移到数据盘。记录器不自动删除历史数据；保留量检查也不是磁盘配额，其他程序同时写盘时仍需观察系统空间。

无需接机器人也可检查原始录制链路：

```bash
source scripts/hardware_env.sh
python scripts/verify_recording.py
```

这个验证脚本固定使用 localhost 和 ROS 域 143，只向 `/aloha2_test/run_...` 发布带合成标签的 14 路测试主题（包含主相机深度与CameraInfo），不使用真实硬件命名空间，也不构造机器人。它启动本项目原始录制器，停止后检查 manifest、rosbag 元数据计数，并反序列化全部消息验证类型和内容。结果写入 `runs/recording-verification/<时间>/`，数据来源明确标为 `synthetic_ros_verification`，不能算作真机或人工示范。

记录配置的 `source` 默认为 `real_hardware_raw`。`synthetic_ros_verification` 只允许 `/aloha2_test/` 下的主题，反向也禁止将该测试前缀标为真实数据；该标签用于来源追踪，不代替操作者确认实际设备来源。

需要人工填写或补充的实验元数据包括：任务描述、物体/场景、操作者、是否成功、失败原因、相机序列号、标定版本、控制模式、动作限幅和时间同步方案。manifest 可能包含设备序列号，只保存在本机数据目录。

## 真机数据到模仿学习的边界

原始 rosbag **还不能直接传给当前仿真训练命令**。本地仿真 schema 明确要求左臂在前、右臂在后，夹爪是单指位移（米）。上游 ALOHA 2 配置列出的 follower 顺序是右臂、左臂；`real_env.py` 按字典插入顺序拼接数据，不能假设它输出左、右顺序。真机夹爪电机角度、驱动推算手指位置、上游归一化值以及 MuJoCo 指位移也不是同一个量。

接入训练前必须实现并验证离线转换：

1. 用 `JointState.name` 和控制器关节组映射显式重排为左 6 关节＋左夹爪、右 6 关节＋右夹爪。
2. 标定夹爪闭合/张开及电机角度到实际指位移的映射，保留原始值、单位和标定版本。
3. 在同一时基上对齐双臂状态、四路图像与动作。保留 message header 和 bag 接收时间；无 header 的命令消息只能得到接收时间，不可宣称精确执行时刻。
4. 验证 observation 在 action 前，剔除丢帧、时间倒退、过期画面和不完整 episode，并按 episode 划分训练/验证集。
5. 单独验证真机动作限速、超时停止和现场干预；当前 `aloha-lab evaluate` 仅在 MuJoCo 中执行，不向真机发布策略动作。

当前已提供的真机入口是驱动工作区、只读审计和原始采集。真机标定、原始数据转换、视觉策略训练及真机策略执行属于接通设备后的下一阶段，不能以仿真测试替代验收。

来源：[官方 ALOHA 2 Bringup & Shutdown](https://docs.trossenrobotics.com/aloha_docs/2.0/operation/bringup_shutdown.html)、[Interbotix ALOHA 2.0 源码](https://github.com/Interbotix/aloha/tree/2.0)。
