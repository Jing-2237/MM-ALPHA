# 本机初步搭建验证报告

日期：2026-10-29。设备：Ubuntu 22.04.5、Python 3.10.12、RTX 5060 Laptop GPU 8GB。项目按用户选择保存在 Linux 分区，位置为 `/home/wj/Documents/Codex/2026-10-29/wo/outputs/aloha2-lab`。清理本轮已完成的下载缓存后，分区可用空间约 42GiB。

本报告证明开发环境和初始工程流程可用；不证明真实机器人已调试完成，也不证明已学会任何抓取任务。

| 要求 | 本机证据 | 结果 |
|---|---|---|
| ROS 2 开发与硬件驱动环境 | `runs/verification/ros-build.log`，`runs/hardware-audit.json` | Humble 桌面版、RViz、ros2_control、RealSense ROS 和 ALOHA Stationary 依赖安装；工作区 11 包构建成功 |
| ALOHA 2 双臂物理仿真 | `runs/environment-verification/environment.json` | MuJoCo 3.3.7，14 维控制，200 步状态有限 |
| 四路相机 | `runs/environment-verification/` 的四张相机 PNG；`data/camera-smoke/episode_000000.hdf5` | 四路实际渲染，并保存 5 帧同步 HDF5 数据 |
| 本机可视化窗口 | `runs/verification/native-viewer.json` | GLFW 窗口启动，100 次同步步进后正常关闭 |
| ROS 通信 | `runs/ros-verification/report.json` | 独立域 142 实际收到 10 条关节状态、10 条时钟，时间递增 |
| 真机软件准备 | `runs/hardware-imports.json` | 官方遥操作、采集、回放、停机脚本导入以及 cv_bridge 往返转换通过；禁止实例化机器人进行验证 |
| 原始录制工具 | `runs/recording-verification/20261029T174241.954719Z/report.json` | 域 143、仅测试前缀的 12 路话题，共 612 条消息，录制后全部反序列化成功；明确标为合成测试来源 |
| 采集存储保护 | `tests/test_recording_config.py` | 默认保留 5GiB，每秒检查；空间不足只停止本工具的录制子进程，保留收尾数据和原因 |
| GPU 训练环境 | `runs/environment-verification/environment.json` | PyTorch 2.7.1+cu128，CUDA 12.8，设备能力 12.0，GPU 前向与反向计算通过 |
| 模仿学习训练闭环 | `runs/initial-bc/metrics.json`、`policy.pt` | 8 段 × 180 帧测试轨迹，6 段训练/2 段验证，GPU 训练 80 轮，按 episode 划分，无验证集归一化泄漏 |
| 策略载入与仿真回放 | `runs/initial-eval/report.json`、`rollout.hdf5`、`final.png` | checkpoint 成功加载，300 步回放完成且数值有限；没有发送真机动作 |
| 自动检查 | `runs/verification/tests-final.log` | 30 项通过，无跳过项 |
| 参考仓库问题核查 | `docs/reference-audit.md`、`docs/reference-bridge-probe.json` | 核实单臂假设与训练链路缺失，并离线复现夹爪旧反馈、超时取消及 effort 限制缺陷；新流程不使用该桥接器 |

## 模型质量边界

当前模型是输入关节位置/速度的动作分块行为克隆基线。虽然已记录图像，训练器尚不使用图像，也不是视觉 ACT。训练样本是明确标记的自动测试轨迹，目的是验证存储、划分、优化、checkpoint 和回放流程。

本次最佳归一化验证 MSE 为 0.3530。300 步回放中 297 步触发控制范围或变化速率裁剪，说明这个测试模型不能作为可部署策略。`task_success_evaluated=false`，没有物体任务成功率。不可把数值有限、控制限幅或无接触当作碰撞安全/抓取成功的证明。

## 真机状态

最新主机审计为 `runs/hardware-audit.json`：官方软件包可发现，USB 列表包含笔记本摄像头，未发现 ALOHA Dynamixel 串口或 RealSense D405。本轮未使能扭矩、未启动硬件驱动、未执行真机动作。

接入后按 `docs/hardware.md` 确认机器人型号、串口和相机序列号、关节顺序、夹爪单位、时间对齐及标定。官方真机数据默认右/左顺序与本项目左/右顺序不同，原始 rosbag 的 `training_ready=false`；完成校准转换前不能直接当成当前仿真 HDF5。正式任务还需要人工示范和独立成功/失败评估。

旧的安装前或隔离环境审计已放入 `runs/archive/`，不代表本机当前状态。

## 可复现信息

- `config/sources.lock.json`：8 个第三方源码提交。
- `config/hardware-sources.lock.json`：Stationary ALOHA 包映射及提交。
- `config/python-learning.lock.txt`、`config/python-hardware.lock.txt`：已安装的环境版本。
- `config/torch-wheels.lock.json`、`runs/verification/torch-wheel-checksums.json`：公开来源和实际核验的 CUDA/PyTorch wheel SHA256。
- `scripts/bootstrap.sh`：系统、源码、两个 Python 环境及完整工作区的重建入口；本轮执行过对应安装/构建步骤，脚本语法通过，未另行重装一台干净系统测试。

系统库继续使用签名保护的 Ubuntu/ROS 仓库，因此不是字节级固定的系统镜像。使用方法以项目 README 为入口。
