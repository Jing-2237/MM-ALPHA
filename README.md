# MM-ALPHA：ALOHA 2 视觉抓杯与模仿学习

ALOHA 2 双臂机器人的 MuJoCo 仿真、RGB-D 示范采集、视觉模仿学习及 ROS 2 Humble 接入工具。参考 [RobotAgent](https://github.com/Jing-2237/RobotAgent)，机器人模型采用 [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/aloha)，真机接口参考 [Interbotix ALOHA 2.0](https://github.com/Interbotix/aloha/tree/2.0)。上游源码按提交锁定，由脚本下载。

**当前推荐模型：`data/paper-rgbd-v4/model/`。** 已包含实际训练权重、该批次示范、独立回放记录和测试结果。对象为下直径5cm、上直径10cm、高12cm的麦当劳纸杯。策略先张开夹爪，再从侧面以35°倾角夹住杯身下部，抬升16cm并保持2秒。

| 最终 v4 实验 | 结果 |
|---|---|
| 合格示范 | 17段训练、6段验证；保存后独立回放23/23通过 |
| 训练 | CUDA，1200轮 |
| 预先划定的10个未见位置 | 严格抓取判据9/10成功，1次未夹稳 |
| RGB置黑 / 错误位置深度 | 0/10、1/10成功 |
| 缺失有效深度 | 拒绝推理 |
| 正常测试杯壁最大接触重叠 | 0.0201mm |

这是固定相机、静态单目标、刚体空杯的仿真结果。策略仅在开始时读取RGB、对齐米制深度和有效性掩码，随后开环执行学习到的关节轨迹；不会在线跟踪移动目标。真实杯子变形、相机标定和真机抓取尚未验证，项目没有自动使能真机的入口。

[模型说明](MODEL_CARD.md) · [侧夹修复与完整实验](docs/side-grasp-retraining.md) · [侧面特写视频](runs/paper-v4-side-demo/closeup.mp4) · [测试明细](data/paper-rgbd-v4/test/trials.json)

## 安装与首次回放

本机已验证环境为 Ubuntu 22.04 x86_64、Python 3.10、ROS 2 Humble、MuJoCo 3.3.7、PyTorch 2.7.1/CUDA 12.8。仿真与学习使用 `.venv`，真机ROS工具另用 `.venv-hardware`。训练模型推理也支持CPU。

```bash
 git clone https://github.com/Jing-2237/MM-ALPHA.git
 cd MM-ALPHA
 # 完整环境安装；system阶段会请求sudo并安装ROS及系统依赖。
 bash scripts/bootstrap.sh
 source scripts/env.sh
 python scripts/verify_release.py
 MUJOCO_GL=glfw python -m aloha2_lab.visual_il rollout \
   --root "$ALOHA_LAB_ROOT/data/paper-rgbd-v4" \
   --output "$ALOHA_LAB_ROOT/runs/my-first-v4" --viewer
```

已有ROS与图形依赖时，可只执行 `bash scripts/bootstrap.sh sources` 和 `bash scripts/bootstrap.sh python`；需要真机工作区时再执行 `hardware-python`、`build` 阶段。安装脚本不安装NVIDIA驱动，需要GPU时应先有兼容驱动。无桌面环境用 `MUJOCO_GL=egl` 并去掉 `--viewer`。已有输出目录不会覆盖，每次回放换新名字。

指定杯子位置示例：

```bash
MUJOCO_GL=glfw python -m aloha2_lab.visual_il rollout \
  --root "$ALOHA_LAB_ROOT/data/paper-rgbd-v4" --cup -0.02 0.06 \
  --output "$ALOHA_LAB_ROOT/runs/my-v4-position-1" --viewer
```

训练网格范围为x∈[-0.06,0.02]m、y∈[0.02,0.105]m。这不是任意位置成功保证；坐标系、实测夹爪开口、失败点及碰撞门槛见[实验报告](docs/side-grasp-retraining.md)。训练权重应与同目录 `split.json` 中的杯子及物理配置一起使用。

## 数据、重新训练与验收

`data/paper-rgbd-v4/` 保留完整的已完成episode、失败结果、训练配置及全部测试对照；2个中断的 `.partial` 文件仅保留在原电脑，不上传。训练器只选择任务成功且独立回放通过的记录。不要直接覆盖已发布模型；新批次请另建目录。

[重新采集和训练命令](docs/side-grasp-retraining.md#重建新批次) · [HDF5数据约定](docs/data-and-training.md) · [RGB-D格式与真机深度](docs/paper-cup-rgbd.md)

```bash
source scripts/env.sh
python -m pytest tests -q
python scripts/verify_release.py
```

`verify_release.py` 校验发布文件SHA-256、成功示范记录、空间划分和模型输出。完整独立物理重放可运行 `python scripts/verify_side_dataset.py --root data/paper-rgbd-v4`。对新模型执行 `visual_il test` 会创建独立 `test/` 目录，已发布的测试目录不可覆盖。原工作区47项测试通过；干净检出验证44项通过、3项因不包含历史模型而跳过，日志位于 `runs/publication-fresh-checkout.log`。干净检出复用了已按锁定提交下载的Menagerie模型。

## 真机与ROS

ROS仿真接口：`ros2 launch aloha2_lab_bringup sim.launch.py`。真机只读检查和原始RGB-D数据录制需先 `source scripts/hardware_env.sh`，详见[硬件接入](docs/hardware.md)与 `config/hardware.example.yaml`。原始真实rosbag需完成时间同步、深度单位、外参和关节单位标定后转换，不能直接套用仿真权重驱动机器人。

## 仓库内容

| 路径 | 内容 |
|---|---|
| `src/aloha2_lab/` | MuJoCo、视觉输入、接触审计、示范记录、训练和ROS接口 |
| `data/paper-rgbd-v4/` | 最终模型、示范数据、划分、训练记录和测试证据 |
| `runs/paper-v4-side-demo/` | 全景、特写、动作、起始RGB-D和成功判据 |
| `config/` | 杯子参数、Python和上游提交锁、硬件配置模板 |
| `scripts/` | 环境安装、校验、采集及ROS检查 |
| `ros2_ws/src/` | 本地bringup包及上游源码相对链接 |
| `tests/` | 数据、物理、学习与录制契约测试 |
| `docs/` | 使用、实验、真机接入和参考代码审查 |

虚拟环境、系统ROS、上游完整源码、构建缓存、旧模型和大体积历史采集不随仓库上传。`docs/` 中历史报告描述原电脑的验证结果，提到的旧实验文件不一定包含在本次发布中。上游依赖保留各自许可证；杯子标志为仿真近似外观，相关商标归权利人所有。发布清单及文件摘要见 [RELEASE_MANIFEST.json](RELEASE_MANIFEST.json)。
