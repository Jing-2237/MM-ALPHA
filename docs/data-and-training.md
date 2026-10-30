> 最新无把手纸杯RGB-D实验见 [纸杯深度训练报告](paper-cup-rgbd.md)。旧带把手杯RGB权重不适用于新纸杯。

> 本页介绍原有状态行为克隆与数据约定。已训练的咖啡杯RGB视觉策略另见 [视觉模仿学习报告](coffee-visual-il.md)，请使用 `visual_il train/rollout` 入口。

# 数据与训练说明

当前项目支持两种来源明确分开的记录：MuJoCo 仿真示范保存为可训练 HDF5，真机已有 ROS 流保存为原始 rosbag。后者保留原消息和时间戳，尚未完成同步、夹爪单位转换与训练导入。

仿真、训练与 CUDA 验证使用 `source scripts/env.sh` 启用学习 `.venv`；真机官方 Python 脚本、只读审计和原始录制使用 `source scripts/hardware_env.sh` 启用 `.venv-hardware`。后者复用系统 OpenCV/NumPy，不需要 GPU Torch 即可检查和录制 ROS 流。

## 仿真 episode 合约

一个文件对应一段连续示范，文件名采用 `episode_000000.hdf5`。文件写入时先使用 `.hdf5.partial`，正常结束、至少两帧且结构校验通过后才发布正式文件；异常中断的文件保留供排查。正式文件不覆盖已有数据。

| 字段 | 形状 | 含义 |
|---|---|---|
| `observations/qpos` | `T × 14`，float32 | 发送该步动作前的当前位置 |
| `observations/qvel` | `T × 14`，float32 | 同时刻速度 |
| `action` | `T × 14`，float32 | 此时发送的绝对位置目标，不是位移增量 |
| `timestamp` | `T`，float64 | 仿真秒，默认间隔 0.02 秒 |
| `observations/images/<camera>` | `T × 240 × 320 × 3`，uint8 | 可选 RGB 图像，采样在动作之前 |

顺序固定为：

```text
left/waist, left/shoulder, left/elbow, left/forearm_roll,
left/wrist_angle, left/wrist_rotate, left/gripper,
right/waist, right/shoulder, right/elbow, right/forearm_roll,
right/wrist_angle, right/wrist_rotate, right/gripper
```

臂关节的位置/目标单位是弧度，速度是弧度/秒；夹爪位置/目标是模型中**单个手指沿导轨的位移，单位米**，速度是米/秒。这既不是两指总开口，也不是电机弧度、PWM 或归一化的 `[0, 1]` 数值。MuJoCo 模型中的两个手指通过约束关联，14 个动作量只包含每侧一个夹爪控制量。

关键文件属性：

- `schema = aloha2-lab/sim-v1`，`sim = true`。
- `joint_names`、`gripper_unit`、`dt`、`observation_action_alignment = observation_before_action`。
- `source = scripted_smoke_not_human`：程序生成的链路测试轨迹。
- `source = human_mujoco_control_sliders`：在交互窗口中操作控制滑块的示范。
- `source = learned_policy_simulation`：策略仿真回放。

训练器不自动筛选“好示范”。任务失败、暂停太久、不相关动作和策略自己产生的回放应由数据管理流程审查，不能混进人工专家训练集后当作专家数据统计。

## 当前行为克隆基线

输入是 14 维 `qpos` 和 14 维 `qvel`，总计 28 维状态。两层 128 单元 MLP 预测连续若干步的 14 维位置目标，默认 chunk 为 10。这是动作分块行为克隆，**不是 ACT 的 Transformer/CVAE 实现**。模型不读取图像、语言或物体位置，不能据此期望它完成依赖视觉变化的泛化抓取。

训练规则：

1. 至少三个完整 episode；按照整个 episode 随机划分约 80% 训练/20% 验证，默认种子为 7。相邻帧不会被随机拆到两个集合。
2. 输入和动作的均值、标准差只从训练 episode 计算，避免验证数据泄漏。
3. chunk 不跨 episode。尾部不足部分用末帧填充，但 padding 通过 mask 排除在损失之外。
4. 训练数据的 `dt` 必须一致。混合 30Hz/50Hz 数据时，应先明确重采样和动作时间关系，不能只修改属性。
5. 验证误差最低的 checkpoint 保存为 `policy.pt`，保存模型参数、训练归一化统计和控制周期。

`metrics.json` 记录训练/验证 episode 路径、随机种子、控制周期和每个 epoch 的标准化均方误差。这个误差不直接等于毫米误差、抓取成功率或安全性指标。比较实验时需要相同任务、采样频率、数据质量和划分方式。

当前训练器将全部状态和动作分块放入内存，再复制到目标设备，适合先验证小规模实验。大量长 episode 应改为按需读取和 mini-batch 数据加载，或接入成熟的 ACT/LeRobot 训练器。独立图像 HDF5 字段目前不被训练器读取，因此录制图像不会自动变成视觉策略。

## 回放解释

`aloha-lab evaluate` 从仿真初始姿态开始，每步重新预测动作分块，执行预测的第一步，再采集新观测。动作会限制在模型范围内，并限制相邻命令变化速率。checkpoint 的控制周期用于回放和记录。

回放报告字段：

- `finite`：该次运行未发现状态数值发散。
- `clipped_steps`：预测命令被范围/速率限制的步数。
- `max_contacts`：模型报告的最大接触数量；不等于碰撞违规次数，也不等于抓取完成。
- `task_success_evaluated = false`：当前未定义和测量物体任务成功率。
- `hardware_motion_commanded = false`：此入口仅运行 MuJoCo。

物体抓取任务需另行加入对象、目标、成功判据、扰动和独立测试集。低离线损失只是数据拟合证据，必须结合闭环任务验证。

## 真机数据进入训练前

硬件步骤见[硬件接入说明](hardware.md)。`hardware_record.py` 检查配置中必需主题的存在性和消息类型，再调用 rosbag2 订阅录制。它保存接收时间和原始消息头，不保证四路图像、主从关节和无 header 的命令已经同步。

转换器需要验证以下事实，当前版本不会猜测填入：

- 实机左右臂和每个 JointState 的名称顺序，不能按收到的数组位置盲目拼接。
- 主臂示范命令与从臂测量值的对应时刻，以及“观测在动作之前”的时间关系。
- 夹爪电机位置、手指位置和开口之间经过测量的转换；当前仿真夹爪单位不能直接当成真机电机命令。
- 相机 serial、图像编码、帧率、时间偏差、丢帧情况；标定参数和数据版本。
- 明确 episode 起止、任务成功/失败、人工介入和示范者信息。

完成这些步骤后，应输出独立的硬件数据 schema 和转换报告，再扩展训练器。当前 sim-v1 校验会拒绝 `sim = false`，避免意外把原始硬件单位当成仿真单位。

## 容量管理

图像是主要存储开销。单帧未压缩 RGB 大小为 `宽 × 高 × 3` 字节，多相机乘以相机数，再乘帧率和时长。LZF 压缩率取决于画面内容，不应预先假定能节省固定比例。采集前可运行 `df -h .`；真机录制入口默认保留 5GiB、每秒检查一次空间，达到保留量时只停止自己的 rosbag 录制并保留收尾数据，manifest 记录原因，不宣称示范成功。它不停止机器人、不清理旧数据，详见[硬件录制的存储配置](hardware.md)。

建议按任务和采集批次划分目录，保留完整来源及校准记录。需要迁移大规模数据时，可以把数据和实验输出路径指向已确认的数据分区，代码与虚拟环境继续保留在 Linux 文件系统。
