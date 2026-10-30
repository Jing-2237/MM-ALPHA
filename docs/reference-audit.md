# RobotAgent 初步代码核查

核查来源：`https://github.com/Jing-2237/RobotAgent`，提交 `73b0c976ecf72710f41701b49ba720f2707309ba`。本地副本位于 `third_party/RobotAgent/`。这是针对搭建路线和夹爪桥接器的局部审查，并非全仓库验收；没有执行真机动作。

## 与本项目目标的差异

| 已核实事实 | 依据 | 本项目处理 |
|---|---|---|
| 仓库面向 ALOHA Solo 单 follower，而用户设备为 ALOHA 2 双臂 | `README.md:3`；`src/arirobot/config.py:267` 的 `require_hardware()` 约束 `follower_left`；`ros2_ws/src/arirobot_ros/launch/aloha_hardware.launch.py` 同样约束单臂 | 使用 ALOHA 2 双臂 MuJoCo 模型和明确的左右顺序；不直接复用单臂硬件 launch |
| 仓库动作是 MoveIt 执行链，不包含本任务所需的示范学习训练链 | `README.md:5` 明确“系统不训练模型”；`src/arirobot/motion/ros_backend.py` 使用 MoveGroup / ExecuteTrajectory | 单独实现采集、动作分块行为克隆和仿真回放；当前基线只读关节状态，尚不是视觉 ACT |
| 现有真机视觉链只使用一台主相机 | `README.md:80` 和 `README.md:411` | 仿真可保存四路相机；真机图像须按实际配置和时间戳单独验收 |
| 仓库自述真机标定和成功率尚待现场验证 | `README.md:7` | 不把软件测试或仿真轨迹回放当作实物抓放成功 |

## 已定位的缺陷

以下文件路径均相对于参考仓库。

1. **夹爪反馈未验证新鲜度。** `ros2_ws/src/arirobot_ros/scripts/arirobot_aloha_gripper_bridge.py:69` 保存反馈时间戳，`execute()` 在第 115–124 行只取位置和力，没有读取或检查保存的时间戳。旧反馈恰好等于目标时仍可返回成功。必须检查时间基准、年龄、关节序列和有限值后才可采用反馈。
2. **夹爪子动作超时后未取消。** 同文件第 98–102 行处理受理超时、第 110–114 行处理结果超时时，只结束外层 Action。受理迟到没有补偿取消；结果超时也没有向已经受理的子动作发出取消。外层结束不能证明硬件停止。
3. **`max_effort` 没有落实到执行层。** 同文件第 74–77 行仅验证请求值为有限正数；第 82–127 行没有使用这个限值，也未向控制器传递限制。不能据此声称请求的夹持力上限已生效。当前状态是否可由硬件配置限制，还需检查具体驱动，不能由这个 Action 参数推断。
4. **README 的构建目录与 source 路径不一致。** `README.md:174–181` 从仓库根目录运行 `colcon build`，随后读取 `ros2_ws/install/setup.bash`。未配置额外 colcon 默认参数时，构建目录由当前工作目录决定，文档应显式进入 `ros2_ws`，或指定 `--build-base`、`--install-base` 和日志目录。

仓库内 `tests/test_aloha_hardware.py` 当前仅测试夹爪几何换算和开度范围，不覆盖上述超时、取消或反馈新鲜度逻辑。测试范围不足以证明这些路径正确。

## 离线复现证据

运行项目中的 `tests/reference_bridge_probe.py` 可在不导入 ROS、不创建任何硬件连接的情况下，用原始 `execute()` 方法体和假的传输对象复现前三项的局部行为。结果保存在 `docs/reference-bridge-probe.json`：

- 时间戳为 0 的反馈仍触发 `succeed`。
- 子动作结果超时只触发外层 `abort`，没有 `cancel_child`。
- 受理超时只触发外层 `abort`，没有注册迟到取消。
- 请求 `max_effort=0.01` 的示例能返回 `effort=0.5` 并成功；这证明该方法未执行上限校验，不是对真实力单位或实际硬件力值的测量。

该探针只验证方法体的本地分支，不验证 ROS executor 调度、真实网络时序和电机行为。参考仓库副本保持原样；本项目当前仿真/训练路径不调用该夹爪桥接器。若将来复用它，必须先补修并通过独立的异步取消和新鲜度测试。

## 新工程本轮修正

`src/aloha2_lab/dataset.py`：发布前校验完整 episode；少于两帧或中断的数据保留为 `.partial`；拒绝夹爪单位、左右关节顺序、仿真标记、观测/动作对齐不一致的数据；检查固定采样周期；图像尺寸改变时先拒绝整帧，避免状态与图像长度不同。

`src/aloha2_lab/learning.py`：拒绝混合控制周期；训练/验证按 episode 切分，归一化只来自训练 episode；动作分块不会跨 episode；尾部 padding 不参与损失；checkpoint 保存控制周期，回放必须使用该周期。

局限：当前状态行为克隆会把完整状态数据和动作分块装入内存，适用于初步小规模实验；长时间、多任务大数据训练应改为按 episode/帧流式读取。相机数据已可记录，但当前训练器不使用图像；机器人标定、真机动作映射和实物任务成功率仍需后续验证。
