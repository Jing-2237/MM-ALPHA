# 历史实验提示

本页记录v2纸杯实验。复查发现旧成功判据未排除大幅穿透，请使用[新版v4侧夹修复与重新训练](side-grasp-retraining.md)中的模型、回放和严格测试结果。下文保留作为历史记录与深度数据格式说明。

# 麦当劳纸杯 RGB-D 模仿学习实验

## 对象与抓法

按用户最终确认：**下直径5cm、上直径10cm、高12cm**，即下半径0.025m、上半径0.05m、高0.12m。参考图保存在 `assets/references/mcdonalds-paper-cup.jpeg`；参数为 `config/paper_cup.json`。

模型为无把手、开口锥台纸杯，包含可碰撞的杯壁、杯底和卷口；米白杯身、红黄标记为代码生成的近似外观，不是单张照片的精确纹理重建。杯口10cm大于夹爪指尖实际全开净间距约5.78cm（原文7.4cm是滑轨行程的误读，现已更正），因此从下部较窄杯身**侧向夹持**，没有偷偷把杯子缩小或把夹爪扩大。

纸杯按空杯刚体近似，质量18g、壁厚1.5mm、接触摩擦等是仿真假设，未通过实物测量；不模拟纸壁压溃、液体或泼洒。本次没有真机动作。

## 深度实际用于训练

训练不是只保存深度文件。输入为5通道：RGB三通道、米制对齐深度、深度有效性掩码。网络用这5个通道生成空间注意力，同时输出图像特征位置及注意力加权的**米制深度特征**；三维特征共同输入从成功动作轨迹拟合的解码器。

- `16UC1` 深度以毫米换算到米；`32FC1` 使用米。
- NaN、Inf、0及范围外深度会被掩码；归一化使用固定米制尺度，不读取测试集统计量。
- 模型没有深度时拒绝执行，不会悄悄退回RGB旧模型。
- 策略依然以起始对齐RGB-D为条件预测整段轨迹；没有逐步视觉纠偏，不支持抓取途中移动杯子。

`rgbd_input.py` 的测试确认深度参与前向计算与反向梯度。推理期间不调用教师、IK、颜色/几何定位器，也不读取物体真实坐标。仿真真实状态只用于独立成功评估。

## 数据和结果

当前有效实验：`data/paper-rgbd-v2/`。

| 项目 | 结果 |
|---|---|
| 训练位置采集 | 25次尝试，21次成功、4次失败 |
| 验证位置采集 | 6次尝试，6次成功 |
| 训练 | GPU，1200轮；仅成功示范用于训练/验证 |
| 未训练坐标 | 训练前预留10个，距离训练/验证坐标至少7.77mm |
| 正确RGB＋正确深度 | **9/10成功** |
| 保留深度、RGB置黑 | **0/10成功** |
| 保留RGB、替换成另一位置的深度 | **2/10成功** |
| 完全缺失有效深度 | **10次均拒绝执行**，不是10次执行后的运动失败 |
| 自动检查 | 42项通过 |

成功判据在测试前确定：杯心高度超过0.12m，抬升后保持2秒期间始终超过此高度，且最终两个实体手指仍接触杯子。没有焊接/附着约束、物体瞬移或改判据来制造成功。

失败位置是 `(-0.0510842, 0.0684147)` 米：杯子仍被双指夹持，但保持段最低杯心高度约0.1170m，未达到已定阈值，**仍计失败**。模型与超参数没有根据这次测试结果继续调整。轨迹解码器的正则参数仅根据验证动作误差选择。

训练范围为 `x∈[-0.06,0.02]`、`y∈[0.02,0.105]` 米。这是适合当前侧夹动作的小范围位置插值测试，不代表整个桌面、不同杯型、遮挡、形变、不同相机或真实机器人上的成功率。

较早的 `data/paper-rgbd-v1/` 保留了沿用原杯位置时的IK不可达/碰撞尝试；未用于训练，也没有替换或合并到新实验的测试分母中。

## 保存内容

每段 HDF5 包含：

- `initial_rgb`、`initial_depth_m`、`initial_depth_valid`。
- `camera_K`、`T_world_optical`、`image_timestamp`、深度单位和对齐说明。
- 820步关节位置、速度、动作、仿真时间戳；控制周期0.02s。
- 杯型参数、示范来源、任务结果与完整性标记。

数据是仿真视觉/几何定位加IK教师生成的成功示范，来源标签为 `vision_ik_teacher_paper_cup_side`。不是人工示范，也没有逐帧保存RGB-D的声明；本策略只需起始RGB-D。

关键文件：

- `split.json`：预定位置、RGB-D模式、完整杯型参数。
- `collection.json`：全部采集尝试，包括失败。
- `model/encoder.pt`、`model/decoder.npz`、`model/training.json`：权重与训练/验证记录。
- `test/trials.json`、`test/report.json`：正常与消融测试的全部逐次记录。
- `runs/paper-rgbd-learned-demo/`：学习策略实测、视频、起始RGB和深度。
- `runs/paper-cup-confirmed/object.png`：确认尺寸后的模型预览。

## 运行已训练纸杯策略

```bash
source /home/wj/Documents/Codex/2026-10-29/wo/outputs/aloha2-lab/scripts/env.sh
MUJOCO_GL=glfw python -m aloha2_lab.visual_il rollout \
  --root "$ALOHA_LAB_ROOT/data/paper-rgbd-v2" \
  --output "$ALOHA_LAB_ROOT/runs/paper-rgbd-demo-1" --viewer
```

不指定 `--cup` 时选该实验第一个预留测试位置；也可显式指定训练区域内坐标。场景坐标只用于生成杯子，不传给策略；模型接收的是渲染的对齐RGB-D。旧 `coffee-visual-v1` 权重属于带把手蓝杯RGB实验，不能复用为新纸杯策略。

复现实验使用新目录：

```bash
paper_root="$ALOHA_LAB_ROOT/data/paper-rgbd-v3"
python -m aloha2_lab.visual_il plan --root "$paper_root" --sensor rgbd \
  --cup-profile "$ALOHA_LAB_ROOT/config/paper_cup.json"
MUJOCO_GL=egl python -m aloha2_lab.visual_il collect --root "$paper_root"
python -m aloha2_lab.visual_il train --root "$paper_root" --epochs 1200
MUJOCO_GL=egl python -m aloha2_lab.visual_il test --root "$paper_root"
```

## 真机相机采集

`hardware.example.yaml` 已把主相机的 aligned depth 和 CameraInfo 列为必需话题；其他相机的深度列为可选话题。RGB-only录制配置不能当作RGB-D训练数据。

在官方机器人配置的 `robot.cameras.common_parameters` 中启用：

```yaml
enable_depth: true
align_depth.enable: true
enable_sync: true
```

确认实际话题名、编码和时间戳后，按 `docs/hardware.md` 使用原始录制工具。真实RGB与深度必须同步、像素对齐，并保留相机标定；`16UC1`与`32FC1`不能混用单位。新模型仍使用仿真起始图像与固定裁剪区域，真实相机分辨率、深度噪声、外参、杯子材料与实际夹爪接触均未现场验收，原始真实rosbag也仍需校准转换后才能训练。

真机原始录制工具的合成ROS测试也已通过：14路话题共714条消息成功读回，包含16UC1毫米深度与CameraInfo；见 `runs/paper-rgbd-recording-verification/20261030T062955.993871Z/report.json`。这是软件通路验证，不是真实相机采集。
