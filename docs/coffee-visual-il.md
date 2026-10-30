# 咖啡杯视觉模仿学习：本机实验

模型已实际训练、测试。咖啡杯具有开口杯壁、可碰撞的实体把手；杯内深色表面只表示咖啡外观，不模拟液体。本轮没有真机动作。

## 结果

| 项目 | 实际结果 |
|---|---|
| 示范采集 | 25个训练位置尝试：23成功、2失败；6个验证位置尝试：5成功、1失败 |
| 训练输入 | 每段示范的起始 overhead RGB；23段训练、5段验证成功示范 |
| 模型 | CNN空间注意力视觉编码器 + 从示范拟合的动作轨迹解码器 |
| 训练 | GPU，1200轮；仅用验证集选择编码器checkpoint |
| 未训练位置 | 预先固定10个位置，与训练/验证坐标最小距离7.22mm |
| 正常图像 | **10/10成功** |
| 所有位置使用同一张训练参考图 | **5/10成功** |
| 全黑图像 | **0/10成功** |
| 自动检查 | 38项通过 |

成功判据：杯心高于桌面0.12m，抬升后保持2秒期间高度始终高于0.12m，最终两个实体手指仍与杯子接触。没有使用焊接、附着约束或把杯子瞬移到夹爪。所有测试结果和失败均保留。

正常测试的动作限幅计数中存在181步浮点边界修正，最大幅度仅4.77e-10（数值舍入）；不能把该计数误解为大幅越界。

## 视觉和动作怎样学习

采集时使用原先的RGB-D视觉定位和IK教师生成动作，来源标为 `vision_ik_teacher_coffee_mug`，不是人工遥操作。失败示范不进入训练，但失败记录不会删掉。

视觉编码器从RGB像素学习咖啡杯的位置特征，辅助监督是由教师RGB-D定位得到的图像点。动作解码器从训练示范的整段关节命令学习视觉特征到动作轨迹的关系；其轨迹均值、PCA基和回归系数全部只用训练集拟合。验证集用于模型选择，不进入轨迹基与回归拟合。

**测试推理只接收起始RGB。** 不接收杯子坐标、仿真物体状态、深度或教师像素标签；不调用颜色定位器或IK。物体真实位姿仅由独立评估代码计算成功指标。`tests/test_visual_il.py` 会禁止教师函数并检查改变图像确实改变预测动作。

当前是**起始图像条件化的开环轨迹模仿学习**，低层执行器仍有位置伺服。它没有每步重看图像纠偏，不支持执行过程中移动杯子。完整轨迹820步，控制周期0.02秒。

## 位置划分和数据

数据根目录：`$ALOHA_LAB_ROOT/data/coffee-visual-v1`。

- `split.json`：采集前固定的训练、验证、测试坐标和随机种子。
- `collection.json`：全部31次采集尝试及失败原因。
- `train/episode_*.hdf5`、`validation/episode_*.hdf5`：关节状态/速度/动作/时间戳、起始RGB、教师像素标签、结果与来源。
- `model/encoder.pt`、`model/decoder.npz`：训练后的视觉编码器和动作解码器。
- `model/training.json`：训练/验证文件列表、指标、架构和训练曲线。
- `test/trials.json`、`test/report.json`：正常和空图像测试的逐次记录与汇总。
- `test/fixed-image-trials.json`：同一参考图的对照结果。

训练位置覆盖 `x ∈ [-0.21,-0.14]`、`y ∈ [0.02,0.105]` 米。测试在此区域的未训练坐标插值；它不是整个允许工作区的随机测试，也没有验证新杯型、相机位置、遮挡、动态目标或真机。

新训练格式保存的是起始图像，不声称包含每一步的RGB。不要把旧的状态学习器 `aloha-lab train` 当作本视觉策略的训练入口。

## 运行已训练策略

```bash
source /home/wj/Documents/Codex/2026-10-29/wo/outputs/aloha2-lab/scripts/env.sh
MUJOCO_GL=glfw python -m aloha2_lab.visual_il rollout \
  --root "$ALOHA_LAB_ROOT/data/coffee-visual-v1" \
  --cup -.16321004092160085 .07221568438742393 \
  --output "$ALOHA_LAB_ROOT/runs/coffee-policy-demo-1" --viewer
```

这条命令运行学习策略。`aloha-lab cup-grasp` 仍是IK教师，不要混淆两个入口。`--cup`只生成测试场景，坐标不传入模型。

运行后输出起始图像、预测动作、物理结果和5fps的过程视频。已有示例为 `runs/coffee-learned-demo/rollout.mp4`，长度16.4秒。

## 重新采集、训练和测试

选择全新根目录，以下命令按顺序运行；保留原实验不覆盖：

```bash
coffee_root="$ALOHA_LAB_ROOT/data/coffee-visual-v2"
python -m aloha2_lab.visual_il plan --root "$coffee_root"
MUJOCO_GL=egl python -m aloha2_lab.visual_il collect --root "$coffee_root"
python -m aloha2_lab.visual_il train --root "$coffee_root" --epochs 1200
MUJOCO_GL=egl python -m aloha2_lab.visual_il test --root "$coffee_root"
```

默认种子和位置划分固定以便复现；改测试集做新的泛化实验时，必须在训练前确定，并将结果作为独立实验报告。本次10个位置样本较少，10/10只说明这个受控仿真实验成功，不是任意位置保证。

复现实验中的同一参考图对照（新建输出目录，不覆盖原结果）：

```bash
MUJOCO_GL=egl python scripts/verify_coffee_reference_ablation.py \
  --root "$ALOHA_LAB_ROOT/data/coffee-visual-v1" \
  --output "$ALOHA_LAB_ROOT/runs/coffee-fixed-reference-recheck"
```
