# 纸杯侧夹修复与重新训练（v4）

最新版使用 `data/paper-rgbd-v4`，杯子仍为用户确认的下直径5cm、上直径10cm、高12cm麦当劳纸杯。已重新采集、独立回放检查并完成1200轮GPU训练。预先固定的10个未见位置中 **9/10成功**；成功必须同时满足抬起、双指侧壁夹持、碰撞预算及稳定保持。1个位置没有夹稳，倾倒后判失败，未更换测试点或降低门槛。

## 修复原因和动作

旧版v2的一个已判成功样例，重放后测得机器人与杯壁最大重叠8.773mm，因此旧报告的“抬起来”不能证明抓法正确。另一个问题是把每侧滑轨0.037m误解释为夹爪净开口7.4cm；按当前指尖碰撞几何测量，全开净间距约 **57.8mm**。本次没有加大机器人关节行程，也没有关闭杯子碰撞。

新版先保持机械臂不动并全开夹爪60步（1.2秒），再将手腕抬高，以35°向下倾角从侧面接近杯身较窄的下部，闭爪后增加小幅夹持余量，再抬升16cm并保持2秒。两指需要分别与杯身侧壁接触，接触法向相反，不能用杯底托住冒充侧夹。TCP目标高度21mm，实际接触位置还取决于指尖几何。

纸杯碰撞参数改为更硬的接触，物理步长由原设置细化至1ms；接触监测逐物理步运行。左夹爪模拟执行器力限幅为±3N，接触力还受惯性和约束影响，瞬时接触峰值可能超过3N。模型仍是刚体空杯，不能据此保证真实纸杯不会被捏瘪。

## 数据和训练证据

- 训练位置尝试25次，17次通过；验证位置6次均通过。8次失败保留在采集结果中，未用于训练。
- 23段成功记录先按保存的float32动作独立重放，**23/23通过**，检查结果在 `data/paper-rgbd-v4/saved-action-replay.json`。新教师执行前也量化动作，避免保存精度导致重放不同。
- 使用17段训练、6段验证，CUDA训练1200轮。验证集用于选模型和解码器正则项，测试数据不参与训练。权重与训练记录在 `data/paper-rgbd-v4/model/`。
- 种子2037预先固定划分；测试坐标距离所有训练/验证坐标至少7.05mm。训练网格范围x∈[-0.06,0.02]m、y∈[0.02,0.105]m，不代表范围内任意位置都能成功。
- v3为中间实验，复查发现一条边缘夹持记录保存后重放失败，已由本次v4取代。v2/v3的旧数据和模型保留，默认演示改用v4。

网络输入为起始RGB、对齐米制深度和深度有效性掩码，学习视觉特征及整段关节轨迹；执行时不调用目标位置、定位器或IK。IK仅用于采集教师示范。当前是起始图像条件化的开环策略，不会追踪动作中移动的杯子。特写观察相机会跟随杯子，观察画面不输入策略。

## 独立测试结果

| 检查 | 结果 |
|---|---:|
| 正常RGB-D，严格成功 | 9/10 |
| RGB置黑 | 0/10 |
| 换为另一位置的深度 | 1/10 |
| 缺少有效深度 | 10/10拒绝推理 |
| 全部正常测试的杯壁最大重叠 | 0.02010mm |
| 闭爪前最大杯壁重叠 | 0.01491mm |
| 手部与桌面最大重叠 | 0.21014mm |
| 非手指机器人部件承力接触杯子 | 0次 |

失败点为x=-0.0033113m、y=0.0974393m：最终无双指接触，杯子倾角77.68°，判为夹持失败。未将这个点补进训练数据后重复声称独立测试。

碰撞门槛：杯壁和手桌重叠各≤0.5mm，闭爪前杯壁≤0.1mm；保持阶段杯子相对夹爪漂移≤5mm、倾角≤20°；末态每指侧壁法向力>0.15N，方向相对。MuJoCo软接触存在小量重叠，此处不声称严格零穿透。全部试验与失败详情见 `data/paper-rgbd-v4/test/trials.json`；统计见同目录 `report.json`、`contact-summary.json`。

47项代码测试通过，日志 `runs/side-v4-tests.log`。典型成功回放在 `runs/paper-v4-side-demo/`，含全景 `rollout.mp4`、标注指尖间距和最大重叠的 `closeup.mp4`、动作及 `report.json`。

## 查看新策略

在项目根目录打开终端，使用新的输出目录名：

```bash
source scripts/env.sh
MUJOCO_GL=glfw python -m aloha2_lab.visual_il rollout \
  --root "$ALOHA_LAB_ROOT/data/paper-rgbd-v4" \
  --output "$ALOHA_LAB_ROOT/runs/my-side-grasp-1" --viewer
```

默认使用第一个预留测试位置。可加 `--cup -0.02 0.06` 指定静态杯子位置；不会为指定位置在线重新规划。再次运行须换输出目录，已有文件不会覆盖。

## 重建新批次

以下会创建新目录、重新采集和训练，不需要为观看回放重复执行：

```bash
source scripts/env.sh
python -m aloha2_lab.visual_il plan --root "$ALOHA_LAB_ROOT/data/paper-side-new" \
  --sensor rgbd --cup-profile config/paper_cup_v4.json --seed 2037
python -m aloha2_lab.visual_il collect --root "$ALOHA_LAB_ROOT/data/paper-side-new"
python scripts/verify_side_dataset.py --root "$ALOHA_LAB_ROOT/data/paper-side-new"
python -m aloha2_lab.visual_il train --root "$ALOHA_LAB_ROOT/data/paper-side-new" --epochs 1200
python -m aloha2_lab.visual_il test --root "$ALOHA_LAB_ROOT/data/paper-side-new"
```

新版本训练器要求先完成独立保存动作回放，只选择通过的成功记录。真实纸杯材质、实际指尖、深度标定和摩擦都需要实测；本次未执行真机动作。
