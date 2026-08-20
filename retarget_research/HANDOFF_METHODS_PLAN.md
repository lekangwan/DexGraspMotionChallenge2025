# 灵巧手重定向方法探索：交接文档

> 本文档交给子 agent 执行后续方法探索。先读第 0–1 节了解现状与上手约定，再按第 2–3 节的方法路线推进。所有实验命令在"用户终端"执行，子 agent 负责写代码、分析、准备命令与解读结果。

## 0. 项目现状（2026-08-17）

### 任务与口径

- 把 1000 条 Shadow Hand 官方成功抓取轨迹（50 类 / 100 物体 / 1000 轨迹，manifest 已冻结）重定向到三只目标手，在 Isaac Gym CPU PhysX 中重放，按 **稳定 30 cm 协议 v2** 判成功：物体相对落稳初始高度抬升 ≥30 cm，末 30 个 60 Hz 物理步持续满足高度、接触、回落（峰值→末段 ≤3 cm）和波动（末段波动 ≤1 cm）门；另有"运输质量"门（末段掌物相对滑移，由 audit_stable_success.py 从 trace 计算）。
- 专家数据 = 运输质量通过的轨迹；进阶策略（MLP / Temporal3 / Diffusion 链路已就绪）在专家数据上训练。

### 三手冻结方法与最新数字

| 手 | DOF | 正式方法 | 正式 1000（30cm 协议） | 开发集 20 条（30cm） |
|---|---|---|---|---|
| Linker O6 | 6 | 渐进夹紧 v1 | 稳定 169 / 运输 136 | 夹紧 4/20 |
| XHand | 12 | 官方基线（待换） | 稳定 516 / 运输 501 | 官方 12/20、指腹 13/20、**向量 14/20** |
| Wuji | 20 | 待用户决策 | n005：431/421；旧无约束：581/557（0 专家） | — |

- 开发集 = `retargeting/configs/linker_independent_validation_v1.json`（10 物体 × 2 轨迹 = 20 条，已用于方法选择的开发集，不是最终测试集）。
- XHand 50 类确认（`manifests/xhand_confirmation_50c_50t_seed20260817.json`，与 Wuji train50 同轨迹）：官方 27/50，纯向量 **30/50**（+6/-3），方向与 dev20 一致。

### 待用户拍板的决策点

1. **Wuji 手型门**：旧无约束方法靠四指反弯（URDF -28.3° 负向范围）拿到 581/557，但手型门 quarantine 全部隔离（0 专家）；修复版 n005 431/421 全合规。用户倾向：成功率优先、撤销 quarantine、正式用旧方法、视觉缺陷写进报告局限性章节；n005 降级为"手型合规修复"消融。**等用户确认后**改 `retargeting/configs/stable_success_protocol_v2.json` 中 wuji 的 `training_anatomy_gate`。
2. **XHand 正式 1000 是否换向量法**（50 类 +6 已确认，重跑正式约 3 小时）。

## 1. 上手约定（子 agent 必读）

### 环境与数据

- 环境：`/home/lekangwan/miniconda3/envs/hand-retarget/bin/python`（Python 3.8），Isaac Gym 在 `/home/lekangwan/isaacgym`。
- 运行环境变量：`MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=4 MKL_NUM_THREADS=4`（所有命令统一前缀）。
- 开发集源轨迹：`retarget_research/reference/HandRetargetTask2026/scripts/data/sorting/seq41proc/*.npy`；开发集物体网格：`.../sorting/object_41/<object_name>/`。
- 正式源轨迹：`/home/lekangwan/projects/DexGraspMotionChallenge2025/external_data/dataset/*.npy`；正式物体：`.../external_data/meshdata/<object_name>/`。
- 工作目录始终为仓库根：`/home/lekangwan/projects/DexGraspMotionChallenge2025-final-release`（object_geometry 依赖相对资产路径，勿在子目录运行）。

### 代码分区与入口

- `retargeting/run/`：生成入口（单文件 `retarget_*.py` / `refine_*.py` + 批量 `run_*_manifest.py`）
- `retargeting/evaluate/`：几何与物理评测（`evaluate_hand_manifest.py` 批量重放；`replay_{linker,xhand,wuji}_isaac.py` 单条；`compare_manifest_methods.py` 配对比较；`audit_stable_success.py` 稳定成功/trace 审计）
- `retargeting/configs/`：映射、解剖、向量、协议配置；`manifests/`：冻结数据清单
- 候选输出统一为 npy dict：`grasp_seqs (N,70,D)`（Linker D=12 `[腕6,主动6]`、XHand D=18 `[腕6,关节12]`、Wuji D=26）+ `source_trajectory_indices` + `retarget_method` + 参数元数据；批量续跑靠元数据核对。

### 工作约定

- **开发期不写注释/docstring**（用户最后统一要求生成文档时再补）；`test_documentation.py` 已退役为空测试。
- **预计 >3 分钟的命令整理成可复制命令交给用户在终端运行**，不后台轮询；<3 分钟的可直接跑；后台批量用 `nohup nice -n 10 bash script.sh > /dev/null 2>&1 &`。
- **新方法跑完一轮不直接抛弃**：先归因（接触步、关节剖面、loss 分解、配对得失），再优化 1–2 轮（权重、warm start、实现 bug），接近现有方法就有优化必要；确实无望才换下一个主流方法。
- 结果记入 `WORK_LOG.md`（追加式）；git 只在方法确定后提交。
- **用户决策节点必须停下来问**：方法选择、扩展到正式 1000、覆盖/删除冻结产物、对外提交。

### 最小跑法模板

```bash
# 单条生成（以 XHand 向量法为例）
cd /home/lekangwan/projects/DexGraspMotionChallenge2025-final-release
MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
retarget_research/retargeting/run/retarget_xhand_vectors.py \
--source retarget_research/reference/HandRetargetTask2026/scripts/data/sorting/seq41proc/ddg-gd_pliers_poisson_017.npy \
--output /tmp/xhand_vec_test.npy --trajectory-indices 2 --maxeval 50

# 单条物理重放
MPLCONFIGDIR=/tmp/matplotlib-retarget OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/home/lekangwan/miniconda3/envs/hand-retarget/bin/python \
retarget_research/retargeting/evaluate/replay_xhand_isaac.py \
--source <源npy> --target /tmp/xhand_vec_test.npy --output /tmp/eval_out \
--object-name ddg-gd_pliers_poisson_017 --source-index 2 --target-index 0

# 批量生成 / 评测 / 对比
retarget_research/retargeting/run/run_xhand_vector_manifest.py \
  --manifest retarget_research/retargeting/configs/linker_independent_validation_v1.json \
  --output-dir retarget_research/outputs/xxx --workers 4 --resume --maxeval 50
retarget_research/retargeting/evaluate/evaluate_hand_manifest.py \
  --hand xhand --manifest <同manifest> --target-dir <候选目录> \
  --output-dir <评测目录> --workers 4 --resume
retarget_research/retargeting/evaluate/compare_manifest_methods.py \
  --manifest <同manifest> --summary 基线名 <基线summary.json> --summary 新方法名 <新summary.json> \
  --output <对比.json>
```

## 2. 方法路线（按合适程度排序）

### 2.1 Linker O6（6 DOF，优先级最高）

**现状**：夹紧 v1 dev20 4/20；纯向量 1/20；向量+grip5 2/20；**向量+接触锚+grip5 4/20（与夹紧打平，配对 +3/-3，p=1.0）**。新方法族已追平现有方法，按用户约定继续优化。

1. **向量+接触锚+抓握偏置的调优（最高优先，代码已就绪）**
   - 实现：`run/retarget_linker_vectors.py`（`--contact-weight`、`--grip-flexion-weight` 开关），配置 `configs/linker_anydex_vectors_v1.json`（向量定义 + `grip_flexion_targets`），批量入口 `run/run_linker_vector_manifest.py`。
   - 已知调优轴：grip 权重（5.0 → 8/10）；接触权重（5.0 → 3/8）；grip 目标剖面（现 `[1.0,0.5,0.9,1.1,1.3,1.4]`，可参照夹紧法最终关节 `[1.16,0.58,0.94,1.26,1.46,1.54]` 微调）；warm start（首帧现从零位开始，可改为夹紧候选首帧）。
   - 失败归因线索：vec_grip5 仍丢 CoughDrops[5]/[30]（薄小物体）、Candle[2]；接触锚版新增 Saucepan[37]、Wingtip[11]。逐条看 `outputs/vector_methods_dev_screen/<dir>_evaluation/<obj>/source_*_physics.json` 的接触步与关节剖面。
2. **关节空间归一化映射（未实现，廉价下限）**
   - 想法：Shadow 22 关节按每指屈曲/侧摆归一化到 [0,1]，映射到 Linker 6 主动关节（拇指 2 + 四指 4，mimic 内部展开），不做优化。半小时可完成，回答"非优化自然手型"下限。参考 `run/retarget_linker_keypoints.py` 的 mimic 展开与 `initial_values()`。
3. **接触重合成（DexGraspNet 式，未实现）**
   - 想法：接触阶段放弃模仿源姿态，直接优化"目标手指尖到物体表面接触点 + 法向对齐 + 防穿透 + 摩擦锥力闭合"（`run/phase_contact.py` 已有 `friction_wrench_residual` 与表面 KD-tree 工具）。闭合段独立于源姿态，抬升段冻结抓形。对 6 DOF 手最有针对性（只优化"抓得住"而非"像"）。
4. **夹紧初始化 + 向量细化（未实现）**
   - 想法：以夹紧 v1 候选为初值，在闭合/抬升期用向量+接触锚目标做局部细化（类似 XHand 指腹细化相对官方的关系）。实现量小，可复用 `refine_linker_squeeze.py` 的参数传递模式。
5. **Residual RL 物理在环（最后手段）**
   - 想法：在运动学候选上加可学习残差，PPO 在 Isaac Gym 中直接优化抬升成功率。链路可参考 `advanced_policy/` 的 `evaluate_policy_isaac.py` 与训练入口。成本高（1–3 天），只在运动学方法全部天花板后启用。

### 2.2 XHand（12 DOF）

**现状**：官方 12/20、指腹 13/20、**纯向量 14/20**；50 类向量 30/50 vs 官方 27/50。接触锚在 XHand 上已负（真实接触权重下 8/20，+2/-7），**不要再在 XHand 上调高接触权重**。

1. **纯向量法正式化（已确认，等用户拍板）**：通过后重跑 XHand 正式 1000（见 §5）。
2. **向量法第 1 轮优化（优先）**：50 类丢失 3 条（Vase[2]、Palette[4]、Snowman[6]）——先归因（关节剖面 vs 官方解、接触步），再试：方向项权重 5→3/8、neutral 权重、首帧 warm start 从官方候选（现在从零位）。实现：`run/retarget_xhand_vectors.py` + `configs/xhand_anydex_vectors_v1.json`。
3. **抓握偏置移植（若 Linker 上有效）**：XHand 脚本目前没有 `--grip-flexion-weight`，可参照 Linker 版补上（目标剖面按 XHand 官方闭合姿态定）。
4. **关节归一化映射（未实现）**：同 Linker §2.1-2。
5. **Residual RL**：同 Linker §2.1-5。

### 2.3 Wuji（20 DOF）

**现状**：正式方法待用户决策（旧无约束 581/557 vs n005 431/421）。已负方法：功能向量 7/20、向量+接触 7/20、phase_open 11–12/20（手型完美但成功率降）。Wuji 的新方法优先级低。

1. **若用户回退旧方法**：无需新方法；把 n005/phase_open/向量族写成报告消融。
2. **若保留 n005**：可试"接触锚 + 抓握偏置"组合（Wuji 的向量+接触是"选接触区"版本且已负，但"源接触位置转移 + grip"未试过；实现参照 Linker 版，注意 Wuji 解剖边界 `configs/wuji_anatomy_coupled_v1.json` 必须保留）。

## 3. 关键实现细节

### 3.1 AnyDex 风格分指段向量（已实现，Linker/XHand）

- 目标函数（`retarget_{xhand,linker}_vectors.py` 的 `*VectorObjective`）：位置向量 Huber 匹配（掌心→指尖/中段，按两手零姿态长度比逐向量缩放）+ 方向向量单位匹配（中段→指尖，Huber δ=0.5，权重 5.0）+ 掌心位置锚（×1000）+ 中性正则 0.0025 + 上一帧时序 0.01。
- 优化：逐帧 SLSQP（nlopt），梯度经 PyTorch autograd；变量布局 XHand `[关节12,平移3,欧拉3]`、Linker `[关节6,平移3,欧拉3]`（mimic 在模型内展开）；首帧零位、后续上一帧 warm start；maxeval 50。
- 阶段相关项（`--contact-weight` >0 或 grip >0 时启用）：`phase_contact.build_phase_contact_plan` 推断 approach/close/lift；接触锚只在 close 期生效（源五指接触掩膜内的指尖世界位置拉目标指尖）；grip 偏置在 close+lift 期生效（单侧 `relu(target - joint)²`）。
- 已知坑：批量入口的 build_command 曾因 `return [` 死代码导致 contact/grip 参数未传入（已修复并回归验证）；`mapping_semantics` 必须存**关键点映射语义**（供 evaluate_linker_geometry 校验），不是向量语义名。

### 3.2 夹紧 v1（Linker 现有正式）

- `run/refine_linker_squeeze.py` + 配置 `configs/linker_o6_squeeze_v1.json`；机制：关键点基线 + 接触期渐进夹紧（接触启动距离、抬升阶段位移、PD 120/5）。

### 3.3 评测协议

- `evaluate/evaluate_hand_manifest.py`：`--hand {linker,xhand,wuji}`，默认 30cm 稳定协议、PD 120/5、20 Hz、60 Hz×3 步、hold 30 步；`--policy-trace-dir` 生成进阶策略 trace。
- 配对比较用 `compare_manifest_methods.py`（对齐同一 manifest 的 (object, source_index) 键，报告 +N/-M、净变化、McNemar exact p）。
- 稳定/运输审计用 `audit_stable_success.py`（从物理报告/ trace 重算，含手型解剖门；Wuji 的 quarantine 值在 `configs/stable_success_protocol_v2.json`）。

### 3.4 结果目录约定

- 新实验输出放 `retarget_research/outputs/<描述性名称>/`（不在 git 内）；候选目录 + `_evaluation` + `_analysis` 平行命名；对比 JSON 放同一层。
- 现有有效结果：`outputs/vector_methods_dev_screen/`（开发集首筛：`xhand_vec`、`linker_vec`、`*_grip5`、`*_contact*`、各 `_evaluation`、baseline 评测与对比 JSON）；`outputs/xhand_method_reassessment_v1/`（50 类确认）。

## 4. 接下来的规划

1. **用户决策**（阻塞）：Wuji 手型门回退与否；XHand 正式换向量法与否。
2. **XHand 正式 1000 重跑**（若换）：生成（`run_xhand_vector_manifest.py` + `manifests/formal_50c_100o_1000t_seed20260808.json`，约 1.5–2.5 小时）+ 评测 + trace + 审计。命令模板照 `run/run_wuji_formal_1000_v1.sh` 改 hand 即可。
3. **Linker 向量族调优**（§2.1-1）→ 若 dev20 超过 4/20 且 50 类确认 → 同样重跑正式。
4. **Wuji 手型门解禁**（若用户回退）：改协议配置后重跑 audit，旧方法 557 条 trace 直接成为专家数据。
5. **进阶策略重启**：三手专家数据齐后，按 `advanced_policy/` 既有链路（对象级 split → 三模型训练 → 500 条闭环评测）执行。
6. **报告收尾**：三手正式表（含新旧口径）、方法探索消融表（phase_open、功能向量、向量族、接触转移的得失与归因）、失败案例分类、渲染视频。

## 5. 参考与背景

- 调研结论与公式来源：`retarget_research/METHOD_REASSESSMENT.md`（功能向量/接触法协议与 Wuji 负结果）、`LITERATURE_REVIEW_ONE_SLIDE.md`；AnyDexRetarget 参考实现审计在 `reference/AnyDexRetarget/`（向量目标 + 方向项 + 捏合 α 混合 + mimic 支持，`optimizer/key_vector_optimizer.py` 可直接抄公式）；接触转移思路来自 Lakshmipathy et al., TOG 2025（物体上的接触点跨手不变）。
- 每日实验记录：`WORK_LOG.md`（搜索 "2026-08-17" 见最新两天）。
