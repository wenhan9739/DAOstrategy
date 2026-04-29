# Poker Exploit Agent — 6-max NLHE（BC + PPO 路线）

基于 `phh-dataset`（10k Pluribus 对局 + ~8M HandHQ 在线手牌）训练的 6-max
无限注德扑剥削型 AI。流程分两步：

1. **Behavior Cloning（BC）** — 用真实玩家数据训练一个"人类池"策略
   `pi_pool`，作为环境中的 5 个对手。
2. **PPO 剥削** — Hero 坐 1 个座，其余 5 座由冻结的 `pi_pool` 填充，用 PPO
   让 Hero 学会针对性剥削这批"平均水平"对手。

配套一个 Flask 网页端，可以作为真人和 5 个 BC bot 坐在一起打牌。

---

## 目录速查

```
poker_agent/
  data/       parser / replay / encoder / Dataset
  env/        6-max NLHE 模拟器（含 PokerVecEnv 批量环境）
  models/     BCModel（Transformer 主干） + PPO ActorCritic 包装
  training/   train_bc.py / train_ppo.py
  eval/       evaluate.py（mbb/hand + 95% CI）
  scripts/    CLI 入口
configs/      YAML 配置
  bc.yaml          旧版 BC（保留供对比）
  bc_v2.yaml       ★ 当前生产配置
  ppo.yaml         PPO 剥削训练
processed/    preprocess 生成的 .npz shard（约 3.7M × 10 = 37M 样本）
checkpoints/
  bc_full/best.pt           ★ 当前生产 BC（v2）
  bc_full/best_v1_backup.pt  旧 v1 备份
  bc_v2/                    v2 全部训练过程
  ppo_full/                 PPO 输出
webgame/      Flask 网页前端（HTML 内联）
```

---

## 数据流

```
.phh / .phhs 原始手牌
  ├─ filter_6max_nlhe.py   → 过滤纯 NL 德扑 6-max
  ├─ parser.iter_phhs      → HandRecord
  ├─ replay.replay_hand    → (state, action) DecisionSample
  └─ features.encode_sample
                            → shard_*.npz（定长张量）
                                    │
                    ┌───────────────┴──────────────┐
                    ▼                              ▼
            train_bc.py                  analyze_bc_selfplay.py
            (BC v2 Transformer)             (6-BC 自对战统计)
                    │                              │
                    └────────►  checkpoints/bc_full/best.pt
                                            │
                                            ▼
                                    train_ppo.py
                                (Hero + 5 BC 对手)
                                            │
                                            ▼
                                    webgame/app.py（人机对战）
```

---

## 快速上手

```bash
# 0) 依赖（CUDA torch 单独装）
pip install -r requirements.txt
pip install torch --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu121

# 1) 预处理：原始手牌 → .npz shard
python -m poker_agent.scripts.preprocess \
    --data-root d:/Poker/DAOstrategy/holdem6max \
    --out-dir   d:/Poker/DAOstrategy/processed \
    --workers   8

# 2) 冒烟测试（50 步，1% 数据）
python -m poker_agent.training.train_bc --config configs/bc_v2_smoke.yaml

# 3) 正式 BC v2 训练（4070S ~2.5 小时，val_acc ≈ 71%）
python -m poker_agent.training.train_bc --config configs/bc_v2.yaml

# 4) 分析 BC 自对战分布（2000 手 ~17 秒 CPU）
python analyze_bc_selfplay.py \
    --ckpt checkpoints/bc_full/best.pt \
    --n-hands 2000 --parallel 128 \
    --out stats_selfplay.json \
    --dump-first 20 --dump-out sample_bc_selfplay.txt

# 5) 把自对战分布 ↔ 训练分布做对比并生成 markdown 报告
python analyze_training_stats.py --out stats_training.json --max-shards 10
python compare_bc_stats.py \
    --train stats_training.json --selfplay stats_selfplay.json \
    --out bc_analysis_report.md

# 6) PPO 剥削训练（数小时到一天）
python -m poker_agent.training.train_ppo --config configs/ppo.yaml

# 7) 评估 Hero vs 冻结 BC pool
python -m poker_agent.eval.evaluate \
    --hero-ckpt checkpoints/ppo_full/best.pt \
    --opp-ckpt  checkpoints/bc_full/best.pt \
    --n-hands   200000

# 8) 网页对战（本机 http://127.0.0.1:5000）
python -m webgame.app --bc-ckpt checkpoints/bc_full/best.pt
```

---

## 核心设计

### 动作空间（10 个桶）

| idx | 名称 | 说明 |
|---|---|---|
| 0 | FOLD | 弃牌（仅在有需跟注时合法） |
| 1 | CHECK/CALL | 过牌或跟注 |
| 2..8 | RAISE 0.33 / 0.5 / 0.75 / 1.0 / 1.5 / 2.0 / 3.0 × pot | 翻后 pot-relative 加注 |
| 9 | ALL_IN | 全下 |

环境里会把 bucket 反解（`desnap_raise_to`）成真实筹码数，并在非法
bucket 上自动降级为 call。

### 状态特征（仅公共信息 + hero 手牌）

- 浮点：`pot_bb / to_call_bb / stack_bb / effective_stack_bb / SPR / 通话价 / num_active_ratio`（7 维）
- 整数：`street / actor_pos / num_players`
- 卡片 id：`board(5) + hole(2)`（52 = 未知/pad）
- 每座位活跃 mask（6 维 bool）
- 行动历史 token：`(actor_pos, street, action_bucket, raise_frac)` × 48

### 模型（`BCModel`）

Transformer Encoder，4 层 / `d_model=192` / 4 head，~1.8M 参数：

```
board_tok(5) + hole_tok(2) + global_ctx_tok(1) + hist_tok(48) → TransformerEncoder → pool global_ctx → {policy_logits(10), rfrac_pred, value}
```

`value_head` 在 BC 训练时不使用，留给 PPO 复用。

---

## BC v2 的改进

BC v1 跑通后，我们用"6 个 BC 互打 2000 手"和**训练数据的决策级分布**
做对比（见 `bc_analysis_report.md`），发现三个问题：

1. **Raise sizing mode collapse** — R0.5x 占所有 raise 的 48%，R1x 只有 6%
   （训练集 R1x 有 23%）。根因：CE 把 10 个桶当独立类，模型聚到中间最"安全"桶。
2. **BTN 开池率翻倍**（56% vs 训练 29%）—— 根因：`replay.py:_assign_positions(2)=[5,1]`
   把 heads-up 数据也映射到 BTN 位，污染了位置 5 的统计。
3. **策略熵仅 0.65 nats** —— softmax 过度集中。

v2 对应的修复：

| 问题 | 改动 |
|---|---|
| Raise mode collapse | `bc_loss` 改为 soft-label：raise bucket 2..8 按序数平滑，70% 中心 + 15%/15% 给 ±1 邻居，ALL_IN 独立 |
| softmax 过集中 | 额外 `label_smoothing=0.05` |
| HU 污染 BTN | dataset `only_num_players=6` 过滤（保留 78% 样本 / 37M） |
| card_emb 样本效率低 | dataset `suit_permute=True`，每个样本 4! 花色随机置换（×24 数据多样性） |
| ctx_proj hack | 拆出 `hist_rfrac_proj`，`ctx_proj` 不再塞 zero padding |

### 结果对比（2000 手 6-BC 自对战）

| 指标 | v1 | **v2** | 训练集 |
|---|---|---|---|
| policy entropy | 0.646 | **0.870** nats | — |
| overall action TV | 12.2% | **9.4%** | — |
| raise-bucket TV | 31.3% | **27.5%** | — |
| R0.5x 占比 | 48% | **31%** | 19% |
| BTN raise% (首次开池) | 56% | **47%** | 29% |
| River 动作 TV | 8.2% | **3.9%** | — |
| val_acc | 70.3% | **71.1%** | — |
| 训练步数 | 123k | **120k** | — |

---

## 分析工具

- `analyze_bc_selfplay.py` — 批量 6-BC 自对战，收集 per-street × per-pos
  动作分布、VPIP/PFR/3bet%、showdown 率、策略熵等。
- `analyze_training_stats.py` — 从 `.npz` shard 采样计算**同维度**的
  分布，用于 apples-to-apples 对比。
- `compare_bc_stats.py` — 两份 JSON → markdown 报告（含诊断 + 改进 roadmap）。
- `compare_v1_v2.py` — v1 ckpt 和 v2 ckpt 的关键数字并排打印。
- `dump_hands.py` — 把 Hero(ppo) vs BC pool 对局写成人类可读文本（用于抽查）。

---

## 硬件 & 时间预算

- 参考机：RTX 4070 SUPER 12GB / Win11 / CUDA 12.1 / PyTorch ≥2.4
- BC v2 全量训练：**~2.5 小时**（120k 步 @ 14 it/s，bf16 AMP，batch 1024）
- PPO 剥削训练：~数小时至 1 天（数据在 `runs/ppo_full/`）
- 2000 手 6-BC 自对战分析：**~17 秒** CPU（批量 128 并行手）

---

## 已知问题 / TODO

以下来自 v2 分析报告，按影响力排序：

- [ ] **SB 开池仍偏松**（v2 35% vs 训练 23%）：label smoothing 把概率微抬
- [ ] **All-in 率升至 8%**（v1 1.6%，训练 2.2%）：`raise_neighbor_weight`
      让 R3x ↔ ALL_IN 互相分享 mass，需要把 all-in 从序数平滑链里独立
- [ ] **DAgger 式状态扩充** — 治 BC 分布漂移的主药
- [ ] **牌力特征**（169-bucket / equity buckets）
- [ ] **RL 微调** — 已有 `train_ppo.py`，用 v2 BC warm-start
- [ ] 把分布分析脚本挂到 CI，每次新 BC ckpt 自动跑

详见 `bc_analysis_report_v2.md` 第 11 节。
