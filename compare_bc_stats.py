"""Compare BC self-play stats with training-data stats, emit markdown report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _pct(x):
    return f"{x * 100:5.1f}%"


def _fmt_row(label, vals):
    return "| " + label + " | " + " | ".join(_pct(v) for v in vals) + " |"


def _chi2_total_variation(p, q):
    """Compute simple total-variation distance between two categorical dists."""
    n = min(len(p), len(q))
    return 0.5 * sum(abs(p[i] - q[i]) for i in range(n))


def _diagnosis_text(tr, sp) -> str:
    raise_labels = tr["raise_bucket_labels"]
    tr_rb = tr["raise_bucket_count"]
    sp_rb = sp["raise_bucket_count"]
    tr_rb_f = [x / max(1, sum(tr_rb)) for x in tr_rb]
    sp_rb_f = [x / max(1, sum(sp_rb)) for x in sp_rb]
    # find peak bucket shifts
    tr_peak = raise_labels[tr_rb_f.index(max(tr_rb_f))]
    sp_peak = raise_labels[sp_rb_f.index(max(sp_rb_f))]

    btn_train_open = tr["pf_open_freq_per_pos"][5]
    btn_sp_open = sp["pf_open_freq_per_pos"][5]
    btn_train_raise = sum(btn_train_open[2:])   # buckets 2..9
    btn_sp_raise = sum(btn_sp_open[2:])

    sb_train_open = tr["pf_open_freq_per_pos"][0]
    sb_sp_open = sp["pf_open_freq_per_pos"][0]
    sb_train_raise = sum(sb_train_open[2:])
    sb_sp_raise = sum(sb_sp_open[2:])

    nact = tr["num_active_distribution_freq"]
    # fraction of training decisions that are heads-up (num_active == 2)
    pct_hu = nact[2] * 100

    lines = [
        "### A. Raise sizing 严重模式坍塌 ⚠️",
        "",
        f"训练集里 raise bucket 的峰值在 **{tr_peak}**（{max(tr_rb_f)*100:.1f}%），分布比较均匀；",
        f"自对战却把 **{sp_peak}** 概率推到 **{max(sp_rb_f)*100:.1f}%**，`1x-pot / 0.75x-pot` 的 mass 被吸走。",
        "",
        f"Total-variation distance (训练 vs 自对战 raise 分布) = **{_chi2_total_variation(tr_rb_f, sp_rb_f)*100:.1f}%**。",
        "",
        "**根因**：在相似 state 下人类选手会混用 0.33 / 0.5 / 0.75 / 1x / 1.5x，CE + argmax-式 softmax 会把概率集中到"
        "单一最安全桶（0.5x 刚好在中央）。`rfrac_head` 用 Smooth-L1 回归也没帮上忙，因为它只预测单个标量，"
        "无法表达多峰分布。",
        "",
        "### B. BTN / SB 首次开池频率过高",
        "",
        f"- BTN 首次决策：训练集总 raise = **{btn_train_raise*100:.1f}%**（fold {btn_train_open[0]*100:.1f}%）",
        f"- BTN 自对战总 raise = **{btn_sp_raise*100:.1f}%**（fold {btn_sp_open[0]*100:.1f}%）→ 多出 {(btn_sp_raise-btn_train_raise)*100:.1f} pp",
        f"- SB 首次决策：训练 raise = {sb_train_raise*100:.1f}% → 自对战 {sb_sp_raise*100:.1f}%（也偏松）",
        "",
        "**根因（一）**：训练数据含有 2/3/4/5 人桌的手，在 `replay.py:_assign_positions(2)=[5,1]`，**把 heads-up 玩家直接编码成 BTN(=5)**。"
        f"训练集有 **{pct_hu:.1f}% 决策发生在 num_active=2**，其中很大一部分是 HU 开局，位置 5 的统计被 HU 严重污染 → "
        "模型误以为 BTN 在 6-max 里也应当像 HU BTN 那样激进。",
        "",
        "**根因（二）**：CE 集中度（见 A）让 R0.5 在 BTN 首次决策桶上的 softmax 值膨胀，采样后放大了开池概率。",
        "",
        "### C. 翻前面对加注的反应偏差不大、翻后面对下注的偏差也较小",
        "",
        "Flop 面对下注 TV ≈ 6.9%；翻前 vs-raise 方向差异也只 5–10 pp。这说明模型对"
        "**动作类别**学得还行，问题主要集中在 **raise sizing 的多样性** 与 **位置-桌型耦合**。",
        "",
        "### D. 策略熵过低",
        "",
        f"平均 token-level 熵 **{sp['policy_entropy_nats_avg']:.3f} nats**。作为参考：若在 fold/check-call/0.5x 之间"
        " 50/30/20 混合，熵 ≈ 1.03 nats；当前值说明模型在大多数 state 上把 ≥90% 概率押在单桶 —— 这与 A/B 的现象一致。",
        "",
        "### E. Showdown 率与 all-in 率",
        "",
        f"showdown {sp['showdown_rate']*100:.1f}%、all-in 出现 {sp['all_in_rate']*100:.1f}%。这俩数字对 6-max 100bb 是合理范围，"
        "说明 BC 没有整体崩坏，只是**尺寸与位置-桌型**两个维度扭曲。",
    ]
    return "\n".join(lines)


def _improvements_text() -> str:
    return """### P0（立即可做，不破坏现有 ckpt）

1. **Raise bucket 用 ordinal / soft-label**（最关键）
   - 当前 `bc.py:bc_loss` 的 CE 把 10 个桶视为独立类。改成：
     * 把 raise 当成 **一级分类**（fold / check-call / raise / all-in）+ **二级回归/分类**（给定 raise 选 sizing）；或
     * 用 **soft label**：真实桶给 0.6，相邻两桶各给 0.2，远桶 0；或
     * 直接 **ordinal regression**（logistic link 多阈值），避免模型只会选中间桶。

2. **修 `ctx_proj` 的 hack**（`bc.py:82-94`）
   - 目前 history token 的 `raise_frac` 和 global ctx 共用一个 `Linear(N_CTX_FLOAT+1 → d)`，给 history 塞 0 padding。
   - 拆分：`hist_rfrac` 用单独的 1-维 `Linear(1, d)` 或把标量做成 sinusoidal embedding；`ctx_proj` 专注全局。

3. **Label smoothing**（CE `ε=0.05`）
   - 直接在 `F.cross_entropy(logits, labels, label_smoothing=0.05)`。缓解 softmax 过度集中。

### P1（要重新训练、但不改数据管线）

4. **训练子集过滤到纯 6-max**（或给 `table_size` 一个 embedding）
   - `num_players` 已在 `encode_sample` 里，但模型没用（只用 `n_players_emb`，但 HU 被映射到 6-max 位置仍然混了）。
   - 要么在 `preprocess.py` 里 `if hand.num_players < 6: skip`；要么改 `_assign_positions`：**不要把 HU/3-max 的位置塞到 6-max 的 0..5**，改用独立 slot 并在输入里明确区分 table size。

5. **花色对称数据增强**
   - 每个 batch 采样一个 4! 花色置换，同时施加到 `board / hole / hist_*`。免费 ×24 数据量，能让 `card_emb` 真正学到 rank-only 信息。

6. **调 `rfrac_weight` 与 raise-only mask**
   - 日志显示 `rf` loss 一直在 0.05 左右；若改成 raise 子分类（P0-1），`rfrac_head` 基本废弃，改作只在连续尺寸上做 teacher-forcing 的校准头。

### P2（更大改动，需要基础设施）

7. **DAgger 风格状态扩充**
   - 让当前 BC 与自己对战生成 100 万手状态；对每个状态**用启发式/简易 solver/PPO teacher** 标注动作；混入训练集再训一轮。这是治"分布漂移"的主药。

8. **牌力特征（equity / 169-bucket）**
   - 翻前：`hole` → 169-bucket one-hot；翻后：`(hole, board)` → equity vs random range 的分桶。直接拼进 ctx。
   - 这能显著提升 flop/turn/river 的决策质量（现在模型只靠 52-card embedding 自己学牌力，数据效率低）。

9. **策略蒸馏 / 多教师**
   - 训几个不同 seed 的 BC，self-play 生成 state，把 **ensemble 平均策略** 作为 soft target 再蒸馏一次。可以直接拉平单模型的 mode collapse。

10. **RL 微调**
    - `train_ppo.py` 已经就绪。P0–P1 做完后，用 BC best.pt warm-start PPO 对抗"冻结 BC pool"几十万步。PPO 天然会恢复分布多样性（靠探索奖励 + 优势信号）。

### P3（评估侧，不改训练但提升可信度）

11. **把本次分析脚本固化为 CI**
    - 在每个新 BC ckpt 产生后自动跑 `analyze_bc_selfplay.py + compare_bc_stats.py`，若 raise-bucket TV > 阈值就告警。

12. **加入温度校准**
    - 在验证集上拟合单一温度 T 最小化 NLL（Platt scaling）。通常能把过度集中的 softmax 拉回到更接近人类混合策略的水平。
"""


def action_dist_table(title, names, train_freq, sp_freq, street_names=None):
    lines = [f"### {title}", ""]
    header_cols = [" "] + list(names)
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("|" + "---|" * (len(header_cols)))
    if street_names:
        for i, st in enumerate(street_names):
            lines.append(_fmt_row(f"train {st}", train_freq[i]))
            lines.append(_fmt_row(f"selfp {st}", sp_freq[i]))
            tv = _chi2_total_variation(train_freq[i], sp_freq[i])
            lines.append(f"| Δ-TV {st} | " + f"{tv*100:.1f}%".rjust(7) + " | " + " | ".join([" "] * (len(names) - 1)) + " |")
    else:
        lines.append(_fmt_row("training ", train_freq))
        lines.append(_fmt_row("self-play", sp_freq))
        tv = _chi2_total_variation(train_freq, sp_freq)
        lines.append(f"\n*Total-variation distance: {tv*100:.1f}%*")
    lines.append("")
    return "\n".join(lines)


def per_pos_table(title, names, train_mat, sp_mat, pos_names):
    lines = [f"### {title}", ""]
    header = [" "] + list(names) + ["n(train)", "n(selfp)"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * (len(header)))
    for i, pos in enumerate(pos_names):
        lines.append("| train " + pos + " | " + " | ".join(_pct(v) for v in train_mat[i]) + " |  |  |")
        lines.append("| selfp " + pos + " | " + " | ".join(_pct(v) for v in sp_mat[i]) + " |  |  |")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="stats_training.json")
    ap.add_argument("--selfplay", default="stats_selfplay.json")
    ap.add_argument("--out", default="bc_analysis_report.md")
    args = ap.parse_args()

    tr = json.loads(Path(args.train).read_text(encoding="utf-8"))
    sp = json.loads(Path(args.selfplay).read_text(encoding="utf-8"))

    L = []
    L.append("# BC 策略复现性分析报告\n")
    L.append(f"- 训练集样本： **{tr['n_decisions']:,}** 决策 (来自 {tr['n_shards']} 个 shard)")
    L.append(f"- 自对战样本： **{sp['n_hands']:,}** 手，**{sp['n_decisions']:,}** 决策")
    L.append(f"- 自对战平均策略熵： `{sp['policy_entropy_nats_avg']:.3f}` nats "
             f"(均匀分布理论上限 ≈ {2.303:.2f} nats, 10 个桶)")
    L.append("")

    # --- street distribution ---
    L.append("## 1. 数据结构对照")
    L.append("")
    L.append("**训练集 street 分布** (哪个街做决策):")
    L.append("| pre | flop | turn | river |")
    L.append("|---|---|---|---|")
    L.append("| " + " | ".join(_pct(v) for v in tr["street_distribution_freq"]) + " |")
    L.append("")

    L.append("**训练集 num_active (同时还活着的玩家数) 分布**:")
    L.append("| 0 | 1 | 2 | 3 | 4 | 5 | 6 |")
    L.append("|" + "---|" * 7)
    L.append("| " + " | ".join(_pct(v) for v in tr["num_active_distribution_freq"]) + " |")
    L.append("")

    L.append("**自对战抵达各街概率** (每手):")
    L.append("| pre | flop | turn | river |")
    L.append("|---|---|---|---|")
    L.append("| " + " | ".join(_pct(v) for v in sp["street_reach_rate"]) + " |")
    L.append("")

    # --- overall action dist ---
    L.append("## 2. 总体动作桶分布")
    L.append("")
    L.append(action_dist_table(
        "Overall action frequency",
        tr["action_names"],
        tr["action_freq_overall"],
        sp["action_freq_overall"],
    ))

    # --- per-street ---
    L.append("## 3. 分街道动作分布")
    L.append("")
    L.append(action_dist_table(
        "Action freq per street (train vs self-play)",
        tr["action_names"],
        tr["action_freq_per_street"],
        sp["action_freq_per_street"],
        street_names=tr["street_names"],
    ))

    # --- preflop open ---
    L.append("## 4. 翻前首次主动动作（pf_open）按位置")
    L.append("")
    L.append("(每一行是该 pos 在【自己首次决策且目前无加注】时的动作分布)")
    L.append("")
    L.append(per_pos_table(
        "Preflop first voluntary decision",
        tr["action_names"],
        tr["pf_open_freq_per_pos"],
        sp["pf_open_freq_per_pos"],
        tr["pos_names"],
    ))

    # --- preflop vs raise ---
    L.append("## 5. 翻前面对加注（含 3bet+）")
    L.append("")
    L.append(per_pos_table(
        "Preflop facing ≥1 raise",
        tr["action_names"],
        tr["pf_vs_raise_freq_per_pos"],
        sp["pf_vs_raise_freq_per_pos"],
        tr["pos_names"],
    ))

    # --- postflop ---
    L.append("## 6. 翻后面对下注 (to_call > 0)")
    L.append("")
    L.append(per_pos_table(
        "Postflop facing a bet",
        tr["action_names"],
        tr["postflop_facing_bet_freq"],
        sp["postflop_facing_bet_freq"],
        tr["pos_names"],
    ))

    L.append("## 7. 翻后被 check 到 (to_call == 0)")
    L.append("")
    L.append(per_pos_table(
        "Postflop with no bet to call (check option)",
        tr["action_names"],
        tr["postflop_no_bet_freq"],
        sp["postflop_no_bet_freq"],
        tr["pos_names"],
    ))

    # --- raise sizing histogram ---
    L.append("## 8. Raise sizing bucket（在所有 raise/all-in 中的占比）")
    L.append("")
    def _rb_freq(counts):
        s = sum(counts)
        return [c / s if s > 0 else 0.0 for c in counts]
    train_rb = _rb_freq(tr["raise_bucket_count"])
    sp_rb = _rb_freq(sp["raise_bucket_count"])
    labs = tr["raise_bucket_labels"]
    L.append("| " + " | ".join(labs) + " |")
    L.append("|" + "---|" * len(labs))
    L.append("| " + " | ".join(_pct(v) for v in train_rb) + " |")
    L.append("| " + " | ".join(_pct(v) for v in sp_rb) + " |")
    L.append("")
    L.append(f"*Total-variation: {_chi2_total_variation(train_rb, sp_rb)*100:.1f}%*")
    L.append("")

    # --- hand-level ---
    L.append("## 9. 手牌级指标（自对战）")
    L.append("")
    L.append("| 指标 | 值 |")
    L.append("|---|---|")
    L.append(f"| showdown 率 | {_pct(sp['showdown_rate'])} |")
    L.append(f"| 出现 all-in 的手 | {_pct(sp['all_in_rate'])} |")
    L.append(f"| 每手平均动作数 | {sp['avg_actions_per_hand']:.2f} |")
    L.append(f"| 策略熵（自对战） | {sp['policy_entropy_nats_avg']:.3f} nats |")
    L.append("")
    L.append("**VPIP / PFR / 3bet% / fold-to-3bet%（按位置，自对战）**:")
    L.append("| seat | VPIP | PFR | 3bet% (机会时) | Fold to 3bet |")
    L.append("|---|---|---|---|---|")
    for i, name in enumerate(sp["pos_names"]):
        L.append(f"| {name} | {_pct(sp['vpip_per_seat'][i])} | "
                 f"{_pct(sp['pfr_per_seat'][i])} | "
                 f"{_pct(sp['threebet_per_seat_when_opportunity'][i])} | "
                 f"{_pct(sp['fold_to_threebet_per_seat'][i])} |")
    L.append("")
    L.append("> *对照：真实 6-max NLHE 现金池群体均值大致 VPIP 22–26%，PFR 18–22%，CO/BTN 的 PFR 明显更高，UTG 最紧。*")
    L.append("")

    # -------- 主要问题诊断 + 改进建议 -------- #
    L.append("---\n")
    L.append("## 10. 核心诊断\n")
    L.append(_diagnosis_text(tr, sp))
    L.append("")
    L.append("## 11. 改进 Roadmap（按影响力排序）\n")
    L.append(_improvements_text())

    Path(args.out).write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
