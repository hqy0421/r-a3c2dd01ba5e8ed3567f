"""Hierarchical markdown report writer (层级报表)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _fmt(x, digits=4):
    if x is None:
        return "—"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(xf):
        return str(xf)
    return f"{xf:,.{digits}f}"


def _sci(x):
    try:
        return f"{float(x):.2e}"
    except (TypeError, ValueError):
        return str(x)


def _table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join([" --- "] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def write_report(out_dir, result: dict, hierarchy) -> Path:
    out_dir = Path(out_dir)
    lines = []
    ok = result["success"]
    lines.append("# 分表能耗层级预测报表")
    lines.append("")
    lines.append(f"- 生成时间: {result['generated_at']}")
    lines.append(f"- 总体状态: {'✅ 成功' if ok else '❌ 失败（见「失败明细」）'}")
    win = result.get("windowing")
    if win:
        lines.append(
            f"- 预测窗口: `{win['predict_window']}`（预测其下一期需求）；"
            f"训练窗口 {len(win['train_windows'])} 个"
        )
    lines.append("")

    # ------------------------------------------------------------------ #
    lines.append("## 方法")
    lines.append("")
    lines.append(
        "1. **先汇总、后提特征**：按声明树把同一窗口、同一时刻的叶子原始信号"
        "逐时刻汇总到各节点，再对每个节点的汇总序列实际提取 tsfresh 特征；"
        "叶子的非线性特征（如 abs_energy、c3、FFT）不能直接相加当节点特征。"
    )
    lines.append(
        "2. **每节点独立回归**：用本节点训练窗口特征与由叶子组成的下一期标签"
        "拟合独立回归器；新窗口沿用训练期拟合的预处理（中位数填补、训练方差"
        "过滤、训练均值/标准差标准化）。"
    )
    lines.append(
        "3. **全局协调**：以各节点实际基线预测 ŷ 为目标，按给定正权重 w 解"
        " `min Σ_n w_n·(S_n·x − ŷ_n)², x ≥ 0`（NNLS 精确全局最优，非简单裁零）；"
        "所有节点值由协调后的叶子汇总，层级天然一致且需求非负。"
    )
    lines.append("")

    # ------------------------------------------------------------------ #
    lines.append("## 声明层级树（* = 叶子表）")
    lines.append("")
    lines.append("```")
    lines.append(hierarchy.render())
    lines.append("```")
    lines.append("")

    # ------------------------------------------------------------------ #
    lines.append("## 数据与窗口")
    lines.append("")
    if win:
        lines.append(
            _table(
                ["项目", "值"],
                [
                    ["窗口长度（样本）", win["window_size"]],
                    ["完整窗口数", win["n_windows"]],
                    ["训练窗口", f"{win['train_windows'][0]} … {win['train_windows'][-1]}"],
                    ["预测窗口", win["predict_window"]],
                    ["丢弃的尾部零头行", win["dropped_trailing_rows"]],
                    ["叶子表", ", ".join(win["leaves"])],
                ],
            )
        )
    else:
        lines.append("数据未能通过载入校验，未形成窗口。")
    lines.append("")

    # ------------------------------------------------------------------ #
    agg = result.get("raw_aggregation")
    if agg and win:
        lines.append(f"## 原始汇总（预测窗口 `{win['predict_window']}`）")
        lines.append("")
        lines.append("叶子原始信号按同一时刻汇总到各节点后的序列统计：")
        lines.append("")
        rows = []
        for node in hierarchy.nodes:
            if node not in agg or win["predict_window"] not in agg[node]:
                continue
            s = np.asarray(agg[node][win["predict_window"]], dtype=float)
            rows.append(
                [node, len(s), _fmt(s.sum()), _fmt(s.mean()), _fmt(s.min()), _fmt(s.max())]
            )
        lines.append(_table(["节点", "样本数", "窗口总量", "均值", "最小", "最大"], rows))
        lines.append("")
        lines.append("完整逐时刻序列见 `raw_aggregation.csv`。")
        lines.append("")

    # ------------------------------------------------------------------ #
    feats = result.get("features") or {}
    if feats and win:
        lines.append(f"## 特征（预测窗口 `{win['predict_window']}`，各节点实际提取）")
        lines.append("")
        nodes = [n for n in hierarchy.nodes if n in feats]
        all_cols = sorted({c for n in nodes for c in feats[n].columns})
        rows = []
        for col in all_cols:
            row = [col]
            for n in nodes:
                v = feats[n].loc[win["predict_window"], col]
                row.append(_fmt(v, 4) if pd.notna(v) else "NaN")
            rows.append(row)
        lines.append(_table(["特征"] + nodes, rows))
        lines.append("")
        lines.append("全部窗口的特征矩阵见 `features/<节点>.csv`。")
        lines.append("")

    # ------------------------------------------------------------------ #
    base = result.get("base_predictions") or {}
    if base:
        lines.append("## 基线预测（各节点独立模型的实际输出）")
        lines.append("")
        rows = [
            [n, _fmt(result["train_r2"].get(n), 4), _fmt(base[n])]
            for n in hierarchy.nodes
            if n in base
        ]
        lines.append(_table(["节点", "训练 R²（样本内）", "基线预测 ŷ"], rows))
        lines.append("")

    # ------------------------------------------------------------------ #
    rec = result.get("reconciliation")
    if rec:
        lines.append("## 协调结果（非负叶子 + 全局最小加权平方改变量）")
        lines.append("")
        lines.append("### 叶子值")
        lines.append("")
        rows = [
            [l, _fmt(rec["base"][l]), _fmt(rec["leaves"][l])] for l in hierarchy.leaves
        ]
        lines.append(_table(["叶子", "基线预测", "协调值 x*"], rows))
        lines.append("")
        lines.append("### 各节点代价明细")
        lines.append("")
        rows = [
            [
                p["node"],
                _fmt(p["weight"], 3),
                _fmt(p["base"]),
                _fmt(p["reconciled"]),
                _fmt(p["delta"]),
                _fmt(p["weighted_sq_change"], 6),
                f"{100.0 * p['cost_share']:.1f}%",
            ]
            for p in rec["per_node"]
        ]
        lines.append(
            _table(
                ["节点", "权重 w", "基线 ŷ", "协调值", "改变量 Δ", "w·Δ²", "代价占比"],
                rows,
            )
        )
        lines.append("")
        clip = rec["clip_baseline"]
        lines.append(
            f"- **总代价 Σ w·Δ² = {_fmt(rec['total_cost'], 6)}**；"
            f"简单裁零基线总代价 = {_fmt(clip['total_cost'], 6)}"
            f"（全局最优 ≤ 裁零，差额 {_fmt(clip['total_cost'] - rec['total_cost'], 6)}）"
        )
        kkt = rec["kkt"]
        lines.append(
            f"- KKT 最优性诊断：驻点最大偏差 {_sci(kkt['stationarity_max_abs'])}，"
            f"对偶最大违例 {_sci(kkt['dual_max_violation'])}，"
            f"叶子最小值 {_fmt(kkt['primal_min_leaf'], 6)}（≥ 0）"
        )
        coh = rec["coherence"]
        lines.append(
            f"- 层级一致性：根节点 {coh['root']} = {_fmt(coh['root_value'])}，"
            f"叶子之和 = {_fmt(coh['leaf_sum'])}，"
            f"最大父子偏差 {_sci(coh['max_parent_child_violation'])}"
        )
        lines.append("")
    elif result.get("base_predictions"):
        lines.append("## 协调结果")
        lines.append("")
        lines.append("未执行协调（存在失败，见下）。")
        lines.append("")

    # ------------------------------------------------------------------ #
    if result["failures"]:
        lines.append("## 失败明细")
        lines.append("")
        rows = [
            [f["stage"], f["code"], f.get("node", "—"), f["message"]]
            for f in result["failures"]
        ]
        lines.append(_table(["阶段", "代码", "节点", "信息"], rows))
        lines.append("")
        lines.append("> 存在上述失败，本报表不得视为完整成功。")
        lines.append("")

    # ------------------------------------------------------------------ #
    lines.append("## 产物清单")
    lines.append("")
    for a in result.get("artifacts", []):
        lines.append(f"- `{a}`")
    lines.append("")

    path = out_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
