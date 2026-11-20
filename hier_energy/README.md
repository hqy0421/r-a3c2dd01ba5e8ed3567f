# hier_energy — 分表能耗层级预测

基于 tsfresh 的层级能耗预测流程：总表等于下属分表之和，需求不为负。

## 流程

1. **先汇总、后提特征**（`windows.py` / `features.py`）：按声明树把同一窗口、
   同一时刻的叶子原始信号逐时刻汇总到各节点，再对每个节点的汇总序列实际提取
   tsfresh 特征。叶子的非线性特征（`abs_energy`、`c3`、FFT 等）不能相加当节点
   特征 —— `abs_energy(a+b) != abs_energy(a)+abs_energy(b)`。
2. **每节点独立回归**（`model.py`）：本节点训练窗口特征 + 由叶子组成的下一期
   标签（叶子下一窗口总量之和）拟合独立回归器；预处理（中位数填补、训练方差
   过滤、训练均值/标准差标准化）只在训练窗口上拟合，预测新窗口时沿用。
3. **全局协调**（`reconcile.py`）：以各节点实际基线预测 ŷ 为目标，按给定正
   权重 w 解 `min Σ_n w_n·(S_n·x − ŷ_n)², x ≥ 0`（S 为层级求和矩阵），用
   `scipy.optimize.nnls` 求得全局最优 —— 不是简单裁零。所有节点值由协调后的
   叶子汇总，层级一致且非负。
4. **诚实报告**（`pipeline.py` / `report.py`）：缺叶（`missing_leaf`）、未声明
   表计（`unexpected_meter`）、人口错位（`population_misalignment`）、拟合失败
   （`fit_failure`）、求解失败（`solve_failure`）等都会记录并使总体状态为失败，
   绝不假报完整成功。

## 运行

```bash
python -m hier_energy.demo --out /workspace/hier_energy_output --also-failure
python -m pytest hier_energy/tests -q
```

## 输入

- 长表数据：`time`, `meter`, `value` 三列；所有叶子表须共享同一时间网格。
- 声明树：`{"total": ["building_a", ...], "building_a": ["floor_a1", ...], ...}`。
- 每个节点的正权重。

## 输出（`out_dir`）

- `report.md` — 层级报表（原始汇总、特征、基线预测、协调值、代价明细、KKT
  与一致性校验、失败明细）
- `status.json` — 总体成功标志与失败列表
- `raw_aggregation.csv` — 各窗口各节点的原始汇总序列
- `features/<节点>.csv` — 各节点实际提取的特征矩阵（全部窗口）
- `labels_train.csv` — 训练标签
- `base_predictions.json` — 各节点基线预测与样本内 R²
- `reconciliation.json` — 协调叶子值、节点值、逐节点代价明细、总代价、
  裁零基线对比、KKT 诊断、一致性校验
