# 阶段三正式开发回测

代码 `87b2021` 在 2014 年 4–7 月四个整月开发窗口、2–3 月两个较早整月窗口以及四个固定 30 天窗口执行 **23 个候选 × 10 折，共 230 组，0 个运行失败**。固定 30 天从 3 月 15 日、5 月 15 日、6 月 15 日、7 月 1 日的历史截止点开始，窗口可与主协议重叠，单独使用。整个开发流程的输入范围截止 2014-07-31。第二次讨论并提交冻结方案后才执行 8 月正式评估。 [实验 Manifest](../results/metrics/stage3-development/manifest.json)

## 完整开发结果

下列指标对 4–7 月四折等权平均。普通日严重误差率先按目标算相对误差超过 30% 的日期比例，再按 45/55 加权。其余指标、逐日方向、日型和跨度单列记录。 [逐折](../results/metrics/stage3-development/fold_metrics.csv) · [逐日](../results/metrics/stage3-development/daily_predictions.csv) · [汇总](../results/metrics/stage3-development/selection_summary.csv) · [图表](../results/figures/stage3_development_comparison.png)

| 候选 | 加权 MAPE | 普通日严重误差率 |
| --- | ---: | ---: |
| `route_special` | **17.13%** | **14.60%** |
| `blend_equal` | 18.12% | 14.60% |
| `cal_workday` | 18.61% | 17.78% |
| `ridge_additive` | 19.44% | 20.22% |
| `ridge_ratio_less_regularized` | 19.48% | 19.34% |
| `lightgbm_ratio` | 19.51% | 17.79% |
| `ridge_ratio` | 19.77% | 19.59% |
| `ridge_ratio_external` | 20.09% | 20.96% |
| `ridge_ratio_business` | 20.32% | 23.22% |
| `trimmed_weekday` | 20.80% | 16.91% |
| `ridge_ratio_stronger_adjustment` | 20.90% | 21.55% |
| `ridge_ratio_long` | 22.51% | 29.13% |
| `ridge_log_amount` | 23.92% | 29.58% |
| `B3` | 24.58% | 27.32% |
| `ridge_amount` | 24.94% | 31.82% |
| `lightgbm_relative_mae` | 26.58% | 34.03% |
| `ridge_trend` | 31.59% | 39.91% |
| `ridge_rolling` | 31.85% | 35.78% |
| `ridge_stat_features` | 32.88% | 42.27% |
| `lightgbm_direct` | 33.35% | 36.81% |
| `huber_direct` | 35.88% | 48.57% |
| `ridge_lags` | 36.69% | 48.61% |
| `ridge_calendar` | 38.29% | 52.66% |

## 观察和消融

- **按日型切换保留了普通日期的表现。** `route_special` 4–7 月为 13.31%／17.57%／21.02%／16.61%。相对 `blend_equal`，全部日期 30% 以内比例由 81.57% 提高到 82.75%，普通日严重误差率均为 14.60%。切换方案在补充固定 30 天窗口平均为 21.08%，2–3 月为 54.25%。同一数据口径下 `cal_workday` 分别为 23.11% 和 50.52%，早期阶段切换的优势不稳定。
- **纯 ML 从日历和近期历史中学到一部分规律，跨月风险仍高。** 相对尺度的 Ridge 从仅日历的 38.29% 改善到加滞后的 36.69%，再到加滚动统计的 31.85% 和加趋势的 31.59%。对同一组特征，用对数金额得到 23.92%，原金额为 24.94%。LightGBM 相对误差加权 MAE 优于普通平方损失的版本（26.58% 对 33.35%）。直接把阶段二预测当作附加特征使 Ridge 从 31.59% 升到 32.88%。
- **先用统计预测再修正，效果优于直接 ML。** Ridge 加法残差为 19.44%，对数倍率为 19.77%。调整窗口从 120 天到 180 天，误差升至 22.51%，扩大残差修正力度也使误差升至 20.90%。从对数倍率的 19.77% 加入历史渠道和人数后为 20.32%，再加外部变量为 20.09%。这两组特征暂缺跨月收益。现有残差模型的主要退化来自赎回和普通周末。 [日型](../results/metrics/stage3-development/day_type_metrics.csv) · [跨度](../results/metrics/stage3-development/horizon_metrics.csv)
- **LightGBM 普通日残差模型的改善集中在申购。** `lightgbm_ratio` 的申购 MAPE 为 13.89%，优于 `route_special` 的 15.13%，但赎回从 18.76% 升到 24.10%。28 日申购水平比在该树模型申购端、赎回端分别占平均 gain 的约 58% 和 70%，可能反映模型较依赖历史阶段差异；该统计只描述模型如何分裂。 [特征重要性](../results/metrics/stage3-development/feature_importance.csv)

在完成上述比较后，额外用已存逐日预测回算了一个分目标组合，只有申购采用 `lightgbm_ratio`，赎回保留 `route_special`。4–7 月加权 MAPE **16.57%**，普通日严重误差率 **13.45%**，普通日月间 MAPE 标准差 **4.12 个百分点**（纯日型切换为 **3.77**）。四个固定 30 天窗口平均 **22.09%**，2–3 月 **60.14%**，两项都高于纯日型切换（21.08%、54.25%）。这是根据开发结果提出的备选假设，初轮配置、指标表和暂定统计方案均保留。

## 8 月固定评估

`f7a4802` 在运行留出评估前固定了主备角色和 24 个候选。使用至 7 月 31 日的已知历史一次性预测 8 月 31 天，共完成 **24 组，0 个运行失败**。此月全为普通日期，`route_special` 与 `blend_equal` 逐日相同。申购增强备选在 8 月的申购 MAPE 为 **14.90%**，统计主模型为 **14.73%**，开发集的申购改善未复现。几个直接 Ridge 在 8 月较强，`ridge_trend` 达到 **15.83%**，4–7 月为 **31.59%**。该结果保留用于理解阶段差异，发布模型沿用事先冻结的角色。 [全部 8 月指标](../results/metrics/stage3-holdout/fold_metrics.csv) · [8 月逐日预测](../results/metrics/stage3-holdout/daily_predictions.csv) · [8 月 Manifest](../results/metrics/stage3-holdout/manifest.json)

## 复验

Windows 和 Python 3.13.5 实际运行 **192 项**测试。开发流程从同一数据和配置再次运行，6,923 条模型逐日预测、逐折指标、分组结果及特征重要性逐值一致，耗时字段按实际运行分别记录。另取 4 月和 7 月两折作真实数据扰动，修改相应截止日之后 122 天和 31 天的申购、赎回、观察及交易人数、银行渠道、转出、余额、基金收益率和隔夜利率。重新构造特征和拟合全部 23 个候选，**46 个拟合预测逐值一致**。阶段二的四个统计对照在开发和 8 月的 **612 条**记录上逐值复现，原有两份 9 月预测 CSV 仍与发布文件逐字节一致。向发布输入追加含错误业务字段的 30 条 9 月模拟记录，阶段三两份 CSV 和日型对照逐字节一致。修改 8 月结果文件和清单内的文件哈希，独立固定的清单指纹仍会拒绝导出。

```powershell
$python = '.\.venv\Scripts\python.exe'
uv pip install --python $python -r requirements-stage3.txt
& $python -m experiments.stage3 develop --output-dir data/cache/stage3-repro/development
& $python -m experiments.stage3_figures --metrics data/cache/stage3-repro/development --output data/cache/stage3-repro/development.png
& $python -m experiments.stage3_makeup --output-dir data/cache/stage3-repro/makeup
& $python -m experiments.stage3_release holdout --output data/cache/stage3-repro/august
& $python -m experiments.stage3_release forecast --output data/cache/stage3-repro/september
& $python -m pytest -q
```

`holdout` 的重跑清单包含新运行时间和版本。默认 `forecast` 使用提交中的正式 8 月证据。独立复验要生成自己的发布产物时，先审核并固定冻结与重跑清单的 LF SHA256，再同时通过 `--expected-freeze-sha256` 和 `--expected-holdout-sha256` 传入，并用 `--holdout` 指向重跑清单。指纹用于验证实验发布的完整性与一致性，信任来源是已审核的 Git 证据。
