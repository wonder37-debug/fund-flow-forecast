# 阶段三总结

阶段三以 427 天 L3 日级序列构造直接多步样本，用相同前向时间边界比较统计、Ridge、Huber、LightGBM、加法和倍率残差。开发阶段完成 **230 组**，冻结后 8 月评估完成 **24 组**，均无运行失败。按日型切换比阶段二等权融合改善 4–7 月平均加权 MAPE **0.99 个百分点**，普通日预测逐日相同。日型切换只改变法定放假和补班日期，保持了用户对普通日稳定性的偏好。 [研究设计](stage3_design.md) · [多步特征](stage3_features.md) · [实际回测](stage3_experiments.md)

| 方法 | 4–7 月加权 MAPE | 8 月加权 MAPE |
| --- | ---: | ---: |
| 阶段一 `B3` | 24.58% | 17.24% |
| 阶段二 `cal_workday` | 18.61% | 17.93% |
| 阶段二 `blend_equal` | 18.12% | 18.55% |
| **阶段三主模型 `route_special`** | **17.13%** | **18.55%** |
| 阶段三备选 `lightgbm_purchase_route_redeem` | 16.57% | 18.63% |
| 开发期最强双目标 ML 残差 `ridge_additive` | 19.44% | 18.31% |
| 开发期最强直接 ML `ridge_log_amount` | 23.92% | 15.96% |

备选的 16.57% 是开发结果之后拼合已存逐日预测得到的。普通日月间 MAPE 标准差为 4.12 个百分点，主模型为 3.77，且备选在两个更早月份和多个固定 30 天窗口的平均表现更弱。8 月没有放假和补班，主模型在该月不改变 `blend_equal` 的预测。8 月个别直接 ML 候选达到较低误差，跨窗口表现仍不够稳定。主备角色在 8 月评估前冻结。 [开发汇总](../results/metrics/stage3-development/selection_summary.csv) · [8 月结果](../results/metrics/stage3-holdout/fold_metrics.csv)

## 9 月预测

两份 CSV 各 30 行三列、无表头、金额为非负整数分。主模型在普通日复用等权融合，在 9 月 6–8 日中秋假期及 28 日补班复用 `cal_workday`。备选只在 26 个普通日修正申购，全部赎回预测与主模型相同。9 月 28 日两份预测均为申购 **2,826,154.48 元**、赎回 **3,147,396.58 元**。主备 9 月 29–30 日赎回一致，未根据单次历史补班加入总量修正。 [主预测](../results/submissions/stage3/stage3_primary.csv) · [备选预测](../results/submissions/stage3/stage3_backup.csv) · [特殊日期对照](../results/submissions/stage3/calendar_comparison.csv) · [文件指纹](../results/submissions/stage3/manifest.json)

当前唯一已报告的真实线上成绩仍是阶段二 `stage2_backup.csv` 的 **111.9770**。阶段三两份文件均未代为提交，线上成绩和提交时间保持空值。加权 MAPE、30% 以内比例和其他误差诊断用于离线比较，官方完整积分映射未知。 [线上提交记录](../results/submissions/online_submissions.json)

## 局限与阶段四

正式开发仍只有四个整月窗口，按截止日展开的训练行共享真实标签，2–3 月存在明显的分布变化。日型切换集中改善少数节假日，8 月无法验证其特殊日泛化。2013 年国庆前周日补班的转出有提前迹象，窗口合计的方向随参照改变。阶段四优先对消费、转出分别建模再加总，比较直接预测与分解预测的误差抵消，并保留事件窗口合计。 [补班分析](stage3_makeup.md) · [研究路线](roadmap.md)
