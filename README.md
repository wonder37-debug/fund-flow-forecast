# 资金流入流出预测

基于阿里天池竞赛数据，预测每日资金申购和赎回总量。

## 本地数据

原始数据保存在项目根目录的 `Purchase Redemption Data/` 中。请将下列 CSV 放在该目录下，并保留原文件名。

| 文件 | 内容 |
| --- | --- |
| `user_balance_table.csv` | 用户每日余额及交易记录 |
| `user_profile_table.csv` | 用户资料 |
| `mfd_day_share_interest.csv` | 基金日收益率 |
| `mfd_bank_shibor.csv` | 银行间同业拆放利率 |
| `comp_predict_table.csv` | 竞赛预测提交模板 |

原始数据和预处理缓存仅在本地使用。经过检查的聚合统计、实验指标和预测 CSV 可以按需纳入版本管理。

## 课程报告

`ml_report.tex` 是课程实验报告模板。课程实践指导书保存在本地。
