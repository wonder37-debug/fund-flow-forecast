# 资金流入流出预测

机器学习课程实践项目。以阿里天池竞赛历史交易为基础，预测 2014 年 9 月全体用户每日申购和赎回总额，金额统一使用分。当前交付阶段一的数据审计、探索性分析和离线基线。

## 环境与数据

实验环境为 Windows 和 Python 3.13。以下 PowerShell 命令在项目根目录执行。

```powershell
uv venv --python 3.13 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

也可以用 `python -m venv .venv` 创建环境，再用 `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` 安装。

将自己取得的输入放在项目根目录下的 `Purchase Redemption Data/`，保留以下文件名。

| 文件 | 内容 |
| --- | --- |
| `user_balance_table.csv` | 用户每日余额及交易记录 |
| `user_profile_table.csv` | 用户资料 |
| `mfd_day_share_interest.csv` | 基金日收益率 |
| `mfd_bank_shibor.csv` | 银行间同业拆放利率 |
| `comp_predict_table.csv` | 无表头的 3 行提交格式示例 |

原始 CSV、Parquet 缓存、课程指导书和报告模板保存在本地。目录 `results/` 包含经检查的聚合证据、实验指标、图表和预测文件。[天池数据集入口](https://tianchi.aliyun.com/dataset/13319)

## 复现实验

```powershell
$python = '.\.venv\Scripts\python.exe'
& $python -m src.data.pipeline
& $python -m src.features.calendar
& $python -m src.eda
& $python -m experiments.baselines
& $python -m pytest -q
```

数据流水线先在 `data/processed/` 生成 L1 用户明细、L2 群体日序列和 L3 全体用户日序列。EDA 读取本地数据，回测模型只接收训练截止日及以前的日级目标与未来已知日期。默认在 2014 年 4—7 月选型、8 月固定评估，随后生成 30 天的 9 月基线预测。加权 MAPE 用于离线模型比较，官方积分映射尚未确认。

| 路径 | 内容 |
| --- | --- |
| `src/data/` | 原始数据校验、分层汇总与数据审计 |
| `src/features/`、`configs/` | 历史日历与因果时间特征 |
| `src/eda.py` | 实际数据探索和绘图 |
| `src/models/`、`src/evaluation.py` | 模型接口与时序回测 |
| `src/submission.py`、`experiments/` | 提交文件校验与可追踪实验入口 |
| `tests/`、`results/` | 正确性测试与可复现产出 |

## 实验材料

- [数据字典](docs/data_dictionary.md)、[数据质量审计](docs/data_audit.md)
- [EDA 报告](docs/eda_report.md)、[日历口径](docs/calendar.md)
- [评估协议](docs/evaluation_protocol.md)、[阶段一结果](docs/stage1_summary.md)
- [后续研究路线](docs/roadmap.md)
