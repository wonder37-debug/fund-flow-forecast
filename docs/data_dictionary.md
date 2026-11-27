# 数据字典与分层产物

本地入口读取 `Purchase Redemption Data/` 中的五张 CSV，默认把 Parquet 写入 `data/processed/`，把聚合审计写入 `results/audit/`。金额以整数分存储，余额均值、分位数和人均金额仍以分计。外部收益和利率保留 CSV 数值尺度。

```powershell
python -m src.data.pipeline --raw-dir 'Purchase Redemption Data' --processed-dir data/processed --output-dir results/audit
```

## 原始表

### `user_balance_table.csv`

一行对应一个 `user_id, report_date`。`report_date` 从 `YYYYMMDD` 解析为 `datetime`。余额、流量和类别金额均可表示为整数分，类别金额使用可空整数保存。

| 列 | 含义 |
| --- | --- |
| `user_id` | 匿名用户键 |
| `report_date` | 记录日期 |
| `tBalance`、`yBalance` | 当日余额、前一日余额 |
| `total_purchase_amt` | 总申购，含 `direct_purchase_amt` 与 `share_amt` |
| `direct_purchase_amt` | 直接申购，含 `purchase_bal_amt` 与 `purchase_bank_amt` |
| `purchase_bal_amt`、`purchase_bank_amt` | 余额渠道、银行卡渠道转入金额 |
| `total_redeem_amt` | 总赎回，含 `consume_amt` 与 `transfer_amt` |
| `consume_amt`、`transfer_amt` | 消费、转出金额 |
| `tftobal_amt`、`tftocard_amt` | `transfer_amt` 的两个子渠道。按字段名解释为转至余额、转至银行卡，渠道业务细节有待源数据说明确认 |
| `share_amt` | 计入总申购的收益金额 |
| `category1` 至 `category4` | 消费类别金额。数据未提供类别名称，保留原编号和空值 |

### `user_profile_table.csv`

一行对应一个 `user_id`。`sex` 和 `city` 是原始整数编码，`constellation` 是字符串。连接以余额交易记录为左表，按 `user_id` 多对一合并。材料未给出编码到真实地名或其他业务标签的映射。缺少资料的交易仍保留，静态分群将相应缺失字段记录进空值组。

### `mfd_day_share_interest.csv` 与 `mfd_bank_shibor.csv`

两表的 `mfd_date` 均从 `YYYYMMDD` 解析为 `datetime`，按日期与 L3 左连接。

| 表 | CSV 数值列 |
| --- | --- |
| `mfd_day_share_interest.csv` | `mfd_daily_yield`（日收益指标）、`mfd_7daily_yield`（7 日收益指标） |
| `mfd_bank_shibor.csv` | `Interest_O_N`、`Interest_1_W`、`Interest_2_W`、`Interest_1_M`、`Interest_3_M`、`Interest_6_M`、`Interest_9_M`、`Interest_1_Y`，对应隔夜和各期限 SHIBOR |

缺少当日发布记录或空值的字段在 L3 中为缺失。源表未提供独立的发布时间戳，回测需按训练截止日限制可用历史。

### `comp_predict_table.csv`

无表头，CSV 的三列依次是预测日期 `YYYYMMDD`、总申购和总赎回，两项金额均采用整数分。`report_date,purchase,redeem` 为代码中的解析列名。原样例只有 2014-09-01 至 2014-09-03 的 3 行，提交要求覆盖 2014-09-01 至 2014-09-30 共 30 天，保持日期升序、逗号分隔和无表头。样例金额只用于检查格式，预测数据来自模型输出。

## 预处理产物

| 层级与文件 | 唯一键 | 内容 |
| --- | --- | --- |
| L1 `transactions.parquet` | `user_id, report_date` | 余额交易原始列、解析后的日期以及 `sex`、`city`、`constellation`。原始金额和缺失值原样保留 |
| L2 `group_daily.parquet` | `report_date, segment_type, segment_value` | 对 `sex`、`city`、`constellation` 分别按日聚合。`segment_type` 是分群字段名，`segment_value` 是该字段原编码的字符串。每种分群单独覆盖当日人群 |
| L3 `daily.parquet` | `report_date` | 全人群每日合计、人数、余额分布与日期匹配的外部数据。只包含原交易表存在记录的日期 |

L2 和 L3 的各余额、交易渠道和消费类别合计列保留源字段名。每列用当天该组的已观测值求和，整组该列全空时结果为空。缺少整日交易记录时保留日期缺口，`data_audit.json` 列出缺日。

| 派生列 | 定义 |
| --- | --- |
| `purchase`、`redeem` | `total_purchase_amt`、`total_redeem_amt` 的聚合别名，是模型目标。L3 按目标分别检查完整性，当天任一已记录用户的对应字段缺失时，该目标为空，原名字段保留已观察小计 |
| `observed_users` | 当天该组有记录的不同用户数 |
| `purchase_observed_users`、`redeem_observed_users` | 对应目标字段在当日有值的用户数，包含金额为 0 的用户 |
| `purchase_users`、`redeem_users` | 对应金额严格大于 0 的用户数。`purchase_users` 采用含收益的总申购口径 |
| `first_observed_users` | 第一次在交易历史出现的用户数。观测窗首日可包含此前已经存在的用户 |
| `purchase_per_active_user`、`redeem_per_active_user` | 对应金额的已观察小计除以当日该金额为正的用户数，分母为 0 时保留缺失 |
| `tBalance_mean`、`tBalance_median`、`tBalance_p90` 等 | L3 当天余额的均值、中位数、90% 分位数，分别对 `tBalance` 和 `yBalance` 计算 |
| 外部收益和 `Interest_*` | L3 按日对齐的原列，缺少匹配日期时保留缺失 |

同一用户可出现在不同 `segment_type` 中。核账按同一分群维度汇总 `segment_value`，分群维度之间不可相加。

### 截至指定日期的行为分群输入

`from src.data.pipeline import behavior_features_as_of` 可读取 L1 的 DataFrame 来生成用户特征。`behavior_features_as_of(transactions, "2014-08-31")` 使用截至 2014-08-31 当天的观测记录，每个在截止日及以前出现的用户一行。特征包括 `observed_days`、`first_observed_date`、`last_observed_date`、申购与赎回合计、对应已观测天数与正金额活动天数，以及 `tBalance_mean` 和最后一条用户记录的 `tBalance_last`。最后一条余额为空时，`tBalance_last` 保留空值。回测应传入该折已知历史的最后一天，再由该折的特征拟合分群规则。结果包含用户键，供本地使用。

原始 CSV 和三层 Parquet 均按 `.gitignore` 仅保留在本地。可提交的审计产物为表结构、覆盖统计、核账汇总和按日总人数，见[数据审计](data_audit.md)。
