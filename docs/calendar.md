# 2013—2014 年历史日历

`configs/holiday_calendar.json` 按国务院办公厅年度通知录入全国放假区间和补班日。项目输出覆盖 2013-07-01 至 2014-09-30，内部额外保留至 2014-10-07，以便计算 9 月末到国庆假期的距离。

- 2013 年通知发布于 2012-12-08，项目范围内包含中秋节、国庆节及相应调休。[国务院办公厅关于2013年部分节假日安排的通知](https://www.sport.gov.cn/bgt/n4962/c649722/content.html)
- 2014 年通知发布于 2013-12-11，2014-09-28（周日）为调休工作日，2014-09-06 至 09-08 为中秋连休，10-01 至 10-07 为国庆连休。[国务院办公厅关于2014年部分节假日安排的通知](https://www.sport.gov.cn/bgt/n4961/c644255/content.html)
- `is_statutory_holiday` 表示法定节日当日或法定天数，`is_holiday` 表示通知公布的整个连休区间。2014 年春节法定日为正月初一至初三。[国务院令第644号](https://www.ln.gov.cn/web/zwgkx/lnsrmzfgb/2014n/qk/2014n_dyq/gwywj/C02298C830654E08B8CDFC5E6A46E0D8/index.shtml)

`is_weekend` 仅根据星期判断，`is_regular_weekend` 排除假期与补班周末，`is_workday` 包括补班。节前、节后窗口分别表示距下一个假期的开始、上一个假期的结束 1–3 个自然日。假期内两个距离均为 0，超出配置区间的假期距离保留为空。月末日期可以同时处于国庆前窗口和调休工作日。

双11、双12、618按固定公历日标记，表示分析事件日期，不代表平台实际促销起止，也不纳入官方假期定义。日历只需要日期和当时已公布的安排，适用于 2014 年各回测窗口和 9 月预测。每次回测的 B0–B5 只使用日期中的星期信息。

```powershell
python -m src.features.calendar
```

生成 `results/calendar/historical_calendar.csv`。
