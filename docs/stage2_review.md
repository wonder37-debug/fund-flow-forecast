# 阶段二 PR #2 审查与验收

## 审查意见和修复

1. **8 月清单完整性。** 原导出接受清单内自带的产物哈希，缺少独立的预期值。正式入口现在从版本管理中的 `configs/stage2_release.json` 取得冻结方案和 8 月清单的 LF SHA256。两者来自先前单独提交的复验记录 `6596238`，运行时不会根据待校验文件生成预期值。通过这两项检查后，再检查候选、配置、源码、日历、环境、开发输入及切分，核对 31 日的 8 月训练输入指纹和指标、逐日预测、失败及分组记录。未知模型、重复和缺漏会被拒绝。完整性失败时不创建预测目录。[审查意见](https://github.com/wonder37-debug/fund-flow-forecast/pull/2#discussion_r4719153918)
2. **奇数等权样本误报。** 累计权重 `[1, 2, 3, 4, 5]` 的中点为 `2.5`，NumPy `left` 和 `right` 均返回索引 `2`。另发现两个实际数值边界，浮点累计舍入会制造假平分，分半后相加会使极小金额下溢。修复后精确累计二进制浮点可表示的权重，对金额中点只舍入一次。归一化后完全不可表示的微小权重按零处理。新增 14 项测试，统计模型专项共 29 项通过。[审查意见](https://github.com/wonder37-debug/fund-flow-forecast/pull/2#discussion_r4719153983)
3. **日期和范围。** 阶段二读入后统一日期精度，修复毫秒、微秒 Parquet 时间戳被误判缺日的问题。事件诊断现在在金额校验和指纹计算前限定开发范围。两处均通过针对性合成回归。

完整性检查可发现意外修改和来源不一致。信任根是已审查的 Git 提交及其中的发布指纹记录。能同时修改代码和所有信任锚的主体超出该机制的适用范围。

## 真实数据回放

从原始 CSV 重建 427 日，阶段一 B0–B5 共 30 组、阶段二开发 76 组、8 月 19 组和 9 个前向事件诊断均完成。除运行耗时、时间戳和版本指纹等元数据，重跑逐折分数、逐日预测和分组数值与原记录一致。95 组真实输入预测通过未来目标、人数、余额和利率扰动检查。主预测仍为 `blend_equal`，备选仍为 `cal_workday`。两份 9 月 CSV 及特殊日期对照均与首发逐字节相同。

原始 [开发记录](../results/metrics/stage2-development/freeze.json) 和 [8 月记录](../results/metrics/stage2-holdout/manifest.json) 保留在 `561c07c` 对应的源码历史。修复后包含源码指纹变化的 [新冻结记录](../results/metrics/stage2-review-development/freeze.json) 和 [新 8 月评估](../results/metrics/stage2-review-holdout/manifest.json) 由实际执行生成。新开发冻结提交 `8bd80d0` 在新 8 月评估前完成。

## 导出与后续研究

默认导出使用仓库中已固定的复验清单。独立重跑的清单有自己的版本、耗时和指纹。核验后先在独立记录中固定两个 LF SHA256，再在导出时明确传入。

```powershell
$python = '.\.venv\Scripts\python.exe'
& $python -m experiments.stage2_forecast --output-dir data/cache/stage2-review-check/september

# 下列变量从事先审核的独立记录取得
& $python -m experiments.stage2_forecast --freeze $freezePath --holdout $holdoutManifestPath `
  --expected-freeze-sha256 $approvedFreezeSha256 --expected-holdout-sha256 $approvedHoldoutSha256 `
  --output-dir data/cache/stage2-independent/september
```

阶段二等权融合弱化了节假日和调休修正。开发集 9 个假期日的合并加权 MAPE 从纯日历的 30.40% 升到融合的 42.32%。按日型调整权重、给趋势模型单独做日历修正和限定参数消融，列入[后续研究路线](roadmap.md)。
