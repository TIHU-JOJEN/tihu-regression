# 鹈鹕回归项目说明

更新日期：2026-09-05

## 部署入口

- Streamlit 入口：`panel_app.py`
- 页面流程：`tihu_ui.py`
- 清洗、变量加工、模型估计与后续分析：`tihu_core.py`
- Stata/Python 复现包：`tihu_export.py`
- 合成数据测试：`test_tihu.py`
- 本机 Stata 对照测试：`verify_stata.py`

Streamlit Community Cloud 从仓库根目录运行 `panel_app.py`。更新依赖时同时修改 `requirements.txt`。

## 产品流程

1. 上传 DTA、CSV 或 XLSX。
2. 按顺序建立可编辑的数据清洗和变量加工步骤。
3. 用户确认数据结构、Y 的实际含义和模型。
4. 设置必选控制、候选控制、标准误及模型专属变量。
5. 用最终模型和所选标准误直接搜索控制组合。
6. 在选定结果的同一有效样本上继续机制、中介、调节、异质性和稳健性分析。
7. 下载项目配置或包含本次数据的完整复现包。

## 模型范围

- 横截面：OLS、Logit/Probit/LPM、有序模型、Poisson/负二项、Tobit、Heckman、PSM、IV/2SLS、Sharp/Fuzzy RD、ANOVA/交互效应、ESR。
- 面板：FE、RE、FE+RE、Pooled OLS、DID、PSM-DID，以及适合结果类型的基础模型。
- 时间序列：ARIMA、AR、MA、ARMA、VAR、GARCH，并提供动态机制和调节分析。
- ESR 首版范围：横截面、连续 Y、0/1 选择变量。网页与导出的 Stata 代码均使用同一高斯完整信息最大似然；ATT/ATU 使用 Delta 法。

## 关键规则

- 同名文件用 SHA-256 内容指纹识别，内容变化会建立新会话工作区。
- 面板重复 `ID+时间` 不自动取平均；要求用户先明确处理。
- 滞后和差分按精确时间键匹配，不把跨期缺口的上一行当作一期。
- 搜索、结果页和 Stata 导出使用同一个模型、样本和标准误设定。
- 最大搜索变量数：核心 X 8 个、候选控制 20 个；组合数由用户设置预算。
- 失败、未收敛或标准误不可计算的组合跳过，并汇总原因。
- 每个成功模型固定显示“继续分析”。可直接交互的模型使用交互项；ESR、PSM、IV、Fuzzy RD 等提供高低组效应差异检验。

## 数据安全

- 上传数据进入 Streamlit 服务器当前会话内存。
- 程序不主动把用户数据写入服务器磁盘，不向 AI 服务发送数据。
- “保存项目设定”不含数据行，并绑定原始文件哈希。
- “复现包”明确包含本次数据，仅在用户点击下载后由浏览器接收。
- 侧栏“清除本次数据和结果”清空当前会话状态。

## 验证

运行：

```bash
python3 -m unittest test_tihu -v
```

本机安装 Stata 时可运行：

```bash
python3 verify_stata.py
```

测试只生成合成数据。当前对照覆盖 OLS 普通/稳健/聚类、Logit、Poisson、FE、RE、DID、Tobit、Heckman、IV、Sharp/Fuzzy RD，以及 ESR 的 ATT/ATU 与稳健标准误。

## 回滚

新版前的线上版本是提交 `023b56a`。发生部署问题时可由该提交恢复；不要使用桌面其他目录中的旧副本覆盖部署入口。
