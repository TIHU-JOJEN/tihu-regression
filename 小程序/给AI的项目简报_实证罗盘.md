# 鹈鹕回归 — 项目简报（喂给任何 AI 即可接续工作）

## 一、项目是什么

一个面向中国经管/金融本研生的计量实证 Web 平台。用户上传数据 → 自动诊断数据结构 → 搜索最优控制变量组合 → 一键出中文三线表 + 可复现 Stata 代码。

**口号**：「我有数据要写实证章节」→ 60 秒出正确模型 + 可复现代码 + 中文三线表。

**定位**：填补 SPSSAU（不出口代码/无决策树）、Julius AI（美元定价/无中文论文规范）、ChatGPT（无工作流/不保证可复现）之间的缺口。

**当前阶段**：MVP v9，已部署 Streamlit Cloud，覆盖面板/横截面/时间序列，单人 Python 开发，零预算。

**部署地址**：`https://tihu-regression-m7qdmweg457db74g5mwnc9.streamlit.app/`
**GitHub 私仓**：`TIHU-JOJEN/tihu-regression`（代码不公开，Streamlit Cloud 部署时临时改 Public 拉取后改回 Private）

**名称变更**：之前叫"实证罗盘 Empirica"，已改为 **「鹈鹕回归」**。

---

## 二、竞品一句话

| 竞品 | 弱点 = 我们的机会 |
|------|-------------------|
| SPSSAU (spssau.com) | 广而不深，不导出 Stata 代码，无决策树，无三线表 |
| Julius AI | 美元定价要国际卡，无中文计量规范，无 DID/RDD 工作流 |
| ChatGPT/Claude 数据分析 | 缺决策树、三线表、方法论文本，不保证可复现 |
| JASP/jamovi | 桌面软件，不原生支持面板 FE/RE/DID/IV/Tobit |
| 经管之家代码包 (¥10-100) | 验证了付费意愿，用户就在 Stata 专版 |

**护城河**：决策树驱动模型推荐 + 可复现 Stata/R 代码 + 中文三线表 + 方法论段落模板 + 稳健性套餐。

---

## 三、技术选型与当前架构

### 全部 Python，零前端
- **框架**：Streamlit（100% Python，自动生成网页 UI，无需 HTML/CSS/JS）
- **核心库**：pandas、statsmodels（OLS/Logit/Probit/OrderedModel/Poisson/NegativeBinomial/ARIMA/VAR）、linearmodels（PanelOLS/RandomEffects/IV2SLS）、scipy、scikit-learn（PSM）、python-docx
- **不调用 Stata**：用 Jinja 模板生成 .do 文本，参数与 Python 模型同源；Python 结果是展示答案，.do 是复现交付物
- **运行方式**：本地 `streamlit run panel_app.py`，浏览器访问 `localhost:8501`
- **Streamlit 路径**：`/Users/hutianhe/Library/Python/3.9/bin/streamlit`

### 托管路线
- 当前：纯本地（localhost），数据不离开用户电脑
- 阶段 0：Streamlit Cloud / Hugging Face Spaces 免费部署（大陆 5-15s 加载），可用 **GitHub 私仓**（代码不公开，但网页链接可访问）
- 阶段 1（有付费用户）：香港 VPS（腾讯云/阿里云香港，¥30-60/月，无需备案）
- 阶段 2（有收入）：大陆阿里云 ECS + ICP 备案
- 收款绕过：爱发电（6% 费，月会员）+ 网站免费层（不处理支付 → 不需经营性 ICP）
- GitHub Student Pack：用 .edu.cn 邮箱申请，得 Copilot Student + $200 DigitalOcean

---

## 四、当前代码：`panel_app.py` v9

**文件位置**：`/Users/hutianhe/Desktop/panel_app.py`

**测试数据位置**：
- `/Users/hutianhe/Desktop/test_panel.csv`（50 个体 × 6 年 = 300 观测）
- `/Users/hutianhe/Desktop/test_cross.csv`（300 横截面观测）

**运行命令**：
```bash
/Users/hutianhe/Library/Python/3.9/bin/streamlit run /Users/hutianhe/Desktop/panel_app.py
```

### 4.1 功能流程（4 步）

**Step 1 · 上传**：支持 .dta / .csv / .xlsx。Session state 用文件名比较避免 `st.rerun()` 导致的 file_uploader 状态丢失。

**Step 2 · 数据结构诊断**：
- 自动猜个体 ID（nunique 适中 + 各值计数分布均匀）和时间列（1990-2030 数值列）
- 用户可选择面板/横截面/时间序列三种结构
- 面板：显示平衡/非平衡、N、T
- 时间序列：ADF 平稳性检验
- 横截面：观测数与数值列数

**Step 3 · 变量分配 & 模型选择 & 搜索**：
- 用户选 Y、核心 X（最多 8 个）、控制变量候选池（最多 20 个）
- 设最少/最多控制变量数（最少可为 0）
- **SE 智能推荐**：根据个体数自动推荐聚类/稳健/普通 SE，用户可选择 💡智能推荐 / 聚类 SE / 稳健 SE(HC1) / 普通 SE
- **核心X显著要求**：分别显著（各X独立搜索） / 同时显著（所有X联合搜索，按最弱|t|排序）
- 聚类变量可自由选择（个体ID/省份/行业等），不限于固定效应 ID
- **三级回退**（智能推荐/聚类SE模式）：clustered → robust → ordinary，聚类后不显著自动回退并黄字说明
- **自动检测 Y 类型**（`detect_y_type()`）：
  - binary：{0,1} 二元 → Logit/Probit/LPM
  - ordered：3-10 个连续整数 → Ordered Logit/Probit
  - count：非负整数 → Poisson/负二项
  - censored：>5% 在边界 → Tobit/OLS+稳健SE
  - continuous：其他 → OLS/FE+RE/Heckman/PSM/IV
- **分类变量自动处理**：object 列或低基数数值列（2-15 个唯一值）自动 `get_dummies(drop_first=True)` 转为虚拟变量，名称如 `region_中部`、`region_西部`，出现在变量选择池
- **模型选项按数据结构 × Y 类型动态生成**（见 4.2 模型表）
- 点击搜索后，枚举候选池所有有效组合，上限 10000 个随机抽样（seed 42）
- **面板 FE+RE**：每个组合跑完整 PanelOLS + RandomEffects + Hausman
- **其他模型**：OLS 快速筛选，选组合后再跑完整模型
- 按核心 X 的 |t| 排序，保留前 50，存入 session_state

**Step 4 · 选组合 → 完整结果 & 代码**：
- 排名表显示每个组合的控制变量、核心 X 系数(SE)、|t|、R²、N
- 下拉选择组合 → 即时显示完整回归结果（所有变量系数+标准误+显著性星号）+ 表注（N、R²、SE 类型）
- Stata .do 代码（中文注释）+ 下载按钮（.do + .csv）

### 4.2 当前支持的模型全集

| 数据结构 | Y 类型 | 模型 | 说明 |
|---------|--------|------|------|
| 面板 | 连续 | FE+RE（固定+随机效应）| PanelOLS + RandomEffects + Hausman + 智能SE（三级回退） |
| 面板 | 连续 | Pooled OLS（忽略面板结构）| 混合回归基线，不控制个体固定效应 |
| 面板 | 连续 | DID 双重差分 | 2×2 DID，自动创建 Post×Treat 交互项 |
| 面板 | 连续 | PSM-DID | 先匹配再双重差分（显示 PSM ATT） |
| 面板 | 连续 | 分组面板回归 | 按分类变量分组，每组分别 FE+RE+Hausman，结果并列 |
| 面板 | 二元 | Logit / Probit / Logit+Probit | 带个体+时间虚拟变量 |
| 横截面 | 连续 | OLS / OLS+稳健SE | OLS 带 HC1 稳健 SE |
| 横截面 | 连续 | Tobit（截断回归）| 自定义 MLE（BFGS+数值 Hessian） |
| 横截面 | 连续 | Heckman（样本选择）| 两步法：Probit 选择方程 + IMR 修正 |
| 横截面 | 连续 | PSM（倾向得分匹配）| Nearest-neighbor/kernel matching on propensity score |
| 横截面 | 连续 | IV/2SLS（工具变量）| linearmodels.IV2SLS + 第一阶段 F |
| 横截面 | 连续 | ANOVA/组间比较 | 分类变量的 F 检验 + 组均值表 |
| 横截面 | 连续/二元 | 交互效应（分类×连续）| 分类虚拟 × 连续 X 交互项 + 联合 F 检验 |
| 横截面 | 二元 | Logit / Probit / Logit+Probit / LPM | + 边际效应 Stata 代码 |
| 横截面 | 有序 | Ordered Logit / Ordered Probit | statsmodels OrderedModel |
| 横截面 | 计数 | Poisson / 负二项 | statsmodels GLM |
| 横截面 | 截断 | OLS+稳健SE / Tobit | 自动检测左截断点 |
| 横截面 | 任意 | PSM / IV/2SLS | 始终可选 |
| 时间序列 | — | ARIMA / AR / MA / ARMA / VAR / GARCH(1,1) | 独立路径，GARCH 需 `pip install arch` |

### 4.3 关键 Helper 函数

- **`tobit_mle(y, X, left, right)`**：scipy.optimize.minimize BFGS + 数值 Hessian 求 SE
- **`heckman_two_step(y, X, Z)`**：Probit 选择方程 → `φ(xβ)/Φ(xβ)` 修正 → OLS 第二阶
- **`psm_att(y, D, X, method, k, caliper)`**：sklearn LogisticRegression 倾向得分 + NearestNeighbors 匹配
- **`check_stationarity(series)`**：ADF 检验，返回统计量 + p + 是否平稳
- **`detect_y_type(y_series)`**：自动判断 binary/ordered/count/censored/continuous

### 4.4 关键设计决策（不可回退）

1. **禁止使用 `st.rerun()`**：会重置 file_uploader 状态导致上传文件丢失。依赖 Streamlit 按钮点击后的自然重渲染周期。
2. **File uploader 用文件名比较**：`st.session_state.df_name != uploaded.name` 避免每次 rerender 重读文件。
3. **Session state 命名用版本号**：如 `key='id7'`、`key='cx7'`，避免不同步骤的 widget 冲突。
4. **linearmodels 的 clusters 参数必须 numpy array**：`td.index.get_level_values(0).to_numpy()`。
5. **绝对不要遍历参数字符串**：之前 `for suf,pk,sk in [('FE','fe'),('RE','re')]` 把 'fe' 拆成 'f' 和 'e' 导致崩溃，已改为 `fe_r.params.get(v, np.nan)` 直接取值。
6. **二元 Y 面板模型用虚拟变量法**：linearmodels 无面板 Logit，用 statsmodels Logit/Probit + 个体虚拟变量 + 时间虚拟变量。
7. **搜索时全部跑完，选组合时不跑回归**：用户要求搜索阶段就完成 FE+RE+Hausman，排名表能看到每个组合的完整信息。
8. **分类变量自动转虚拟变量**：object 型或低基数数值列（2-15 unique）→ `get_dummies(drop_first=True)` → 合并到变量池。
9. **搜索用 OLS 快速筛选，面板 FE+RE 除外**：跨截面/时间序列模型在搜索时用 OLS 排序（快），选组合后才跑真正模型（慢但只跑一次）。
10. **Heckman IMR 用正确公式**：`φ(xβ)/Φ(xβ)`，不是 `resid_response`（那是广义残差，数值不对）。

### 4.5 Session State 变量

| 变量 | 类型 | 说明 |
|------|------|------|
| `df` | DataFrame | 上传的原始数据 |
| `df_name` | str | 文件名，用于判断是否新文件 |
| `data_type` | str | 'panel' / 'cross' / 'timeseries' |
| `diagnosed` | bool | 是否已完成数据结构诊断 |
| `id_col` / `time_col` | str | 个体/时间列名（面板） |
| `n_units` / `n_periods` | int | 个体数/时期数 |
| `balanced` | bool | 是否平衡面板 |
| `search_results` | dict | `{cx: [result_dict, ...]}` 每个核心 X 的搜索结果 |
| `_y` | str | 搜索时的 Y 列名 |
| `_y_type` | str | Y 类型：continuous/binary/ordered/count/censored |
| `_is_panel` | bool | 是否面板数据 |
| `_model` | str | 选中模型名 |
| `_sub` | DataFrame | 搜索时使用的子集数据 |
| `_df_aug` | DataFrame | 含虚拟变量的增强数据框 |
| `_group_var` | str | 分类变量模型的分组列 |
| `_did_var` / `_use_cl` | var/bool | DID 处理变量 / 是否聚类 |
| `_cluster_col` | str | 聚类变量名（可为 ID/省份/行业等） |
| `_se_mode` | str | SE 类型：cluster/robust/ordinary |
| `_use_rob` | bool | 是否使用稳健 SE |
| `_search_mode` | str | 分别显著 / 同时显著 |
| `_heckman_sel` | str | Heckman 选择变量 |
| `_iv_endog` / `_iv_insts` | str/list | IV 内生变量和工具变量 |

### 4.6 搜索结果数据结构（result_dict）

**OLS 搜索**：
```python
{'controls': tuple, 'n': int,
 'tstat': float, 'pval': float,
 'rsq': float, 'rsq_adj': float,
 'ols_params': {var_name: {'b': float, 'se': float}}}
```

**面板 FE+RE 搜索**：
```python
{'controls': tuple, 'n': int, 'n_units': int, 'n_periods': int,
 'fe_beta': float, 'fe_se': float, 're_beta': float, 're_se': float,
 'tstat': float, 'pval': float, 'fe_rsq': float, 're_rsq': float,
 'hausman_chi2': float|None, 'hausman_p': float|None, 'hausman_df': int,
 'cluster_ok': bool, 'se_used': str, 'fell_back': bool,
 'tstat_ord': float, 'tstat_rob': float|None, 'tstat_clust': float|None,
 'fe_params': {var: {'b': float, 'se': float}},
 're_params': {var: {'b': float, 'se': float}},
 'const_fe_b/se': float, 'const_re_b/se': float, 'Xv': [str]}
```

---

## 五、完整 MVP 愿景（来自设计手册）

### 决策树叶节点（★ = 已完成）

| 数据结构 × Y 类型 | 模型 | 状态 |
|-------------------|------|------|
| 面板 × 连续 | ★ FE vs RE + Hausman | 已完成 |
| 面板 × 连续 | ★ DID (2×2) | 已完成 |
| 面板 × 连续 | ★ 分组面板回归 | 已完成 |
| 面板 × 二元 | ★ Logit/Probit | 已完成 |
| 面板 × 政策冲击 | DID (Callaway-Sant'Anna, TWFE) | 待开发 |
| 面板 × 计数 | Poisson FE / 负二项 FE | 待开发 |
| 横截面 × 连续 | ★ OLS + 稳健SE | 已完成 |
| 横截面 × 连续 | ★ Tobit I (MLE) | 已完成 |
| 横截面 × 连续 | ★ Heckman 两步法 | 已完成 |
| 横截面 × 连续 | ★ PSM ATT | 已完成 |
| 横截面 × 连续 | ★ IV/2SLS | 已完成 |
| 横截面 × 连续 | ★ ANOVA/组间比较 | 已完成 |
| 横截面 × 二元 | ★ Logit/Probit/LPM | 已完成 |
| 横截面 × 有序 | ★ Ordered Logit/Probit | 已完成 |
| 横截面 × 计数 | ★ Poisson + 负二项 | 已完成 |
| 横截面 × 截断 | ★ Tobit / OLS+稳健SE | 已完成 |
| 横截面 × 分类 X | ★ 交互效应（分类×连续）| 已完成 |
| 时间序列 × 连续 | ★ ARIMA/VAR/GARCH | 已完成（GARCH 需 arch 包）|
| 所有 | ★ 分类变量→虚拟变量 | 已完成 |
| 横截面 × 删失 | Cragg 双栏 / Heckman | 待开发 |
| 面板 × 交错 DID | csdid/did_imputation 警告 | 待开发 |
| 横截面 × 久期 | Cox PH / Weibull AFT | 待开发 |
| 内生面板 | IV/2SLS 面板 / GMM | 待开发 |
| 空间面板 | SAR/SEM/SDM/SAC | 待开发（课件已提取）|

### P1-P6 页面规划（当前只有 P4 结果页）

| 页面 | 功能 | 状态 |
|------|------|------|
| P1 上传 | 手机登录、拖拽上传、隐私声明 | 未开始 |
| P2 诊断 | 结构识别、变量表、缺失率 | 已完成（简化版） |
| P3 推荐 | 决策树结果卡+白话理由 | 未开始（当前直接跳模型选择） |
| P4 结果 | 三线表、Stata代码、方法论段落、稳健性 | v7 已覆盖全模型家族 |
| P5 项目 | 历史列表、重下载、删除 | 未开始 |
| P6 定价 | 免费/Pro/终身 | 未开始 |

### 定价（来自手册）
- 免费：3 次/月，<5k 行
- Pro 月：¥19
- Pro 年：¥99（对标 SPSSAU ¥288）
- 终身：¥299 前 500 限量
- 毕业论文包：¥49/30 天
- 验证：先卖终身买断，50 人预付 = 有 PMF

---

## 六、16 周路线图（当前约在第 8-9 周水平）

已完成面板 FE/RE + DID + 横截面全模型 + 时间序列 + 分类变量支持 + 三线表 + Stata 代码生成。对标手册：
- W1-6 已完成（环境、上传、诊断、OLS、Logit、面板 FE/RE）
- W7-8 基本完成（面板深度、控制变量搜索、DID）
- W9 部分完成（Tobit/Heckman/PSM/IV 已做，决策树推荐器未做）
- 下一步：决策树推荐器、方法论段落、部署到 Streamlit Cloud

---

## 七、用户偏好与协作方式

1. **用户是零基础初学者**：只会打开 Python，所有概念（虚拟环境、localhost、session state）需要通俗解释。
2. **用户想要完整结果而非分步**：搜索时就把所有模型跑完，不要"选组合后再点一下跑回归"。
3. **用户关注中文论文规范**：三线表必须在表注含 N、R²、SE 类型、模型设定。
4. **用户数据绝不离开本地**：当前纯本地运行，未来部署后内存处理 24h 内删除。
5. **中文交流，代码注释中文**。
6. **不要擅自添加用户没要求的功能**。当前 MVP 就聚焦面板+横截面。
7. **不要用 emoji**。
8. **每次修改代码后提醒用户重启 Streamlit 测试**。
9. **用户视代码为知识产权**：部署用 GitHub 私仓，代码不公开，仅服务可公开访问。

---

## 八、已知问题与注意事项

1. 面板 FE+RE 搜索每组合跑完整模型 + Hausman，10000+ 组合可能需数分钟（已放弃两阶段加速，保持全量搜索精度）。
2. 二元 Y 面板模型用虚拟变量法，大 N 时虚拟变量过多可能内存溢出。
3. Hausman 检验可能在非正定协方差矩阵差时失败（返回 NaN）。
4. **SE 三级回退**：智能推荐/聚类SE模式下，clustered→robust→ordinary，回退时黄字警告并说明具体|t|值。
5. Stata 代码根据实际使用的 SE 类型自动匹配 vce(cluster)/vce(robust)/不加 vce。
6. 重复观测自动取均值，但会静默丢信息。
7. 学术诚信定位：辅助学习 + 可复现代码，避开"代跑/代写/包过"。
8. Tobit MLE 可能不收敛（BFGS 敏感），数值 Hessian SE 可能不稳定。
9. Heckman 两步法的排除变量选择较简单，用户需自行判断模型设定合理性。
10. GARCH 需要可选安装 `arch` 包，未预装时会提示。
11. PSM-DID 目前仅显示 PSM ATT，未做匹配后 DID 第二阶。

---

## 九、关键文件清单

| 文件 | 说明 |
|------|------|
| `/Users/hutianhe/Desktop/panel_app.py` | 当前唯一运行的 Streamlit 应用（v9） |
| `/Users/hutianhe/Desktop/test_panel.csv` | 面板测试数据（50×6=300） |
| `/Users/hutianhe/Desktop/test_cross.csv` | 横截面测试数据（300 观测） |
| `/Users/hutianhe/Desktop/小程序/计量实证平台_完整落地手册.md` | 完整产品手册（竞品、决策树、MVP规格、16周路线） |
| `/Users/hutianhe/Desktop/小程序/原型_决策树与五页草图(1)(1).html` | HTML 原型（决策树可视化 + P1-P5 页面草图） |
| `/Users/hutianhe/Desktop/小程序/ai聊天记录.docx` | 产品设计讨论（学术诚信、替代方案、定价） |
| `/Users/hutianhe/Desktop/小程序/给AI的项目简报_实证罗盘.md` | 本文档 |
| `/Users/hutianhe/Desktop/计量上课课件/` | 课程课件 PDF（7 个专题） |

---

## 十、下一步工作（待用户确认）

1. ~~部署到 Streamlit Cloud~~ 已完成：`https://tihu-regression-m7qdmweg457db74g5mwnc9.streamlit.app/`
2. ~~发给 10 个朋友测试，验证需求~~ 已发，收到首批反馈（速度/聚类/OLS/显著性），v8-v9 已修复
3. 决策树推荐器（约 200 行字典，根据数据结构×Y类型自动推荐模型）
4. 方法论段落模板（Jinja2 自动填空）
5. 扩展模型：DID Callaway-Sant'Anna、Poisson/NB 面板、Cox PH
6. 登录系统 + 项目历史页
7. GMM 与空间计量（已从课件提取内容，待实现）
