"""Model-specific instrument screening and linear IV estimation, without I/O."""
from dataclasses import replace
import warnings
import numpy as np
import pandas as pd
import pyhdfe
import statsmodels.api as sm
from scipy import stats
from linearmodels.iv import IV2SLS
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from tihu_core import Fit, frame, design, unique, binary, validate_panel, covariance_options, effect_row


def candidates(data, s, excluded=()):
    blocked = set(unique([s.y, s.entity, s.time, s.cluster, s.treatment, s.post, s.running, s.group]
                         +s.core+s.controls+list(excluded)))
    lineage = data.attrs.get("tihu_lineage", {})
    family = set([s.y]+s.core+lineage.get(s.y, [])+[source for c in s.core for source in lineage.get(c, [])])
    blocked.update(family)
    for col, sources in lineage.items():
        if family.intersection(sources):
            blocked.add(col)
    output = []
    for col in data:
        if col in blocked or col.startswith("__") or col.lower() in {"id", "pid", "编号", "序号", "code", "year", "年份"} or col.lower().endswith("_id"):
            continue
        values = data[col].dropna()
        if values.nunique() < 2:
            continue
        if not pd.api.types.is_numeric_dtype(values) and values.nunique() > 20:
            continue
        output.append(col)
    return output


def iv_arrays(data, s):
    if len(s.core) != 1 or not s.instruments:
        raise ValueError("当前 2SLS 需要一个内生 X 和至少一个工具变量")
    if set(s.instruments) & set(s.controls+s.core+[s.y]):
        raise ValueError("工具变量不能同时作为 Y、X 或结果方程的控制变量")
    if s.interactions:
        raise ValueError("含内生交互项的 IV 需要另配工具，本入口不自动沿用交互项")
    d = frame(data, s)
    x = design(d, s.controls, s.categorical)
    z = design(d, s.instruments, s.categorical, constant=False)
    endog = d[s.core].astype(float)
    y = d[s.y].astype(float)
    absorbed, removed = 0, []
    if s.entity:
        validate_panel(d, s.entity, s.time)
        effects = ([s.entity] if s.entity_effects else [])+([s.time] if s.time_effects else [])
        if effects:
            # Absorb all stages together and retain the full FE rank for covariance correction.
            ids = np.column_stack([pd.factorize(d[c])[0] for c in effects])
            algorithm = pyhdfe.create(ids, drop_singletons=False)
            absorbed = algorithm.degrees
            parts = [y.to_frame(), endog, x, z]
            widths = np.cumsum([part.shape[1] for part in parts])
            transformed = np.split(algorithm.residualize(np.column_stack(parts)), widths[:-1], axis=1)
            y = pd.Series(transformed[0][:, 0], index=d.index, name=s.y)
            endog = pd.DataFrame(transformed[1], index=d.index, columns=endog.columns)
            original_x = x
            x = pd.DataFrame(transformed[2], index=d.index, columns=x.columns)
            z_original = z
            z = pd.DataFrame(transformed[3], index=d.index, columns=z.columns)
            keep_x = np.linalg.norm(x, axis=0) > 1e-9*np.maximum(np.linalg.norm(original_x, axis=0), 1.)
            removed = list(x.columns[~keep_x])
            x = x.loc[:, keep_x]
            keep_z = np.linalg.norm(z, axis=0) > 1e-9*np.maximum(np.linalg.norm(z_original, axis=0), 1.)
            z = z.loc[:, keep_z]
    if z.empty or np.linalg.norm(endog.to_numpy()) < 1e-9:
        raise ValueError("工具变量或内生 X 被固定效应完全吸收")
    if len(d) <= x.shape[1]+z.shape[1]+absorbed+3:
        raise ValueError("有效样本不足以估计第一阶段")
    for matrix in [np.column_stack([x, z]), np.column_stack([x, endog])]:
        if np.linalg.matrix_rank(matrix) < matrix.shape[1]:
            raise ValueError("工具变量或解释变量在当前设定下完全共线")
    return d, y, x, endog, z, absorbed, removed


def linear_test(y, x, tested, s, d, absorbed=0):
    result = sm.OLS(y, x).fit(**covariance_options(s, d))
    df = len(d)-x.shape[1]-absorbed
    if df <= 0:
        raise ValueError("没有剩余自由度")
    cov = np.asarray(result.cov_params())*(len(d)-x.shape[1])/df
    indices = [x.columns.get_loc(col) for col in tested]
    b = np.asarray(result.params)[indices]
    vc = cov[np.ix_(indices, indices)]
    if np.linalg.matrix_rank(vc) < len(indices):
        raise ValueError("检验协方差不可逆")
    f = float(b @ np.linalg.solve(vc, b)/len(indices))
    denominator = d[s.cluster].nunique()-1 if s.se == "cluster" else df
    p = float(stats.f.sf(f, len(indices), denominator))
    return result, f, p


def first_stage(data, s, arrays=None):
    d, y, x, endog, z, absorbed, _ = arrays or iv_arrays(data, s)
    full = pd.concat([x, z], axis=1)
    result, f, p = linear_test(endog.iloc[:, 0], full, list(z), s, d, absorbed)
    reduced = endog.to_numpy().ravel()
    if x.shape[1]:
        reduced = reduced-x.to_numpy() @ np.linalg.lstsq(x, reduced, rcond=None)[0]
    partial = 1-float(np.sum(result.resid**2))/float(reduced @ reduced)
    if partial >= 1-1e-10:
        raise ValueError("第一阶段近乎完全拟合，可能是 X 的复制或确定性变换")
    kind = {"ordinary": "普通 F", "robust": "稳健 Wald F", "cluster": "聚类 Wald F"}[s.se]
    return {"第一阶段 F": f, "相关性 p值": p, "偏 R²": partial, "有效样本": len(d),
            "检验": kind, "通过初筛": bool(np.isfinite(f) and f > 10 and p < .05)}, result


def iv_candidates(data, s, excluded=()):
    rows = []
    for col in candidates(data, s, excluded):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                row, _ = first_stage(data, replace(s, instruments=[col]))
            rows.append({"变量": col, **row})
        except (ValueError, np.linalg.LinAlgError, ZeroDivisionError, PerfectSeparationError, FloatingPointError):
            continue
    columns = ["变量", "第一阶段 F", "相关性 p值", "偏 R²", "有效样本", "检验", "通过初筛"]
    return pd.DataFrame(rows, columns=columns).sort_values("第一阶段 F", ascending=False)


def esr_test(data, s, instruments):
    if s.entity:
        raise ValueError("ESR 识别变量筛选仅用于横截面选择方程")
    if not binary(data[s.treatment]):
        raise ValueError("ESR 的 D 必须同时包含 0 和 1")
    d = frame(data, replace(s, instruments=list(instruments), selection=unique(s.controls+list(instruments))))
    x = design(d, s.controls+list(instruments), s.categorical)
    z = design(d, instruments, s.categorical, constant=False)
    if min(d[s.treatment].value_counts()) < 10:
        raise ValueError("选择组或未选择组有效样本不足")
    kw = covariance_options(s, d)
    kw.pop("use_t", None)
    result = sm.Probit(d[s.treatment], x).fit(disp=False, maxiter=100, **kw)
    if not result.mle_retvals.get("converged") or not np.isfinite(result.bse).all():
        raise ValueError("Probit 选择方程未收敛或发生完全分离")
    restriction = np.zeros((z.shape[1], x.shape[1]))
    for i, col in enumerate(z):
        restriction[i, x.columns.get_loc(col)] = 1
    test = result.wald_test(restriction, scalar=True)
    p = float(test.pvalue)
    return {"选择方程 Wald χ²": float(test.statistic), "相关性 p值": p,
            "有效样本": len(d), "通过初筛": bool(np.isfinite(p) and p < .05)}


def esr_candidates(data, s, excluded=()):
    rows = []
    for col in candidates(data, s, excluded):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                row = esr_test(data, s, [col])
            rows.append({"变量": col, **row})
        except (ValueError, np.linalg.LinAlgError, ZeroDivisionError, PerfectSeparationError, FloatingPointError):
            continue
    columns = ["变量", "选择方程 Wald χ²", "相关性 p值", "有效样本", "通过初筛"]
    return pd.DataFrame(rows, columns=columns).sort_values("相关性 p值")


def fit_iv(data, s):
    arrays = iv_arrays(data, s)
    d, y, x, endog, z, absorbed, removed = arrays
    stage, first = first_stage(data, s, arrays)
    if s.auto_instrument and not stage["通过初筛"]:
        raise ValueError("当前控制组合的第一阶段未通过 F>10 且 p<0.05 初筛")
    kw = {"cov_type": "robust" if s.se == "robust" else "unadjusted", "debiased": True}
    if s.se == "cluster":
        kw = {"cov_type": "clustered", "clusters": pd.factorize(d[s.cluster])[0], "debiased": True}
    r = IV2SLS(y, x if x.shape[1] else None, endog, z).fit(**kw)
    k, n = len(r.params), len(d)
    df = n-k-absorbed
    if df <= 0:
        raise ValueError("固定效应和解释变量耗尽了自由度")
    se = r.std_errors*np.sqrt((n-k)/df)
    df_test = d[s.cluster].nunique()-1 if s.se == "cluster" else df
    quantile = stats.t.ppf(.975, df_test)
    table = pd.DataFrame({"term": r.params.index, "coef": r.params.values, "se": se.values,
                          "p": 2*stats.t.sf(np.abs(r.params/se), df_test),
                          "lower": (r.params-quantile*se).values, "upper": (r.params+quantile*se).values})
    details = {"第一阶段": pd.DataFrame([stage]), "说明": "F>10 是强度初筛，不是外生性或排除限制的证明。"}
    if absorbed:
        details["固定效应自由度"] = absorbed
        details["被吸收的外生项"] = "、".join(removed)
    try:
        augmented = pd.concat([x, endog, pd.Series(first.resid, index=d.index, name="__cf")], axis=1)
        _, f, p = linear_test(y, augmented, ["__cf"], s, d, absorbed)
        details["内生性检验（控制函数）"] = pd.DataFrame([{"F": f, "p": p}])
    except (ValueError, np.linalg.LinAlgError):
        details["内生性检验"] = "当前设定不可计算"
    if z.shape[1] > endog.shape[1]:
        if s.se == "cluster":
            details["过度识别检验"] = "聚类情形请使用导出代码的稳健过度识别检验"
        else:
            try:
                over = r.sargan if s.se == "ordinary" else r.wooldridge_overid
                details["过度识别检验"] = pd.DataFrame([{"检验": "Sargan" if s.se == "ordinary" else "Wooldridge score",
                                                                 "统计量": over.stat, "p": over.pval}])
            except (ValueError, np.linalg.LinAlgError):
                details["过度识别检验"] = "当前设定不可计算"
    else:
        details["过度识别检验"] = "恰好识别，无法进行过度识别检验"
    return Fit(s, table, d, effect_row(table, s.core[0]), details, x, y, r)


def iv_followup_spec(base, endog):
    supported = {"OLS", "Pooled OLS", "FE", "FE+RE", "LPM", "IV/2SLS"}
    if base.model not in supported or base.interactions or endog not in base.core:
        raise ValueError("本入口支持无交互项的线性基准；不会把 DID、RD、ESR 或非线性模型机械改成普通 2SLS")
    return replace(base, model="IV/2SLS", core=[endog],
                   controls=unique(base.controls+[c for c in base.core if c != endog]), instruments=[],
                   entity_effects=base.entity_effects if base.model in {"FE", "FE+RE", "IV/2SLS"} else False,
                   time_effects=base.time_effects if base.model in {"FE", "FE+RE", "IV/2SLS"} else False,
                   auto_instrument=True)
