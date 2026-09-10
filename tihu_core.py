"""Shared estimation and reproducible data pipeline. No UI or persistent user data."""
from dataclasses import dataclass, field, asdict, replace
from itertools import combinations
import hashlib
import json
import math
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import minimize
from scipy.special import log_ndtr
from statsmodels.tools.numdiff import approx_hess, approx_fprime
from statsmodels.miscmodels.ordinal_model import OrderedModel
from linearmodels.iv import IV2SLS
from linearmodels.panel import PanelOLS, RandomEffects

VERSION = "2026.09.10"


def unique(values):
    return list(dict.fromkeys(v for v in values if v is not None and v != ""))


def fingerprint(content):
    return hashlib.sha256(content).hexdigest()


def binary(values):
    return set(pd.Series(values).dropna().unique()) == {0, 1}


def infer_type(values):
    y = pd.to_numeric(values, errors="coerce").dropna()
    if binary(y):
        return "二元"
    # Integer-valued income is not automatically a count or an ordinal scale.
    return "连续"


def esr_identification_candidates(data, y, treatment, candidates, alpha=.05):
    """Compatibility entry: Probit relevance only, never an exclusion certification."""
    from tihu_iv import esr_candidates
    s = ModelSpec("ESR", y, treatment=treatment)
    screened = esr_candidates(data, s, [c for c in data if c not in candidates])
    return screened[screened["相关性 p值"] < alpha]


def validate_panel(df, entity, time):
    if not entity or not time or entity == time:
        raise ValueError("请选择不同的个体和时间列")
    if df[[entity, time]].isna().any().any():
        raise ValueError("个体或时间列有缺失，请先处理")
    if df.duplicated([entity, time]).any():
        raise ValueError("存在重复的个体+时间记录，请在清洗中明确处理；不会自动取平均")


def apply_steps(raw, steps):
    work = raw.copy(deep=True)
    audit = []
    for step in steps:
        before = work.copy()
        op, cols = step["op"], step.get("cols", [])
        if any(c not in work for c in cols):
            raise ValueError("操作依赖的变量不存在，请检查规则顺序")
        if op == "deduplicate":
            work = work.drop_duplicates(subset=cols or None, keep="first")
        elif op == "drop_missing":
            work = work.dropna(subset=cols)
        elif op == "numeric":
            for c in cols:
                work[c] = pd.to_numeric(work[c], errors="coerce")
        elif op in {"mean", "median", "interpolate"}:
            group = step.get("group")
            for c in cols:
                if not pd.api.types.is_numeric_dtype(work[c]):
                    raise ValueError(f"{c} 请先转换为数值")
                if op == "interpolate":
                    time = step.get("time")
                    if not time or time in cols or group in cols:
                        raise ValueError("插值需要独立的数值时间列")
                    keys = unique([group, time])
                    if work[keys].isna().any().any() or work.duplicated(keys).any():
                        raise ValueError("插值排序键存在缺失或重复")
                    if not pd.api.types.is_numeric_dtype(work[time]):
                        raise ValueError("插值时间列需为数值")
                    groups = work.groupby(group, sort=False).groups.values() if group else [work.index]
                    for idx in groups:
                        part = work.loc[idx].sort_values(time)
                        s = pd.Series(part[c].to_numpy(), index=part[time].to_numpy())
                        filled = s.interpolate(method="index", limit_area="inside")
                        work.loc[part.index, c] = filled.to_numpy()
                else:
                    val = work.groupby(group)[c].transform(op) if group else getattr(work[c], op)()
                    work[c] = work[c].fillna(val)
        elif op == "winsor":
            pct = float(step["pct"])
            if not 0 < pct < .5:
                raise ValueError("缩尾比例应在 0 与 0.5 之间")
            for c in cols:
                if not pd.api.types.is_numeric_dtype(work[c]):
                    raise ValueError("缩尾变量必须是数值")
                work[c] = work[c].clip(*work[c].quantile([pct, 1-pct]))
        elif op == "filter":
            work = work[work[cols[0]].between(step["lower"], step["upper"])].copy()
        else:
            name = step["name"]
            if not name or name in work.columns:
                raise ValueError("新变量名为空或已经存在")
            x = pd.to_numeric(work[cols[0]], errors="coerce")
            if op == "log":
                work[name] = np.log(x.where(x > 0))
            elif op == "square":
                work[name] = x ** 2
            elif op == "interact":
                work[name] = x * pd.to_numeric(work[cols[1]], errors="coerce")
            elif op == "zscore":
                if not x.std() > 0:
                    raise ValueError("常量无法标准化")
                work[name] = (x-x.mean()) / x.std()
            elif op == "recode":
                work[name] = (work[cols[0]] == step["one"]).astype(float).where(work[cols[0]].notna())
            elif op in {"lag", "diff"}:
                entity, time = step.get("group"), step["time"]
                keys = unique([entity, time])
                if work[keys].isna().any().any() or work.duplicated(keys).any():
                    raise ValueError("滞后/差分的个体时间键存在缺失或重复")
                interval = float(step.get("interval", 1))
                if interval <= 0 or not pd.api.types.is_numeric_dtype(work[time]):
                    raise ValueError("时间必须为数值，间隔必须为正数")
                lookup = work[keys + [cols[0]]].copy()
                lookup[time] = lookup[time] + interval
                lookup = lookup.rename(columns={cols[0]: "__lag_value"})
                lag = work[keys].merge(lookup, on=keys, how="left", validate="one_to_one")["__lag_value"].to_numpy()
                work[name] = lag if op == "lag" else x.to_numpy()-lag
            else:
                raise ValueError("不支持的操作")
            lineage = dict(work.attrs.get("tihu_lineage", {}))
            lineage[name] = unique(cols+[source for c in cols for source in lineage.get(c, [])])
            work.attrs["tihu_lineage"] = lineage
        work = work.replace([np.inf, -np.inf], np.nan)
        common = before.index.intersection(work.index)
        compared = cols if "name" not in step else []
        changed = sum(int((~before.loc[common, c].eq(work.loc[common, c]) & ~(before.loc[common, c].isna() & work.loc[common, c].isna())).sum()) for c in compared)
        audit.append({"操作": op, "处理前行数": len(before), "处理后行数": len(work), "修改值数": changed, "新变量": step.get("name", "")})
    return work, audit


@dataclass
class ModelSpec:
    model: str
    y: str
    core: list = field(default_factory=list)
    controls: list = field(default_factory=list)
    categorical: list = field(default_factory=list)
    se: str = "robust"
    cluster: str = ""
    entity: str = ""
    time: str = ""
    entity_effects: bool = True
    time_effects: bool = True
    treatment: str = ""
    post: str = ""
    policy: float = 0
    running: str = ""
    cutoff: float = 0
    bandwidth: float = 0
    degree: int = 1
    instruments: list = field(default_factory=list)
    selection: list = field(default_factory=list)
    regime0: list = field(default_factory=list)
    regime1: list = field(default_factory=list)
    effect: str = "ATT"
    left: object = None
    right: object = None
    match_k: int = 1
    caliper: float = .1
    interactions: list = field(default_factory=list)
    group: str = ""
    contrast: str = ""
    auto_instrument: bool = False


@dataclass
class Fit:
    spec: ModelSpec
    table: pd.DataFrame
    sample: pd.DataFrame
    effect: dict
    details: dict = field(default_factory=dict)
    design: object = None
    response: object = None
    fitted: object = None


def required_columns(s):
    return unique([s.y] + s.core + s.controls + [s.cluster if s.se == "cluster" else "", s.entity, s.time, s.treatment, s.post, s.running, s.group] + s.instruments + s.selection + s.regime0 + s.regime1 + [v for pair in s.interactions for v in pair])


def frame(data, s):
    need = required_columns(s)
    missing = [c for c in need if c not in data]
    if missing:
        raise ValueError("缺少模型变量：" + "、".join(missing))
    d = data.copy().replace([np.inf, -np.inf], np.nan)
    drop = [c for c in need if not (s.model == "Heckman" and c == s.y)]
    d = d.dropna(subset=drop).copy()
    if s.model == "Heckman":
        d = d[(d[s.treatment] == 0) | d[s.y].notna()].copy()
    if s.model in {"Sharp RD", "Fuzzy RD"} and s.bandwidth > 0:
        d = d[(d[s.running]-s.cutoff).abs() <= s.bandwidth].copy()
    if d.empty or d[s.y].dropna().nunique() < 2:
        raise ValueError("结果变量无变化或没有有效样本")
    if s.se == "cluster" and (not s.cluster or d[s.cluster].nunique() < 2):
        raise ValueError("聚类至少需要两个有效组")
    return d


def design(d, columns, categorical=(), constant=True):
    parts = []
    for c in unique(columns):
        if c in categorical or not pd.api.types.is_numeric_dtype(d[c]):
            parts.append(pd.get_dummies(d[c].astype(str), prefix=c, drop_first=True, dtype=float))
        else:
            parts.append(d[[c]].astype(float))
    x = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=d.index)
    x = x.loc[:, x.nunique() > 1]
    if constant:
        x.insert(0, "const", 1.0)
    if x.shape[1] == 0 or len(x) <= x.shape[1] + 2:
        raise ValueError("有效样本不足以估计这些参数")
    if np.linalg.matrix_rank(x.to_numpy()) < x.shape[1]:
        raise ValueError("解释变量完全共线")
    return x


def table_from_result(r):
    p = r.params
    if not isinstance(p, pd.Series):
        p = pd.Series(p, index=r.model.exog_names)
    se = r.std_errors if hasattr(r, "std_errors") else r.bse
    ci = np.asarray(r.conf_int())
    return pd.DataFrame({"term": p.index, "coef": np.asarray(p), "se": np.asarray(se), "p": np.asarray(r.pvalues), "lower": ci[:, 0], "upper": ci[:, 1]})


def effect_row(table, term):
    rows = table[table.term == term]
    if rows.empty:
        raise ValueError(f"目标项 {term} 被吸收、无变化或不可估计")
    row = rows.iloc[0].to_dict()
    if not np.isfinite([row["coef"], row["se"], row["p"]]).all() or row["se"] <= 0:
        raise ValueError("目标效应或标准误无效")
    return row


def covariance_options(s, d, panel=False):
    if s.se == "cluster":
        labels = pd.factorize(d[s.cluster])[0]
        if panel:
            return {"cov_type": "clustered", "clusters": pd.DataFrame({"cluster": labels}, index=d.index), "group_debias": True}
        return {"cov_type": "cluster", "cov_kwds": {"groups": labels, "use_correction": True}, "use_t": True}
    if panel:
        return {"cov_type": "robust" if s.se == "robust" else "unadjusted"}
    return {"cov_type": "HC1" if s.se == "robust" else "nonrobust", "use_t": True}


def likelihood_fit(logobs, initial, s, d):
    def objective(p):
        values = logobs(p)
        return -np.mean(values) if np.isfinite(values).all() else 1e50
    best = minimize(objective, initial, method="BFGS", options={"maxiter": 700, "gtol": 1e-6})
    if not np.isfinite(best.fun) or np.max(np.abs(best.jac)) > 2e-4:
        raise ValueError("最大似然未收敛，该组合已跳过")
    h = approx_hess(best.x, lambda p: -np.sum(logobs(p)))
    h = (h+h.T)/2
    eig = np.linalg.eigvalsh(h)
    if not np.isfinite(eig).all() or eig.min() <= max(1e-8, eig.max()*1e-9):
        raise ValueError("信息矩阵不可逆，无法可靠计算标准误")
    cov = np.linalg.inv(h)
    if s.se != "ordinary":
        scores = approx_fprime(best.x, logobs)
        if s.se == "cluster":
            scores = pd.DataFrame(scores).groupby(pd.factorize(d[s.cluster])[0]).sum().to_numpy()
            g = len(scores)
            correction = g/(g-1)
        else:
            correction = len(d)/(len(d)-1)
        cov = cov @ (scores.T @ scores) @ cov * correction
    return best.x, cov, float(-len(d)*best.fun)


def delta_table(p, cov, transform, names):
    values = np.atleast_1d(transform(p))
    jac = np.atleast_2d(approx_fprime(p, transform))
    v = np.diag(jac @ cov @ jac.T)
    if np.any(v <= 0) or not np.isfinite(v).all():
        raise ValueError("效应标准误不可计算")
    se = np.sqrt(v)
    return pd.DataFrame({"term": names, "coef": values, "se": se, "p": 2*stats.norm.sf(np.abs(values/se)), "lower": values-1.95996398454*se, "upper": values+1.95996398454*se})


def fit_esr(d, s):
    if not binary(d[s.treatment]):
        raise ValueError("ESR 的选择变量必须明确编码为 0/1")
    if not s.selection:
        raise ValueError("请选择选择方程变量")
    xs = [design(d, s.regime0 or s.controls, s.categorical), design(d, s.regime1 or s.controls, s.categorical)]
    z = design(d, s.selection, s.categorical)
    y = d[s.y].to_numpy(float)
    target = d[s.treatment].to_numpy(int)
    means, scales, arrays = [], [], []
    for x in [z] + xs:
        mu = x.mean().to_numpy(); mu[0] = 0
        sd = x.std().to_numpy(); sd[0] = 1
        means.append(mu); scales.append(sd); arrays.append((x.to_numpy()-mu)/sd)
    Z, X0, X1 = arrays
    ym, ys = y.mean(), y.std(ddof=1)
    yn = (y-ym)/ys
    dims = [Z.shape[1], X0.shape[1], X1.shape[1]]
    a, b, c = np.cumsum(dims)
    for value, x in [(0, X0), (1, X1)]:
        subset = x[target == value]
        if len(subset) <= x.shape[1] + 5 or np.linalg.matrix_rank(subset) < x.shape[1]:
            raise ValueError("ESR 某组样本不足或组内变量共线")
    pr = sm.Probit(target, Z).fit(disp=False)
    if not pr.mle_retvals.get("converged", False):
        raise ValueError("选择方程未收敛")
    r0 = sm.OLS(yn[target == 0], X0[target == 0]).fit()
    r1 = sm.OLS(yn[target == 1], X1[target == 1]).fit()
    initial = np.r_[pr.params, r0.params, r1.params, np.log(np.sqrt(r0.scale)), np.log(np.sqrt(r1.scale)), 0., 0.]
    def parts(p):
        return Z@p[:a], X0@p[a:b], X1@p[b:c], np.exp(p[c:c+2]), np.tanh(p[c+2:c+4])
    def logobs(p):
        eta, mu0, mu1, sig, rho = parts(p)
        if np.any(sig < 1e-8) or np.any(np.abs(rho) > .9999):
            return np.full(len(y), -1e50)
        e0, e1 = (yn-mu0)/sig[0], (yn-mu1)/sig[1]
        l0 = stats.norm.logpdf(e0)-np.log(sig[0])+log_ndtr(-(eta+rho[0]*e0)/np.sqrt(1-rho[0]**2))
        l1 = stats.norm.logpdf(e1)-np.log(sig[1])+log_ndtr((eta+rho[1]*e1)/np.sqrt(1-rho[1]**2))
        return np.where(target == 1, l1, l0)
    p, cov, ll = likelihood_fit(logobs, initial, s, d)
    def counterfactual(p):
        eta, m0, m1, sig, rho = parts(p)
        l1 = np.exp(stats.norm.logpdf(eta)-log_ndtr(eta))
        l0 = np.exp(stats.norm.logpdf(eta)-log_ndtr(-eta))
        factual1 = ym+ys*(m1+sig[1]*rho[1]*l1)
        cf0 = ym+ys*(m0+sig[0]*rho[0]*l1)
        factual0 = ym+ys*(m0-sig[0]*rho[0]*l0)
        cf1 = ym+ys*(m1-sig[1]*rho[1]*l0)
        t, u = target == 1, target == 0
        return np.array([(factual1[t]-cf0[t]).mean(), (cf1[u]-factual0[u]).mean(), factual1[t].mean(), cf0[t].mean(), factual0[u].mean(), cf1[u].mean()])
    effects = delta_table(p, cov, counterfactual, ["ATT", "ATU", "E[Y1|D=1]", "E[Y0|D=1]", "E[Y0|D=0]", "E[Y1|D=0]"])
    def original_params(p):
        output = []
        for i, (lo, hi) in enumerate([(0, a), (a, b), (b, c)]):
            v = p[lo:hi].copy()/scales[i]
            v[0] -= np.dot(means[i][1:], v[1:])
            if i:
                v *= ys; v[0] += ym
            output.extend(v)
        return np.r_[output, np.exp(p[c:c+2])*ys, np.tanh(p[c+2:c+4])]
    names = ["选择:"+v for v in z.columns] + ["D=0:"+v for v in xs[0].columns] + ["D=1:"+v for v in xs[1].columns] + ["sigma0", "sigma1", "rho0", "rho1"]
    parameters = delta_table(p, cov, original_params, names)
    return Fit(s, parameters, d, effect_row(effects, s.effect), {"effects": effects, "loglik": ll-len(y)*np.log(ys), "n0": int((target == 0).sum()), "n1": int(target.sum()), "converged": True})


def match_pairs(d, treatment, columns, categorical=(), k=1, caliper=.1):
    if not columns or not binary(d[treatment]):
        raise ValueError("匹配需要协变量及 0/1 两组")
    x = design(d, columns, categorical)
    pr = sm.Logit(d[treatment], x).fit(disp=False, maxiter=150)
    if not pr.mle_retvals.get("converged", False):
        raise ValueError("倾向得分模型未收敛")
    ps = np.asarray(pr.predict(x))
    treated, controls = np.where(d[treatment].to_numpy() == 1)[0], np.where(d[treatment].to_numpy() == 0)[0]
    if k > len(controls):
        raise ValueError("对照组不足以完成匹配")
    pairs = []
    # One-to-one without replacement gives explicit independent matched pairs.
    available = set(controls.tolist())
    for t in treated:
        candidates = sorted(available, key=lambda i: (abs(ps[i]-ps[t]), i))[:k]
        if len(candidates) == k and all(abs(ps[i]-ps[t]) <= caliper for i in candidates):
            pairs.append((t, candidates))
            available.difference_update(candidates)
    if len(pairs) < 3:
        raise ValueError("卡尺内有效匹配不足")
    return pairs, ps


def fit_model(data, spec):
    s = replace(spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = frame(data, s)
        if s.model == "ESR":
            if s.auto_instrument:
                from tihu_iv import instrument_test, identification_report
                check, _ = instrument_test(d, s)
                if not check["通过初筛"]:
                    raise ValueError("当前控制组合的识别变量未通过 Probit 选择方程 p<0.05 初筛")
            result = fit_esr(d, s)
            if s.auto_instrument:
                result.details["识别诊断报告"] = identification_report(check)
                result.details["识别变量检验"] = pd.DataFrame([{k: check[k] for k in ["选择方程 Wald χ²", "相关性 p值", "有效样本", "检验", "通过初筛"]}])
            return result
        if s.model == "IV/2SLS":
            from tihu_iv import fit_iv
            return fit_iv(d, s)
        columns = unique(s.core+s.controls)
        focus = s.core[0] if s.core else "const"
        details = {}
        if s.model in {"ANOVA", "交互效应"}:
            levels = sorted(d[s.group].astype(str).unique())
            if len(levels) < 2 or s.contrast not in levels[1:]:
                raise ValueError("比较组或参考组在当前样本中不存在")
            s.categorical = unique(s.categorical+[s.group])
            columns = unique(columns+[s.group])
            focus = f"{s.group}_{s.contrast}"
            if s.model == "交互效应":
                for i, level in enumerate(levels[1:]):
                    term = f"__gx{i}"
                    d[term] = (d[s.group].astype(str) == level).astype(float)*d[s.core[0]]
                    columns.append(term)
                    if level == s.contrast:
                        focus = term
            details["参考组"] = levels[0]
            details["比较组"] = s.contrast
        if s.model in {"DID", "PSM-DID"}:
            validate_panel(d, s.entity, s.time)
            if not binary(d[s.treatment]):
                raise ValueError("Treat 必须为 0/1")
            if d.groupby(s.entity)[s.treatment].nunique().max() != 1:
                raise ValueError("本模型要求 Treat 在个体内不随时间改变")
            d["__post"] = d[s.post] if s.post else (d[s.time] >= s.policy).astype(float)
            if not binary(d["__post"]):
                raise ValueError("Post 必须同时存在政策前 0 和政策后 1")
            if d.groupby(s.time)["__post"].nunique().max() != 1:
                raise ValueError("当前 DID 支持共同政策时间；分期实施请勿使用此设定")
            if d.groupby(s.time)["__post"].first().sort_index().diff().dropna().lt(0).any():
                raise ValueError("Post 必须按时间从 0 转为 1")
            cells = d.groupby([s.treatment, "__post"]).size()
            if len(cells) != 4:
                raise ValueError("DID 需要处理/对照组在政策前后都有样本")
            if s.model == "PSM-DID" and "__match_pair" not in d:
                agg = {v: ("first" if v in s.categorical else "mean") for v in unique([s.treatment]+s.controls)}
                pre = d[d.__post == 0].groupby(s.entity).agg(agg)
                pairs, ps = match_pairs(pre, s.treatment, s.controls, s.categorical, 1, s.caliper)
                ids = pre.iloc[unique([i for t, cs in pairs for i in [t]+cs])].index
                d = d[d[s.entity].isin(ids)].copy()
                mapping = {pre.index[row]: pair for pair, (t, cs) in enumerate(pairs) for row in [t]+cs}
                d["__match_pair"] = d[s.entity].map(mapping)
                details["matched_pairs"] = len(pairs)
            d["__did"] = d[s.treatment]*d["__post"]
            columns, focus = unique(["__did"]+s.controls+([] if s.time_effects else ["__post"])), "__did"
        if s.model in {"Sharp RD", "Fuzzy RD"}:
            r = d[s.running]-s.cutoff
            z = (r >= 0).astype(float)
            if z.nunique() < 2:
                raise ValueError("断点两侧都需要有效样本")
            d["__rd_z"], d["__rd_r"] = z, r
            terms = []
            for power in range(1, s.degree+1):
                d[f"__rd_r{power}"] = r ** power
                d[f"__rd_zr{power}"] = z*r ** power
                terms += [f"__rd_r{power}", f"__rd_zr{power}"]
            columns = unique(([] if s.model == "Fuzzy RD" else ["__rd_z"])+terms+s.controls)
            focus = s.treatment if s.model == "Fuzzy RD" else "__rd_z"
            details.update(left_n=int((z == 0).sum()), right_n=int(z.sum()))
        for i, (left, right) in enumerate(s.interactions):
            d[f"__interaction{i}"] = d[left]*d[right]
            columns = unique(columns+[left, right, f"__interaction{i}"])
            focus = f"__interaction{i}"
        if s.model == "PSM":
            if "__match_pair" in d:
                pairs = []
                for _, part in d.groupby("__match_pair", sort=True):
                    treated = np.where((d.index.isin(part.index)) & (d[s.treatment].to_numpy() == 1))[0]
                    control = np.where((d.index.isin(part.index)) & (d[s.treatment].to_numpy() == 0))[0]
                    if len(treated) == 1 and len(control):
                        pairs.append((int(treated[0]), control.tolist()))
                ps = None
            else:
                original_pairs, ps = match_pairs(d, s.treatment, s.controls, s.categorical, s.match_k, s.caliper)
                selected = []
                for pair, (treated, controls) in enumerate(original_pairs):
                    for row in [treated]+controls:
                        part = d.iloc[[row]].copy()
                        part["__match_pair"] = pair
                        selected.append(part)
                d = pd.concat(selected)
                pairs = []
                for _, part in d.groupby("__match_pair", sort=True):
                    positions = np.where(d.index.isin(part.index))[0]
                    treated = [i for i in positions if d.iloc[i][s.treatment] == 1]
                    controls = [i for i in positions if d.iloc[i][s.treatment] == 0]
                    pairs.append((treated[0], controls))
            y = d[s.y].to_numpy()
            differences = np.array([y[t]-y[cs].mean() for t, cs in pairs])
            b, se = differences.mean(), stats.sem(differences)
            q = stats.t.ppf(.975, len(pairs)-1)
            table = pd.DataFrame([{"term": "ATT", "coef": b, "se": se, "p": 2*stats.t.sf(abs(b/se), len(pairs)-1), "lower": b-q*se, "upper": b+q*se}])
            return Fit(s, table, d, effect_row(table, "ATT"), {"matched_pairs": len(pairs), "pairs": pairs, "ps": ps, "inference": "匹配后组对差值的条件 t 推断，未计入倾向得分估计不确定性"})
        if s.model in {"FE", "RE", "FE+RE", "DID", "PSM-DID"}:
            validate_panel(d, s.entity, s.time)
            original = d.copy()
            d = d.set_index([s.entity, s.time], drop=False)
            x = design(d, columns, s.categorical)
            kw = covariance_options(s, d, True)
            if s.model == "RE":
                r = RandomEffects(d[s.y], x).fit(**kw)
            else:
                if not s.entity_effects and not s.time_effects:
                    raise ValueError("FE 至少需要一种固定效应")
                if s.model == "FE+RE" and not s.entity_effects:
                    raise ValueError("FE+RE 比较需要启用个体固定效应")
                fe_options = {"auto_df": False, "count_effects": False} if s.se == "cluster" else {}
                r = PanelOLS(d[s.y], x, entity_effects=s.entity_effects, time_effects=s.time_effects, drop_absorbed=True).fit(**kw, **fe_options)
                if s.model == "FE+RE":
                    re = RandomEffects(d[s.y], x).fit(**kw)
                    details["RE"] = table_from_result(re)
                    details["排名依据"] = "固定效应模型"
                    if s.se == "ordinary":
                        common = [c for c in r.params.index.intersection(re.params.index) if c != "const"]
                        vdiff = r.cov.loc[common, common]-re.cov.loc[common, common]
                        if common and np.linalg.eigvalsh(vdiff).min() > 0:
                            diff = r.params[common]-re.params[common]
                            statistic = float(diff @ np.linalg.solve(vdiff, diff))
                            details["Hausman p"] = float(stats.chi2.sf(statistic, len(common)))
            table = table_from_result(r)
            if s.se == "cluster" and s.model != "RE":
                nonnested = 0
                effects = ([s.entity] if s.entity_effects else [])+([s.time] if s.time_effects else [])
                for effect in effects:
                    if original.groupby(effect)[s.cluster].nunique().max() > 1:
                        nonnested += original[effect].nunique()-1
                denominator = len(d)-len(r.params)-nonnested
                if denominator <= 0:
                    raise ValueError("固定效应与聚类设定没有剩余自由度")
                table["se"] *= np.sqrt((len(d)-len(r.params))/denominator)
                df_test = original[s.cluster].nunique()-1
                table["p"] = 2*stats.t.sf(np.abs(table.coef/table.se), df_test)
                q = stats.t.ppf(.975, df_test)
                table["lower"] = table.coef-q*table.se
                table["upper"] = table.coef+q*table.se
            details["rsquared"] = r.rsquared
            return Fit(s, table, original, effect_row(table, focus), details, x, d[s.y], r)
        if s.model in {"IV/2SLS", "Fuzzy RD"}:
            endog = s.treatment if s.model == "Fuzzy RD" else s.core[0]
            x = design(d, [v for v in columns if v != endog], s.categorical)
            instruments = ["__rd_z"] if s.model == "Fuzzy RD" else s.instruments
            z = design(d, instruments, s.categorical, constant=False)
            if len(set(x.columns) & set(z.columns)):
                raise ValueError("排除工具变量不能同时进入外生解释项")
            kw = {"cov_type": "robust" if s.se == "robust" else "unadjusted", "debiased": True}
            if s.se == "cluster":
                kw = {"cov_type": "clustered", "clusters": pd.factorize(d[s.cluster])[0], "debiased": True}
            r = IV2SLS(d[s.y], x, d[[endog]], z).fit(**kw)
            table = table_from_result(r)
            try:
                diagnostics = {"first_stage": r.first_stage.diagnostics}
            except np.linalg.LinAlgError:
                diagnostics = {"first_stage": "第一阶段完全拟合，常规 F 统计量不可计算"}
            return Fit(s, table, d, effect_row(table, endog), diagnostics, x, d[s.y], r)
        if s.model in {"Tobit", "Heckman"}:
            return fit_limited(d, s, columns, focus)
        constant = not s.model.startswith("Ordered")
        x = design(d, columns, s.categorical, constant)
        kw = covariance_options(s, d)
        if s.model in {"OLS", "Pooled OLS", "LPM", "ANOVA", "交互效应", "Sharp RD"}:
            r = sm.OLS(d[s.y], x).fit(**kw)
        else:
            kw.pop("use_t", None)
            if s.model in {"Logit", "Probit", "Logit+Probit"}:
                if not binary(d[s.y]):
                    raise ValueError("结果变量需要 0/1 编码")
                model = sm.Probit if s.model == "Probit" else sm.Logit
                r = model(d[s.y], x).fit(disp=False, maxiter=200, **kw)
                if s.model == "Logit+Probit":
                    details["Probit"] = table_from_result(sm.Probit(d[s.y], x).fit(disp=False, maxiter=200, **kw))
            elif s.model.startswith("Ordered"):
                r = OrderedModel(d[s.y], x, distr="probit" if "Probit" in s.model else "logit").fit(method="bfgs", disp=False, maxiter=200, **kw)
            elif s.model in {"Poisson", "负二项"}:
                if (d[s.y] < 0).any() or not np.allclose(d[s.y], np.floor(d[s.y])):
                    raise ValueError("计数结果必须是非负整数")
                model = sm.Poisson if s.model == "Poisson" else sm.NegativeBinomial
                r = model(d[s.y], x).fit(disp=False, maxiter=200, **kw)
            else:
                raise ValueError("未知模型")
            if not r.mle_retvals.get("converged", False):
                raise ValueError("模型未收敛")
        table = table_from_result(r)
        if s.model not in {"OLS", "Pooled OLS", "LPM", "ANOVA", "交互效应", "Sharp RD"} and s.se != "ordinary":
            # Stata ML uses N/(N-1), or G/(G-1), rather than regression df corrections.
            factor = len(d)/(len(d)-1) if s.se == "robust" else (len(d)-len(r.params))/(len(d)-1)
            table["se"] *= np.sqrt(factor)
            table["p"] = 2*stats.norm.sf(np.abs(table.coef/table.se))
            table["lower"] = table.coef-stats.norm.ppf(.975)*table.se
            table["upper"] = table.coef+stats.norm.ppf(.975)*table.se
        details["rsquared"] = getattr(r, "rsquared", getattr(r, "prsquared", np.nan))
        return Fit(s, table, d, effect_row(table, focus), details, x, d[s.y], r)


def fit_limited(d, s, columns, focus):
    x = design(d, columns, s.categorical)
    y = d[s.y].to_numpy(float)
    k = x.shape[1]
    X = x.to_numpy()
    if s.model == "Tobit":
        if s.left is None and s.right is None:
            raise ValueError("请选择至少一个实际审查边界")
        if s.left is not None and s.right is not None and s.left >= s.right:
            raise ValueError("下界必须小于上界")
        r = sm.OLS(y, x).fit()
        def logobs(p):
            sig = np.exp(p[-1]); err = (y-X@p[:-1])/sig
            ll = stats.norm.logpdf(err)-np.log(sig)
            if s.left is not None:
                ll = np.where(y <= s.left, log_ndtr((s.left-X@p[:-1])/sig), ll)
            if s.right is not None:
                ll = np.where(y >= s.right, log_ndtr((X@p[:-1]-s.right)/sig), ll)
            return ll
        p, cov, ll = likelihood_fit(logobs, np.r_[r.params, np.log(np.sqrt(r.scale))], s, d)
        table = delta_table(p, cov, lambda p: p[:-1], list(x.columns))
    else:
        if not binary(d[s.treatment]) or not s.selection:
            raise ValueError("Heckman 需要 0/1 选择变量和选择方程")
        z = design(d, s.selection, s.categorical)
        observed = d[s.treatment].to_numpy() == 1
        Z = z.to_numpy(); q = z.shape[1]
        r = sm.OLS(y[observed], X[observed]).fit()
        prob = sm.Probit(observed, Z).fit(disp=False)
        def logobs(p):
            eta = Z@p[:q]; sig = np.exp(p[-2]); rho = np.tanh(p[-1])
            err = (np.nan_to_num(y)-X@p[q:q+k])/sig
            seen = stats.norm.logpdf(err)-np.log(sig)+log_ndtr((eta+rho*err)/np.sqrt(1-rho**2))
            return np.where(observed, seen, log_ndtr(-eta))
        p, cov, ll = likelihood_fit(logobs, np.r_[prob.params, r.params, np.log(np.sqrt(r.scale)), 0.], s, d)
        table = delta_table(p, cov, lambda p: p[q:q+k], list(x.columns))
    return Fit(s, table, d, effect_row(table, focus), {"loglik": ll}, x, d[s.y])


def candidate_specs(base, pool, minimum, maximum, limit=500, seed=42, joint=True):
    pool = [c for c in unique(pool) if c not in base.controls+base.core]
    if len(pool) > 20 or len(base.core) > 8:
        raise ValueError("候选控制变量最多 20 个，核心 X 最多 8 个")
    maximum = min(int(maximum), len(pool))
    if minimum > maximum:
        raise ValueError("最少候选控制数大于可用数量")
    total = sum(math.comb(len(pool), k) for k in range(int(minimum), maximum+1))
    rng = np.random.default_rng(seed)
    positions = set(rng.choice(total, min(total, limit), replace=False).tolist())
    counter = 0
    for size in range(int(minimum), maximum+1):
        for combo in combinations(pool, size):
            if counter in positions:
                cores = [base.core] if joint or len(base.core) <= 1 else [[v] for v in base.core]
                for core in cores:
                    candidate = replace(base, core=core, controls=unique(base.controls+list(combo)))
                    if candidate.model == "ESR" and candidate.instruments:
                        candidate = replace(
                            candidate,
                            selection=unique(candidate.controls+candidate.instruments),
                            regime0=[],
                            regime1=[],
                        )
                    yield candidate
            counter += 1


def fixed_sample(data, base, pool):
    s = replace(base, controls=unique(base.controls+pool))
    return frame(data, s)


def rank_fit(fit, joint=False):
    if joint and len(fit.spec.core) > 1:
        return max(effect_row(fit.table, c)["p"] for c in fit.spec.core)
    return fit.effect["p"]


def config_payload(file_hash, steps, settings):
    def plain(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {k: plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        return value
    return plain({"version": VERSION, "file_hash": file_hash, "steps": steps, "settings": settings})


def mediation_bootstrap(data, spec, mediator, reps=199, seed=42):
    if spec.model not in {"OLS", "Pooled OLS", "FE", "RE", "FE+RE", "DID", "PSM-DID", "Sharp RD"}:
        raise ValueError("间接效应分解仅用于当前支持的线性模型")
    common = frame(data.dropna(subset=[mediator]), spec)
    def indirect(d):
        a = fit_model(d, replace(spec, y=mediator)).effect["coef"]
        full = fit_model(d, replace(spec, controls=unique(spec.controls+[mediator])))
        return a*effect_row(full.table, mediator)["coef"]
    point = indirect(common)
    rng = np.random.default_rng(seed)
    unit = spec.cluster if spec.se == "cluster" else spec.entity
    values = []
    for _ in range(reps):
        if unit:
            groups = list(common.groupby(unit, sort=False))
            pieces = []
            for j, idx in enumerate(rng.integers(len(groups), size=len(groups))):
                part = groups[idx][1].copy()
                if spec.entity:
                    part[spec.entity] = part[spec.entity].astype(str)+f"_b{j}"
                if spec.cluster:
                    part[spec.cluster] = f"cluster_{j}"
                pieces.append(part)
            sampled = pd.concat(pieces, ignore_index=True)
        else:
            sampled = common.iloc[rng.integers(len(common), size=len(common))].reset_index(drop=True)
        try:
            value = indirect(sampled)
            if np.isfinite(value):
                values.append(value)
        except (ValueError, np.linalg.LinAlgError):
            continue
    if len(values) < max(30, int(.8*reps)):
        raise ValueError("Bootstrap 有效重复不足，未输出间接效应区间")
    lo, hi = np.quantile(values, [.025, .975])
    return {"term": "间接项 a×b", "coef": point, "se": float(np.std(values, ddof=1)), "p": np.nan, "lower": lo, "upper": hi, "有效重复": len(values), "抽样单位": unit or "观测行"}


def mechanism_analysis(data, spec, mediator):
    """Run a model-aware channel path on one inherited complete-case sample."""
    common = frame(data.dropna(subset=[mediator]), spec)
    baseline = fit_model(common, spec)
    mediator_spec = replace(spec, y=mediator)
    # Binary/count outcome families describe Y, not necessarily the mediator.
    if spec.model in {"Logit", "Probit", "Logit+Probit", "Ordered Logit", "Ordered Probit", "Poisson", "负二项", "LPM", "Tobit"}:
        mediator_spec = replace(spec, model="LPM" if binary(common[mediator]) else "OLS", y=mediator)
    elif spec.model == "ESR" and binary(common[mediator]):
        excluded = [v for v in spec.selection if v not in spec.controls]
        if excluded:
            mediator_spec = ModelSpec("IV/2SLS", mediator, [spec.treatment], spec.controls,
                                      categorical=spec.categorical, se=spec.se,
                                      cluster=spec.cluster, instruments=excluded)
        else:
            mediator_spec = ModelSpec("LPM", mediator, [spec.treatment], spec.controls,
                                      categorical=spec.categorical, se=spec.se, cluster=spec.cluster)
    channel = fit_model(common, mediator_spec)
    adjusted = None
    if spec.model != "PSM":
        if spec.model == "ESR":
            adjusted_spec = replace(spec, controls=unique(spec.controls+[mediator]),
                                    regime0=unique(spec.regime0+[mediator]) if spec.regime0 else [],
                                    regime1=unique(spec.regime1+[mediator]) if spec.regime1 else [])
        else:
            adjusted_spec = replace(spec, controls=unique(spec.controls+[mediator]))
        adjusted = fit_model(common, adjusted_spec)
    else:
        # Preserve the original matched design and report the conditional M-Y association.
        exposure = spec.treatment
        aux = ModelSpec("OLS", spec.y, [mediator], [exposure]+spec.controls, categorical=spec.categorical, se="robust")
        adjusted = fit_model(common, aux)
    if spec.model == "ESR":
        m_effects = adjusted.table[adjusted.table.term.isin([f"D=0:{mediator}", f"D=1:{mediator}"])].to_dict("records")
        if len(m_effects) != 2:
            raise ValueError("机制变量在 ESR 结果方程中不可估计")
    else:
        m_effects = [effect_row(adjusted.table, mediator)]
    return {"baseline": baseline, "channel": channel, "adjusted": adjusted, "mediator_effects": m_effects}


def grouped_moderation(data, spec, moderator):
    """Compare inherited model effects in independent low/high moderator groups."""
    common = frame(data.dropna(subset=[moderator]), spec)
    values = pd.to_numeric(common[moderator], errors="coerce")
    if values.notna().sum() != len(common):
        raise ValueError("分组调节变量必须为数值")
    if values.nunique() < 2:
        raise ValueError("调节变量没有足够变化")
    if "__match_pair" in common:
        pair_values = common.assign(__moderator=values).groupby("__match_pair")["__moderator"].mean()
        cutoff = float(pair_values.median())
        low_pairs = pair_values[pair_values < cutoff].index
        groups = [("低组", common[common["__match_pair"].isin(low_pairs)]),
                  ("高组", common[~common["__match_pair"].isin(low_pairs)])]
    elif spec.entity:
        unit_values = common.assign(__moderator=values).groupby(spec.entity)["__moderator"].mean()
        cutoff = float(unit_values.median())
        low_units = unit_values[unit_values < cutoff].index
        groups = [("低组", common[common[spec.entity].isin(low_units)]),
                  ("高组", common[~common[spec.entity].isin(low_units)])]
    else:
        cutoff = float(values.median())
        groups = [("低组", common[values < cutoff]), ("高组", common[values >= cutoff])]
    results = {}
    for name, part in groups:
        if len(part) < 15:
            raise ValueError(f"{name}有效样本少于 15")
        results[name] = fit_model(part, spec)
    low, high = results["低组"].effect, results["高组"].effect
    diff = high["coef"]-low["coef"]
    se = math.sqrt(high["se"]**2+low["se"]**2)
    if not se > 0:
        raise ValueError("组间差异标准误无法计算")
    z = diff/se
    contrast = {"term": "高组−低组", "coef": diff, "se": se, "p": float(2*stats.norm.sf(abs(z))),
                "lower": diff-stats.norm.ppf(.975)*se, "upper": diff+stats.norm.ppf(.975)*se,
                "分组点": cutoff}
    return results, contrast


def load_config(content, file_hash):
    if len(content) > 2_000_000:
        raise ValueError("配置文件过大")
    obj = json.loads(content)
    if obj.get("version") != VERSION or obj.get("file_hash") != file_hash:
        raise ValueError("项目版本或数据内容不匹配，请上传保存项目时的原始文件")
    if not isinstance(obj.get("steps"), list) or not isinstance(obj.get("settings"), dict):
        raise ValueError("项目配置格式无效")
    if len(obj["steps"]) > 100:
        raise ValueError("项目操作超过 100 步")
    numeric_keys = {"cfg_policy", "cfg_cutoff", "cfg_bandwidth", "cfg_degree", "cfg_left", "cfg_right", "cfg_caliper", "cfg_min", "cfg_max", "cfg_budget"}
    list_keys = {"cfg_categorical", "cfg_core", "cfg_instruments", "cfg_controls", "cfg_pool", "cfg_selection", "cfg_regime0", "cfg_regime1", "cfg_ts_exog"}
    bool_keys = {"cfg_entity_effects", "cfg_time_effects", "cfg_separate_regimes"}
    for key, value in obj["settings"].items():
        if key in numeric_keys and (not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError("项目的数值设定无效")
        if key in list_keys and (not isinstance(value, list) or not all(isinstance(v, str) for v in value)):
            raise ValueError("项目的变量列表无效")
        if key in bool_keys and not isinstance(value, bool):
            raise ValueError("项目的开关设定无效")
    for key in ["cfg_min", "cfg_max"]:
        if key in obj["settings"] and not 0 <= obj["settings"][key] <= 20:
            raise ValueError("项目的控制变量数量超出范围")
    if "cfg_budget" in obj["settings"] and not 1 <= obj["settings"]["cfg_budget"] <= 10000:
        raise ValueError("项目的搜索预算超出范围")
    return obj
