"""Session-scoped, stepwise econometrics workbench."""
from dataclasses import asdict, replace
from io import BytesIO
import json
import math
import time
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
import statsmodels.api as sm

from tihu_core import (VERSION, ModelSpec, apply_steps, binary, candidate_specs,
                       config_payload, fingerprint, fit_model, fixed_sample,
                       grouped_moderation, infer_type, esr_identification_candidates,
                       load_config, mechanism_analysis,
                       mediation_bootstrap, rank_fit, required_columns, unique,
                       validate_panel)
from tihu_export import dta_bytes, reproducibility_bundle, stata_script


def invalidate(ws):
    ws.pop("job", None)
    ws.pop("fits", None)
    ws.pop("follow", None)
    ws.pop("bundle", None)
    ws.pop("indirect", None)
    ws.pop("follow_bundle", None)
    st.session_state.pop("ts_result", None)


def reset_session():
    for key in list(st.session_state):
        del st.session_state[key]


def choose(label, options, key, default=None):
    options = list(options)
    if st.session_state.get(key) not in options:
        st.session_state[key] = default if default in options else (options[0] if options else None)
    return st.selectbox(label, options, key=key)


def multiple(label, options, key, default=()):
    options = list(options)
    st.session_state[key] = [v for v in st.session_state.get(key, list(default)) if v in options]
    return st.multiselect(label, options, key=key)


def number(label, key, default, **kwargs):
    return st.number_input(label, value=default, key=key, **kwargs)


def display_table(table):
    renamed = table.rename(columns={"term": "变量/效应", "coef": "系数", "se": "标准误", "p": "p 值", "lower": "95%下限", "upper": "95%上限"})
    if "p 值" in renamed:
        renamed["显著性"] = renamed["p 值"].map(lambda p: "***" if p < .01 else ("**" if p < .05 else ("*" if p < .1 else "")))
    st.dataframe(renamed, width="stretch", hide_index=True, column_config={c: st.column_config.NumberColumn(format="%.6f") for c in ["系数", "标准误", "p 值", "95%下限", "95%上限"]})


@st.cache_data(show_spinner=False)
def cached_esr_identification_candidates(data, y, treatment, candidates):
    return esr_identification_candidates(data, y, treatment, candidates)


def pipeline_panel(ws, raw):
    st.subheader("1 · 数据清洗与变量加工")
    clean_labels = {"删除完全重复行": "deduplicate", "删除所选变量缺失的行": "drop_missing", "数字文本转数值": "numeric", "均值插补": "mean", "中位数插补": "median", "按时间插值": "interpolate", "缩尾": "winsor", "按数值范围保留样本": "filter"}
    trans_labels = {"取对数 ln(x)": "log", "平方项": "square", "相乘/交互项": "interact", "标准化": "zscore", "滞后一期": "lag", "一阶差分": "diff", "二元重编码": "recode"}
    if "pending_rule_edit" in ws:
        idx = ws.pop("pending_rule_edit")
        edit = ws["steps"][idx]
        ws["editing_rule"] = idx
        if edit["op"] in clean_labels.values():
            st.session_state["clean_method"] = next(k for k, v in clean_labels.items() if v == edit["op"])
            for field, key in {"cols": "clean_cols", "time": "clean_time", "pct": "clean_pct", "lower": "clean_lower", "upper": "clean_upper"}.items():
                if field in edit:
                    st.session_state[key] = edit[field]
            st.session_state["clean_group"] = edit.get("group") or "不分组"
        else:
            st.session_state["trans_method"] = next(k for k, v in trans_labels.items() if v == edit["op"])
            st.session_state["trans_col"] = edit["cols"][0]
            st.session_state["trans_name"] = edit["name"]
            if len(edit["cols"]) > 1:
                st.session_state["trans_second"] = edit["cols"][1]
            for field in ["time", "interval", "one"]:
                if field in edit:
                    st.session_state["trans_"+field] = edit[field]
            st.session_state["trans_group"] = edit.get("group") or "单一时间序列"
    editing = ws.get("editing_rule")
    if editing is not None:
        st.info(f"正在编辑第 {editing+1} 个操作")
        if st.button("取消编辑", key="cancel_rule_edit"):
            ws.pop("editing_rule", None); st.rerun()
    def commit_step(step):
        proposed = list(ws.get("steps", []))
        if editing is None:
            proposed.append(step)
        else:
            proposed[editing] = step
        apply_steps(raw, proposed)
        ws["steps"] = proposed
        ws.pop("editing_rule", None)
        invalidate(ws)
    try:
        data, audit = apply_steps(raw, ws.get("steps", []))
    except Exception as e:
        st.error(f"规则无法应用：{e}")
        data, audit = raw.copy(), []
    st.caption(f"原始 {len(raw)} 行 · 当前 {len(data)} 行 · {len(data.columns)} 列 · 缺失值 {int(data.isna().sum().sum())} 个")
    with st.expander("数据预览"):
        st.dataframe(data.head(20), width="stretch")
    with st.expander("添加清洗规则"):
        labels = clean_labels
        method = choose("清洗方式", labels, "clean_method")
        op = labels[method]
        numeric = list(data.select_dtypes(include="number").columns)
        available = numeric if op in {"mean", "median", "interpolate", "winsor", "filter"} else list(data.columns)
        selected = multiple("处理变量", available, "clean_cols") if op != "deduplicate" else []
        step = {"op": op, "cols": selected}
        if op in {"mean", "median", "interpolate"}:
            group = choose("分组列", ["不分组"]+[c for c in data if c not in selected], "clean_group")
            step["group"] = "" if group == "不分组" else group
        if op == "interpolate":
            step["time"] = choose("时间列", [c for c in numeric if c not in selected and c != step["group"]], "clean_time")
        if op == "winsor":
            step["pct"] = st.select_slider("双侧缩尾比例", options=[.01, .025, .05, .1], format_func=lambda v: f"{v*100:g}%", key="clean_pct")
        if op == "filter" and selected:
            selected = selected[:1]; step["cols"] = selected
            step["lower"] = number("保留下界", "clean_lower", float(data[selected[0]].min()))
            step["upper"] = number("保留上界", "clean_upper", float(data[selected[0]].max()))
        if st.button("应用并加入规则", key="clean_add"):
            try:
                if op != "deduplicate" and not selected:
                    raise ValueError("请选择处理变量")
                commit_step(step); st.rerun()
            except Exception as e:
                st.error(str(e))
    with st.expander("生成新变量"):
        labels = trans_labels
        method = choose("生成方式", labels, "trans_method")
        op = labels[method]
        choices = list(data.columns) if op == "recode" else list(data.select_dtypes(include="number").columns)
        c = choose("来源变量", choices, "trans_col")
        name = st.text_input("新变量名", key="trans_name")
        step = {"op": op, "cols": [c], "name": name}
        if op == "interact":
            step["cols"].append(choose("相乘变量", choices, "trans_second"))
        if op == "recode" and c:
            one = choose("编码为 1 的取值", data[c].dropna().unique(), "trans_one")
            step["one"] = one.item() if isinstance(one, np.generic) else one
        if op in {"lag", "diff"}:
            group = choose("个体列", ["单一时间序列"]+[v for v in data if v != c], "trans_group")
            step["group"] = "" if group == "单一时间序列" else group
            step["time"] = choose("数值时间列", [v for v in choices if v not in [c, step["group"]]], "trans_time")
            step["interval"] = number("一期的时间间隔", "trans_interval", 1.0, min_value=.000001)
        if st.button("生成变量", key="trans_add"):
            try:
                commit_step(step); st.rerun()
            except Exception as e:
                st.error(str(e))
    if ws.get("steps"):
        with st.expander(f"已应用操作 · {len(ws['steps'])}"):
            st.dataframe(pd.DataFrame(audit), hide_index=True, width="stretch")
            index = choose("选择操作", range(len(ws["steps"])), "rule_index")
            a, b, c, e, f = st.columns(5)
            if f.button("编辑", icon=":material/edit:", key="rule_edit"):
                ws["pending_rule_edit"] = index; st.rerun()
            action = None
            if a.button("删除", icon=":material/delete:", key="rule_delete"):
                action = "delete"
            if b.button("上移", icon=":material/arrow_upward:", key="rule_up"):
                action = "up"
            if c.button("下移", icon=":material/arrow_downward:", key="rule_down"):
                action = "down"
            if e.button("撤销最后一步", icon=":material/undo:", key="rule_undo"):
                action = "undo"
            if action:
                proposed = list(ws["steps"])
                if action == "delete":
                    proposed.pop(index)
                elif action == "undo":
                    proposed.pop()
                elif action == "up" and index > 0:
                    proposed[index-1], proposed[index] = proposed[index], proposed[index-1]
                elif action == "down" and index < len(proposed)-1:
                    proposed[index+1], proposed[index] = proposed[index], proposed[index+1]
                try:
                    apply_steps(raw, proposed)
                    ws["steps"] = proposed; ws.pop("editing_rule", None); invalidate(ws); st.rerun()
                except Exception as exc:
                    st.error(f"后续操作依赖此规则：{exc}")
    return data


def specification(data):
    st.subheader("2 · 数据结构与模型")
    cols = list(data.columns)
    numeric = list(data.select_dtypes(include="number").columns)
    structure = choose("数据结构", ["横截面", "面板", "时间序列"], "cfg_structure")
    entity = time_col = ""
    if structure == "面板":
        a, b = st.columns(2)
        with a:
            entity = choose("个体 ID", cols, "cfg_entity")
        with b:
            time_col = choose("时间", [c for c in numeric if c != entity], "cfg_time")
        try:
            validate_panel(data, entity, time_col)
        except ValueError as exc:
            st.error(str(exc)); return None
    if structure == "时间序列":
        time_col = choose("时间列", numeric, "cfg_time")
        time_series_ui(data, time_col)
        return None
    excluded = [entity, time_col]
    y = choose("被解释变量 Y", [c for c in numeric if c not in excluded], "cfg_y")
    if not y:
        st.info("请先准备数值型结果变量")
        return None
    if st.session_state.get("_previous_y") != y:
        st.session_state["cfg_y_type"] = infer_type(data[y])
        st.session_state["_previous_y"] = y
    kind = choose("Y 的含义", ["连续", "二元", "有序", "计数", "受限连续"], "cfg_y_type")
    other = [c for c in cols if c not in excluded+[y]]
    bins = [c for c in other if binary(data[c])]
    if structure == "面板":
        models = ["FE", "RE", "FE+RE", "Pooled OLS"]
        if kind == "二元":
            models += ["Logit", "Probit", "Logit+Probit", "LPM"]
        elif kind == "计数":
            models += ["Poisson", "负二项"]
        if bins:
            models += ["DID", "PSM-DID"]
    elif kind == "二元":
        models = ["Logit", "Probit", "Logit+Probit", "LPM"]
    elif kind == "有序":
        models = ["Ordered Logit", "Ordered Probit"]
    elif kind == "计数":
        models = ["Poisson", "负二项", "OLS"]
    else:
        models = ["OLS", "Tobit", "Sharp RD", "Fuzzy RD"]
        if bins:
            models += ["ESR", "Heckman"]
    if structure == "横截面":
        models += ["IV/2SLS"] + (["PSM"] if bins else [])
        if kind in {"连续", "受限连续"} and any(2 <= data[c].nunique() <= 20 for c in other):
            models += ["ANOVA", "交互效应"]
    model = choose("模型", models, "cfg_model")
    cat_default = [c for c in other if not pd.api.types.is_numeric_dtype(data[c])]
    cats = multiple("按分类变量处理", other, "cfg_categorical", cat_default)
    s = ModelSpec(model=model, y=y, categorical=cats, entity=entity, time=time_col)
    if model in {"ANOVA", "交互效应"}:
        s.group = choose("分组变量", [c for c in other if 2 <= data[c].nunique() <= 20], "cfg_group")
        s.categorical = unique(s.categorical+[s.group])
        levels = sorted(data[s.group].dropna().astype(str).unique())
        st.caption(f"参考组：{levels[0]}")
        s.contrast = choose("比较组", levels[1:], "cfg_contrast")
    if model in {"FE", "FE+RE"}:
        a, b = st.columns(2)
        with a:
            s.entity_effects = st.checkbox("个体固定效应", value=True, key="cfg_entity_effects")
        with b:
            s.time_effects = st.checkbox("时间固定效应", value=True, key="cfg_time_effects")
        if not s.entity_effects and not s.time_effects:
            st.warning("FE 至少需要选择个体固定效应或时间固定效应")
            return None
        if model == "FE+RE" and not s.entity_effects:
            st.warning("FE+RE 比较需要启用个体固定效应")
            return None
    elif model in {"DID", "PSM-DID"}:
        s.entity_effects = True
        s.time_effects = st.checkbox("时间固定效应", value=True, key="cfg_time_effects")
    if model in {"DID", "PSM-DID", "PSM", "ESR", "Heckman", "Fuzzy RD"}:
        label = "选择变量 D（1=选择，0=未选择）" if model in {"ESR", "Heckman"} else "处理变量 D（1=处理，0=对照）"
        s.treatment = choose(label, bins, "cfg_treatment")
        if not s.treatment:
            st.warning("请先将处理/选择变量明确编码为 0/1")
            return None
    if model == "ESR":
        screened = cached_esr_identification_candidates(
            data,
            y,
            s.treatment,
            tuple(c for c in other if c != s.treatment),
        )
        eligible = screened["变量"].tolist()
        if not eligible:
            st.warning("没有变量同时通过识别变量筛选：与 D 相关 p<0.10，控制 D 后对 Y 的直接关系 p≥0.10。")
            return None
        exclusion = choose(
            "识别变量 Z（仅进入选择方程）",
            eligible,
            "cfg_esr_exclusion",
        )
        if not exclusion:
            st.warning("ESR 需要先选择一个识别变量")
            return None
        s.instruments = [exclusion]
        with st.expander("识别变量筛选结果"):
            st.dataframe(screened, hide_index=True, width="stretch")
            st.caption("相关性使用稳健标准误的一阶段线性概率检验；排他性使用控制 D 后的无直接关系代理检验。")
    if "DID" in model:
        post_mode = choose("Post 来源", ["政策时间", "已有 Post"], "cfg_post_mode")
        if post_mode == "已有 Post":
            s.post = choose("Post 变量", [v for v in bins if v != s.treatment], "cfg_post")
            if not s.post:
                st.warning("没有可用 Post，请选择政策时间")
                return None
        else:
            s.policy = float(choose("政策开始时间", sorted(data[time_col].dropna().unique()), "cfg_policy"))
    if model in {"Sharp RD", "Fuzzy RD"}:
        s.running = choose("断点变量", [v for v in numeric if v not in excluded+[y, s.treatment]], "cfg_running")
        if not s.running:
            st.warning("请准备数值型断点变量")
            return None
        a, b, c = st.columns(3)
        with a:
            s.cutoff = number("断点值", "cfg_cutoff", float(data[s.running].median()))
        with b:
            s.bandwidth = number("带宽（0=全样本）", "cfg_bandwidth", 0.0, min_value=0.0)
        with c:
            s.degree = choose("多项式阶数", [1, 2], "cfg_degree")
    special = {"DID", "PSM-DID", "PSM", "ESR", "Heckman", "Sharp RD", "Fuzzy RD", "ANOVA"}
    selectable = [c for c in other if c not in [s.treatment, s.post, s.running, s.group]]
    if model not in special or model == "Heckman":
        core_options = [c for c in selectable if c not in cats]
        if model == "IV/2SLS":
            core = choose("内生变量", core_options, "cfg_endog")
            s.core = [core] if core else []
        else:
            s.core = multiple("核心 X", core_options, "cfg_core", core_options[:1])
    if model == "IV/2SLS":
        s.instruments = multiple("排除工具变量", [c for c in selectable if c not in s.core], "cfg_instruments")
    control_options = [c for c in selectable if c not in s.core+s.instruments]
    s.controls = multiple("必选控制变量", control_options, "cfg_controls")
    pool = multiple("候选控制变量", [c for c in control_options if c not in s.controls], "cfg_pool")
    if model == "ESR":
        s.selection = unique(s.controls+s.instruments)
        s.effect = choose("搜索效应", ["ATT", "ATU"], "cfg_effect")
        st.caption("每个控制变量组合同时进入 D=0、D=1 结果方程和选择方程；识别变量 Z 只进入选择方程。")
    elif model == "Heckman":
        selection_options = [c for c in other if c != s.treatment]
        s.selection = multiple("选择方程变量（含你指定的排除变量）", selection_options, "cfg_selection", s.controls)
    if model == "Tobit":
        boundary = choose("审查边界", ["下界", "上界", "双侧"], "cfg_boundary")
        if boundary in {"下界", "双侧"}:
            s.left = number("下界值", "cfg_left", 0.0)
        if boundary in {"上界", "双侧"}:
            s.right = number("上界值", "cfg_right", 1.0)
    if model in {"PSM", "PSM-DID"}:
        s.caliper = number("倾向得分卡尺", "cfg_caliper", .1, min_value=.0001, max_value=1.0)
        st.caption("最近邻、1:1、不放回匹配；PSM-DID 按个体政策前协变量均值匹配。")
    if model == "PSM":
        s.se = "ordinary"
    else:
        mapping = {"异方差稳健": "robust", "聚类": "cluster", "普通 / 信息矩阵": "ordinary"}
        se_label = choose("标准误", mapping, "cfg_se", "聚类" if structure == "面板" else "异方差稳健")
        s.se = mapping[se_label]
        if s.se == "cluster":
            s.cluster = choose("聚类列", [c for c in cols if c != y], "cfg_cluster", entity)
    return s, pool


def search_tick(ws):
    job = ws.get("job")
    if not job or job["done"]:
        return
    cancel = st.button("停止搜索并保留结果", icon=":material/stop:", key="cancel_search")
    if cancel:
        job["done"] = True; st.rerun()
    limit = min(job["index"]+1, len(job["specs"]))
    while job["index"] < limit:
        spec = job["specs"][job["index"]]
        try:
            result = fit_model(job["data"], spec)
            score = rank_fit(result, job["joint"])
            result.fitted = None; result.design = None; result.response = None
            for _, previous in job["fits"]:
                if result.sample.index.equals(previous.sample.index) and result.sample.columns.equals(previous.sample.columns) and result.sample.equals(previous.sample):
                    result.sample = previous.sample
                    break
            job["fits"].append((score, result))
            job["fits"].sort(key=lambda pair: pair[0])
            job["fits"] = job["fits"][:50]
        except Exception as exc:
            reason = str(exc)[:150]
            job["failures"][reason] = job["failures"].get(reason, 0)+1
        job["index"] += 1
    ws["fits"] = [f for _, f in job["fits"]]
    st.progress(job["index"]/max(len(job["specs"]), 1), text=f"已完成 {job['index']}/{len(job['specs'])} · 有效 {job['index']-sum(job['failures'].values())} · 跳过 {sum(job['failures'].values())}")
    if job["index"] >= len(job["specs"]):
        job["done"] = True
        st.rerun()


def search_controls(ws, data, s, pool):
    st.subheader("3 · 估计与组合搜索")
    a, b, c = st.columns(3)
    with a:
        minimum = number("最少候选控制数", "cfg_min", 0, min_value=0, max_value=20)
    with b:
        maximum = number("最多候选控制数", "cfg_max", min(3, len(pool)), min_value=0, max_value=20)
    with c:
        budget = number("最多尝试组合数", "cfg_budget", 50 if s.model == "ESR" else 200, min_value=1, max_value=10000)
    sample_mode = choose("样本规则", ["固定样本（整个候选池共同有效）", "每个组合自身有效样本"], "cfg_sample_mode")
    joint = True
    if len(s.core) > 1:
        joint = choose("核心 X 搜索方式", ["联合显著", "各自搜索"], "cfg_joint") == "联合显著"
    total = sum(math.comb(len(pool), k) for k in range(minimum, min(maximum, len(pool))+1))
    st.caption(f"候选组合 {total:,} · 本次最多尝试 {min(total, budget):,} · 无候选变量时直接估计必选设定")
    job = ws.get("job")
    active = job and not job["done"]
    if st.button("开始估计" if not pool else "开始搜索", type="primary", disabled=bool(active), key="run_search"):
        try:
            if s.model == "ESR" and not s.instruments:
                raise ValueError("请选择识别变量 Z")
            if s.model in {"ESR", "Heckman"} and not s.selection:
                raise ValueError("请选择选择方程变量")
            if s.model == "IV/2SLS" and (not s.core or not s.instruments):
                raise ValueError("请选择内生变量和工具变量")
            specs = list(candidate_specs(s, pool, minimum, maximum, budget, joint=joint))
            if not specs:
                raise ValueError("没有可运行组合，请调整候选控制数")
            search_data = fixed_sample(data, s, pool) if sample_mode.startswith("固定") else data.copy()
            invalidate(ws)
            ws["job"] = {"data": search_data, "specs": specs, "index": 0, "fits": [], "failures": {}, "joint": joint, "done": False, "signature": json.dumps(asdict(s), sort_keys=True, default=str)}
        except Exception as e:
            st.error(str(e))
    job = ws.get("job")
    if job and not job["done"]:
        st.fragment(run_every=1)(search_tick)(ws)
        job = ws.get("job")
    if job and job["done"]:
        st.caption(f"已完成 {job['index']} 个组合 · 保留 {len(ws.get('fits', []))} 个有效结果")
        if job["failures"]:
            with st.expander("跳过的组合"):
                st.dataframe(pd.DataFrame([{"原因": k, "数量": v} for k, v in job["failures"].items()]), hide_index=True)


def followup_ui(ws, fit):
    s, sample = fit.spec, fit.sample
    st.subheader("继续分析")
    label = {"变量": s.y, "模型": s.model, "样本量": len(sample), "标准误": s.se}
    st.caption(" · ".join(f"{k}：{v}" for k, v in label.items()))
    tabs = st.tabs(["机制 / 中介", "调节", "异质性", "稳健性"])
    reserved = unique([s.y, s.entity, s.time, s.treatment, s.post, s.running]+s.core)
    options = [c for c in sample.select_dtypes(include="number") if c not in reserved and not c.startswith("__")]
    with tabs[0]:
        med = choose("机制变量 M", options, "follow_med")
        indirect_supported = s.model in {"OLS", "Pooled OLS", "FE", "RE", "FE+RE", "DID", "PSM-DID", "Sharp RD"}
        decomposition = st.checkbox("Bootstrap 间接效应", disabled=not indirect_supported, key="follow_bootstrap")
        reps = number("Bootstrap 重复次数", "follow_reps", 199, min_value=49, max_value=1999, step=50) if decomposition and indirect_supported else 0
        if not indirect_supported:
            st.caption("该模型进行渠道路径检验：同样本基准效应、对 M 的效应、加入 M 后的结果方程及 M 的系数。")
        if st.button("运行机制分析", disabled=not med, key="run_med"):
            try:
                common = sample.dropna(subset=[med])
                analysis = mechanism_analysis(common, s, med)
                result = {"同样本基准效应": analysis["baseline"], "对机制变量的效应": analysis["channel"]}
                for med_effect in analysis["mediator_effects"]:
                    adjusted = replace(analysis["adjusted"])
                    adjusted.effect = med_effect
                    result[f"加入机制变量后 · {med_effect['term']}"] = adjusted
                ws["follow"] = result
                ws.pop("indirect", None)
                if reps:
                    with st.spinner("计算间接效应区间..."):
                        ws["indirect"] = mediation_bootstrap(common, s, med, reps)
            except Exception as e:
                st.error(f"本次机制分析未成功：{e}")
    with tabs[1]:
        mod = choose("调节变量 M", options, "follow_mod")
        direct = s.model in {"OLS", "Pooled OLS", "FE", "RE", "FE+RE", "Logit", "Probit", "Logit+Probit", "Ordered Logit", "Ordered Probit", "Poisson", "负二项", "LPM", "DID", "PSM-DID", "Sharp RD", "Tobit", "Heckman", "ANOVA", "交互效应"} and (bool(s.core) or s.model in {"DID", "PSM-DID", "Sharp RD"})
        mode = choose("调节检验方式", ["交互项" if direct else "高低组效应差异", "高低组效应差异"] if direct else ["高低组效应差异"], "follow_mod_mode")
        if mode == "高低组效应差异":
            st.caption("沿用当前模型和有效样本，按 M 的中位数分组，并检验两组效应差异。")
        if st.button("运行调节分析", disabled=not mod, key="run_mod"):
            try:
                if mode == "交互项":
                    focus = "__did" if "DID" in s.model else ("__rd_z" if s.model == "Sharp RD" else s.core[0])
                    candidate = replace(s, interactions=s.interactions+[[focus, mod]])
                    ws["follow"] = {"调节效应": fit_model(sample, candidate)}
                else:
                    results, contrast = grouped_moderation(sample, s, mod)
                    contrast_fit = replace(results["高组"])
                    contrast_fit.effect = contrast
                    ws["follow"] = {"低组效应": results["低组"], "高组效应": results["高组"], "组间差异": contrast_fit}
            except Exception as e:
                st.error(str(e))
    with tabs[2]:
        group = choose("分组变量", [c for c in sample if c not in [s.y, s.entity, s.time] and not c.startswith("__")], "follow_group")
        method = choose("分组方式", ["按取值", "中位数", "三分位数", "四分位数"], "follow_group_method")
        if st.button("运行分组回归", disabled=not group, key="run_group"):
            try:
                valid = sample.dropna(subset=[group]).copy()
                if method == "按取值":
                    if valid[group].nunique() > 20:
                        raise ValueError("分组超过 20 个，请选择分位数方式")
                    groups = valid.groupby(group, observed=True)
                else:
                    q = {"中位数": 2, "三分位数": 3, "四分位数": 4}[method]
                    groups = valid.groupby(pd.qcut(valid[group], q=q, duplicates="drop"), observed=True)
                results = {}
                failures = []
                for name, part in groups:
                    try:
                        results[str(name)] = fit_model(part, s)
                    except Exception as e:
                        failures.append(f"{name}：{e}")
                ws["follow"] = results
                for error in failures:
                    st.warning(error)
            except Exception as e:
                st.error(str(e))
    with tabs[3]:
        mode = choose("稳健性方式", ["仅核心 X 缩尾", "Y 缩尾", "替换 Y", "替换核心 X", "按变量范围筛选"], "follow_robust")
        pct = .01
        replacement = None
        if "缩尾" in mode:
            pct = st.select_slider("缩尾比例", options=[.01, .05, .1], format_func=lambda v: f"{v*100:g}%", key="follow_pct")
        elif mode in {"替换 Y", "替换核心 X"}:
            replacement = choose("替换变量", [c for c in sample.select_dtypes(include="number") if not c.startswith("__")], "follow_replacement")
        else:
            replacement = choose("筛选变量", [c for c in sample.select_dtypes(include="number") if not c.startswith("__")], "follow_filter")
            lo = number("保留 ≥", "follow_lower", float(sample[replacement].min())) if replacement else 0
            hi = number("保留 ≤", "follow_upper", float(sample[replacement].max())) if replacement else 0
        if st.button("运行稳健性分析", key="run_robust"):
            try:
                changed = sample.copy(deep=True)
                candidate = s
                if "缩尾" in mode:
                    targets = s.core if mode.startswith("仅") else [s.y]
                    if not targets:
                        raise ValueError("此模型没有普通核心 X，请选择其他稳健性方式")
                    for c in targets:
                        if binary(changed[c]):
                            raise ValueError("二元变量无需缩尾")
                        changed[c] = changed[c].clip(*changed[c].quantile([pct, 1-pct]))
                elif mode == "替换 Y":
                    candidate = replace(s, y=replacement)
                elif mode == "替换核心 X":
                    if len(s.core) != 1:
                        raise ValueError("此操作需要一个普通核心 X")
                    candidate = replace(s, core=[replacement])
                else:
                    changed = changed[changed[replacement].between(lo, hi)]
                ws["follow"] = {"基准": fit, mode: fit_model(changed, candidate)}
            except Exception as e:
                st.error(str(e))
    if ws.get("follow"):
        rows = []
        for title, result in ws["follow"].items():
            rows.append({"分析": title, "N": len(result.sample), **result.effect})
        follow_token = fingerprint(json.dumps(rows, sort_keys=True, default=str).encode())
        if ws.get("follow_token") != follow_token:
            ws["follow_token"] = follow_token
            ws.pop("follow_bundle", None)
        display_table(pd.DataFrame(rows))
        if ws.get("indirect"):
            display_table(pd.DataFrame([ws["indirect"]]))
        st.download_button("下载后续分析表", pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig"), "followup.csv", key="follow_download")
        with st.expander("导出后续模型"):
            selected = choose("后续模型", list(ws["follow"]), "follow_export_choice")
            if st.button("生成后续模型复现包", key="follow_build_export"):
                try:
                    f = ws["follow"][selected]
                    # This source is the inherited, already-processed estimation sample.
                    source = f.sample.copy()
                    used = required_columns(f.spec)
                    source = source[unique([c for c in source if not c.startswith("__")]+used)]
                    packed, do = reproducibility_bundle(source, [], f, ws["hash"])
                    ws["follow_bundle"] = packed
                except Exception as exc:
                    st.error(str(exc))
            if ws.get("follow_bundle"):
                st.download_button("下载后续复现包（含本次分析样本）", ws["follow_bundle"], "tihu_followup.zip", key="follow_download_bundle")


def results_ui(ws, raw):
    fits = ws.get("fits", [])
    if not fits:
        return
    st.subheader("4 · 结果与复现")
    summary = pd.DataFrame([{"编号": i+1, "模型": f.spec.model, "目标": f.effect["term"], "控制变量": "、".join(f.spec.controls) or "无", "N": len(f.sample), "系数": f.effect["coef"], "p 值": f.effect["p"]} for i, f in enumerate(fits)])
    st.dataframe(summary, hide_index=True, width="stretch")
    result_number = choose("选定结果", range(1, len(fits)+1), "selected_result")
    idx = result_number-1
    fit = fits[idx]
    token = fingerprint(json.dumps(asdict(fit.spec), sort_keys=True, default=str).encode())
    if ws.get("selected_token") != token:
        ws["selected_token"] = token
        ws.pop("follow", None); ws.pop("bundle", None)
    st.caption(f"模型 {fit.spec.model} · 有效 N={len(fit.sample)} · 标准误 {fit.spec.se}" + (f" · 聚类 {fit.spec.cluster}" if fit.spec.cluster else ""))
    display_table(fit.table)
    for key, value in fit.details.items():
        if isinstance(value, pd.DataFrame):
            st.write(key)
            display_table(value) if "term" in value else st.dataframe(value, width="stretch")
        elif key not in {"pairs", "ps"}:
            st.caption(f"{key}: {value}")
    try:
        do, _ = stata_script(raw, ws.get("steps", []), fit)
        with st.expander("Stata 复现代码"):
            st.code(do, language="stata")
    except Exception as e:
        st.warning(f"Stata 代码暂时无法生成：{e}")
    if st.button("生成完整复现文件包", icon=":material/download:", key="build_bundle"):
        try:
            ws["bundle"] = reproducibility_bundle(raw, ws.get("steps", []), fit, ws["hash"])
        except Exception as e:
            st.error(f"导出失败：{e}")
    if ws.get("bundle"):
        bundle, do = ws["bundle"]
        st.download_button("下载复现包（包含本次数据）", bundle, "tihu_reproducible.zip", mime="application/zip", key="download_bundle")
        st.download_button("下载 Stata .do", do, "analysis.do", key="download_do")
    with st.expander("对比多个结果"):
        selected = st.multiselect("结果编号", range(len(fits)), format_func=lambda i: str(i+1), key="compare_results")
        if selected:
            st.dataframe(summary.iloc[selected], hide_index=True, width="stretch")
            st.download_button("下载对比表", summary.iloc[selected].to_csv(index=False).encode("utf-8-sig"), "comparison.csv")
    followup_ui(ws, fit)


def time_series_ui(data, time_col):
    if not time_col:
        return
    numeric = [c for c in data.select_dtypes(include="number") if c != time_col]
    y = choose("时间序列 Y", numeric, "cfg_ts_y")
    exog = multiple("其他变量", [c for c in numeric if c != y], "cfg_ts_exog")
    model = choose("时间序列模型", ["ARIMA", "AR", "MA", "ARMA", "VAR", "GARCH"], "cfg_ts_model")
    if st.button("估计时间序列", key="run_ts"):
        try:
            d = data[unique([time_col, y]+exog)].dropna().sort_values(time_col)
            if d[time_col].duplicated().any() or len(d) < 15:
                raise ValueError("时间重复或有效观测不足")
            gaps = d[time_col].diff().dropna()
            if gaps.nunique() != 1:
                raise ValueError("时间间隔不规则，请先处理缺失时期")
            adf = sm.tsa.adfuller(d[y])
            st.caption(f"ADF p={adf[1]:.4f}")
            if model == "VAR":
                if not exog:
                    raise ValueError("VAR 至少需要另一个变量")
                r = sm.tsa.VAR(d[[y]+exog]).fit(maxlags=2, ic="aic")
                if r.k_ar == 0:
                    r = sm.tsa.VAR(d[[y]+exog]).fit(1)
                table = r.params
            elif model == "GARCH":
                from arch import arch_model
                r = arch_model(d[y], p=1, q=1).fit(disp="off")
                table = pd.DataFrame({"系数": r.params, "标准误": r.std_err})
            else:
                order = {"ARIMA": (1, 1, 1), "AR": (1, 0, 0), "MA": (0, 0, 1), "ARMA": (1, 0, 1)}[model]
                r = sm.tsa.arima.ARIMA(d[y], order=order, exog=d[exog] if exog else None).fit()
                table = pd.DataFrame({"系数": r.params, "标准误": r.bse, "p": r.pvalues})
            names = {c: f"v{i+1:04d}" for i, c in enumerate(d.columns)}
            out = BytesIO()
            rhs = " ".join(names[c] for c in exog)
            lines = ["version 16", "clear all", 'use "timeseries.dta", clear', f"tsset {names[time_col]}, delta({float(gaps.iloc[0]):.17g})", f"dfuller {names[y]}"]
            if model == "VAR":
                lines += [f"var {names[y]} {rhs}, lags(1/{r.k_ar})"]
            elif model == "GARCH":
                lines += [f"arch {names[y]}, arch(1) garch(1)"]
            else:
                lines += [f"arima {names[y]} {rhs}, arima({order[0]},{order[1]},{order[2]})"]
            with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr("timeseries.dta", dta_bytes(d.rename(columns=names)))
                z.writestr("analysis.do", "\n".join(lines))
                z.writestr("result.csv", table.to_csv())
                z.writestr("column_names.json", json.dumps(names, ensure_ascii=False))
                z.writestr("README.txt", "该时间序列复现包包含当前处理后的有效分析样本。请在解压目录运行 analysis.do。模型初始化等实现细节可能造成软件间数值差异。")
            st.session_state["ts_result"] = {"model": model, "table": table, "aic": float(r.aic), "text": str(r.summary()), "bundle": out.getvalue(), "data": d, "y": y, "exog": exog, "time": time_col, "names": names}
        except Exception as e:
            st.error(str(e))
    if st.session_state.get("ts_result"):
        saved = st.session_state["ts_result"]
        st.caption(f"上次估计：{saved['model']} · AIC={saved['aic']:.4f}")
        st.dataframe(saved["table"], width="stretch")
        st.download_button("下载时间序列复现包（含当前分析样本）", saved["bundle"], "tihu_timeseries.zip", key="ts_download")
        st.subheader("继续分析")
        st.caption("时间序列路径使用含一期自身滞后的动态方程，并用 HAC(1) 标准误检验机制和调节。")
        dynamic_options = [c for c in data.select_dtypes(include="number") if c not in [time_col, y]]
        if not dynamic_options:
            st.info("需要至少一个数值型解释变量后才能进行机制或调节分析。")
            return
        core = choose("核心时间序列 X", dynamic_options, "ts_follow_core")
        mediators = [c for c in dynamic_options if c != core]
        tabs = st.tabs(["机制 / 中介", "调节"])
        def dynamic_frame(mediator):
            cols = unique([time_col, y, core, mediator]+saved["exog"])
            d = data[cols].dropna().sort_values(time_col).copy()
            d["__lag_y"] = d[y].shift(1)
            d["__lag_m"] = d[mediator].shift(1)
            return d.dropna()
        with tabs[0]:
            mediator = choose("机制变量 M", mediators, "ts_follow_med")
            if st.button("运行时间序列机制分析", disabled=not mediator, key="ts_run_med"):
                try:
                    d = dynamic_frame(mediator)
                    controls = [c for c in saved["exog"] if c not in [core, mediator]]
                    a_vars = unique([core, "__lag_m"]+controls)
                    b_vars = unique([core, mediator, "__lag_y"]+controls)
                    a = sm.OLS(d[mediator], sm.add_constant(d[a_vars])).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
                    b = sm.OLS(d[y], sm.add_constant(d[b_vars])).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
                    result = pd.DataFrame([
                        {"路径": "X→M", "系数": a.params[core], "标准误": a.bse[core], "p 值": a.pvalues[core]},
                        {"路径": "M→Y | X", "系数": b.params[mediator], "标准误": b.bse[mediator], "p 值": b.pvalues[mediator]},
                        {"路径": "X→Y | M", "系数": b.params[core], "标准误": b.bse[core], "p 值": b.pvalues[core]},
                    ])
                    st.session_state["ts_follow_result"] = result
                    names = saved["names"]
                    other = " ".join(names[c] for c in controls)
                    st.session_state["ts_follow_do"] = f"use \"timeseries.dta\", clear\ntsset {names[time_col]}\nnewey {names[mediator]} {names[core]} L.{names[mediator]} {other}, lag(1)\nnewey {names[y]} {names[core]} {names[mediator]} L.{names[y]} {other}, lag(1)\n"
                except Exception as e:
                    st.error(str(e))
        with tabs[1]:
            moderator = choose("调节变量 M", mediators, "ts_follow_mod")
            if st.button("运行时间序列调节分析", disabled=not moderator, key="ts_run_mod"):
                try:
                    d = dynamic_frame(moderator)
                    d["__interaction"] = d[core]*d[moderator]
                    controls = [c for c in saved["exog"] if c not in [core, moderator]]
                    variables = unique([core, moderator, "__interaction", "__lag_y"]+controls)
                    r = sm.OLS(d[y], sm.add_constant(d[variables])).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
                    st.session_state["ts_follow_result"] = pd.DataFrame([{"路径": "X×M", "系数": r.params["__interaction"], "标准误": r.bse["__interaction"], "p 值": r.pvalues["__interaction"]}])
                    names = saved["names"]
                    other = " ".join(names[c] for c in controls)
                    st.session_state["ts_follow_do"] = f"use \"timeseries.dta\", clear\ntsset {names[time_col]}\ngen double __interaction={names[core]}*{names[moderator]}\nnewey {names[y]} {names[core]} {names[moderator]} __interaction L.{names[y]} {other}, lag(1)\n"
                except Exception as e:
                    st.error(str(e))
        if st.session_state.get("ts_follow_result") is not None:
            st.dataframe(st.session_state["ts_follow_result"], hide_index=True, width="stretch")
            st.download_button("下载后续结果", st.session_state["ts_follow_result"].to_csv(index=False).encode("utf-8-sig"), "timeseries_followup.csv", key="ts_follow_download")
            st.download_button("下载后续 Stata 代码", st.session_state["ts_follow_do"], "timeseries_followup.do", key="ts_follow_do_download")


def render_workbench():
    ws = st.session_state.setdefault("workspace_v2", {"steps": []})
    with st.sidebar:
        st.caption(f"版本 {VERSION}")
        st.caption("上传数据在本次服务器会话内处理；本程序不主动持久保存用户数据，也不向 AI 服务发送数据。")
        st.button("清除本次数据和结果", icon=":material/delete:", on_click=reset_session, key="clear_all")
    st.subheader("上传数据")
    uploaded = st.file_uploader("数据文件", type=["dta", "csv", "xlsx"], key="data_upload")
    if uploaded is None:
        return
    content = uploaded.getvalue()
    if len(content) > 50*1024*1024:
        st.error("当前版本单个文件上限为 50 MB，请先选择需要的样本与变量")
        return
    digest = fingerprint(content)
    if ws.get("hash") != digest:
        try:
            if uploaded.name.lower().endswith(".dta"):
                raw = pd.read_stata(BytesIO(content), convert_categoricals=False)
            elif uploaded.name.lower().endswith(".csv"):
                try:
                    raw = pd.read_csv(BytesIO(content))
                except UnicodeDecodeError:
                    raw = pd.read_csv(BytesIO(content), encoding="gb18030")
            else:
                raw = pd.read_excel(BytesIO(content))
            raw.columns = raw.columns.astype(str)
            if raw.columns.duplicated().any() or raw.empty:
                raise ValueError("数据为空或包含重名变量")
            if any(c.startswith("__") for c in raw.columns):
                raise ValueError("双下划线前缀为内部保留变量名，请重命名这些列")
            ws.clear(); ws.update(hash=digest, raw=raw, name=uploaded.name, steps=[])
            st.session_state.pop("ts_result", None)
            st.session_state.pop("ts_follow_result", None)
            st.session_state.pop("ts_follow_do", None)
            for k in list(st.session_state):
                if k.startswith(("cfg_", "follow_")) or k == "_previous_y":
                    del st.session_state[k]
        except Exception as e:
            st.error(f"读取失败：{e}"); return
    raw = ws["raw"]
    if ws.get("pending_config"):
        pending = ws.pop("pending_config")
        for key, val in pending["settings"].items():
            if key.startswith("cfg_"):
                st.session_state[key] = val
        st.session_state["_previous_y"] = pending["settings"].get("cfg_y")
        ws["steps"] = pending["steps"]
        invalidate(ws)
    with st.expander("恢复项目"):
        config = st.file_uploader("项目配置 JSON", type=["json"], key="project_upload")
        if st.button("恢复设定", disabled=config is None, key="restore_project"):
            try:
                pending = load_config(config.getvalue(), digest)
                apply_steps(raw, pending["steps"])
                ws["pending_config"] = pending
                st.rerun()
            except Exception as e:
                st.error(str(e))
    data = pipeline_panel(ws, raw)
    result = specification(data)
    if result:
        s, pool = result
        search_controls(ws, data, s, pool)
    settings = {k: v for k, v in st.session_state.items() if k.startswith("cfg_")}
    payload = config_payload(digest, ws.get("steps", []), settings)
    with st.sidebar:
        st.download_button("保存项目设定（不含数据行）", json.dumps(payload, ensure_ascii=False, indent=2, default=str), "tihu_project.json", mime="application/json", key="save_project")
    results_ui(ws, raw)
