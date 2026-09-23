"""Guided workflow; all uploaded data and jobs remain in the user's session."""
from dataclasses import asdict, replace
from io import BytesIO
from itertools import islice
import json
import math
from time import perf_counter
import zipfile

import numpy as np
import pandas as pd
import streamlit as st

from tihu_core import (ModelSpec, binary, candidate_specs, fingerprint, fit_model,
                       fixed_sample, mechanism_analysis, prepare_panel_time, unique, validate_panel)
from tihu_export import reproducibility_bundle, stata_script
from tihu_ui import choose, multiple, display_table, iv_followup_panel


def detect_structure(data):
    time_names = {"year", "time", "wave", "date", "年份", "年度", "时间", "调查年份"}
    id_names = {"id", "pid", "fid", "firmid", "cityid", "code", "个体", "个体id", "城市", "省份", "企业代码", "地区代码", "家庭编号"}
    times = [c for c in data if c.lower() in time_names and data[c].nunique() > 1
             and (pd.api.types.is_numeric_dtype(data[c])
                  or (not pd.api.types.is_datetime64_any_dtype(data[c])
                      and pd.to_numeric(data[c], errors="coerce")[data[c].notna()].notna().all()))]
    entities = [c for c in data if c.lower() in id_names or c.lower().endswith("_id")]
    pairs = []
    for entity in entities:
        for time in times:
            if entity == time or data[entity].nunique() < 2:
                continue
            if data.groupby(entity)[time].nunique().max() < 2:
                continue
            if not data.duplicated([entity, time]).any() and not data[[entity, time]].isna().any().any():
                pairs.append((entity, time))
    if len(pairs) == 1:
        return "面板", *pairs[0]
    if pairs or times:
        return "待确认", "", times[0] if len(times) == 1 else ""
    return "横截面", "", ""


def basic_spec(data, y, core, entity="", time="", did=False, treatment="", policy=0):
    if data[y].dropna().nunique() < 2:
        raise ValueError("结果变量没有变化，无法估计")
    if entity or time:
        validate_panel(data, entity, time)
    if did:
        if not entity or not time:
            raise ValueError("DID 需要个体和时间列")
        if not binary(data[treatment]):
            raise ValueError("处理组变量必须同时包含 0 和 1")
        model = "DID"
        core = []
    else:
        model = "Probit" if binary(data[y]) else ("FE" if entity else "OLS")
    return ModelSpec(model, y, core, entity=entity, time=time,
                     treatment=treatment if did else "", policy=float(policy),
                     se="cluster" if entity else "robust", cluster=entity)


def search_size(pool, core_count, minimum=0, maximum=10):
    return max(1, core_count)*sum(math.comb(len(pool), size)
                                  for size in range(minimum, min(maximum, len(pool))+1))


def start_job(data, base, pool, mediators, moderators, minimum=0, maximum=10, budget=10000):
    pool = unique(pool)
    if len(pool) > 20 or len(base.core) > 6 or len(mediators) > 5 or len(moderators) > 5:
        raise ValueError("本次支持最多 6 个核心因素、20 个候选控制变量、各 5 个中介和调节变量")
    if not 0 <= minimum <= maximum <= 10 or minimum > len(pool):
        raise ValueError("控制变量范围须在 0–10 个内，且下限不能超过候选变量数")
    if not 1 <= budget <= 10000:
        raise ValueError("单次主模型搜索预算须在 1–10,000 组内")
    reserved = unique([base.y, base.entity, base.time, base.treatment]+base.core)
    if set(pool) & set(reserved+mediators+moderators):
        raise ValueError("控制变量与研究变量角色重复，请重新选择")
    if not base.core and base.model != "DID":
        raise ValueError("请选择至少一个核心因素")
    # Round-robin X allocation keeps the total budget bounded and each X represented.
    core_count = max(1, len(base.core))
    if budget < core_count:
        raise ValueError("搜索预算不能少于核心因素数量")
    specs = list(islice(candidate_specs(base, pool, minimum, min(maximum, len(pool)),
                                       math.ceil(budget/core_count), joint=False), budget))
    common = fixed_sample(data, base, pool)
    return {"data": common, "specs": specs, "index": 0, "best": {}, "records": [],
            "followups": [], "tasks": [], "task_index": 0, "stage": "base",
            "failures": {}, "mediators": list(mediators), "moderators": list(moderators),
            "done": False, "cancelled": False, "original_n": len(data), "elapsed": 0.,
            "search": {"minimum": minimum, "maximum": min(maximum, len(pool)),
                       "budget": budget, "possible": search_size(pool, len(base.core), minimum, maximum),
                       "planned": len(specs), "seed": 42}}


def slim(fit):
    fit.fitted = None
    fit.design = None
    fit.response = None
    return fit


def advance_job(job, batch=50, seconds=None):
    started = perf_counter()
    for i in range(batch):
        if job["done"]:
            break
        # Yield between fits so slow models cannot turn a batch into a long UI lock.
        if i and seconds is not None and perf_counter()-started >= seconds:
            break
        try:
            if job["stage"] == "base":
                spec = job["specs"][job["index"]]
                job["index"] += 1
                key = spec.core[0] if spec.core else "DID"
                result = slim(fit_model(job["data"], spec))
                job["records"].append({"核心因素": key, "控制变量": "、".join(spec.controls) or "无",
                                       "N": len(result.sample), **result.effect})
                old = job["best"].get(key)
                if old is None or result.effect["p"] < old.effect["p"]:
                    job["best"][key] = result
            else:
                key, kind, variable = job["tasks"][job["task_index"]]
                job["task_index"] += 1
                fit = job["best"][key]
                spec = fit.spec
                if kind == "中介":
                    parts = mechanism_analysis(fit.sample, spec, variable)
                    a = parts["channel"].effect
                    b = parts["mediator_effects"][0]
                    job["followups"].append({"核心因素": key, "分析": kind, "变量": variable,
                        "路径一 p": a["p"], "路径二 p": b["p"],
                        "筛选 p": max(a["p"], b["p"]),
                        "fits": [("同样本基准", slim(parts["baseline"])),
                                 ("因素到中介", slim(parts["channel"])),
                                 ("加入中介", slim(parts["adjusted"]))]})
                else:
                    exposure = "__did" if spec.model == "DID" else spec.core[0]
                    result = slim(fit_model(fit.sample, replace(spec, interactions=[[exposure, variable]])))
                    job["followups"].append({"核心因素": key, "分析": kind, "变量": variable,
                        "筛选 p": result.effect["p"], "fits": [("调节模型", result)]})
        except Exception as exc:
            reason = str(exc)[:180]
            job["failures"][reason] = job["failures"].get(reason, 0)+1
        if job["stage"] == "base" and job["index"] >= len(job["specs"]):
            job["stage"] = "followup"
            job["tasks"] = [(key, kind, var) for key in job["best"]
                            for kind, variables in [("中介", job["mediators"]), ("调节", job["moderators"])]
                            for var in variables]
        if job["stage"] == "followup" and job["task_index"] >= len(job["tasks"]):
            job["done"] = True
    job["elapsed"] = job.get("elapsed", 0.)+perf_counter()-started


def job_tick(ws):
    job = ws["job"]
    if st.button("停止并保留已完成结果", icon=":material/stop:", key="nv_stop"):
        job["done"] = True
        job["cancelled"] = True
        st.rerun()
    advance_job(job, batch=250, seconds=.75)
    main = job["stage"] == "base"
    total = len(job["specs"]) if main else len(job["tasks"])
    completed = job["index"] if main else job["task_index"]
    label = f"{'主模型' if main else '中介与调节'} · 已完成 {completed:,}/{total:,} · 计算用时 {job['elapsed']:.0f} 秒"
    if main and job["index"] >= 20:
        remaining = job["elapsed"]/job["index"]*(total-completed)
        label += f" · 估算剩余计算 {remaining:.0f} 秒"
    st.progress(completed/max(total, 1), text=label)
    if job["done"]:
        st.rerun()


def all_results(job):
    output = [(f"主模型 · {key}", fit) for key, fit in job["best"].items()]
    for row in sorted(job["followups"], key=lambda item: item["筛选 p"]):
        output.extend((f"{row['核心因素']} · {row['分析']} · {row['变量']} · {label}", fit)
                      for label, fit in row["fits"])
    return output


def full_bundle(job, digest):
    output = BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        catalog = []
        for i, (title, fit) in enumerate(all_results(job), 1):
            packed, _ = reproducibility_bundle(fit.sample, [], fit, digest)
            folder = f"model_{i:03d}"
            with zipfile.ZipFile(BytesIO(packed)) as model_archive:
                for name in model_archive.namelist():
                    archive.writestr(f"{folder}/{name}", model_archive.read(name))
            catalog.append({"目录": folder, "分析": title, "模型": fit.spec.model, "设定": asdict(fit.spec)})
        archive.writestr("index.json", json.dumps(catalog, ensure_ascii=False, indent=2))
        archive.writestr("search.csv", pd.DataFrame(job["records"]).to_csv(index=False))
        archive.writestr("search_plan.json", json.dumps(job.get("search", {}), ensure_ascii=False, indent=2))
        archive.writestr("README.txt", "每个 model 目录包含该分析实际使用的样本与 analysis.do。将 Stata 工作目录切换至对应目录运行。\n主模型按 p 值排序；中介为路径筛选，不等同于已识别的因果间接效应。\n")
    return output.getvalue()


def results(ws):
    job = ws.get("job")
    if not job or not job["done"]:
        return
    st.divider()
    st.subheader("实证结果")
    if job["cancelled"] and st.button("继续搜索", icon=":material/play_arrow:", key="nv_resume"):
        job.update(done=False, cancelled=False)
        ws.pop("bundle", None)
        st.rerun()
    a, b, c = st.columns(3)
    a.metric("完成组合", job["index"])
    b.metric("主模型", len(job["best"]))
    c.metric("显著主结果", sum(f.effect["p"] < .1 for f in job["best"].values()))
    st.caption(f"原始 {job['original_n']} 行 · 共同有效样本 {len(job['data'])} 行 · 显著阈值 p<0.10 · {'已停止' if job['cancelled'] else '已完成'}")
    tabs = st.tabs(["主结果", "中介与调节", "Stata 复现", "内生性处理"])
    with tabs[0]:
        if not job["best"]:
            st.info("本次没有可估计的主模型，请查看未完成项目。")
        for key, fit in job["best"].items():
            st.markdown(f"**{key} · {fit.spec.model}**")
            st.caption("控制变量："+("、".join(fit.spec.controls) or "无"))
            display_table(pd.DataFrame([fit.effect]))
        with st.expander("全部有效组合"):
            display_table(pd.DataFrame(job["records"]))
    with tabs[1]:
        if not job["mediators"] and not job["moderators"]:
            st.info("本次未选择中介或调节变量。")
        elif not job["followups"]:
            st.info("本次没有可估计的中介或调节结果。")
        else:
            table = pd.DataFrame([{k: v for k, v in row.items() if k != "fits"} for row in job["followups"]])
            table["通过筛选"] = table["筛选 p"] < .1
            st.dataframe(table.sort_values("筛选 p"), hide_index=True, width="stretch")
            st.caption("中介按两条路径分别显著筛选；不代表已验证因果中介效应。调节按交互项检验。")
    with tabs[2]:
        entries = all_results(job)
        if entries:
            idx = st.selectbox("分析结果", range(len(entries)), format_func=lambda i: entries[i][0], key="nv_export")
            fit = entries[idx][1]
            do, _ = stata_script(fit.sample, [], fit)
            with st.expander("Stata 复现代码", expanded=True):
                st.code(do, language="stata")
            st.caption("代码使用复现包对应 model 目录内的 source.dta 和 sample.dta。")
            if st.button("生成整套复现包", icon=":material/download:", key="nv_pack"):
                with st.spinner("正在整理分析结果"):
                    ws["bundle"] = full_bundle(job, ws["hash"])
            if ws.get("bundle"):
                st.download_button("下载全部结果与代码", ws["bundle"], "tihu_beginner.zip", "application/zip", key="nv_download")
    with tabs[3]:
        available = {key: fit for key, fit in job["best"].items() if fit.spec.model in {"OLS", "FE", "LPM"}}
        if available:
            key = choose("基准结果", available, "nv_iv_baseline")
            iv_followup_panel(ws, available[key], "nv_iv", job["mediators"]+job["moderators"])
        else:
            st.info("当前基准不是普通线性模型，不会自动替换为 2SLS。")
    if job["failures"]:
        with st.expander("未完成项目"):
            st.dataframe(pd.DataFrame([{"原因": key, "数量": value} for key, value in job["failures"].items()]), hide_index=True)


def render_novice():
    ws = st.session_state.setdefault("novice_workspace", {})
    for key, value in ws.get("settings", {}).items():
        if key not in st.session_state:
            st.session_state[key] = value
    with st.sidebar:
        st.caption("新手模式 · 基础实证")
        st.caption("数据仅在本次服务器会话中处理，不发送给 AI 服务。")
        if st.button("清除新手模式数据", icon=":material/delete:", key="nv_clear"):
            for key in list(st.session_state):
                if key.startswith("nv_") or key == "novice_workspace":
                    del st.session_state[key]
            st.rerun()
    did = st.radio("是否做 DID（政策前后对比）？", ["不做 DID", "做 DID"], horizontal=True, key="nv_did") == "做 DID"
    def upload_changed():
        if (st.session_state.get("nv_upload") is None
                and st.session_state.get("novice_upload_present", False)
                and st.session_state.get("experience_mode") == "我是新手"
                and not st.session_state.get("mode_back", False)):
            st.session_state["novice_workspace"].clear()
    upload = st.file_uploader("上传数据", type=["csv", "dta", "xlsx"], key="nv_upload", on_change=upload_changed)
    st.session_state["novice_upload_present"] = upload is not None
    if upload is None and "data" not in ws:
        return
    content = upload.getvalue() if upload is not None else None
    if content is not None and len(content) > 50*1024*1024:
        st.error("单个文件不能超过 50 MB")
        return
    digest = fingerprint(content) if content is not None else ws["hash"]
    if ws.get("hash") != digest:
        try:
            if upload.name.lower().endswith(".dta"):
                data = pd.read_stata(BytesIO(content), convert_categoricals=False)
            elif upload.name.lower().endswith(".xlsx"):
                data = pd.read_excel(BytesIO(content))
            else:
                try:
                    data = pd.read_csv(BytesIO(content))
                except UnicodeDecodeError:
                    data = pd.read_csv(BytesIO(content), encoding="gb18030")
            data.columns = data.columns.astype(str)
            if data.empty or data.columns.duplicated().any() or any(c.startswith("__") for c in data):
                raise ValueError("数据为空、包含重名列或使用了内部保留的双下划线列名")
            data = data.replace([np.inf, -np.inf], np.nan)
            ws.clear()
            ws.update(hash=digest, data=data, name=upload.name)
            for key in list(st.session_state):
                if key.startswith("nv_") and key not in {"nv_upload", "nv_did"}:
                    del st.session_state[key]
        except Exception as exc:
            st.error(f"文件无法读取：{exc}")
            return
    data = ws["data"]
    if upload is None:
        st.caption("已载入："+ws.get("name", "本次数据"))
    structure, entity, time = detect_structure(data)
    st.caption(f"{len(data):,} 条记录 · {len(data.columns)} 个变量 · {'需要确认数据结构' if structure == '待确认' else '识别为'+structure+'数据'}")
    with st.expander("数据预览"):
        st.dataframe(data.head(8), hide_index=True, width="stretch")
    panel = st.checkbox("按面板数据分析（同一个体跨期记录）", value=structure == "面板" or did,
                        disabled=did, key="nv_panel") or did
    if panel:
        a, b = st.columns(2)
        with a:
            entity = choose("个体 ID", data.columns, "nv_entity", entity)
        with b:
            time = choose("时间列", data.columns, "nv_time", time)
        try:
            data = prepare_panel_time(data, time)
            validate_panel(data, entity, time)
        except ValueError as exc:
            st.error(str(exc))
            return
    else:
        entity, time = "", ""
    st.subheader("确定研究变量")
    numeric = [c for c in data.select_dtypes(include="number") if c not in {entity, time}]
    if not numeric:
        st.info("需要至少一个数值型结果变量")
        return
    y = choose("想解释的结果 Y", numeric, "nv_y", "y")
    treatment, policy = "", 0
    if did:
        treatment = choose("处理组（1=受到政策影响，0=对照）", [c for c in numeric if c != y and binary(data[c])], "nv_treat", "D")
        periods = sorted(data[time].dropna().unique()) if time else []
        if st.session_state.get("nv_policy") not in periods:
            st.session_state["nv_policy"] = None
        policy = st.selectbox("政策开始时间", periods, index=None, placeholder="选择实际政策开始时间", key="nv_policy")
        cores = []
    else:
        cores = multiple("候选核心因素 X", [c for c in numeric if c != y], "nv_core")
    available = [c for c in data if c not in [y, treatment, entity, time]+cores]
    with st.expander("中介与调节 · 可选"):
        follow_choices = [c for c in available if c in numeric]
        mediators = multiple("候选中介变量", follow_choices, "nv_mediators")
        moderators = multiple("候选调节变量", follow_choices, "nv_moderators")
    pool = multiple("候选控制变量", [c for c in available if c not in mediators+moderators], "nv_pool")
    with st.expander("搜索范围"):
        minimum, maximum = st.slider("每组控制变量个数", 0, 10, (0, 10), key="nv_control_range")
        budget = st.number_input("主模型搜索预算（全部核心因素合计）", min_value=1, max_value=10000,
                                 value=10000, step=100, key="nv_budget")
    setting_keys = ["nv_did", "nv_panel", "nv_entity", "nv_time", "nv_y", "nv_treat", "nv_policy", "nv_core", "nv_mediators", "nv_moderators", "nv_pool", "nv_control_range", "nv_budget"]
    ws["settings"] = {key: st.session_state[key] for key in setting_keys if key in st.session_state}
    try:
        if did and (not treatment or policy is None):
            raise ValueError("请补全处理组和政策时间")
        base = basic_spec(data, y, cores, entity, time, did, treatment, policy or 0)
        base.categorical = [c for c in pool if not pd.api.types.is_numeric_dtype(data[c])]
        st.info(f"本次模型：{base.model}"+(" · 个体与时间双向固定效应" if base.model in {"FE", "DID"} else "")+(" · 按个体聚类" if entity else " · 稳健标准误"))
        if base.model == "Probit" and entity:
            st.caption("二元面板结果采用合并 Probit，按个体聚类，不包含个体固定效应。")
        signature = json.dumps([asdict(base), pool, mediators, moderators, minimum, maximum, budget], sort_keys=True)
        stale = bool(ws.get("job") and ws.get("signature") != signature)
        if stale:
            ws.pop("job", None)
            ws.pop("bundle", None)
        active = bool(ws.get("job") and not ws["job"]["done"])
        if st.button("开始自动实证", icon=":material/play_arrow:", type="primary", disabled=active, key="nv_run"):
            ws["job"] = start_job(data, base, pool, mediators, moderators, minimum, maximum, budget)
            ws["signature"] = signature
            ws.pop("bundle", None)
            st.session_state.pop("nv_export", None)
        possible = search_size(pool, len(cores), minimum, maximum)
        st.caption(f"符合范围 {possible:,} 组 · 本次计划 {min(possible, budget):,} 组 · "
                   +( "全部组合" if possible <= budget else "固定种子无重复抽样")
                   +"；中介与调节沿用各核心因素的优选主模型。")
    except Exception as exc:
        st.error(str(exc))
        return
    if ws.get("job") and not ws["job"]["done"]:
        st.fragment(run_every=.5)(job_tick)(ws)
    results(ws)


def render_entry(home):
    st.markdown("""<style>
    .stMainBlockContainer {max-width:1180px;padding-top:2rem;}
    div[data-testid="stVerticalBlockBorderWrapper"] {border-radius:8px;}
    button[kind="primary"] {background:#12665b;border-color:#12665b;}
    .mode-brand {font-size:26px;font-weight:750;color:inherit;margin:0 0 8px;}
    .stTabs [aria-selected="true"] {color:#12665b;}
    .stTabs [data-baseweb="tab-highlight"] {background:#12665b;}
    .stMultiSelect [data-baseweb="tag"] {background:#e3f1ec;color:#175747;}
    section.pelican-hero {grid-template-columns:1fr;gap:22px;min-height:0;padding:16px 0;
      margin:0 0 12px;border:0;border-radius:0;background:none;box-shadow:none;}
    .pelican-hero .pelican-brand {flex-direction:row;align-items:center;gap:16px;}
    .pelican-hero .pelican-title {font-size:48px;}
    .pelican-hero .pelican-subtitle {display:none;}
    .pelican-hero .pixel-stage {margin:0;}
    .pelican-hero .pelican-mark {width:190px;height:150px;}
    .pelican-hero .pelican-mascot {filter:none;}
    .pelican-hero .pelican-flow {padding:16px 0;border:0;border-radius:0;background:none;border-top:1px solid #dfe7e2;}
    .pelican-hero .flow-title,.pelican-hero .flow-note {display:none;}
    .pelican-hero .flow-list {grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;}
    .pelican-hero .flow-item {grid-template-columns:26px minmax(0,1fr);gap:10px;padding:0;border:0;}
    .pelican-hero .flow-num {width:26px;height:26px;border-radius:4px;font-size:12px;}
    .pelican-hero .flow-copy {font-size:14px;font-weight:500;}
    @media(max-width:600px) {
      .pelican-hero .pelican-title {font-size:36px;}
      .pelican-hero .pelican-mark {width:110px;height:130px;}
      .pelican-hero .pelican-kicker {font-size:12px;}
      .pelican-hero .flow-list {grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;}
      .pelican-hero .flow-copy {font-size:12px;}
    }
    </style>""", unsafe_allow_html=True)
    mode = st.session_state.get("experience_mode")
    if mode is None:
        home()
        st.subheader("选择你的实证方式")
        left, right = st.columns(2, gap="large")
        with left:
            with st.container(border=True):
                st.markdown("### 我是新手")
                if st.button("开始自动实证", icon=":material/auto_awesome:", type="primary", width="stretch", key="mode_novice"):
                    st.session_state["experience_mode"] = "我是新手"
                    st.rerun()
        with right:
            with st.container(border=True):
                st.markdown("### 我懂计量")
                if st.button("进入实证工作台", icon=":material/tune:", width="stretch", key="mode_expert"):
                    st.session_state["experience_mode"] = "我懂计量"
                    st.rerun()
        return
    st.markdown('<div class="mode-brand">鹈鹕回归</div>', unsafe_allow_html=True)
    left, right = st.columns([4, 1])
    left.caption(mode)
    if right.button("切换模式", icon=":material/swap_horiz:", key="mode_back"):
        # Widget teardown is not a request to remove the cached dataset.
        st.session_state["novice_upload_present"] = False
        st.session_state["expert_upload_present"] = False
        st.session_state.pop("experience_mode", None)
        st.rerun()
    st.divider()
    if mode == "我是新手":
        render_novice()
    else:
        from tihu_ui import render_workbench
        for key, value in st.session_state.get("expert_settings", {}).items():
            if key not in st.session_state:
                st.session_state[key] = value
        render_workbench()
        st.session_state["expert_settings"] = {key: st.session_state[key] for key in st.session_state if key.startswith("cfg_")}
