"""In-memory, explicit-download reproducibility bundles."""
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
import json
import zipfile
import importlib.metadata
import numpy as np
import pandas as pd
from tihu_core import VERSION, apply_steps, unique


def dta_bytes(frame):
    frame = frame.copy()
    for c in frame:
        if pd.api.types.is_bool_dtype(frame[c]):
            frame[c] = frame[c].astype(float)
        elif pd.api.types.is_object_dtype(frame[c]) or isinstance(frame[c].dtype, pd.CategoricalDtype):
            frame[c] = frame[c].map(lambda v: "" if pd.isna(v) else str(v))
        elif pd.api.types.is_extension_array_dtype(frame[c]):
            frame[c] = frame[c].astype(float)
    out = BytesIO()
    frame.to_stata(out, write_index=False, version=118)
    return out.getvalue()


def stata_pipeline(raw, steps, names):
    lines = ['use "source.dta", clear']
    before = raw.copy()
    for i, step in enumerate(steps):
        op, cols = step["op"], step.get("cols", [])
        vs = " ".join(names[c] for c in cols)
        group = names.get(step.get("group"), "")
        time = names.get(step.get("time"), "")
        new = names.get(step.get("name"), "")
        lines.append(f"* Step {i+1}: {op}")
        if op == "deduplicate":
            subset = vs or " ".join(names[c] for c in before)
            lines += [f"sort {subset} __rowid", f"by {subset}: keep if _n == 1"]
        elif op == "numeric":
            for c in cols:
                if not pd.api.types.is_numeric_dtype(before[c]):
                    lines.append(f"destring {names[c]}, replace force")
        elif op == "drop_missing":
            lines.append("drop if " + " | ".join(f"missing({names[c]})" for c in cols))
        elif op in {"mean", "median"}:
            for j, c in enumerate(cols):
                temp = f"__fill{i}_{j}"
                lines += [f"egen double {temp} = {op}({names[c]})" + (f", by({group})" if group else ""), f"replace {names[c]} = {temp} if missing({names[c]})", f"drop {temp}"]
        elif op == "interpolate":
            for j, c in enumerate(cols):
                temp = f"__fill{i}_{j}"
                prefix = f"bysort {group}: " if group else ""
                lines += [f"{prefix}ipolate {names[c]} {time}, gen({temp})", f"replace {names[c]} = {temp} if missing({names[c]})", f"drop {temp}"]
        elif op == "winsor":
            for c in cols:
                lo, hi = before[c].quantile([step["pct"], 1-step["pct"]])
                lines += [f"* pandas linear quantiles, preserved at full precision", f"replace {names[c]} = max({lo:.17g}, min({hi:.17g}, {names[c]})) if !missing({names[c]})"]
        elif op == "filter":
            lines.append(f"keep if inrange({names[cols[0]]}, {step['lower']:.17g}, {step['upper']:.17g})")
        elif op == "log":
            lines.append(f"gen double {new} = ln({vs}) if {vs}>0 & !missing({vs})")
        elif op == "square":
            lines.append(f"gen double {new} = {vs}^2")
        elif op == "interact":
            lines.append(f"gen double {new} = {names[cols[0]]} * {names[cols[1]]}")
        elif op == "zscore":
            lines.append(f"gen double {new} = ({vs} - {before[cols[0]].mean():.17g}) / {before[cols[0]].std():.17g}")
        elif op == "recode":
            value = step["one"]
            if isinstance(value, str):
                value = " + ".join(f"uchar({ord(char)})" for char in value) or '""'
                value = f"({value})"
            lines.append(f"gen double {new} = ({vs} == {value}) if !missing({vs})")
        elif op in {"lag", "diff"}:
            # Match exact time keys; a previous row across a gap is not a lag.
            temp = f"__lag{i}"
            lines += ["preserve", f"keep {group} {time} {vs}", f"replace {time} = {time} + {float(step.get('interval', 1)):.17g}", f"rename {vs} {temp}", f"tempfile lag{i}", f"save `lag{i}'", "restore", f"merge 1:1 {group} {time} using `lag{i}', keep(master match) nogen", f"gen double {new} = " + (temp if op == "lag" else f"{vs}-{temp}"), f"drop {temp}"]
        before, _ = apply_steps(before, [step])
    return lines


def stata_model(fit, names):
    s = fit.spec
    def n(v):
        return names[v]
    def terms(cols):
        return " ".join(("i." if c in s.categorical else "")+n(c) for c in unique(cols))
    vce = f"vce(cluster {n(s.cluster)})" if s.se == "cluster" else ("vce(robust)" if s.se == "robust" else "")
    option = f", {vce}" if vce else ""
    y = n(s.y)
    rhs = terms(s.core+s.controls)
    lines = []
    if s.model in {"ANOVA", "交互效应"}:
        rhs += " " + terms([s.group])
        if s.model == "交互效应":
            levels = sorted(fit.sample[s.group].astype(str).unique())
            for i, level in enumerate(levels[1:]):
                lines += [f"gen double __gx{i} = ({n(s.group)}=={i+2})*{n(s.core[0])}"]
                rhs += f" __gx{i}"
    for i, (a, b) in enumerate(s.interactions):
        lines.append(f"gen double __interaction{i} = {n(a)}*{n(b)}")
        rhs += " " + terms([v for v in [a, b] if v not in s.core+s.controls]) + f" __interaction{i}"
    if s.model in {"OLS", "Pooled OLS", "LPM", "ANOVA", "交互效应"}:
        lines += [f"regress {y} {rhs}{option}"]
    elif s.model in {"Logit", "Probit", "Logit+Probit", "Ordered Logit", "Ordered Probit", "Poisson", "负二项"}:
        commands = {"Logit": "logit", "Probit": "probit", "Logit+Probit": "logit", "Ordered Logit": "ologit", "Ordered Probit": "oprobit", "Poisson": "poisson", "负二项": "nbreg"}
        lines += [f"{commands[s.model]} {y} {rhs}{option}"]
        if s.model == "Logit+Probit":
            lines += [f"probit {y} {rhs}{option}"]
        lines += ["margins, dydx(*)"]
    elif s.model in {"FE", "RE", "FE+RE", "DID", "PSM-DID"}:
        lines += [f"egen long __panel = group({n(s.entity)})", f"xtset __panel {n(s.time)}"]
        if "DID" in s.model:
            post = n(s.post) if s.post else f"({n(s.time)} >= {s.policy:.17g})"
            lines += [f"gen byte __post = {post}", f"gen byte __did = {n(s.treatment)}*__post"]
            rhs = "__did " + terms(s.controls) + ("" if s.time_effects else " __post")
            for i, (a, b) in enumerate(s.interactions):
                rhs += " " + terms([v for v in [a, b] if not v.startswith("__") and v not in s.controls]) + f" __interaction{i}"
        if s.model == "RE":
            lines += [f"xtreg {y} {rhs}, re {vce}"]
        else:
            absorb = f"__panel {n(s.time)}" if s.time_effects else "__panel"
            lines += ["capture which reghdfe", "if _rc ssc install reghdfe", f"reghdfe {y} {rhs}, absorb({absorb}) keepsingletons {vce}"]
            if s.model == "FE+RE":
                lines += [f"xtreg {y} {rhs}, re {vce}"]
    elif s.model in {"Sharp RD", "Fuzzy RD"}:
        lines += [f"gen double __rd_r = {n(s.running)}-{s.cutoff:.17g}", "gen byte __rd_z = __rd_r>=0"]
        extra = []
        for power in range(1, s.degree+1):
            lines += [f"gen double __rd_r{power} = __rd_r^{power}", f"gen double __rd_zr{power} = __rd_z*__rd_r^{power}"]
            extra += [f"__rd_r{power}", f"__rd_zr{power}"]
        rhs = " ".join(extra) + " " + terms(s.controls)
        if s.bandwidth:
            lines += [f"keep if abs(__rd_r)<={s.bandwidth:.17g}"]
        if s.model == "Sharp RD":
            lines += [f"regress {y} __rd_z {rhs}{option}"]
        else:
            observed = fit.sample[s.treatment].to_numpy()
            threshold = (fit.sample[s.running] >= s.cutoff).to_numpy()
            if np.array_equal(observed, threshold) or np.array_equal(observed, 1-threshold):
                lines += ["* Deterministic assignment: Fuzzy RD reduces to equivalent Sharp RD.", f"regress {y} {n(s.treatment)} {rhs}{option}"]
            else:
                lines += [f"ivregress 2sls {y} {rhs} ({n(s.treatment)}=__rd_z), small {vce}", "estat firststage"]
    elif s.model == "IV/2SLS":
        lines += [f"ivregress 2sls {y} {terms(s.controls)} ({n(s.core[0])}={terms(s.instruments)}), small {vce}", "estat firststage"]
    elif s.model == "Tobit":
        bounds = " ".join(([f"ll({s.left})"] if s.left is not None else [])+([f"ul({s.right})"] if s.right is not None else []))
        lines += [f"tobit {y} {rhs}, {bounds} {vce}"]
    elif s.model == "Heckman":
        lines += [f"heckman {y} {rhs}, select({n(s.treatment)}={terms(s.selection)}) {vce}"]
    elif s.model == "PSM":
        lines += ["* Same fixed matched pairs as the web result; propensity score uncertainty is not included.", 'merge 1:1 __rowid using "matches.dta", keep(match) nogen', f"collapse (mean) {y}, by(__pair {n(s.treatment)})", f"reshape wide {y}, i(__pair) j({n(s.treatment)})", f"gen double __diff = {y}1-{y}0", "ttest __diff == 0"]
    elif s.model == "ESR":
        z, x0, x1, treatment = terms(s.selection), terms(s.regime0 or s.controls), terms(s.regime1 or s.controls), n(s.treatment)
        lines += [
            "* Full-information Gaussian endogenous switching likelihood; no movestay dependency.",
            "capture program drop tihu_esr_ll", "program define tihu_esr_ll", "    version 16",
            "    args lnf eta mu0 mu1 lns0 lns1 ar0 ar1", "    tempvar e0 e1",
            "    quietly gen double `e0' = ($ML_y2-`mu0')/exp(`lns0')",
            "    quietly gen double `e1' = ($ML_y3-`mu1')/exp(`lns1')",
            "    quietly replace `lnf' = -.5*ln(2*_pi)-.5*`e0'^2-`lns0' + lnnormal(-(`eta'+tanh(`ar0')*`e0')/sqrt(1-tanh(`ar0')^2)) if $ML_y1==0",
            "    quietly replace `lnf' = -.5*ln(2*_pi)-.5*`e1'^2-`lns1' + lnnormal((`eta'+tanh(`ar1')*`e1')/sqrt(1-tanh(`ar1')^2)) if $ML_y1==1", "end",
            f"quietly probit {treatment} {z}", "matrix __start = e(b)",
            f"quietly regress {y} {x0} if {treatment}==0", "matrix __start = __start, e(b)", "scalar __s0 = ln(e(rmse))",
            f"quietly regress {y} {x1} if {treatment}==1", "matrix __start = __start, e(b)", "scalar __s1 = ln(e(rmse))",
            "matrix __start = __start, __s0, __s1, 0, 0",
            f"ml model lf tihu_esr_ll (selection: {treatment} = {z}) (regime0: {y} = {x0}) (regime1: {y} = {x1}) /lns0 /lns1 /ar0 /ar1 {option}",
            "ml init __start, copy", "ml maximize, difficult iterate(1000)",
            "if e(converged)!=1 error 430",
        ]
        eta = "predict(xb equation(selection))"
        difference = "predict(xb equation(regime1))-predict(xb equation(regime0))"
        delta = "(exp(_b[/lns1])*tanh(_b[/ar1])-exp(_b[/lns0])*tanh(_b[/ar0]))"
        lines += [f"margins if {treatment}==1, expression({difference}+{delta}*normalden({eta})/normal({eta}))", f"margins if {treatment}==0, expression({difference}-{delta}*normalden({eta})/normal(-({eta})))"]
    return lines


def stata_script(raw, steps, fit):
    processed, _ = apply_steps(raw, steps)
    cols = unique(list(raw.columns)+list(processed.columns))
    names = {c: f"v{i+1:04d}" for i, c in enumerate(cols)}
    categorical = [c for c in fit.spec.categorical if c in processed]
    lines = ["version 16", "clear all", "set more off", "set seed 42", "* Run from the extracted bundle directory."]
    lines += stata_pipeline(raw, steps, names)
    lines += ['merge 1:1 __rowid using "sample.dta", keep(match) nogen', "sort __rowid"]
    for i, c in enumerate(categorical):
        if pd.api.types.is_numeric_dtype(processed[c]):
            # Python encodes the string representations in lexical order.
            levels = sorted(fit.sample[c].dropna().astype(str).unique())
            lines += [f"gen double __encoded{i} = ."]
            for code, value in enumerate(levels, 1):
                lines += [f"replace __encoded{i} = {code} if {names[c]} == {float(value):.17g}"]
            lines += [f"drop {names[c]}", f"rename __encoded{i} {names[c]}"]
        else:
            lines += [f"encode {names[c]}, gen(__encoded{i})", f"drop {names[c]}", f"rename __encoded{i} {names[c]}"]
    lines += stata_model(fit, names)
    return "\n".join(lines), names


def reproducibility_bundle(raw, steps, fit, file_hash):
    processed, _ = apply_steps(raw, steps)
    do, names = stata_script(raw, steps, fit)
    source = raw.rename(columns=names).copy()
    source["__rowid"] = raw.index.to_numpy(dtype=int)
    selected = pd.DataFrame({"__rowid": fit.sample.index.to_numpy(dtype=int)})
    manifest = {"version": VERSION, "file_hash": file_hash, "steps": steps, "spec": asdict(fit.spec), "column_names": names, "sample_rows": fit.sample.index.tolist()}
    out = BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("source.dta", dta_bytes(source))
        z.writestr("sample.dta", dta_bytes(selected))
        z.writestr("analysis.do", do+"\n")
        z.writestr("result.csv", fit.table.to_csv(index=False))
        if "effects" in fit.details:
            z.writestr("effects.csv", fit.details["effects"].to_csv(index=False))
        z.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
        z.writestr("columns.csv", pd.DataFrame({"原变量": list(names), "Stata变量": list(names.values())}).to_csv(index=False))
        if fit.spec.model == "PSM":
            pairs = [{"__rowid": int(fit.sample.index[row]), "__pair": i} for i, (t, controls) in enumerate(fit.details["pairs"]) for row in [t]+controls]
            z.writestr("matches.dta", dta_bytes(pd.DataFrame(pairs)))
        z.writestr("tihu_core.py", Path(__file__).with_name("tihu_core.py").read_text())
        z.writestr("replay.py", '''import json
from dataclasses import fields
import pandas as pd
from tihu_core import ModelSpec, apply_steps, fit_model
with open("manifest.json", encoding="utf-8") as f:
    config = json.load(f)
raw = pd.read_stata("source.dta", convert_categoricals=False)
raw.index = raw.pop("__rowid").astype(int)
raw = raw.rename(columns={v:k for k,v in config["column_names"].items()})
data, _ = apply_steps(raw, config["steps"])
result = fit_model(data.loc[config["sample_rows"]], ModelSpec(**config["spec"]))
print(result.table.to_string(index=False))
print(result.effect)
''')
        packages = ["streamlit", "pandas", "numpy", "statsmodels", "linearmodels", "scipy", "scikit-learn", "openpyxl"]
        z.writestr("requirements.txt", "\n".join(f"{package}=={importlib.metadata.version(package)}" for package in packages)+"\n")
        z.writestr("README.txt", "此文件包包含本次原始数据，请自行保管。\nanalysis.do: 在 Stata 中将工作目录切换到解压目录后运行。\nsource.dta: 原始数据；sample.dta: 选定估计样本行号。\n清洗及变量加工在 analysis.do 中重做；匹配结果使用本次固定匹配样本。\nreplay.py: 使用相同 Python 依赖重新计算。\n网页与 Stata 的有限样本标准误修正可能存在差异；未经 Stata 实机数值验收的模型不能保证逐位一致。\nESR 使用完整联合似然，ATT/ATU 对固定协变量样本采用 Delta 法。\n")
    return out.getvalue(), do
