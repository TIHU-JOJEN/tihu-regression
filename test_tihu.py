"""Synthetic-data regression checks; no real user data."""
import io
import json
import unittest
import zipfile
from dataclasses import replace
import numpy as np
import pandas as pd
import statsmodels.api as sm
from tihu_core import *
from tihu_export import reproducibility_bundle, stata_model, stata_script


def cross(n=240, seed=42):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame(rng.normal(size=(n, 4)), columns=["x", "c", "z", "m"])
    d["D"] = (d.z + rng.normal(size=n) > 0).astype(int)
    d["y"] = 1+1.5*d.x+.4*d.c+rng.normal(size=n)
    d["group"] = np.arange(n)//10
    d["binary"] = (d.x+rng.normal(size=n) > 0).astype(int)
    d["count"] = rng.poisson(np.exp(.3+.2*d.x))
    d["ordinal"] = pd.qcut(d.y, 4, labels=False)
    return d


def panel():
    d = cross(400)
    d["id"] = np.repeat(np.arange(80), 5)
    d["year"] = np.tile(np.arange(2015, 2020), 80)
    d["D"] = (d.id >= 40).astype(int)
    d["y"] += d.D*(d.year >= 2017)*2+d.id*.03
    return d


class CoreTests(unittest.TestCase):
    def test_app_workflow(self):
        from streamlit.testing.v1 import AppTest
        script = '''import streamlit as st
from io import BytesIO
from unittest.mock import patch
from test_tihu import cross
from tihu_ui import render_workbench
st.set_page_config(layout="wide")
upload = BytesIO(cross().to_csv(index=False).encode())
upload.name = "synthetic.csv"
with patch("streamlit.file_uploader", side_effect=lambda *a, **kw: upload if kw.get("key")=="data_upload" else None):
    render_workbench()
'''
        a = AppTest.from_string(script, default_timeout=30).run()
        self.assertEqual(list(a.exception), [])
        a.selectbox(key="cfg_y").select("y").run()
        a.multiselect(key="cfg_core").set_value(["x"]).run()
        a.button(key="run_search").click().run()
        self.assertEqual(list(a.exception), [])
        self.assertEqual(len(a.session_state["workspace_v2"]["fits"]), 1)
        a.button(key="build_bundle").click().run()
        self.assertEqual(list(a.exception), [])
        self.assertIn("bundle", a.session_state["workspace_v2"])

    def test_app_esr_workflow(self):
        from streamlit.testing.v1 import AppTest
        script = '''import streamlit as st
from io import BytesIO
from unittest.mock import patch
import numpy as np
import pandas as pd
from tihu_ui import render_workbench
st.set_page_config(layout="wide")
rng=np.random.default_rng(31); n=500
x,z,u,e0,e1=rng.normal(size=(5,n)); D=(z+.4*x+u>0).astype(int)
df=pd.DataFrame({"y":np.where(D,3+.7*x+e1,1+.7*x+e0),"x":x,"z":z,"m":e0,"D":D})
upload=BytesIO(df.to_csv(index=False).encode()); upload.name="esr.csv"
with patch("streamlit.file_uploader", side_effect=lambda *a, **kw: upload if kw.get("key")=="data_upload" else None): render_workbench()
'''
        a = AppTest.from_string(script, default_timeout=60).run()
        a.selectbox(key="cfg_model").select("ESR").run()
        a.selectbox(key="cfg_esr_exclusion").select("z").run()
        a.multiselect(key="cfg_controls").set_value(["x"]).run()
        a.button(key="run_search").click().run(timeout=60)
        self.assertEqual(list(a.exception), [])
        fits = a.session_state["workspace_v2"]["fits"]
        self.assertEqual(len(fits), 1)
        self.assertEqual(fits[0].effect["term"], "ATT")
        self.assertEqual(fits[0].spec.controls, ["x"])
        self.assertEqual(fits[0].spec.selection, ["x", "z"])
        self.assertEqual(fits[0].spec.regime0, [])
        self.assertEqual(fits[0].spec.regime1, [])
        self.assertIn("Stata 复现代码", [item.label for item in a.expander])
        self.assertEqual([tab.label for tab in a.tabs[-4:]], ["机制 / 中介", "调节", "异质性", "稳健性"])

    def test_pipeline(self):
        raw = pd.DataFrame({"id": [1, 1, 1, 2, 2], "t": [2018, 2019, 2021, 2018, 2019], "x": [2., np.nan, 8., 10., 20.]})
        original = raw.copy(deep=True)
        steps = [{"op": "interpolate", "cols": ["x"], "time": "t", "group": "id"}, {"op": "lag", "cols": ["x"], "time": "t", "group": "id", "interval": 1, "name": "lag"}]
        d, _ = apply_steps(raw, steps)
        self.assertEqual(d.loc[1, "x"], 4.)
        self.assertTrue(np.isnan(d.loc[2, "lag"]))
        self.assertTrue(np.isnan(d.loc[3, "lag"]))
        self.assertEqual(d.loc[4, "lag"], 10.)
        pd.testing.assert_frame_equal(raw, original)

    def test_ols_se_and_candidates(self):
        d = cross()
        for se in ["ordinary", "robust", "cluster"]:
            s = ModelSpec("OLS", "y", ["x"], ["c"], se=se, cluster="group")
            r = fit_model(d, s)
            expected = sm.OLS(d.y, sm.add_constant(d[["x", "c"]])).fit(**covariance_options(s, d))
            self.assertAlmostEqual(r.effect["p"], expected.pvalues.x, places=12)
            self.assertAlmostEqual(r.effect["se"], expected.bse.x, places=12)
        self.assertEqual(len(list(candidate_specs(s, [], 0, 0))), 1)

    def test_families(self):
        d = cross()
        for model, y in [("Logit", "binary"), ("Probit", "binary"), ("Logit+Probit", "binary"), ("Ordered Logit", "ordinal"), ("Ordered Probit", "ordinal"), ("Poisson", "count")]:
            with self.subTest(model=model):
                r = fit_model(d, ModelSpec(model, y, ["x"], ["c"]))
                self.assertTrue(np.isfinite(r.effect["se"]))
        rng = np.random.default_rng(42)
        d["nb"] = rng.negative_binomial(2, 2/(2+np.exp(.5+.3*d.x)))
        self.assertIn("alpha", fit_model(d, ModelSpec("负二项", "nb", ["x"], ["c"])).table.term.tolist())

    def test_panel_and_did(self):
        d = panel()
        for model in ["FE", "RE", "FE+RE", "DID", "PSM-DID"]:
            with self.subTest(model=model):
                s = ModelSpec(model, "y", ["x"] if "DID" not in model else [], ["c"], entity="id", time="year", treatment="D" if "DID" in model else "", policy=2017, se="cluster", cluster="id", caliper=.3)
                r = fit_model(d, s)
                self.assertTrue(np.isfinite(r.effect["p"]))
                refit = fit_model(r.sample, s)
                self.assertAlmostEqual(r.effect["coef"], refit.effect["coef"], places=8)
        bad = pd.concat([d, d.iloc[[0]]])
        with self.assertRaises(ValueError):
            validate_panel(bad, "id", "year")

    def test_missing_panel_keys_only_remove_affected_rows(self):
        d = panel().rename(columns={"id": "participant_id"})
        d["participant_id"] = "P" + d["participant_id"].astype(str)
        d.loc[0, "participant_id"] = ""
        d.loc[1, "participant_id"] = None
        d.loc[2, "year"] = np.nan
        s = ModelSpec("FE", "y", ["x"], entity="participant_id", time="year",
                      se="cluster", cluster="participant_id")
        self.assertEqual(validate_panel(d, "participant_id", "year"), 3)
        fitted = fit_model(d, s)
        expected = fit_model(d.drop(index=[0, 1, 2]), s)
        self.assertEqual(len(fitted.sample), len(d) - 3)
        self.assertEqual(fitted.sample.index.tolist(), expected.sample.index.tolist())
        self.assertAlmostEqual(fitted.effect["coef"], expected.effect["coef"], places=8)
        self.assertEqual(d.loc[0, "participant_id"], "")
        self.assertTrue(pd.isna(d.loc[2, "year"]))
        duplicate = pd.concat([d, d.iloc[[3]]])
        with self.assertRaisesRegex(ValueError, "重复"):
            validate_panel(duplicate, "participant_id", "year")
        with self.assertRaisesRegex(ValueError, "没有共同有效"):
            validate_panel(d.iloc[[0, 1, 2]], "participant_id", "year")

    def test_fe_effect_dimensions(self):
        d = panel()
        names = {c: c for c in d.columns}
        for entity_effects, time_effects, absorb in [
            (True, False, "absorb(__panel)"),
            (False, True, "absorb(year)"),
            (True, True, "absorb(__panel year)"),
        ]:
            with self.subTest(entity=entity_effects, time=time_effects):
                spec = ModelSpec("FE", "y", ["x"], ["c"], entity="id", time="year",
                                 entity_effects=entity_effects, time_effects=time_effects,
                                 se="cluster", cluster="id")
                fit = fit_model(d, spec)
                self.assertTrue(np.isfinite(fit.effect["p"]))
                commands = "\n".join(stata_model(fit, names))
                self.assertIn(absorb, commands)
        with self.assertRaises(ValueError):
            fit_model(d, ModelSpec("FE", "y", ["x"], entity="id", time="year",
                                   entity_effects=False, time_effects=False))

    def test_rd_and_iv(self):
        d = cross(600)
        d["D"] = (d.x >= 0).astype(float)
        d["y"] += 2*d.D
        for model in ["Sharp RD", "Fuzzy RD"]:
            s = ModelSpec(model, "y", controls=["c"], running="x", bandwidth=.9, treatment="D" if model == "Fuzzy RD" else "")
            r = fit_model(d, s)
            self.assertTrue((r.sample.x.abs() <= .9).all())
            self.assertAlmostEqual(fit_model(r.sample, s).effect["coef"], r.effect["coef"], places=9)
        d["endog"] = d.z + d.x
        d["outcome"] = 2*d.endog+d.c+d.m
        r = fit_model(d, ModelSpec("IV/2SLS", "outcome", ["endog"], ["c"], instruments=["z"]))
        self.assertAlmostEqual(r.effect["coef"], 2., delta=.2)

    def test_limited(self):
        d = cross(500)
        d["censored"] = d.y.clip(lower=0)
        r = fit_model(d, ModelSpec("Tobit", "censored", ["x"], ["c"], left=0.0, se="ordinary"))
        self.assertAlmostEqual(r.effect["coef"], 1.5, delta=.2)
        d.loc[d.D == 0, "y"] = np.nan
        r = fit_model(d, ModelSpec("Heckman", "y", ["x"], ["c"], treatment="D", selection=["x", "c", "z"], se="ordinary"))
        self.assertEqual(len(r.sample), len(d))
        self.assertAlmostEqual(r.effect["coef"], 1.5, delta=.3)

    def test_psm(self):
        d = cross()
        r = fit_model(d, ModelSpec("PSM", "y", controls=["c", "z"], treatment="D", caliper=.2))
        self.assertGreater(r.details["matched_pairs"], 2)
        used = [i for t, cs in r.details["pairs"] for i in [t]+cs]
        self.assertEqual(len(used), len(set(used)))

    def test_esr(self):
        rng = np.random.default_rng(31)
        n = 900
        x, z, u, e0, e1 = rng.normal(size=(5, n))
        d = (z+.4*x+u > 0).astype(int)
        y0 = 1+.7*x+.35*u+e0
        y1 = 3+.7*x+.35*u+e1
        df = pd.DataFrame({"y": np.where(d, y1, y0), "x": x, "z": z, "D": d})
        r = fit_model(df, ModelSpec("ESR", "y", controls=["x"], treatment="D", selection=["x", "z"], se="ordinary"))
        self.assertTrue(r.details["converged"])
        self.assertAlmostEqual(r.effect["coef"], 2., delta=.45)
        self.assertTrue(r.effect["lower"] < r.effect["coef"] < r.effect["upper"])
        effects = r.details["effects"].set_index("term").coef
        self.assertAlmostEqual(effects.ATT, effects["E[Y1|D=1]"]-effects["E[Y0|D=1]"], places=10)

    def test_esr_candidate_controls_enter_all_equations(self):
        base = ModelSpec("ESR", "y", controls=["x"], treatment="D", instruments=["z"], selection=["x", "z"])
        specs = list(candidate_specs(base, ["c"], 0, 1, 10))
        self.assertEqual(len(specs), 2)
        for spec in specs:
            self.assertEqual(spec.selection, spec.controls+["z"])
            self.assertEqual(spec.regime0, [])
            self.assertEqual(spec.regime1, [])
        final = specs[-1]
        names = {c: c for c in ["y", "D", "x", "c", "z"]}
        commands = "\n".join(stata_model(Fit(final, None, None, None), names))
        self.assertIn("(selection: D = x c z)", commands)
        self.assertIn("(regime0: y = x c)", commands)
        self.assertIn("(regime1: y = x c)", commands)

    def test_esr_identification_screen(self):
        rng = np.random.default_rng(9)
        n = 1200
        z, bad, noise = rng.normal(size=(3, n))
        d = (1.2*z+rng.normal(size=n) > 0).astype(int)
        y = 2*d+1.5*bad+noise
        data = pd.DataFrame({"y": y, "D": d, "z": z, "bad": bad, "noise": rng.normal(size=n)})
        screened = esr_identification_candidates(data, "y", "D", ["z", "bad", "noise"])
        self.assertEqual(screened["变量"].tolist(), ["z"])

    def test_config_and_bundle(self):
        raw = cross()
        steps = [{"op": "square", "cols": ["x"], "name": "x_sq"}]
        data, _ = apply_steps(raw, steps)
        fit = fit_model(data, ModelSpec("OLS", "y", ["x"], ["x_sq"], se="ordinary"))
        packed, do = reproducibility_bundle(raw, steps, fit, "hash")
        script, _ = stata_script(raw, steps, fit)
        self.assertEqual(script, do)
        with zipfile.ZipFile(io.BytesIO(packed)) as z:
            self.assertEqual(len(pd.read_stata(io.BytesIO(z.read("source.dta")))), len(raw))
            self.assertIn("replay.py", z.namelist())
            self.assertIn("^2", do)
        config = json.dumps(config_payload("hash", steps, {})).encode()
        self.assertEqual(load_config(config, "hash")["steps"], steps)
        with self.assertRaises(ValueError):
            load_config(config, "wrong")

    def test_followups_and_categorical(self):
        d = panel()
        s = ModelSpec("DID", "y", controls=["c"], entity="id", time="year", treatment="D", policy=2017, cluster="id", se="cluster")
        baseline = fit_model(d, s)
        mod = fit_model(baseline.sample, replace(s, interactions=[["__did", "m"]]))
        self.assertEqual(mod.effect["term"], "__interaction0")
        self.assertEqual(len(mod.sample), len(baseline.sample))
        for model in ["ANOVA", "交互效应"]:
            f = fit_model(d, ModelSpec(model, "y", core=["x"] if model == "交互效应" else [], controls=["c"], group="D", contrast="1"))
            self.assertTrue(np.isfinite(f.effect["p"]))
        x = cross(200)
        x["m"] = .8*x.x+x.m
        x["y"] = 1.1*x.m+.4*x.x+x.z
        effect = mediation_bootstrap(x, ModelSpec("OLS", "y", ["x"], ["c"]), "m", reps=49)
        self.assertGreater(effect["有效重复"], 30)
        self.assertGreater(effect["lower"], 0)
        analysis = mechanism_analysis(x, ModelSpec("Logit", "binary", ["x"], ["c"]), "m")
        self.assertEqual(analysis["channel"].spec.model, "OLS")
        grouped, contrast = grouped_moderation(x, ModelSpec("PSM", "y", controls=["c", "z"], treatment="D", se="ordinary", caliper=.3), "m")
        self.assertEqual(set(grouped), {"低组", "高组"})
        self.assertTrue(np.isfinite(contrast["p"]))

    def test_fixed_and_per_combo_samples(self):
        d = cross()
        d.loc[:49, "m"] = np.nan
        s = ModelSpec("OLS", "y", ["x"])
        self.assertEqual(len(fit_model(d, s).sample), len(d))
        fixed = fixed_sample(d, s, ["m"])
        self.assertEqual(len(fit_model(fixed, s).sample), len(d)-50)
        self.assertEqual(config_payload("x", [], {"cfg_policy": np.int64(2017)})["settings"]["cfg_policy"], 2017)


if __name__ == "__main__":
    unittest.main()
