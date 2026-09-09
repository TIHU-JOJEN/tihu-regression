"""Synthetic validation of separate IV and ESR recommendation pipelines."""
import io
import unittest
import zipfile
from dataclasses import replace
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
from streamlit.testing.v1 import AppTest
from tihu_core import ModelSpec, fit_model, apply_steps, covariance_options
from tihu_iv import iv_candidates, esr_candidates, first_stage, iv_followup_spec
from tihu_export import reproducibility_bundle


def instrument_data(n=800):
    rng = np.random.default_rng(191)
    z, c, u, noise, z2 = rng.normal(size=(5, n))
    x = 1.4*z+.6*z2+2*c+u
    d = pd.DataFrame({"y": 1.7*x+.5*c+.8*u+noise, "x": x, "c": c, "z": z,
                      "z2": z2, "weak": rng.normal(size=n), "confounded": c+rng.normal(size=n)*.1})
    d["D"] = (z+.6*c+rng.normal(size=n) > 0).astype(int)
    d["id"] = np.arange(n)//5
    d["year"] = np.tile(np.arange(2015, 2020), n//5)
    d["invariant"] = d.id%7
    return d


class InstrumentTests(unittest.TestCase):
    def test_recommendation_is_conditional_and_not_y_significance(self):
        d = instrument_data()
        s = ModelSpec("IV/2SLS", "y", ["x"], ["c"])
        before = d.copy(deep=True)
        table = iv_candidates(d, s).set_index("变量")
        self.assertTrue(table.loc["z", "通过初筛"])
        self.assertFalse(table.loc["weak", "通过初筛"])
        self.assertFalse(table.loc["confounded", "通过初筛"])
        self.assertNotIn("id", table.index)
        changed = d.copy(); changed["y"] += 100*changed.z
        again = iv_candidates(changed, s).set_index("变量")
        pd.testing.assert_series_equal(table["第一阶段 F"], again["第一阶段 F"])
        pd.testing.assert_frame_equal(d, before)
        processed, _ = apply_steps(d, [{"op": "log", "cols": ["y"], "name": "lny"},
                                       {"op": "square", "cols": ["x"], "name": "x2"}])
        found = iv_candidates(processed, s)["变量"].tolist()
        self.assertNotIn("lny", found)
        self.assertNotIn("x2", found)

    def test_stage_statistics_match_selected_covariance(self):
        d = instrument_data()
        for se in ["ordinary", "robust", "cluster"]:
            s = ModelSpec("IV/2SLS", "y", ["x"], ["c"], se=se, cluster="id", instruments=["z"])
            stage, _ = first_stage(d, s)
            reference = sm.OLS(d.x, sm.add_constant(d[["c", "z"]])).fit(**covariance_options(s, d))
            self.assertAlmostEqual(stage["第一阶段 F"], reference.tvalues.z**2, places=8)
        with self.assertRaises(ValueError):
            fit_model(d, replace(s, instruments=["weak"], auto_instrument=True))

    def test_fe_iv_matches_full_dummy_iv(self):
        d = instrument_data(200)
        for entity, time in [(True, True), (True, False), (False, True)]:
            for se in ["ordinary", "robust", "cluster"]:
                with self.subTest(entity=entity, time=time, se=se):
                    s = ModelSpec("IV/2SLS", "y", ["x"], ["c"], entity="id", time="year",
                                  entity_effects=entity, time_effects=time, se=se, cluster="id", instruments=["z", "z2"])
                    parts = [sm.add_constant(d[["c"]])]
                    for col, enabled in [("id", entity), ("year", time)]:
                        if enabled:
                            parts.append(pd.get_dummies(d[col], prefix=col, drop_first=True, dtype=float))
                    exog = pd.concat(parts, axis=1)
                    kw = dict(cov_type="robust" if se == "robust" else "unadjusted", debiased=True)
                    if se == "cluster": kw = dict(cov_type="clustered", clusters=d.id, debiased=True)
                    expected = IV2SLS(d.y, exog, d[["x"]], d[["z", "z2"]]).fit(**kw)
                    actual = fit_model(d, s)
                    self.assertAlmostEqual(actual.effect["coef"], expected.params.x, places=8)
                    self.assertAlmostEqual(actual.effect["se"], expected.std_errors.x, places=8)
                    stage, _ = first_stage(d, s)
                    first = sm.OLS(d.x, pd.concat([exog, d[["z", "z2"]]], axis=1)).fit(**covariance_options(s, d))
                    test = first.f_test("z=0,z2=0")
                    self.assertAlmostEqual(stage["第一阶段 F"], float(test.fvalue), places=7)
        s = replace(s, entity_effects=True)
        found = iv_candidates(d, replace(s, instruments=[]))
        self.assertNotIn("invariant", found["变量"].tolist())

    def test_esr_has_its_own_probit_screen(self):
        d = instrument_data()
        s = ModelSpec("ESR", "y", controls=["c"], treatment="D")
        table = esr_candidates(d, s).set_index("变量")
        self.assertTrue(table.loc["z", "通过初筛"])
        self.assertIn("选择方程 Wald χ²", table)
        self.assertNotIn("第一阶段 F", table)
        self.assertNotIn("排他性代理 p值", table)
        changed = d.copy(); changed.y += changed.z*100
        again = esr_candidates(changed, s).set_index("变量")
        pd.testing.assert_series_equal(table["相关性 p值"], again["相关性 p值"])
        self.assertTrue(esr_candidates(d, replace(s, entity="id")).empty)

    def test_followup_samples_and_export(self):
        d = instrument_data(200)
        baseline = fit_model(d, ModelSpec("FE", "y", ["x"], ["c"], entity="id", time="year", se="cluster", cluster="id"))
        s = iv_followup_spec(baseline.spec, "x")
        s.instruments = ["z"]
        d.loc[:9, "z"] = np.nan
        iv = fit_model(d, s)
        same = fit_model(iv.sample, baseline.spec)
        self.assertEqual(len(iv.sample), len(same.sample))
        packed, do = reproducibility_bundle(d, [], iv, "synthetic")
        self.assertIn("i.__iv_id", do)
        self.assertIn("i.__iv_time", do)
        self.assertIn("estat firststage", do)
        with zipfile.ZipFile(io.BytesIO(packed)) as archive:
            self.assertIn("tihu_iv.py", archive.namelist())
        for model in ["DID", "ESR", "Fuzzy RD", "Probit"]:
            with self.assertRaises(ValueError):
                iv_followup_spec(replace(baseline.spec, model=model), "x")

    def test_automatic_ui_and_shared_followup(self):
        script = '''import streamlit as st
from io import BytesIO
from unittest.mock import patch
from test_instruments import instrument_data
from tihu_ui import render_workbench
d = instrument_data(200)
upload=BytesIO(d.to_csv(index=False).encode()); upload.name="synthetic.csv"
with patch("streamlit.file_uploader", side_effect=lambda *a, **kw: upload if kw.get("key")=="data_upload" else None): render_workbench()
'''
        app = AppTest.from_string(script, default_timeout=90).run()
        app.selectbox(key="cfg_y").select("y").run()
        app.multiselect(key="cfg_core").set_value(["x"]).run()
        app.multiselect(key="cfg_controls").set_value(["c"]).run()
        app.button(key="run_search").click().run()
        app.selectbox(key="follow_robust").select("内生性处理：2SLS").run()
        self.assertIn("z", app.multiselect(key="follow_iv_z").options)
        app.multiselect(key="follow_iv_z").set_value(["z"]).run()
        app.button(key="follow_iv_run").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(list(app.session_state["workspace_v2"]["follow_iv"]["results"]), ["原基准", "同样本基准", "2SLS"])
        app.selectbox(key="cfg_model").select("IV/2SLS").run()
        self.assertTrue(app.multiselect(key="cfg_instruments").value)
        app.multiselect(key="cfg_instruments").set_value(["z"]).run()
        app.button(key="run_search").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state["workspace_v2"]["fits"][0].spec.instruments, ["z"])
        app.selectbox(key="cfg_model").select("ESR").run()
        self.assertIn("z", app.selectbox(key="cfg_esr_exclusion").options)
        self.assertEqual(len(app.exception), 0)


if __name__ == "__main__":
    unittest.main()
