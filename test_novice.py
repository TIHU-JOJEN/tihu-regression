"""Synthetic checks for mode isolation, routing, search and exports."""
import io
import json
import math
from collections import Counter
import unittest
import zipfile
from dataclasses import replace
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest

from test_tihu import cross, panel
from tihu_core import prepare_panel_time
from tihu_novice import (detect_structure, basic_spec, start_job, advance_job,
                         full_bundle)


class NoviceTests(unittest.TestCase):
    def test_expanded_search_coverage_and_budget(self):
        d = cross()
        rng = np.random.default_rng(23)
        pool = [f"control{i}" for i in range(20)]
        for col in pool:
            d[col] = rng.normal(size=len(d))
        base = basic_spec(d, "y", ["x"])
        exhaustive = start_job(d, base, pool[:10], [], [])
        self.assertEqual(len(exhaustive["specs"]), 1024)
        self.assertEqual({len(s.controls) for s in exhaustive["specs"]}, set(range(11)))
        ranged = start_job(d, base, pool[:10], [], [], minimum=6)
        self.assertEqual(len(ranged["specs"]), sum(math.comb(10, k) for k in range(6, 11)))
        multi = replace(base, core=["x", "z", "m"])
        sampled = start_job(d, multi, pool, [], [], minimum=6)
        identifiers = [(tuple(s.core), tuple(s.controls)) for s in sampled["specs"]]
        self.assertEqual(len(identifiers), 10000)
        self.assertEqual(len(set(identifiers)), 10000)
        self.assertTrue(all(6 <= len(s.controls) <= 10 for s in sampled["specs"]))
        counts = Counter(s.core[0] for s in sampled["specs"])
        self.assertLessEqual(max(counts.values())-min(counts.values()), 1)
        again = start_job(d, multi, pool, [], [], minimum=6)
        self.assertEqual(identifiers, [(tuple(s.core), tuple(s.controls)) for s in again["specs"]])
        self.assertEqual(len(start_job(d, base, pool, [], [], budget=1)["specs"]), 1)
        with self.assertRaises(ValueError):
            start_job(d, multi, pool, [], [], budget=2)
        with self.assertRaises(ValueError):
            start_job(d, base, pool, [], [], budget=10001)
        with self.assertRaises(ValueError):
            start_job(d, base, pool[:2], [], [], minimum=6)

    def test_timed_batch_and_resume(self):
        d = cross()
        job = start_job(d, basic_spec(d, "y", ["x"]), ["c", "z"], [], [])
        with patch("tihu_novice.perf_counter", side_effect=[0., 2., 3.]):
            advance_job(job, batch=50, seconds=.75)
        self.assertEqual(job["index"], 1)
        job.update(done=True, cancelled=True)
        advance_job(job)
        self.assertEqual(job["index"], 1)
        job.update(done=False, cancelled=False)
        advance_job(job)
        self.assertTrue(job["done"])
        self.assertEqual(len(job["records"]), 4)

    def test_ui_stop_resume_and_range_invalidation(self):
        script = '''import streamlit as st
from io import BytesIO
from unittest.mock import patch
import numpy as np
from test_tihu import cross
from tihu_novice import render_novice
d = cross()
for i in range(10): d[f"c{i}"] = np.random.default_rng(i).normal(size=len(d))
upload=BytesIO(d.to_csv(index=False).encode()); upload.name="range.csv"
with patch("streamlit.file_uploader", return_value=upload): render_novice()
'''
        app = AppTest.from_string(script, default_timeout=60).run()
        app.multiselect(key="nv_core").set_value(["x"]).run()
        app.multiselect(key="nv_pool").set_value([f"c{i}" for i in range(10)]).run()
        self.assertEqual(app.number_input(key="nv_budget").value, 10000)
        app.button(key="nv_run").click().run()
        self.assertFalse(app.session_state["novice_workspace"]["job"]["done"])
        app.button(key="nv_stop").click().run()
        job = app.session_state["novice_workspace"]["job"]
        self.assertTrue(job["cancelled"])
        paused_at = job["index"]
        app.button(key="nv_resume").click().run()
        self.assertGreater(app.session_state["novice_workspace"]["job"]["index"], paused_at)
        app.slider(key="nv_control_range").set_range(6, 10).run()
        self.assertNotIn("job", app.session_state["novice_workspace"])
        self.assertEqual(len(app.exception), 0)

    def test_routes_and_ambiguous_panel(self):
        self.assertEqual(detect_structure(cross())[0], "横截面")
        self.assertEqual(detect_structure(panel()), ("面板", "id", "year"))
        duplicate = panel()
        duplicate.loc[1, "year"] = duplicate.loc[0, "year"]
        self.assertEqual(detect_structure(duplicate)[0], "待确认")
        self.assertEqual(basic_spec(cross(), "binary", ["x"]).model, "Probit")
        self.assertEqual(basic_spec(panel(), "y", ["x"], "id", "year").model, "FE")
        self.assertEqual(basic_spec(panel(), "y", [], "id", "year", True, "D", 2017).model, "DID")

    def test_manual_panel_columns_and_numeric_text_time(self):
        script = '''import streamlit as st
from io import BytesIO
from unittest.mock import patch
from test_tihu import panel
from tihu_novice import render_novice
d = panel()
d["firm_code"] = "F" + d["id"].astype(str)
d["wave_text"] = d["year"].astype(str)
upload = BytesIO(); d.to_stata(upload, write_index=False); upload.seek(0); upload.name = "synthetic_panel.dta"
with patch("streamlit.file_uploader", return_value=upload): render_novice()
'''
        app = AppTest.from_string(script, default_timeout=60).run()
        self.assertEqual(detect_structure(panel()), ("面板", "id", "year"))
        self.assertEqual(app.selectbox(key="nv_entity").value, "id")
        self.assertIn("wave_text", app.selectbox(key="nv_time").options)
        app.selectbox(key="nv_entity").select("firm_code").run()
        app.selectbox(key="nv_time").select("wave_text").run()
        app.selectbox(key="nv_y").select("y").run()
        app.multiselect(key="nv_core").set_value(["x"]).run()
        app.button(key="nv_run").click().run()
        self.assertEqual(list(app.exception), [])
        job = app.session_state["novice_workspace"]["job"]
        self.assertEqual((job["specs"][0].entity, job["specs"][0].time), ("firm_code", "wave_text"))
        self.assertTrue(np.issubdtype(job["data"]["wave_text"].dtype, np.number))
        self.assertEqual(app.session_state["novice_workspace"]["data"]["wave_text"].dtype, object)

    def test_panel_missing_keys_do_not_block_novice(self):
        script = '''import streamlit as st
from io import BytesIO
from unittest.mock import patch
from test_tihu import panel
from tihu_novice import render_novice
d = panel().rename(columns={"id": "participant_id"})
d["participant_id"] = "P" + d["participant_id"].astype(str)
d.loc[0, "participant_id"] = ""
d.loc[1, "year"] = float("nan")
upload = BytesIO(); d.to_stata(upload, write_index=False); upload.seek(0); upload.name = "missing_keys.dta"
with patch("streamlit.file_uploader", return_value=upload): render_novice()
'''
        app = AppTest.from_string(script, default_timeout=60).run()
        self.assertEqual(app.radio(key="nv_structure").value, "面板")
        self.assertTrue(any("2 行缺失" in item.value for item in app.info))
        app.selectbox(key="nv_y").select("y").run()
        app.multiselect(key="nv_core").set_value(["x"]).run()
        app.button(key="nv_run").click().run()
        self.assertEqual(list(app.exception), [])
        ws = app.session_state["novice_workspace"]
        self.assertEqual(len(ws["data"]), 400)
        self.assertEqual(len(ws["job"]["data"]), 398)

    def test_uncertain_detection_does_not_block_manual_panel_choice(self):
        script = '''import streamlit as st
from io import BytesIO
from unittest.mock import patch
from test_tihu import panel
from tihu_novice import render_novice
d = panel()
d["period_code"] = d["year"]
d.loc[1, "year"] = d.loc[0, "year"]
upload = BytesIO(d.to_csv(index=False).encode()); upload.name = "ambiguous.csv"
with patch("streamlit.file_uploader", return_value=upload): render_novice()
'''
        app = AppTest.from_string(script, default_timeout=60).run()
        self.assertEqual(app.radio(key="nv_structure").value, "横截面")
        app.radio(key="nv_structure").set_value("面板").run()
        app.selectbox(key="nv_time").select("period_code").run()
        self.assertEqual(list(app.exception), [])
        self.assertIsNotNone(app.selectbox(key="nv_y"))

    def test_switch_from_household_head_to_respondent_id(self):
        script = '''import streamlit as st
from io import BytesIO
from unittest.mock import patch
from test_tihu import panel
from tihu_novice import render_novice
d = panel()
d.insert(0, "户主", d["id"] // 2)
d.insert(1, "受访者", d["id"])
d = d.drop(columns="id")
upload = BytesIO(d.to_csv(index=False).encode()); upload.name = "synthetic_household.csv"
with patch("streamlit.file_uploader", return_value=upload): render_novice()
'''
        app = AppTest.from_string(script, default_timeout=60).run()
        app.radio(key="nv_structure").set_value("面板").run()
        app.selectbox(key="nv_time").select("year").run()
        app.selectbox(key="nv_entity").select("受访者").run()
        self.assertEqual(app.selectbox(key="nv_time").value, "year")
        app.selectbox(key="nv_y").select("y").run()
        app.multiselect(key="nv_core").set_value(["x"]).run()
        app.button(key="nv_run").click().run()
        self.assertEqual(list(app.exception), [])
        self.assertEqual(app.session_state["novice_workspace"]["job"]["specs"][0].entity, "受访者")

    def test_panel_time_conversion_keeps_upload_unchanged(self):
        d = panel()
        d["text_year"] = d["year"].astype(str)
        prepared = prepare_panel_time(d, "text_year")
        self.assertTrue(np.issubdtype(prepared["text_year"].dtype, np.number))
        self.assertEqual(d["text_year"].dtype, object)

    def test_complete_search_and_bundle(self):
        d = cross()
        d["m"] = .8*d.x+d.m
        d["y"] += .8*d.m
        job = start_job(d, basic_spec(d, "y", ["x", "z"]), ["c"], ["m"], ["group"])
        original = d.copy(deep=True)
        for _ in range(20):
            advance_job(job)
            if job["done"]:
                break
        self.assertTrue(job["done"])
        self.assertEqual(set(job["best"]), {"x", "z"})
        self.assertEqual(len(job["records"]), 4)
        self.assertEqual(len(job["followups"]), 4)
        self.assertTrue(d.equals(original))
        with zipfile.ZipFile(io.BytesIO(full_bundle(job, "synthetic"))) as archive:
            self.assertIn("index.json", archive.namelist())
            self.assertIn("model_001/analysis.do", archive.namelist())
            self.assertIn("model_001/sample.dta", archive.namelist())
            self.assertEqual(json.loads(archive.read("search_plan.json"))["planned"], 4)

    def test_panel_did_and_probit_followups(self):
        for model in ["FE", "DID", "Probit"]:
            with self.subTest(model=model):
                d = panel() if model != "Probit" else cross()
                if model == "DID":
                    spec = basic_spec(d, "y", [], "id", "year", True, "D", 2017)
                elif model == "FE":
                    spec = basic_spec(d, "y", ["x"], "id", "year")
                else:
                    spec = basic_spec(d, "binary", ["x"])
                job = start_job(d, spec, ["c"], ["m"], ["z"])
                advance_job(job, batch=20)
                self.assertTrue(job["done"])
                self.assertEqual(len(job["best"]), 1)
                self.assertEqual(len(job["followups"]), 2, job["failures"])
                self.assertTrue(all(np.isfinite(row["筛选 p"]) for row in job["followups"]))

    def test_modes_and_new_file_invalidation(self):
        script = '''import streamlit as st
from io import BytesIO
from unittest.mock import patch
from test_tihu import cross
from tihu_novice import render_entry
st.set_page_config(layout="wide")
upload=BytesIO(cross(seed=st.session_state.get("test_seed",42)).to_csv(index=False).encode()); upload.name="synthetic.csv"
with patch("streamlit.file_uploader", side_effect=lambda *a, **kw: upload if not st.session_state.get("test_no_upload", False) and kw.get("key") in {"nv_upload","data_upload"} else None):
    render_entry(lambda: st.title("鹈鹕回归"))
'''
        app = AppTest.from_string(script, default_timeout=60).run()
        self.assertEqual(len(app.exception), 0)
        app.button(key="mode_novice").click().run()
        app.multiselect(key="nv_core").set_value(["x"]).run()
        app.multiselect(key="nv_mediators").set_value(["m"]).run()
        app.multiselect(key="nv_moderators").set_value(["z"]).run()
        app.button(key="nv_run").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(app.session_state["novice_workspace"]["job"]["done"])
        app.button(key="nv_pack").click().run()
        self.assertIn("bundle", app.session_state["novice_workspace"])
        app.button(key="mode_back").click().run()
        app.button(key="mode_expert").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.selectbox(key="cfg_y").options) > 0, True)
        expert_y = app.selectbox(key="cfg_y").value
        app.button(key="mode_back").click().run()
        app.session_state["test_no_upload"] = True
        app.button(key="mode_novice").click().run()
        self.assertTrue(app.session_state["novice_workspace"]["job"]["done"])
        self.assertEqual(app.multiselect(key="nv_core").value, ["x"])
        app.button(key="mode_back").click().run()
        app.button(key="mode_expert").click().run()
        self.assertEqual(app.selectbox(key="cfg_y").value, expert_y)
        app.button(key="mode_back").click().run()
        app.button(key="mode_novice").click().run()
        app.session_state["test_no_upload"] = False
        app.session_state["test_seed"] = 8
        app.run()
        self.assertNotIn("job", app.session_state["novice_workspace"])

    def test_did_ui_requires_policy_time(self):
        script = '''import streamlit as st
from io import BytesIO
from unittest.mock import patch
from test_tihu import panel
from tihu_novice import render_novice
upload=BytesIO(panel().to_csv(index=False).encode()); upload.name="panel.csv"
with patch("streamlit.file_uploader", return_value=upload): render_novice()
'''
        app = AppTest.from_string(script, default_timeout=60).run()
        app.radio(key="nv_did").set_value("做 DID").run()
        self.assertEqual(app.selectbox(key="nv_entity").value, "id")
        self.assertEqual(app.selectbox(key="nv_time").value, "year")
        self.assertIsNone(app.selectbox(key="nv_policy").value)
        self.assertNotIn("nv_run", [button.key for button in app.button])
        app.selectbox(key="nv_policy").select(2017).run()
        self.assertEqual(len(app.error), 0)
        app.button(key="nv_run").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state["novice_workspace"]["job"]["best"]["DID"].spec.model, "DID")


if __name__ == "__main__":
    unittest.main()
