"""Budget, batching and UI regression checks for expert searches."""
from collections import Counter
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest
from test_tihu import cross
from tihu_core import ModelSpec
from tihu_ui import expert_specs, advance_search


def job_for(data, specs):
    return dict(data=data, specs=specs, index=0, fits=[], failures={}, records=[],
                joint=False, done=False, cancelled=False, elapsed=0.)


class ExpertSearchTests(unittest.TestCase):
    def test_switch_from_household_head_to_respondent_id(self):
        script = '''import streamlit as st
from io import BytesIO
from unittest.mock import patch
from test_tihu import panel
from tihu_ui import render_workbench
d = panel()
d.insert(0, "户主", d["id"] // 2)
d.insert(1, "受访者", d["id"])
d = d.drop(columns="id")
upload = BytesIO(d.to_csv(index=False).encode()); upload.name = "synthetic_household.csv"
with patch("streamlit.file_uploader", side_effect=lambda *a, **kw: upload if kw.get("key") == "data_upload" else None):
    render_workbench()
'''
        app = AppTest.from_string(script, default_timeout=60).run()
        app.selectbox(key="cfg_structure").select("面板").run()
        self.assertEqual(app.selectbox(key="cfg_entity").value, "户主")
        app.selectbox(key="cfg_entity").select("受访者").run()
        self.assertEqual(app.selectbox(key="cfg_time").value, "year")
        app.selectbox(key="cfg_y").select("y").run()
        app.multiselect(key="cfg_core").set_value(["x"]).run()
        app.number_input(key="cfg_budget").set_value(1).run()
        app.button(key="run_search").click().run()
        self.assertEqual(list(app.exception), [])
        self.assertEqual(app.session_state["workspace_v2"]["job"]["specs"][0].entity, "受访者")

    def test_panel_columns_can_be_changed_to_numeric_text_time(self):
        script = '''import streamlit as st
from test_tihu import panel
from tihu_ui import specification
d = panel()
d["firm_code"] = "F" + d["id"].astype(str)
d["wave_text"] = d["year"].astype(str)
st.session_state["test_result"] = specification(d)
'''
        app = AppTest.from_string(script, default_timeout=60).run()
        app.selectbox(key="cfg_structure").select("面板").run()
        self.assertEqual((app.selectbox(key="cfg_entity").value,
                          app.selectbox(key="cfg_time").value), ("id", "year"))
        app.selectbox(key="cfg_entity").select("firm_code").run()
        app.selectbox(key="cfg_time").select("wave_text").run()
        self.assertEqual(list(app.exception), [])
        spec, _, prepared = app.session_state["test_result"]
        self.assertEqual((spec.entity, spec.time), ("firm_code", "wave_text"))
        self.assertTrue(np.issubdtype(prepared["wave_text"].dtype, np.number))

    def test_text_time_is_recorded_in_expert_replay(self):
        script = '''import streamlit as st
from io import BytesIO
from unittest.mock import patch
from test_tihu import panel
from tihu_ui import render_workbench
d = panel()
d["firm_code"] = "F" + d["id"].astype(str)
d["wave_text"] = d["year"].astype(str)
d["bad_time"] = "not a period"
upload = BytesIO(); d.to_stata(upload, write_index=False); upload.seek(0); upload.name = "synthetic_panel.dta"
with patch("streamlit.file_uploader", side_effect=lambda *a, **kw: upload if kw.get("key") == "data_upload" else None):
    render_workbench()
'''
        app = AppTest.from_string(script, default_timeout=60).run()
        app.selectbox(key="cfg_structure").select("面板").run()
        app.selectbox(key="cfg_entity").select("firm_code").run()
        app.selectbox(key="cfg_time").select("wave_text").run()
        app.selectbox(key="cfg_y").select("y").run()
        app.multiselect(key="cfg_core").set_value(["x"]).run()
        app.number_input(key="cfg_budget").set_value(1).run()
        app.button(key="run_search").click().run()
        ws = app.session_state["workspace_v2"]
        self.assertEqual(ws["job"]["export_steps"], [{"op": "numeric", "cols": ["wave_text"]}])
        self.assertEqual(list(app.exception), [])
        app.button(key="build_bundle").click().run()
        self.assertIn("bundle", app.session_state["workspace_v2"])
        _, do = app.session_state["workspace_v2"]["bundle"]
        self.assertIn("destring", do)
        app.selectbox(key="cfg_time").select("bad_time").run()
        self.assertNotIn("job", app.session_state["workspace_v2"])
        self.assertNotIn("fits", app.session_state["workspace_v2"])

    def test_budget_counts_each_x_once(self):
        spec = ModelSpec("OLS", "y", ["x", "z", "m"], controls=["required"])
        pool = [f"c{i}" for i in range(20)]
        specs = expert_specs(spec, pool, 6, 10, 10000, joint=False)
        keys = [(tuple(s.core), tuple(s.controls)) for s in specs]
        self.assertEqual(len(keys), 10000)
        self.assertEqual(len(set(keys)), 10000)
        counts = Counter(s.core[0] for s in specs)
        self.assertLessEqual(max(counts.values())-min(counts.values()), 1)
        self.assertTrue(all(7 <= len(s.controls) <= 11 and "required" in s.controls for s in specs))
        self.assertEqual(keys, [(tuple(s.core), tuple(s.controls)) for s in expert_specs(spec, pool, 6, 10, 10000, False)])
        self.assertEqual(len(expert_specs(spec, pool[:3], 0, 10, 10000, False)), 24)
        self.assertEqual(len(expert_specs(spec, pool[:3], 0, 10, 10000, True)), 8)
        self.assertEqual(len(expert_specs(spec, pool, 0, 20, 7, False)), 7)
        with self.assertRaises(ValueError):
            expert_specs(spec, pool, 0, 10, 2, False)
        esr = ModelSpec("ESR", "y", controls=["required"], instruments=["iv"], treatment="D")
        for s in expert_specs(esr, pool[:3], 0, 3, 10000):
            self.assertEqual(s.selection, s.controls+["iv"])

    def test_batches_record_all_and_keep_best(self):
        d = cross()
        pool = [f"c{i}" for i in range(6)]
        for i, col in enumerate(pool):
            d[col] = np.random.default_rng(i).normal(size=len(d))
        original = d.copy(deep=True)
        job = job_for(d, expert_specs(ModelSpec("OLS", "y", ["x"]), pool, 0, 6, 10000))
        with patch("tihu_ui.time.perf_counter", side_effect=[0., 1., 2.]):
            advance_search(job, seconds=.75)
        self.assertEqual(job["index"], 1)
        job.update(done=True, cancelled=True)
        advance_search(job)
        self.assertEqual(job["index"], 1)
        job.update(done=False, cancelled=False)
        advance_search(job, seconds=None)
        self.assertEqual(len(job["records"]), 64)
        self.assertEqual(len(job["fits"]), 50)
        self.assertTrue(job["done"])
        self.assertEqual([p for p, _ in job["fits"]], sorted(row["排序 p"] for row in job["records"])[:50])
        self.assertTrue(d.equals(original))
        failed = job_for(d, [replace(job["specs"][0], core=["absent"])])
        advance_search(failed)
        self.assertTrue(failed["done"])
        self.assertEqual(sum(failed["failures"].values()), 1)

    def test_ui_defaults_resume_reload_and_invalidation(self):
        script = '''import streamlit as st
import numpy as np
from test_tihu import cross
from tihu_core import ModelSpec
from tihu_ui import search_controls, results_ui
d = cross()
for i in range(10): d[f"c{i}"] = np.random.default_rng(i).normal(size=len(d))
ws = st.session_state.setdefault("workspace_v2", {"hash":"synthetic"})
search_controls(ws, d, ModelSpec("OLS", "y", ["x"]), [f"c{i}" for i in range(10)])
results_ui(ws, d)
'''
        app = AppTest.from_string(script, default_timeout=60).run()
        self.assertEqual(app.number_input(key="cfg_budget").value, 10000)
        self.assertEqual(app.number_input(key="cfg_max").value, 10)
        app.button(key="run_search").click().run()
        self.assertFalse(app.session_state["workspace_v2"]["job"]["done"])
        app.button(key="cancel_search").click().run()
        job = app.session_state["workspace_v2"]["job"]
        paused = job["index"]
        self.assertTrue(job["cancelled"])
        app.selectbox(key="inspect_combination").select(job["records"][-1]["组合编号"]).run()
        app.button(key="load_combination").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state["workspace_v2"]["fits"][0].spec, job["specs"][job["records"][-1]["组合编号"]-1])
        app.button(key="resume_search").click().run()
        self.assertGreater(app.session_state["workspace_v2"]["job"]["index"], paused)
        app.number_input(key="cfg_min").set_value(6).run()
        self.assertNotIn("job", app.session_state["workspace_v2"])
        self.assertNotIn("fits", app.session_state["workspace_v2"])
        self.assertEqual(len(app.exception), 0)


if __name__ == "__main__":
    unittest.main()
