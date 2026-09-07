"""Optional synthetic 10,000-fit benchmark; never loads user datasets."""
from time import perf_counter
import resource
import sys
import numpy as np
import pandas as pd
from tihu_novice import basic_spec, start_job, advance_job


if __name__ == "__main__":
    rng = np.random.default_rng(29)
    data = pd.DataFrame(rng.normal(size=(400, 15)), columns=[f"c{i}" for i in range(15)])
    pool = list(data.columns)
    data["x"] = rng.normal(size=len(data))
    data["y"] = 1.2*data.x+.5*data.c0+rng.normal(size=len(data))
    started = perf_counter()
    expert = "--expert" in sys.argv
    if expert:
        from tihu_ui import expert_specs, advance_search
        specs = expert_specs(basic_spec(data, "y", ["x"]), pool, 6, 10, 10000)
        job = dict(data=data, specs=specs, index=0, fits=[], failures={}, records=[],
                   joint=True, done=False, cancelled=False, elapsed=0.)
    else:
        job = start_job(data, basic_spec(data, "y", ["x"]), pool, [], [], minimum=6)
    while not job["done"]:
        if expert:
            advance_search(job, batch=1000, seconds=None)
        else:
            advance_job(job, batch=1000)
        print(f"{job['index']:,} fits; {perf_counter()-started:.1f}s", flush=True)
    assert len(job["records"]) == 10000, job["failures"]
    assert all(6 <= len(s.controls) <= 10 for s in job["specs"])
    print({"mode": "expert" if expert else "novice", "rows": len(data), "model": "OLS, HC1", "successful": len(job["records"]),
           "seconds": round(perf_counter()-started, 2),
           "peak_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
