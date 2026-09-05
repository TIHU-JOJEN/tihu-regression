"""Optional local Stata parity audit using generated data only."""
import io
import json
import re
from pathlib import Path
import subprocess
import zipfile
import numpy as np
import pandas as pd
from test_tihu import cross, panel
from tihu_core import ModelSpec, fit_model
from tihu_export import reproducibility_bundle

ROOT = Path('/private/tmp/tihu-stata-audit')
BIN = '/Applications/Stata/StataMP.app/Contents/MacOS/stata-mp'
EXTRACT = '''
matrix __b = e(b)
matrix __v = e(V)
local __names : colfullnames __b
tempname __handle
file open `__handle' using "stata_estimates.csv", write replace
file write `__handle' "term,coef,se" _n
forvalues j = 1/`=colsof(__b)' {
    local term : word `j' of `__names'
    file write `__handle' "`term'," %24.17g (__b[1,`j']) "," %24.17g (sqrt(__v[`j',`j'])) _n
}
file close `__handle'
'''


def run(name, d, spec):
    destination = ROOT/name
    destination.mkdir(parents=True, exist_ok=True)
    (destination/'stata_estimates.csv').unlink(missing_ok=True)
    fit = fit_model(d, spec)
    bundle, do = reproducibility_bundle(d, [], fit, 'synthetic')
    with zipfile.ZipFile(io.BytesIO(bundle)) as z:
        z.extractall(destination)
    # Capture every marginal effect before the next command overwrites r().
    lines = []
    margins = 0
    for line in do.splitlines():
        lines.append(line)
        if line.startswith('margins if'):
            margins += 1
            lines += ['matrix __effect = r(table)', 'tempname __eh', f'file open `__eh\' using "effect{margins}.csv", write replace', 'file write `__eh\' "coef,se" _n', 'file write `__eh\' %24.17g (__effect[1,1]) "," %24.17g (__effect[2,1]) _n', 'file close `__eh\'']
    (destination/'analysis.do').write_text('\n'.join(lines)+EXTRACT)
    completed = subprocess.run([BIN, '-q', '-b', 'do', 'analysis.do'], cwd=destination, timeout=90)
    log = (destination/'analysis.log').read_text(errors='replace')
    if not (destination/'stata_estimates.csv').exists() or re.search(r'\nr\([1-9][0-9]*\);', log):
        raise AssertionError(name+' FAILED '+log[-3500:])
    table = pd.read_csv(destination/'stata_estimates.csv')
    manifest = json.loads((destination/'manifest.json').read_text())
    names = manifest['column_names']
    if spec.model != 'ESR':
        focus = fit.effect['term']
        stataterm = '_cons' if focus == 'const' else names.get(focus, focus)
        row = table[table.term.str.split(':').str[-1] == stataterm].iloc[0]
        print(name, 'coef_delta', float(row.coef-fit.effect['coef']), 'se_delta', float(row.se-fit.effect['se']))
        assert abs(row.coef-fit.effect['coef']) < 2e-4, name
        assert abs(row.se-fit.effect['se']) < 2e-4, name
    else:
        for i, effect in enumerate(['ATT', 'ATU'], 1):
            actual = pd.read_csv(destination/f'effect{i}.csv').iloc[0]
            expected = fit.details['effects'].set_index('term').loc[effect]
            print(name, effect, 'coef_delta', float(actual.coef-expected.coef), 'se_delta', float(actual.se-expected.se))
            assert abs(actual.coef-expected.coef) < 2e-4
            assert abs(actual.se-expected.se) < 2e-4
    return True


if __name__ == '__main__':
    d = cross(700)
    cases = [('ols', ModelSpec('OLS','y',['x'],['c'],se='robust')),
             ('cluster', ModelSpec('OLS','y',['x'],['c'],se='cluster',cluster='group')),
             ('logit', ModelSpec('Logit','binary',['x'],['c'])),
             ('poisson', ModelSpec('Poisson','count',['x'],['c']))]
    for name, spec in cases:
        run(name, d, spec)
    rng = np.random.default_rng(31)
    n = 900
    x,z,u,e0,e1 = rng.normal(size=(5,n))
    D = (z+.4*x+u>0).astype(int)
    df = pd.DataFrame({'y':np.where(D,3+.7*x+.35*u+e1,1+.7*x+.35*u+e0),'x':x,'z':z,'D':D})
    run('esr', df, ModelSpec('ESR','y',controls=['x'],treatment='D',selection=['x','z'],se='ordinary'))
    run('esr_robust', df, ModelSpec('ESR','y',controls=['x'],treatment='D',selection=['x','z'],se='robust'))
    p = panel()
    run('fe',p,ModelSpec('FE','y',['x'],['c'],entity='id',time='year',se='ordinary'))
    run('re',p,ModelSpec('RE','y',['x'],['c'],entity='id',time='year',se='ordinary'))
    run('did',p,ModelSpec('DID','y',controls=['c'],entity='id',time='year',treatment='D',policy=2017,se='cluster',cluster='id'))
    d['censored'] = d.y.clip(lower=0)
    run('tobit',d,ModelSpec('Tobit','censored',['x'],['c'],left=0.,se='ordinary'))
    limited = d.copy(); limited.loc[limited.D==0,'y'] = np.nan
    run('heckman',limited,ModelSpec('Heckman','y',['x'],['c'],treatment='D',selection=['x','c','z'],se='ordinary'))
    d['endog'] = d.z+d.x; d['outcome'] = 2*d.endog+d.c+d.m
    run('iv',d,ModelSpec('IV/2SLS','outcome',['endog'],['c'],instruments=['z']))
    d['D'] = (np.random.default_rng(3).uniform(size=len(d)) < (.2+.6*(d.x>=0))).astype(int)
    d['y'] += 2*d.D
    run('sharp',d,ModelSpec('Sharp RD','y',controls=['c'],running='x',bandwidth=.9))
    run('fuzzy',d,ModelSpec('Fuzzy RD','y',controls=['c'],running='x',bandwidth=.9,treatment='D'))
