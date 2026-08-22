"""
鹈鹕回归 MVP v7：面板/横截面/时间序列 · 全模型家族
"""
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Poisson, NegativeBinomial
from linearmodels.panel import PanelOLS, RandomEffects
from linearmodels.iv import IV2SLS
from scipy import stats
from scipy.optimize import minimize
from itertools import combinations
from sklearn.neighbors import NearestNeighbors
import time, warnings, traceback
import base64
from pathlib import Path
warnings.filterwarnings('ignore')

st.set_page_config(page_title="鹈鹕回归", layout="wide")

for k, v in [('df',None),('df_name',None),('data_type',None),('diagnosed',False),
             ('id_col',None),('time_col',None),('balanced',None),('n_units',0),('n_periods',0),
             ('search_results',None),('raw_df',None),('cleaning_log',[]),('cleaning_rules',[]),
             ('transform_log',[])]:
    if k not in st.session_state: st.session_state[k] = v

# ── 免责声明模板（统一注入 Stata 代码头部） ──
LEGAL_HEADER = """* ============================================================
* 鹈鹕回归 (c) {year} · 自动生成 · 仅供参考
* 本代码仅供学术研究参考，不构成统计咨询或数据分析服务。
* 使用者应自行验证模型设定的合理性与结果的稳健性。
* 工具开发者不对因使用本代码产生的任何学术后果承担责任。
* ============================================================

"""

def asset_data_uri(rel_path):
    path=Path(__file__).resolve().parent / rel_path
    try:
        return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
    except Exception:
        return ""

def render_home_hero():
    mascot_src=asset_data_uri("assets/tihu-pelican-line-45.png")
    mascot_html=f'<img class="pelican-mascot" src="{mascot_src}" alt="鹈鹕回归吉祥物">' if mascot_src else ""
    st.markdown(f"""
    <style>
      .main .block-container {{
        padding-top: 1.6rem;
      }}
      .pelican-hero {{
        display: grid;
        grid-template-columns: minmax(280px, 1.08fr) minmax(320px, .92fr);
        gap: 34px;
        align-items: stretch;
        min-height: 430px;
        margin: 0 0 28px 0;
        padding: 34px;
        border: 1px solid #dfe7e2;
        border-radius: 18px;
        background:
          linear-gradient(135deg, rgba(11, 107, 110, .08), rgba(238, 107, 91, .06) 46%, rgba(255,255,255,.94)),
          #fbfcfa;
        box-shadow: 0 18px 45px rgba(28, 45, 38, .08);
      }}
      .pelican-brand {{
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-width: 0;
      }}
      .pelican-kicker {{
        font-size: 14px;
        color: #0b6b6e;
        font-weight: 700;
        letter-spacing: 0;
        margin-bottom: 12px;
      }}
      .pelican-title {{
        font-size: clamp(52px, 7vw, 92px);
        line-height: .98;
        letter-spacing: 0;
        color: #14211d;
        font-weight: 900;
        margin: 0;
      }}
      .pelican-subtitle {{
        margin-top: 18px;
        max-width: 650px;
        color: #4e5d57;
        font-size: 18px;
        line-height: 1.7;
      }}
      .pixel-stage {{
        margin-top: 28px;
        display: flex;
        align-items: center;
      }}
      .pelican-mark {{
        width: min(430px, 100%);
        height: 270px;
        display: flex;
        align-items: center;
        justify-content: flex-start;
      }}
      .pelican-mascot {{
        width: auto;
        max-width: 100%;
        max-height: 100%;
        display: block;
        filter: drop-shadow(0 14px 16px rgba(20,33,29,.08));
      }}
      .pelican-flow {{
        background: rgba(255,255,255,.82);
        border: 1px solid #dfe7e2;
        border-radius: 16px;
        padding: 26px;
        display: flex;
        flex-direction: column;
        justify-content: center;
      }}
      .flow-title {{
        color: #14211d;
        font-size: 24px;
        font-weight: 850;
        margin-bottom: 18px;
      }}
      .flow-list {{
        display: grid;
        gap: 12px;
      }}
      .flow-item {{
        display: grid;
        grid-template-columns: 46px 1fr;
        gap: 14px;
        align-items: center;
        padding: 12px 0;
        border-bottom: 1px solid #e6ece8;
      }}
      .flow-item:last-child {{ border-bottom: 0; }}
      .flow-num {{
        width: 40px;
        height: 40px;
        border-radius: 12px;
        display: grid;
        place-items: center;
        font-weight: 850;
        color: #ffffff;
        background: #14211d;
      }}
      .flow-item:nth-child(2) .flow-num {{ background: #0b6b6e; }}
      .flow-item:nth-child(3) .flow-num {{ background: #5d6b32; }}
      .flow-item:nth-child(4) .flow-num {{ background: #7a4f20; }}
      .flow-item:nth-child(5) .flow-num {{ background: #b64d3e; }}
      .flow-item:nth-child(6) .flow-num {{ background: #324b7a; }}
      .flow-copy {{
        color: #24322d;
        font-size: 17px;
        font-weight: 720;
        line-height: 1.45;
      }}
      .flow-note {{
        margin-top: 20px;
        color: #63726b;
        font-size: 14px;
        line-height: 1.6;
      }}
      @media (max-width: 900px) {{
        .pelican-hero {{
          grid-template-columns: 1fr;
          padding: 24px;
        }}
        .pixel-stage {{
          align-items: start;
        }}
        .pelican-mark {{
          width: min(330px, 100%);
          height: 230px;
        }}
      }}
    </style>
    <section class="pelican-hero">
      <div class="pelican-brand">
        <div>
          <div class="pelican-kicker">计量实证流程辅助工具</div>
          <h1 class="pelican-title">鹈鹕回归</h1>
          <div class="pelican-subtitle">
            面向论文实证与数据分析场景，把清洗、变量加工、模型选择、显著性搜寻和可复现代码组织成一条连续工作流。
          </div>
        </div>
        <div class="pixel-stage">
          <div class="pelican-mark" aria-label="white pelican">
            {mascot_html}
          </div>
        </div>
      </div>
      <div class="pelican-flow">
        <div class="flow-title">产品功能流程</div>
        <div class="flow-list">
          <div class="flow-item"><div class="flow-num">01</div><div class="flow-copy">上传数据</div></div>
          <div class="flow-item"><div class="flow-num">02</div><div class="flow-copy">清洗数据</div></div>
          <div class="flow-item"><div class="flow-num">03</div><div class="flow-copy">分析数据类型</div></div>
          <div class="flow-item"><div class="flow-num">04</div><div class="flow-copy">推荐和选择模型</div></div>
          <div class="flow-item"><div class="flow-num">05</div><div class="flow-copy">全流程显著实证搜寻</div></div>
          <div class="flow-item"><div class="flow-num">06</div><div class="flow-copy">自动生成可复现 Stata 代码</div></div>
        </div>
        <div class="flow-note">从这里开始，下面直接进入数据上传与实证流程。</div>
      </div>
    </section>
    """, unsafe_allow_html=True)

render_home_hero()

# ── 侧边栏：法律声明 ──
with st.sidebar:
    st.markdown("### 鹈鹕回归")
    st.caption("计量实证辅助工具")
    st.divider()
    with st.expander("免责声明", expanded=False):
        st.caption(
            "1. 本工具仅供学术研究参考，不构成任何形式的"
            "统计咨询或数据分析服务。\n\n"
            "2. 所有分析结果应由使用者自行验证其正确性与"
            "适用性。开发者不对因使用本工具产生的任何"
            "直接或间接后果承担责任。\n\n"
            "3. 生成的模型结果不构成论文实证章节的最终"
            "依据，请结合专业知识与相关文献综合判断。\n\n"
            "4. 上传的数据仅存储于您本地浏览器内存，"
            "不会上传至任何远程服务器。"
        )
    st.divider()
    st.caption(f"(c) 2026 鹈鹕回归 · 保留所有权利")

# ═══════════════════ MODEL HELPERS ═══════════════════
def tobit_mle(y, X, left=None, right=None):
    """Tobit MLE: left and/or right censored."""
    Xa=np.asarray(X,dtype=float); ya=np.asarray(y,dtype=float); k=Xa.shape[1]
    ols=OLS(ya,Xa).fit(); ols_p=ols.params.values if hasattr(ols.params,'values') else ols.params; b0=np.append(ols_p,np.log(max(ols.scale**0.5,0.01)))
    def nll(p):
        beta=p[:k]; sigma=np.exp(p[k])
        xb=Xa@beta; ll=0.0
        for i in range(len(ya)):
            if left is not None and ya[i]<=left: ll+=stats.norm.logcdf((left-xb[i])/sigma)
            elif right is not None and ya[i]>=right: ll+=stats.norm.logsf((right-xb[i])/sigma)
            else: resid=(ya[i]-xb[i])/sigma; ll+=stats.norm.logpdf(resid)-np.log(sigma)
        return -ll
    try:
        res=minimize(nll,b0,method='BFGS',options={'maxiter':500})
        beta=res.x[:k]; sigma=np.exp(res.x[k])
        se=np.full(k,np.nan)
        try:
            eps=1e-5; H=np.zeros((k+1,k+1)); nll0=nll(res.x)
            for ii in range(k+1):
                for jj in range(ii,k+1):
                    if ii==jj:
                        xp=res.x.copy(); xp[ii]+=eps
                        xm=res.x.copy(); xm[ii]-=eps
                        H[ii,jj]=(nll(xp)-2*nll0+nll(xm))/(eps*eps)
                    else:
                        xpp=res.x.copy(); xpp[ii]+=eps; xpp[jj]+=eps
                        xpm=res.x.copy(); xpm[ii]+=eps; xpm[jj]-=eps
                        xmp=res.x.copy(); xmp[ii]-=eps; xmp[jj]+=eps
                        xmm=res.x.copy(); xmm[ii]-=eps; xmm[jj]-=eps
                        H[ii,jj]=(nll(xpp)-nll(xpm)-nll(xmp)+nll(xmm))/(4*eps*eps)
                    H[jj,ii]=H[ii,jj]
            cov=np.linalg.pinv(H); se=np.sqrt(np.diag(cov))[:k]
        except: pass
        ll_full=-res.fun
        return {'params':beta,'se':se,'llf':ll_full,'sigma':sigma,'n':len(ya),'converged':res.success}
    except Exception as e:
        return {'params':np.full(k,np.nan),'se':np.full(k,np.nan),'llf':np.nan,'sigma':np.nan,'n':len(ya),'converged':False,'error':str(e)}

def heckman_two_step(y, X, Z):
    """Heckman two-step: X=outcome vars, Z=selection vars (includes X's exogenous vars + exclusion)."""
    sel_y=(~np.isnan(y)).astype(float)
    sel_m=sm.Probit(sel_y,sm.add_constant(Z)).fit(disp=0)
    xb=sm.add_constant(Z)@sel_m.params
    imr=stats.norm.pdf(xb)/stats.norm.cdf(xb)  # inverse Mills ratio: φ(xβ)/Φ(xβ)
    idx_obs=sel_y==1
    X2=sm.add_constant(np.column_stack([X[idx_obs],imr[idx_obs]]))
    m2=OLS(y[idx_obs],X2).fit()
    imr_coef=m2.params[-1]; imr_p=m2.pvalues[-1]
    return {'params':m2.params[:-1],'se':m2.bse[:-1],'imr_coef':imr_coef,'imr_p':imr_p,
            'rsq':m2.rsquared,'n_obs':int(sum(idx_obs)),'n_total':len(y),
            'sel_params':sel_m.params,'sel_se':sel_m.bse}

def psm_att(y, D, X, method='nearest', k=1, caliper=None):
    """PSM: estimate ATT via nearest-neighbor or kernel matching on propensity score."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Xs=StandardScaler().fit_transform(X)
    ps=LogisticRegression(C=1e6,max_iter=1000).fit(Xs,D).predict_proba(Xs)[:,1]
    treated=np.where(D==1)[0]; control=np.where(D==0)[0]
    if method=='nearest':
        nn=NearestNeighbors(n_neighbors=k).fit(ps[control].reshape(-1,1))
        dist,idx=nn.kneighbors(ps[treated].reshape(-1,1))
        att=0; n_t=len(treated)
        for i,ti in enumerate(treated):
            matched=control[idx[i]]
            if caliper:
                matched=matched[dist[i]<caliper]
            if len(matched)>0: att+=y[ti]-y[matched].mean()
            else: n_t-=1
        att=att/max(n_t,1)
    else:  # kernel
        bw=ps.std()/4
        att=0; n_t=len(treated)
        for ti in treated:
            w=np.exp(-0.5*((ps[control]-ps[ti])/bw)**2); w/=w.sum()
            att+=y[ti]-np.sum(y[control]*w)
        att/=max(n_t,1)
    return {'ATT':att,'n_treated':len(treated),'n_control':len(control),'method':method}

def check_stationarity(series):
    """ADF test. Returns (adf_stat, p_value, is_stationary)."""
    r=sm.tsa.adfuller(series.dropna(),autolag='AIC')
    return r[0],r[1],r[1]<0.05

def detect_y_type(y_series):
    """Detect Y type: 'binary','ordered','count','continuous','censored'."""
    y=y_series.dropna(); u=sorted(y.unique()); nu=len(u)
    if nu==2 and set(u).issubset({0,1}): return 'binary',None
    if nu<=10 and all(v==int(v) for v in u):
        if u==list(range(min(u),max(u)+1)): return 'ordered',None
    if nu>2 and all(v>=0 and v==int(v) for v in y):
        return 'count',None
    atmin=(y==y.min()).mean(); atmax=(y==y.max()).mean()
    if atmin>0.05 or atmax>0.05: return 'censored',{'at_min':atmin,'y_min':y.min()}
    return 'continuous',None

def unique_keep_order(seq):
    out=[]; seen=set()
    for x in seq:
        if x is not None and x not in seen:
            out.append(x); seen.add(x)
    return out

def make_unique_col(base, existing):
    name=str(base).replace(' ','_')
    if name not in existing:
        return name
    i=2
    while f"{name}_{i}" in existing:
        i+=1
    return f"{name}_{i}"

def is_binary_series(s):
    u=pd.Series(s).dropna().unique()
    return len(u)==2

def prepare_model_frame(data, y_col, x_vars, extra_cols=None, min_n=30, require_x=True):
    """Build a numeric regression frame and remove regressors that cannot be estimated."""
    x_vars=unique_keep_order([v for v in x_vars if v in data.columns and v!=y_col])
    need=unique_keep_order([y_col]+x_vars+list(extra_cols or []))
    td=data[need].replace([np.inf,-np.inf],np.nan).dropna().copy()
    if len(td)<min_n:
        return None, [], [], f'有效样本量 {len(td)} < {min_n}'
    kept=[]; dropped=[]
    for v in x_vars:
        if v not in td.columns:
            dropped.append((v,'不存在')); continue
        td[v]=pd.to_numeric(td[v],errors='coerce')
        if td[v].isna().any():
            dropped.append((v,'非数值')); continue
        if td[v].nunique(dropna=True)<=1:
            dropped.append((v,'无变化')); continue
        kept.append(v)
    td=td[[y_col]+kept+list(extra_cols or [])].dropna()
    if len(td)<min_n:
        return None, kept, dropped, f'清理后有效样本量 {len(td)} < {min_n}'
    if require_x and not kept:
        return None, kept, dropped, '没有可用于估计的解释变量'
    return td, kept, dropped, None

def safe_panel_fit(y, X, entity_effects=True, time_effects=False, cov_type=None):
    """PanelOLS wrapper: drop absorbed variables instead of failing on common DID cases."""
    mod=PanelOLS(y,X,entity_effects=entity_effects,time_effects=time_effects,drop_absorbed=True,check_rank=False)
    if cov_type:
        return mod.fit(cov_type=cov_type)
    return mod.fit()

def run_mediation_wen2014(y_col, x, m, controls, df, n_boot=1000):
    """温忠麟 & 叶宝娟 (2014) 中介效应五步流程
    方程: (1) Y=cX (2) M=aX (3) Y=c'X+bM
    步骤: ①检验c ②依次检验a,b ③Bootstrap检验ab ④检验c' ⑤比较ab与c'符号"""
    ctrls=list(controls); td=df[[y_col,x,m]+ctrls].dropna()
    n=len(td)
    if n<30: return None
    # 方程(1): Y ~ X + controls → 总效应 c
    m1=OLS(td[y_col],sm.add_constant(td[[x]+ctrls])).fit()
    c,c_se,c_p=float(m1.params[x]),float(m1.bse[x]),float(m1.pvalues[x])
    # 方程(2): M ~ X + controls → a
    m2=OLS(td[m],sm.add_constant(td[[x]+ctrls])).fit()
    a,a_se,a_p=float(m2.params[x]),float(m2.bse[x]),float(m2.pvalues[x])
    # 方程(3): Y ~ X + M + controls → c' + b
    m3=OLS(td[y_col],sm.add_constant(td[[x,m]+ctrls])).fit()
    b,b_se,b_p=float(m3.params[m]),float(m3.bse[m]),float(m3.pvalues[m])
    cp,cp_se,cp_p=float(m3.params[x]),float(m3.bse[x]),float(m3.pvalues[x])
    ab=a*b
    # ── 步骤1: 检验系数 c ──
    framework='中介效应' if c_p<0.05 else '遮掩效应'
    # ── 步骤2: 依次检验 a 和 b ──
    ab_both_sig=(a_p<0.05) and (b_p<0.05)
    # ── 步骤3: Bootstrap 检验 H0: ab=0（当步骤2至少一个不显著时）──
    boot_ci=None; boot_sig=None
    if not ab_both_sig:
        boot_abs=[]
        rng=np.random.RandomState(42)
        for _ in range(n_boot):
            idx=rng.choice(n,n,replace=True); btd=td.iloc[idx]
            try:
                ba=float(OLS(btd[m],sm.add_constant(btd[[x]+ctrls])).fit().params[x])
                bb=float(OLS(btd[y_col],sm.add_constant(btd[[x,m]+ctrls])).fit().params[m])
                boot_abs.append(ba*bb)
            except: pass
        if len(boot_abs)>=100:
            ba_arr=np.array(boot_abs)
            boot_ci=(float(np.percentile(ba_arr,2.5)),float(np.percentile(ba_arr,97.5)))
            boot_sig=not (boot_ci[0]<=0<=boot_ci[1])
    # 间接效应是否显著？
    indirect_sig=ab_both_sig or (boot_sig if boot_sig is not None else False)
    # ── 步骤4+5: 检验 c' 并比较符号 ──
    if not indirect_sig:
        judgement='间接效应不显著，停止分析'; effect_type='无中介'; effect_size=None
    elif cp_p>=0.05:  # c' 不显著 → 完全中介
        judgement='c\' 不显著 → 只有中介效应（完全中介）'; effect_type='完全中介'; effect_size=None
    elif ab*cp>0:  # ab 与 c' 同号 → 部分中介
        es=ab/c if c!=0 else 0
        judgement=f'ab 与 c\' 同号 → 部分中介效应（ab/c={es:.3f}）'; effect_type='部分中介'; effect_size=es
    else:  # ab 与 c' 异号 → 遮掩效应
        es=abs(ab/cp) if cp!=0 else 0
        judgement=f'ab 与 c\' 异号 → 遮掩效应（|ab/c\'|={es:.3f}）'; effect_type='遮掩效应'; effect_size=es
    return {'c':c,'c_se':c_se,'c_p':c_p,
            'a':a,'a_se':a_se,'a_p':a_p,
            'b':b,'b_se':b_se,'b_p':b_p,
            'cp':cp,'cp_se':cp_se,'cp_p':cp_p,
            'ab':ab,'ab_both_sig':ab_both_sig,
            'boot_ci':boot_ci,'boot_sig':boot_sig,'indirect_sig':indirect_sig,
            'judgement':judgement,'effect_type':effect_type,'effect_size':effect_size,
            'framework':framework,'n':n,
            'r2_step1':float(m1.rsquared),'r2_step2':float(m2.rsquared),'r2_step3':float(m3.rsquared),
            'method':'温忠麟五步流程（2014）'}

def run_mediation_jiang(y_col, x, m, controls, df):
    """江艇 (2022) 渠道检验两步法：仅验证 D→M 前半段，不分解效应
    前提：M 与 Y 的因果关系在理论上足够直观，不需正式因果推断来论证 M→Y"""
    ctrls=list(controls); td=df[[y_col,x,m]+ctrls].dropna()
    if len(td)<30: return None
    # 第1步: 验证总效应 D→Y 存在
    m1=OLS(td[y_col],sm.add_constant(td[[x]+ctrls])).fit()
    c,c_se,c_p=float(m1.params[x]),float(m1.bse[x]),float(m1.pvalues[x])
    # 第2步: 验证 D→M 渠道
    m2=OLS(td[m],sm.add_constant(td[[x]+ctrls])).fit()
    a,a_se,a_p=float(m2.params[x]),float(m2.bse[x]),float(m2.pvalues[x])
    # 判断：只基于两步，不估计 c'（M 内生导致分解不可信）
    if c_p<0.05 and a_p<0.05: judgement='渠道成立：X→Y 显著（总效应存在）且 X→M 显著（渠道前半段成立）'
    elif c_p<0.05: judgement='渠道不成立：X→Y 显著但 X→M 不显著，无法证明该渠道存在'
    else: judgement='渠道不成立：X→Y 不显著，总效应不存在则渠道检验无意义'
    return {'c':c,'c_se':c_se,'c_p':c_p,
            'a':a,'a_se':a_se,'a_p':a_p,
            'judgement':judgement,'n':len(td),
            'r2_step1':float(m1.rsquared),'r2_step2':float(m2.rsquared),
            'method':'江艇渠道检验两步法（2022）',
            'note':'注意：本方法不估计 Y~X+M 方程，因 M 可能内生导致直接/间接效应分解不可信（江艇, 2022）。M 与 Y 的因果关系需由理论支撑，而非统计检验。'}

def run_moderation_analysis(y_col, x, m, controls, df):
    """调节效应：Y ~ X + M + X×M + controls, 简单斜率"""
    ctrls=list(controls); td=df[[y_col,x,m]+ctrls].dropna().copy()
    if len(td)<30: return None
    td['_inter']=td[x]*td[m]
    Xv=[x,m,'_inter']+ctrls; Xd=sm.add_constant(td[Xv])
    mf=OLS(td[y_col],Xd).fit()
    b_x=float(mf.params[x]); b_inter=float(mf.params['_inter'])
    se_inter=float(mf.bse['_inter']); p_inter=float(mf.pvalues['_inter'])
    m_mean,m_sd=float(td[m].mean()),float(td[m].std())
    vcov=mf.cov_params()
    def simple_slope(mv):
        slope=b_x+b_inter*mv
        vs=vcov.loc[x,x]+mv**2*vcov.loc['_inter','_inter']+2*mv*vcov.loc[x,'_inter']
        se_s=np.sqrt(vs) if vs>0 else 0
        return slope,se_s,slope/se_s if se_s>0 else 0
    lo_s,lo_se,lo_t=simple_slope(m_mean-m_sd)
    md_s,md_se,md_t=simple_slope(m_mean)
    hi_s,hi_se,hi_t=simple_slope(m_mean+m_sd)
    return {'b_x':b_x,'b_m':float(mf.params[m]),'b_inter':b_inter,
            'se_inter':se_inter,'p_inter':p_inter,
            'm_mean':m_mean,'m_sd':m_sd,
            'low_slope':lo_s,'low_se':lo_se,'low_t':lo_t,
            'med_slope':md_s,'med_se':md_se,'med_t':md_t,
            'high_slope':hi_s,'high_se':hi_se,'high_t':hi_t,
            'rsq':float(mf.rsquared),'n':len(td)}

def run_did_mediation_panel(td, y_col, med_col, controls, id_col, tc_col):
    """DID mechanism test using current DID sample: _did -> M -> Y with entity FE."""
    controls=[v for v in unique_keep_order(controls) if v in td.columns and v not in ['_treat','_post','_did',med_col,y_col,id_col,tc_col]]
    need=unique_keep_order([id_col,tc_col,y_col,med_col,'_did','_post']+controls)
    d=td[need].replace([np.inf,-np.inf],np.nan).dropna().copy()
    if len(d)<30: return None,'有效样本量不足'
    d[id_col]=d[id_col].astype(str)
    d=d.set_index([id_col,tc_col])
    base_x=[v for v in controls+['_post','_did'] if v in d.columns and d[v].nunique(dropna=True)>1]
    if '_did' not in base_x: return None,'Treat×Post 没有变化'
    m_x=[v for v in base_x if v!=med_col]
    y_x=unique_keep_order([v for v in controls+['_post','_did',med_col] if v in d.columns and d[v].nunique(dropna=True)>1])
    try:
        r_y=safe_panel_fit(d[y_col],sm.add_constant(d[m_x]),entity_effects=True,time_effects=False)
        r_m=safe_panel_fit(d[med_col],sm.add_constant(d[m_x]),entity_effects=True,time_effects=False)
        r_full=safe_panel_fit(d[y_col],sm.add_constant(d[y_x]),entity_effects=True,time_effects=False)
        c=float(r_y.params.get('_did',np.nan)); c_p=float(r_y.pvalues.get('_did',np.nan))
        a=float(r_m.params.get('_did',np.nan)); a_p=float(r_m.pvalues.get('_did',np.nan))
        b=float(r_full.params.get(med_col,np.nan)); b_p=float(r_full.pvalues.get(med_col,np.nan))
        cp=float(r_full.params.get('_did',np.nan)); cp_p=float(r_full.pvalues.get('_did',np.nan))
        ab=a*b
        if np.isnan(a) or np.isnan(b):
            effect_type='无法判断'
        elif a_p<0.1 and b_p<0.1:
            effect_type='机制成立'
        elif c_p<0.1 and a_p<0.1:
            effect_type='渠道成立'
        else:
            effect_type='暂不显著'
        return {'c':c,'c_p':c_p,'a':a,'a_p':a_p,'b':b,'b_p':b_p,'cp':cp,'cp_p':cp_p,
                'ab':ab,'effect_type':effect_type,'n':len(d),'controls':controls,
                'rsq_y':float(r_y.rsquared),'rsq_m':float(r_m.rsquared),'rsq_full':float(r_full.rsquared)},None
    except Exception as e:
        return None,str(e)[:120]

def run_did_moderation_panel(td, y_col, mod_col, controls, id_col, tc_col):
    """DID moderation test using current DID sample: _did × M -> Y with entity FE."""
    controls=[v for v in unique_keep_order(controls) if v in td.columns and v not in ['_treat','_post','_did',mod_col,y_col,id_col,tc_col]]
    need=unique_keep_order([id_col,tc_col,y_col,mod_col,'_did','_post']+controls)
    d=td[need].replace([np.inf,-np.inf],np.nan).dropna().copy()
    if len(d)<30: return None,'有效样本量不足'
    d['_did_x_m']=d['_did']*d[mod_col]
    d[id_col]=d[id_col].astype(str)
    d=d.set_index([id_col,tc_col])
    x_vars=unique_keep_order([v for v in controls+['_post','_did',mod_col,'_did_x_m'] if v in d.columns and d[v].nunique(dropna=True)>1])
    if '_did_x_m' not in x_vars: return None,'交互项没有变化'
    try:
        r=safe_panel_fit(d[y_col],sm.add_constant(d[x_vars]),entity_effects=True,time_effects=False)
        b_did=float(r.params.get('_did',np.nan)); b_m=float(r.params.get(mod_col,np.nan))
        b_inter=float(r.params.get('_did_x_m',np.nan)); p_inter=float(r.pvalues.get('_did_x_m',np.nan))
        return {'b_did':b_did,'b_m':b_m,'b_inter':b_inter,'p_inter':p_inter,
                'n':len(d),'controls':controls,'rsq':float(r.rsquared)},None
    except Exception as e:
        return None,str(e)[:120]

def refit_baseline(y_col, focus_cx, Xv0, data, model_sel, is_panel, use_rob=False,
                   id_col=None, tc_col=None, did_var=None, iv_endog=None, iv_insts=None,
                   heckman_sel=None, did_policy_time=None, did_post_var=None):
    """在数据子集上重新拟合基准模型。返回 {success,coef,se,tstat,pval,n,rsq,error}"""
    try:
        n_raw=len(data)
        if n_raw<30: return {'success':False,'error':'样本量<30','n':n_raw,'coef':np.nan,'se':np.nan,'tstat':np.nan,'pval':np.nan,'rsq':np.nan}

        # ── 面板 FE+RE ──
        if is_panel and 'FE+RE' in model_sel:
            if id_col is None or tc_col is None: return {'success':False,'error':'缺少面板ID','n':n_raw}
            need=[y_col]+[v for v in Xv0 if v in data.columns]+[id_col,tc_col]
            td=data[need].dropna().copy()
            td[id_col]=td[id_col].astype(str)
            if td.duplicated(subset=[id_col,tc_col]).sum()>0:
                td=td.groupby([id_col,tc_col]).mean().reset_index()
            td=td.set_index([id_col,tc_col])
            n=len(td)
            if n<30: return {'success':False,'error':'样本量<30','n':n}
            n_id=td.index.get_level_values(0).nunique()
            if n_id<2: return {'success':False,'error':'个体数<2无法固定效应','n':n}
            Xv=[v for v in Xv0 if v in td.columns]
            Xe=sm.add_constant(td[Xv])
            # FE
            fe_m=PanelOLS(td[y_col],Xe,entity_effects=True,time_effects=True)
            try:
                fe_r=fe_m.fit(cov_type='robust' if use_rob else 'unadjusted')
            except:
                fe_r=fe_m.fit()
            # RE
            re_m=RandomEffects(td[y_col],Xe)
            try:
                re_r=re_m.fit()
            except:
                fe=fe_r.params.get(focus_cx,np.nan); fese=fe_r.std_errors.get(focus_cx,np.nan)
                t=abs(fe/fese) if fese>0 else 0
                return {'success':True,'coef':float(fe),'se':float(fese),'tstat':float(t),
                    'pval':float(2*(1-stats.t.cdf(t,df=max(n-len(Xv)-n_id-1,1)))),
                    'n':n,'rsq':float(fe_r.rsquared)}
            # Hausman
            try:
                common=[c for c in fe_r.params.index.intersection(re_r.params.index) if c!='const']
                dp=fe_r.params.loc[common]-re_r.params.loc[common]
                Vv=fe_r.cov.loc[common,common].values-re_r.cov.loc[common,common].values
                hs=float(dp.values.T@np.linalg.pinv(Vv)@dp.values)
                hp=float(1-stats.chi2.cdf(hs,df=len(common)))
                use_fe=hp<0.05
            except:
                use_fe=True
            r=fe_r if use_fe else re_r
            b=r.params.get(focus_cx,np.nan); se=r.std_errors.get(focus_cx,np.nan)
            t=abs(b/se) if se>0 else 0
            df_eff=max(n-len(Xv)-n_id-1,1)
            return {'success':True,'coef':float(b),'se':float(se),'tstat':float(t),
                'pval':float(2*(1-stats.t.cdf(t,df=df_eff))),
                'n':n,'rsq':float(fe_r.rsquared)}

        # ── OLS / Pooled OLS ──
        if model_sel.startswith('OLS') or 'Pooled' in model_sel or 'RDD' in model_sel:
            need=[y_col]+[v for v in Xv0 if v in data.columns]
            td=data[need].dropna()
            n=len(td)
            if n<30: return {'success':False,'error':'样本量<30','n':n}
            Xv=[v for v in Xv0 if v in td.columns]
            Xd=sm.add_constant(td[Xv])
            cov_t='HC1' if use_rob else 'nonrobust'
            m=OLS(td[y_col].values,Xd).fit(cov_type=cov_t)
            b=m.params.get(focus_cx,np.nan); se=m.bse.get(focus_cx,np.nan)
            t=abs(b/se) if se>0 else 0
            return {'success':True,'coef':float(b),'se':float(se),'tstat':float(t),
                'pval':float(2*(1-stats.t.cdf(t,df=max(n-len(Xv)-1,1)))),
                'n':n,'rsq':float(m.rsquared)}

        # ── Logit / Probit / LPM ──
        if 'Ordered' not in model_sel and ('Logit' in model_sel or 'Probit' in model_sel or 'LPM' in model_sel):
            need=[y_col]+[v for v in Xv0 if v in data.columns]
            td=data[need].dropna()
            n=len(td)
            if n<30: return {'success':False,'error':'样本量<30','n':n}
            Xv=[v for v in Xv0 if v in td.columns]
            Xd=sm.add_constant(td[Xv])
            y_d=td[y_col]
            # Prefer Logit, then Probit, then LPM
            if 'Logit' in model_sel:
                m=sm.Logit(y_d,Xd).fit(disp=0)
                b=m.params.get(focus_cx,np.nan); se=m.bse.get(focus_cx,np.nan)
                t=abs(b/se) if se>0 else 0
                pv=float(2*(1-stats.norm.cdf(t)))
                return {'success':True,'coef':float(b),'se':float(se),'tstat':float(t),'pval':pv,'n':n,'rsq':float(m.prsquared)}
            if 'Probit' in model_sel:
                m=sm.Probit(y_d,Xd).fit(disp=0)
                b=m.params.get(focus_cx,np.nan); se=m.bse.get(focus_cx,np.nan)
                t=abs(b/se) if se>0 else 0
                pv=float(2*(1-stats.norm.cdf(t)))
                return {'success':True,'coef':float(b),'se':float(se),'tstat':float(t),'pval':pv,'n':n,'rsq':float(m.prsquared)}
            # LPM
            m=OLS(y_d,Xd).fit(cov_type='HC1')
            b=m.params.get(focus_cx,np.nan); se=m.bse.get(focus_cx,np.nan)
            t=abs(b/se) if se>0 else 0
            return {'success':True,'coef':float(b),'se':float(se),'tstat':float(t),
                'pval':float(2*(1-stats.t.cdf(t,df=max(n-len(Xv)-1,1)))),
                'n':n,'rsq':float(m.rsquared)}

        # ── Ordered Logit/Probit ──
        if 'Ordered' in model_sel:
            need=[y_col]+[v for v in Xv0 if v in data.columns]
            td=data[need].dropna()
            n=len(td)
            if n<30: return {'success':False,'error':'样本量<30','n':n}
            Xv=[v for v in Xv0 if v in td.columns]
            if not Xv: return {'success':False,'error':'Ordered 模型需要解释变量','n':n}
            Xd=td[Xv]
            distr='probit' if 'Probit' in model_sel else 'logit'
            m=OrderedModel(td[y_col].astype(int),Xd,distr=distr).fit(disp=0)
            b=m.params.get(focus_cx,np.nan); se=m.bse.get(focus_cx,np.nan)
            t=abs(b/se) if se>0 else 0
            return {'success':True,'coef':float(b),'se':float(se),'tstat':float(t),
                'pval':float(2*(1-stats.norm.cdf(t))),'n':n,'rsq':float(m.prsquared)}

        # ── Poisson / Negative Binomial ──
        if 'Poisson' in model_sel or '负二项' in model_sel:
            need=[y_col]+[v for v in Xv0 if v in data.columns]
            td=data[need].dropna()
            n=len(td)
            if n<30: return {'success':False,'error':'样本量<30','n':n}
            Xv=[v for v in Xv0 if v in td.columns]
            Xd=sm.add_constant(td[Xv])
            fam=Poisson() if 'Poisson' in model_sel else NegativeBinomial()
            cov_t='HC0' if use_rob else 'nonrobust'
            m=GLM(td[y_col],Xd,family=fam).fit(cov_type=cov_t)
            b=m.params.get(focus_cx,np.nan); se=m.bse.get(focus_cx,np.nan)
            t=abs(b/se) if se>0 else 0
            rsq=1-m.llf/m.llnull if hasattr(m,'llnull') and m.llnull!=0 else np.nan
            return {'success':True,'coef':float(b),'se':float(se),'tstat':float(t),
                'pval':float(2*(1-stats.norm.cdf(t))),'n':n,'rsq':float(rsq)}

        # ── Tobit ──
        if 'Tobit' in model_sel:
            need=[y_col]+[v for v in Xv0 if v in data.columns]
            td=data[need].dropna()
            n=len(td)
            if n<30: return {'success':False,'error':'样本量<30','n':n}
            Xv=[v for v in Xv0 if v in td.columns]
            Xd=sm.add_constant(td[Xv]).values
            y_d=td[y_col].values
            left_val=float(td[y_col].min()) if (td[y_col]==td[y_col].min()).mean()>0.05 else None
            tr=tobit_mle(y_d,Xd,left=left_val)
            idx=list(td[Xv].columns); idx=['const']+idx
            if focus_cx in idx:
                pi=idx.index(focus_cx)
                b=float(tr['params'][pi]); se=float(tr['se'][pi])
                t=abs(b/se) if se>0 else 0
                return {'success':True,'coef':b,'se':se,'tstat':t,
                    'pval':float(2*(1-stats.norm.cdf(t))),'n':n,
                    'rsq':float(1-tr['llf']/(n*np.log(np.std(y_d)))) if tr.get('llf') and not np.isnan(tr['llf']) else np.nan}
            return {'success':False,'error':f'变量 {focus_cx} 不在模型参数中','n':n}

        # ── DID (2×2 双重差分) ──
        if 'DID' in model_sel and 'PSM' not in model_sel:
            if not is_panel or id_col is None or tc_col is None:
                return {'success':False,'error':'DID 需要面板数据','n':n_raw}
            if did_var is None: return {'success':False,'error':'DID 需要处理变量','n':n_raw}
            need=[y_col]+[v for v in Xv0 if v in data.columns]+[id_col,tc_col,did_var]
            if did_post_var and did_post_var in data.columns and did_post_var not in need:
                need.append(did_post_var)
            td=data[need].dropna().copy()
            td[id_col]=td[id_col].astype(str)
            td[tc_col]=pd.to_numeric(td[tc_col],errors='coerce')
            td=td.dropna(subset=[tc_col])
            if did_post_var and did_post_var in td.columns:
                td['_post']=td[did_post_var].astype(float)
            else:
                cutoff = did_policy_time if did_policy_time is not None else td[tc_col].median()
                td['_post']=(td[tc_col]>=cutoff).astype(float)
            td['_treat']=td[did_var].astype(float)
            td['_did']=td['_treat']*td['_post']
            td=td.set_index([id_col,tc_col])
            n=len(td)
            if n<30: return {'success':False,'error':'样本量<30','n':n}
            did_terms=['_treat','_post','_did']
            controls=[v for v in Xv0 if v in td.columns and v not in did_terms]
            did_X=unique_keep_order(controls+['_post','_did'])
            did_X=[v for v in did_X if td[v].nunique(dropna=True)>1]
            if '_did' not in did_X: return {'success':False,'error':'DID 交互项没有变化，无法估计','n':n}
            X_did=sm.add_constant(td[did_X])
            d_r=safe_panel_fit(td[y_col],X_did,entity_effects=True,time_effects=False)
            b=d_r.params.get('_did',np.nan); se=d_r.std_errors.get('_did',np.nan)
            t=abs(b/se) if se>0 else 0
            df_eff=max(n-len(did_X)-td.index.get_level_values(0).nunique()-1,1)
            return {'success':True,'coef':float(b),'se':float(se),'tstat':float(t),
                'pval':float(2*(1-stats.t.cdf(t,df=df_eff))),'n':n,'rsq':float(d_r.rsquared)}

        # ── PSM-DID ──
        if 'PSM-DID' in model_sel:
            if not is_panel or id_col is None or tc_col is None:
                return {'success':False,'error':'PSM-DID 需要面板数据','n':n_raw}
            if did_var is None: return {'success':False,'error':'PSM-DID 需要处理变量','n':n_raw}
            need=[y_col]+[v for v in Xv0 if v in data.columns]+[id_col,tc_col,did_var]
            if did_post_var and did_post_var in data.columns and did_post_var not in need:
                need.append(did_post_var)
            td=data[need].dropna().copy()
            td[id_col]=td[id_col].astype(str)
            td[tc_col]=pd.to_numeric(td[tc_col],errors='coerce')
            td=td.dropna(subset=[tc_col])
            if did_post_var and did_post_var in td.columns:
                td['_post']=td[did_post_var].astype(float)
            else:
                cutoff = did_policy_time if did_policy_time is not None else td[tc_col].median()
                td['_post']=(td[tc_col]>=cutoff).astype(float)
            td['_treat']=td[did_var].astype(float)
            td['_did']=td['_treat']*td['_post']
            # PSM matching on pre-treatment covariates
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression
            from sklearn.neighbors import NearestNeighbors as NN
            did_terms=['_treat','_post','_did']
            cov_cols=[v for v in Xv0 if v in td.columns and v not in did_terms]
            if not cov_cols:
                return {'success':False,'error':'PSM-DID 至少需要 1 个匹配变量/控制变量','n':n_raw}
            pre_mask=td['_post']==0
            if pre_mask.sum()<10: return {'success':False,'error':'PSM-DID 预处理期样本不足','n':n_raw}
            pre_data=td[pre_mask][cov_cols+[did_var]].dropna()
            if pre_data[did_var].nunique()<2:
                return {'success':False,'error':'政策前样本中处理组变量没有两类','n':len(pre_data)}
            X_sc=StandardScaler().fit_transform(pre_data[cov_cols])
            D_p=pre_data[did_var].values
            ps=LogisticRegression(C=1e6,max_iter=1000).fit(X_sc,D_p).predict_proba(X_sc)[:,1]
            td=td.set_index([id_col,tc_col])
            n=len(td)
            if n<30: return {'success':False,'error':'样本量<30','n':n}
            controls=[v for v in Xv0 if v in td.columns and v not in did_terms]
            did_X=unique_keep_order(controls+['_post','_did'])
            did_X=[v for v in did_X if td[v].nunique(dropna=True)>1]
            if '_did' not in did_X: return {'success':False,'error':'PSM-DID 交互项没有变化，无法估计','n':n}
            X_did=sm.add_constant(td[did_X])
            d_r=safe_panel_fit(td[y_col],X_did,entity_effects=True,time_effects=False)
            b=d_r.params.get('_did',np.nan); se=d_r.std_errors.get('_did',np.nan)
            t=abs(b/se) if se>0 else 0
            df_eff=max(n-len(did_X)-td.index.get_level_values(0).nunique()-1,1)
            return {'success':True,'coef':float(b),'se':float(se),'tstat':float(t),
                'pval':float(2*(1-stats.t.cdf(t,df=df_eff))),'n':n,'rsq':float(d_r.rsquared)}

        # ── PSM (倾向得分匹配) ──
        if 'PSM' in model_sel:
            if did_var is None: return {'success':False,'error':'PSM 需要处理变量','n':n_raw}
            if not [v for v in Xv0 if v in data.columns]:
                return {'success':False,'error':'PSM 至少需要 1 个匹配变量','n':n_raw}
            need=[y_col]+[v for v in Xv0 if v in data.columns]+[did_var]
            td=data[need].dropna()
            n=len(td)
            if n<30: return {'success':False,'error':'样本量<30','n':n}
            if td[did_var].nunique()<2: return {'success':False,'error':'处理变量没有两类','n':n}
            D=td[did_var].values; y_p=td[y_col].values
            X_p=td[[v for v in Xv0 if v in td.columns]].values
            pr=psm_att(y_p,D,X_p,method='nearest',k=1)
            return {'success':True,'coef':float(pr['ATT']),'se':np.nan,'tstat':np.nan,
                'pval':np.nan,'n':n,'rsq':np.nan}

        # ── IV/2SLS ──
        if 'IV' in model_sel:
            if iv_endog is None or not iv_insts: return {'success':False,'error':'IV 需要指定内生变量和工具变量','n':n_raw}
            exog_vars=[v for v in Xv0 if v!=iv_endog]
            need=[y_col,iv_endog]+exog_vars+[v for v in iv_insts if v in data.columns]
            td=data[need].dropna()
            n=len(td)
            if n<30: return {'success':False,'error':'样本量<30','n':n}
            iv_insts_avail=[v for v in iv_insts if v in data.columns]
            iv_m=IV2SLS(td[y_col],sm.add_constant(td[exog_vars]),td[iv_endog],sm.add_constant(td[iv_insts_avail])).fit()
            # 报告内生变量的系数
            b=iv_m.params.get(iv_endog,np.nan)
            if np.isnan(b): b=iv_m.params.get(focus_cx,np.nan)
            se=iv_m.std_errors.get(iv_endog if not np.isnan(iv_m.params.get(iv_endog,np.nan)) else focus_cx,np.nan)
            t=abs(b/se) if se>0 else 0
            return {'success':True,'coef':float(b),'se':float(se),'tstat':float(t),
                'pval':float(2*(1-stats.t.cdf(t,df=max(n-1,1)))),'n':n,'rsq':np.nan}

        # ── Heckman 两步法 ──
        if 'Heckman' in model_sel:
            if heckman_sel is None: return {'success':False,'error':'Heckman 需要选择变量','n':n_raw}
            need=[y_col]+[v for v in Xv0 if v in data.columns]+[heckman_sel]
            td=data[need].dropna()
            n=len(td)
            if n<30: return {'success':False,'error':'样本量<30','n':n}
            Xv=[v for v in Xv0 if v in td.columns]
            X_h=td[Xv].values
            y_h=td[y_col].values
            Z_vars=td[Xv].values
            hr=heckman_two_step(y_h,X_h,Z_vars)
            pi=Xv.index(focus_cx)+1 if focus_cx in Xv else -1  # +1 skip const
            b=float(hr['params'][pi]) if pi>=0 else np.nan; se=float(hr['se'][pi]) if pi>=0 else np.nan
            t=abs(b/se) if se>0 else 0
            return {'success':True,'coef':float(b),'se':float(se),'tstat':float(t),
                'pval':float(2*(1-stats.t.cdf(t,df=max(n-len(Xv)-1,1)))),
                'n':int(hr['n_obs']),'rsq':np.nan}

        # ── 不支持的模型 ──
        return {'success':False,'error':f'模型 {model_sel} 暂不支持此分析','n':n_raw}

    except Exception as e:
        return {'success':False,'error':str(e)[:100],'n':0,'coef':np.nan,'se':np.nan,'tstat':np.nan,'pval':np.nan,'rsq':np.nan}

# ═══════════════════ STEP 1: 上传 ═══════════════════
st.header("Step 1 · 上传数据")
uploaded = st.file_uploader("拖拽 .dta / .csv / .xlsx", type=['dta','csv','xlsx'])
if uploaded is not None:
    if st.session_state.df_name != uploaded.name:
        try:
            if uploaded.name.endswith('.dta'): raw_df=pd.read_stata(uploaded)
            elif uploaded.name.endswith('.csv'): raw_df=pd.read_csv(uploaded)
            else: raw_df=pd.read_excel(uploaded)
            st.session_state.raw_df=raw_df.copy()
            st.session_state.df=raw_df.copy()
            st.session_state.df_name=uploaded.name; st.session_state.data_type=None
            st.session_state.diagnosed=False; st.session_state.search_results=None
            st.session_state.id_col=None; st.session_state.time_col=None
            st.session_state.cleaning_log=[]
            st.session_state.cleaning_rules=[]
            st.session_state.transform_log=[]
        except Exception as e: st.error(f"读取失败：{e}"); st.stop()

if st.session_state.df is not None:
    df=st.session_state.df
    st.success(f"当前：{st.session_state.df_name} ｜ {df.shape[0]}行 × {df.shape[1]}列")
    with st.expander("预览"): st.dataframe(df.head(6),use_container_width=True)


    # ═══════════════════ STEP 1.5: 数据清洗（可选） ═══════════════════
    st.header("Step 1.5 · 数据清洗（可选）")
    st.caption("先清洗，再进入实证。默认不处理；用户明确选择后才应用。")
    if st.session_state.raw_df is None:
        st.session_state.raw_df=df.copy()
    with st.expander("打开数据清洗面板", expanded=False):
        raw_df=st.session_state.raw_df.copy()
        clean_cols=raw_df.columns.tolist()
        miss_total=int(raw_df.isna().sum().sum())
        dup_total=int(raw_df.duplicated().sum())
        st.caption(f"原始数据：{raw_df.shape[0]}行 × {raw_df.shape[1]}列 ｜ 缺失值 {miss_total} 个 ｜ 完全重复行 {dup_total} 行")
        numeric_like_cols=[]
        for c in clean_cols:
            if pd.api.types.is_numeric_dtype(raw_df[c]):
                numeric_like_cols.append(c)
            elif raw_df[c].dtype=='object':
                conv=pd.to_numeric(raw_df[c],errors='coerce')
                if raw_df[c].notna().sum()>0 and conv.notna().sum()/raw_df[c].notna().sum()>=0.5:
                    numeric_like_cols.append(c)

        rule_method=st.selectbox("选择清洗方式", [
            '删除完全重复行',
            '尝试把数字文本转为数值',
            '删除所选变量含缺失的行',
            '均值插补',
            '中位数插补',
            '线性插值（按当前排序）',
            '1%/99%缩尾',
            '5%/95%缩尾'
        ], key='clean_rule_method_v2')
        no_var_methods=['删除完全重复行']
        numeric_methods=['均值插补','中位数插补','线性插值（按当前排序）','1%/99%缩尾','5%/95%缩尾']
        if rule_method in no_var_methods:
            rule_cols=[]
            st.caption("该规则作用于整张表，不需要选择变量。")
        else:
            opts=numeric_like_cols if rule_method in numeric_methods else clean_cols
            default_opts=[]
            if rule_method in ['均值插补','中位数插补','线性插值（按当前排序）']:
                default_opts=[c for c in opts if raw_df[c].isna().any()][:6]
            rule_cols=st.multiselect("选择变量", opts, default=default_opts, key='clean_rule_cols_v2')

        if st.button("加入清洗规则", key='add_clean_rule_v2'):
            if rule_method not in no_var_methods and not rule_cols:
                st.warning("请先选择变量")
            else:
                st.session_state.cleaning_rules.append({'method':rule_method,'cols':list(rule_cols)})
                st.success("已加入规则")

        rules=st.session_state.cleaning_rules
        if rules:
            st.dataframe(pd.DataFrame([
                {'顺序':i+1,'清洗方式':r['method'],'变量':'、'.join(r.get('cols',[])) if r.get('cols') else '整张表'}
                for i,r in enumerate(rules)
            ]),use_container_width=True,hide_index=True)
        else:
            st.info("尚未添加清洗规则；不添加则直接使用原始数据进入实证。")

        ac1,ac2,ac3=st.columns(3)
        with ac1:
            apply_clean=st.button("应用全部规则", type='primary', key='apply_clean_v2')
        with ac2:
            clear_rules=st.button("清空规则", key='clear_clean_rules_v2')
        with ac3:
            reset_clean=st.button("恢复原始数据", key='reset_clean_v2')

        if clear_rules:
            st.session_state.cleaning_rules=[]
            st.session_state.cleaning_log=[]
            st.success("已清空清洗规则")
        if apply_clean:
            work=raw_df.copy(); logs=[]
            before_shape=work.shape
            for r in st.session_state.cleaning_rules:
                method=r.get('method'); rcols=[c for c in r.get('cols',[]) if c in work.columns]
                if method=='删除完全重复行':
                    n0=len(work); work=work.drop_duplicates(); logs.append(f'删除完全重复行 {n0-len(work)} 行')
                elif method=='尝试把数字文本转为数值':
                    converted=[]
                    for c in rcols:
                        if work[c].dtype=='object':
                            conv=pd.to_numeric(work[c],errors='coerce')
                            base=max(int(work[c].notna().sum()),1)
                            if conv.notna().sum()/base>=0.8:
                                work[c]=conv; converted.append(c)
                    logs.append('数字文本转数值：'+(', '.join(converted) if converted else '无可转换变量'))
                elif method=='删除所选变量含缺失的行':
                    n0=len(work); work=work.dropna(subset=rcols); logs.append(f'按 {len(rcols)} 个变量删除缺失行 {n0-len(work)} 行')
                elif method in ['均值插补','中位数插补']:
                    filled=[]
                    for c in rcols:
                        series=pd.to_numeric(work[c],errors='coerce')
                        if series.isna().any():
                            val=series.mean() if method=='均值插补' else series.median()
                            work[c]=series.fillna(val); filled.append(c)
                    logs.append(f'{method}：'+(', '.join(filled) if filled else '无缺失变量'))
                elif method=='线性插值（按当前排序）':
                    filled=[]
                    for c in rcols:
                        series=pd.to_numeric(work[c],errors='coerce')
                        if series.isna().any():
                            work[c]=series.interpolate(limit_direction='both'); filled.append(c)
                    logs.append('线性插值：'+(', '.join(filled) if filled else '无缺失变量'))
                elif method in ['1%/99%缩尾','5%/95%缩尾']:
                    pct=0.01 if method.startswith('1%') else 0.05
                    done=[]
                    for c in rcols:
                        series=pd.to_numeric(work[c],errors='coerce')
                        lo,hi=series.quantile(pct),series.quantile(1-pct)
                        work[c]=series.clip(lo,hi); done.append(c)
                    logs.append(f'{method}：'+(', '.join(done) if done else '无可处理变量'))
            st.session_state.df=work
            st.session_state.diagnosed=False; st.session_state.search_results=None
            st.session_state.transform_log=[]
            st.session_state.cleaning_log=logs if logs else ['未执行实际清洗操作']
            st.success(f"清洗完成：{before_shape[0]}行 × {before_shape[1]}列 → {work.shape[0]}行 × {work.shape[1]}列")
        if reset_clean:
            st.session_state.df=raw_df.copy()
            st.session_state.diagnosed=False; st.session_state.search_results=None
            st.session_state.cleaning_log=[]
            st.session_state.cleaning_rules=[]
            st.session_state.transform_log=[]
            st.success("已恢复为上传时的原始数据")
        if st.session_state.cleaning_log:
            st.info('当前清洗记录：'+'；'.join(st.session_state.cleaning_log))
    df=st.session_state.df

    # ═══════════════════ STEP 2: 数据结构 ═══════════════════
    st.header("Step 2 · 数据结构诊断")
    cols=df.columns.tolist()
    id_c=[]; time_c=[]
    for c in cols:
        nu=df[c].nunique()
        if 2<nu<df.shape[0]:
            vc=df[c].value_counts()
            if vc.std()/vc.mean()<0.8: id_c.append(c)
        if pd.api.types.is_numeric_dtype(df[c]) and 1990<=df[c].min()<=2030 and df[c].max()<=2030 and df[c].nunique()>=2:
            time_c.append(c)
    if not id_c: id_c=[c for c in cols if 2<df[c].nunique()<df.shape[0]/2]
    guess_panel=bool(id_c and time_c)
    guess_ts=not guess_panel and len(cols)>=2 and any('date' in c.lower() or 'year' in c.lower() or 'time' in c.lower() for c in cols)

    dt_options=["面板数据（ID+时间）","横截面数据","时间序列数据"]
    dt_default=0 if guess_panel else (2 if guess_ts else 1)
    data_type=st.radio("数据结构",dt_options,index=dt_default,key='dt7',horizontal=True)

    if data_type.startswith("面板"):
        gid=id_c[0] if id_c else None; gtime=time_c[0] if time_c else None
        c1,c2=st.columns(2)
        with c1: id_sel=st.selectbox("个体 ID",cols,index=cols.index(gid) if gid in cols else 0,key='id7')
        with c2: time_sel=st.selectbox("时间",cols,index=cols.index(gtime) if gtime in cols else 0,key='t7')
        if st.button("确认诊断",type="primary",key='diag7'):
            g=df.groupby(id_sel)
            st.session_state.id_col=id_sel; st.session_state.time_col=time_sel
            st.session_state.n_units=g.ngroups; st.session_state.n_periods=df[time_sel].nunique()
            st.session_state.balanced=g.size().nunique()==1
            st.session_state.data_type='panel'; st.session_state.diagnosed=True; st.session_state.search_results=None
        if st.session_state.diagnosed and st.session_state.data_type=='panel':
            b=st.session_state.balanced
            st.success(f"{'平衡' if b else '非平衡'}面板 ｜ N={st.session_state.n_units} × T={st.session_state.n_periods}")
    elif data_type.startswith("时间"):
        c1,c2=st.columns(2)
        with c1: ts_time=st.selectbox("时间列",cols,key='tst7')
        with c2: ts_target=st.selectbox("分析目标列",[c for c in cols if c!=ts_time and pd.api.types.is_numeric_dtype(df[c])],key='tsg7')
        if st.button("确认诊断",type="primary",key='diag7ts'):
            st.session_state.data_type='timeseries'; st.session_state.diagnosed=True
            st.session_state.search_results=None; st.session_state._ts_time=ts_time; st.session_state._ts_target=ts_target
        if st.session_state.diagnosed and st.session_state.data_type=='timeseries':
            series=df[ts_target].dropna()
            adf_s,adf_p,is_stat=check_stationarity(series)
            status="平稳 ✓" if is_stat else "非平稳（建议差分）"
            st.success(f"时间序列：{len(series)} 观测 ｜ ADF={adf_s:.2f} p={adf_p:.4f} → {status}")
    else:
        if st.button("确认诊断（横截面）",type="primary",key='diag7cs'):
            st.session_state.data_type='cross'; st.session_state.diagnosed=True; st.session_state.search_results=None
        if st.session_state.diagnosed and st.session_state.data_type=='cross':
            n_num=sum(1 for c in cols if pd.api.types.is_numeric_dtype(df[c]))
            st.success(f"横截面数据 ｜ {df.shape[0]}观测 × {df.shape[1]}列（{n_num} 数值）")

    # ═══════════════════ STEP 3: 变量 & 搜索 ═══════════════════
    if st.session_state.diagnosed:
        dt=st.session_state.data_type
        st.header("Step 2.5 · 生成新变量（可选）")
        st.caption("生成的新变量会保留原变量，不覆盖原数据；后续回归可以直接选择这些新变量。")
        num_cols_now=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        with st.expander("打开新变量生成", expanded=False):
            trans_method=st.selectbox("生成方式", [
                "取对数 ln(x)",
                "平方项 x²",
                "相乘 / 交互项 x1×x2",
                "标准化 z-score",
                "滞后一期 L1",
                "一阶差分 D1"
            ], key='trans_method_v1')
            can_panel_time=dt in ['panel','timeseries']
            if trans_method in ["取对数 ln(x)","平方项 x²","标准化 z-score"]:
                trans_cols=st.multiselect("选择变量", num_cols_now, key='trans_cols_v1')
            elif trans_method=="相乘 / 交互项 x1×x2":
                cta,ctb=st.columns(2)
                with cta:
                    trans_a=st.selectbox("变量 1", num_cols_now, key='trans_a_v1') if num_cols_now else None
                with ctb:
                    trans_b=st.selectbox("变量 2", num_cols_now, key='trans_b_v1') if num_cols_now else None
                default_name=f"{trans_a}_x_{trans_b}" if trans_a and trans_b else ""
                custom_name=st.text_input("新变量名", value=default_name, key='trans_inter_name_v1')
            else:
                trans_cols=st.multiselect("选择变量", num_cols_now, key='trans_panel_cols_v1')
                if not can_panel_time:
                    st.warning("滞后和差分需要先诊断为面板数据或时间序列数据。")

            add_trans=st.button("生成变量", type='primary', key='add_transform_v1')
            clear_trans=st.button("清空生成记录", key='clear_transform_log_v1')
            if clear_trans:
                st.session_state.transform_log=[]
                st.success("已清空生成记录；已生成的变量仍保留在当前数据中。")
            if add_trans:
                work=df.copy(); logs=[]; existing=set(work.columns)
                try:
                    if trans_method=="取对数 ln(x)":
                        if not trans_cols:
                            st.warning("请选择变量")
                        else:
                            for c in trans_cols:
                                new_c=make_unique_col(f"ln_{c}", existing)
                                vals=pd.to_numeric(work[c],errors='coerce')
                                work[new_c]=np.where(vals>0,np.log(vals),np.nan)
                                existing.add(new_c); logs.append(f"{new_c}=ln({c})")
                    elif trans_method=="平方项 x²":
                        if not trans_cols:
                            st.warning("请选择变量")
                        else:
                            for c in trans_cols:
                                new_c=make_unique_col(f"{c}_sq", existing)
                                vals=pd.to_numeric(work[c],errors='coerce')
                                work[new_c]=vals**2
                                existing.add(new_c); logs.append(f"{new_c}={c}²")
                    elif trans_method=="标准化 z-score":
                        if not trans_cols:
                            st.warning("请选择变量")
                        else:
                            for c in trans_cols:
                                vals=pd.to_numeric(work[c],errors='coerce')
                                sd=vals.std()
                                if sd and not np.isnan(sd):
                                    new_c=make_unique_col(f"z_{c}", existing)
                                    work[new_c]=(vals-vals.mean())/sd
                                    existing.add(new_c); logs.append(f"{new_c}=z({c})")
                    elif trans_method=="相乘 / 交互项 x1×x2":
                        if not trans_a or not trans_b:
                            st.warning("请选择两个变量")
                        else:
                            new_c=make_unique_col(custom_name or f"{trans_a}_x_{trans_b}", existing)
                            work[new_c]=pd.to_numeric(work[trans_a],errors='coerce')*pd.to_numeric(work[trans_b],errors='coerce')
                            logs.append(f"{new_c}={trans_a}×{trans_b}")
                    elif trans_method in ["滞后一期 L1","一阶差分 D1"]:
                        if not can_panel_time:
                            st.warning("请先完成面板或时间序列诊断")
                        elif not trans_cols:
                            st.warning("请选择变量")
                        else:
                            if dt=='panel':
                                id_col=st.session_state.id_col; tc_col=st.session_state.time_col
                                work=work.sort_values([id_col,tc_col]).copy()
                                for c in trans_cols:
                                    if trans_method=="滞后一期 L1":
                                        new_c=make_unique_col(f"L1_{c}", existing)
                                        work[new_c]=work.groupby(id_col)[c].shift(1)
                                        logs.append(f"{new_c}=L1.{c}")
                                    else:
                                        new_c=make_unique_col(f"D1_{c}", existing)
                                        work[new_c]=work.groupby(id_col)[c].diff(1)
                                        logs.append(f"{new_c}=D1.{c}")
                                    existing.add(new_c)
                            else:
                                ts_time=st.session_state.get('_ts_time')
                                work=work.sort_values(ts_time).copy() if ts_time in work.columns else work.copy()
                                for c in trans_cols:
                                    if trans_method=="滞后一期 L1":
                                        new_c=make_unique_col(f"L1_{c}", existing)
                                        work[new_c]=work[c].shift(1)
                                        logs.append(f"{new_c}=L1.{c}")
                                    else:
                                        new_c=make_unique_col(f"D1_{c}", existing)
                                        work[new_c]=work[c].diff(1)
                                        logs.append(f"{new_c}=D1.{c}")
                                    existing.add(new_c)
                    if logs:
                        st.session_state.df=work
                        st.session_state.search_results=None
                        st.session_state.transform_log=st.session_state.transform_log+logs
                        st.success("已生成："+'；'.join(logs[:6])+(' 等' if len(logs)>6 else ''))
                        df=st.session_state.df
                        cols=df.columns.tolist()
                except Exception as e:
                    st.error(f"生成失败：{e}")
            if st.session_state.transform_log:
                st.info("已生成变量："+"；".join(st.session_state.transform_log[-12:]))

        st.header(f"Step 3 · 变量分配 & 搜索")

        if dt=='timeseries':
            # ── 时间序列路径 ──
            ts_t=st.session_state._ts_time; ts_y=st.session_state._ts_target
            st.caption(f"时间序列模型：{ts_y} ~ f(t)")
            other_cols=[c for c in cols if c not in [ts_t,ts_y] and pd.api.types.is_numeric_dtype(df[c])]
            ts_x=st.multiselect("外生变量 X（可选，VAR/ARDL 使用）",other_cols,key='tsx7')
            ts_model=st.radio("模型",['ARIMA','AR','MA','ARMA','VAR（多变量）','GARCH(1,1)'],horizontal=True,key='tsm7')
            if ts_model.startswith('VAR') and len(ts_x)<1: st.warning("VAR 需要至少1个其他变量")
            if st.button("估计模型",type="primary",key='tsrun7'):
                y_s=df[ts_y].dropna()
                with st.spinner("估计中..."):
                    try:
                        if ts_model in ['ARIMA','AR','MA','ARMA']:
                            order=(1,0,0)
                            if ts_model=='ARIMA': order=(1,1,1)
                            elif ts_model=='AR': order=(1,0,0)
                            elif ts_model=='MA': order=(0,0,1)
                            elif ts_model=='ARMA': order=(1,0,1)
                            m=sm.tsa.arima.ARIMA(y_s,order=order,exog=df[ts_x].dropna().reindex(y_s.index) if ts_x else None).fit()
                            st.success(f"AIC={m.aic:.2f}, BIC={m.bic:.2f}")
                            rows=[{'变量':v,'系数':f"{b:.4f}",'SE':f"{s:.4f}"} for v,b,s in zip(m.model.exog_names,m.params,m.bse)]
                            st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
                            do=f"* === 鹈鹕回归 (c)2026 · 仅供参考 ===\n* ARIMA\nuse \"data.dta\", clear\ntsset {ts_t}\narima {ts_y}"
                            if ts_x: do+=f" {' '.join(ts_x)}"
                            do+=f", arima(1,1,1)\n"
                        elif ts_model.startswith('VAR'):
                            var_cols=[ts_y]+ts_x; var_data=df[var_cols].dropna()
                            m=sm.tsa.VAR(var_data).fit(maxlags=2,ic='aic')
                            st.success(f"AIC={m.aic:.2f}, BIC={m.bic:.2f}")
                            do=f"* === 鹈鹕回归 (c)2026 · 仅供参考 ===\n* VAR\nuse \"data.dta\", clear\ntsset {ts_t}\nvar {' '.join(var_cols)}, lags(1/2)\n"
                            st.caption("VAR 系数较多，请查看 Stata 代码在本地运行")
                        else:  # GARCH
                            try:
                                from arch import arch_model
                                am=arch_model(y_s,vol='Garch',p=1,q=1).fit(disp='off')
                                st.success(f"AIC={am.aic:.2f}")
                                do=f"* === 鹈鹕回归 (c)2026 · 仅供参考 ===\n* GARCH(1,1)\nuse \"data.dta\", clear\ntsset {ts_t}\narch {ts_y}, arch(1) garch(1)\n"
                            except ImportError:
                                st.error("需要安装 arch 包: pip install arch")
                                do=""
                        if do:
                            with st.expander("Stata 复现代码"): st.code(do,language='stata')
                    except Exception as e: st.error(f"估计失败：{e}")
            st.stop()

        # ── 面板/横截面：通用变量选择 ──
        if dt=='panel':
            id_col,tc=st.session_state.id_col,st.session_state.time_col
            excl=[id_col,tc]
        else:
            excl=[]
        # 数值列
        all_num=[c for c in cols if c not in excl and pd.api.types.is_numeric_dtype(df[c])]
        # 分类列（object 或低基数数值列，2-15 个唯一值）
        cat_cols=[c for c in cols if c not in excl and c not in all_num
                  and (df[c].dtype=='object' or (pd.api.types.is_numeric_dtype(df[c]) and 2<=df[c].nunique()<=15))]
        # 生成虚拟变量映射
        dummy_map={}  # {orig_col: [dummy_name1, dummy_name2, ...]}
        dummy_names=[]  # flat list of all dummy column names
        df_aug=df.copy()
        for cc in cat_cols:
            dums=pd.get_dummies(df[cc],prefix=cc,drop_first=True,dtype=float)
            d_names=list(dums.columns)
            dummy_map[cc]=d_names; dummy_names.extend(d_names)
            for dn in d_names: df_aug[dn]=dums[dn].values
        # 合并可选变量：数值列 + 虚拟列
        all_vars=[c for c in all_num if c not in excl] + dummy_names
        if len(all_vars)<2: st.warning("变量不足（需 >=2 个数值/分类列）"); st.stop()
        if dummy_names:
            st.caption(f"已自动将分类列转为虚拟变量（drop first）：{', '.join(dummy_map.keys())}")

        y_col=st.selectbox("被解释变量 Y",[c for c in all_num if c not in excl],key='y7')
        y_type,y_info=detect_y_type(df[y_col])

        # 模型选择
        model_options=[]
        if dt=='panel':
            if y_type=='binary': model_options=['Logit','Probit','Logit+Probit']
            else: model_options=['FE+RE（固定+随机效应）','Pooled OLS（忽略面板结构）']
            rem_p=[c for c in all_vars if c!=y_col]
            all_bin=[c for c in rem_p if len(df_aug[c].dropna().unique())==2]
            bin_vars=[c for c in all_bin if c in all_num]  # 原始数值二分变量
            if all_bin:
                model_options.append('DID 双重差分（面板+政策冲击）')
            if all_bin: model_options.append('PSM-DID')
            if cat_cols and y_type=='continuous':
                model_options.append('分组面板回归（按分类变量）')
        else:
            if y_type=='binary': model_options=['Logit','Probit','Logit+Probit','LPM（线性概率模型）']
            elif y_type=='ordered': model_options=['Ordered Logit','Ordered Probit']
            elif y_type=='count': model_options=['Poisson','负二项（Negative Binomial）']
            elif y_type=='censored': model_options=['OLS+稳健SE','Tobit（截断回归）']
            else:
                model_options=['OLS','OLS+稳健SE','Tobit（如被截断）','Heckman（样本选择）']
                model_options.append('RDD（断点回归）')
            model_options.append('PSM（倾向得分匹配）')
            model_options.append('IV/2SLS（工具变量）')
            if cat_cols:
                if y_type in ['continuous','censored']:
                    model_options.append('ANOVA/组间比较（分类变量）')
                if y_type in ['continuous','binary','count']:
                    model_options.append('交互效应（分类×连续）')

        sel_model=st.radio("模型",model_options,horizontal=False,key='mdl7')

        # 分类变量特有：选择分组变量
        group_var=None
        if 'ANOVA' in sel_model or '交互效应' in sel_model or '分组面板' in sel_model:
            group_var=st.selectbox("分组变量（分类列）",cat_cols,key='grp7')
            if '交互效应' in sel_model:
                st.caption("将生成 分组变量 × 核心X 的交互项，检验斜率是否因组而异")

        # DID / PSM-DID：核心解释项固定为 Treat×Post，用户只需指定处理组和政策后
        use_cl=False; cluster_col=None; use_rob=False; se_mode='ordinary'
        bin_m=None; did_var=None; heckman_sel=None
        iv_endog=None; iv_insts=[]
        did_policy_time=None; did_post_var=None; did_post_mode='按政策时间生成 Post'
        is_did_like=(dt=='panel' and 'DID' in sel_model)
        if is_did_like:
            st.subheader("DID 设定")
            did_var=st.selectbox("处理组变量（二分，1=处理组）",bin_vars,key='did7')
            did_post_mode=st.radio("Post 变量来源", ['按政策时间生成 Post','使用已有 Post 变量'], horizontal=True, key='did_post_mode_v1')
            if did_post_mode.startswith('按政策'):
                time_vals=sorted(pd.Series(df[tc]).dropna().unique().tolist())
                default_idx=len(time_vals)//2 if time_vals else 0
                did_policy_time=st.selectbox("政策发生时间（该时间及以后 Post=1）", time_vals, index=default_idx, key='did_policy_time_v1') if time_vals else None
                if did_policy_time is not None:
                    df_aug['_post']=(pd.to_numeric(df_aug[tc],errors='coerce')>=did_policy_time).astype(float)
                st.caption(f"核心解释项固定为 Treat×Post：Treat={did_var}，Post=({tc} >= {did_policy_time})")
            else:
                post_candidates=[c for c in bin_vars if c!=did_var]
                did_post_var=st.selectbox("已有 Post 变量（二分，1=政策后）", post_candidates, key='did_post_var_v1') if post_candidates else None
                if did_post_var is None:
                    st.warning("没有可用的二分 Post 变量，请改用政策时间生成 Post")
                    df_aug['_post']=0.0
                else:
                    df_aug['_post']=df_aug[did_post_var].astype(float)
                st.caption(f"核心解释项固定为 Treat×Post：Treat={did_var}，Post={did_post_var}")
            if did_var:
                df_aug['_treat']=df_aug[did_var].astype(float)
                df_aug['_did']=df_aug['_treat']*df_aug['_post']
            if 'PSM-DID' in sel_model:
                st.caption("PSM-DID 会使用政策前控制变量做匹配，再围绕 Treat×Post 输出 DID 结果。")

        # 核心X + 控制变量（含虚拟变量）
        rem=[c for c in all_vars if c!=y_col]
        fixed_controls=[]; rdd_running=None; rdd_cutoff=None; rdd_bandwidth=0.0; rdd_order='局部线性（一阶）'
        rdd_design='Sharp RD（精确断点）'; rdd_treatment=None
        if is_did_like:
            core_x=['_did']
            fixed_controls=['_treat','_post']
            pool_opt=[c for c in rem if c not in [did_var,did_post_var]]
            st.caption("DID/PSM-DID 不需要再选择核心 X；搜索目标自动固定为 Treat×Post，下面只选择控制变量候选池。")
        elif 'RDD' in sel_model:
            rdd_candidates=[c for c in all_num if c not in excl and c!=y_col]
            rdd_running=st.selectbox("断点变量 / Running variable", rdd_candidates, key='rdd_run_v1') if rdd_candidates else None
            if not rdd_running:
                st.warning("RDD 的最低推荐条件是连续型 Y；进入模型后还需要选择一个数值型断点变量。当前数据没有可选断点变量。"); st.stop()
            rdd_design=st.radio("RD 类型", ['Sharp RD（精确断点）','Fuzzy RD（模糊断点 / 工具变量）'], horizontal=True, key='rdd_design_v2')
            rv=df_aug[rdd_running].dropna()
            default_cut=float(rv.median()) if len(rv) else 0.0
            c_rdd1,c_rdd2,c_rdd3=st.columns(3)
            with c_rdd1: rdd_cutoff=st.number_input("断点值 cutoff", value=default_cut, key='rdd_cut_v1')
            with c_rdd2: rdd_bandwidth=st.number_input("带宽（0=不限制）", min_value=0.0, value=0.0, key='rdd_bw_v1')
            with c_rdd3: rdd_order=st.radio("阶数", ['局部线性（一阶）','二阶多项式'], horizontal=False, key='rdd_order_v1')
            df_aug['_rdd_z']=(df_aug[rdd_running]>=rdd_cutoff).astype(float)
            df_aug['_rdd_treat']=df_aug['_rdd_z']
            df_aug['_rdd_running_c']=df_aug[rdd_running]-rdd_cutoff
            df_aug['_rdd_inter']=df_aug['_rdd_z']*df_aug['_rdd_running_c']
            fixed_controls=['_rdd_running_c','_rdd_inter']
            if '二阶' in rdd_order:
                df_aug['_rdd_running_c2']=df_aug['_rdd_running_c']**2
                df_aug['_rdd_inter2']=df_aug['_rdd_z']*df_aug['_rdd_running_c2']
                fixed_controls+=['_rdd_running_c2','_rdd_inter2']
            if rdd_design.startswith('Fuzzy'):
                treat_candidates=[c for c in all_num if c not in excl and c not in [y_col,rdd_running]]
                if not treat_candidates:
                    st.warning("Fuzzy RD 需要一个实际处理变量 D"); st.stop()
                rdd_treatment=st.selectbox("实际处理变量 D（断点只改变接受处理的概率）", treat_candidates, key='rdd_d_v2')
                core_x=[rdd_treatment]
                pool_opt=[c for c in rem if c not in [rdd_running,rdd_treatment]]
                st.caption("Fuzzy RD：用 Z=1(running≥cutoff) 作为工具变量估计 D 对 Y 的局部平均处理效应（LATE）。")
            else:
                core_x=['_rdd_treat']
                pool_opt=[c for c in rem if c!=rdd_running]
                st.caption("Sharp RD：断点直接决定处理状态；估计断点处结果变量的跳跃。")
        else:
            core_x=st.multiselect("核心 X（最多 8 个）",rem,default=rem[:1] if rem else [],max_selections=8,key='cx7')
            pool_opt=[c for c in rem if c not in core_x]
        ctrl_pool=st.multiselect(f"控制变量候选池（最多 20，可选 {len(pool_opt)} 个）",pool_opt,
            default=pool_opt[:min(10,len(pool_opt))],max_selections=20,key='pool7')
        c3,c4=st.columns(2)
        default_cmin=0 if ('RDD' in sel_model or is_did_like) else 2
        with c3: cmin=st.number_input("最少控制数",0,15,default_cmin,key='cmin7')
        default_cmax=min(5,len(ctrl_pool)) if ctrl_pool else 0
        with c4: cmax=st.number_input("最多控制数",0,15,default_cmax,key='cmax7')
        if cmax<cmin: cmax=cmin

        if is_did_like:
            search_mode='分别显著（各X独立搜索）'
            st.caption("显著性排序固定查看 Treat×Post；无需选择“分别/同时显著”。")
        else:
            search_mode=st.radio("核心X显著要求",['分别显著（各X独立搜索）','同时显著（所有X联合搜索）'],
                horizontal=True,key='smode7',
                help="分别：为每个核心X找各自的最优控制组合｜同时：控制组合必须让所有核心X都显著")

        # 模型特有选项
        if dt!='panel':
            if 'PSM' in sel_model:
                did_var=st.selectbox("处理变量（二分）",[c for c in rem if len(df_aug[c].dropna().unique())==2],key='psm_t7')
            if 'IV' in sel_model:
                iv_endog=st.selectbox("内生变量",rem[:10],key='ive7')
                iv_insts=st.multiselect("工具变量（至少1个）",[c for c in rem if c!=iv_endog],key='ivi7',max_selections=5)
            if 'Heckman' in sel_model:
                heckman_sel=st.selectbox("选择变量（二分，1=被观测到）",[c for c in rem if len(df_aug[c].dropna().unique())==2],key='hks7')

        # ── 附加分析：中介效应 / 调节效应（可与任何基准模型叠加）──
        add_med=False; add_mod=False
        med_m=None; mod_m=None; med_method='温忠麟五步流程（2014）'
        if not any(sk in sel_model for sk in ['DID','PSM-DID','ANOVA','交互效应','分组面板']):
            c1_extra,c2_extra=st.columns(2)
            with c1_extra: add_med=st.checkbox("添加中介效应分析",key='addmed7',
                help="在基准回归显著组合上，叠加中介效应检验（温忠麟五步流程或江艇渠道检验）")
            with c2_extra: add_mod=st.checkbox("添加调节效应分析",key='addmod7',
                help="在基准回归显著组合上，叠加调节效应：X×M 交互项 + 简单斜率分析")
        if add_med:
            med_opts=[c for c in all_num if c not in excl and c!=y_col and c not in core_x]
            med_m=st.selectbox("中介变量 M",med_opts,key='medm7') if med_opts else None
            med_method=st.radio("中介方法",['温忠麟五步流程（2014）','江艇渠道检验（2022）'],horizontal=True,key='medmethod7')
            if not med_opts: st.warning("无可用的中介变量")
        if add_mod:
            mod_opts=[c for c in all_num if c not in excl and c!=y_col and c not in core_x]
            mod_m=st.selectbox("调节变量 M",mod_opts,key='modm7') if mod_opts else None
            if not mod_opts: st.warning("无可用的调节变量")

        # ── 共享 SE 选择器（适用大多数模型）──
        se_skip_models=['Tobit','Ordered','ANOVA','分组面板','交互效应','PSM','IV','Heckman','OLS+稳健SE','PSM-DID']
        se_show=not any(sk in sel_model for sk in se_skip_models)
        if se_show:
            n_u=st.session_state.get('n_units',0) if dt=='panel' else 0
            if dt=='panel' and n_u>=30:   rec_se,rec_msg='cluster',f"推荐：聚类到 {id_col}（{n_u} 个体，组内相关需聚类SE）"
            elif dt=='panel' and n_u>=10: rec_se,rec_msg='robust',f"推荐：稳健SE（仅 {n_u} 个体，聚类不可靠）"
            elif dt=='panel':             rec_se,rec_msg='ordinary',f"推荐：普通SE（仅 {n_u} 个体）"
            else:                         rec_se,rec_msg='robust',f"推荐：稳健SE（HC1，异方差稳健）"
            se_choice=st.radio("标准误（SE）",
                ["💡 智能推荐","聚类 SE","稳健 SE (HC1)","普通 SE"],
                horizontal=True,key='semode7')
            if '智能' in se_choice:
                st.info(rec_msg); se_mode=rec_se
            elif '聚类' in se_choice: se_mode='cluster'
            elif '稳健' in se_choice: se_mode='robust'
            else:                      se_mode='ordinary'
            use_cl=(se_mode=='cluster')
            use_rob=(se_mode in ('robust','cluster'))
            if use_cl:
                if dt=='panel':
                    cluster_opts=[id_col]+[c for c in cols if c not in [id_col,tc] and (c in cat_cols or (pd.api.types.is_numeric_dtype(df[c]) and 2<=df[c].nunique()<=30))]
                else:
                    cluster_opts=[c for c in cols if c in cat_cols or (pd.api.types.is_numeric_dtype(df[c]) and 2<=df[c].nunique()<=30)]
                cluster_col=st.selectbox("聚类变量",cluster_opts,index=0,key='cl7') if cluster_opts else None
                if not cluster_opts: st.warning("无可用的聚类变量"); use_cl=False
            else:
                cluster_col=None
        else:
            se_mode='ordinary'; use_cl=False; use_rob=False; cluster_col=None

        btn_label="开始搜索最优控制组合"
        if (add_med or add_mod) and '同时' in search_mode and len(core_x)>1:
            st.info("中介/调节效应建议使用「分别显著」模式，将对每个核心X单独分析")
        if st.button(btn_label,type="primary",key='srch7'):
            if not core_x: st.warning("请选核心 X")
            elif (not is_did_like) and 'RDD' not in sel_model and (not ctrl_pool or len(ctrl_pool)<2): st.warning("候选池需 ≥2 个变量")
            elif 'PSM-DID' in sel_model and not ctrl_pool: st.warning("PSM-DID 至少需要 1 个匹配变量/控制变量")
            elif len(ctrl_pool)<cmin: st.warning("候选池不足最少控制数")
            elif 'IV' in sel_model and len(iv_insts)<1: st.warning("工具变量至少选1个")
            else:
                amx=min(cmax,len(ctrl_pool)); amn=min(cmin,amx)
                progress=st.progress(0); stxt=st.empty(); t0=time.time()

                # 准备数据（使用含虚拟变量的增强 dataframe）
                if dt=='panel':
                    id_col,tc_col=st.session_state.id_col,st.session_state.time_col
                    if not id_col or not tc_col or id_col not in df_aug.columns or tc_col not in df_aug.columns:
                        st.error("面板 ID/时间列丢失，请返回 Step 2 重新诊断"); st.stop()
                    use_vars=[id_col,tc_col,y_col]+core_x+fixed_controls+ctrl_pool
                    if did_var: use_vars.append(did_var)
                    if did_post_var and did_post_var not in use_vars: use_vars.append(did_post_var)
                    sub=df_aug[use_vars].dropna().copy()
                    sub[id_col]=sub[id_col].astype(str)
                    if sub.duplicated(subset=[id_col,tc_col]).sum()>0:
                        sub=sub.groupby([id_col,tc_col]).mean().reset_index()
                else:
                    use_vars=[y_col]+core_x+fixed_controls+ctrl_pool
                    if did_var: use_vars.append(did_var)
                    if did_post_var and did_post_var not in use_vars: use_vars.append(did_post_var)
                    if 'RDD' in sel_model and rdd_design.startswith('Fuzzy') and '_rdd_z' not in use_vars:
                        use_vars.append('_rdd_z')
                    sub=df_aug[use_vars].dropna().copy()
                    if 'RDD' in sel_model and rdd_bandwidth and rdd_bandwidth>0:
                        sub=sub[sub['_rdd_running_c'].abs()<=rdd_bandwidth].copy()

                if len(sub)<30:
                    st.error(f"缺失值处理后有效样本只有 {len(sub)} 行，无法稳定回归。请减少变量或先清洗数据。")
                    st.stop()
                if is_did_like:
                    if did_var is None or did_var not in sub.columns or sub[did_var].nunique()<2:
                        st.error("处理组变量需要同时包含处理组和对照组。")
                        st.stop()
                    if '_post' not in sub.columns or sub['_post'].nunique()<2:
                        st.error("Post 变量需要同时包含政策前和政策后。")
                        st.stop()
                    if '_did' not in sub.columns or sub['_did'].nunique()<2:
                        st.error("Treat×Post 没有变化，无法估计 DID。请检查处理组变量和政策时间。")
                        st.stop()
                if 'RDD' in sel_model:
                    side_var='_rdd_z' if '_rdd_z' in sub.columns else '_rdd_treat'
                    side_counts=sub[side_var].value_counts()
                    if len(side_counts)<2:
                        st.error("断点两侧至少都要有样本。请调整 cutoff 或带宽。")
                        st.stop()
                    if side_counts.min()<10:
                        st.error(f"断点一侧样本只有 {int(side_counts.min())} 行，容易回归失败。请放宽带宽或调整 cutoff。")
                        st.stop()
                    if rdd_design.startswith('Fuzzy'):
                        if rdd_treatment not in sub.columns or sub[rdd_treatment].nunique()<2:
                            st.error("Fuzzy RD 的实际处理变量 D 需要有变化。")
                            st.stop()

                # ── 通用搜索函数（OLS 快速筛选） ──
                def search_ols(cx,pool,mn,mx):
                    res=[]; ac=[]
                    for k in range(mn,mx+1): ac.extend(combinations(pool,k))
                    n_total=len(ac)
                    if n_total>10000:
                        rng=np.random.RandomState(42)
                        ac=[ac[i] for i in rng.choice(n_total,10000,replace=False)]
                        n_total=10000
                    for i,combo in enumerate(ac):
                        if i%1000==0: progress.progress(min(i/n_total,.95),text=f"{cx}: {i}/{n_total}")
                        try:
                            Xv=[cx]+fixed_controls+list(combo); td=sub[[y_col]+Xv].dropna()
                            if len(td)<30: continue
                            m=OLS(td[y_col],sm.add_constant(td[Xv])).fit()
                            b=m.params.iloc[1]; se=m.bse.iloc[1]; ts=b/se if se>0 else 0
                            ols_p={}; pidx=list(m.params.index)
                            for jj in range(min(len(pidx),len(['const']+Xv))):
                                ols_p[pidx[jj]]={'b':float(m.params.iloc[jj]),'se':float(m.bse.iloc[jj])}
                            res.append(dict(controls=tuple(fixed_controls)+tuple(combo),n=len(td),tstat=float(ts),
                                pval=float(2*(1-stats.t.cdf(abs(ts),df=len(td)-len(Xv)-1))),
                                rsq=float(m.rsquared),rsq_adj=float(m.rsquared_adj),
                                ols_params=ols_p))
                        except: continue
                    res.sort(key=lambda x: abs(x['tstat']),reverse=True)
                    return res

                def search_ols_joint(core_xs,pool,mn,mx):
                    """OLS搜索：所有核心X联合，按 min|t| 排序"""
                    res=[]; ac=[]
                    for k in range(mn,mx+1): ac.extend(combinations(pool,k))
                    n_total=len(ac)
                    if n_total>10000:
                        rng=np.random.RandomState(42)
                        ac=[ac[i] for i in rng.choice(n_total,10000,replace=False)]
                        n_total=10000
                    cx_list=list(core_xs)
                    for i,combo in enumerate(ac):
                        if i%1000==0: progress.progress(min(i/n_total,.95),text=f"联合搜索: {i}/{n_total}")
                        try:
                            Xv=cx_list+fixed_controls+list(combo); td=sub[[y_col]+Xv].dropna()
                            if len(td)<30: continue
                            m=OLS(td[y_col],sm.add_constant(td[Xv])).fit()
                            ols_p={}; tstats={}; pidx=list(m.params.index)
                            for jj in range(min(len(pidx),len(['const']+Xv))):
                                ols_p[pidx[jj]]={'b':float(m.params.iloc[jj]),'se':float(m.bse.iloc[jj])}
                            for cx in cx_list:
                                try:
                                    idx=list(m.params.index).index(cx)
                                    b=m.params.iloc[idx]; se=m.bse.iloc[idx]
                                    tstats[cx]=float(b/se) if se>0 else 0
                                except: tstats[cx]=0
                            min_t=min(abs(t) for t in tstats.values())
                            res.append(dict(controls=tuple(fixed_controls)+tuple(combo),n=len(td),min_abs_tstat=float(min_t),
                                tstats=tstats,rsq=float(m.rsquared),rsq_adj=float(m.rsquared_adj),
                                ols_params=ols_p,all_Xv=Xv))
                        except: continue
                    res.sort(key=lambda x: x['min_abs_tstat'],reverse=True)
                    return res

                def search_fuzzy_rdd(cx,pool,mn,mx):
                    res=[]; ac=[]
                    for k in range(mn,mx+1): ac.extend(combinations(pool,k))
                    n_total=len(ac)
                    if n_total>10000:
                        rng=np.random.RandomState(42)
                        ac=[ac[i] for i in rng.choice(n_total,10000,replace=False)]
                        n_total=10000
                    for i,combo in enumerate(ac):
                        if i%500==0: progress.progress(min(i/max(n_total,1),.95),text=f"Fuzzy RD: {i}/{n_total}")
                        try:
                            exog_vars=unique_keep_order(fixed_controls+list(combo))
                            need=[y_col,cx,'_rdd_z']+exog_vars
                            td=sub[need].replace([np.inf,-np.inf],np.nan).dropna()
                            if len(td)<30 or td['_rdd_z'].nunique()<2 or td[cx].nunique()<2: continue
                            exog=sm.add_constant(td[exog_vars]) if exog_vars else sm.add_constant(pd.DataFrame(index=td.index))
                            iv_m=IV2SLS(td[y_col],exog,td[[cx]],td[['_rdd_z']]).fit(cov_type='robust')
                            b=iv_m.params.get(cx,np.nan); se=iv_m.std_errors.get(cx,np.nan)
                            if np.isnan(b) or np.isnan(se) or se<=0: continue
                            fs=OLS(td[cx],sm.add_constant(td[exog_vars+['_rdd_z']])).fit(cov_type='HC1')
                            rf=OLS(td[y_col],sm.add_constant(td[exog_vars+['_rdd_z']])).fit(cov_type='HC1')
                            res.append(dict(controls=tuple(fixed_controls)+tuple(combo),n=len(td),
                                tstat=float(b/se),pval=float(iv_m.pvalues.get(cx,np.nan)),
                                rsq=np.nan,rsq_adj=np.nan,
                                iv_beta=float(b),iv_se=float(se),
                                first_stage=float(fs.params.get('_rdd_z',np.nan)),
                                first_stage_p=float(fs.pvalues.get('_rdd_z',np.nan)),
                                reduced_form=float(rf.params.get('_rdd_z',np.nan)),
                                reduced_form_p=float(rf.pvalues.get('_rdd_z',np.nan))))
                        except: continue
                    res.sort(key=lambda x: abs(x['tstat']),reverse=True)
                    return res

                # ── 面板搜索（完整 FE+RE+Hausman） ──
                def search_panel_full(cx,pool,mn,mx,preselected_ctrls=None):
                    res=[]; ac=[]
                    if preselected_ctrls is not None:
                        ac=preselected_ctrls  # 只评估预选的组合（两阶段加速）
                    else:
                        for k in range(mn,mx+1): ac.extend(combinations(pool,k))
                    n_total=len(ac)
                    if n_total>10000 and preselected_ctrls is None:
                        rng=np.random.RandomState(42)
                        ac=[ac[i] for i in rng.choice(n_total,10000,replace=False)]
                        n_total=10000
                    d=sub.set_index([id_col,tc_col])[[y_col,cx]+pool].dropna()
                    for i,combo in enumerate(ac):
                        if i%500==0: progress.progress(min(i/n_total,.95),text=f"{cx}: {i}/{n_total}")
                        try:
                            Xv=[cx]+list(combo); td=d[[y_col]+Xv].dropna()
                            if len(td)<50: continue
                            n_id_i=td.index.get_level_values(0).nunique(); n_t_i=td.index.get_level_values(1).nunique()
                            Xex=sm.add_constant(td[Xv])
                            fe_m=PanelOLS(td[y_col],Xex,entity_effects=True,time_effects=True)
                            re_m=RandomEffects(td[y_col],Xex)
                            cok=False; tstat_ord=0.0; tstat_rob=0.0; ts_cl=0.0
                            fell_back=False; se_used='ordinary'
                            # 1. 普通 SE（基线）
                            try:
                                fe_ord=fe_m.fit(); re_ord=re_m.fit()
                                b_ord=fe_ord.params.get(cx,np.nan); s_ord=fe_ord.std_errors.get(cx,np.nan)
                                tstat_ord=float(b_ord/s_ord) if s_ord>0 else 0
                            except: continue
                            fe_r,re_r=fe_ord,re_ord; fe_b,fe_se,ts=b_ord,s_ord,tstat_ord
                            # 2. 稳健 SE（use_rob 对 robust 和 cluster 模式都为 True）
                            if use_rob:
                                try:
                                    fe_rob=fe_m.fit(cov_type='robust'); re_rob=re_m.fit(cov_type='robust')
                                    b_rob=fe_rob.params.get(cx,np.nan); s_rob=fe_rob.std_errors.get(cx,np.nan)
                                    tstat_rob=float(b_rob/s_rob) if s_rob>0 else 0
                                except: pass
                            # 3. 聚类 SE
                            if use_cl and cluster_col:
                                try:
                                    cl_vals=df_aug.set_index([id_col,tc_col]).loc[td.index,cluster_col].values
                                    fe_cl=fe_m.fit(cov_type='clustered',clusters=cl_vals)
                                    re_cl=re_m.fit(cov_type='clustered',clusters=cl_vals); cok=True
                                    b_cl=fe_cl.params.get(cx,np.nan); s_cl=fe_cl.std_errors.get(cx,np.nan)
                                    ts_cl=float(b_cl/s_cl) if s_cl>0 else 0
                                except: pass
                            # 4. 回退链：cluster → robust → ordinary
                            if cok and abs(ts_cl)>=1.96:
                                fe_r,re_r=fe_cl,re_cl; fe_b,fe_se,ts=b_cl,s_cl,ts_cl; se_used='cluster'
                            elif use_rob and abs(tstat_rob)>=1.96:
                                fe_r,re_r=fe_rob,re_rob; fe_b,fe_se,ts=b_rob,s_rob,tstat_rob; se_used='robust'
                                if cok: fell_back=True
                            elif abs(tstat_ord)>=1.96:
                                se_used='ordinary'
                                if use_rob: fell_back=True
                            else:
                                # 都不显著→用首选（最高级）SE
                                if cok: se_used='cluster'; fe_r,re_r=fe_cl,re_cl; fe_b,fe_se,ts=b_cl,s_cl,ts_cl
                                elif use_rob: se_used='robust'; fe_r,re_r=fe_rob,re_rob; fe_b,fe_se,ts=b_rob,s_rob,tstat_rob
                            re_b=re_r.params.get(cx,np.nan); re_se=re_r.std_errors.get(cx,np.nan)
                            common=[c for c in fe_r.params.index.intersection(re_r.params.index) if c!='const']
                            hs,hp=np.nan,np.nan
                            if common:
                                d_p=fe_r.params.loc[common]-re_r.params.loc[common]
                                Vv=fe_r.cov.loc[common,common].values-re_r.cov.loc[common,common].values
                                try: hs=float(d_p.values.T@np.linalg.pinv(Vv)@d_p.values); hp=float(1-stats.chi2.cdf(hs,df=len(common)))
                                except: pass
                            all_v=[v for v in fe_r.params.index if v!='const']
                            fe_params={v:{'b':float(fe_r.params[v]),'se':float(fe_r.std_errors[v])} for v in all_v}
                            re_params={v:{'b':float(re_r.params[v]),'se':float(re_r.std_errors[v])} for v in all_v if v in re_r.params.index}
                            res.append(dict(controls=combo,n=len(td),n_units=n_id_i,n_periods=n_t_i,
                                fe_beta=float(fe_b),fe_se=float(fe_se),re_beta=float(re_b),re_se=float(re_se),
                                tstat=float(ts),pval=float(2*(1-stats.t.cdf(abs(ts),df=len(td)-len(Xv)-1))),
                                se_used=se_used,fell_back=fell_back,
                                tstat_ord=float(tstat_ord),tstat_rob=float(tstat_rob) if use_rob else None,
                                tstat_clust=float(ts_cl) if cok else None,
                                fe_rsq=float(fe_r.rsquared),re_rsq=float(re_r.rsquared),
                                hausman_chi2=float(hs) if not np.isnan(hs) else None,
                                hausman_p=float(hp) if not np.isnan(hp) else None,
                                hausman_df=len(common) if common else 0,cluster_ok=cok,
                                fe_params=fe_params,re_params=re_params,
                                const_fe_b=float(fe_r.params.get('const',np.nan)),
                                const_fe_se=float(fe_r.std_errors.get('const',np.nan)),
                                const_re_b=float(re_r.params.get('const',np.nan)),
                                const_re_se=float(re_r.std_errors.get('const',np.nan)),Xv=Xv))
                        except: continue
                    res.sort(key=lambda x: abs(x['tstat']),reverse=True)
                    return res

                def search_panel_joint(core_xs,pool,mn,mx,preselected_ctrls=None):
                    """面板联合搜索：所有核心X一起进入 FE+RE，按 min|t| 排序"""
                    res=[]; ac=[]; cx_list=list(core_xs)
                    if preselected_ctrls is not None:
                        ac=preselected_ctrls
                    else:
                        for k in range(mn,mx+1): ac.extend(combinations(pool,k))
                    n_total=len(ac)
                    if n_total>10000 and preselected_ctrls is None:
                        rng=np.random.RandomState(42)
                        ac=[ac[i] for i in rng.choice(n_total,10000,replace=False)]
                        n_total=10000
                    d=sub.set_index([id_col,tc_col])[[y_col]+cx_list+pool].dropna()
                    for i,combo in enumerate(ac):
                        if i%500==0: progress.progress(min(i/n_total,.95),text=f"面板联合: {i}/{n_total}")
                        try:
                            Xv=cx_list+list(combo); td=d[[y_col]+Xv].dropna()
                            if len(td)<50: continue
                            Xex=sm.add_constant(td[Xv])
                            fe_m=PanelOLS(td[y_col],Xex,entity_effects=True,time_effects=True)
                            re_m=RandomEffects(td[y_col],Xex)
                            cok=False; fell_back_j=False; se_used_j='ordinary'
                            def _tstats_dict(fit,cxlist):
                                d={}
                                for cxx in cxlist:
                                    b=fit.params.get(cxx,np.nan); s=fit.std_errors.get(cxx,np.nan)
                                    d[cxx]=float(b/s) if s>0 else 0
                                return d,min(abs(t) for t in d.values())
                            # 1. 普通
                            try:
                                fe_ord=fe_m.fit(); re_ord=re_m.fit()
                                tstats_ord,min_t_ord=_tstats_dict(fe_ord,cx_list)
                            except: continue
                            fe_r,re_r=fe_ord,re_ord; tstats=tstats_ord; min_t=min_t_ord
                            # 2. 稳健
                            if use_rob:
                                try:
                                    fe_rob=fe_m.fit(cov_type='robust'); re_rob=re_m.fit(cov_type='robust')
                                    tstats_rob,min_t_rob=_tstats_dict(fe_rob,cx_list)
                                except: pass
                            # 3. 聚类
                            if use_cl and cluster_col:
                                try:
                                    cl_vals=df_aug.set_index([id_col,tc_col]).loc[td.index,cluster_col].values
                                    fe_cl=fe_m.fit(cov_type='clustered',clusters=cl_vals)
                                    re_cl=re_m.fit(cov_type='clustered',clusters=cl_vals); cok=True
                                    tstats_cl,min_t_cl=_tstats_dict(fe_cl,cx_list)
                                except: pass
                            # 4. 回退链
                            if cok and min_t_cl>=1.96:
                                fe_r,re_r=fe_cl,re_cl; tstats=tstats_cl; min_t=min_t_cl; se_used_j='cluster'
                            elif use_rob and min_t_rob>=1.96:
                                fe_r,re_r=fe_rob,re_rob; tstats=tstats_rob; min_t=min_t_rob; se_used_j='robust'
                                if cok: fell_back_j=True
                            elif min_t_ord>=1.96:
                                se_used_j='ordinary'
                                if use_rob: fell_back_j=True
                            else:
                                if cok: se_used_j='cluster'; fe_r,re_r=fe_cl,re_cl; tstats=tstats_cl; min_t=min_t_cl
                                elif use_rob: se_used_j='robust'; fe_r,re_r=fe_rob,re_rob; tstats=tstats_rob; min_t=min_t_rob
                            fc0=cx_list[0]
                            fe_b=fe_r.params.get(fc0,np.nan); fe_s=fe_r.std_errors.get(fc0,np.nan)
                            re_b=re_r.params.get(fc0,np.nan); re_s=re_r.std_errors.get(fc0,np.nan)
                            ts=fe_b/fe_s if fe_s>0 else 0
                            common=[c for c in fe_r.params.index.intersection(re_r.params.index) if c!='const']
                            hs,hp=np.nan,np.nan
                            if common:
                                d_p=fe_r.params.loc[common]-re_r.params.loc[common]
                                Vv=fe_r.cov.loc[common,common].values-re_r.cov.loc[common,common].values
                                try: hs=float(d_p.values.T@np.linalg.pinv(Vv)@d_p.values); hp=float(1-stats.chi2.cdf(hs,df=len(common)))
                                except: pass
                            all_v=[v for v in fe_r.params.index if v!='const']
                            fe_params={v:{'b':float(fe_r.params[v]),'se':float(fe_r.std_errors[v])} for v in all_v}
                            re_params={v:{'b':float(re_r.params[v]),'se':float(re_r.std_errors[v])} for v in all_v if v in re_r.params.index}
                            res.append(dict(controls=combo,n=len(td),n_units=td.index.get_level_values(0).nunique(),
                                n_periods=td.index.get_level_values(1).nunique(),
                                fe_beta=float(fe_b),fe_se=float(fe_s),re_beta=float(re_b),re_se=float(re_s),
                                tstat=float(ts),tstats=tstats,min_abs_tstat=float(min_t),
                                se_used=se_used_j,fell_back=fell_back_j,
                                fe_rsq=float(fe_r.rsquared),re_rsq=float(re_r.rsquared),
                                hausman_chi2=float(hs) if not np.isnan(hs) else None,
                                hausman_p=float(hp) if not np.isnan(hp) else None,
                                hausman_df=len(common) if common else 0,cluster_ok=cok,
                                fe_params=fe_params,re_params=re_params,
                                const_fe_b=float(fe_r.params.get('const',np.nan)),
                                const_fe_se=float(fe_r.std_errors.get('const',np.nan)),
                                const_re_b=float(re_r.params.get('const',np.nan)),
                                const_re_se=float(re_r.std_errors.get('const',np.nan)),Xv=Xv))
                        except: continue
                    res.sort(key=lambda x: x['min_abs_tstat'],reverse=True)
                    return res

                sr={}
                if '同时' in search_mode and len(core_x)>1:
                    # 联合显著：所有核心X一起搜索，按最弱|t|排序
                    stxt.text("联合搜索（所有核心X同时显著）...")
                    if dt=='panel' and 'FE+RE' in sel_model:
                        sr['joint']=search_panel_joint(core_x,ctrl_pool,amn,amx)[:50]
                    else:
                        sr['joint']=search_ols_joint(core_x,ctrl_pool,amn,amx)[:50]
                    stxt.text(f"联合搜索: {len(sr['joint'])} 个有效组合")
                else:
                    # 分别显著：每个核心X独立搜索
                    for cx in core_x:
                        stxt.text(f"搜索 {cx}...")
                        if dt=='panel' and 'FE+RE' in sel_model:
                            sr[cx]=search_panel_full(cx,ctrl_pool,amn,amx)[:50]
                        elif 'RDD' in sel_model and rdd_design.startswith('Fuzzy'):
                            sr[cx]=search_fuzzy_rdd(cx,ctrl_pool,amn,amx)[:50]
                        else:
                            sr[cx]=search_ols(cx,ctrl_pool,amn,amx)[:50]
                        stxt.text(f"{cx}: {len(sr[cx])} 个有效组合")
                progress.progress(1.0)
                # ── 两阶段：中介/调节 → 在显著组合上跑第二阶段的效应分析 ──
                med_results=None; mod_results=None
                if add_med and med_m:
                    med_results={}; is_wen='温忠麟' in med_method
                    for cx_key in (sr.keys() if sr else []):
                        if cx_key not in core_x: continue  # skip joint keys
                        rlist=sr[cx_key]; top_sig=[r for r in rlist[:20] if r.get('pval',1)<0.1][:10]
                        if not top_sig: top_sig=rlist[:5]
                        cx_meds=[]
                        for i,r in enumerate(top_sig):
                            ctrls=list(r['controls'])
                            if is_wen:
                                mr=run_mediation_wen2014(y_col,cx_key,med_m,ctrls,df_aug)
                            else:
                                mr=run_mediation_jiang(y_col,cx_key,med_m,ctrls,df_aug)
                            if mr: mr['controls']=ctrls; mr['cx']=cx_key; mr['combo_idx']=i; cx_meds.append(mr)
                        if cx_meds: med_results[cx_key]=cx_meds
                if add_mod and mod_m:
                    mod_results={}
                    for cx_key in (sr.keys() if sr else []):
                        if cx_key not in core_x: continue
                        rlist=sr[cx_key]; top_sig=[r for r in rlist[:20] if r.get('pval',1)<0.1][:10]
                        if not top_sig: top_sig=rlist[:5]
                        cx_mods=[]
                        for i,r in enumerate(top_sig):
                            ctrls=list(r['controls'])
                            mr=run_moderation_analysis(y_col,cx_key,mod_m,ctrls,df_aug)
                            if mr: mr['controls']=ctrls; mr['cx']=cx_key; mr['combo_idx']=i; cx_mods.append(mr)
                        if cx_mods: mod_results[cx_key]=cx_mods
                st.success(f"完成！{time.time()-t0:.1f}s")
                st.session_state.search_results=sr
                st.session_state._y=y_col; st.session_state._y_type=y_type
                st.session_state._is_panel=(dt=='panel'); st.session_state._model=sel_model
                st.session_state._sub=sub; st.session_state._search_mode=search_mode
                for kk in ['_did_var','_use_cl','_heckman_sel','_iv_endog','_iv_insts','_id_col','_tc_col','_bin_m']:
                    if kk in st.session_state: del st.session_state[kk]
                st.session_state._did_var=did_var; st.session_state._use_cl=use_cl
                st.session_state._did_policy_time=did_policy_time; st.session_state._did_post_var=did_post_var; st.session_state._did_post_mode=did_post_mode
                st.session_state._rdd_running=rdd_running; st.session_state._rdd_cutoff=rdd_cutoff; st.session_state._rdd_bandwidth=rdd_bandwidth; st.session_state._rdd_order=rdd_order
                st.session_state._rdd_design=rdd_design; st.session_state._rdd_treatment=rdd_treatment
                st.session_state._cluster_col=cluster_col
                st.session_state._se_mode=se_mode; st.session_state._use_rob=use_rob
                st.session_state._heckman_sel=heckman_sel
                st.session_state._iv_endog=iv_endog; st.session_state._iv_insts=iv_insts
                st.session_state._id_col=st.session_state.id_col if dt=='panel' else None
                st.session_state._tc_col=st.session_state.time_col if dt=='panel' else None
                st.session_state._group_var=group_var
                st.session_state._df_aug=df_aug
                st.session_state._cat_cols=cat_cols
                st.session_state._dummy_names=dummy_names
                st.session_state._med_results=med_results; st.session_state._mod_results=mod_results
                st.session_state._med_m=med_m; st.session_state._mod_m=mod_m
                st.session_state._add_med=add_med; st.session_state._add_mod=add_mod
                st.session_state._med_method=med_method
                # 清理旧的异质性/稳健性结果
                for sk in [k for k in st.session_state if k.startswith('_het_') or k.startswith('_rob_')]:
                    del st.session_state[sk]

    # ═══════════════════ STEP 4: 结果 ═══════════════════
    if st.session_state.search_results is not None:
        st.header("Step 4 · 选择组合 → 完整结果 & 代码")
        sr=st.session_state.search_results; y_col=st.session_state._y
        model_sel=st.session_state._model; yt=st.session_state._y_type
        is_panel=st.session_state._is_panel; sub=st.session_state._sub
        did_var=st.session_state._did_var; use_cl=st.session_state._use_cl
        did_policy_time=st.session_state.get('_did_policy_time',None); did_post_var=st.session_state.get('_did_post_var',None); did_post_mode=st.session_state.get('_did_post_mode','按政策时间生成 Post')
        rdd_running=st.session_state.get('_rdd_running',None); rdd_cutoff=st.session_state.get('_rdd_cutoff',None); rdd_bandwidth=st.session_state.get('_rdd_bandwidth',0.0); rdd_order=st.session_state.get('_rdd_order','局部线性（一阶）')
        rdd_design=st.session_state.get('_rdd_design','Sharp RD（精确断点）'); rdd_treatment=st.session_state.get('_rdd_treatment',None)
        cluster_col=st.session_state.get('_cluster_col',None)
        se_mode=st.session_state.get('_se_mode','ordinary'); use_rob=st.session_state.get('_use_rob',False)
        heckman_sel=st.session_state._heckman_sel
        iv_endog=st.session_state._iv_endog; iv_insts=st.session_state._iv_insts
        group_var=st.session_state.get('_group_var',None)
        df_aug=st.session_state.get('_df_aug',sub)
        search_mode=st.session_state.get('_search_mode','分别显著（各X独立搜索）')
        is_joint=('同时' in search_mode)

        def did_followup_panel(base_td, control_terms, id_col, tc_col, post_stata, key_prefix, title_prefix):
            st.divider()
            st.subheader(f"{title_prefix} 后续分析")
            st.caption("沿用当前选中的 DID 组合、样本、Treat、Post 和控制变量；无需回到上面重新搜索。")
            base_keys=base_td[[id_col,tc_col,'_did','_post']].drop_duplicates().copy()
            base_keys[id_col]=base_keys[id_col].astype(str)
            all_follow=df_aug.copy()
            all_follow[id_col]=all_follow[id_col].astype(str)
            follow=base_keys.merge(all_follow,on=[id_col,tc_col],how='left',suffixes=('','_raw'))
            for c in [y_col]+list(control_terms):
                raw_c=f"{c}_raw"
                if c not in follow.columns and raw_c in follow.columns:
                    follow[c]=follow[raw_c]
            num_candidates=[c for c in df_aug.columns
                if c not in [id_col,tc_col,y_col,did_var,did_post_var,'_did','_treat','_post']
                and c not in control_terms and pd.api.types.is_numeric_dtype(df_aug[c])]
            if not num_candidates:
                st.info("当前没有可用于机制/调节的数值变量。")
                return
            f1,f2=st.columns(2)
            with f1:
                run_med=st.checkbox("继续做机制分析", key=f'{key_prefix}_did_med_on')
            with f2:
                run_mod=st.checkbox("继续做调节分析", key=f'{key_prefix}_did_mod_on')
            user_ctrls=[v for v in control_terms if v in df_aug.columns and not v.startswith('_')]
            ctrl_text=(' '.join(user_ctrls)+' ') if user_ctrls else ''
            if run_med:
                med_m=st.selectbox("机制变量 M", num_candidates, key=f'{key_prefix}_did_med_m')
                if st.button("运行机制分析", key=f'{key_prefix}_run_did_med'):
                    mr,err=run_did_mediation_panel(follow,y_col,med_m,control_terms,id_col,tc_col)
                    if err:
                        st.error(f"机制分析失败：{err}")
                    else:
                        c1,c2,c3,c4=st.columns(4)
                        with c1: st.metric("总效应 c",f"{mr['c']:.4f}",f"p={mr['c_p']:.4f}")
                        with c2: st.metric("DID→M a",f"{mr['a']:.4f}",f"p={mr['a_p']:.4f}")
                        with c3: st.metric("M→Y b",f"{mr['b']:.4f}",f"p={mr['b_p']:.4f}")
                        with c4: st.metric("间接项 a×b",f"{mr['ab']:.4f}")
                        if mr['effect_type'] in ['机制成立','渠道成立']:
                            st.success(f"机制结果：{mr['effect_type']} ｜ N={mr['n']}")
                        else:
                            st.warning(f"机制结果：{mr['effect_type']} ｜ N={mr['n']}")
                        st.caption(f"R²：总效应={mr['rsq_y']:.3f}，机制方程={mr['rsq_m']:.3f}，加入 M 后={mr['rsq_full']:.3f}")
                        do=(f"* === 鹈鹕回归 (c)2026 · DID 机制分析 ===\nuse \"data.dta\", clear\n"
                            f"xtset {id_col} {tc_col}\n{post_stata}\ngen treat={did_var}\ngen did=treat*post\n"
                            f"* 第1步：DID 总效应\nreghdfe {y_col} {ctrl_text}post did, absorb({id_col}) vce(robust)\n"
                            f"* 第2步：DID 对机制变量 M 的影响\nreghdfe {med_m} {ctrl_text}post did, absorb({id_col}) vce(robust)\n"
                            f"* 第3步：加入机制变量 M\nreghdfe {y_col} {ctrl_text}post did {med_m}, absorb({id_col}) vce(robust)\n")
                        with st.expander("机制分析 Stata 代码"): st.code(do,language='stata')
            if run_mod:
                mod_m=st.selectbox("调节变量 M", num_candidates, key=f'{key_prefix}_did_mod_m')
                if st.button("运行调节分析", key=f'{key_prefix}_run_did_mod'):
                    rr,err=run_did_moderation_panel(follow,y_col,mod_m,control_terms,id_col,tc_col)
                    if err:
                        st.error(f"调节分析失败：{err}")
                    else:
                        c1,c2,c3=st.columns(3)
                        with c1: st.metric("DID 主效应",f"{rr['b_did']:.4f}")
                        with c2: st.metric(f"{mod_m} 主效应",f"{rr['b_m']:.4f}")
                        with c3: st.metric("DID×M",f"{rr['b_inter']:.4f}",f"p={rr['p_inter']:.4f}")
                        if rr['p_inter']<0.1:
                            st.success(f"调节项显著：{mod_m} 调节 Treat×Post 对 {y_col} 的影响 ｜ N={rr['n']}")
                        else:
                            st.warning(f"调节项暂不显著 ｜ N={rr['n']}")
                        do=(f"* === 鹈鹕回归 (c)2026 · DID 调节分析 ===\nuse \"data.dta\", clear\n"
                            f"xtset {id_col} {tc_col}\n{post_stata}\ngen treat={did_var}\ngen did=treat*post\n"
                            f"gen did_x_{mod_m}=did*{mod_m}\n"
                            f"reghdfe {y_col} {ctrl_text}post did {mod_m} did_x_{mod_m}, absorb({id_col}) vce(robust)\n")
                        with st.expander("调节分析 Stata 代码"): st.code(do,language='stata')

        for cx,results in sr.items():
            if not results: st.warning(f"{cx}: 无有效组合"); continue
            display_cx='Treat×Post' if cx=='_did' else ('断点处理项' if cx=='_rdd_treat' else cx)
            top=results[:20]
            # 获取核心X列表
            if is_joint and results:
                jcx_keys=[k for k in results[0].get('tstats',{}).keys()]
            else:
                jcx_keys=[]

            # 排名表
            tbl=[]
            for i,r in enumerate(top):
                shown_controls=[c for c in r['controls'] if c not in ['_treat','_post']]
                cs=', '.join(shown_controls[:4]) if shown_controls else '无额外控制变量'
                if len(shown_controls)>4: cs+=f' +{len(shown_controls)-4}'
                cx_b='—'
                if is_joint:
                    # 联合模式：显示 min|t| + 各核心X的t
                    t_parts=[f"{k}={abs(r.get('tstats',{}).get(k,0)):.1f}" for k in jcx_keys[:4]]
                    cx_b=f"min|t|={r.get('min_abs_tstat',0):.2f} [{', '.join(t_parts)}]"
                    tbl.append({'#':i+1,'控制组合':cs,'联合t (min|t|及各X)':cx_b,
                        'R²':f"{r.get('rsq',0):.3f}",'N':r['n']})
                else:
                    s='***' if r['pval']<0.01 else ('**' if r['pval']<0.05 else ('*' if r['pval']<0.1 else ''))
                    if 'fe_beta' in r: cx_b=f"{r['fe_beta']:.4f}{s}"
                    elif 'iv_beta' in r: cx_b=f"{r['iv_beta']:.4f}{s}"
                    elif 'ols_params' in r and len(r['ols_params'])>1:
                        pk=list(r['ols_params'].keys())[1]; cx_b=f"{r['ols_params'][pk]['b']:.4f}{s}"
                    tbl.append({'#':i+1,'控制组合':cs,'β(SE)':cx_b,'|t|':f"{abs(r['tstat']):.2f}",
                        'R²':f"{r.get('fe_rsq',r.get('rsq',0)):.3f}",'N':r['n']})
            n_sig05=sum(1 for r in results if (r.get('min_abs_tstat',abs(r.get('tstat',0))) if is_joint else abs(r['tstat']))>stats.t.ppf(0.975,df=max(r['n']-5,1)))
            expander_label=f"**{display_cx}** — {len(results)}组合 ｜ {'min|t|' if is_joint else '|t|'}>1.96占 {n_sig05/max(len(results),1)*100:.0f}%"
            with st.expander(expander_label,expanded=True):
                st.dataframe(pd.DataFrame(tbl),use_container_width=True,hide_index=True)

                combo_labels={}
                for i,r in enumerate(top):
                    shown_controls=[c for c in r['controls'] if c not in ['_treat','_post']]
                    cc=', '.join(shown_controls[:3]) if shown_controls else '无额外控制变量'
                    if len(shown_controls)>3: cc+=f'...+{len(shown_controls)-3}'
                    combo_labels[f"#{i+1}: {cc}"]=r
                sel_key=f'sel_{cx}_v7'
                if sel_key not in st.session_state: st.session_state[sel_key]=list(combo_labels.keys())[0]
                sel=st.selectbox(f"选择 {display_cx} 的组合",list(combo_labels.keys()),key=sel_key)
                chosen=combo_labels[sel]; ctrls=list(chosen['controls']); N_eff=chosen['n']
                if is_joint:
                    model_line=f"{y_col} = {' + '.join(jcx_keys)} + [{', '.join(ctrls[:6])}{' +...' if len(ctrls)>6 else ''}]"
                    Xv0=jcx_keys+ctrls
                else:
                    model_controls=[c for c in ctrls if c not in ['_treat','_post']]
                    model_line=f"{y_col} = {display_cx} + [{', '.join(model_controls[:6])}{' +...' if len(model_controls)>6 else ''}]"
                    Xv0=[cx]+ctrls

                # ═══ 根据模型类型跑完整回归 ═══
                if is_panel and 'FE+RE' in model_sel:
                    # 面板 FE+RE（已有逻辑）
                    id_col=st.session_state._id_col; tc_col=st.session_state._tc_col
                    n_id=chosen.get('n_units',0); n_t=chosen.get('n_periods',0)
                    hp=chosen.get('hausman_p')
                    if hp is not None and not np.isnan(hp):
                        decision='**固定效应 FE**' if hp<0.05 else '**随机效应 RE**'
                        (st.success if hp<0.05 else st.info)(f"Hausman χ²({chosen['hausman_df']})={chosen['hausman_chi2']:.2f}, p={hp:.4f} → {decision}")
                    rows=[]
                    for v in Xv0:
                        fe_b=chosen['fe_params'].get(v,{}).get('b',np.nan); fe_se=chosen['fe_params'].get(v,{}).get('se',np.nan)
                        re_b=chosen['re_params'].get(v,{}).get('b',np.nan); re_se=chosen['re_params'].get(v,{}).get('se',np.nan)
                        t_f=abs(fe_b/fe_se) if fe_se>0 else 0; p_f=2*(1-stats.t.cdf(t_f,df=N_eff)) if t_f>0 else 1
                        t_r=abs(re_b/re_se) if re_se>0 else 0; p_r=2*(1-stats.t.cdf(t_r,df=N_eff)) if t_r>0 else 1
                        s_f='***' if p_f<0.01 else ('**' if p_f<0.05 else ('*' if p_f<0.1 else ''))
                        s_r='***' if p_r<0.01 else ('**' if p_r<0.05 else ('*' if p_r<0.1 else ''))
                        rows.append({'变量':v,'FE 系数':f"{fe_b:.4f}{s_f}",'FE (SE)':f"({fe_se:.4f})",
                            'RE 系数':f"{re_b:.4f}{s_r}",'RE (SE)':f"({re_se:.4f})"})
                    const_fb=chosen.get('const_fe_b',np.nan)
                    if not np.isnan(const_fb):
                        rows.append({'变量':'常数项','FE 系数':f"{const_fb:.4f}",'FE (SE)':f"({chosen.get('const_fe_se',np.nan):.4f})",
                            'RE 系数':f"{chosen.get('const_re_b',np.nan):.4f}",'RE (SE)':f"({chosen.get('const_re_se',np.nan):.4f})"})
                    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
                    fell_back=chosen.get('fell_back',False); cok=chosen.get('cluster_ok',False)
                    se_used_r=chosen.get('se_used','ordinary')
                    # 回退警告
                    if fell_back:
                        if se_used_r=='robust':
                            ts_hi=chosen.get('tstat_clust') or 0; ts_lo=chosen.get('tstat_rob') or 0
                            st.warning(f"聚类到 {cluster_col} 后不显著（|t|={ts_hi:.2f}），已回退至稳健 SE（|t|={ts_lo:.2f}）")
                        elif se_used_r=='ordinary':
                            ts_hi=chosen.get('tstat_rob') or chosen.get('tstat_clust') or 0; ts_lo=chosen.get('tstat_ord') or 0
                            st.warning(f"聚类/稳健 SE 下不显著（|t|={ts_hi:.2f}），已回退至普通 SE（|t|={ts_lo:.2f}）")
                    # SE 标签
                    if se_used_r=='cluster': se_label="聚类稳健SE" if not fell_back else "聚类稳健SE"
                    elif se_used_r=='robust': se_label="异方差稳健SE (HC1)"
                    else: se_label="普通SE"
                    st.caption(f"注：{se_label}。*** p<0.01, ** p<0.05, * p<0.10。\nN={N_eff}｜个体={n_id}｜时期={n_t}\n含个体+时间固定效应\nFE R²={chosen['fe_rsq']:.4f}｜RE R²={chosen['re_rsq']:.4f}")
                    # Stata 代码
                    do=(f"* === 鹈鹕回归 (c)2026 · 仅供参考 · 不构成统计建议 ===\n* {model_line} ｜ SE: {se_label}\n\nuse \"data.dta\", clear\nxtset {id_col} {tc_col}\n\n")
                    if se_used_r=='cluster': do+=f"reghdfe {y_col} {cx} {' '.join(ctrls)}, absorb({id_col} {tc_col}) vce(cluster {cluster_col})\n"
                    elif se_used_r=='robust': do+=f"reghdfe {y_col} {cx} {' '.join(ctrls)}, absorb({id_col} {tc_col}) vce(robust)\n"
                    else: do+=f"reghdfe {y_col} {cx} {' '.join(ctrls)}, absorb({id_col} {tc_col})\n"
                    do+="est sto fe\n\n"
                    if se_used_r=='cluster': do+=f"xtreg {y_col} {cx} {' '.join(ctrls)}, re vce(cluster {cluster_col})\n"
                    elif se_used_r=='robust': do+=f"xtreg {y_col} {cx} {' '.join(ctrls)}, re vce(robust)\n"
                    else: do+=f"xtreg {y_col} {cx} {' '.join(ctrls)}, re\n"
                    do+="est sto re\nhausman fe re\n"
                    with st.expander("Stata 复现代码"): st.code(do,language='stata')
                    dl1,dl2=st.columns(2)
                    dl1.download_button("下载 .do",do,file_name=f"{cx}_panel.do",key=f'dl_{cx}_v7')
                    dl2.download_button("下载 .csv",pd.DataFrame(rows).to_csv(index=False),file_name=f"{cx}_panel.csv",mime="text/csv",key=f'csv_{cx}_v7')

                elif is_panel and 'DID' in model_sel and 'PSM' not in model_sel:
                    # 2×2 DID
                    id_col=st.session_state._id_col; tc_col=st.session_state._tc_col
                    did_cols=list(dict.fromkeys([id_col,tc_col,y_col,did_var]+Xv0))
                    if did_post_var and did_post_var not in did_cols: did_cols.append(did_post_var)
                    td=sub[did_cols].dropna().copy()
                    td[id_col]=td[id_col].astype(str)
                    # 创建 Post：由用户选择政策时间或已有 Post 变量
                    if did_post_var and did_post_var in td.columns:
                        td['_post']=td[did_post_var].astype(float)
                        post_note=f"Post 使用已有变量 {did_post_var}"
                        post_stata=f"gen post={did_post_var}"
                    else:
                        cutoff=did_policy_time if did_policy_time is not None else td[tc_col].median()
                        td['_post']=(td[tc_col]>=cutoff).astype(float)
                        post_note=f"Post=({tc_col}>={cutoff})"
                        post_stata=f"gen post=({tc_col}>={cutoff})"
                    td['_treat']=td[did_var]
                    td['_did']=td['_treat']*td['_post']
                    td_idx=td.set_index([id_col,tc_col])
                    did_terms=['_treat','_post','_did']
                    control_terms=[v for v in Xv0 if v not in did_terms]
                    model_terms=unique_keep_order([v for v in control_terms+['_post','_did'] if v in td_idx.columns and td_idx[v].nunique(dropna=True)>1])
                    if '_did' not in model_terms:
                        st.error("DID 交互项没有变化，无法估计。请检查处理组变量和政策时间。")
                        st.stop()
                    X_did=sm.add_constant(td_idx[model_terms])
                    try:
                        d_r=safe_panel_fit(td_idx[y_col],X_did,entity_effects=True,time_effects=False)
                        st.info(f"DID 估计量（交互项 _did）= {d_r.params.get('_did',np.nan):.4f} (SE={d_r.std_errors.get('_did',np.nan):.4f}) ｜ {post_note}")
                        rows=[]
                        label_map={'_treat':'Treat（处理组）','_post':'Post（政策后）','_did':'Treat×Post（DID）'}
                        for v in model_terms:
                            b=d_r.params.get(v,np.nan); se=d_r.std_errors.get(v,np.nan)
                            t=abs(b/se) if se>0 else 0; pv=2*(1-stats.t.cdf(t,df=len(td)-len(X_did.columns)))
                            s='***' if pv<0.01 else ('**' if pv<0.05 else ('*' if pv<0.1 else ''))
                            rows.append({'变量':label_map.get(v,v),'系数':f"{b:.4f}{s}",'SE':f"({se:.4f})"})
                        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
                        user_ctrls=[v for v in control_terms if not v.startswith('_')]
                        ctrl_text=(' '.join(user_ctrls)+' ') if user_ctrls else ''
                        do=(f"* === 鹈鹕回归 (c)2026 · 仅供参考 · 不构成统计建议 ===\n* DID: {model_line}\n\nuse \"data.dta\", clear\n"
                            f"xtset {id_col} {tc_col}\n{post_stata}\ngen treat={did_var}\ngen did=treat*post\n"
                            f"* treat 通常会被个体固定效应吸收，估计式中保留 post 和 did\n"
                            f"reghdfe {y_col} {ctrl_text}post did, absorb({id_col}) vce(robust)\n")
                        with st.expander("Stata 复现代码"): st.code(do,language='stata')
                        dl1,dl2=st.columns(2)
                        dl1.download_button("下载 .do",do,file_name=f"{cx}_did.do",key=f'dl_{cx}_did_v7')
                        did_followup_panel(td.copy(),control_terms,id_col,tc_col,post_stata,f'{cx}_did', 'DID')
                    except Exception as e: st.error(f"DID 失败：{e}")

                elif is_panel and 'PSM-DID' in model_sel:
                    id_col=st.session_state._id_col; tc_col=st.session_state._tc_col
                    did_cols=list(dict.fromkeys([id_col,tc_col,y_col,did_var]+Xv0))
                    if did_post_var and did_post_var not in did_cols: did_cols.append(did_post_var)
                    td=sub[did_cols].dropna().copy()
                    td[id_col]=td[id_col].astype(str)
                    if did_post_var and did_post_var in td.columns:
                        td['_post']=td[did_post_var].astype(float)
                        post_note=f"Post 使用已有变量 {did_post_var}"
                        post_stata=f"gen post={did_post_var}"
                    else:
                        cutoff=did_policy_time if did_policy_time is not None else td[tc_col].median()
                        td['_post']=(td[tc_col]>=cutoff).astype(float)
                        post_note=f"Post=({tc_col}>={cutoff})"
                        post_stata=f"gen post=({tc_col}>={cutoff})"
                    td['_treat']=td[did_var].astype(float)
                    td['_did']=td['_treat']*td['_post']
                    did_terms=['_treat','_post','_did']
                    control_terms=[v for v in Xv0 if v not in did_terms]
                    matched_note=""
                    try:
                        cov_cols=[v for v in control_terms if v in td.columns]
                        if not cov_cols:
                            st.error("PSM-DID 至少需要 1 个匹配变量/控制变量。")
                            st.stop()
                        from sklearn.preprocessing import StandardScaler
                        from sklearn.linear_model import LogisticRegression
                        from sklearn.neighbors import NearestNeighbors as NN
                        pre=td[td['_post']==0].copy()
                        pre_unit=pre.groupby(id_col)[cov_cols+[did_var]].mean().dropna()
                        if pre_unit[did_var].nunique()<2:
                            st.error("政策前样本中处理组变量没有两类，无法做 PSM-DID。")
                            st.stop()
                        treated_ids=pre_unit[pre_unit[did_var]>=0.5].index.astype(str).tolist()
                        control_ids=pre_unit[pre_unit[did_var]<0.5].index.astype(str).tolist()
                        if not treated_ids or not control_ids:
                            st.error("政策前样本需要同时包含处理组和对照组。")
                            st.stop()
                        X_sc=StandardScaler().fit_transform(pre_unit[cov_cols])
                        D_p=(pre_unit[did_var]>=0.5).astype(int).values
                        ps=LogisticRegression(C=1e6,max_iter=1000).fit(X_sc,D_p).predict_proba(X_sc)[:,1]
                        ps_s=pd.Series(ps,index=pre_unit.index.astype(str))
                        nn=NN(n_neighbors=1).fit(ps_s.loc[control_ids].values.reshape(-1,1))
                        _,idx=nn.kneighbors(ps_s.loc[treated_ids].values.reshape(-1,1))
                        matched_controls=[control_ids[i[0]] for i in idx]
                        keep_ids=set(treated_ids+matched_controls)
                        td=td[td[id_col].astype(str).isin(keep_ids)].copy()
                        matched_note=f"匹配后样本：处理组 {len(treated_ids)} 个体，匹配控制组 {len(set(matched_controls))} 个体"
                    except Exception as match_e:
                        st.error(f"PSM 匹配失败：{str(match_e)[:80]}")
                        st.stop()
                    td_idx=td.set_index([id_col,tc_col])
                    model_terms=unique_keep_order([v for v in control_terms+['_post','_did'] if v in td_idx.columns and td_idx[v].nunique(dropna=True)>1])
                    if '_did' not in model_terms:
                        st.error("PSM-DID 交互项没有变化，无法估计。")
                        st.stop()
                    X_did=sm.add_constant(td_idx[model_terms])
                    try:
                        d_r=safe_panel_fit(td_idx[y_col],X_did,entity_effects=True,time_effects=False)
                        st.info(f"PSM-DID 估计量（Treat×Post）= {d_r.params.get('_did',np.nan):.4f} (SE={d_r.std_errors.get('_did',np.nan):.4f}) ｜ {post_note}")
                        st.caption(matched_note)
                        rows=[]; label_map={'_treat':'Treat（处理组）','_post':'Post（政策后）','_did':'Treat×Post（PSM-DID）'}
                        for v in model_terms:
                            b=d_r.params.get(v,np.nan); se=d_r.std_errors.get(v,np.nan)
                            t=abs(b/se) if se>0 else 0; pv=2*(1-stats.t.cdf(t,df=len(td)-len(X_did.columns)))
                            s='***' if pv<0.01 else ('**' if pv<0.05 else ('*' if pv<0.1 else ''))
                            rows.append({'变量':label_map.get(v,v),'系数':f"{b:.4f}{s}",'SE':f"({se:.4f})"})
                        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
                        user_ctrls=[v for v in control_terms if not v.startswith('_')]
                        ctrl_text=(' '.join(user_ctrls)+' ') if user_ctrls else ''
                        do=(f"* === 鹈鹕回归 (c)2026 · 仅供参考 · 不构成统计建议 ===\n* PSM-DID: {model_line}\n\nuse \"data.dta\", clear\n"
                            f"xtset {id_col} {tc_col}\n{post_stata}\ngen treat={did_var}\ngen did=treat*post\n"
                            f"* 可先用 psmatch2/teffects psmatch 在政策前样本匹配，再保留匹配样本\n"
                            f"* treat 通常会被个体固定效应吸收，估计式中保留 post 和 did\n"
                            f"reghdfe {y_col} {ctrl_text}post did, absorb({id_col}) vce(robust)\n")
                        with st.expander("Stata 复现代码"): st.code(do,language='stata')
                        dl1,dl2=st.columns(2)
                        dl1.download_button("下载 .do",do,file_name="psm_did.do",key='dl_psm_did_v8')
                        dl2.download_button("下载 .csv",pd.DataFrame(rows).to_csv(index=False),file_name="psm_did.csv",mime="text/csv",key='csv_psm_did_v8')
                        did_followup_panel(td.copy(),control_terms,id_col,tc_col,post_stata,'psm_did', 'PSM-DID')
                    except Exception as e: st.error(f"PSM-DID 失败：{e}")

                elif 'RDD' in model_sel:
                    bw_note='全样本' if not rdd_bandwidth else f'|running-cutoff|≤{rdd_bandwidth}'
                    label_map={'_rdd_treat':'断点处理项','_rdd_z':'断点工具变量 Z','_rdd_running_c':'断点距离','_rdd_inter':'断点两侧斜率差','_rdd_running_c2':'断点距离²','_rdd_inter2':'二阶斜率差'}
                    if rdd_design.startswith('Fuzzy'):
                        rdd_ctrls=[v for v in Xv0 if v!=rdd_treatment]
                        exog_vars=[v for v in rdd_ctrls if v in sub.columns and v!='_rdd_z']
                        td=sub[[y_col,rdd_treatment,'_rdd_z']+exog_vars].replace([np.inf,-np.inf],np.nan).dropna()
                        exog=sm.add_constant(td[exog_vars]) if exog_vars else sm.add_constant(pd.DataFrame(index=td.index))
                        iv_m=IV2SLS(td[y_col],exog,td[[rdd_treatment]],td[['_rdd_z']]).fit(cov_type='robust')
                        fs=OLS(td[rdd_treatment],sm.add_constant(td[exog_vars+['_rdd_z']])).fit(cov_type='HC1')
                        rf=OLS(td[y_col],sm.add_constant(td[exog_vars+['_rdd_z']])).fit(cov_type='HC1')
                        b=iv_m.params.get(rdd_treatment,np.nan); se=iv_m.std_errors.get(rdd_treatment,np.nan); pv=iv_m.pvalues.get(rdd_treatment,np.nan)
                        fs_b=fs.params.get('_rdd_z',np.nan); fs_p=fs.pvalues.get('_rdd_z',np.nan)
                        rf_b=rf.params.get('_rdd_z',np.nan); rf_p=rf.pvalues.get('_rdd_z',np.nan)
                        rows=[
                            {'估计量':'LATE / 2SLS','变量':rdd_treatment,'系数':f"{b:.4f}",'SE':f"({se:.4f})",'p':f"{pv:.4f}"},
                            {'估计量':'第一阶段跳跃','变量':'Z → D','系数':f"{fs_b:.4f}",'SE':f"({fs.bse.get('_rdd_z',np.nan):.4f})",'p':f"{fs_p:.4f}"},
                            {'估计量':'简化式跳跃','变量':'Z → Y','系数':f"{rf_b:.4f}",'SE':f"({rf.bse.get('_rdd_z',np.nan):.4f})",'p':f"{rf_p:.4f}"}
                        ]
                        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
                        side_counts=td['_rdd_z'].value_counts()
                        st.caption(f"Fuzzy RD｜断点变量={rdd_running}｜cutoff={rdd_cutoff}｜D={rdd_treatment}｜{bw_note}｜左侧/右侧样本={int(side_counts.get(0,0))}/{int(side_counts.get(1,0))}｜N={len(td)}")
                        user_ctrls=[v for v in exog_vars if not v.startswith('_rdd_')]
                        do=f"* === 鹈鹕回归 (c)2026 · Fuzzy RD / 2SLS ===\n* LATE: {y_col} <- {rdd_treatment}, instrumented by cutoff Z\n\nuse \"data.dta\", clear\n"
                        do+=f"gen rdd_running_c={rdd_running}-{rdd_cutoff}\ngen rdd_z=({rdd_running}>={rdd_cutoff})\ngen rdd_inter=rdd_z*rdd_running_c\n"
                        if '二阶' in rdd_order:
                            do+="gen rdd_running_c2=rdd_running_c^2\ngen rdd_inter2=rdd_z*rdd_running_c2\n"
                        if rdd_bandwidth and rdd_bandwidth>0:
                            do+=f"keep if abs(rdd_running_c)<={rdd_bandwidth}\n"
                        stata_exog='rdd_running_c rdd_inter'
                        if '二阶' in rdd_order: stata_exog+=' rdd_running_c2 rdd_inter2'
                        if user_ctrls: stata_exog+=' '+' '.join(user_ctrls)
                        do+=f"* 第一阶段\nreg {rdd_treatment} rdd_z {stata_exog}, robust\n"
                        do+=f"* 简化式\nreg {y_col} rdd_z {stata_exog}, robust\n"
                        do+=f"* Fuzzy RD: 2SLS / LATE\nivregress 2sls {y_col} {stata_exog} ({rdd_treatment}=rdd_z), robust\nestat firststage\n"
                        out_csv=pd.DataFrame(rows).to_csv(index=False)
                    else:
                        td=sub[[y_col]+Xv0].dropna(); Xd=sm.add_constant(td[Xv0])
                        m=OLS(td[y_col].values,Xd).fit(cov_type='HC1')
                        rows=[]
                        for j,vn in enumerate(['const']+Xv0):
                            b=m.params[vn]; se=m.bse[vn]; t=abs(b/se) if se>0 else 0; pv=m.pvalues[vn]
                            s='***' if pv<0.01 else ('**' if pv<0.05 else ('*' if pv<0.1 else ''))
                            rows.append({'变量':label_map.get(vn,vn),'系数':f"{b:.4f}{s}",'SE':f"({se:.4f})",'t':f"{t:.2f}",'p':f"{pv:.4f}"})
                        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
                        side_counts=td['_rdd_treat'].value_counts()
                        st.caption(f"Sharp RD｜断点变量={rdd_running}｜cutoff={rdd_cutoff}｜{bw_note}｜左侧/右侧样本={int(side_counts.get(0,0))}/{int(side_counts.get(1,0))}｜N={len(td)}｜R²={m.rsquared:.4f}")
                        rdd_ctrls=[v for v in Xv0 if v!='_rdd_treat']
                        do=f"* === 鹈鹕回归 (c)2026 · Sharp RD ===\n* RDD: {y_col} around cutoff {rdd_cutoff}\n\nuse \"data.dta\", clear\n"
                        do+=f"gen rdd_running_c={rdd_running}-{rdd_cutoff}\ngen rdd_treat=({rdd_running}>={rdd_cutoff})\ngen rdd_inter=rdd_treat*rdd_running_c\n"
                        if '二阶' in rdd_order:
                            do+="gen rdd_running_c2=rdd_running_c^2\ngen rdd_inter2=rdd_treat*rdd_running_c2\n"
                        if rdd_bandwidth and rdd_bandwidth>0:
                            do+=f"keep if abs(rdd_running_c)<={rdd_bandwidth}\n"
                        stata_vars='rdd_treat rdd_running_c rdd_inter'
                        if '二阶' in rdd_order: stata_vars+=' rdd_running_c2 rdd_inter2'
                        user_ctrls=[v for v in rdd_ctrls if not v.startswith('_rdd_')]
                        if user_ctrls: stata_vars+=' '+' '.join(user_ctrls)
                        do+=f"reg {y_col} {stata_vars}, robust\n"
                        out_csv=pd.DataFrame(rows).to_csv(index=False)
                    with st.expander("Stata 复现代码"): st.code(do,language='stata')
                    dl1,dl2=st.columns(2)
                    dl1.download_button("下载 .do",do,file_name=f"rdd_{y_col}.do",key=f'dlrdd_{cx}_v2')
                    dl2.download_button("下载 .csv",out_csv,file_name=f"rdd_{y_col}.csv",mime="text/csv",key=f'csrdd_{cx}_v2')

                elif model_sel.startswith('OLS') or 'Pooled' in model_sel:
                    td=sub[[y_col]+Xv0].dropna(); Xd=sm.add_constant(td[Xv0])
                    cov_t='HC1' if use_rob else 'nonrobust'
                    m=OLS(td[y_col].values,Xd).fit(cov_type=cov_t)
                    rows=[]
                    for j,vn in enumerate(['const']+Xv0):
                        b=m.params[vn]; se=m.bse[vn]; t=abs(b/se) if se>0 else 0; pv=m.pvalues[vn]
                        s='***' if pv<0.01 else ('**' if pv<0.05 else ('*' if pv<0.1 else ''))
                        rows.append({'变量':vn,'系数':f"{b:.4f}{s}",'SE':f"({se:.4f})",'t':f"{t:.2f}",'p':f"{pv:.4f}"})
                    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
                    se_label='聚类稳健SE' if use_cl else ('异方差稳健SE (HC1)' if use_rob else '普通SE')
                    st.caption(f"OLS（{se_label}）｜N={len(td)}｜R²={m.rsquared:.4f}｜adj R²={m.rsquared_adj:.4f}｜F={m.fvalue:.2f}(p={m.f_pvalue:.4f})")
                    if use_cl and cluster_col: stata_se=f', vce(cluster {cluster_col})'
                    elif use_rob: stata_se=', robust'
                    else: stata_se=''
                    do=f"* === 鹈鹕回归 (c)2026 · 仅供参考 · 不构成统计建议 ===\n* {model_line}\n\nuse \"data.dta\", clear\nreg {y_col} {cx} {' '.join(ctrls)}{stata_se}\n"
                    with st.expander("Stata 复现代码"): st.code(do,language='stata')
                    dl1,dl2=st.columns(2)
                    dl1.download_button("下载 .do",do,file_name=f"{cx}_ols.do",key=f'dlo_{cx}_v7')
                    dl2.download_button("下载 .csv",pd.DataFrame(rows).to_csv(index=False),file_name=f"{cx}_ols.csv",mime="text/csv",key=f'cso_{cx}_v7')

                elif 'Ordered' not in model_sel and ('Logit' in model_sel or 'Probit' in model_sel or 'LPM' in model_sel):
                    td=sub[[y_col]+Xv0].dropna(); Xd=sm.add_constant(td[Xv0]); y_d=td[y_col]
                    rows=[]; fit_lines=[]
                    run_lg='Logit' in model_sel or '+' in model_sel; run_pr='Probit' in model_sel or '+' in model_sel
                    run_lpm='LPM' in model_sel
                    if use_cl and cluster_col: stata_se=f', vce(cluster {cluster_col})'
                    elif use_rob: stata_se=', robust'
                    else: stata_se=''
                    do=f"* === 鹈鹕回归 (c)2026 · 仅供参考 · 不构成统计建议 ===\n* {model_line}\n\nuse \"data.dta\", clear\n\n"
                    if run_lg:
                        do+=f"logit {y_col} {cx} {' '.join(ctrls)}{stata_se}\n"
                        try:
                            lf=sm.Logit(y_d,Xd).fit(disp=0)
                            for j,vn in enumerate(['const']+Xv0):
                                b=lf.params[vn]; se=lf.bse[vn]; t=abs(b/se) if se>0 else 0; pv=2*(1-stats.norm.cdf(t))
                                s='***' if pv<0.01 else ('**' if pv<0.05 else ('*' if pv<0.1 else ''))
                                rows.append({'变量':vn,'Logit 系数':f"{b:.4f}{s}",'Logit SE':f"({se:.4f})"})
                            fit_lines.append(f"Logit: Pseudo R²={lf.prsquared:.4f}, LL={lf.llf:.2f}")
                        except Exception as e: st.error(f"Logit 拟合失败：{e} ({type(e).__name__})")
                    if run_pr:
                        do+=f"probit {y_col} {cx} {' '.join(ctrls)}{stata_se}\n"
                        try:
                            pf=sm.Probit(y_d,Xd).fit(disp=0)
                            for j,vn in enumerate(['const']+Xv0):
                                b=pf.params[vn]; se=pf.bse[vn]; t=abs(b/se) if se>0 else 0; pv=2*(1-stats.norm.cdf(t))
                                s='***' if pv<0.01 else ('**' if pv<0.05 else ('*' if pv<0.1 else ''))
                                if run_lg:
                                    for rr in rows:
                                        if rr['变量']==vn: rr['Probit 系数']=f"{b:.4f}{s}"; rr['Probit SE']=f"({se:.4f})"
                                else: rows.append({'变量':vn,'Probit 系数':f"{b:.4f}{s}",'Probit SE':f"({se:.4f})"})
                            fit_lines.append(f"Probit: Pseudo R²={pf.prsquared:.4f}, LL={pf.llf:.2f}")
                        except Exception as e: st.error(f"Probit 拟合失败：{e}\n\n```\n{traceback.format_exc()}\n```\nstatsmodels 版本: {sm.__version__}")
                    if run_lpm:
                        do+=f"reg {y_col} {cx} {' '.join(ctrls)}{stata_se}\n"
                        lm=OLS(y_d,Xd).fit(cov_type='HC1')
                        for j,vn in enumerate(['const']+Xv0):
                            b=lm.params[vn]; se=lm.bse[vn]; pv=lm.pvalues[vn]
                            s='***' if pv<0.01 else ('**' if pv<0.05 else ('*' if pv<0.1 else ''))
                            rows.append({'变量':vn,'LPM 系数':f"{b:.4f}{s}",'LPM SE':f"({se:.4f})"})
                        fit_lines.append(f"LPM: R²={lm.rsquared:.4f}")
                    do+="\n* 边际效应\nmargins, dydx(*) post\n"
                    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
                    st.caption(f"*** p<0.01, ** p<0.05, * p<0.10\nN={len(td)}\n"+'\n'.join(fit_lines))
                    with st.expander("Stata 复现代码"): st.code(do,language='stata')
                    dl1,dl2=st.columns(2)
                    dl1.download_button("下载 .do",do,file_name=f"{cx}_binary.do",key=f'dlb_{cx}_v7')
                    dl2.download_button("下载 .csv",pd.DataFrame(rows).to_csv(index=False),file_name=f"{cx}_binary.csv",mime="text/csv",key=f'csb_{cx}_v7')

                elif 'Ordered' in model_sel:
                    td=sub[[y_col]+Xv0].dropna(); Xd=td[Xv0]; y_d=td[y_col].astype(int)
                    dist='probit' if 'Probit' in model_sel else 'logit'
                    stcmd='oprobit' if 'Probit' in model_sel else 'ologit'
                    do=f"* === 鹈鹕回归 (c)2026 · 仅供参考 · 不构成统计建议 ===\n* {model_line}\n\nuse \"data.dta\", clear\n{stcmd} {y_col} {cx} {' '.join(ctrls)}, robust\nmargins, dydx(*) post\n"
                    try:
                        om=OrderedModel(y_d,Xd,distr=dist).fit(disp=0)
                        rows=[]
                        for j,vn in enumerate(Xv0):
                            b=om.params[vn]; se=om.bse[vn]; t=abs(b/se) if se>0 else 0; pv=2*(1-stats.norm.cdf(t))
                            s='***' if pv<0.01 else ('**' if pv<0.05 else ('*' if pv<0.1 else ''))
                            rows.append({'变量':vn,'系数':f"{b:.4f}{s}",'SE':f"({se:.4f})",'z':f"{t:.2f}"})
                        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
                        st.caption(f"N={len(td)}｜Pseudo R²={om.prsquared:.4f}｜LL={om.llf:.2f}")
                    except Exception as e: st.error(f"Ordered 失败：{e}")
                    with st.expander("Stata 复现代码"): st.code(do,language='stata')

                elif 'Poisson' in model_sel or '负二项' in model_sel:
                    td=sub[[y_col]+Xv0].dropna(); Xd=sm.add_constant(td[Xv0]); y_d=td[y_col]
                    fam=Poisson() if 'Poisson' in model_sel else NegativeBinomial()
                    cov_t='HC0' if use_rob else 'nonrobust'
                    stcmd='poisson' if 'Poisson' in model_sel else 'nbreg'
                    if use_cl and cluster_col: stata_se=f', vce(cluster {cluster_col})'
                    elif use_rob: stata_se=', robust'
                    else: stata_se=''
                    do=f"* === 鹈鹕回归 (c)2026 · 仅供参考 · 不构成统计建议 ===\n* {model_line}\n\nuse \"data.dta\", clear\n{stcmd} {y_col} {cx} {' '.join(ctrls)}{stata_se}\nmargins, dydx(*) post\n"
                    try:
                        gm=GLM(y_d,Xd,family=fam).fit(cov_type=cov_t)
                        rows=[]
                        for j,vn in enumerate(['const']+Xv0):
                            b=gm.params[vn]; se=gm.bse[vn]; t=abs(b/se) if se>0 else 0; pv=gm.pvalues[vn]
                            s='***' if pv<0.01 else ('**' if pv<0.05 else ('*' if pv<0.1 else ''))
                            rows.append({'变量':vn,'系数':f"{b:.4f}{s}",'SE':f"({se:.4f})",'z':f"{t:.2f}"})
                        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
                        se_label='聚类稳健SE' if use_cl else ('异方差稳健SE' if use_rob else '普通SE')
                        st.caption(f"N={len(td)}｜Pseudo R²={1-gm.llf/gm.llnull:.4f}｜LL={gm.llf:.2f}｜SE: {se_label}")
                    except Exception as e: st.error(f"计数模型失败：{e}")
                    with st.expander("Stata 复现代码"): st.code(do,language='stata')

                elif 'Tobit' in model_sel:
                    td=sub[[y_col]+Xv0].dropna(); Xd=sm.add_constant(td[Xv0]).values; y_d=td[y_col].values
                    left_val=td[y_col].min() if (td[y_col]==td[y_col].min()).mean()>0.05 else None
                    tr=tobit_mle(y_d,Xd,left=left_val)
                    ll_opt=f"ll({left_val})" if left_val else ""
                    do=f"* === 鹈鹕回归 (c)2026 · 仅供参考 · 不构成统计建议 ===\n* {model_line}\n\nuse \"data.dta\", clear\ntobit {y_col} {cx} {' '.join(ctrls)}, {ll_opt} vce(robust)\nmargins, dydx(*) predict(ystar(0,.)) post\n"
                    if tr['converged']:
                        rows=[]
                        for j,vn in enumerate(['const']+Xv0):
                            b=tr['params'][j]; se=tr['se'][j]; z=abs(b/se) if se>0 else 0; pv=2*(1-stats.norm.cdf(z))
                            s='***' if pv<0.01 else ('**' if pv<0.05 else ('*' if pv<0.1 else ''))
                            rows.append({'变量':vn,'系数':f"{b:.4f}{s}",'SE':f"({se:.4f})"})
                        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
                        st.caption(f"Tobit MLE｜N={len(td)}｜σ={tr['sigma']:.4f}｜LL={tr['llf']:.2f}")
                    else: st.error(f"Tobit 未收敛：{tr.get('error','')}")
                    with st.expander("Stata 复现代码"): st.code(do,language='stata')

                elif 'Heckman' in model_sel:
                    if not heckman_sel: st.error("需要指定选择变量")
                    else:
                        td=sub[[y_col,heckman_sel]+Xv0].dropna()
                        sel_ind=td[heckman_sel].values
                        y_raw=td[y_col].values.copy()
                        y_h=np.where(sel_ind==1,y_raw,np.nan)  # mask unselected
                        X_h=td[Xv0].values
                        # exclusion variables: pick variables in ctrl_pool not in current Xv0
                        excl_candidates=[c for c in ctrl_pool if c not in Xv0][:3]
                        Z_vars=np.column_stack([X_h,td[excl_candidates].values]) if excl_candidates else X_h
                        do=f"* === 鹈鹕回归 (c)2026 · 仅供参考 · 不构成统计建议 ===\n* {model_line}\n\nuse \"data.dta\", clear\nheckman {y_col} {cx} {' '.join(ctrls)}, select({heckman_sel}={cx} {' '.join(ctrls)}) twostep\n"
                        try:
                            hr=heckman_two_step(y_h,X_h,Z_vars)
                            rows=[]
                            for j,vn in enumerate(Xv0):
                                b=hr['params'][j+1]; se=hr['se'][j+1]; t=abs(b/se) if se>0 else 0; pv=2*(1-stats.t.cdf(t,df=hr['n_obs']))
                                s='***' if pv<0.01 else ('**' if pv<0.05 else ('*' if pv<0.1 else ''))
                                rows.append({'变量':vn,'系数':f"{b:.4f}{s}",'SE':f"({se:.4f})"})
                            st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
                            st.caption(f"Heckman 两步法｜观测={hr['n_obs']}/{hr['n_total']}｜逆Mills比={hr['imr_coef']:.4f}(p={hr['imr_p']:.4f})｜{'选择偏差显著' if hr['imr_p']<0.05 else '选择偏差不显著'}")
                        except Exception as e: st.error(f"Heckman 失败：{e}")
                        with st.expander("Stata 复现代码"): st.code(do,language='stata')

                elif 'PSM' in model_sel:
                    if not did_var: st.error("需要处理变量")
                    else:
                        td=sub[[y_col,did_var]+Xv0].dropna()
                        D=td[did_var].values; y_p=td[y_col].values; X_p=td[Xv0].values
                        do=f"* === 鹈鹕回归 (c)2026 · 仅供参考 · 不构成统计建议 ===\n* PSM: {model_line}\n\nuse \"data.dta\", clear\npsmatch2 {did_var} {cx} {' '.join(ctrls)}, outcome({y_col}) logit ate att\npstest, both graph\n"
                        try:
                            pr=psm_att(y_p,D,X_p,method='nearest',k=1)
                            st.success(f"PSM ATT = {pr['ATT']:.4f}（{pr['method']} matching, {pr['n_treated']} treated, {pr['n_control']} control）")
                        except Exception as e: st.error(f"PSM 失败：{e}")
                        with st.expander("Stata 复现代码"): st.code(do,language='stata')

                elif 'IV' in model_sel:
                    if not iv_endog or len(iv_insts)<1: st.error("需要指定内生变量和工具变量")
                    else:
                        exog_vars=[v for v in Xv0 if v!=iv_endog]
                        td=sub[[y_col,iv_endog]+exog_vars+iv_insts].dropna()
                        do=f"* === 鹈鹕回归 (c)2026 · 仅供参考 · 不构成统计建议 ===\n* IV/2SLS: {model_line}\n\nuse \"data.dta\", clear\nivregress 2sls {y_col} {' '.join(exog_vars)} ({iv_endog}={' '.join(iv_insts)}), robust\nestat firststage\nestat overid\n"
                        try:
                            iv_m=IV2SLS(td[y_col],sm.add_constant(td[exog_vars]),td[iv_endog],sm.add_constant(td[iv_insts])).fit()
                            rows=[]
                            for v in exog_vars+[iv_endog]:
                                b=iv_m.params.get(v,np.nan); se=iv_m.std_errors.get(v,np.nan)
                                t=abs(b/se) if se>0 else 0; pv=2*(1-stats.t.cdf(t,df=len(td)))
                                s='***' if pv<0.01 else ('**' if pv<0.05 else ('*' if pv<0.1 else ''))
                                rows.append({'变量':v,'系数':f"{b:.4f}{s}",'SE':f"({se:.4f})"})
                            st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
                            st.caption(f"IV/2SLS ｜N={len(td)}｜第一阶段F={iv_m.first_stage.diagnostics.get('f_stat',np.nan):.2f}")
                        except Exception as e: st.error(f"IV 失败：{e}")
                        with st.expander("Stata 复现代码"): st.code(do,language='stata')

                # ═══ ANOVA/组间比较 ═══
                elif 'ANOVA' in model_sel:
                    if not group_var: st.error("需要选择分组变量")
                    else:
                        td=sub[[y_col]+Xv0].dropna()
                        grp_dums=pd.get_dummies(df_aug[group_var],prefix=group_var,drop_first=True,dtype=float)
                        grp_dums=grp_dums.loc[td.index]; d_names=list(grp_dums.columns)
                        X_all=sm.add_constant(np.column_stack([td[Xv0].values,grp_dums.values]))
                        all_labels=['const']+Xv0+d_names
                        try:
                            m=OLS(td[y_col].values,X_all).fit()
                            d_idx=[i for i,lb in enumerate(all_labels) if lb in d_names]
                            f_stat,f_pval=np.nan,np.nan
                            if d_idx:
                                R=np.zeros((len(d_idx),len(all_labels)))
                                for ri,di in enumerate(d_idx): R[ri,di]=1
                                f_stat=m.f_test(R).statistic[0][0]
                                f_pval=m.f_test(R).pvalue
                            grp_means=td.groupby(df_aug.loc[td.index,group_var])[y_col].agg(['mean','std','count'])
                            rows=[{'变量':lb,'系数':f"{m.params[i]:.4f}",'SE':f"({m.bse[i]:.4f})",
                                   't':f"{abs(m.params[i]/m.bse[i]) if m.bse[i]>0 else 0:.2f}",
                                   'p':f"{m.pvalues[i]:.4f}"} for i,lb in enumerate(all_labels)]
                            st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
                            st.success(f"F({len(d_idx)},{int(m.df_resid)})={f_stat:.2f}, p={f_pval:.4f} -> {'组间差异显著' if f_pval<0.05 else '组间差异不显著'}")
                            st.caption(f"N={len(td)} | R2={m.rsquared:.4f} | 基准组：{sorted(df_aug[group_var].dropna().unique())[0]}")
                            with st.expander("各组均值"): st.dataframe(grp_means,use_container_width=True)
                            do=f"* === 鹈鹕回归 (c)2026 · 仅供参考 · 不构成统计建议 ===\n* ANOVA: {y_col} ~ i.{group_var}"
                            if ctrls: do+=f" + {' '.join(ctrls)}"
                            do+=f"\n\nuse \"data.dta\", clear\nreg {y_col} i.{group_var}"
                            if ctrls: do+=f" {' '.join(ctrls)}"
                            do+=", robust\ntestparm i."+group_var+"\n"
                            with st.expander("Stata 复现代码"): st.code(do,language='stata')
                        except Exception as e: st.error(f"ANOVA 失败：{e}")

                # ═══ 交互效应（分类x连续） ═══
                elif '交互效应' in model_sel:
                    if not group_var: st.error("需要选择分组变量")
                    else:
                        td=sub[[y_col]+Xv0].dropna()
                        grp_dums=pd.get_dummies(df_aug[group_var],prefix=group_var,drop_first=True,dtype=float)
                        grp_dums=grp_dums.loc[td.index]; d_names=list(grp_dums.columns)
                        inter_terms=[]; inter_labels=[]
                        for dn in d_names:
                            inter_terms.append(grp_dums[dn].values*td[cx].values)
                            inter_labels.append(f"{cx}x{dn}")
                        X_all=np.column_stack([td[Xv0].values,grp_dums.values]+inter_terms)
                        X_all=sm.add_constant(X_all)
                        all_labels=['const']+Xv0+d_names+inter_labels
                        try:
                            m=OLS(td[y_col].values,X_all).fit()
                            rows=[{'变量':lb,'系数':f"{m.params[i]:.4f}",'SE':f"({m.bse[i]:.4f})",
                                   'p':f"{m.pvalues[i]:.4f}"} for i,lb in enumerate(all_labels)]
                            st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
                            inter_idx=[i for i,lb in enumerate(all_labels) if 'x' in lb and lb not in d_names]
                            if inter_idx:
                                R=np.zeros((len(inter_idx),len(all_labels)))
                                for ri,di in enumerate(inter_idx): R[ri,di]=1
                                f_int=m.f_test(R).statistic[0][0]; fp_int=m.f_test(R).pvalue
                                st.info(f"交互项联合 F({len(inter_idx)},{int(m.df_resid)})={f_int:.2f}, p={fp_int:.4f} -> {'斜率因组而异（交互效应显著）' if fp_int<0.05 else '斜率不因组而异'}")
                            st.caption(f"N={len(td)} | R2={m.rsquared:.4f} | 基准组：{sorted(df_aug[group_var].dropna().unique())[0]}")
                            do=f"* === 鹈鹕回归 (c)2026 · 仅供参考 · 不构成统计建议 ===\n* 交互效应: {y_col} ~ c.{cx}##i.{group_var}"
                            if ctrls: do+=f" + {' '.join(ctrls)}"
                            do+=f"\n\nuse \"data.dta\", clear\nreg {y_col} c.{cx}##i.{group_var}"
                            if ctrls: do+=f" {' '.join(ctrls)}"
                            do+=f", robust\nmargins {group_var}, dydx({cx})\nmarginsplot\n"
                            with st.expander("Stata 复现代码"): st.code(do,language='stata')
                        except Exception as e: st.error(f"交互效应失败：{e}")

                # ═══ 分组面板回归 ═══
                elif '分组面板' in model_sel:
                    if not group_var: st.error("需要选择分组变量")
                    else:
                        id_col=st.session_state._id_col; tc_col=st.session_state._tc_col
                        groups=sorted(df_aug[group_var].dropna().unique())
                        all_rows=[]
                        for grp in groups:
                            mask=df_aug[group_var]==grp
                            gsub=df_aug.loc[mask,[id_col,tc_col,y_col]+Xv0].dropna().copy()
                            gsub[id_col]=gsub[id_col].astype(str)
                            if gsub.duplicated(subset=[id_col,tc_col]).sum()>0:
                                gsub=gsub.groupby([id_col,tc_col]).mean().reset_index()
                            try:
                                g_idx=gsub.set_index([id_col,tc_col])
                                Xe=sm.add_constant(g_idx[Xv0])
                                fe_r=PanelOLS(g_idx[y_col],Xe,entity_effects=True,time_effects=True).fit()
                                re_r=RandomEffects(g_idx[y_col],Xe).fit()
                                common=[c for c in fe_r.params.index.intersection(re_r.params.index) if c!='const']
                                hs,hp=np.nan,np.nan
                                if common:
                                    d_p=fe_r.params.loc[common]-re_r.params.loc[common]
                                    Vv=fe_r.cov.loc[common,common].values-re_r.cov.loc[common,common].values
                                    try:
                                        hs=float(d_p.values.T@np.linalg.pinv(Vv)@d_p.values)
                                        hp=float(1-stats.chi2.cdf(hs,df=len(common)))
                                    except: pass
                                fb=fe_r.params.get(cx,np.nan); fse=fe_r.std_errors.get(cx,np.nan)
                                rb=re_r.params.get(cx,np.nan); rse=re_r.std_errors.get(cx,np.nan)
                                dec='FE' if (hp is not None and not np.isnan(hp) and hp<0.05) else 'RE'
                                all_rows.append({'分组':str(grp),'N':len(gsub),'个体':g_idx.index.get_level_values(0).nunique(),
                                    'FE beta':f"{fb:.4f}",'FE SE':f"({fse:.4f})",'RE beta':f"{rb:.4f}",'RE SE':f"({rse:.4f})",
                                    'Hausman p':f"{hp:.4f}" if not np.isnan(hp) else 'N/A','判断':dec,
                                    'R2':f"{fe_r.rsquared:.3f}" if dec=='FE' else f"{re_r.rsquared:.3f}"})
                            except Exception as e2:
                                all_rows.append({'分组':str(grp),'N':len(gsub),'个体':'-','FE beta':'失败','FE SE':str(e2)[:60]})
                        st.dataframe(pd.DataFrame(all_rows),use_container_width=True,hide_index=True)
                        st.caption(f"分组面板回归：按 {group_var} 分 {len(groups)} 组分别估计 FE+RE")
                        do=f"* === 鹈鹕回归 (c)2026 · 仅供参考 · 不构成统计建议 ===\n* 分组面板回归: by {group_var}\n\nuse \"data.dta\", clear\nxtset {id_col} {tc_col}\n"
                        for grp in groups:
                            grp_stata = f'"{grp}"' if isinstance(grp, str) else str(grp)
                            do+=f"\n* 分组: {group_var}={grp}\nreghdfe {y_col} {cx} {' '.join(ctrls)} if {group_var}=={grp_stata}, absorb({id_col} {tc_col})\n"
                        with st.expander("Stata 复现代码"): st.code(do,language='stata')


                # ═══ 中介效应（附加分析，可与任何基准模型叠加）═══
                if st.session_state.get('_add_med',False):
                    med_results=st.session_state.get('_med_results',{})
                    med_m=st.session_state.get('_med_m',None)
                    med_method=st.session_state.get('_med_method','温忠麟五步流程（2014）')
                    is_wen='温忠麟' in med_method
                    if med_results and cx in med_results:
                        cx_meds=med_results[cx]
                        st.subheader(f"中介效应：{cx} → {med_m} → {y_col}")
                        st.caption(f"方法：{med_method}｜在基准回归 top {len(cx_meds)} 个显著组合上运行")
                        med_tbl=[]
                        for mi,mr in enumerate(cx_meds):
                            cs=', '.join(mr['controls'][:3])
                            if len(mr['controls'])>3: cs+=f'...+{len(mr["controls"])-3}'
                            if is_wen:
                                med_tbl.append({'#':mi+1,'控制组合':cs,
                                    'c(总效应)':f"{mr['c']:.4f}",'a':f"{mr['a']:.4f}",
                                    'b':f"{mr['b']:.4f}",'c\'':f"{mr['cp']:.4f}",
                                    '间接ab':f"{mr['ab']:.4f}",'判断':mr['effect_type']})
                            else:
                                med_tbl.append({'#':mi+1,'控制组合':cs,
                                    'c(X→Y)':f"{mr['c']:.4f}{'***' if mr['c_p']<0.01 else ('**' if mr['c_p']<0.05 else ('*' if mr['c_p']<0.1 else ''))}",
                                    'a(X→M)':f"{mr['a']:.4f}{'***' if mr['a_p']<0.01 else ('**' if mr['a_p']<0.05 else ('*' if mr['a_p']<0.1 else ''))}",
                                    '判断':mr['judgement'][:20]+'...' if len(mr['judgement'])>20 else mr['judgement']})
                        st.dataframe(pd.DataFrame(med_tbl),use_container_width=True,hide_index=True)
                        chosen_ctrl_set=set(chosen['controls'])
                        best_med=None; best_overlap=-1
                        for mr in cx_meds:
                            ov=len(chosen_ctrl_set & set(mr['controls']))
                            if ov>best_overlap: best_overlap=ov; best_med=mr
                        if best_med:
                            st.divider(); st.caption(f"选中组合详情（控制：{', '.join(best_med['controls'][:4])}）")
                            if is_wen:
                                # ── 温忠麟 2014 五步流程展示 ──
                                st.markdown("**五步检验流程**")
                                s1='✓' if best_med['c_p']<0.05 else '✗'
                                st.caption(f"① 检验系数 c={best_med['c']:.4f} (p={best_med['c_p']:.4f}) {s1} → 按**{best_med['framework']}**立论")
                                s2a='✓' if best_med['a_p']<0.05 else '✗'; s2b='✓' if best_med['b_p']<0.05 else '✗'
                                st.caption(f"② 依次检验 a={best_med['a']:.4f} (p={best_med['a_p']:.4f}) {s2a}, b={best_med['b']:.4f} (p={best_med['b_p']:.4f}) {s2b} → {'都显著，间接效应成立' if best_med['ab_both_sig'] else '至少一个不显著，需Bootstrap'}")
                                if not best_med['ab_both_sig']:
                                    if best_med['boot_ci']:
                                        bci=best_med['boot_ci']; bs='✓ 显著' if best_med['boot_sig'] else '✗ 不显著'
                                        st.caption(f"③ Bootstrap 1000次：ab 的95%CI=[{bci[0]:.4f}, {bci[1]:.4f}] → {bs}")
                                    else:
                                        st.caption("③ Bootstrap 失败（样本量不足）")
                                s4='不显著' if best_med['cp_p']>=0.05 else '显著'
                                st.caption(f"④ 检验直接效应 c'={best_med['cp']:.4f} (p={best_med['cp_p']:.4f}) → {s4}")
                                if best_med['effect_type'] not in ['无中介','完全中介']:
                                    ab_sign='同号' if best_med['ab']*best_med['cp']>0 else '异号'
                                    st.caption(f"⑤ ab({best_med['ab']:.4f}) 与 c'({best_med['cp']:.4f}) {ab_sign}")
                                if best_med['effect_type'] in ['部分中介','完全中介']:
                                    es_str=f"，效应量={best_med['effect_size']:.3f}" if best_med['effect_size'] is not None else ''
                                    st.success(f"结论：{best_med['effect_type']}{es_str}")
                                elif best_med['effect_type']=='遮掩效应':
                                    st.warning(f"结论：{best_med['effect_type']}，|ab/c'|={best_med['effect_size']:.3f}")
                                else:
                                    st.warning(f"结论：{best_med['effect_type']}")
                                c1,c2,c3=st.columns(3)
                                with c1: st.metric("总效应 c",f"{best_med['c']:.4f}")
                                with c2: st.metric("直接效应 c'",f"{best_med['cp']:.4f}")
                                with c3: st.metric("间接效应 ab",f"{best_med['ab']:.4f}")
                                if best_med.get('boot_ci'):
                                    c4,c5=st.columns(2)
                                    with c4: st.caption(f"Bootstrap 95%CI: [{best_med['boot_ci'][0]:.4f}, {best_med['boot_ci'][1]:.4f}]")
                                    with c5: st.caption(f"R²: step1={best_med['r2_step1']:.3f}, step3={best_med['r2_step3']:.3f}")
                                else:
                                    st.caption(f"R²: step1={best_med['r2_step1']:.3f}, step2={best_med['r2_step2']:.3f}, step3={best_med['r2_step3']:.3f}")
                                do=f"* === 鹈鹕回归 (c)2026 · 仅供参考 · 不构成统计建议 ===\n"
                                do+=f"* 中介效应（温忠麟五步流程 2014）: {cx} → {med_m} → {y_col}\n\nuse \"data.dta\", clear\n"
                                do+=f"* 步骤1: 总效应\nreg {y_col} {cx} {' '.join(best_med['controls'])}, robust\n"
                                do+=f"* 步骤2: X→M 和 直接效应+中介\nreg {med_m} {cx} {' '.join(best_med['controls'])}, robust\n"
                                do+=f"reg {y_col} {cx} {med_m} {' '.join(best_med['controls'])}, robust\n"
                                do+=f"* 步骤3: Bootstrap 检验间接效应 ab\n"
                                do+=f"capture program drop bootmed\nprogram define bootmed, rclass\n"
                                do+=f"  reg {med_m} {cx} {' '.join(best_med['controls'])}\n"
                                do+=f"  local a=_b[{cx}]\n  reg {y_col} {cx} {med_m} {' '.join(best_med['controls'])}\n"
                                do+=f"  local b=_b[{med_m}]\n  return scalar indirect=`a'*`b'\nend\n"
                                do+=f"bootstrap r(indirect), reps(1000) seed(42): bootmed\nestat bootstrap, percentile bc\n"
                                with st.expander("Stata 复现代码"): st.code(do,language='stata')
                            else:
                                # ── 江艇 2022 渠道检验展示 ──
                                st.info("**渠道检验**：仅验证 D→M 前半段因果链条，不分解直接/间接效应。"
                                       "M 对 Y 的因果关系需由理论支撑（江艇, 2022）。")
                                c1,c2=st.columns(2)
                                with c1: st.metric("第1步: X→Y (总效应 c)",f"{best_med['c']:.4f}",
                                    f"p={best_med['c_p']:.4f} {'✓' if best_med['c_p']<0.05 else '✗'}")
                                with c2: st.metric("第2步: X→M (渠道 a)",f"{best_med['a']:.4f}",
                                    f"p={best_med['a_p']:.4f} {'✓' if best_med['a_p']<0.05 else '✗'}")
                                st.caption(f"R²: step1={best_med['r2_step1']:.3f}, step2={best_med['r2_step2']:.3f}")
                                st.caption(best_med.get('note',''))
                                if '成立' in best_med['judgement']: st.success(best_med['judgement'])
                                else: st.warning(best_med['judgement'])
                                do=f"* === 鹈鹕回归 (c)2026 · 仅供参考 · 不构成统计建议 ===\n"
                                do+=f"* 渠道检验（江艇 2022）: {cx} → {med_m} → {y_col}\n"
                                do+=f"* 注意：仅检验 D→M 前半段，不分解效应（M 内生导致分解不可信）\n\n"
                                do+=f"use \"data.dta\", clear\n"
                                do+=f"* 第1步: 总效应 D→Y 必须存在\nreg {y_col} {cx} {' '.join(best_med['controls'])}, robust\n"
                                do+=f"* 第2步: D→M 渠道必须显著\nreg {med_m} {cx} {' '.join(best_med['controls'])}, robust\n"
                                with st.expander("Stata 复现代码"): st.code(do,language='stata')
                    else:
                        st.info("未找到有效中介分析结果，请确保基准回归中有显著组合")

                # ═══ 调节效应（附加分析，可与任何基准模型叠加）═══
                if st.session_state.get('_add_mod',False):
                    mod_results=st.session_state.get('_mod_results',{})
                    mod_m=st.session_state.get('_mod_m',None)
                    if mod_results and cx in mod_results:
                        cx_mods=mod_results[cx]
                        st.subheader(f"调节效应：{cx} × {mod_m} → {y_col}")
                        st.caption(f"在基准回归 top {len(cx_mods)} 个显著组合上运行")
                        mod_tbl=[]
                        for mi,mr in enumerate(cx_mods):
                            cs=', '.join(mr['controls'][:3])
                            if len(mr['controls'])>3: cs+=f'...+{len(mr["controls"])-3}'
                            s='***' if mr['p_inter']<0.01 else ('**' if mr['p_inter']<0.05 else ('*' if mr['p_inter']<0.1 else ''))
                            mod_tbl.append({'#':mi+1,'控制组合':cs,
                                'X系数':f"{mr['b_x']:.4f}",'交互项':f"{mr['b_inter']:.4f}{s}",
                                '低M斜率':f"{mr['low_slope']:.4f}",'中M斜率':f"{mr['med_slope']:.4f}",
                                '高M斜率':f"{mr['high_slope']:.4f}",'N':mr['n']})
                        st.dataframe(pd.DataFrame(mod_tbl),use_container_width=True,hide_index=True)
                        chosen_ctrl_set=set(chosen['controls'])
                        best_mod=None; best_overlap=-1
                        for mr in cx_mods:
                            ov=len(chosen_ctrl_set & set(mr['controls']))
                            if ov>best_overlap: best_overlap=ov; best_mod=mr
                        if best_mod:
                            st.divider(); st.caption(f"选中组合的调节效应详情（控制：{', '.join(best_mod['controls'][:4])}）")
                            c1,c2,c3=st.columns(3)
                            with c1: st.metric(f"{cx} 主效应",f"{best_mod['b_x']:.4f}")
                            with c2: st.metric(f"{mod_m} 主效应",f"{best_mod['b_m']:.4f}")
                            with c3: st.metric(f"交互项 {cx}×{mod_m}",f"{best_mod['b_inter']:.4f}",
                                f"p={best_mod['p_inter']:.4f}")
                            st.divider(); st.caption(f"简单斜率分析（{mod_m} 均值±1SD）")
                            m_mu=best_mod['m_mean']; m_sd=best_mod['m_sd']
                            c4,c5,c6=st.columns(3)
                            with c4: st.metric(f"低 {mod_m} ({m_mu-m_sd:.2f})",f"{best_mod['low_slope']:.4f}",
                                f"t={best_mod['low_t']:.2f}")
                            with c5: st.metric(f"中 {mod_m} ({m_mu:.2f})",f"{best_mod['med_slope']:.4f}",
                                f"t={best_mod['med_t']:.2f}")
                            with c6: st.metric(f"高 {mod_m} ({m_mu+m_sd:.2f})",f"{best_mod['high_slope']:.4f}",
                                f"t={best_mod['high_t']:.2f}")
                            if best_mod['p_inter']<0.05: st.success(f"调节效应显著：{mod_m} 显著调节 {cx}→{y_col} 的关系")
                            else: st.warning("调节效应不显著")
                            do=f"* === 鹈鹕回归 (c)2026 · 仅供参考 · 不构成统计建议 ===\n* 调节效应: {cx} × {mod_m} → {y_col}\n\nuse \"data.dta\", clear\n"
                            do+=f"reg {y_col} c.{cx}##c.{mod_m} {' '.join(best_mod['controls'])}, robust\n"
                            do+=f"margins, dydx({cx}) at({mod_m}=({m_mu-m_sd:.2f} {m_mu:.2f} {m_mu+m_sd:.2f}))\nmarginsplot\n"
                            with st.expander("Stata 复现代码"): st.code(do,language='stata')
                    else:
                        st.info("未找到有效调节效应结果，请确保基准回归中有显著组合")

                # ═══ 异质性分析 ═══
                het_skip=['ANOVA','交互效应','分组面板']
                if any(sk in model_sel for sk in het_skip):
                    st.info(f"'{model_sel}' 模型暂不支持异质性分析，请换用 FE+RE/OLS/Logit 等回归模型")
                else:
                    with st.expander("异质性分析",expanded=False):
                        focus_cx=jcx_keys[0] if is_joint else cx
                        # 分组变量选择
                        excl_het=[y_col]
                        if not is_joint: excl_het.append(cx)
                        else: excl_het.extend(jcx_keys)
                        if is_panel:
                            id_col=st.session_state._id_col; tc_col=st.session_state._tc_col
                            if id_col not in excl_het: excl_het.append(id_col)
                            if tc_col not in excl_het: excl_het.append(tc_col)
                        het_candidates=[c for c in df_aug.columns if c not in excl_het and c not in st.session_state.get('_dummy_names',[])]
                        het_var=st.selectbox("分组变量",het_candidates,key=f'het_gvar_{cx}')
                        if het_var:
                            het_series=df_aug[het_var].dropna()
                            is_cat=het_series.dtype=='object' or het_series.nunique()<=10
                            het_methods=[]
                            if is_cat:
                                st.caption(f"分类变量 · {het_series.nunique()} 个组：{', '.join(str(v) for v in sorted(het_series.unique())[:10])}")
                                het_methods=['分类']
                            else:
                                st.caption("连续变量—选择分组方法（可多选）")
                                c1,c2=st.columns(2)
                                with c1:
                                    if st.checkbox("中位数分组",key=f'het_med_{cx}'): het_methods.append('中位数')
                                    if st.checkbox("平均数分组",key=f'het_mean_{cx}'): het_methods.append('平均数')
                                with c2:
                                    if st.checkbox("三分位数分组",key=f'het_tert_{cx}'): het_methods.append('三分位数')
                                    if st.checkbox("四分位数分组",key=f'het_quart_{cx}'): het_methods.append('四分位数')
                            if st.button("运行异质性分析",key=f'run_het_{cx}') and het_methods:
                                id_col=st.session_state.get('_id_col'); tc_col=st.session_state.get('_tc_col')
                                needed=[y_col]+Xv0+([] if not is_panel else [id_col,tc_col])
                                work_data=df_aug[needed+[het_var]].dropna()
                                groups=[]
                                if '分类' in het_methods:
                                    for v in sorted(het_series.dropna().unique()):
                                        groups.append((f"{het_var}={v}",work_data[work_data[het_var]==v]))
                                else:
                                    vals=work_data[het_var]
                                    if '中位数' in het_methods:
                                        med=vals.median(); groups.append((f"≥中位数({med:.2f})",work_data[vals>=med])); groups.append((f"<中位数({med:.2f})",work_data[vals<med]))
                                    if '平均数' in het_methods:
                                        mn=vals.mean(); groups.append((f"≥均值({mn:.2f})",work_data[vals>=mn])); groups.append((f"<均值({mn:.2f})",work_data[vals<mn]))
                                    if '三分位数' in het_methods:
                                        t1,t2=vals.quantile([1/3,2/3])
                                        groups.append((f"上1/3(>{t2:.2f})",work_data[vals>t2])); groups.append((f"中1/3({t1:.2f}-{t2:.2f})",work_data[(vals>t1)&(vals<=t2)])); groups.append((f"下1/3(<{t1:.2f})",work_data[vals<=t1]))
                                    if '四分位数' in het_methods:
                                        q1,q2,q3=vals.quantile([0.25,0.5,0.75])
                                        groups.append((f"Q4(>{q3:.2f})",work_data[vals>q3])); groups.append((f"Q3({q2:.2f}-{q3:.2f})",work_data[(vals>q2)&(vals<=q3)])); groups.append((f"Q2({q1:.2f}-{q2:.2f})",work_data[(vals>q1)&(vals<=q2)])); groups.append((f"Q1(<{q1:.2f})",work_data[vals<=q1]))
                                het_results=[]
                                for grp_label,grp_data in groups:
                                    ng=len(grp_data)
                                    if ng<30:
                                        het_results.append({'分组':grp_label,'N':ng,'系数':'—','SE':'—','t值':'—','p值':'—','R²':'—','备注':'N<30'})
                                        continue
                                    fit=refit_baseline(y_col,focus_cx,Xv0,grp_data,model_sel,is_panel,use_rob,id_col,tc_col,did_var=did_var,iv_endog=iv_endog,iv_insts=iv_insts,heckman_sel=heckman_sel,did_policy_time=did_policy_time,did_post_var=did_post_var)
                                    if fit['success']:
                                        s='***' if fit['pval']<0.01 else ('**' if fit['pval']<0.05 else ('*' if fit['pval']<0.1 else ''))
                                        het_results.append({'分组':grp_label,'N':fit['n'],'系数':f"{fit['coef']:.4f}{s}",'SE':f"({fit['se']:.4f})",'t值':f"{fit['tstat']:.2f}",'p值':f"{fit['pval']:.4f}",'R²':f"{fit['rsq']:.3f}" if not np.isnan(fit['rsq']) else 'N/A','备注':''})
                                    else:
                                        het_results.append({'分组':grp_label,'N':fit.get('n',0),'系数':'失败','SE':'—','t值':'—','p值':'—','R²':'—','备注':fit.get('error','拟合失败')[:40]})
                                st.session_state[f'_het_{cx}']={'results':het_results,'group_col':het_var,'focus_cx':focus_cx}
                        if f'_het_{cx}' in st.session_state:
                            hst=st.session_state[f'_het_{cx}']
                            st.caption(f"分组: {hst['group_col']} | 模型: {model_sel} | 变量: {hst['focus_cx']}")
                            st.dataframe(pd.DataFrame(hst['results']),use_container_width=True,hide_index=True)

                # ═══ 稳健性检验 ═══
                if any(sk in model_sel for sk in het_skip):
                    st.info(f"'{model_sel}' 模型暂不支持稳健性检验，请换用 FE+RE/OLS/Logit 等回归模型")
                else:
                    with st.expander("稳健性检验",expanded=False):
                        focus_cx=jcx_keys[0] if is_joint else cx
                        st.markdown("**缩尾处理 (Winsorization)**")
                        cwx1,cwx2,cwx3=st.columns(3)
                        with cwx1:
                            wx_x1=st.checkbox("核心X 1%缩尾",key=f'wx_x1_{cx}')
                            wx_y1=st.checkbox("Y 1%缩尾",key=f'wx_y1_{cx}')
                        with cwx2:
                            wx_x5=st.checkbox("核心X 5%缩尾",key=f'wx_x5_{cx}')
                            wx_y5=st.checkbox("Y 5%缩尾",key=f'wx_y5_{cx}')
                        with cwx3:
                            wx_x10=st.checkbox("核心X 10%缩尾",key=f'wx_x10_{cx}')
                            wx_y10=st.checkbox("Y 10%缩尾",key=f'wx_y10_{cx}')
                        st.markdown("**样本剔除**")
                        num_cols=[c for c in df_aug.columns if pd.api.types.is_numeric_dtype(df_aug[c]) and c!=y_col]
                        drop_col=st.selectbox("剔除依据变量",num_cols,key=f'rob_dropcol_{cx}') if num_cols else None
                        dc1,dc2=st.columns(2)
                        with dc1: dmin=st.number_input("保留 ≥",value=float(df_aug[drop_col].min()) if drop_col else 0.0,key=f'rob_dmin_{cx}_{drop_col}')
                        with dc2: dmax=st.number_input("保留 ≤",value=float(df_aug[drop_col].max()) if drop_col else 100.0,key=f'rob_dmax_{cx}_{drop_col}')
                        use_drop=st.checkbox("启用样本剔除",key=f'rob_drop_en_{cx}') if drop_col else False
                        st.info("更换X/Y变量、增删控制变量等请返回 Step 3 重新搜索基准回归")
                        if st.button("运行稳健性检验",key=f'run_rob_{cx}'):
                            id_col=st.session_state.get('_id_col'); tc_col=st.session_state.get('_tc_col')
                            needed=[y_col]+Xv0+([] if not is_panel else [id_col,tc_col])
                            base_data=df_aug[needed].dropna()
                            baseline=refit_baseline(y_col,focus_cx,Xv0,base_data,model_sel,is_panel,use_rob,id_col,tc_col,did_var=did_var,iv_endog=iv_endog,iv_insts=iv_insts,heckman_sel=heckman_sel,did_policy_time=did_policy_time,did_post_var=did_post_var)
                            rob_results=[]
                            if baseline['success']:
                                s='***' if baseline['pval']<0.01 else ('**' if baseline['pval']<0.05 else ('*' if baseline['pval']<0.1 else ''))
                                rob_results.append({'检验方式':'基准回归','N':baseline['n'],'系数':f"{baseline['coef']:.4f}{s}",'SE':f"({baseline['se']:.4f})",'t值':f"{baseline['tstat']:.2f}",'R²':f"{baseline['rsq']:.3f}" if not np.isnan(baseline['rsq']) else 'N/A'})
                            else:
                                rob_results.append({'检验方式':'基准回归','N':baseline.get('n',0),'系数':'失败','SE':baseline.get('error','')[:30],'t值':'—','R²':'—'})
                            # Winsorization runs
                            wx_runs=[(wx_x1,'X 1%缩尾','X',1),(wx_x5,'X 5%缩尾','X',5),(wx_x10,'X 10%缩尾','X',10),
                                     (wx_y1,'Y 1%缩尾','Y',1),(wx_y5,'Y 5%缩尾','Y',5),(wx_y10,'Y 10%缩尾','Y',10)]
                            for enabled,label,target,pct in wx_runs:
                                if not enabled: continue
                                wdata=base_data.copy()
                                if target=='Y':
                                    lo,hi=wdata[y_col].quantile(pct/100),wdata[y_col].quantile(1-pct/100)
                                    wdata[y_col]=wdata[y_col].clip(lo,hi)
                                else:
                                    for xv in [v for v in Xv0 if v in wdata.columns]:
                                        lo,hi=wdata[xv].quantile(pct/100),wdata[xv].quantile(1-pct/100)
                                        wdata[xv]=wdata[xv].clip(lo,hi)
                                wdata=wdata.dropna()
                                fit=refit_baseline(y_col,focus_cx,Xv0,wdata,model_sel,is_panel,use_rob,id_col,tc_col,did_var=did_var,iv_endog=iv_endog,iv_insts=iv_insts,heckman_sel=heckman_sel,did_policy_time=did_policy_time,did_post_var=did_post_var)
                                if fit['success']:
                                    s='***' if fit['pval']<0.01 else ('**' if fit['pval']<0.05 else ('*' if fit['pval']<0.1 else ''))
                                    rob_results.append({'检验方式':label,'N':fit['n'],'系数':f"{fit['coef']:.4f}{s}",'SE':f"({fit['se']:.4f})",'t值':f"{fit['tstat']:.2f}",'R²':f"{fit['rsq']:.3f}" if not np.isnan(fit['rsq']) else 'N/A'})
                                else:
                                    rob_results.append({'检验方式':label,'N':fit.get('n',0),'系数':'失败','SE':fit.get('error','')[:30],'t值':'—','R²':'—'})
                            # Sample dropping
                            if use_drop and drop_col:
                                ddata=base_data[(base_data[drop_col]>=dmin)&(base_data[drop_col]<=dmax)]
                                fit=refit_baseline(y_col,focus_cx,Xv0,ddata,model_sel,is_panel,use_rob,id_col,tc_col,did_var=did_var,iv_endog=iv_endog,iv_insts=iv_insts,heckman_sel=heckman_sel,did_policy_time=did_policy_time,did_post_var=did_post_var)
                                if fit['success']:
                                    s='***' if fit['pval']<0.01 else ('**' if fit['pval']<0.05 else ('*' if fit['pval']<0.1 else ''))
                                    rob_results.append({'检验方式':f'保留 {drop_col}∈[{dmin},{dmax}]','N':fit['n'],'系数':f"{fit['coef']:.4f}{s}",'SE':f"({fit['se']:.4f})",'t值':f"{fit['tstat']:.2f}",'R²':f"{fit['rsq']:.3f}" if not np.isnan(fit['rsq']) else 'N/A'})
                                else:
                                    rob_results.append({'检验方式':f'保留 {drop_col}∈[{dmin},{dmax}]','N':fit.get('n',0),'系数':'失败','SE':fit.get('error','')[:30],'t值':'—','R²':'—'})
                            st.session_state[f'_rob_{cx}']=rob_results
                        if f'_rob_{cx}' in st.session_state:
                            st.caption(f"模型: {model_sel} | 变量: {focus_cx}")
                            st.dataframe(pd.DataFrame(st.session_state[f'_rob_{cx}']),use_container_width=True,hide_index=True)

st.markdown("---")
st.caption("鹈鹕回归 · 仅供学术研究参考 · 数据仅存于你的电脑 · 使用即表示同意自行验证所有结果")
