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
import time, warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="鹈鹕回归", layout="wide")

for k, v in [('df',None),('df_name',None),('data_type',None),('diagnosed',False),
             ('id_col',None),('time_col',None),('balanced',None),('n_units',0),('n_periods',0),
             ('search_results',None)]:
    if k not in st.session_state: st.session_state[k] = v

# ── 免责声明模板（统一注入 Stata 代码头部） ──
LEGAL_HEADER = """* ============================================================
* 鹈鹕回归 (c) {year} · 自动生成 · 仅供参考
* 本代码仅供学术研究参考，不构成统计咨询或数据分析服务。
* 使用者应自行验证模型设定的合理性与结果的稳健性。
* 工具开发者不对因使用本代码产生的任何学术后果承担责任。
* ============================================================

"""

st.title("鹈鹕回归")
st.caption("面板 FE/RE/DID · 横截面 OLS/Logit/Tobit/有序/Poisson/Heckman/PSM/IV · 时间序列 ARIMA/VAR")

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
    ols=OLS(ya,Xa).fit(); b0=np.append(ols.params.values,np.log(max(ols.scale**0.5,0.01)))
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
            eps=1e-5; H=np.zeros((k+1,k+1))
            for ii in range(k+1):
                for jj in range(ii,k+1):
                    xp=res.x.copy(); xp[ii]+=eps; xp[jj]+=eps
                    xm=res.x.copy(); xm[ii]-=eps; xm[jj]-=eps
                    H[ii,jj]=(nll(xp)-nll(xm))/(2*eps) if ii==jj else (nll(xp)-nll(xm))/(4*eps*eps)
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

# ═══════════════════ STEP 1: 上传 ═══════════════════
st.header("Step 1 · 上传数据")
uploaded = st.file_uploader("拖拽 .dta / .csv / .xlsx", type=['dta','csv','xlsx'])
if uploaded is not None:
    if st.session_state.df_name != uploaded.name:
        try:
            if uploaded.name.endswith('.dta'): st.session_state.df=pd.read_stata(uploaded)
            elif uploaded.name.endswith('.csv'): st.session_state.df=pd.read_csv(uploaded)
            else: st.session_state.df=pd.read_excel(uploaded)
            st.session_state.df_name=uploaded.name; st.session_state.data_type=None
            st.session_state.diagnosed=False; st.session_state.search_results=None
        except Exception as e: st.error(f"读取失败：{e}"); st.stop()

if st.session_state.df is not None:
    df=st.session_state.df
    st.success(f"当前：{st.session_state.df_name} ｜ {df.shape[0]}行 × {df.shape[1]}列")
    with st.expander("预览"): st.dataframe(df.head(6),use_container_width=True)

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
            if all_bin and y_type=='continuous':
                model_options.append('DID 双重差分（面板+政策冲击）')
            if all_bin: model_options.append('PSM-DID')
            if cat_cols and y_type=='continuous':
                model_options.append('分组面板回归（按分类变量）')
            model_options.append('中介效应（X→M→Y）')
            model_options.append('调节效应（X×M 交互）')
        else:
            if y_type=='binary': model_options=['Logit','Probit','Logit+Probit','LPM（线性概率模型）']
            elif y_type=='ordered': model_options=['Ordered Logit','Ordered Probit']
            elif y_type=='count': model_options=['Poisson','负二项（Negative Binomial）']
            elif y_type=='censored': model_options=['OLS+稳健SE','Tobit（截断回归）']
            else: model_options=['OLS','OLS+稳健SE','Tobit（如被截断）','Heckman（样本选择）']
            model_options.append('PSM（倾向得分匹配）')
            model_options.append('IV/2SLS（工具变量）')
            model_options.append('中介效应（X→M→Y）')
            model_options.append('调节效应（X×M 交互）')
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

        # 核心X + 控制变量（含虚拟变量）
        rem=[c for c in all_vars if c!=y_col]
        core_x=st.multiselect("核心 X（最多 8 个）",rem,default=rem[:1] if rem else [],max_selections=8,key='cx7')
        pool_opt=[c for c in rem if c not in core_x]
        ctrl_pool=st.multiselect(f"控制变量候选池（最多 20，可选 {len(pool_opt)} 个）",pool_opt,
            default=pool_opt[:min(10,len(pool_opt))],max_selections=20,key='pool7')
        c3,c4=st.columns(2)
        with c3: cmin=st.number_input("最少控制数",0,15,2,key='cmin7')
        with c4: cmax=st.number_input("最多控制数",1,15,min(5,len(ctrl_pool) if ctrl_pool else 5),key='cmax7')
        if cmax<cmin: cmax=cmin

        search_mode=st.radio("核心X显著要求",['分别显著（各X独立搜索）','同时显著（所有X联合搜索）'],
            horizontal=True,key='smode7',
            help="分别：为每个核心X找各自的最优控制组合｜同时：控制组合必须让所有核心X都显著")

        # 模型特有选项
        use_cl=False; cluster_col=None; use_rob=False; se_mode='ordinary'
        bin_m=None; did_var=None; heckman_sel=None
        iv_endog=None; iv_insts=[]
        if dt=='panel':
            if 'DID' in sel_model and 'PSM' not in sel_model:
                did_var=st.selectbox("处理变量（二分，1=处理组）",bin_vars,key='did7')
                st.caption("模型将自动创建 Post×Treat 交互项")
            if 'PSM-DID' in sel_model:
                did_var=st.selectbox("处理变量",bin_vars,key='psmdid7')
                st.caption("PSM-DID：先匹配再双重差分")
        else:
            if 'PSM' in sel_model:
                did_var=st.selectbox("处理变量（二分）",[c for c in rem if len(df_aug[c].dropna().unique())==2],key='psm_t7')
            if 'IV' in sel_model:
                iv_endog=st.selectbox("内生变量",rem[:10],key='ive7')
                iv_insts=st.multiselect("工具变量（至少1个）",[c for c in rem if c!=iv_endog],key='ivi7',max_selections=5)
            if 'Heckman' in sel_model:
                heckman_sel=st.selectbox("选择变量（二分，1=被观测到）",[c for c in rem if len(df_aug[c].dropna().unique())==2],key='hks7')

        # 中介/调节特有选项
        med_m=None; mod_m=None; med_method='温忠麟五步流程（2014）'
        if '中介' in sel_model:
            med_opts=[c for c in all_num if c not in excl and c!=y_col and c not in core_x]
            med_m=st.selectbox("中介变量 M",med_opts,key='medm7') if med_opts else None
            med_method=st.radio("中介方法",['温忠麟五步流程（2014）','江艇渠道检验（2022）'],horizontal=True,key='medmethod7')
            if not med_opts: st.warning("无可用的中介变量")
        if '调节' in sel_model:
            mod_opts=[c for c in all_num if c not in excl and c!=y_col and c not in core_x]
            mod_m=st.selectbox("调节变量 M",mod_opts,key='modm7') if mod_opts else None
            if not mod_opts: st.warning("无可用的调节变量")

        # ── 共享 SE 选择器（适用大多数模型）──
        se_skip_models=['Tobit','Ordered','ANOVA','分组面板','交互效应','PSM','IV','Heckman','OLS+稳健SE','PSM-DID','中介','调节']
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
        if ('中介' in sel_model or '调节' in sel_model) and '同时' in search_mode and len(core_x)>1:
            st.info("中介/调节效应建议使用「分别显著」模式，将对每个核心X单独分析")
        if st.button(btn_label,type="primary",key='srch7'):
            if not core_x: st.warning("请选核心 X")
            elif not ctrl_pool or len(ctrl_pool)<2: st.warning("候选池需 ≥2 个变量")
            elif len(ctrl_pool)<cmin: st.warning("候选池不足最少控制数")
            elif 'IV' in sel_model and len(iv_insts)<1: st.warning("工具变量至少选1个")
            else:
                amx=min(cmax,len(ctrl_pool)); amn=min(cmin,amx)
                progress=st.progress(0); stxt=st.empty(); t0=time.time()

                # 准备数据（使用含虚拟变量的增强 dataframe）
                if dt=='panel':
                    id_col,tc_col=st.session_state.id_col,st.session_state.time_col
                    use_vars=[id_col,tc_col,y_col]+core_x+ctrl_pool
                    if did_var: use_vars.append(did_var)
                    sub=df_aug[use_vars].dropna().copy()
                    sub[id_col]=sub[id_col].astype(str)
                    if sub.duplicated(subset=[id_col,tc_col]).sum()>0:
                        sub=sub.groupby([id_col,tc_col]).mean().reset_index()
                else:
                    use_vars=[y_col]+core_x+ctrl_pool
                    if did_var: use_vars.append(did_var)
                    sub=df_aug[use_vars].dropna().copy()

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
                            Xv=[cx]+list(combo); td=sub[[y_col]+Xv].dropna()
                            if len(td)<30: continue
                            m=OLS(td[y_col].values,sm.add_constant(td[Xv].values)).fit()
                            b=m.params.iloc[1]; se=m.bse.iloc[1]; ts=b/se if se>0 else 0
                            ols_p={}; pidx=list(m.params.index)
                            for jj in range(min(len(pidx),len(['const']+Xv))):
                                ols_p[pidx[jj]]={'b':float(m.params.iloc[jj]),'se':float(m.bse.iloc[jj])}
                            res.append(dict(controls=combo,n=len(td),tstat=float(ts),
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
                            Xv=cx_list+list(combo); td=sub[[y_col]+Xv].dropna()
                            if len(td)<30: continue
                            m=OLS(td[y_col].values,sm.add_constant(td[Xv].values)).fit()
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
                            res.append(dict(controls=combo,n=len(td),min_abs_tstat=float(min_t),
                                tstats=tstats,rsq=float(m.rsquared),rsq_adj=float(m.rsquared_adj),
                                ols_params=ols_p,all_Xv=Xv))
                        except: continue
                    res.sort(key=lambda x: x['min_abs_tstat'],reverse=True)
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
                        else:
                            sr[cx]=search_ols(cx,ctrl_pool,amn,amx)[:50]
                        stxt.text(f"{cx}: {len(sr[cx])} 个有效组合")
                progress.progress(1.0)
                # ── 两阶段：中介/调节 → 在显著组合上跑第二阶段的效应分析 ──
                med_results=None; mod_results=None
                if '中介' in sel_model and med_m:
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
                if '调节' in sel_model and mod_m:
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
                st.session_state._cluster_col=cluster_col
                st.session_state._se_mode=se_mode; st.session_state._use_rob=use_rob
                st.session_state._heckman_sel=heckman_sel
                st.session_state._iv_endog=iv_endog; st.session_state._iv_insts=iv_insts
                st.session_state._id_col=st.session_state.id_col if dt=='panel' else None
                st.session_state._tc_col=st.session_state.time_col if dt=='panel' else None
                st.session_state._group_var=group_var
                st.session_state._df_aug=df_aug
                st.session_state._cat_cols=cat_cols
                st.session_state._med_results=med_results; st.session_state._mod_results=mod_results
                st.session_state._med_m=med_m; st.session_state._mod_m=mod_m
                st.session_state._med_method=med_method

    # ═══════════════════ STEP 4: 结果 ═══════════════════
    if st.session_state.search_results is not None:
        st.header("Step 4 · 选择组合 → 完整结果 & 代码")
        sr=st.session_state.search_results; y_col=st.session_state._y
        model_sel=st.session_state._model; yt=st.session_state._y_type
        is_panel=st.session_state._is_panel; sub=st.session_state._sub
        did_var=st.session_state._did_var; use_cl=st.session_state._use_cl
        cluster_col=st.session_state.get('_cluster_col',None)
        se_mode=st.session_state.get('_se_mode','ordinary'); use_rob=st.session_state.get('_use_rob',False)
        heckman_sel=st.session_state._heckman_sel
        iv_endog=st.session_state._iv_endog; iv_insts=st.session_state._iv_insts
        group_var=st.session_state.get('_group_var',None)
        df_aug=st.session_state.get('_df_aug',sub)
        search_mode=st.session_state.get('_search_mode','分别显著（各X独立搜索）')
        is_joint=('同时' in search_mode)

        for cx,results in sr.items():
            if not results: st.warning(f"{cx}: 无有效组合"); continue
            top=results[:20]
            # 获取核心X列表
            if is_joint and results:
                jcx_keys=[k for k in results[0].get('tstats',{}).keys()]
            else:
                jcx_keys=[]

            # 排名表
            tbl=[]
            for i,r in enumerate(top):
                cs=', '.join(r['controls'][:4])
                if len(r['controls'])>4: cs+=f' +{len(r["controls"])-4}'
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
                    elif 'ols_params' in r and len(r['ols_params'])>1:
                        pk=list(r['ols_params'].keys())[1]; cx_b=f"{r['ols_params'][pk]['b']:.4f}{s}"
                    tbl.append({'#':i+1,'控制组合':cs,'β(SE)':cx_b,'|t|':f"{abs(r['tstat']):.2f}",
                        'R²':f"{r.get('fe_rsq',r.get('rsq',0)):.3f}",'N':r['n']})
            n_sig05=sum(1 for r in results if (r.get('min_abs_tstat',abs(r.get('tstat',0))) if is_joint else abs(r['tstat']))>stats.t.ppf(0.975,df=max(r['n']-5,1)))
            expander_label=f"**{cx}** — {len(results)}组合 ｜ {'min|t|' if is_joint else '|t|'}>1.96占 {n_sig05/max(len(results),1)*100:.0f}%"
            with st.expander(expander_label,expanded=True):
                st.dataframe(pd.DataFrame(tbl),use_container_width=True,hide_index=True)

                combo_labels={}
                for i,r in enumerate(top):
                    cc=', '.join(r['controls'][:3])
                    if len(r['controls'])>3: cc+=f'...+{len(r["controls"])-3}'
                    combo_labels[f"#{i+1}: {cc}"]=r
                sel_key=f'sel_{cx}_v7'
                if sel_key not in st.session_state: st.session_state[sel_key]=list(combo_labels.keys())[0]
                sel=st.selectbox(f"选择 {cx} 的组合",list(combo_labels.keys()),key=sel_key)
                chosen=combo_labels[sel]; ctrls=list(chosen['controls']); N_eff=chosen['n']
                if is_joint:
                    model_line=f"{y_col} = {' + '.join(jcx_keys)} + [{', '.join(ctrls[:6])}{' +...' if len(ctrls)>6 else ''}]"
                    Xv0=jcx_keys+ctrls
                else:
                    model_line=f"{y_col} = {cx} + [{', '.join(ctrls[:6])}{' +...' if len(ctrls)>6 else ''}]"
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
                    td=sub[[id_col,tc_col,y_col,did_var]+Xv0].dropna().copy()
                    td[id_col]=td[id_col].astype(str)
                    # 创建 Post: 时间中位数之后为1
                    t_med=td[tc_col].median()
                    td['_post']=(td[tc_col]>=t_med).astype(float)
                    td['_treat']=td[did_var]
                    td['_did']=td['_treat']*td['_post']
                    td_idx=td.set_index([id_col,tc_col])
                    X_did=sm.add_constant(td_idx[Xv0+['_treat','_post','_did']])
                    try:
                        d_m=PanelOLS(td_idx[y_col],X_did,entity_effects=True,time_effects=False)
                        d_r=d_m.fit()
                        st.info(f"DID 估计量（交互项 _did）= {d_r.params.get('_did',np.nan):.4f} (SE={d_r.std_errors.get('_did',np.nan):.4f})")
                        rows=[]
                        for v in Xv0+['_treat','_post','_did']:
                            b=d_r.params.get(v,np.nan); se=d_r.std_errors.get(v,np.nan)
                            t=abs(b/se) if se>0 else 0; pv=2*(1-stats.t.cdf(t,df=len(td)-len(X_did.columns)))
                            s='***' if pv<0.01 else ('**' if pv<0.05 else ('*' if pv<0.1 else ''))
                            rows.append({'变量':v,'系数':f"{b:.4f}{s}",'SE':f"({se:.4f})"})
                        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
                        do=(f"* === 鹈鹕回归 (c)2026 · 仅供参考 · 不构成统计建议 ===\n* DID: {model_line}\n\nuse \"data.dta\", clear\n"
                            f"xtset {id_col} {tc_col}\ngen post=({tc_col}>={t_med})\ngen treat={did_var}\ngen did=treat*post\n"
                            f"reghdfe {y_col} {cx} {' '.join(ctrls)} treat post did, absorb({id_col}) vce(robust)\n"
                            f"* 平行趋势检验: 画图看 pre-trend\n")
                        with st.expander("Stata 复现代码"): st.code(do,language='stata')
                        dl1,dl2=st.columns(2)
                        dl1.download_button("下载 .do",do,file_name=f"{cx}_did.do",key=f'dl_{cx}_did_v7')
                    except Exception as e: st.error(f"DID 失败：{e}")

                elif model_sel.startswith('OLS') or 'Pooled' in model_sel:
                    td=sub[[y_col]+Xv0].dropna(); Xd=sm.add_constant(td[Xv0])
                    cov_t='HC1' if use_rob else 'nonrobust'
                    m=OLS(td[y_col].values,Xd).fit(cov_type=cov_t)
                    rows=[]
                    for j,vn in enumerate(['const']+Xv0):
                        b=m.params[j]; se=m.bse[j]; t=abs(b/se) if se>0 else 0; pv=m.pvalues[j]
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

                elif 'Logit' in model_sel or 'Probit' in model_sel or 'LPM' in model_sel:
                    td=sub[[y_col]+Xv0].dropna(); Xd=sm.add_constant(td[Xv0]); y_d=td[y_col]
                    rows=[]; fit_lines=[]
                    run_lg='Logit' in model_sel or '+' in model_sel; run_pr='Probit' in model_sel or '+' in model_sel
                    run_lpm='LPM' in model_sel
                    if use_cl and cluster_col: stata_se=f', vce(cluster {cluster_col})'
                    elif use_rob: stata_se=', robust'
                    else: stata_se=''
                    do=f"* === 鹈鹕回归 (c)2026 · 仅供参考 · 不构成统计建议 ===\n* {model_line}\n\nuse \"data.dta\", clear\n\n"
                    if run_lg:
                        try:
                            lf=sm.Logit(y_d,Xd).fit(disp=0)
                            for j,vn in enumerate(['const']+Xv0):
                                b=lf.params[j]; se=lf.bse[j]; t=abs(b/se) if se>0 else 0; pv=2*(1-stats.norm.cdf(t))
                                s='***' if pv<0.01 else ('**' if pv<0.05 else ('*' if pv<0.1 else ''))
                                rows.append({'变量':vn,'Logit 系数':f"{b:.4f}{s}",'Logit SE':f"({se:.4f})"})
                            fit_lines.append(f"Logit: Pseudo R²={lf.prsquared:.4f}, LL={lf.llf:.2f}")
                            do+=f"logit {y_col} {cx} {' '.join(ctrls)}{stata_se}\n"
                        except Exception as e: st.error(f"Logit: {e}")
                    if run_pr:
                        try:
                            pf=sm.Probit(y_d,Xd).fit(disp=0)
                            for j,vn in enumerate(['const']+Xv0):
                                b=pf.params[j]; se=pf.bse[j]; t=abs(b/se) if se>0 else 0; pv=2*(1-stats.norm.cdf(t))
                                s='***' if pv<0.01 else ('**' if pv<0.05 else ('*' if pv<0.1 else ''))
                                if run_lg:
                                    for rr in rows:
                                        if rr['变量']==vn: rr['Probit 系数']=f"{b:.4f}{s}"; rr['Probit SE']=f"({se:.4f})"
                                else: rows.append({'变量':vn,'Probit 系数':f"{b:.4f}{s}",'Probit SE':f"({se:.4f})"})
                            fit_lines.append(f"Probit: Pseudo R²={pf.prsquared:.4f}, LL={pf.llf:.2f}")
                            do+=f"probit {y_col} {cx} {' '.join(ctrls)}{stata_se}\n"
                        except Exception as e: st.error(f"Probit: {e}")
                    if run_lpm:
                        lm=OLS(y_d,Xd).fit(cov_type='HC1')
                        for j,vn in enumerate(['const']+Xv0):
                            b=lm.params[j]; se=lm.bse[j]; pv=lm.pvalues[j]
                            s='***' if pv<0.01 else ('**' if pv<0.05 else ('*' if pv<0.1 else ''))
                            rows.append({'变量':vn,'LPM 系数':f"{b:.4f}{s}",'LPM SE':f"({se:.4f})"})
                        fit_lines.append(f"LPM: R²={lm.rsquared:.4f}")
                        do+=f"reg {y_col} {cx} {' '.join(ctrls)}{stata_se}\n"
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
                    try:
                        om=OrderedModel(y_d,Xd,distr=dist).fit(disp=0)
                        rows=[]
                        for j,vn in enumerate(Xv0):
                            b=om.params[j]; se=om.bse[j]; t=abs(b/se) if se>0 else 0; pv=2*(1-stats.norm.cdf(t))
                            s='***' if pv<0.01 else ('**' if pv<0.05 else ('*' if pv<0.1 else ''))
                            rows.append({'变量':vn,'系数':f"{b:.4f}{s}",'SE':f"({se:.4f})",'z':f"{t:.2f}"})
                        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
                        st.caption(f"N={len(td)}｜Pseudo R²={om.prsquared:.4f}｜LL={om.llf:.2f}")
                        stcmd='oprobit' if 'Probit' in model_sel else 'ologit'
                        do=f"* === 鹈鹕回归 (c)2026 · 仅供参考 · 不构成统计建议 ===\n* {model_line}\n\nuse \"data.dta\", clear\n{stcmd} {y_col} {cx} {' '.join(ctrls)}, robust\nmargins, dydx(*) post\n"
                        with st.expander("Stata 复现代码"): st.code(do,language='stata')
                    except Exception as e: st.error(f"Ordered 失败：{e}")

                elif 'Poisson' in model_sel or '负二项' in model_sel:
                    td=sub[[y_col]+Xv0].dropna(); Xd=sm.add_constant(td[Xv0]); y_d=td[y_col]
                    fam=Poisson() if 'Poisson' in model_sel else NegativeBinomial()
                    cov_t='HC0' if use_rob else 'nonrobust'
                    try:
                        gm=GLM(y_d,Xd,family=fam).fit(cov_type=cov_t)
                        rows=[]
                        for j,vn in enumerate(['const']+Xv0):
                            b=gm.params[j]; se=gm.bse[j]; t=abs(b/se) if se>0 else 0; pv=gm.pvalues[j]
                            s='***' if pv<0.01 else ('**' if pv<0.05 else ('*' if pv<0.1 else ''))
                            rows.append({'变量':vn,'系数':f"{b:.4f}{s}",'SE':f"({se:.4f})",'z':f"{t:.2f}"})
                        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
                        se_label='聚类稳健SE' if use_cl else ('异方差稳健SE' if use_rob else '普通SE')
                        st.caption(f"N={len(td)}｜Pseudo R²={1-gm.llf/gm.llnull:.4f}｜LL={gm.llf:.2f}｜SE: {se_label}")
                        stcmd='poisson' if 'Poisson' in model_sel else 'nbreg'
                        if use_cl and cluster_col: stata_se=f', vce(cluster {cluster_col})'
                        elif use_rob: stata_se=', robust'
                        else: stata_se=''
                        do=f"* === 鹈鹕回归 (c)2026 · 仅供参考 · 不构成统计建议 ===\n* {model_line}\n\nuse \"data.dta\", clear\n{stcmd} {y_col} {cx} {' '.join(ctrls)}{stata_se}\nmargins, dydx(*) post\n"
                        with st.expander("Stata 复现代码"): st.code(do,language='stata')
                    except Exception as e: st.error(f"计数模型失败：{e}")

                elif 'Tobit' in model_sel:
                    td=sub[[y_col]+Xv0].dropna(); Xd=sm.add_constant(td[Xv0]).values; y_d=td[y_col].values
                    left_val=td[y_col].min() if (td[y_col]==td[y_col].min()).mean()>0.05 else None
                    tr=tobit_mle(y_d,Xd,left=left_val)
                    if tr['converged']:
                        rows=[]
                        for j,vn in enumerate(['const']+Xv0):
                            b=tr['params'][j]; se=tr['se'][j]; z=abs(b/se) if se>0 else 0; pv=2*(1-stats.norm.cdf(z))
                            s='***' if pv<0.01 else ('**' if pv<0.05 else ('*' if pv<0.1 else ''))
                            rows.append({'变量':vn,'系数':f"{b:.4f}{s}",'SE':f"({se:.4f})"})
                        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
                        st.caption(f"Tobit MLE｜N={len(td)}｜σ={tr['sigma']:.4f}｜LL={tr['llf']:.2f}")
                        ll_opt=f"ll({left_val})" if left_val else ""
                        do=f"* === 鹈鹕回归 (c)2026 · 仅供参考 · 不构成统计建议 ===\n* {model_line}\n\nuse \"data.dta\", clear\ntobit {y_col} {cx} {' '.join(ctrls)}, {ll_opt} vce(robust)\nmargins, dydx(*) predict(ystar(0,.)) post\n"
                        with st.expander("Stata 复现代码"): st.code(do,language='stata')
                    else: st.error(f"Tobit 未收敛：{tr.get('error','')}")

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
                        try:
                            hr=heckman_two_step(y_h,X_h,Z_vars)
                            rows=[]
                            for j,vn in enumerate(Xv0):
                                b=hr['params'][j]; se=hr['se'][j]; t=abs(b/se) if se>0 else 0; pv=2*(1-stats.t.cdf(t,df=hr['n_obs']))
                                s='***' if pv<0.01 else ('**' if pv<0.05 else ('*' if pv<0.1 else ''))
                                rows.append({'变量':vn,'系数':f"{b:.4f}{s}",'SE':f"({se:.4f})"})
                            st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
                            st.caption(f"Heckman 两步法｜观测={hr['n_obs']}/{hr['n_total']}｜逆Mills比={hr['imr_coef']:.4f}(p={hr['imr_p']:.4f})｜{'选择偏差显著' if hr['imr_p']<0.05 else '选择偏差不显著'}")
                            do=f"* === 鹈鹕回归 (c)2026 · 仅供参考 · 不构成统计建议 ===\n* {model_line}\n\nuse \"data.dta\", clear\nheckman {y_col} {cx} {' '.join(ctrls)}, select({heckman_sel}={cx} {' '.join(ctrls)}) twostep\n"
                            with st.expander("Stata 复现代码"): st.code(do,language='stata')
                        except Exception as e: st.error(f"Heckman 失败：{e}")

                elif 'PSM' in model_sel:
                    if not did_var: st.error("需要处理变量")
                    else:
                        td=sub[[y_col,did_var]+Xv0].dropna()
                        D=td[did_var].values; y_p=td[y_col].values; X_p=td[Xv0].values
                        try:
                            pr=psm_att(y_p,D,X_p,method='nearest',k=1)
                            st.success(f"PSM ATT = {pr['ATT']:.4f}（{pr['method']} matching, {pr['n_treated']} treated, {pr['n_control']} control）")
                            do=f"* === 鹈鹕回归 (c)2026 · 仅供参考 · 不构成统计建议 ===\n* PSM: {model_line}\n\nuse \"data.dta\", clear\npsmatch2 {did_var} {cx} {' '.join(ctrls)}, outcome({y_col}) logit ate att\npstest, both graph\n"
                            with st.expander("Stata 复现代码"): st.code(do,language='stata')
                        except Exception as e: st.error(f"PSM 失败：{e}")

                elif 'IV' in model_sel:
                    if not iv_endog or len(iv_insts)<1: st.error("需要指定内生变量和工具变量")
                    else:
                        exog_vars=[v for v in Xv0 if v!=iv_endog]
                        td=sub[[y_col,iv_endog]+exog_vars+iv_insts].dropna()
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
                            do=f"* === 鹈鹕回归 (c)2026 · 仅供参考 · 不构成统计建议 ===\n* IV/2SLS: {model_line}\n\nuse \"data.dta\", clear\nivregress 2sls {y_col} {' '.join(exog_vars)} ({iv_endog}={' '.join(iv_insts)}), robust\nestat firststage\nestat overid\n"
                            with st.expander("Stata 复现代码"): st.code(do,language='stata')
                        except Exception as e: st.error(f"IV 失败：{e}")

                # ═══ 中介效应 ═══
                elif '中介' in model_sel:
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
                                # Step 1
                                s1='✓' if best_med['c_p']<0.05 else '✗'
                                st.caption(f"① 检验系数 c={best_med['c']:.4f} (p={best_med['c_p']:.4f}) {s1} → 按**{best_med['framework']}**立论")
                                # Step 2
                                s2a='✓' if best_med['a_p']<0.05 else '✗'; s2b='✓' if best_med['b_p']<0.05 else '✗'
                                st.caption(f"② 依次检验 a={best_med['a']:.4f} (p={best_med['a_p']:.4f}) {s2a}, b={best_med['b']:.4f} (p={best_med['b_p']:.4f}) {s2b} → {'都显著，间接效应成立' if best_med['ab_both_sig'] else '至少一个不显著，需Bootstrap'}")
                                # Step 3 (Bootstrap, if needed)
                                if not best_med['ab_both_sig']:
                                    if best_med['boot_ci']:
                                        bci=best_med['boot_ci']; bs='✓ 显著' if best_med['boot_sig'] else '✗ 不显著'
                                        st.caption(f"③ Bootstrap 1000次：ab 的95%CI=[{bci[0]:.4f}, {bci[1]:.4f}] → {bs}")
                                    else:
                                        st.caption("③ Bootstrap 失败（样本量不足）")
                                # Step 4
                                s4='不显著' if best_med['cp_p']>=0.05 else '显著'
                                st.caption(f"④ 检验直接效应 c'={best_med['cp']:.4f} (p={best_med['cp_p']:.4f}) → {s4}")
                                # Step 5
                                if best_med['effect_type'] not in ['无中介','完全中介']:
                                    ab_sign='同号' if best_med['ab']*best_med['cp']>0 else '异号'
                                    st.caption(f"⑤ ab({best_med['ab']:.4f}) 与 c'({best_med['cp']:.4f}) {ab_sign}")
                                # 最终判断
                                if best_med['effect_type'] in ['部分中介','完全中介']:
                                    es_str=f"，效应量={best_med['effect_size']:.3f}" if best_med['effect_size'] is not None else ''
                                    st.success(f"结论：{best_med['effect_type']}{es_str}")
                                elif best_med['effect_type']=='遮掩效应':
                                    st.warning(f"结论：{best_med['effect_type']}，|ab/c'|={best_med['effect_size']:.3f}")
                                else:
                                    st.warning(f"结论：{best_med['effect_type']}")
                                # 汇总指标
                                c1,c2,c3=st.columns(3)
                                with c1: st.metric("总效应 c",f"{best_med['c']:.4f}")
                                with c2: st.metric("直接效应 c'",f"{best_med['cp']:.4f}")
                                with c3: st.metric("间接效应 ab",f"{best_med['ab']:.4f}")
                                # Bootstrap CI (if available)
                                if best_med.get('boot_ci'):
                                    c4,c5=st.columns(2)
                                    with c4: st.caption(f"Bootstrap 95%CI: [{best_med['boot_ci'][0]:.4f}, {best_med['boot_ci'][1]:.4f}]")
                                    with c5: st.caption(f"R²: step1={best_med['r2_step1']:.3f}, step3={best_med['r2_step3']:.3f}")
                                else:
                                    st.caption(f"R²: step1={best_med['r2_step1']:.3f}, step2={best_med['r2_step2']:.3f}, step3={best_med['r2_step3']:.3f}")
                                # Stata 代码
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

                # ═══ 调节效应 ═══
                elif '调节' in model_sel:
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
                            do+=f"\n* 分组: {group_var}={grp}\nreghdfe {y_col} {cx} {' '.join(ctrls)} if {group_var}=={grp}, absorb({id_col} {tc_col})\n"
                        with st.expander("Stata 复现代码"): st.code(do,language='stata')

st.markdown("---")
st.caption("鹈鹕回归 · 仅供学术研究参考 · 数据仅存于你的电脑 · 使用即表示同意自行验证所有结果")

