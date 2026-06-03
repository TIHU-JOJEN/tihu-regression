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
            idc,tc=st.session_state.id_col,st.session_state.time_col
            excl=[idc,tc]
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
            else: model_options=['FE+RE（固定+随机效应）']
            rem_p=[c for c in all_vars if c!=y_col]
            all_bin=[c for c in rem_p if len(df_aug[c].dropna().unique())==2]
            bin_vars=[c for c in all_bin if c in all_num]  # 原始数值二分变量
            if all_bin and y_type=='continuous':
                model_options.append('DID 双重差分（面板+政策冲击）')
            if all_bin: model_options.append('PSM-DID')
            if cat_cols and y_type=='continuous':
                model_options.append('分组面板回归（按分类变量）')
        else:
            if y_type=='binary': model_options=['Logit','Probit','Logit+Probit','LPM（线性概率模型）']
            elif y_type=='ordered': model_options=['Ordered Logit','Ordered Probit']
            elif y_type=='count': model_options=['Poisson','负二项（Negative Binomial）']
            elif y_type=='censored': model_options=['OLS+稳健SE','Tobit（截断回归）']
            else: model_options=['OLS','OLS+稳健SE','Tobit（如被截断）','Heckman（样本选择）']
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

        # 模型特有选项
        use_cl=False; bin_m=None; did_var=None; heckman_sel=None
        iv_endog=None; iv_insts=[]
        if dt=='panel':
            if 'FE+RE' in sel_model: use_cl=st.checkbox("聚类标准误（到个体 ID）",value=True,key='cl7')
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

        btn_label="开始搜索最优控制组合"
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

                # ── 面板搜索（完整 FE+RE+Hausman） ──
                def search_panel_full(cx,pool,mn,mx):
                    res=[]; ac=[]
                    for k in range(mn,mx+1): ac.extend(combinations(pool,k))
                    n_total=len(ac)
                    if n_total>10000:
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
                            cok=False
                            try:
                                if use_cl:
                                    ca=td.index.get_level_values(0).to_numpy()
                                    fe_r=fe_m.fit(cov_type='clustered',clusters=ca)
                                    re_r=re_m.fit(cov_type='clustered',clusters=ca); cok=True
                                else: fe_r=fe_m.fit(); re_r=re_m.fit()
                            except:
                                try: fe_r=fe_m.fit(); re_r=re_m.fit()
                                except: continue
                            fe_b=fe_r.params.get(cx,np.nan); fe_se=fe_r.std_errors.get(cx,np.nan)
                            ts=fe_b/fe_se if fe_se>0 else 0
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

                sr={}
                for cx in core_x:
                    stxt.text(f"搜索 {cx}...")
                    if dt=='panel' and 'FE+RE' in sel_model:
                        sr[cx]=search_panel_full(cx,ctrl_pool,amn,amx)[:50]
                    else:
                        sr[cx]=search_ols(cx,ctrl_pool,amn,amx)[:50]
                    stxt.text(f"{cx}: {len(sr[cx])} 个有效组合")
                progress.progress(1.0)
                st.success(f"完成！{time.time()-t0:.1f}s")
                st.session_state.search_results=sr
                st.session_state._y=y_col; st.session_state._y_type=y_type
                st.session_state._is_panel=(dt=='panel'); st.session_state._model=sel_model
                st.session_state._sub=sub
                for kk in ['_did_var','_use_cl','_heckman_sel','_iv_endog','_iv_insts','_id_col','_tc_col','_bin_m']:
                    if kk in st.session_state: del st.session_state[kk]
                st.session_state._did_var=did_var; st.session_state._use_cl=use_cl
                st.session_state._heckman_sel=heckman_sel
                st.session_state._iv_endog=iv_endog; st.session_state._iv_insts=iv_insts
                st.session_state._id_col=st.session_state.id_col if dt=='panel' else None
                st.session_state._tc_col=st.session_state.time_col if dt=='panel' else None
                st.session_state._group_var=group_var
                st.session_state._df_aug=df_aug
                st.session_state._cat_cols=cat_cols

    # ═══════════════════ STEP 4: 结果 ═══════════════════
    if st.session_state.search_results is not None:
        st.header("Step 4 · 选择组合 → 完整结果 & 代码")
        sr=st.session_state.search_results; y_col=st.session_state._y
        model_sel=st.session_state._model; yt=st.session_state._y_type
        is_panel=st.session_state._is_panel; sub=st.session_state._sub
        did_var=st.session_state._did_var; use_cl=st.session_state._use_cl
        heckman_sel=st.session_state._heckman_sel
        iv_endog=st.session_state._iv_endog; iv_insts=st.session_state._iv_insts
        group_var=st.session_state.get('_group_var',None)
        df_aug=st.session_state.get('_df_aug',sub)

        for cx,results in sr.items():
            if not results: st.warning(f"{cx}: 无有效组合"); continue
            top=results[:20]; sig05=sum(1 for r in results if r['pval']<0.05)

            # 排名表
            tbl=[]
            for i,r in enumerate(top):
                cs=', '.join(r['controls'][:4])
                if len(r['controls'])>4: cs+=f' +{len(r["controls"])-4}'
                s='***' if r['pval']<0.01 else ('**' if r['pval']<0.05 else ('*' if r['pval']<0.1 else ''))
                cx_b='—'
                if 'fe_beta' in r: cx_b=f"{r['fe_beta']:.4f}{s}"
                elif 'ols_params' in r and len(r['ols_params'])>1:
                    pk=list(r['ols_params'].keys())[1]; cx_b=f"{r['ols_params'][pk]['b']:.4f}{s}"
                tbl.append({'#':i+1,'控制组合':cs,'β(SE)':cx_b,'|t|':f"{abs(r['tstat']):.2f}",
                    'R²':f"{r.get('fe_rsq',r.get('rsq',0)):.3f}",'N':r['n']})
            expander_label=f"**{cx}** — {len(results)}组合 ｜ p<0.05占 {sig05/max(len(results),1)*100:.0f}%"
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
                model_line=f"{y_col} = {cx} + [{', '.join(ctrls[:6])}{' +...' if len(ctrls)>6 else ''}]"

                st.divider(); st.markdown("### 完整回归结果")
                st.markdown(f"**模型设定**: {model_line}（{model_sel}）")

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
                    se_n="聚类稳健" if chosen.get('cluster_ok') else "普通稳健"
                    st.caption(f"注：{se_n}SE。*** p<0.01, ** p<0.05, * p<0.10。\nN={N_eff}｜个体={n_id}｜时期={n_t}\n含个体+时间固定效应\nFE R²={chosen['fe_rsq']:.4f}｜RE R²={chosen['re_rsq']:.4f}")
                    do=(f"* === 鹈鹕回归 (c)2026 · 仅供参考 · 不构成统计建议 ===\n* {model_line}\n\nuse \"data.dta\", clear\nxtset {id_col} {tc_col}\n\n")
                    if chosen.get('cluster_ok'): do+=f"reghdfe {y_col} {cx} {' '.join(ctrls)}, absorb({id_col} {tc_col}) vce(cluster {id_col})\n"
                    else: do+=f"reghdfe {y_col} {cx} {' '.join(ctrls)}, absorb({id_col} {tc_col})\n"
                    do+="est sto fe\n\n"
                    if chosen.get('cluster_ok'): do+=f"xtreg {y_col} {cx} {' '.join(ctrls)}, re vce(cluster {id_col})\n"
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

                elif model_sel.startswith('OLS'):
                    td=sub[[y_col]+Xv0].dropna(); Xd=sm.add_constant(td[Xv0])
                    use_rob='稳健' in model_sel; cov_t='HC1' if use_rob else 'nonrobust'
                    m=OLS(td[y_col].values,Xd).fit(cov_type=cov_t)
                    rows=[]
                    for j,vn in enumerate(['const']+Xv0):
                        b=m.params[j]; se=m.bse[j]; t=abs(b/se) if se>0 else 0; pv=m.pvalues[j]
                        s='***' if pv<0.01 else ('**' if pv<0.05 else ('*' if pv<0.1 else ''))
                        rows.append({'变量':vn,'系数':f"{b:.4f}{s}",'SE':f"({se:.4f})",'t':f"{t:.2f}",'p':f"{pv:.4f}"})
                    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
                    st.caption(f"OLS（{'HC1稳健' if use_rob else '普通'}SE）｜N={len(td)}｜R²={m.rsquared:.4f}｜adj R²={m.rsquared_adj:.4f}｜F={m.fvalue:.2f}(p={m.f_pvalue:.4f})")
                    do=f"* === 鹈鹕回归 (c)2026 · 仅供参考 · 不构成统计建议 ===\n* {model_line}\n\nuse \"data.dta\", clear\nreg {y_col} {cx} {' '.join(ctrls)}{', robust' if use_rob else ''}\n"
                    with st.expander("Stata 复现代码"): st.code(do,language='stata')
                    dl1,dl2=st.columns(2)
                    dl1.download_button("下载 .do",do,file_name=f"{cx}_ols.do",key=f'dlo_{cx}_v7')
                    dl2.download_button("下载 .csv",pd.DataFrame(rows).to_csv(index=False),file_name=f"{cx}_ols.csv",mime="text/csv",key=f'cso_{cx}_v7')

                elif 'Logit' in model_sel or 'Probit' in model_sel or 'LPM' in model_sel:
                    td=sub[[y_col]+Xv0].dropna(); Xd=sm.add_constant(td[Xv0]); y_d=td[y_col]
                    rows=[]; fit_lines=[]
                    run_lg='Logit' in model_sel or '+' in model_sel; run_pr='Probit' in model_sel or '+' in model_sel
                    run_lpm='LPM' in model_sel
                    do=f"* === 鹈鹕回归 (c)2026 · 仅供参考 · 不构成统计建议 ===\n* {model_line}\n\nuse \"data.dta\", clear\n\n"
                    if run_lg:
                        try:
                            lf=sm.Logit(y_d,Xd).fit(disp=0)
                            for j,vn in enumerate(['const']+Xv0):
                                b=lf.params[j]; se=lf.bse[j]; t=abs(b/se) if se>0 else 0; pv=2*(1-stats.norm.cdf(t))
                                s='***' if pv<0.01 else ('**' if pv<0.05 else ('*' if pv<0.1 else ''))
                                rows.append({'变量':vn,'Logit 系数':f"{b:.4f}{s}",'Logit SE':f"({se:.4f})"})
                            fit_lines.append(f"Logit: Pseudo R²={lf.prsquared:.4f}, LL={lf.llf:.2f}")
                            do+=f"logit {y_col} {cx} {' '.join(ctrls)}, robust\n"
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
                            do+=f"probit {y_col} {cx} {' '.join(ctrls)}, robust\n"
                        except Exception as e: st.error(f"Probit: {e}")
                    if run_lpm:
                        lm=OLS(y_d,Xd).fit(cov_type='HC1')
                        for j,vn in enumerate(['const']+Xv0):
                            b=lm.params[j]; se=lm.bse[j]; pv=lm.pvalues[j]
                            s='***' if pv<0.01 else ('**' if pv<0.05 else ('*' if pv<0.1 else ''))
                            rows.append({'变量':vn,'LPM 系数':f"{b:.4f}{s}",'LPM SE':f"({se:.4f})"})
                        fit_lines.append(f"LPM: R²={lm.rsquared:.4f}")
                        do+=f"reg {y_col} {cx} {' '.join(ctrls)}, robust\n"
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
                    try:
                        gm=GLM(y_d,Xd,family=fam).fit()
                        rows=[]
                        for j,vn in enumerate(['const']+Xv0):
                            b=gm.params[j]; se=gm.bse[j]; t=abs(b/se) if se>0 else 0; pv=gm.pvalues[j]
                            s='***' if pv<0.01 else ('**' if pv<0.05 else ('*' if pv<0.1 else ''))
                            rows.append({'变量':vn,'系数':f"{b:.4f}{s}",'SE':f"({se:.4f})",'z':f"{t:.2f}"})
                        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
                        st.caption(f"N={len(td)}｜Pseudo R²={1-gm.llf/gm.llnull:.4f}｜LL={gm.llf:.2f}")
                        stcmd='poisson' if 'Poisson' in model_sel else 'nbreg'
                        do=f"* === 鹈鹕回归 (c)2026 · 仅供参考 · 不构成统计建议 ===\n* {model_line}\n\nuse \"data.dta\", clear\n{stcmd} {y_col} {cx} {' '.join(ctrls)}, robust\nmargins, dydx(*) post\n"
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
