"""Pelican Regression: versioned entrypoint."""
import base64
from pathlib import Path
import streamlit as st
from tihu_ui import render_workbench

st.set_page_config(page_title="鹈鹕回归", layout="wide")

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


render_workbench()
