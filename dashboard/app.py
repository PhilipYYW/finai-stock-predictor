# dashboard/app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib, json, os, sys, subprocess, time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset.build_historical_dataset import FEATURE_COLS_HIST, LABEL_COLS

st.set_page_config(page_title="FinAI", page_icon="📈",
                   layout="wide", initial_sidebar_state="collapsed")

MODEL_DIR   = "models"
DATA_DIR    = "data"
CONFIG_PATH = "config.json"
HIST_FEAT   = FEATURE_COLS_HIST + ["ticker_enc"]

C = {
    "bg":"#080D18","card":"#0F1724","card2":"#162034",
    "border":"#1C2E45","border2":"#243852",
    "teal":"#00D4FF","green":"#00E5A0","red":"#FF4560",
    "orange":"#FF9F1C","purple":"#A78BFA","yellow":"#FFD60A",
    "text":"#E2E8F0","sub":"#5A7A9A","muted":"#2E4A6A",
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
*, html, body {{ font-family:'Inter',sans-serif !important; }}
.stApp {{ background:{C['bg']} !important; }}
#MainMenu,footer,header {{ visibility:hidden; }}
/* Streamlit's actual content container */
.block-container {{
    padding: 2rem 4rem 4rem 4rem !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
}}

/* Fallback page wrap */
.page-wrap {{ display: contents; }}

/* NAV */
.topbar {{
    background:{C['card']}; border-bottom:1px solid {C['border']};
    padding:0 4rem; height:56px; display:flex;
    align-items:center; justify-content:space-between;
    position:sticky; top:0; z-index:999;
}}
.logo {{ font-size:16px; font-weight:800; color:{C['teal']}; letter-spacing:1px; }}
.logo span {{ color:{C['text']}; font-weight:400; }}

/* Apple-style pill nav — override Streamlit buttons */
div[data-testid="column"]:has(button[key^="nav_"]) button {{
    background: transparent !important;
    border: none !important;
    color: {C['sub']} !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    padding: 6px 16px !important;
    transition: all 0.15s !important;
    box-shadow: none !important;
}}
div[data-testid="column"]:has(button[key^="nav_"]) button:hover {{
    background: {C['card2']} !important;
    color: {C['text']} !important;
}}

/* Streamlit button style overrides */
.stButton > button {{
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    transition: all 0.15s !important;
}}
.stButton > button[kind="primary"] {{
    background: {C['teal']} !important;
    color: {C['bg']} !important;
    border: none !important;
    font-weight: 600 !important;
}}
.stButton > button[kind="secondary"] {{
    background: transparent !important;
    border: 1px solid {C['border']} !important;
    color: {C['sub']} !important;
}}
.stButton > button[kind="secondary"]:hover {{
    border-color: {C['teal']} !important;
    color: {C['text']} !important;
    background: {C['card2']} !important;
}}

/* Expander */
details summary {{
    font-size: 12px !important;
    font-weight: 600 !important;
    color: {C['sub']} !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}}
details summary:hover {{ color: {C['text']} !important; }}
details {{ border: none !important; background: transparent !important; }}

/* Checkbox */
.stCheckbox label {{
    font-size: 13px !important;
    color: {C['text']} !important;
}}
.stCheckbox label span {{ font-weight: 500 !important; }}

/* Selectbox */
.stSelectbox label {{ display: none !important; }}
.stSelectbox > div > div {{
    background: {C['card2']} !important;
    border: 1px solid {C['border']} !important;
    border-radius: 8px !important;
    color: {C['text']} !important;
    font-size: 13px !important;
}}

/* Input */
.stTextInput > div > div {{
    background: {C['card2']} !important;
    border: 1px solid {C['border']} !important;
    border-radius: 8px !important;
    color: {C['text']} !important;
    font-size: 13px !important;
}}

/* Metrics */
[data-testid="stMetric"] {{
    background: {C['card']} !important;
    border: 1px solid {C['border']} !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
}}

/* Progress bar */
.stProgress > div > div {{
    background: {C['teal']} !important;
    border-radius: 4px !important;
}}
.stProgress > div {{
    background: {C['border']} !important;
    border-radius: 4px !important;
}}

/* Dataframe */
[data-testid="stDataFrame"] {{
    background: {C['card']} !important;
    border: 1px solid {C['border']} !important;
    border-radius: 10px !important;
}}

/* Warning/Info/Success boxes */
.stAlert {{
    border-radius: 8px !important;
    font-size: 13px !important;
}}

/* Caption */
.stCaption {{ color: {C['sub']} !important; font-size: 11px !important; }}

/* Radio */
.stRadio > div {{
    gap: 4px !important;
}}
.stRadio label {{
    background: {C['card2']} !important;
    border: 1px solid {C['border']} !important;
    border-radius: 6px !important;
    padding: 4px 12px !important;
    font-size: 12px !important;
    color: {C['sub']} !important;
    transition: all 0.15s !important;
}}
.stRadio label:has(input:checked) {{
    background: {C['muted']} !important;
    border-color: {C['teal']} !important;
    color: {C['teal']} !important;
}}

/* PAGE LAYOUT */
.page {{ display:flex; height:calc(100vh - 52px); overflow:hidden; }}
.sidebar {{
    width:260px; min-width:260px; background:{C['card']};
    border-right:1px solid {C['border']}; overflow-y:auto;
    padding:20px 16px; display:flex; flex-direction:column; gap:4px;
}}
.content {{ flex:1; overflow-y:auto; padding:24px 28px; }}

/* CARDS */
.card {{
    background:{C['card']}; border:1px solid {C['border']};
    border-radius:14px; padding:24px;
}}
.card-sm {{
    background:{C['card']}; border:1px solid {C['border']};
    border-radius:12px; padding:18px 20px;
}}
.card-accent {{
    background:{C['card2']}; border:1px solid {C['border2']};
    border-radius:12px; padding:18px 20px;
}}

/* KPI */
.kpi {{ display:flex; flex-direction:column; gap:4px; }}
.kpi-label {{ font-size:10px; font-weight:600; color:{C['sub']};
              text-transform:uppercase; letter-spacing:1px; }}
.kpi-value {{ font-size:28px; font-weight:800; color:{C['text']}; line-height:1; margin:6px 0 4px; }}
.kpi-sub   {{ font-size:12px; font-weight:500; }}

/* SIGNAL LIST */
.sig-row {{
    display:flex; align-items:center; padding:12px 16px;
    border-radius:10px; margin-bottom:6px;
    background:{C['card2']}; border:1px solid {C['border']};
    cursor:pointer; transition:border-color 0.15s;
}}
.sig-row:hover {{ border-color:{C['teal']}; }}
.sig-ticker {{ font-size:13px; font-weight:700; color:{C['text']}; width:48px; }}
.sig-name   {{ font-size:11px; color:{C['sub']}; flex:1; }}
.badge-long  {{ background:rgba(0,229,160,.12); color:{C['green']};
               border:1px solid rgba(0,229,160,.25); padding:2px 8px;
               border-radius:4px; font-size:11px; font-weight:700; }}
.badge-short {{ background:rgba(255,69,96,.12);  color:{C['red']};
               border:1px solid rgba(255,69,96,.25);  padding:2px 8px;
               border-radius:4px; font-size:11px; font-weight:700; }}
.badge-pend  {{ background:rgba(90,122,154,.10); color:{C['sub']};
               border:1px solid {C['muted']}; padding:2px 8px;
               border-radius:4px; font-size:11px; }}
.sig-prob {{ font-size:11px; color:{C['sub']}; width:72px; text-align:right; }}
.sig-sent {{ font-size:11px; font-weight:600; width:52px; text-align:right; }}

/* SECTION TITLE */
.sec {{ font-size:11px; font-weight:700; color:{C['sub']};
        text-transform:uppercase; letter-spacing:1px;
        margin-bottom:14px; margin-top:4px; }}

/* TICKER STRIP */
.tstrip {{
    display:flex; gap:36px; padding:12px 4rem;
    background:{C['card']}; border-bottom:1px solid {C['border']};
    overflow-x:auto; scrollbar-width:none;
}}
.titem {{ display:flex; flex-direction:column; min-width:72px; }}
.tsym  {{ font-size:10px; font-weight:700; color:{C['sub']}; letter-spacing:.5px; }}
.tprc  {{ font-size:14px; font-weight:700; color:{C['text']}; }}
.tup   {{ font-size:10px; color:{C['green']}; font-weight:600; }}
.tdn   {{ font-size:10px; color:{C['red']};   font-weight:600; }}

/* SIDEBAR NAV ITEM */
.snav {{
    display:flex; align-items:center; gap:10px; padding:9px 12px;
    border-radius:8px; margin-bottom:2px; cursor:pointer;
    font-size:13px; font-weight:500; color:{C['sub']};
    border:1px solid transparent; transition:all 0.15s;
}}
.snav.active {{
    background:{C['card2']}; color:{C['teal']};
    border-color:{C['border2']};
}}
.snav:hover {{ color:{C['text']}; background:{C['card2']}; }}

/* SETTINGS FORM */
.settings-section {{ margin-bottom:24px; }}

/* News */
.nitem {{ padding:10px 0; border-bottom:1px solid {C['border']}; }}
.ndate {{ font-size:10px; color:{C['sub']}; }}
.nhead {{ font-size:12px; color:{C['text']}; line-height:1.4; margin:3px 0; }}

/* Divider */
.hdiv {{ height:1px; background:{C['border']}; margin:16px 0; }}

section[data-testid="stSidebar"] {{ display:none !important; }}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────
def load_config():
    d = {"tickers":["AAPL","MSFT","GOOGL","AMZN","NVDA","TSLA","META","JPM","BAC","AMD"],
         "active_tickers":["AAPL","MSFT","GOOGL","AMZN","NVDA","TSLA","META","JPM","BAC","AMD"],
         "ticker_names":{"AAPL":"Apple","MSFT":"Microsoft","GOOGL":"Alphabet",
                         "AMZN":"Amazon","NVDA":"Nvidia","TSLA":"Tesla",
                         "META":"Meta","JPM":"JPMorgan","BAC":"Bank of America","AMD":"AMD"}}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f: return {**d,**json.load(f)}
    return d

def save_config(c):
    with open(CONFIG_PATH,"w") as f: json.dump(c,f,indent=2)

cfg = load_config()

@st.cache_data(ttl=300)
def load_sentiment():
    p=f"{DATA_DIR}/news_with_sentiment.csv"
    if not os.path.exists(p): return pd.DataFrame()
    df=pd.read_csv(p); df["date"]=pd.to_datetime(df["date"]); return df

@st.cache_data(ttl=300)
def load_historical():
    p=f"{DATA_DIR}/dataset_historical.csv"
    if not os.path.exists(p): return pd.DataFrame()
    df=pd.read_csv(p); df["date"]=pd.to_datetime(df["date"]); return df

@st.cache_data(ttl=300)
def load_backtest():
    p=f"{DATA_DIR}/backtest_results.csv"
    if not os.path.exists(p): return pd.DataFrame()
    return pd.read_csv(p)

@st.cache_resource
def load_models():
    m={}
    for lc in LABEL_COLS:
        mp=f"{MODEL_DIR}/xgb_v2_{lc}.pkl"; sp=f"{MODEL_DIR}/scaler_v2_{lc}.pkl"
        if os.path.exists(mp) and os.path.exists(sp):
            m[lc]={"model":joblib.load(mp),"scaler":joblib.load(sp)}
    return m

def get_prob(hist_df, models, ticker, lc="label_1d"):
    m=models.get(lc)
    if not m or hist_df.empty: return None
    row=hist_df[hist_df["ticker"]==ticker]
    if row.empty: return None
    row=row.sort_values("date").iloc[-1:]
    X=m["scaler"].transform(row[HIST_FEAT].fillna(0))
    return float(m["model"].predict_proba(X)[0,1])

def sent_col(s):
    return C["green"] if s>0.1 else C["red"] if s<-0.1 else C["sub"]

def dark_chart(fig, h=340):
    fig.update_layout(height=h, paper_bgcolor=C["card"], plot_bgcolor=C["card"],
                      font_color=C["text"], showlegend=True,
                      legend=dict(bgcolor="rgba(0,0,0,0)", font_size=11),
                      margin=dict(l=0,r=0,t=24,b=0))
    fig.update_xaxes(gridcolor=C["border"], showgrid=True)
    fig.update_yaxes(gridcolor=C["border"], showgrid=True)
    return fig

# ── Session state ─────────────────────────────────────────
for k,v in [("page","Dashboard"),("ticker","AAPL")]:
    if k not in st.session_state: st.session_state[k]=v

sent_df  = load_sentiment()
hist_df  = load_historical()
bt_df    = load_backtest()
xgb_mdls = load_models()
TICKERS  = cfg["active_tickers"] or cfg["tickers"]

# ── TOP BAR ───────────────────────────────────────────────
st.markdown(f"""
<div class="topbar">
  <div style="display:flex;align-items:center;gap:10px">
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
      <path d="M3 17l4-8 4 4 4-6 4 10" stroke="#00D4FF" stroke-width="2"
            stroke-linecap="round" stroke-linejoin="round"/>
      <circle cx="20" cy="5" r="2" fill="#00E5A0"/>
    </svg>
    <div class="logo">FinAI <span>Stock Predictor</span></div>
  </div>
  <div style="display:flex;align-items:center;gap:20px">
    <div style="display:flex;align-items:center;gap:6px">
      <span style="width:6px;height:6px;background:#00E5A0;border-radius:50%;display:inline-block"></span>
      <span style="font-size:11px;color:{C['sub']}">Live</span>
    </div>
    <span style="font-size:11px;color:{C['border2']}">|</span>
    <span style="font-size:11px;color:{C['sub']}">{len(TICKERS)} stocks</span>
    <span style="font-size:11px;color:{C['border2']}">|</span>
    <span style="font-size:11px;color:{C['sub']}">{len(hist_df):,} rows</span>
    <span style="font-size:11px;color:{C['border2']}">|</span>
    <span style="font-size:11px;color:{C['sub']}">FinBERT · XGBoost · LSTM</span>
  </div>
</div>""", unsafe_allow_html=True)

# ── TICKER STRIP ──────────────────────────────────────────
if not hist_df.empty:
    lp = hist_df[hist_df["ticker"].isin(TICKERS)].sort_values("date").groupby("ticker").last().reset_index()
    items="".join([f"""
<div class="titem">
  <span class="tsym">{r['ticker']}</span>
  <span class="tprc">${r['close']:.2f}</span>
  <span class="{'tup' if (r.get('ret_1d',0) or 0)>=0 else 'tdn'}">
    {'▲' if (r.get('ret_1d',0) or 0)>=0 else '▼'} {abs((r.get('ret_1d',0) or 0)*100):.2f}%
  </span>
</div>""" for _,r in lp.iterrows()])
    st.markdown(f'<div class="tstrip">{items}</div>', unsafe_allow_html=True)

# ── NAV BUTTONS — Apple pill style ───────────────────────
pages = [("Dashboard","dashboard"),("Analysis","analysis"),("Deep Dive","deepdive")]
page_labels = {
    "Dashboard": "Dashboard",
    "Analysis":  "Analysis",
    "Deep Dive": "Deep Dive",
}

# Apple-style segmented control via columns
nav_cols = st.columns([1,1,1,4])
for i,(p,_) in enumerate(pages):
    with nav_cols[i]:
        is_active = st.session_state.page == p
        # Render active state via HTML overlay trick
        st.markdown(f"""
<style>
div[data-testid="column"]:nth-child({i+1}) button {{
    background: {"rgba(0,212,255,0.12)" if is_active else "transparent"} !important;
    border: {"1px solid rgba(0,212,255,0.4)" if is_active else "1px solid transparent"} !important;
    color: {"#00D4FF" if is_active else "#5A7A9A"} !important;
    font-weight: {"600" if is_active else "400"} !important;
    font-size: 13px !important;
    border-radius: 8px !important;
    transition: all 0.2s !important;
}}
div[data-testid="column"]:nth-child({i+1}) button:hover {{
    color: #E2E8F0 !important;
    background: rgba(255,255,255,0.06) !important;
    border-color: rgba(255,255,255,0.1) !important;
}}
</style>
""", unsafe_allow_html=True)
        if st.button(p, key=f"nav_{p}", use_container_width=True):
            st.session_state.page=p; st.rerun()

page = st.session_state.page

# ════════════════════════════════════════════════════════════
# DASHBOARD — Overview (left) + Settings (right sidebar)
# ════════════════════════════════════════════════════════════
if page == "Dashboard":
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)

    STEPS=[("📰 News","crawler/news_spider.py"),
           ("🧠 FinBERT","nlp/finbert_scorer.py"),
           ("📊 Dataset","dataset/build_historical_dataset.py"),
           ("🤖 XGBoost","models/train_xgboost_v2.py")]

    # ── KPI Row ───────────────────────────────────────────
    latest_sent=pd.DataFrame()
    if not sent_df.empty:
        latest_sent=sent_df[sent_df["ticker"].isin(TICKERS)].sort_values("date").groupby("ticker").last().reset_index()

    avg_sent=latest_sent["sentiment_score"].mean() if not latest_sent.empty else 0
    n_long=sum(1 for t in TICKERS if (get_prob(hist_df,xgb_mdls,t) or 0.5)>0.5)
    n_short=len(TICKERS)-n_long

    k1,k2,k3,k4=st.columns(4)
    for col,(lbl,val,sub,clr) in zip([k1,k2,k3,k4],[
        ("Market Sentiment",f"{avg_sent:+.3f}","Bullish" if avg_sent>0 else "Bearish",C["green"] if avg_sent>0 else C["red"]),
        ("Long Signals",str(n_long),f"of {len(TICKERS)} stocks",C["green"]),
        ("Short Signals",str(n_short),f"of {len(TICKERS)} stocks",C["red"]),
        ("Data Points",f"{len(hist_df):,}" if not hist_df.empty else "—","historical rows",C["teal"]),
    ]):
        with col:
            st.markdown(f"""
<div class="card-sm kpi">
  <div class="kpi-label">{lbl}</div>
  <div class="kpi-value">{val}</div>
  <div class="kpi-sub" style="color:{clr}">{sub}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    # ── Signal List + Heatmap (side by side, equal weight) ──
    sig_col, heat_col = st.columns([1, 2], gap="large")

    with sig_col:
        st.markdown(f'<div class="sec">Latest Signals — T+1</div>', unsafe_allow_html=True)
        real_tickers=set(sent_df["ticker"].unique()) if not sent_df.empty else set()
        sent_lkp=latest_sent.set_index("ticker")[["sentiment_score"]].to_dict("index") if not latest_sent.empty else {}

        for t in TICKERS:
            prob=get_prob(hist_df,xgb_mdls,t)
            nm=cfg["ticker_names"].get(t,"")
            ss=float((sent_lkp.get(t,{}) or {}).get("sentiment_score",0))
            sc=sent_col(ss)
            pm="" if t in real_tickers else "*"

            if prob is None:
                badge='<span class="badge-pend">PENDING</span>'; pt="—"
            elif prob>0.5:
                badge='<span class="badge-long">▲ LONG</span>'; pt=f"P↑{prob:.2f}"
            else:
                badge='<span class="badge-short">▼ SHORT</span>'; pt=f"P↑{prob:.2f}"

            st.markdown(f"""
<div class="sig-row">
  <span class="sig-ticker">{t}</span>
  <span class="sig-name">{nm}{pm}</span>
  {badge}
  <span class="sig-prob">{pt}</span>
  <span class="sig-sent" style="color:{sc}">{ss:+.2f}</span>
</div>""", unsafe_allow_html=True)

    with heat_col:
        st.markdown(f'<div class="sec">Sentiment Heatmap — Last 20 Trading Days</div>', unsafe_allow_html=True)
        hm_src=hist_df[hist_df["ticker"].isin(TICKERS)][["ticker","date","sentiment_score"]].copy() if not hist_df.empty else pd.DataFrame()
        if not hm_src.empty:
            hm_src["date"]=pd.to_datetime(hm_src["date"]).dt.strftime("%Y-%m-%d")
            pv=hm_src.groupby(["ticker","date"])["sentiment_score"].mean().reset_index().pivot(
                index="ticker",columns="date",values="sentiment_score").fillna(0)
            pv=pv.iloc[:,-20:]
            pv.columns=[c[-5:] for c in pv.columns]
            real_set=set(sent_df["ticker"].unique()) if not sent_df.empty else set()
            pv.index=[f"{t}*" if t not in real_set else t for t in pv.index]
            fig=px.imshow(pv,
                color_continuous_scale=[[0,"#FF4560"],[0.5,C["card"]],[1,"#00E5A0"]],
                zmin=-1, zmax=1, aspect="auto", text_auto=".1f")
            fig.update_traces(textfont_size=10)
            fig.update_layout(
                height=max(320, len(TICKERS)*34),
                paper_bgcolor=C["card"], plot_bgcolor=C["card"],
                font_color=C["text"], font_size=11,
                coloraxis_showscale=True,
                coloraxis_colorbar=dict(
                    thickness=12, len=0.8,
                    tickfont=dict(size=10, color=C["sub"]),
                    title=dict(text="", side="right"),
                ),
                margin=dict(l=0,r=8,t=0,b=0),
                xaxis=dict(tickangle=-30, tickfont_size=10, side="bottom"),
                yaxis=dict(tickfont_size=11),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("* Proxy model sentiment (no real news yet)")

    # ── Bottom: Update + Stock Management ────────────────
    st.markdown(f"<div style='height:1px;background:{C['border']};margin:20px 0'></div>", unsafe_allow_html=True)

    bot_left, bot_mid, bot_right = st.columns([1.4, 1, 1], gap="large")

    # Pipeline
    with bot_left:
        st.markdown(f'<div class="sec">Update Data & Models</div>', unsafe_allow_html=True)
        if st.button("▶  Run Full Pipeline", type="primary", use_container_width=True):
            root=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
            pb=st.progress(0.0, text="Starting...")
            slots=[st.empty() for _ in STEPS]
            for i,(l,_) in enumerate(STEPS):
                slots[i].markdown(f"<div style='font-size:12px;color:{C['muted']};padding:2px 0'>⬜ {l}</div>",unsafe_allow_html=True)
            log,err=[],""
            for i,(l,s) in enumerate(STEPS):
                pb.progress(i/len(STEPS), text=f"{i+1}/{len(STEPS)}: {l}")
                slots[i].markdown(f"<div style='font-size:12px;color:{C['orange']};padding:2px 0'>⏳ {l}</div>",unsafe_allow_html=True)
                try:
                    r=subprocess.run([sys.executable,s],capture_output=True,text=True,cwd=root,timeout=600)
                    if r.returncode==0:
                        slots[i].markdown(f"<div style='font-size:12px;color:{C['green']};padding:2px 0'>✅ {l}</div>",unsafe_allow_html=True)
                        log.append(f"✅ {l}")
                    else:
                        e=r.stderr.strip().split("\n")[-1]
                        slots[i].markdown(f"<div style='font-size:12px;color:{C['red']};padding:2px 0'>❌ {l}</div>",unsafe_allow_html=True)
                        err=e; break
                except Exception as ex:
                    err=str(ex); break
            if not err:
                pb.progress(1.0, text="✅ Complete!")
                st.success("Pipeline complete! Refreshing...")
                st.cache_data.clear(); st.cache_resource.clear()
            else:
                st.error(f"Failed: {err}")
            if log: st.code("\n".join(log), language=None)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        step_c1, step_c2 = st.columns(2)
        for i,(l,s) in enumerate(STEPS):
            col = step_c1 if i%2==0 else step_c2
            with col:
                if st.button(l, use_container_width=True, key=f"ind_{s}"):
                    root=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
                    with st.spinner("Running..."):
                        r=subprocess.run([sys.executable,s],capture_output=True,text=True,cwd=root,timeout=600)
                    if r.returncode==0: st.success("✅ Done!"); st.cache_data.clear()
                    else: st.error(r.stderr.strip().split("\n")[-1])

        if st.button("Clear Cache", use_container_width=True):
            st.cache_data.clear(); st.cache_resource.clear()
            st.success("Cleared!"); st.rerun()

    # Active Stocks checkboxes
    with bot_mid:
        st.markdown(f'<div class="sec">Active Stocks</div>', unsafe_allow_html=True)
        new_active=[]
        for t in cfg["tickers"]:
            nm=cfg["ticker_names"].get(t,"")
            lbl=f"{t}  {nm}" if nm else t
            if st.checkbox(lbl, value=t in cfg["active_tickers"], key=f"chk_{t}"):
                new_active.append(t)
        if set(new_active)!=set(cfg["active_tickers"]):
            cfg["active_tickers"]=new_active; save_config(cfg); st.rerun()

    # Add / Remove stock
    with bot_right:
        st.markdown(f'<div class="sec">Add New Stock</div>', unsafe_allow_html=True)
        with st.form("add", clear_on_submit=True):
            nt=st.text_input("Ticker", placeholder="e.g. NFLX").upper().strip()
            nn=st.text_input("Company Name", placeholder="e.g. Netflix")
            if st.form_submit_button("Add Stock", use_container_width=True):
                if nt in cfg["tickers"]: st.warning(f"{nt} already exists")
                elif not nt.isalpha() or len(nt)>6: st.error("Invalid ticker")
                else:
                    cfg["tickers"].append(nt); cfg["active_tickers"].append(nt)
                    if nn: cfg["ticker_names"][nt]=nn
                    save_config(cfg); st.success(f"✅ {nt} added — Run pipeline to activate"); st.rerun()

        removable=[t for t in cfg["tickers"] if t not in
                   ["AAPL","MSFT","GOOGL","AMZN","NVDA","TSLA","META","JPM","BAC","AMD"]]
        if removable:
            st.markdown(f'<div class="sec" style="margin-top:16px">Remove Stock</div>', unsafe_allow_html=True)
            rc1, rc2 = st.columns([2,1])
            rm=rc1.selectbox("Remove", removable, label_visibility="collapsed")
            if rc2.button("Remove", use_container_width=True):
                cfg["tickers"].remove(rm)
                if rm in cfg["active_tickers"]: cfg["active_tickers"].remove(rm)
                cfg["ticker_names"].pop(rm,None); save_config(cfg); st.rerun()



elif page == "Analysis":
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)

    # ── Backtest first ────────────────────────────────────
    st.markdown(f'<div class="sec">Backtest Results — Jan–Jun 2026 · $100k Capital · 0.1% Cost/Trade</div>',unsafe_allow_html=True)


    if bt_df.empty:
        st.warning("No backtest results. Run `backtest/run_backtest.py` first.")
    else:
        bh=bt_df[bt_df["strategy"]=="Buy & Hold"].iloc[0]
        strs=bt_df[bt_df["strategy"]!="Buy & Hold"]
        best=strs.loc[strs["sharpe"].idxmax()]

        b1,b2,b3,b4=st.columns(4)
        for col,(lbl,val,sub,clr) in zip([b1,b2,b3,b4],[
            ("Best Strategy",best["strategy"].split(".")[1].strip() if "." in best["strategy"] else best["strategy"],f"Sharpe {best['sharpe']:.2f}",C["teal"]),
            ("Best Return",f"{strs['total_return'].max():+.1f}%",f"vs B&H {bh['total_return']:+.1f}%",C["green"]),
            ("Best Sharpe",f"{strs['sharpe'].max():.2f}","risk-adjusted",C["orange"]),
            ("Lowest Drawdown",f"{strs['max_drawdown'].max():.2f}%","least loss from peak",C["purple"]),
        ]):
            with col:
                st.markdown(f'<div class="card-sm kpi"><div class="kpi-label">{lbl}</div><div class="kpi-value" style="font-size:20px">{val}</div><div class="kpi-sub" style="color:{clr}">{sub}</div></div>',unsafe_allow_html=True)

        st.markdown("<br>",unsafe_allow_html=True)
        sc={"1. Sentiment Long/Short":C["teal"],"2. Momentum + Sentiment":C["orange"],
            "3. Mean Reversion":C["red"],"4. SNR + Sentiment":C["purple"],"Buy & Hold":C["muted"]}

        bc1,bc2=st.columns(2,gap="large")
        for col,(met,title,fmt) in zip([bc1,bc2],[
            ("total_return","Total Return (%)","▲" ),
            ("sharpe","Sharpe Ratio",""),
        ]):
            with col:
                fig=go.Figure(go.Bar(
                    x=bt_df["strategy"],y=bt_df[met],
                    marker_color=[sc.get(s,C["teal"]) for s in bt_df["strategy"]],
                    text=[f"{v:+.1f}%" if met=="total_return" else f"{v:.2f}" for v in bt_df[met]],
                    textposition="outside"))
                if met=="sharpe":
                    fig.add_hline(y=1.0,line_dash="dash",line_color=C["muted"],annotation_text="1.0")
                dark_chart(fig,280)
                fig.update_layout(showlegend=False,xaxis=dict(tickangle=-15,tickfont_size=10),
                    title=dict(text=title,font_size=12,font_color=C["sub"]))
                st.plotly_chart(fig,use_container_width=True)

        st.markdown(f'<div class="sec" style="margin-top:8px">Full Results Table</div>', unsafe_allow_html=True)
        d=bt_df.copy()
        d["total_return"]=d["total_return"].map(lambda x:f"{x:+.2f}%")
        d["sharpe"]=d["sharpe"].map(lambda x:f"{x:.3f}")
        d["max_drawdown"]=d["max_drawdown"].map(lambda x:f"{x:.2f}%")
        d["win_rate"]=d["win_rate"].map(lambda x:f"{x:.1f}%")
        st.dataframe(d.set_index("strategy"),use_container_width=True)
        st.markdown(f"""
<div style="background:rgba(255,159,28,0.08);border:1px solid rgba(255,159,28,0.25);
border-radius:8px;padding:10px 16px;margin-top:12px;font-size:12px;color:{C['sub']}">
⚠️ Sentiment proxy may introduce look-ahead bias. Past performance ≠ future results.
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="hdiv"></div>',unsafe_allow_html=True)

    # ── Models below ─────────────────────────────────────
    st.markdown(f'<div class="sec">Model Performance — Out-of-Sample Test Set (975 rows)</div>', unsafe_allow_html=True)

    xgb_r=pd.DataFrame([{"H":"T+1d","Acc":0.5395,"F1":0.5198,"AUC":0.5416},
                         {"H":"T+3d","Acc":0.5200,"F1":0.4742,"AUC":0.5666},
                         {"H":"T+5d","Acc":0.5467,"F1":0.5562,"AUC":0.5687}])
    lstm_r=pd.DataFrame([{"H":"T+1d","Acc":0.5500,"F1":0.5826,"AUC":0.5602},
                          {"H":"T+3d","Acc":0.5333,"F1":0.5211,"AUC":0.5544},
                          {"H":"T+5d","Acc":0.5603,"F1":0.6284,"AUC":0.5783}])

    mc1,mc2,mc3=st.columns(3)
    metric=mc1.radio("Metric",["Acc","F1","AUC"],horizontal=True)
    metric_full={"Acc":"Accuracy","F1":"F1 Score","AUC":"AUC-ROC"}[metric]

    m_left,m_right=st.columns([1.6,1],gap="large")

    with m_left:
        fig=go.Figure()
        fig.add_trace(go.Bar(name="XGBoost v2",x=xgb_r["H"],y=xgb_r[metric],
            marker_color=C["teal"],text=xgb_r[metric].map(lambda x:f"{x:.4f}"),textposition="outside"))
        fig.add_trace(go.Bar(name="LSTM",x=lstm_r["H"],y=lstm_r[metric],
            marker_color=C["orange"],text=lstm_r[metric].map(lambda x:f"{x:.4f}"),textposition="outside"))
        fig.add_hline(y=0.5,line_dash="dash",line_color=C["muted"],annotation_text="Random 50%",annotation_font_color=C["sub"])
        fig.update_layout(barmode="group",yaxis=dict(range=[0.45,0.70]))
        dark_chart(fig,320)
        fig.update_layout(title=dict(text=f"{metric_full}: XGBoost vs LSTM",font_size=12,font_color=C["sub"]))
        st.plotly_chart(fig,use_container_width=True)

    with m_right:
        st.markdown(f'<div class="sec">Results Table</div>',unsafe_allow_html=True)
        combined=pd.DataFrame([
            {"Model":"XGBoost","Horizon":"T+1d","Accuracy":0.5395,"F1":0.5198,"AUC-ROC":0.5416},
            {"Model":"XGBoost","Horizon":"T+3d","Accuracy":0.5200,"F1":0.4742,"AUC-ROC":0.5666},
            {"Model":"XGBoost","Horizon":"T+5d","Accuracy":0.5467,"F1":0.5562,"AUC-ROC":0.5687},
            {"Model":"LSTM",   "Horizon":"T+1d","Accuracy":0.5500,"F1":0.5826,"AUC-ROC":0.5602},
            {"Model":"LSTM",   "Horizon":"T+3d","Accuracy":0.5333,"F1":0.5211,"AUC-ROC":0.5544},
            {"Model":"LSTM",   "Horizon":"T+5d","Accuracy":0.5603,"F1":0.6284,"AUC-ROC":0.5783},
        ])
        st.dataframe(combined.set_index(["Model","Horizon"]),use_container_width=True)
        st.markdown("<br>",unsafe_allow_html=True)
        st.success("**T+1 & T+5** → LSTM wins")
        st.warning("**T+3** → XGBoost wins")
        st.info("📌 52–58% AUC beats random baseline")


# ════════════════════════════════════════════════════════════
# DEEP DIVE
# ════════════════════════════════════════════════════════════

elif page == "Deep Dive":
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
    dc1,dc2,dc3=st.columns([1,1,2])
    ticker=dc1.selectbox("Stock",TICKERS)
    horizon=dc2.selectbox("Horizon",["T+1 Tomorrow","T+3 (3 Days)","T+5 (5 Days)"])
    lc={"T+1 Tomorrow":"label_1d","T+3 (3 Days)":"label_3d","T+5 (5 Days)":"label_5d"}[horizon]

    t_hist=hist_df[hist_df["ticker"]==ticker].sort_values("date") if not hist_df.empty else pd.DataFrame()
    t_sent=sent_df[sent_df["ticker"]==ticker].sort_values("date") if not sent_df.empty else pd.DataFrame()
    prob=get_prob(hist_df,xgb_mdls,ticker,lc)

    # Banner
    if prob is not None:
        il=prob>0.5
        sig="▲  LONG  —  Price likely to rise" if il else "▼  SHORT  —  Price likely to fall"
        bc=C["green"] if il else C["red"]
        conf="High" if abs(prob-.5)>.15 else "Medium" if abs(prob-.5)>.07 else "Low"
        st.markdown(f"""
<div style="background:rgba({'0,229,160' if il else '255,69,96'},.07);
border:1px solid {bc};border-radius:10px;padding:12px 20px;margin-bottom:16px;
display:flex;align-items:center;justify-content:space-between">
<span style="font-size:17px;font-weight:700;color:{bc}">{sig}</span>
<span style="font-size:12px;color:{C['sub']}">
  {horizon} · P(UP)={prob:.3f} · P(DOWN)={1-prob:.3f} ·
  <span style="color:{C['orange']}">Confidence: {conf}</span>
</span>
</div>""",unsafe_allow_html=True)

    ch_col, info_col = st.columns([2.2, 1], gap="large")

    with ch_col:
        if not t_hist.empty:
            fig=make_subplots(rows=3,cols=1,shared_xaxes=True,
                              row_heights=[0.55,0.22,0.23],
                              subplot_titles=[f"{ticker}  Price + SMA20","FinBERT Sentiment","RSI (14)"],
                              vertical_spacing=0.05)
            fig.add_trace(go.Scatter(x=t_hist["date"],y=t_hist["close"],
                name="Price",line=dict(color=C["teal"],width=2)),row=1,col=1)
            if "sma20" in t_hist.columns:
                fig.add_trace(go.Scatter(x=t_hist["date"],y=t_hist["sma20"],
                    name="SMA20",line=dict(color=C["orange"],width=1,dash="dash")),row=1,col=1)
            sd=t_hist[["date","sentiment_score"]].dropna()
            fig.add_trace(go.Bar(x=sd["date"],y=sd["sentiment_score"],name="Sentiment",
                marker_color=[C["green"] if s>0 else C["red"] for s in sd["sentiment_score"]]),row=2,col=1)
            if "rsi14" in t_hist.columns:
                fig.add_trace(go.Scatter(x=t_hist["date"],y=t_hist["rsi14"],
                    name="RSI",line=dict(color=C["purple"],width=1.5)),row=3,col=1)
                fig.add_hrect(y0=70,y1=100,fillcolor="#FF4560",opacity=0.06,line_width=0,row=3,col=1)
                fig.add_hrect(y0=0, y1=30, fillcolor="#00E5A0",opacity=0.06,line_width=0,row=3,col=1)
                fig.add_hline(y=70,line_dash="dash",line_color=C["red"],  line_width=1,row=3,col=1)
                fig.add_hline(y=30,line_dash="dash",line_color=C["green"],line_width=1,row=3,col=1)
            dark_chart(fig,620); st.plotly_chart(fig,use_container_width=True)

    with info_col:
        # Gauge
        if prob is not None:
            fig_g=go.Figure(go.Indicator(
                mode="gauge+number+delta",value=prob*100,
                delta={"reference":50,"suffix":"%"},
                title={"text":f"P(UP) — {horizon}","font":{"color":C["text"],"size":12}},
                number={"suffix":"%","font":{"color":C["text"],"size":28}},
                gauge={"axis":{"range":[0,100],"tickcolor":C["sub"],"tickfont":{"color":C["sub"],"size":9}},
                       "bar":{"color":C["teal"]},"bgcolor":C["card2"],"bordercolor":C["border"],
                       "steps":[{"range":[0,40],"color":"rgba(255,69,96,.12)"},
                                 {"range":[40,60],"color":"rgba(75,98,128,.08)"},
                                 {"range":[60,100],"color":"rgba(0,229,160,.12)"}],
                       "threshold":{"line":{"color":"white","width":2},"value":50}},
            ))
            fig_g.update_layout(height=220,paper_bgcolor=C["card"],font_color=C["text"],margin=dict(l=16,r=16,t=28,b=0))
            st.plotly_chart(fig_g,use_container_width=True)

        # News
        st.markdown(f'<div class="sec">Recent News</div>',unsafe_allow_html=True)
        if not t_sent.empty:
            for _,row in t_sent.sort_values("date",ascending=False).head(10).iterrows():
                sc=row["sentiment_score"]
                sc_c=sent_col(sc)
                st.markdown(f"""
<div class="nitem">
  <span class="ndate">{str(row['date'])[:10]}</span>
  <div class="nhead">{row['headline'][:110]}{"…" if len(row['headline'])>110 else ""}</div>
  <span style="font-size:10px;font-weight:600;color:{sc_c}">{sc:+.3f}</span>
</div>""",unsafe_allow_html=True)
        else:
            st.markdown(f'<p style="color:{C["sub"]};font-size:12px">No news. Run the crawler.</p>',unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
