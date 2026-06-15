# dashboard/app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import json
import os
import sys
import subprocess
import threading
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset.build_historical_dataset import FEATURE_COLS_HIST, LABEL_COLS

st.set_page_config(
    page_title="FinAI Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_DIR   = "models"
DATA_DIR    = "data"
CONFIG_PATH = "config.json"
HIST_FEAT   = FEATURE_COLS_HIST + ["ticker_enc"]

COLORS = {
    "positive": "#00C896",
    "negative": "#FF4B4B",
    "neutral":  "#8E9AAF",
    "bg":       "#0E1117",
    "card":     "#1E2130",
    "teal":     "#00B4D8",
    "orange":   "#FF9F1C",
    "purple":   "#A78BFA",
}

st.markdown("""
<style>
.signal-card {
    background: #1E2130; border-radius: 12px; padding: 16px;
    text-align: center; margin-bottom: 8px; border: 1px solid #2D3250;
}
.signal-card .ticker { font-size: 18px; font-weight: 700; color: #fff; }
.signal-card .signal { font-size: 22px; font-weight: 800; margin: 6px 0; }
.signal-card .prob   { font-size: 13px; color: #8E9AAF; margin: 0; }
.signal-card .sent   { font-size: 12px; margin-top: 4px; }
.long  { color: #00C896; }
.short { color: #FF4B4B; }
</style>
""", unsafe_allow_html=True)

# ── Config helpers ────────────────────────────────────────
def load_config() -> dict:
    default = {
        "tickers": ["AAPL","MSFT","GOOGL","AMZN","NVDA",
                    "TSLA","META","JPM","BAC","AMD"],
        "active_tickers": ["AAPL","MSFT","GOOGL","AMZN","NVDA",
                           "TSLA","META","JPM","BAC","AMD"],
        "ticker_names": {}
    }
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return {**default, **json.load(f)}
    return default

def save_config(cfg: dict):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)

# ── Data loaders ──────────────────────────────────────────
@st.cache_data
def load_sentiment():
    path = f"{DATA_DIR}/news_with_sentiment.csv"
    if not os.path.exists(path): return pd.DataFrame()
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data
def load_historical():
    path = f"{DATA_DIR}/dataset_historical.csv"
    if not os.path.exists(path): return pd.DataFrame()
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data
def load_backtest():
    path = f"{DATA_DIR}/backtest_results.csv"
    if not os.path.exists(path): return pd.DataFrame()
    return pd.read_csv(path)

@st.cache_resource
def load_xgb_models():
    models = {}
    for lc in LABEL_COLS:
        mp = f"{MODEL_DIR}/xgb_v2_{lc}.pkl"
        sp = f"{MODEL_DIR}/scaler_v2_{lc}.pkl"
        if os.path.exists(mp) and os.path.exists(sp):
            models[lc] = {"model": joblib.load(mp), "scaler": joblib.load(sp)}
    return models

def sentiment_emoji(label):
    return {"positive":"🟢","negative":"🔴","neutral":"⚪"}.get(label,"⚪")

def plotly_dark(fig, height=400, title=""):
    fig.update_layout(
        height=height, title=title,
        paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"],
        font_color="white",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="#2D3250")
    fig.update_yaxes(gridcolor="#2D3250")
    return fig

# ── Load config ───────────────────────────────────────────
cfg = load_config()

# ════════════════════════════════════════════════════════════
# SIDEBAR — Ticker Management
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📈 FinAI Predictor")
    st.caption("AI-Powered Financial News Analysis")
    st.divider()

    page = st.radio("Navigation", [
        "📊 Market Overview",
        "🔍 Stock Deep Dive",
        "🤖 Model Performance",
        "📈 Backtest Results",
    ], label_visibility="collapsed")

    st.divider()

    # ── Ticker Management ─────────────────────────────────
    st.markdown("### 🏢 Manage Stocks")

    # Show checkboxes for all tickers
    all_tickers     = cfg["tickers"]
    active_tickers  = cfg["active_tickers"]
    ticker_names    = cfg.get("ticker_names", {})

    new_active = []
    for t in all_tickers:
        name    = ticker_names.get(t, "")
        label   = f"**{t}**  {name}" if name else f"**{t}**"
        checked = t in active_tickers
        if st.checkbox(label, value=checked, key=f"chk_{t}"):
            new_active.append(t)

    # Save if changed
    if set(new_active) != set(active_tickers):
        cfg["active_tickers"] = new_active
        save_config(cfg)
        st.rerun()

    st.divider()

    # ── Add new ticker ────────────────────────────────────
    st.markdown("### ➕ Add New Stock")
    with st.form("add_ticker_form", clear_on_submit=True):
        new_ticker = st.text_input(
            "Ticker Symbol",
            placeholder="e.g. NFLX",
            help="Enter US stock ticker (e.g. NFLX, BABA, INTC)"
        ).upper().strip()
        new_name = st.text_input(
            "Company Name (optional)",
            placeholder="e.g. Netflix"
        ).strip()
        submitted = st.form_submit_button("Add Stock", use_container_width=True)

        if submitted and new_ticker:
            if new_ticker in cfg["tickers"]:
                st.warning(f"{new_ticker} already in list")
            elif len(new_ticker) > 6 or not new_ticker.isalpha():
                st.error("Invalid ticker format")
            else:
                cfg["tickers"].append(new_ticker)
                cfg["active_tickers"].append(new_ticker)
                if new_name:
                    cfg["ticker_names"][new_ticker] = new_name
                save_config(cfg)
                st.success(f"✅ {new_ticker} added!")
                st.info(
                    f"⚠️ To get predictions for {new_ticker}, "
                    f"re-run:\n"
                    f"1. `crawler/news_spider.py`\n"
                    f"2. `nlp/finbert_scorer.py`\n"
                    f"3. `dataset/build_historical_dataset.py`\n"
                    f"4. `models/train_xgboost_v2.py`"
                )
                st.rerun()

    # ── Remove ticker ─────────────────────────────────────
    st.divider()
    st.markdown("### 🗑️ Remove Stock")
    removable = [t for t in cfg["tickers"]
                 if t not in ["AAPL","MSFT","GOOGL","AMZN","NVDA",
                              "TSLA","META","JPM","BAC","AMD"]]
    if removable:
        to_remove = st.selectbox("Select to remove", removable)
        if st.button("Remove", type="secondary", use_container_width=True):
            cfg["tickers"].remove(to_remove)
            if to_remove in cfg["active_tickers"]:
                cfg["active_tickers"].remove(to_remove)
            cfg["ticker_names"].pop(to_remove, None)
            save_config(cfg)
            st.success(f"✅ {to_remove} removed")
            st.rerun()
    else:
        st.caption("Add custom stocks above to enable removal\n(Default 10 stocks are protected)")

    # ── Run Pipeline ──────────────────────────────────────
    st.divider()
    st.markdown("### ⚙️ Update Data & Models")
    st.caption("Re-fetches news, re-scores sentiment, rebuilds dataset and retrains models.")

    PIPELINE_STEPS = [
        ("📰 Crawling news...",       "crawler/news_spider.py"),
        ("🧠 Running FinBERT...",     "nlp/finbert_scorer.py"),
        ("📊 Building dataset...",    "dataset/build_historical_dataset.py"),
        ("🤖 Training XGBoost...",    "models/train_xgboost_v2.py"),
    ]

    # ── Session state init ────────────────────────────────
    for key, val in [("pipeline_running", False),
                     ("pipeline_step", 0),
                     ("pipeline_log", []),
                     ("pipeline_done", False),
                     ("pipeline_error", "")]:
        if key not in st.session_state:
            st.session_state[key] = val

    N_STEPS = len(PIPELINE_STEPS)

    if st.button("🔄 Run Full Pipeline", type="primary",
                 use_container_width=True):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        # Build progress UI directly in sidebar
        prog_bar  = st.progress(0.0, text="Starting pipeline...")
        step_slots = [st.empty() for _ in PIPELINE_STEPS]

        # Init all steps as pending
        for idx, (label, _) in enumerate(PIPELINE_STEPS):
            step_slots[idx].markdown(
                f"<div style='font-size:12px;color:#8E9AAF'>⬜ {label}</div>",
                unsafe_allow_html=True)

        log_lines = []
        error_msg = ""

        for idx, (label, script) in enumerate(PIPELINE_STEPS):
            fraction = idx / len(PIPELINE_STEPS)
            prog_bar.progress(fraction, text=f"Step {idx+1}/{len(PIPELINE_STEPS)}: {label}")

            # Mark current step as running
            step_slots[idx].markdown(
                f"<div style='font-size:12px;color:#FF9F1C'>⏳ {label}</div>",
                unsafe_allow_html=True)

            log_lines.append(f"▶ {label}")
            try:
                result = subprocess.run(
                    [sys.executable, script],
                    capture_output=True, text=True,
                    cwd=root, timeout=600
                )
                if result.returncode == 0:
                    log_lines.append("  ✅ Done")
                    step_slots[idx].markdown(
                        f"<div style='font-size:12px;color:#00C896'>✅ {label}</div>",
                        unsafe_allow_html=True)
                else:
                    err = result.stderr.strip().split("\n")[-1]
                    log_lines.append(f"  ❌ {err}")
                    step_slots[idx].markdown(
                        f"<div style='font-size:12px;color:#FF4B4B'>❌ {label}</div>",
                        unsafe_allow_html=True)
                    error_msg = err
                    break
            except subprocess.TimeoutExpired:
                log_lines.append("  ⏱️ Timeout")
                step_slots[idx].markdown(
                    f"<div style='font-size:12px;color:#FF4B4B'>⏱️ {label} (timeout)</div>",
                    unsafe_allow_html=True)
                error_msg = "Timeout"
                break
            except Exception as e:
                log_lines.append(f"  ❌ {e}")
                error_msg = str(e)
                break

        # Final state
        if not error_msg:
            prog_bar.progress(1.0, text="✅ All steps complete!")
            st.success("✅ Pipeline complete! Refreshing data...")
            st.cache_data.clear()
            st.cache_resource.clear()
            time.sleep(1)
            st.rerun()
        else:
            prog_bar.progress(
                (idx) / len(PIPELINE_STEPS),
                text=f"❌ Failed at step {idx+1}"
            )
            st.error(f"❌ Pipeline failed: {error_msg}")

        with st.expander("View log"):
            st.code("\n".join(log_lines), language=None)

    with st.expander("Run individual steps"):
        for label, script in PIPELINE_STEPS:
            btn_label = label.split("...")[0].strip()
            if st.button(btn_label, use_container_width=True, key=f"btn_{script}"):
                root = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), ".."))
                with st.spinner(f"Running {script}..."):
                    result = subprocess.run(
                        [sys.executable, script],
                        capture_output=True, text=True,
                        cwd=root, timeout=300
                    )
                if result.returncode == 0:
                    st.success(f"✅ {btn_label} done!")
                    st.cache_data.clear()
                    st.cache_resource.clear()
                else:
                    err = result.stderr.strip().split("\n")[-1]
                    st.error(f"❌ {err}")

    st.divider()
    st.caption("Data: Yahoo Finance + NewsAPI")
    st.caption("Model: FinBERT + XGBoost + LSTM")

# Use active tickers for all pages
TICKERS = cfg["active_tickers"] if cfg["active_tickers"] else cfg["tickers"]

# ════════════════════════════════════════════════════════════
# PAGE 1 — Market Overview
# ════════════════════════════════════════════════════════════
if page == "📊 Market Overview":
    st.title("📊 Market Overview")
    st.caption(f"Showing {len(TICKERS)} stocks · XGBoost T+1 predictions · FinBERT sentiment")

    sent_df = load_sentiment()
    hist_df = load_historical()
    models  = load_xgb_models()

    if sent_df.empty:
        st.warning("No sentiment data found. Run `nlp/finbert_scorer.py` first.")
        st.stop()

    # Filter to active tickers only
    sent_df = sent_df[sent_df["ticker"].isin(TICKERS)]
    hist_df = hist_df[hist_df["ticker"].isin(TICKERS)] if not hist_df.empty else hist_df

    latest_sent = (sent_df.sort_values("date")
                          .groupby("ticker").last().reset_index())

    # ── KPI Row ───────────────────────────────────────────
    avg_sent = latest_sent["sentiment_score"].mean()
    n_pos    = (latest_sent["sentiment_label"]=="positive").sum()
    n_neg    = (latest_sent["sentiment_label"]=="negative").sum()
    n_neu    = (latest_sent["sentiment_label"]=="neutral").sum()

    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Market Sentiment", f"{avg_sent:+.3f}",
              "🟢 Bullish" if avg_sent > 0 else "🔴 Bearish")
    k2.metric("🟢 Bullish", f"{n_pos} stocks")
    k3.metric("🔴 Bearish", f"{n_neg} stocks")
    k4.metric("⚪ Neutral",  f"{n_neu} stocks")

    st.divider()
    st.subheader("Latest Signals")

    # Get XGBoost predictions for tickers that have historical data
    tickers_with_data = set(hist_df["ticker"].unique()) if not hist_df.empty else set()
    latest_hist = pd.DataFrame()
    if not hist_df.empty:
        latest_hist = (hist_df[hist_df["ticker"].isin(TICKERS)]
                       .sort_values("date")
                       .groupby("ticker").last().reset_index())
        xgb_1d = models.get("label_1d")
        if xgb_1d and not latest_hist.empty:
            X    = xgb_1d["scaler"].transform(latest_hist[HIST_FEAT].fillna(0))
            prob = xgb_1d["model"].predict_proba(X)[:,1]
            latest_hist["prob_up"] = prob

    # Build per-ticker sentiment lookup
    sent_lookup = (latest_sent.set_index("ticker")[
                   ["sentiment_label","sentiment_score"]].to_dict("index"))

    # Signal cards — dynamic column count
    n_cols = min(5, len(TICKERS))
    cols   = st.columns(n_cols)

    for i, ticker_sym in enumerate(TICKERS):
        col  = cols[i % n_cols]
        name = cfg["ticker_names"].get(ticker_sym, "")
        has_data = ticker_sym in tickers_with_data

        # Get sentiment (real or neutral)
        sent_info = sent_lookup.get(ticker_sym, {})
        sent_l    = sent_info.get("sentiment_label", "neutral")
        sent_s    = float(sent_info.get("sentiment_score", 0.0))
        sent_e    = sentiment_emoji(sent_l)

        if not has_data:
            # Pending card for newly added tickers
            with col:
                st.markdown(f"""
<div class="signal-card">
  <div class="ticker">{ticker_sym}</div>
  {"<div style='font-size:11px;color:#8E9AAF;margin-bottom:4px'>" + name + "</div>" if name else ""}
  <div class="signal" style="color:#8E9AAF;font-size:16px">⏳ PENDING</div>
  <div class="prob" style="margin-top:6px">No model data yet</div>
  <div style="background:#2D3250;border-radius:4px;height:6px;margin:8px 0">
    <div style="background:#8E9AAF;width:50%;height:6px;border-radius:4px"></div>
  </div>
  <div class="sent" style="color:#8E9AAF;font-size:11px">Re-run pipeline to activate</div>
</div>
""", unsafe_allow_html=True)
            continue

        # Get model prediction
        hist_row = latest_hist[latest_hist["ticker"]==ticker_sym]
        prob_up  = float(hist_row["prob_up"].values[0]) \
                   if not hist_row.empty else 0.5
        is_long  = prob_up > 0.5
        sig_cls  = "long" if is_long else "short"
        sig_txt  = "▲ LONG" if is_long else "▼ SHORT"
        bar_pct  = int(prob_up * 100)
        bar_col  = "#00C896" if is_long else "#FF4B4B"

        with col:
            st.markdown(f"""
<div class="signal-card">
  <div class="ticker">{ticker_sym}</div>
  {"<div style='font-size:11px;color:#8E9AAF;margin-bottom:4px'>" + name + "</div>" if name else ""}
  <div class="signal {sig_cls}">{sig_txt}</div>
  <div class="prob">P(UP)={prob_up:.2f} | P(DN)={1-prob_up:.2f}</div>
  <div style="background:#2D3250;border-radius:4px;height:6px;margin:8px 0">
    <div style="background:{bar_col};width:{bar_pct}%;height:6px;border-radius:4px"></div>
  </div>
  <div class="sent">{sent_e} {sent_l.capitalize()} ({sent_s:+.2f})</div>
</div>
""", unsafe_allow_html=True)

    st.divider()

    # Heatmap
    st.subheader("Sentiment Heatmap — Last 20 Trading Days")
    hm = (sent_df.groupby(["ticker","date"])["sentiment_score"]
                 .mean().reset_index())
    hm_pivot = hm.pivot(index="ticker", columns="date",
                        values="sentiment_score").fillna(0)
    hm_pivot = hm_pivot.iloc[:, -20:]
    hm_pivot.columns = [str(c)[:10] for c in hm_pivot.columns]

    fig_heat = px.imshow(
        hm_pivot,
        color_continuous_scale=[[0,"#FF4B4B"],[0.5,"#1E2130"],[1,"#00C896"]],
        zmin=-1, zmax=1, aspect="auto", text_auto=".2f",
    )
    fig_heat.update_traces(textfont_size=9)
    plotly_dark(fig_heat, height=max(300, len(TICKERS)*35))
    st.plotly_chart(fig_heat, use_container_width=True)


# ════════════════════════════════════════════════════════════
# PAGE 2 — Stock Deep Dive
# ════════════════════════════════════════════════════════════
elif page == "🔍 Stock Deep Dive":
    st.title("🔍 Stock Deep Dive")

    c1, c2 = st.columns([1,2])
    ticker  = c1.selectbox("Ticker", TICKERS)
    horizon = c2.selectbox("Prediction Horizon", [
        "T+1 (Tomorrow)", "T+3 (3 Days)", "T+5 (5 Days)"])
    horizon_map = {"T+1 (Tomorrow)":"label_1d",
                   "T+3 (3 Days)":"label_3d",
                   "T+5 (5 Days)":"label_5d"}
    label_col = horizon_map[horizon]

    sent_df = load_sentiment()
    hist_df = load_historical()
    models  = load_xgb_models()

    if hist_df.empty:
        st.warning("No historical data."); st.stop()

    t_hist = hist_df[hist_df["ticker"]==ticker].sort_values("date")
    t_sent = sent_df[sent_df["ticker"]==ticker].sort_values("date") \
             if not sent_df.empty else pd.DataFrame()

    # Check if ticker has model data
    has_model_data = ticker in hist_df["ticker"].unique()

    if not has_model_data:
        st.warning(
            f"⚠️ **{ticker}** has no historical model data yet.\n\n"
            f"To generate predictions, re-run the pipeline:\n"
            f"1. `python crawler/news_spider.py`\n"
            f"2. `python nlp/finbert_scorer.py`\n"
            f"3. `python dataset/build_historical_dataset.py`\n"
            f"4. `python models/train_xgboost_v2.py`"
        )

    # Prediction banner
    xgb_info = models.get(label_col)
    if xgb_info and not t_hist.empty and has_model_data:
        last_row = t_hist.iloc[-1:]
        X        = xgb_info["scaler"].transform(
                    last_row[HIST_FEAT].fillna(0))
        prob_up  = xgb_info["model"].predict_proba(X)[0,1]
        is_long  = prob_up > 0.5
        signal   = "▲ LONG — Price likely to rise" if is_long \
                   else "▼ SHORT — Price likely to fall"
        color    = COLORS["positive"] if is_long else COLORS["negative"]
        conf     = ("High"   if abs(prob_up-0.5)>0.15
                    else "Medium" if abs(prob_up-0.5)>0.07
                    else "Low")

        st.markdown(f"""
<div style="background:{color}22;border:1.5px solid {color};border-radius:10px;
padding:16px 24px;margin-bottom:16px">
  <span style="font-size:22px;font-weight:800;color:{color}">{signal}</span><br>
  <span style="color:#ccc;font-size:14px">
    {horizon} &nbsp;|&nbsp; P(UP)={prob_up:.3f} &nbsp;|&nbsp;
    P(DOWN)={1-prob_up:.3f} &nbsp;|&nbsp; Confidence: {conf}
  </span>
</div>""", unsafe_allow_html=True)

    # Chart
    if not t_hist.empty:
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.55,0.22,0.23],
            subplot_titles=[f"{ticker} Price + SMA20",
                            "FinBERT Sentiment Score","RSI (14)"],
            vertical_spacing=0.06,
        )
        fig.add_trace(go.Scatter(
            x=t_hist["date"], y=t_hist["close"],
            name="Price", line=dict(color=COLORS["teal"],width=2)
        ), row=1, col=1)
        if "sma20" in t_hist.columns:
            fig.add_trace(go.Scatter(
                x=t_hist["date"], y=t_hist["sma20"],
                name="SMA20", line=dict(color=COLORS["orange"],
                                        width=1,dash="dash")
            ), row=1, col=1)

        sent_daily = t_hist[["date","sentiment_score"]].dropna()
        bar_colors = [COLORS["positive"] if s>0 else COLORS["negative"]
                      for s in sent_daily["sentiment_score"]]
        fig.add_trace(go.Bar(
            x=sent_daily["date"], y=sent_daily["sentiment_score"],
            name="Sentiment", marker_color=bar_colors,
        ), row=2, col=1)
        fig.add_hline(y=0.1,  line_dash="dot",
                      line_color="rgba(0,200,150,0.4)", row=2, col=1)
        fig.add_hline(y=-0.1, line_dash="dot",
                      line_color="rgba(255,75,75,0.4)",  row=2, col=1)

        if "rsi14" in t_hist.columns:
            fig.add_trace(go.Scatter(
                x=t_hist["date"], y=t_hist["rsi14"],
                name="RSI14", line=dict(color=COLORS["purple"],width=1.5)
            ), row=3, col=1)
            fig.add_hrect(y0=70, y1=100, line_width=0,
                          fillcolor="#FF4B4B", opacity=0.08, row=3, col=1)
            fig.add_hrect(y0=0,  y1=30,  line_width=0,
                          fillcolor="#00C896", opacity=0.08, row=3, col=1)
            fig.add_hline(y=70, line_dash="dash",
                          line_color="#FF4B4B", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash",
                          line_color="#00C896", row=3, col=1)

        plotly_dark(fig, height=720)
        st.plotly_chart(fig, use_container_width=True)

    # Gauge
    if xgb_info and not t_hist.empty and has_model_data:
        st.subheader("Prediction Confidence")
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob_up*100,
            delta={"reference":50,"suffix":"%"},
            title={"text":f"P(UP) — {horizon}","font":{"color":"white"}},
            number={"suffix":"%","font":{"color":"white"}},
            gauge={
                "axis":{"range":[0,100],"tickcolor":"white",
                        "tickfont":{"color":"white"}},
                "bar":{"color":COLORS["teal"]},
                "bgcolor":"#1E2130",
                "bordercolor":"#2D3250",
                "steps":[
                    {"range":[0,40],  "color":"rgba(255,75,75,0.25)"},
                    {"range":[40,60], "color":"rgba(142,154,175,0.15)"},
                    {"range":[60,100],"color":"rgba(0,200,150,0.25)"},
                ],
                "threshold":{"line":{"color":"white","width":3},"value":50},
            },
        ))
        fig_g.update_layout(height=280, paper_bgcolor=COLORS["bg"],
                            font_color="white")
        st.plotly_chart(fig_g, use_container_width=True)

    # News
    if not t_sent.empty:
        st.subheader("📰 Recent News")
        recent = t_sent[["date","headline","sentiment_label",
                          "sentiment_score"]].sort_values(
                          "date", ascending=False).head(8)
        for _, row in recent.iterrows():
            emoji = sentiment_emoji(row["sentiment_label"])
            score = row["sentiment_score"]
            color = (COLORS["positive"] if score>0.1
                     else COLORS["negative"] if score<-0.1
                     else COLORS["neutral"])
            st.markdown(f"""
<div style="padding:10px 14px;margin:4px 0;background:#1E2130;
border-radius:8px;border-left:3px solid {color}">
<span style="color:#8E9AAF;font-size:12px">{str(row['date'])[:10]}</span>
&nbsp; {emoji} &nbsp;
<span style="color:white">{row['headline']}</span>
&nbsp; <span style="color:{color};font-size:12px">({score:+.3f})</span>
</div>""", unsafe_allow_html=True)
    elif has_model_data:
        st.info("No news data for this ticker. Run the crawler to fetch news.")


# ════════════════════════════════════════════════════════════
# PAGE 3 — Model Performance
# ════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance")
    st.caption("XGBoost v2 vs LSTM — Out-of-sample test set (975 rows)")

    xgb_res  = pd.DataFrame([
        {"Horizon":"T+1d","Accuracy":0.5395,"F1":0.5198,"AUC-ROC":0.5416},
        {"Horizon":"T+3d","Accuracy":0.5200,"F1":0.4742,"AUC-ROC":0.5666},
        {"Horizon":"T+5d","Accuracy":0.5467,"F1":0.5562,"AUC-ROC":0.5687},
    ])
    lstm_res = pd.DataFrame([
        {"Horizon":"T+1d","Accuracy":0.5500,"F1":0.5826,"AUC-ROC":0.5602},
        {"Horizon":"T+3d","Accuracy":0.5333,"F1":0.5211,"AUC-ROC":0.5544},
        {"Horizon":"T+5d","Accuracy":0.5603,"F1":0.6284,"AUC-ROC":0.5783},
    ])

    st.info("📌 **Realistic expectations**: 52~58% accuracy is good for financial prediction. "
            "All models beat the random 50% baseline.")

    metric = st.radio("Metric", ["Accuracy","F1","AUC-ROC"], horizontal=True)

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        name="XGBoost v2", x=xgb_res["Horizon"], y=xgb_res[metric],
        marker_color=COLORS["teal"],
        text=xgb_res[metric].map(lambda x:f"{x:.4f}"),
        textposition="outside",
    ))
    fig_comp.add_trace(go.Bar(
        name="LSTM", x=lstm_res["Horizon"], y=lstm_res[metric],
        marker_color=COLORS["orange"],
        text=lstm_res[metric].map(lambda x:f"{x:.4f}"),
        textposition="outside",
    ))
    fig_comp.add_hline(y=0.5, line_dash="dash", line_color="#8E9AAF",
                       annotation_text="Random Baseline")
    fig_comp.update_layout(barmode="group",
                           yaxis=dict(range=[0.45,0.72]))
    plotly_dark(fig_comp, height=400,
                title=f"{metric}: XGBoost vs LSTM")
    st.plotly_chart(fig_comp, use_container_width=True)

    c1,c2 = st.columns(2)
    with c1:
        st.subheader("XGBoost v2")
        st.dataframe(xgb_res.set_index("Horizon"), use_container_width=True)
    with c2:
        st.subheader("LSTM")
        st.dataframe(lstm_res.set_index("Horizon"), use_container_width=True)

    st.divider()
    st.subheader("🏆 Winner by Metric")
    w1,w2,w3 = st.columns(3)
    w1.success("**T+1 AUC** → LSTM (0.5602 vs 0.5416)")
    w2.warning("**T+3 AUC** → XGBoost (0.5666 vs 0.5544)")
    w3.success("**T+5 AUC** → LSTM (0.5783 vs 0.5687)")


# ════════════════════════════════════════════════════════════
# PAGE 4 — Backtest Results
# ════════════════════════════════════════════════════════════
elif page == "📈 Backtest Results":
    st.title("📈 Backtest Results")
    st.caption("Jan–Jun 2026 · $100k initial capital · 0.1% transaction cost")

    bt_df = load_backtest()
    if bt_df.empty:
        st.warning("No backtest results. Run `backtest/run_backtest.py` first.")
        st.stop()

    bh         = bt_df[bt_df["strategy"]=="Buy & Hold"].iloc[0]
    strategies = bt_df[bt_df["strategy"]!="Buy & Hold"]
    best       = strategies.loc[strategies["sharpe"].idxmax()]

    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Best Strategy",
              best["strategy"].split(".")[1].strip()
              if "." in best["strategy"] else best["strategy"],
              f"Sharpe {best['sharpe']:.2f}")
    k2.metric("Best Return",   f"{strategies['total_return'].max():+.1f}%",
              f"vs B&H {bh['total_return']:+.1f}%")
    k3.metric("Best Sharpe",   f"{strategies['sharpe'].max():.2f}")
    k4.metric("Lowest Drawdown",
              f"{strategies['max_drawdown'].max():.2f}%")

    st.divider()

    s_colors = {
        "1. Sentiment Long/Short": COLORS["teal"],
        "2. Momentum + Sentiment": COLORS["orange"],
        "3. Mean Reversion":       COLORS["negative"],
        "4. SNR + Sentiment":      COLORS["purple"],
        "Buy & Hold":              COLORS["neutral"],
    }

    c1,c2 = st.columns(2)
    with c1:
        fig_r = go.Figure(go.Bar(
            x=bt_df["strategy"], y=bt_df["total_return"],
            marker_color=[s_colors.get(s,COLORS["teal"])
                          for s in bt_df["strategy"]],
            text=[f"{v:+.1f}%" for v in bt_df["total_return"]],
            textposition="outside",
        ))
        plotly_dark(fig_r, height=400, title="Total Return (%)")
        fig_r.update_xaxes(tickangle=-20)
        st.plotly_chart(fig_r, use_container_width=True)

    with c2:
        fig_s = go.Figure(go.Bar(
            x=bt_df["strategy"], y=bt_df["sharpe"],
            marker_color=[s_colors.get(s,COLORS["teal"])
                          for s in bt_df["strategy"]],
            text=[f"{v:.2f}" for v in bt_df["sharpe"]],
            textposition="outside",
        ))
        fig_s.add_hline(y=1.0, line_dash="dash", line_color="#8E9AAF",
                        annotation_text="Sharpe=1.0")
        plotly_dark(fig_s, height=400, title="Sharpe Ratio")
        fig_s.update_xaxes(tickangle=-20)
        st.plotly_chart(fig_s, use_container_width=True)

    fig_dd = go.Figure(go.Bar(
        x=bt_df["strategy"], y=bt_df["max_drawdown"],
        marker_color=[COLORS["negative"] if v<-5
                      else COLORS["orange"] if v<-2
                      else COLORS["positive"]
                      for v in bt_df["max_drawdown"]],
        text=[f"{v:.2f}%" for v in bt_df["max_drawdown"]],
        textposition="outside",
    ))
    plotly_dark(fig_dd, height=320,
                title="Max Drawdown (closer to 0% = better)")
    fig_dd.update_xaxes(tickangle=-20)
    st.plotly_chart(fig_dd, use_container_width=True)

    st.subheader("Full Results Table")
    disp = bt_df.copy()
    disp["total_return"] = disp["total_return"].map(lambda x:f"{x:+.2f}%")
    disp["sharpe"]       = disp["sharpe"].map(lambda x:f"{x:.3f}")
    disp["max_drawdown"] = disp["max_drawdown"].map(lambda x:f"{x:.2f}%")
    disp["win_rate"]     = disp["win_rate"].map(lambda x:f"{x:.1f}%")
    st.dataframe(disp.set_index("strategy"), use_container_width=True)

    st.divider()
    st.warning("⚠️ **Disclaimer**: Sentiment proxy model may introduce look-ahead bias. "
               "Past performance ≠ future results. 0.1% transaction cost included.")
