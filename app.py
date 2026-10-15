from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Swing Trade Portfolio (STP)", page_icon="📈", layout="wide")

REPO_NAME = "Swing Trade Portfolio (STP)"

PRESET_CONFIGS = {
    "Portfolio": {
        "tickers": "SPYI, QQQI, JAAA, JPST, SCYB",
        "buffer": 1.5,
        "rvol": 0.75,
        "min_atr": 0.25,
        "rationale": "Core portfolio allocation containing your income, credit, and safe-harbor rotation assets.",
    },
    "Watching": {
        "tickers": "VOO, VGT",
        "buffer": 2.0,
        "rvol": 1.00,
        "min_atr": 0.50,
        "rationale": "Core equity growth watch list for tracking broad market and tech recovery phases.",
    },
}

# --- Technical Helper Functions ---

def get_atr(df, window=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return float(tr.rolling(window).mean().iloc[-1])

def get_rsi(close, window=14):
    delta = close.diff()
    g = delta.where(delta > 0, 0).rolling(window).mean()
    l = (-delta.where(delta < 0, 0)).rolling(window).mean()
    return float((100 - (100 / (1 + (g / l)))).iloc[-1])

def get_adx(df, window=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    up, down = h - h.shift(1), l.shift(1) - l
    p_dm = np.where((up > down) & (up > 0), up, 0.0)
    m_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    tr_s = tr.rolling(window).mean()
    p_di = 100 * (pd.Series(p_dm, index=df.index).rolling(window).mean() / tr_s)
    m_di = 100 * (pd.Series(m_dm, index=df.index).rolling(window).mean() / tr_s)
    dx = 100 * (abs(p_di - m_di) / (p_di + m_di))
    return float(dx.rolling(window).mean().iloc[-1])

def get_stoch(df, window=14):
    l_min = df["Low"].rolling(window).min()
    h_max = df["High"].rolling(window).max()
    k = 100 * ((df["Close"] - l_min) / (h_max - l_min))
    return float(k.iloc[-1])

def get_bollinger(close, price):
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    u_bb = float((sma20 + (std20 * 2)).iloc[-1])
    l_bb = float((sma20 - (std20 * 2)).iloc[-1])
    bw = ((sma20 + (std20 * 2)) - (sma20 - (std20 * 2))) / sma20
    bw_c = float(bw.iloc[-1]) * 100
    bw_a = float(bw.rolling(50).mean().iloc[-1]) * 100
    bw_m = float(bw.rolling(50).min().iloc[-1]) * 100
    is_sq = bw_c <= (bw_m * 1.15)
    
    if is_sq:
        return f"SQUEEZE ({bw_c:.1f}%)", True
    if bw_c > (bw_a * 1.20):
        return f"EXPANDING VOL ({bw_c:.1f}%)", False
    if price >= u_bb:
        return f"UPPER BAND (${u_bb:.2f})", False
    if price <= l_bb:
        return f"LOWER BAND (${l_bb:.2f})", False
    return f"NORMAL ({bw_c:.1f}%)", False

def get_benchmark_regime(ticker_symbol):
    """Evaluates 50 SMA and 200 SMA benchmark hierarchy for SPY/QQQ linked assets."""
    bench = "SPY" if "SPY" in ticker_symbol or "VOO" in ticker_symbol else ("QQQ" if "QQQ" in ticker_symbol or "VGT" in ticker_symbol else ticker_symbol)
    try:
        df = yf.download(bench, period="1y", interval="1d", auto_adjust=True, progress=False)
        if df.empty:
            return "GREEN", "Full Risk-On"
        c = df["Close"].squeeze()
        price = float(c.iloc[-1])
        sma50 = float(c.rolling(50).mean().iloc[-1])
        sma200 = float(c.rolling(200).mean().iloc[-1])

        if price > sma200 and price > sma50:
            return "GREEN", f"Full Risk-On ({bench} > 50 SMA & 200 SMA)"
        elif price > sma200 and price <= sma50:
            return "YELLOW", f"Caution / Pullback ({bench} broke below 50 SMA)"
        else:
            return "RED", f"Full Bear ({bench} < 200 SMA)"
    except:
        return "GREEN", "Default"

def analyze_tickers(tickers, default_buffer_pct=0.020, rvol_threshold=0.80):
    action_results = []
    detail_results = []

    for t in tickers:
        try:
            df = yf.download(t, period="2y", interval="1d", auto_adjust=True, progress=False)
            if df.empty or len(df) < 205:
                continue

            c = df["Close"].squeeze()
            h = df["High"].squeeze()
            l = df["Low"].squeeze()
            o = df["Open"].squeeze()
            v = df["Volume"].squeeze()

            df_clean = pd.DataFrame({"High": h, "Low": l, "Close": c, "Open": o})
            price = float(c.iloc[-1])
            open_p = float(o.iloc[-1])
            vol = float(v.iloc[-1])

            v_sma20 = float(v.rolling(20).mean().iloc[-1])
            rvol = vol / v_sma20 if v_sma20 > 0 else 1.0
            rvol_ok = rvol >= rvol_threshold

            is_green = price >= open_p
            vol_status = "BULLISH (HEAVY BUY)" if (rvol >= 1.25 and is_green) else ("BEARISH (HEAVY SELL)" if (rvol >= 1.25 and not is_green) else "NORMAL / MODERATE")

            ema20 = float(c.ewm(span=20, adjust=False).mean().iloc[-1])
            sma50 = float(c.rolling(50).mean().iloc[-1])
            sma200_series = c.rolling(200).mean()
            p_sma200 = float(sma200_series.iloc[-1])

            atr = get_atr(df_clean)
            atr_pct = (atr / price) * 100
            rsi = get_rsi(c)
            adx = get_adx(df_clean)
            stoch = get_stoch(df_clean)

            dir_s = np.sign(c.diff()).fillna(0)
            obv_s = float((dir_s * v).cumsum().diff(10).iloc[-1])
            obv_label = "INFLOW" if obv_s > 0 else "OUTFLOW"
            bb_text, _ = get_bollinger(c, price)

            regime_color, regime_msg = get_benchmark_regime(t)

            if regime_color == "GREEN":
                status = "🟢 Bullish"
                existing_action = "HOLD EXISTING"
                new_action = "ACCUMULATE"
                msg = f"Market regime healthy. {regime_msg}."
            elif regime_color == "YELLOW":
                status = "🟡 Caution / Pullback"
                existing_action = "⚠️ TRIM 50% TO SAFE HARBOR"
                new_action = "WAIT (Trim Mode)"
                msg = f"Benchmark pullback detected. {regime_msg}. **Action:** Trim 50% of position to JAAA/JPST."
            else:
                status = "🔴 Bear / Risk-Off"
                existing_action = "🚨 ROTATE 100% TO SAFE HARBOR"
                new_action = "SELL"
                msg = f"Macro bear trigger hit. {regime_msg}. **Action:** Rotate 100% to SCYB/JAAA/JPST."

            pct_sma200 = ((price - p_sma200) / p_sma200) * 100.0
            ma_alignment = "ABOVE 20 & 50" if (price > ema20 and price > sma50) else "BELOW SHORT MAs"

            action_results.append({
                "Ticker": t,
                "Price": round(price, 2),
                "Macro Regime": status,
                "👉 Existing Position": existing_action,
                "🚀 New Entry": new_action,
                "% vs 200 SMA": f"{pct_sma200:+.2f}%",
                "Details & Rationale": msg,
            })

            detail_results.append({
                "Ticker": t,
                "RVOL": round(rvol, 2),
                "Vol State": vol_status,
                "RSI (14)": round(rsi, 1),
                "Stoch %K": round(stoch, 1),
                "ADX": round(adx, 1),
                "OBV Flow": obv_label,
                "Bollinger State": bb_text,
                "ATR (%)": round(atr_pct, 2),
                "Short MAs": ma_alignment,
            })
        except Exception as e:
            st.error(f"Error processing {t}: {str(e)}")

    return pd.DataFrame(action_results), pd.DataFrame(detail_results)

# --- Portfolio Rotation Engine ---
def evaluate_portfolio_rotation():
    st.subheader("🛡️ Dynamic Portfolio Rotation Engine (50 SMA & 200 SMA Matrix)")
    st.markdown("Evaluates benchmark indexes (**SPY** and **QQQ**) to determine whether to hold **Risk-On** or rotate to **Safe Harbor** (SCYB / JAAA / JPST).")

    benchmarks = ["SPY", "QQQ"]
    rotation_data = []

    for b in benchmarks:
        try:
            df = yf.download(b, period="1y", interval="1d", auto_adjust=True, progress=False)
            if df.empty:
                continue
            c = df["Close"].squeeze()
            price = float(c.iloc[-1])
            sma50 = float(c.rolling(50).mean().iloc[-1])
            sma200 = float(c.rolling(200).mean().iloc[-1])
            
            if price > sma200 and price > sma50:
                regime = "🟢 FULL RISK-ON"
                action = "Hold Risk-On Assets (SPYI / QQQI / VOO / VGT)"
            elif price > sma200 and price <= sma50:
                regime = "🟡 CAUTION / PULLBACK"
                action = "Trim 50% Exposure (to JAAA/JPST)"
            else:
                regime = "🔴 FULL RISK-OFF (BEAR)"
                action = "Rotate 100% to Safe Harbor (SCYB / JAAA / JPST)"

            rotation_data.append({
                "Benchmark": b,
                "Current Price": round(price, 2),
                "50 SMA": round(sma50, 2),
                "200 SMA": round(sma200, 2),
                "Market Regime": regime,
                "Rotation Guidance": action
            })
        except Exception as e:
            st.warning(f"Could not load data for {b}: {e}")

    if rotation_data:
        df_rot = pd.DataFrame(rotation_data)
        st.table(df_rot)

# --- UI Layout ---

st.title("📈 " + REPO_NAME)
st.caption("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

with st.expander("📖 **Methodology Guide & Portfolio Rotation Rules**"):
    st.markdown("### **Portfolio Rotation Hierarchy:**")
    st.markdown("* **50 SMA Lost:** Intermediate pullback. Trim 50% of risk-on exposure to JAAA/JPST while keeping 50% active for income.")
    st.markdown("* **200 SMA Lost:** Secular bear market. Rotate 100% to Safe Harbor (SCYB/JAAA/JPST).")

st.sidebar.header("Screener Configuration")

url_params = st.query_params
init_preset = url_params.get("preset", "Portfolio")
if init_preset not in PRESET_CONFIGS:
    init_preset = "Portfolio"

def on_preset_change():
    sel = st.session_state["preset_radio"]
    st.session_state["ticker_text"] = PRESET_CONFIGS[sel]["tickers"]
    st.query_params["preset"] = sel
    st.query_params["tickers"] = st.session_state["ticker_text"]

def on_ticker_change():
    st.query_params["preset"] = st.session_state["preset_radio"]
    st.query_params["tickers"] = st.session_state["ticker_text"]

if "preset_radio" not in st.session_state or st.session_state["preset_radio"] not in PRESET_CONFIGS:
    st.session_state["preset_radio"] = init_preset

if "ticker_text" not in st.session_state:
    st.session_state["ticker_text"] = url_params.get(
        "tickers", PRESET_CONFIGS[init_preset]["tickers"]
    )

run_screener = st.sidebar.button("🚀 Run Screener & Rotation Check", type="primary", use_container_width=True)
st.sidebar.divider()

selected_preset = st.sidebar.radio(
    "Target Instance Preset:",
    options=list(PRESET_CONFIGS.keys()),
    key="preset_radio",
    on_change=on_preset_change,
)

active_defaults = PRESET_CONFIGS[selected_preset]

ticker_input = st.sidebar.text_area(
    "Watchlist Tickers (Editable):",
    key="ticker_text",
    on_change=on_ticker_change,
    help="Edit tickers freely. Updates refresh browser URL for bookmarking.",
)

buffer_setting = st.sidebar.slider("Base Buffer Noise Filter (%)", 1.0, 5.0, active_defaults["buffer"], 0.5)
rvol_setting = st.sidebar.slider("Min RVOL Breakout Confirmation (x)", 0.1, 2.5, active_defaults["rvol"], 0.05)
min_atr_setting = st.sidebar.slider("Min Volatility / ATR (%)", 0.0, 5.0, active_defaults["min_atr"], 0.25)

evaluate_portfolio_rotation()
st.divider()

if run_screener:
    tickers_list = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    if tickers_list:
        with st.spinner("Analyzing market data & portfolio rotation signals..."):
            df_action, df_detail = analyze_tickers(
                tickers_list,
                default_buffer_pct=buffer_setting / 100.0,
                rvol_threshold=rvol_setting,
            )
        if not df_action.empty:
            st.subheader("📋 Primary Action Table (Hold vs. Rotate)")
            st.dataframe(df_action, use_container_width=True, hide_index=True)

            st.markdown("### 📊 Detailed Technical Metrics & Indicators")
            st.dataframe(df_detail, use_container_width=True, hide_index=True)

            st.divider()
            full_df = pd.merge(df_action, df_detail, on="Ticker")
            csv_data = full_df.to_csv(index=False)
            st.download_button(
                "📥 Download Full Combined CSV",
                csv_data,
                "swing_trade_portfolio_full.csv",
                "text/csv",
            )
