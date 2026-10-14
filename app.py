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

def get_logic_result(price, u_th, l_th, slope, adx, rsi, stoch, obv_s, is_sq, ema20, sma50, rvol_ok, rvol, buf):
    if price > u_th and slope in ["UP", "FLAT"]:
        if adx < 20:
            return "⚪ Consolidating", "WAIT", f"Above 200 EMA, but ADX ({adx:.1f}) < 20 signals choppy consolidation. Wait for trend strength expansion."
        if rsi > 70 or stoch > 80:
            return "🟡 Overbought", "HOLD", f"Macro UP, but short-term RSI ({rsi:.1f}) / Stoch ({stoch:.1f}) extended. Hold existing position, avoid new entry."
        if obv_s < 0 and rsi < 50:
            return "🔴 Divergence", "WARNING", "Negative OBV volume outflow signals distribution."
        if is_sq:
            return "🟡 Squeeze", "WATCH", "Bollinger Band Squeeze active—prepare for breakout expansion."
        if price > ema20 and price > sma50 and rvol_ok:
            return "🟢 Bullish", "ACCUMULATE", f"Confirmed entry: Macro UP + Short MAs aligned + RVOL ({rvol:.2f}x)."
        if price <= ema20 or price <= sma50:
            return "🟡 Pullback", "WAIT", "Macro UP, but price is pulling back below 20 EMA / 50 SMA. Wait for short-term bounce."
        return "🟡 Warning", "LOW VOLUME", f"Price above buffer, but short-term RVOL ({rvol:.2f}x) lacks volume confirmation."

    if price < l_th and slope == "DOWN":
        if adx < 20:
            return "⚪ Consolidating", "WEAK BEAR", f"Below 200 EMA, but weak ADX ({adx:.1f})."
        if rvol_ok:
            return "🔴 Bearish", "SELL", f"Selling volume confirmed below buffer (RVOL: {rvol:.2f}x)."
        return "🔴 Bearish", "SELL LOW VOL", f"Below buffer with DOWN slope (RVOL: {rvol:.2f}x)."

    if l_th <= price <= u_th:
        return "⚪ Neutral", "BUFFER ZONE", f"Price within +/-{buf*100:.1f}% noise buffer of 200 EMA."

    return "🟡 Warning", "CAUTION", f"Crossed 200 EMA but slope ({slope}) unconfirmed."

def analyze_tickers(tickers, default_buffer_pct=0.020, rvol_threshold=0.80):
    combined_results = []

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
            if rvol >= 1.25 and is_green:
                vol_status = "BULLISH (HEAVY BUY)"
            elif rvol >= 1.25 and not is_green:
                vol_status = "BEARISH (HEAVY SELL)"
            else:
                vol_status = "NORMAL / MODERATE"

            ema20 = float(c.ewm(span=20, adjust=False).mean().iloc[-1])
            sma50 = float(c.rolling(50).mean().iloc[-1])
            ema200 = c.ewm(span=200, adjust=False).mean()
            p_ema200 = float(ema200.iloc[-1])
            prev_ema200 = float(ema200.iloc[-6])

            atr = get_atr(df_clean)
            atr_pct = (atr / price) * 100
            eff_buf = max(default_buffer_pct, (atr / price) * 1.5)

            u_th = p_ema200 * (1.0 + eff_buf)
            l_th = p_ema200 * (1.0 - eff_buf)

            rsi = get_rsi(c)
            adx = get_adx(df_clean)
            stoch = get_stoch(df_clean)

            dir_s = np.sign(c.diff()).fillna(0)
            obv_s = float((dir_s * v).cumsum().diff(10).iloc[-1])
            obv_label = "INFLOW" if obv_s > 0 else "OUTFLOW"

            bb_text, is_sq = get_bollinger(c, price)

            ema_diff = ((p_ema200 - prev_ema200) / prev_ema200) * 100
            slope = "UP" if ema_diff > 0.05 else ("DOWN" if ema_diff < -0.05 else "FLAT")

            status, signal, msg = get_logic_result(
                price, u_th, l_th, slope, adx, rsi, stoch,
                obv_s, is_sq, ema20, sma50, rvol_ok, rvol, eff_buf
            )

            pct_ema200 = ((price - p_ema200) / p_ema200) * 100.0
            ma_alignment = "ABOVE 20 & 50" if (price > ema20 and price > sma50) else "BELOW SHORT MAs"

            combined_results.append({
                "Ticker": t,
                "Price": round(price, 2),
                "Macro Status": status,
                "Action Signal": signal,
                "% vs 200 EMA": f"{pct_ema200:+.2f}%",
                "200 EMA Slope": slope,
                "Short MAs": ma_alignment,
                "RVOL": round(rvol, 2),
                "RSI (14)": round(rsi, 1),
                "ADX": round(adx, 1),
                "Details & Rationale": msg,
            })
        except Exception as e:
            st.error(f"Error processing {t}: {str(e)}")

    return pd.DataFrame(combined_results)

# --- Portfolio Rotation Engine (50/200 EMA Rule) ---
def evaluate_portfolio_rotation():
    st.subheader("🛡️ Dynamic Portfolio Rotation Engine (50 & 200 EMA Trigger Matrix)")
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
            ema200 = float(c.ewm(span=200, adjust=False).mean().iloc[-1])
            
            if price > ema200 and price > sma50:
                regime = "🟢 FULL RISK-ON"
                action = "Hold Risk-On Assets (SPYI / QQQI / VOO / VGT)"
            elif price > ema200 and price <= sma50:
                regime = "🟡 CAUTION / PULLBACK"
                action = "Trim Risk-On Exposure (50% to JAAA/JPST)"
            else:
                regime = "🔴 FULL RISK-OFF (BEAR)"
                action = "Rotate 100% to Safe Harbor (SCYB / JAAA / JPST)"

            rotation_data.append({
                "Benchmark": b,
                "Current Price": round(price, 2),
                "50 SMA": round(sma50, 2),
                "200 EMA": round(ema200, 2),
                "Market Regime": regime,
                "Rotation Guidance": action
            })
        except Exception as e:
            st.warning(f"Could not load data for {b}: {e}")

    if rotation_data:
        df_rot = pd.DataFrame(rotation_data)
        st.table(df_rot)
    else:
        st.info("Loading rotation benchmark metrics...")

# --- UI Layout ---

st.title("📈 " + REPO_NAME)
st.caption("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

with st.expander("📖 **Methodology Guide & Portfolio Rotation Rules**"):
    st.markdown("### **1. Dynamic Portfolio Rotation (50/200 EMA)**")
    st.markdown("* **Full Risk-On:** Price > 200 EMA and > 50 SMA. Maintain full allocation to income/growth engines.")
    st.markdown("* **Caution / Warning:** Price > 200 EMA but breaks below 50 SMA. Trim exposure or rotate 50% to safe harbor.")
    st.markdown("* **Full Risk-Off:** Price < 200 EMA. Immediate rotation 100% into SCYB / JAAA / JPST safe harbor to prevent NAV erosion.")
    st.markdown("---")
    st.markdown("### **2. Preset Parameter Rationale**")

    p_data = [
        {
            "Preset Name": f"**{name}**",
            "Buffer (%)": f"{cfg['buffer']:.1f}%",
            "Min RVOL": f"{cfg['rvol']:.2f}x",
            "Min ATR (%)": f"{cfg['min_atr']:.2f}%",
            "Calibration Rationale": cfg["rationale"],
        }
        for name, cfg in PRESET_CONFIGS.items()
    ]
    st.table(pd.DataFrame(p_data))

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
        with st.spinner("Analyzing market data & rotation signals..."):
            df_combined = analyze_tickers(
                tickers_list,
                default_buffer_pct=buffer_setting / 100.0,
                rvol_threshold=rvol_setting,
            )
        if not df_combined.empty:
            st.subheader("📋 Unified Portfolio Matrix & Execution Table")
            st.dataframe(df_combined, use_container_width=True, hide_index=True)

            st.divider()
            csv_data = df_combined.to_csv(index=False)
            st.download_button(
                "📥 Download Unified Matrix CSV",
                csv_data,
                "swing_trade_portfolio_unified.csv",
                "text/csv",
            )
