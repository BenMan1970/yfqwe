# forex_scanner_yf.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import time
import yfinance as yf
import traceback

st.set_page_config(page_title="Scanner Confluence Forex (yfinance)", page_icon="⭐", layout="wide")
st.title("🔍 Scanner Confluence Forex Premium (Données yfinance)")
st.markdown("*Utilisation de yfinance pour les données de marché H1*")

# Liste des paires forex supportées
FOREX_PAIRS_YF = [
    'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X',
    'AUDUSD=X', 'USDCAD=X', 'NZDUSD=X', 'EURJPY=X',
    'GBPJPY=X', 'EURGBP=X'
]

# Fonctions techniques
def ema(s, p): return s.ewm(span=p, adjust=False).mean()
def rma(s, p): return s.ewm(alpha=1/p, adjust=False).mean()

def hull_ma_pine(dc, p=20):
    hl = int(p / 2)
    sl = int(np.sqrt(p))
    wma1 = dc.rolling(window=hl).apply(lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), raw=True)
    wma2 = dc.rolling(window=p).apply(lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), raw=True)
    diff = 2 * wma1 - wma2
    return diff.rolling(window=sl).apply(lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), raw=True)

def rsi_pine(po4, p=10):
    d = po4.diff()
    g = d.where(d > 0, 0.0)
    l = -d.where(d < 0, 0.0)
    ag = rma(g, p)
    al = rma(l, p)
    rs = ag / al.replace(0, 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def adx_pine(h, l, c, p=14):
    tr1 = h - l
    tr2 = abs(h - c.shift(1))
    tr3 = abs(l - c.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = rma(tr, p)
    um = h.diff()
    dm = l.shift(1) - l
    pdm = pd.Series(np.where((um > dm) & (um > 0), um, 0.0), index=h.index)
    mdm = pd.Series(np.where((dm > um) & (dm > 0), dm, 0.0), index=h.index)
    satr = atr.replace(0, 1e-9)
    pdi = 100 * (rma(pdm, p) / satr)
    mdi = 100 * (rma(mdm, p) / satr)
    dxden = (pdi + mdi).replace(0, 1e-9)
    dx = 100 * (abs(pdi - mdi) / dxden)
    return rma(dx, p).fillna(0)

def heiken_ashi_pine(dfo):
    ha = pd.DataFrame(index=dfo.index)
    if dfo.empty:
        ha['HA_Open'] = pd.Series(dtype=float)
        ha['HA_Close'] = pd.Series(dtype=float)
        return ha['HA_Open'], ha['HA_Close']
    ha['HA_Close'] = (dfo['Open'] + dfo['High'] + dfo['Low'] + dfo['Close']) / 4
    ha['HA_Open'] = np.nan
    if not dfo.empty:
        ha.iloc[0, ha.columns.get_loc('HA_Open')] = (dfo['Open'].iloc[0] + dfo['Close'].iloc[0]) / 2
        for i in range(1, len(dfo)):
            ha.iloc[i, ha.columns.get_loc('HA_Open')] = (ha.iloc[i - 1, ha.columns.get_loc('HA_Open')] + ha.iloc[i - 1, ha.columns.get_loc('HA_Close')]) / 2
    return ha['HA_Open'], ha['HA_Close']

def smoothed_heiken_ashi_pine(dfo, l1=10, l2=10):
    eo = ema(dfo['Open'], l1)
    eh = ema(dfo['High'], l1)
    el = ema(dfo['Low'], l1)
    ec = ema(dfo['Close'], l1)
    hai = pd.DataFrame({'Open': eo, 'High': eh, 'Low': el, 'Close': ec}, index=dfo.index)
    hao_i, hac_i = heiken_ashi_pine(hai)
    sho = ema(hao_i, l2)
    shc = ema(hac_i, l2)
    return sho, shc

def ichimoku_pine_signal(df_high, df_low, df_close, tenkan_p=9, kijun_p=26, senkou_b_p=52):
    min_len_req = max(tenkan_p, kijun_p, senkou_b_p)
    if len(df_high) < min_len_req or len(df_low) < min_len_req or len(df_close) < min_len_req:
        print(f"Ichi:Data<{len(df_close)} vs req {min_len_req}.")
        return 0
    ts = (df_high.rolling(window=tenkan_p).max() + df_low.rolling(window=tenkan_p).min()) / 2
    ks = (df_high.rolling(window=kijun_p).max() + df_low.rolling(window=kijun_p).min()) / 2
    sa = (ts + ks) / 2
    sb = (df_high.rolling(window=senkou_b_p).max() + df_low.rolling(window=senkou_b_p).min()) / 2
    if pd.isna(df_close.iloc[-1]) or pd.isna(sa.iloc[-1]) or pd.isna(sb.iloc[-1]):
        print("Ichi:NaN close/spans.")
        return 0
    ccl = df_close.iloc[-1]
    cssa = sa.iloc[-1]
    cssb = sb.iloc[-1]
    ctn = max(cssa, cssb)
    cbn = min(cssa, cssb)
    sig = 0
    if ccl > ctn:
        sig = 1
    elif ccl < cbn:
        sig = -1
    return sig

@st.cache_data(ttl=300)
def get_data_yf(symbol_yf: str, interval_yf: str = '1h', period_yf: str = '30d'):
    try:
        ticker = yf.Ticker(symbol_yf)
        data = ticker.history(period=period_yf, interval=interval_yf)
        if data.empty or len(data) < 100:
            return None
        if data.index.tz is not None:
            data.index = data.index.tz_convert('UTC')
        else:
            data.index = data.index.tz_localize('UTC')
        return data.dropna()
    except Exception as e:
        return None

def calculate_all_signals_pine(data):
    if data is None or len(data) < 60:
        return None
    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in data.columns for col in required_cols):
        return None
    close = data['Close']
    high = data['High']
    low = data['Low']
    open_price = data['Open']
    ohlc4 = (open_price + high + low + close) / 4
    bull_confluences = 0
    bear_confluences = 0
    signal_details_pine = {}

    # HMA
    try:
        hma_series = hull_ma_pine(close, 20)
        if len(hma_series) >= 2 and not hma_series.iloc[-2:].isna().any():
            hma_val = hma_series.iloc[-1]
            hma_prev = hma_series.iloc[-2]
            if hma_val > hma_prev:
                bull_confluences += 1
                signal_details_pine['HMA'] = "▲"
            elif hma_val < hma_prev:
                bear_confluences += 1
                signal_details_pine['HMA'] = "▼"
            else:
                signal_details_pine['HMA'] = "─"
        else:
            signal_details_pine['HMA'] = "N/A"
    except:
        signal_details_pine['HMA'] = "Err"

    # RSI
    try:
        rsi_series = rsi_pine(ohlc4, 10)
        if len(rsi_series) >= 1 and not pd.isna(rsi_series.iloc[-1]):
            rsi_val = rsi_series.iloc[-1]
            signal_details_pine['RSI_val'] = f"{rsi_val:.0f}"
            if rsi_val > 50:
                bull_confluences += 1
                signal_details_pine['RSI'] = f"▲({rsi_val:.0f})"
            elif rsi_val < 50:
                bear_confluences += 1
                signal_details_pine['RSI'] = f"▼({rsi_val:.0f})"
            else:
                signal_details_pine['RSI'] = f"─({rsi_val:.0f})"
        else:
            signal_details_pine['RSI'] = "N/A"
    except:
        signal_details_pine['RSI'] = "Err"

    # ADX
    try:
        adx_series = adx_pine(high, low, close, 14)
        if len(adx_series) >= 1 and not pd.isna(adx_series.iloc[-1]):
            adx_val = adx_series.iloc[-1]
            signal_details_pine['ADX_val'] = f"{adx_val:.0f}"
            if adx_val >= 20:
                bull_confluences += 1
                bear_confluences += 1
                signal_details_pine['ADX'] = f"✔({adx_val:.0f})"
            else:
                signal_details_pine['ADX'] = f"✖({adx_val:.0f})"
        else:
            signal_details_pine['ADX'] = "N/A"
    except:
        signal_details_pine['ADX'] = "Err"

    # HA
    try:
        ha_open, ha_close = heiken_ashi_pine(data)
        if len(ha_open) >= 1 and len(ha_close) >= 1 and not pd.isna(ha_open.iloc[-1]) and not pd.isna(ha_close.iloc[-1]):
            if ha_close.iloc[-1] > ha_open.iloc[-1]:
                bull_confluences += 1
                signal_details_pine['HA'] = "▲"
            elif ha_close.iloc[-1] < ha_open.iloc[-1]:
                bear_confluences += 1
                signal_details_pine['HA'] = "▼"
            else:
                signal_details_pine['HA'] = "─"
        else:
            signal_details_pine['HA'] = "N/A"
    except:
        signal_details_pine['HA'] = "Err"

    # SHA
    try:
        sha_open, sha_close = smoothed_heiken_ashi_pine(data, 10, 10)
        if len(sha_open) >= 1 and len(sha_close) >= 1 and not pd.isna(sha_open.iloc[-1]) and not pd.isna(sha_close.iloc[-1]):
            if sha_close.iloc[-1] > sha_open.iloc[-1]:
                bull_confluences += 1
                signal_details_pine['SHA'] = "▲"
            elif sha_close.iloc[-1] < sha_open.iloc[-1]:
                bear_confluences += 1
                signal_details_pine['SHA'] = "▼"
            else:
                signal_details_pine['SHA'] = "─"
        else:
            signal_details_pine['SHA'] = "N/A"
    except:
        signal_details_pine['SHA'] = "Err"

    # Ichi
    try:
        ichimoku_signal_val = ichimoku_pine_signal(high, low, close)
        if ichimoku_signal_val == 1:
            bull_confluences += 1
            signal_details_pine['Ichi'] = "▲"
        elif ichimoku_signal_val == -1:
            bear_confluences += 1
            signal_details_pine['Ichi'] = "▼"
        else:
            signal_details_pine['Ichi'] = "─"
    except:
        signal_details_pine['Ichi'] = "Err"

    confluence_value = max(bull_confluences, bear_confluences)
    direction = "NEUTRE"
    if bull_confluences > bear_confluences:
        direction = "HAUSSIER"
    elif bear_confluences > bull_confluences:
        direction = "BAISSIER"
    elif bull_confluences == bear_confluences and bull_confluences > 0:
        direction = "CONFLIT"

    return {
        'confluence_P': confluence_value,
        'direction_P': direction,
        'bull_P': bull_confluences,
        'bear_P': bear_confluences,
        'rsi_P': signal_details_pine.get('RSI_val', "N/A"),
        'adx_P': signal_details_pine.get('ADX_val', "N/A"),
        'signals_P': signal_details_pine
    }

def get_stars_pine(confluence_value):
    if confluence_value == 6: return "⭐⭐⭐⭐⭐⭐"
    elif confluence_value == 5: return "⭐⭐⭐⭐⭐"
    elif confluence_value == 4: return "⭐⭐⭐⭐"
    elif confluence_value == 3: return "⭐⭐⭐"
    elif confluence_value == 2: return "⭐⭐"
    elif confluence_value == 1: return "⭐"
    else: return "WAIT"

# Interface utilisateur
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("⚙️ Paramètres")
    min_conf = st.selectbox("Confluence min (0-6)", options=[0,1,2,3,4,5,6], index=3)
    show_all = st.checkbox("Voir toutes les paires")
    pair_to_debug = st.selectbox("🔍 Afficher OHLC pour:", ["Aucune"] + FOREX_PAIRS_YF, index=0)
    scan_btn = st.button("🔍 Scanner (Données yfinance H1)", type="primary", use_container_width=True)

with col2:
    if scan_btn:
        st.info("🔄 Scan en cours (yfinance H1)...")
        pr_res = []
        pb = st.progress(0)
        stx = st.empty()

        if pair_to_debug != "Aucune":
            st.subheader(f"Données OHLC pour {pair_to_debug} (yfinance):")
            debug_data = get_data_yf(pair_to_debug, interval_yf="1h", period_yf="5d")
            if debug_data is not None:
                st.dataframe(debug_data[['Open','High','Low','Close']].tail(10))
            st.divider()

        for i, symbol_yf_scan in enumerate(FOREX_PAIRS_YF):
            pnd = symbol_yf_scan.replace('=X','').replace('=F','')
            cp = (i+1)/len(FOREX_PAIRS_YF)
            pb.progress(cp)
            stx.text(f"Analyse (yfinance H1): {pnd} ({i+1}/{len(FOREX_PAIRS_YF)})")
            d_h1_yf = get_data_yf(symbol_yf_scan, interval_yf="1h", period_yf="30d")
            if d_h1_yf is not None:
                sigs = calculate_all_signals_pine(d_h1_yf)
                if sigs:
                    strs = get_stars_pine(sigs['confluence_P'])
                    rd = {
                        'Paire': pnd,
                        'Direction': sigs['direction_P'],
                        'Conf. (0-6)': sigs['confluence_P'],
                        'Étoiles': strs,
                        'RSI': sigs['rsi_P'],
                        'ADX': sigs['adx_P'],
                        'Bull': sigs['bull_P'],
                        'Bear': sigs['bear_P'],
                        'details': sigs['signals_P']
                    }
                    pr_res.append(rd)
            time.sleep(0.5)

        pb.empty()
        stx.empty()

        if pr_res:
            dfa = pd.DataFrame(pr_res)
            dfd = dfa[dfa['Conf. (0-6)'] >= min_conf].copy() if not show_all else dfa.copy()
            if not dfd.empty:
                dfds = dfd.sort_values('Conf. (0-6)', ascending=False)
                vcs = [c for c in ['Paire','Direction','Conf. (0-6)','Étoiles','RSI','ADX','Bull','Bear'] if c in dfds.columns]
                st.dataframe(dfds[vcs], use_container_width=True, hide_index=True)
                with st.expander("📊 Détails des signaux"):
                    for _, r in dfds.iterrows():
                        sm = r.get('details', {})
                        if not isinstance(sm, dict): sm = {'Info': 'Détails non dispo'}
                        st.write(f"**{r.get('Paire','N/A')}** - {r.get('Étoiles','N/A')} ({r.get('Conf. (0-6)','N/A')}) - Dir: {r.get('Direction','N/A')}")
                        dc = st.columns(6)
                        so = ['HMA', 'RSI', 'ADX', 'HA', 'SHA', 'Ichi']
                        for idx, sk in enumerate(so):
                            dc[idx].metric(label=sk, value=sm.get(sk, "N/P"))
                        st.divider()
            else:
                st.warning("❌ Aucune paire ne correspond aux critères.")
        else:
            st.error("❌ Aucune donnée traitée.")

with st.expander("ℹ️ Comment ça marche"):
    st.markdown("**6 Signaux Confluence:** HMA(20), RSI(10), ADX(14)>=20, HA(Simple), SHA(10,10), Ichi(9,26,52). **Comptage & Étoiles:** Pine Script. **Source:** yfinance API.")

st.caption("Scanner H1 (yfinance). Multi-TF non actif.")
