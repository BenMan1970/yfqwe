import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Configuration
DATA_DIR = "data"
st.set_page_config(page_title="Scanner Forex Local Optimisé", page_icon="📊", layout="wide")
st.title("🔍 Scanner Confluence Forex (Données Locales)")
st.markdown("*Utilise des fichiers CSV au lieu de yfinance – Plus rapide et gratuit*")

# Liste des paires forex
FOREX_PAIRS_LOCAL = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF',
    'AUDUSD', 'USDCAD', 'NZDUSD', 'EURJPY',
    'GBPJPY', 'EURGBP'
]

# === FONCTIONS TECHNIQUES (identiques à ton code original) ===
def ema(s, p): return s.ewm(span=p, adjust=False).mean()
def rma(s, p): return s.ewm(alpha=1/p, adjust=False).mean()

def hull_ma_pine(dc, p=20):
    hl=int(p/2); sl=int(np.sqrt(p))
    wma1=dc.rolling(window=hl).apply(lambda x:np.sum(x*np.arange(1,len(x)+1))/np.sum(np.arange(1,len(x)+1)),raw=True)
    wma2=dc.rolling(window=p).apply(lambda x:np.sum(x*np.arange(1,len(x)+1))/np.sum(np.arange(1,len(x)+1)),raw=True)
    diff=2*wma1-wma2; return diff.rolling(window=sl).apply(lambda x:np.sum(x*np.arange(1,len(x)+1))/np.sum(np.arange(1,len(x)+1)),raw=True)

def rsi_pine(po4,p=10):
    d=po4.diff();g=d.where(d>0,0.0);l=-d.where(d<0,0.0)
    ag=rma(g,p);al=rma(l,p);rs=ag/al.replace(0,1e-9)
    rsi=100-(100/(1+rs));return rsi.fillna(50)

def adx_pine(h,l,c,p=14):
    tr1=h-l;tr2=abs(h-c.shift(1));tr3=abs(l-c.shift(1))
    tr=pd.concat([tr1,tr2,tr3],axis=1).max(axis=1);atr=rma(tr,p)
    um=h.diff();dm=l.shift(1)-l
    pdm=pd.Series(np.where((um>dm)&(um>0),um,0.0),index=h.index)
    mdm=pd.Series(np.where((dm>um)&(dm>0),dm,0.0),index=h.index)
    satr=atr.replace(0,1e-9);pdi=100*(rma(pdm,p)/satr);mdi=100*(rma(mdm,p)/satr)
    dxden=(pdi+mdi).replace(0,1e-9);dx=100*(abs(pdi-mdi)/dxden)
    return rma(dx,p).fillna(0)

def heiken_ashi_pine(dfo):
    ha=pd.DataFrame(index=dfo.index)
    if dfo.empty:
        ha['HA_Open']=pd.Series(dtype=float);ha['HA_Close']=pd.Series(dtype=float)
        return ha['HA_Open'],ha['HA_Close']
    ha['HA_Close']=(dfo['Open']+dfo['High']+dfo['Low']+dfo['Close'])/4;ha['HA_Open']=np.nan
    if not dfo.empty:
        ha.iloc[0,ha.columns.get_loc('HA_Open')]=(dfo['Open'].iloc[0]+dfo['Close'].iloc[0])/2
        for i in range(1,len(dfo)):
            ha.iloc[i,ha.columns.get_loc('HA_Open')] = (ha.iloc[i-1, ha.columns.get_loc('HA_Open')] + ha.iloc[i-1, ha.columns.get_loc('HA_Close')]) / 2
    return ha['HA_Open'],ha['HA_Close']

def smoothed_heiken_ashi_pine(dfo,l1=10,l2=10):
    eo=ema(dfo['Open'],l1);eh=ema(dfo['High'],l1);el=ema(dfo['Low'],l1);ec=ema(dfo['Close'],l1)
    hai=pd.DataFrame({'Open':eo,'High':eh,'Low':el,'Close':ec},index=dfo.index)
    hao_i,hac_i=heiken_ashi_pine(hai);sho=ema(hao_i,l2);shc=ema(hac_i,l2);return sho,shc

def ichimoku_pine_signal(df_high, df_low, df_close, tenkan_p=9, kijun_p=26, senkou_b_p=52):
    min_len_req=max(tenkan_p,kijun_p,senkou_b_p)
    if len(df_high)<min_len_req or len(df_low)<min_len_req or len(df_close)<min_len_req:
        print(f"Ichi:Data<{len(df_close)} vs req {min_len_req}.");return 0
    ts=(df_high.rolling(window=tenkan_p).max()+df_low.rolling(window=tenkan_p).min())/2
    ks=(df_high.rolling(window=kijun_p).max()+df_low.rolling(window=kijun_p).min())/2
    sa=(ts+ks)/2;sb=(df_high.rolling(window=senkou_b_p).max()+df_low.rolling(window=senkou_b_p).min())/2
    if pd.isna(df_close.iloc[-1]) or pd.isna(sa.iloc[-1]) or pd.isna(sb.iloc[-1]):
        print("Ichi:NaN close/spans.");return 0
    ccl=df_close.iloc[-1];cssa=sa.iloc[-1];cssb=sb.iloc[-1]
    ctn=max(cssa,cssb);cbn=min(cssa,cssb);sig=0
    if ccl>ctn:sig=1
    elif ccl<cbn:sig=-1
    return sig

# === CHARGEMENT DES DONNÉES LOCALES ===
def load_local_data(pair_name):
    file_path = os.path.join(DATA_DIR, f"{pair_name}.csv")
    try:
        df = pd.read_csv(file_path)
        df.columns = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        return df[['Open', 'High', 'Low', 'Close']]
    except Exception as e:
        st.warning(f"Impossible de charger {file_path} : {e}")
        return None

# === ANALYSE DE CONFLUENCE ===
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

    # Calcul des indicateurs
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

# === INTERFACE STREAMLIT ===
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("⚙️ Paramètres")
    min_conf = st.selectbox("Confluence min (0-6)", options=[0,1,2,3,4,5,6], index=3)
    show_all = st.checkbox("Voir toutes les paires")
    pair_to_debug = st.selectbox("🔍 Afficher OHLC pour:", ["Aucune"] + FOREX_PAIRS_LOCAL)
    scan_btn = st.button("🔍 Scanner (Local)", type="primary", use_container_width=True)

with col2:
    if scan_btn:
        st.info("🔄 Scan en cours (données locales)..."); pr_res = []; pb = st.progress(0); stx = st.empty()
        if pair_to_debug != "Aucune":
            st.subheader(f"Données OHLC pour {pair_to_debug}:")
            debug_data = load_local_data(pair_to_debug)
            if debug_data is not None:
                st.dataframe(debug_data[['Open','High','Low','Close']].tail(10))
            st.divider()
        for i, pair in enumerate(FOREX_PAIRS_LOCAL):
            progress = (i+1) / len(FOREX_PAIRS_LOCAL)
            pb.progress(progress); stx.text(f"Analyse : {pair} ({i+1}/{len(FOREX_PAIRS_LOCAL)})")
            data = load_local_data(pair)
            if data is not None:
                signals = calculate_all_signals_pine(data)
                if signals:
                    strs = get_stars_pine(signals['confluence_P'])
                    pr_res.append({
                        'Paire': pair,
                        'Direction': signals['direction_P'],
                        'Conf. (0-6)': signals['confluence_P'],
                        'Étoiles': strs,
                        'Bull': signals['bull_P'],
                        'Bear': signals['bear_P'],
                        'details': signals['signals_P']
                    })
            time.sleep(0.1)
        pb.empty(); stx.empty()
        if pr_res:
            df = pd.DataFrame(pr_res)
            df_filtered = df[df['Conf. (0-6)'] >= min_conf] if not show_all else df.copy()
            if not df_filtered.empty:
                df_sorted = df_filtered.sort_values('Conf. (0-6)', ascending=False)
                st.success(f"🎯 {len(df_sorted)} paire(s) trouvée(s) avec une confluence ≥ {min_conf}")
                st.dataframe(df_sorted[['Paire', 'Direction', 'Conf. (0-6)', 'Étoiles']], use_container_width=True, hide_index=True)
            else:
                st.warning("❌ Aucune paire ne correspond aux critères.")
        else:
            st.error("❌ Aucune donnée traitée.")

with st.expander("ℹ️ Comment ça marche"):
    st.markdown("**Version utilisant des données locales CSV au lieu de yfinance. Tous les indicateurs sont conservés.**")
           
