"""
app.py — EOD Stock Analysis Dashboard & Multi-Bagger Screener (All-in-One)
=====================================================================
Versi Pro+ (disempurnakan):
- Filter Likuiditas (>10 Miliar), Bollinger Bands, Valuasi Spesifik Sektor
- PERBAIKAN AKURASI:
  * Position sizing kini mengecek modal (peringatan overbudget) + Risk/Reward
  * Support/Resistance selalu punya fallback level psikologis ATAS & BAWAH
    (tidak lagi mengarang resistance = harga x 1.1 saat saham di puncak)
  * auto_adjust=True agar MA tidak melompat saat dividen/split
  * Label Multi-Bagger memakai nilai numerik (bukan parsing string yang rapuh)
- TAMBAHAN:
  * Panel edukasi anatomi Candlestick (body, shadow/wick, bullish/bearish)
  * Garis Support (batas bawah) & Resistance (batas atas) tegas di chart
"""
import json
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

st.set_page_config(page_title="EOD Dashboard IDX", layout="wide", page_icon="📈")

# ============================================================
# MESIN ANALITIK TEKNIKAL & VOLUMETRIK
# ============================================================
def sma(s: pd.Series, n: int) -> pd.Series: return s.rolling(n).mean()
def ema(s: pd.Series, n: int) -> pd.Series: return s.ewm(span=n, adjust=False).mean()

def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    gain = d.clip(lower=0).ewm(alpha=1/n, min_periods=n).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1/n, min_periods=n).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))

def macd(s: pd.Series, fast=12, slow=26, signal=9):
    line = ema(s, fast) - ema(s, slow)
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, min_periods=n).mean()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Trend & Volume
    df["MA20"] = sma(df["Close"], 20)
    df["MA50"] = sma(df["Close"], 50)
    df["MA200"] = sma(df["Close"], 200)
    df["VolMA20"] = sma(df["Volume"], 20)
    df["VolRatio"] = df["Volume"] / df["VolMA20"].replace(0, np.nan)

    # Nilai Transaksi (Likuiditas)
    df["TxValue"] = df["Close"] * df["Volume"]
    df["AvgValue20"] = sma(df["TxValue"], 20)

    # Bollinger Bands
    std20 = df["Close"].rolling(20).std()
    df["Upper_BB"] = df["MA20"] + (2 * std20)
    df["Lower_BB"] = df["MA20"] - (2 * std20)

    # Momentum
    df["RSI"] = rsi(df["Close"], 14)
    df["MACD"], df["MACD_SIG"], df["MACD_HIST"] = macd(df["Close"])
    df["ATR"] = atr(df, 14)
    return df

def detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Deteksi pola candlestick klasik per baris. Menambah kolom boolean:
    Hammer, InvHammer, Doji, BullEngulf, BearEngulf, dan kolom 'Pattern'
    (label teks untuk baris terakhir yang terdeteksi).

    Definisi (berbasis rasio, tahan beda harga antar saham):
      body   = |Close - Open|
      range  = High - Low
      upper  = High - max(Open,Close)   (sumbu atas)
      lower  = min(Open,Close) - Low    (sumbu bawah)
    """
    d = df.copy()
    o, c, h, l = d["Open"], d["Close"], d["High"], d["Low"]
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    upper = h - o.combine(c, max)
    lower = o.combine(c, min) - l
    body_pct = body / rng          # porsi badan thd rentang
    bullish = c >= o

    # --- HAMMER: badan kecil di ATAS, sumbu bawah panjang (>=2x badan),
    #     sumbu atas pendek → penolakan harga rendah (potensi pembalikan naik).
    #     Diperiksa SEBELUM doji karena hammer juga berbadan kecil; yang
    #     membedakan adalah sumbu bawah yang dominan. ---
    small_body = body_pct < 0.35
    d["Hammer"] = small_body & (lower >= 2 * body) & (upper <= body)

    # --- INVERTED HAMMER: cerminan hammer, sumbu atas dominan ---
    d["InvHammer"] = small_body & (upper >= 2 * body) & (lower <= body)

    # --- DOJI: badan sangat kecil DAN kedua sumbu relatif seimbang
    #     (bukan hammer/inverted hammer yang sumbunya berat sebelah) ---
    d["Doji"] = (body_pct < 0.10) & (~d["Hammer"]) & (~d["InvHammer"])

    # --- BULLISH ENGULFING: candle hijau hari ini "menelan" badan merah kemarin ---
    prev_o, prev_c = o.shift(1), c.shift(1)
    prev_bear = prev_c < prev_o
    d["BullEngulf"] = bullish & prev_bear & (c >= prev_o) & (o <= prev_c) & (body > (prev_o - prev_c).abs())

    # --- BEARISH ENGULFING: kebalikannya (info tambahan, tanda waspada) ---
    prev_bull = prev_c > prev_o
    d["BearEngulf"] = (~bullish) & prev_bull & (o >= prev_c) & (c <= prev_o) & (body > (prev_c - prev_o).abs())

    # label teks per baris (prioritas: engulfing > hammer > doji)
    def label(row):
        if row["BullEngulf"]: return "Bullish Engulfing"
        if row["BearEngulf"]: return "Bearish Engulfing"
        if row["Hammer"]:     return "Hammer"
        if row["InvHammer"]:  return "Inverted Hammer"
        if row["Doji"]:       return "Doji"
        return ""
    # Hammer & InvHammer diprioritaskan; jika keduanya menyala (jarang), pilih
    # berdasarkan sumbu yang lebih panjang.
    both = d["Hammer"] & d["InvHammer"]
    d.loc[both & (lower >= upper), "InvHammer"] = False
    d.loc[both & (upper > lower), "Hammer"] = False
    d["Pattern"] = d.apply(label, axis=1)
    return d


def screen_score(df: pd.DataFrame, min_liquidity: float) -> dict:
    if len(df) < 60 or df[["MA50", "RSI", "ATR", "AvgValue20"]].iloc[-1].isna().any():
        return {"valid": False}

    last = df.iloc[-1]
    # Filter Likuiditas: Buang saham gorengan sepi
    if last["AvgValue20"] < min_liquidity:
        return {"valid": False, "reason": "Kurang Likuid"}

    close = last["Close"]
    score = 0

    # 1. Volume spike (Max 25)
    vr = last["VolRatio"]
    score += 25 if vr > 3 else (20 if vr > 2 else (15 if vr > 1.5 else (10 if vr > 1 else 0)))

    # 2. Tren MA (Max 20)
    if close > last["MA20"]: score += 7
    if last["MA20"] > last["MA50"]: score += 7
    if not np.isnan(last["MA200"]) and last["MA50"] > last["MA200"]: score += 6

    # 3. Bollinger Bands & Breakout (Max 25)
    hi20 = df["High"].iloc[-21:-1].max()
    if close > last["Upper_BB"]: score += 15       # Tembus pita atas BB
    elif close > last["Upper_BB"] * 0.98: score += 10
    if close > hi20: score += 10                   # Breakout resistance lokal

    # 4. RSI (Max 15)
    r = last["RSI"]
    score += 15 if 55 <= r <= 70 else (10 if 50 <= r < 55 else (5 if 70 < r <= 78 else 0))

    # 5. MACD (Max 15)
    if last["MACD"] > last["MACD_SIG"]:
        score += 15 if last["MACD"] > 0 else 10

    return {
        "valid": True, "score": int(score), "close": float(close),
        "vol_ratio": float(vr), "rsi": float(r), "avg_val": float(last["AvgValue20"]),
        "atr": float(last["ATR"]),
    }

# ============================================================
# POSISI & LEVEL HARGA
# ============================================================
def position_size_percent(modal, risk_pct, entry, stop):
    """Position sizing dengan pengecekan modal. 1 lot = 100 lembar.
    PERBAIKAN: kini menandai jika modal terpakai > modal (overbudget),
    dan membatasi lot agar tidak melebihi modal tersedia."""
    if entry <= 0 or stop <= 0 or entry <= stop:
        return {"valid": False, "reason": "Entry harus > Stop dan keduanya > 0."}
    risk_rp = modal * risk_pct / 100
    lots_by_risk = int((risk_rp / (entry - stop)) // 100)
    if lots_by_risk < 1:
        return {"valid": False, "reason": "Risiko terlalu kecil untuk 1 lot. Perbesar risk% atau perlebar stop."}

    # Batas modal: berapa lot yang MAMPU dibeli
    lots_by_capital = int((modal / entry) // 100)
    lots = min(lots_by_risk, lots_by_capital)
    terpakai = lots * 100 * entry
    overbudget = lots_by_risk > lots_by_capital

    return {
        "valid": True, "lots": lots, "lots_by_risk": lots_by_risk,
        "lots_by_capital": lots_by_capital, "terpakai": terpakai,
        "risk_rp": risk_rp, "pct_modal": terpakai / modal * 100 if modal else 0,
        "overbudget": overbudget,
        "risk_actual": lots * 100 * (entry - stop),  # kerugian nyata jika stop kena
    }

def risk_reward(entry, stop, target):
    if min(entry, stop, target) <= 0 or entry == stop:
        return None
    risk = abs(entry - stop); reward = abs(target - entry)
    return reward / risk if risk else None

def support_resistance(df: pd.DataFrame, lookback: int = 120, bins: int = 24) -> dict:
    d = df.tail(lookback)
    close = d["Close"].iloc[-1]

    # --- Volume by Price (POC = harga dgn volume terpadat) ---
    lo, hi = d["Low"].min(), d["High"].max()
    edges = np.linspace(lo, hi, bins + 1)
    mid = (edges[:-1] + edges[1:]) / 2
    vol_at = np.zeros(bins)
    for _, row in d.iterrows():
        b0 = max(0, np.searchsorted(edges, row["Low"], "right") - 1)
        b1 = min(bins - 1, np.searchsorted(edges, row["High"], "right") - 1)
        span = b1 - b0 + 1
        if span > 0:
            vol_at[b0:b1+1] += row["Volume"] / span
    poc = sorted(zip(mid, vol_at), key=lambda x: -x[1])[0][0]

    # --- Swing pivots ---
    hh, ll = d["High"].values, d["Low"].values
    highs = [hh[i] for i in range(3, len(d)-3) if hh[i] == max(hh[i-3:i+4])]
    lows = [ll[i] for i in range(3, len(d)-3) if ll[i] == min(ll[i-3:i+4])]
    resistance = sorted({round(x) for x in highs if x > close})[:3]
    support = sorted({round(x) for x in lows if x < close}, reverse=True)[:3]

    # --- PERBAIKAN: level psikologis ATAS & BAWAH (kelipatan bulat) ---
    step = 50 if close < 1000 else (100 if close < 5000 else 250)
    psy_below = int(close // step * step)
    psy_above = psy_below + step

    # Fallback jujur: jika tak ada pivot (mis. harga di puncak), pakai
    # level psikologis / ATR — BUKAN mengarang harga x 1.1
    atr_val = df["ATR"].iloc[-1] if "ATR" in df.columns and not np.isnan(df["ATR"].iloc[-1]) else close * 0.03
    if not resistance:
        # saham di area tertinggi: proyeksikan resistance dari ATR & psikologis
        resistance = sorted({psy_above, round(close + 2*atr_val), round(hi)})
        resistance = [r for r in resistance if r > close][:3]
    if not support:
        support = sorted({psy_below, round(close - 2*atr_val), round(poc)}, reverse=True)
        support = [s for s in support if s < close][:3]

    return {
        "close": float(close),
        "poc": round(poc),
        "resistance": resistance,
        "support": support,
        "psy_below": psy_below,
        "psy_above": psy_above,
    }

# ============================================================
# STATE & DATA LOADERS
# ============================================================
if "modal" not in st.session_state: st.session_state.modal = 100_000_000
if "target" not in st.session_state: st.session_state.target = 500_000_000
if "watchlist_default" not in st.session_state:
    st.session_state.watchlist_default = "BBRI, BBCA, BMRI, TLKM, ASII, ADRO, ANTM, MDKA, GOTO, BRIS, KLBF, ICBP, UNTR, PTBA, INCO, ISAT, EXCL, AMMN"

@st.cache_data(ttl=3600, show_spinner=False)
def load_eod(ticker: str):
    if not HAS_YF: return None
    try:
        # PERBAIKAN: auto_adjust=True agar MA tidak melompat saat dividen/split
        df = yf.Ticker(ticker + ".JK").history(period="1y", auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna() if len(df) >= 60 else None
    except Exception: return None

@st.cache_data(ttl=3600, show_spinner=False)
def load_fundamental_yf(ticker: str):
    if not HAS_YF: return {}
    try:
        info = yf.Ticker(ticker + ".JK").info or {}
        eps, bvps = info.get("trailingEps", 0), info.get("bookValue", 0)
        return {
            "eps": eps, "bvps": bvps,
            "pbv": info.get("priceToBook"),
            "roe": (info.get("returnOnEquity") or 0) * 100 if info.get("returnOnEquity") else None,
            "der": info.get("debtToEquity", 0) / 100 if info.get("debtToEquity") else None,
            "fair_value": np.sqrt(22.5 * eps * bvps) if eps and bvps and eps > 0 and bvps > 0 else None
        }
    except Exception: return {}

def scan_one(tk: str, min_liquidity: float, fund_custom: dict, sector_map: dict) -> dict | None:
    """Proses SATU ticker sepenuhnya (unduh + indikator + skor + fundamental).
    Fungsi ini AMAN dipanggil dari thread karena TIDAK menyentuh objek `st`.
    Dipakai oleh ThreadPoolExecutor agar puluhan ticker diproses serentak."""
    df = load_eod(tk)
    if df is None:
        return None
    df = add_indicators(df)
    sc = screen_score(df, min_liquidity=min_liquidity)
    if not sc.get("valid"):
        return None

    # pola candlestick pada baris terakhir
    patt = detect_patterns(df.tail(3))["Pattern"].iloc[-1]

    row = {"Saham": tk, "Harga": sc["close"], "Skor Tech": sc["score"],
           "Tx/Hari (M)": round(sc["avg_val"]/1_000_000_000, 1),
           "Pola": patt or "-"}

    f = fund_custom.get(tk, load_fundamental_yf(tk))
    roe_val = num(f.get("roe") if f.get("roe") is not None else f.get("ROE"))
    der_val = num(f.get("der") if f.get("der") is not None else f.get("DER"))
    is_bank = is_bank_ticker(tk, sector_map)

    mos_val = None; layak_valuasi = False
    if is_bank:
        pbv = f.get("pbv") if f.get("pbv") is not None else f.get("PBV")
        pbv = num(pbv, None) if pbv is not None else None
        row["Valuasi"] = f"PBV: {round(pbv, 2)}" if pbv else "-"
        layak_valuasi = bool(pbv and pbv < 1.5)
        row["MoS/Diskon"] = "Layak (Bank)" if layak_valuasi else ("Mahal" if pbv else "-")
    else:
        fv = f.get("fair_value")
        if fv is None:
            eps_c = num(f.get("EPS") if f.get("EPS") is not None else f.get("eps"))
            bvps_c = num(f.get("BVPS") if f.get("BVPS") is not None else f.get("bvps"))
            fv = np.sqrt(22.5 * eps_c * bvps_c) if eps_c > 0 and bvps_c > 0 else None
        if fv and fv > 0:
            mos_val = (fv - sc["close"]) / fv * 100
            row["Valuasi"] = f"Harga Wajar: {round(fv)}"
            row["MoS/Diskon"] = f"{round(mos_val, 1)}%"
            layak_valuasi = mos_val > 15
        else:
            row["Valuasi"] = "-"; row["MoS/Diskon"] = "-"

    row["ROE (%)"] = round(roe_val, 1)
    row["DER"] = round(der_val, 2)
    sehat = (roe_val >= 15) and (0 < der_val <= 1.2)
    row["Status"] = "🔥 Multi-Bagger" if (sc["score"] >= 60 and sehat and layak_valuasi) else "✅ Masuk Radar"
    return row


# Fallback daftar bank bila CSV sektor tidak diunggah (agar tetap jalan)
DEFAULT_BANKS = {"BBCA","BBRI","BMRI","BBNI","BRIS","ARTO","BBTN","NISP","BDMN","MEGA",
                 "BJBR","BJTM","BNGA","PNBN","BTPS","AGRO","BANK","BBHI","BBYB"}

def build_sector_map(sector_file) -> dict:
    """Baca CSV sektor (kolom: Saham, Sektor) → dict {KODE: sektor_lower}.
    Jika tidak ada file, kembalikan {} dan screener pakai DEFAULT_BANKS.
    Ini menggantikan hardcode daftar bank di dalam skrip."""
    if sector_file is None:
        return {}
    try:
        sdf = pd.read_csv(sector_file)
        # toleran terhadap nama kolom
        cols = {c.lower(): c for c in sdf.columns}
        kode_c = cols.get("saham") or cols.get("kode") or cols.get("ticker") or list(sdf.columns)[0]
        sekt_c = cols.get("sektor") or cols.get("sector") or list(sdf.columns)[1]
        return {str(r[kode_c]).strip().upper(): str(r[sekt_c]).strip().lower()
                for _, r in sdf.iterrows() if pd.notna(r[kode_c])}
    except Exception:
        return {}

def is_bank_ticker(tk: str, sector_map: dict) -> bool:
    """Tentukan bank dari CSV sektor bila ada; jika tidak, dari daftar bawaan."""
    if sector_map:
        return "bank" in sector_map.get(tk, "")
    return tk in DEFAULT_BANKS

def rupiah(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)): return "-"
        return "Rp{:,.0f}".format(x).replace(",", ".")
    except Exception:
        return "-"

def num(x, default=0.0):
    """Ambil angka dengan aman dari nilai fundamental yang bisa None/str."""
    try:
        if x is None: return default
        return float(x)
    except (ValueError, TypeError):
        return default

# ============================================================
# UI DASHBOARD
# ============================================================
st.title("📊 EOD Stock Analysis Dashboard & Multi-Bagger Hub")
if not HAS_YF: st.error("Pustaka `yfinance` belum terpasang. Jalankan `pip install yfinance`.")

tab1, tab2, tab3 = st.tabs(["① Screener Cerdas", "② Analisis Chart & Level", "③ Manajemen Risiko"])

# --- TAB 1: SCREENER ---
with tab1:
    st.subheader("Pencari Saham Multi-Bagger (Fundamental + Teknikal Likuid)")
    st.caption("Skor = konfluensi sinyal teknikal likuid. **Skor tinggi = layak diteliti, "
               "bukan sinyal beli, bukan jaminan multi-bagger.** Label hanya penanda kandidat.")

    col1, col2, col3 = st.columns([2, 1, 1])
    wl = col1.text_area("Daftar Saham (Pisahkan koma)", st.session_state.watchlist_default, height=130)
    min_score = col2.slider("Skor Teknikal Minimal", 0, 100, 50, 5)
    min_liq_miliar = col3.number_input("Min. Transaksi Harian (Miliar Rp)", value=10, step=5,
                                       help="Menyaring saham gorengan yang sepi peminat.")

    fu1, fu2 = st.columns(2)
    fund_file = fu1.file_uploader("CSV Fundamental (Saham, EPS, BVPS, ROE, DER, PBV)",
                                  type=["csv"], help="Abaikan jika ingin otomatis via Yahoo Finance.")
    sector_file = fu2.file_uploader("CSV Sektor (Saham, Sektor) — untuk penanda bank/non-bank",
                                    type=["csv"], help="Kolom Sektor berisi mis. 'Bank', 'Energi'. "
                                    "Jika kosong, dipakai daftar bank bawaan. Ini menggantikan "
                                    "hardcode daftar bank di dalam skrip.")
    max_workers = st.slider("Jumlah thread paralel", 2, 30, 12, 1,
                            help="Makin tinggi makin cepat, tapi jangan berlebihan agar tidak "
                                 "diblokir sumber data. 8–15 biasanya optimal.")

    if st.button("🚀 Jalankan Screener", type="primary"):
        fund_custom = pd.read_csv(fund_file).set_index("Saham").to_dict("index") if fund_file else {}
        sector_map = build_sector_map(sector_file)
        tickers = [t.strip().upper() for t in wl.replace("\n", ",").split(",") if t.strip()]

        min_liq = min_liq_miliar * 1_000_000_000
        rows = []
        prog = st.progress(0.0, text=f"Memindai {len(tickers)} saham secara paralel...")
        done = 0

        # ---- MULTI-THREADING: proses puluhan ticker serentak ----
        # I/O jaringan (unduh yfinance) adalah bottleneck; ThreadPoolExecutor
        # menjalankan banyak unduhan bersamaan sehingga total waktu ~ waktu
        # satu unduhan, bukan jumlah semua unduhan.
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(scan_one, tk, min_liq, fund_custom, sector_map): tk
                       for tk in tickers}
            for fut in as_completed(futures):
                done += 1
                prog.progress(done/len(tickers), text=f"Selesai {done}/{len(tickers)}...")
                try:
                    r = fut.result()
                    if r: rows.append(r)
                except Exception:
                    pass  # ticker gagal diabaikan, screener tetap lanjut

        prog.empty()
        if rows:
            res = pd.DataFrame(rows)
            res = res[res["Skor Tech"] >= min_score].sort_values("Skor Tech", ascending=False).reset_index(drop=True)
            st.dataframe(res.style.background_gradient(subset=["Skor Tech"], cmap="Blues"),
                         use_container_width=True)
            st.download_button("⬇️ Unduh Watchlist (CSV)", res.to_csv(index=False).encode("utf-8"),
                               f"watchlist_{dt.date.today()}.csv", "text/csv")
        else:
            st.warning("Tidak ada saham yang lolos filter (Coba turunkan Skor atau syarat Likuiditas).")

# --- TAB 2: CHART & LEVEL ---
with tab2:
    tk3 = st.text_input("Analisis Kode Saham", "BBRI").strip().upper()
    df3 = load_eod(tk3)
    if df3 is not None:
        df3 = add_indicators(df3)
        df3 = detect_patterns(df3)
        sr = support_resistance(df3)
        last = df3.iloc[-1]

        c1, c2, c3, c4 = st.columns(4)
        chg = (last["Close"]/df3["Close"].iloc[-2]-1)*100
        c1.metric("Harga Terakhir", rupiah(last["Close"]), f"{chg:+.2f}%")
        c2.metric("🟢 Support (Antre Beli)", rupiah(sr["support"][0] if sr["support"] else sr["psy_below"]))
        c3.metric("🔴 Resistance (Jual)", rupiah(sr["resistance"][0] if sr["resistance"] else sr["psy_above"]))
        c4.metric("POC (Volume Terpadat)", rupiah(sr["poc"]))

        # ---- candle terakhir: bullish/bearish ----
        o, cl, hi, lo = last["Open"], last["Close"], last["High"], last["Low"]
        arah = "Bullish (naik) 🟢" if cl >= o else "Bearish (turun) 🔴"
        body = abs(cl - o); rng = hi - lo if hi != lo else np.nan
        body_pct = (body / rng * 100) if rng and not np.isnan(rng) else 0

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                            vertical_spacing=0.03,
                            subplot_titles=("Candlestick + Bollinger Bands + Support/Resistance", "MACD Histogram"))
        d = df3.tail(120)

        # Chart Harga + BB (warna candle: hijau bullish, merah bearish)
        fig.add_trace(go.Candlestick(
            x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
            name="Harga",
            increasing_line_color="#26A69A", increasing_fillcolor="#26A69A",
            decreasing_line_color="#EF5350", decreasing_fillcolor="#EF5350",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["MA20"], line=dict(color="#2962FF", width=1.5), name="MA20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["Upper_BB"], line=dict(color="gray", width=1, dash="dot"), name="Upper BB"), row=1, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["Lower_BB"], line=dict(color="gray", width=1, dash="dot"),
                                 name="Lower BB", fill="tonexty", fillcolor="rgba(128,128,128,0.08)"), row=1, col=1)

        # ---- GARIS SUPPORT (batas bawah, hijau) & RESISTANCE (batas atas, merah) ----
        for i, r in enumerate(sr["resistance"]):
            fig.add_hline(y=r, line=dict(color="#EF5350", width=1.3, dash="dash"),
                          annotation_text=f"R{i+1} {rupiah(r)}", annotation_position="right",
                          annotation_font_color="#EF5350", row=1, col=1)
        for i, s in enumerate(sr["support"]):
            fig.add_hline(y=s, line=dict(color="#26A69A", width=1.3, dash="dash"),
                          annotation_text=f"S{i+1} {rupiah(s)}", annotation_position="right",
                          annotation_font_color="#26A69A", row=1, col=1)
        # POC (garis ungu tipis)
        fig.add_hline(y=sr["poc"], line=dict(color="#8E44AD", width=1, dash="dot"),
                      annotation_text=f"POC {rupiah(sr['poc'])}", annotation_position="left",
                      annotation_font_color="#8E44AD", row=1, col=1)

        # ---- PENANDA POLA CANDLESTICK otomatis di chart ----
        # Marker ditempatkan di dekat candle yang membentuk pola. Pola bullish
        # (hijau) di bawah Low, pola bearish (merah) di atas High, netral (doji) di atas.
        pat_cfg = {
            "Bullish Engulfing": ("▲", "#00897B", "below"),
            "Hammer":            ("▲", "#00897B", "below"),
            "Inverted Hammer":   ("△", "#00897B", "below"),
            "Doji":              ("◆", "#8E44AD", "above"),
            "Bearish Engulfing": ("▼", "#D32F2F", "above"),
        }
        for name, (sym, color, pos) in pat_cfg.items():
            sel = d[d["Pattern"] == name]
            if sel.empty:
                continue
            if pos == "below":
                ys = sel["Low"] * 0.985
            else:
                ys = sel["High"] * 1.015
            fig.add_trace(go.Scatter(
                x=sel.index, y=ys, mode="text", text=[sym]*len(sel),
                textfont=dict(size=15, color=color), name=name,
                hovertext=[name]*len(sel), hoverinfo="text+x",
            ), row=1, col=1)

        # Chart MACD
        fig.add_trace(go.Bar(x=d.index, y=d["MACD_HIST"],
                             marker_color=np.where(d["MACD_HIST"] > 0, "#26A69A", "#EF5350"),
                             name="MACD Hist"), row=2, col=1)

        fig.update_layout(height=680, margin=dict(t=40, b=10, l=10, r=60),
                          xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, use_container_width=True)

        # ---- ringkasan pola candlestick terdeteksi (30 hari terakhir) ----
        recent = df3.tail(30)
        found = recent[recent["Pattern"] != ""][["Pattern"]].copy()
        if not found.empty:
            counts = found["Pattern"].value_counts()
            badge = "  ·  ".join(f"**{k}**: {v}×" for k, v in counts.items())
            last_pat = df3["Pattern"].iloc[-1]
            st.markdown(f"🕯️ **Pola candlestick 30 hari terakhir** — {badge}")
            if last_pat:
                arti = {
                    "Bullish Engulfing": "sinyal potensi pembalikan NAIK (pembeli mengambil alih)",
                    "Hammer": "penolakan harga rendah — potensi pembalikan NAIK di area support",
                    "Inverted Hammer": "potensi pembalikan naik, perlu konfirmasi candle berikutnya",
                    "Doji": "keraguan pasar / keseimbangan — waspada perubahan arah",
                    "Bearish Engulfing": "sinyal potensi pembalikan TURUN (penjual mengambil alih)",
                }.get(last_pat, "")
                st.info(f"Candle **terakhir** membentuk **{last_pat}** — {arti}. "
                        "Pola candlestick bukan sinyal pasti; selalu konfirmasi dengan tren, "
                        "volume, dan level support/resistance.")

        # ============================================================
        # PANEL EDUKASI: ANATOMI CANDLESTICK
        # ============================================================
        with st.expander("📚 Cara Membaca Candlestick (klik untuk belajar) — candle terakhir dianalisis di bawah"):
            ce1, ce2 = st.columns([1, 1])
            with ce1:
                st.markdown("""
**Anatomi satu batang candle**

- **Body (Badan):** rentang antara harga **Open** (pembukaan) dan **Close** (penutupan). Badan tebal = pergerakan open→close besar.
- **Shadow / Wick (Sumbu):** garis tipis di atas & bawah badan; menunjukkan **High** (tertinggi) dan **Low** (terendah) hari itu.
- **🟢 Bullish (naik):** Close **lebih tinggi** dari Open → badan hijau.
- **🔴 Bearish (turun):** Close **lebih rendah** dari Open → badan merah.

**Membaca sumbu:**
- Sumbu bawah panjang = ada tekanan beli (harga sempat turun lalu ditolak naik) — sering sinyal *support*.
- Sumbu atas panjang = ada tekanan jual (harga sempat naik lalu ditolak turun) — sering sinyal *resistance*.

**Pola yang ditandai otomatis di chart:**
- **▲ Hammer** — badan kecil di atas, sumbu bawah panjang. Muncul setelah penurunan → potensi pembalikan naik.
- **◆ Doji** — badan sangat tipis (open ≈ close). Pasar ragu → sering mendahului perubahan arah.
- **▲ Bullish Engulfing** — candle hijau menelan penuh badan merah kemarin. Sinyal pembeli mengambil alih.
- **▼ Bearish Engulfing** — kebalikannya (merah menelan hijau); tanda waspada tekanan jual.
""")
            with ce2:
                # SVG anatomi candle (bullish & bearish) — statis, edukatif
                st.markdown("**Ilustrasi:**")
                svg = """
<svg width="300" height="240" viewBox="0 0 300 240" xmlns="http://www.w3.org/2000/svg">
  <style>.lbl{font:11px sans-serif;fill:#444}.hd{font:bold 12px sans-serif}</style>
  <!-- Bullish -->
  <line x1="80" y1="20" x2="80" y2="60" stroke="#26A69A" stroke-width="2"/>
  <line x1="80" y1="150" x2="80" y2="200" stroke="#26A69A" stroke-width="2"/>
  <rect x="62" y="60" width="36" height="90" fill="#26A69A" rx="2"/>
  <text x="40" y="16" class="hd" fill="#26A69A">Bullish</text>
  <text x="105" y="30" class="lbl">High (sumbu atas)</text>
  <text x="105" y="70" class="lbl">Close</text>
  <text x="105" y="148" class="lbl">Open</text>
  <text x="105" y="200" class="lbl">Low (sumbu bawah)</text>
  <!-- Bearish -->
  <line x1="230" y1="20" x2="230" y2="55" stroke="#EF5350" stroke-width="2"/>
  <line x1="230" y1="145" x2="230" y2="205" stroke="#EF5350" stroke-width="2"/>
  <rect x="212" y="55" width="36" height="90" fill="#EF5350" rx="2"/>
  <text x="196" y="16" class="hd" fill="#EF5350">Bearish</text>
  <text x="255" y="60" class="lbl">Open</text>
  <text x="255" y="140" class="lbl">Close</text>
</svg>
"""
                st.markdown(svg, unsafe_allow_html=True)

            st.info(f"**Candle terakhir {tk3}:** {arah} · "
                    f"Open {rupiah(o)} → Close {rupiah(cl)} · "
                    f"High {rupiah(hi)} / Low {rupiah(lo)} · "
                    f"Badan mengisi {body_pct:.0f}% dari rentang hari itu "
                    f"({'dominan — konviksi kuat' if body_pct>60 else 'kecil — ragu-ragu/indecision' if body_pct<30 else 'sedang'}).")

        # ---- ringkasan level teks ----
        lv1, lv2, lv3 = st.columns(3)
        with lv1:
            st.markdown("**🟢 Support (batas bawah)**")
            for s in sr["support"]:
                st.write(f"• {rupiah(s)}  ({(s/sr['close']-1)*100:+.1f}%)")
        with lv2:
            st.markdown("**🔴 Resistance (batas atas)**")
            for r in sr["resistance"]:
                st.write(f"• {rupiah(r)}  ({(r/sr['close']-1)*100:+.1f}%)")
        with lv3:
            st.markdown("**📊 Level kunci**")
            st.write(f"POC: **{rupiah(sr['poc'])}**")
            st.write(f"Psikologis: {rupiah(sr['psy_below'])} / {rupiah(sr['psy_above'])}")
        st.caption("Support = area harga sering memantul naik (kandidat antre beli). "
                   "Resistance = area harga sering tertahan (kandidat ambil untung). "
                   "Level dihitung dari data harga & volume NYATA, bukan order book.")
    else:
        st.info("Masukkan kode saham yang valid (butuh yfinance & koneksi internet).")

# --- TAB 3: MANAJEMEN RISIKO ---
with tab3:
    c1, c2, c3 = st.columns(3)
    st.session_state.modal = c1.number_input("Modal saat ini (Rp)", value=int(st.session_state.modal), step=1_000_000)
    risk_pct = c2.slider("Risiko per transaksi (%)", 0.25, 3.0, 1.0, 0.25)
    st.session_state.target = c3.number_input("Target aset (Rp)", value=int(st.session_state.target), step=10_000_000)

    # progress
    pct = min(st.session_state.modal / st.session_state.target, 1.0) if st.session_state.target else 0
    st.progress(pct, text=f"Progres menuju target: {rupiah(st.session_state.modal)} / {rupiah(st.session_state.target)} ({pct*100:.1f}%)")

    st.markdown("#### Kalkulator Position Sizing")
    p1, p2, p3 = st.columns(3)
    entry = p1.number_input("Harga Beli (Rp)", value=1000, step=5)
    stop = p2.number_input("Stop Loss (Rp)", value=950, step=5)
    target_price = p3.number_input("Take Profit (Rp)", value=1150, step=5)

    ps = position_size_percent(st.session_state.modal, risk_pct, entry, stop)
    rr = risk_reward(entry, stop, target_price)

    if ps.get("valid"):
        m1, m2, m3 = st.columns(3)
        m1.metric("Beli Maksimal", f"{ps['lots']:,} Lot", f"Modal: {rupiah(ps['terpakai'])} ({ps['pct_modal']:.0f}%)")
        m2.metric("Risiko Terukur", rupiah(ps['risk_actual']),
                  "kerugian jika stop kena", delta_color="inverse")
        if rr:
            m3.metric("Risk/Reward", f"1 : {rr:.2f}",
                      "layak (≥1:2)" if rr >= 2 else "kurang ideal",
                      delta_color="normal" if rr >= 2 else "inverse")
        # PERBAIKAN: peringatan overbudget yang sebelumnya tidak ada
        if ps.get("overbudget"):
            st.warning(f"⚠️ Berdasarkan risiko {risk_pct}%, idealnya {ps['lots_by_risk']:,} lot, "
                       f"tapi modal hanya cukup untuk {ps['lots_by_capital']:,} lot. "
                       f"Jumlah dibatasi ke {ps['lots']:,} lot. Stop loss terlalu dekat dengan entry — "
                       f"pertimbangkan perlebar stop atau pilih saham lain agar risiko tetap terukur.")
        if rr and rr < 2:
            st.info("Risk/Reward di bawah 1:2. Banyak trader disiplin melewati setup seperti ini.")
    else:
        st.warning(ps.get("reason", "Lengkapi input: Harga Beli harus di atas Stop Loss."))
