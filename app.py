import io
import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.stats import poisson

st.set_page_config(page_title="Maç Analiz Paneli", layout="wide")

BASE = "https://www.football-data.co.uk"
SEASONS = {"2627": 1.0, "2526": 0.6}  # güncel sezon tam ağırlık, geçen sezon 0.6
LEAGUES = {
    "Süper Lig": "T1",
    "Premier Lig": "E0",
    "La Liga": "SP1",
    "Bundesliga": "D1",
    "Serie A": "I1",
    "Ligue 1": "F1",
}
MAX_GOALS = 8
SHRINK = 3  # az maçı olan takımları lig ortalamasına yaklaştırır


@st.cache_data(ttl=6 * 3600)  # 6 saatte bir veriyi yeniler
def load_csv(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.content.decode("latin-1")), on_bad_lines="skip")


def load_results(code):
    frames = []
    for season, weight in SEASONS.items():
        try:
            d = load_csv(f"{BASE}/mmz4281/{season}/{code}.csv")
            d = d[["HomeTeam", "AwayTeam", "FTHG", "FTAG"]].dropna()
            d["w"] = weight
            frames.append(d)
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def wmean(x, w):
    return (x * w).sum() / w.sum()


def shrink(value, n):
    return (n * value + SHRINK * 1.0) / (n + SHRINK)


def strengths(df):
    avg_h = wmean(df.FTHG, df.w)
    avg_a = wmean(df.FTAG, df.w)
    s = {}
    for t in set(df.HomeTeam) | set(df.AwayTeam):
        h = df[df.HomeTeam == t]
        a = df[df.AwayTeam == t]
        nh, na = h.w.sum(), a.w.sum()
        s[t] = {
            "h_att": shrink(wmean(h.FTHG, h.w) / avg_h, nh) if nh else 1.0,
            "h_def": shrink(wmean(h.FTAG, h.w) / avg_a, nh) if nh else 1.0,
            "a_att": shrink(wmean(a.FTAG, a.w) / avg_a, na) if na else 1.0,
            "a_def": shrink(wmean(a.FTHG, a.w) / avg_h, na) if na else 1.0,
        }
    return s, avg_h, avg_a


def predict(home, away, s, avg_h, avg_a):
    d = {"h_att": 1, "h_def": 1, "a_att": 1, "a_def": 1}
    sh, sa = s.get(home, d), s.get(away, d)
    lam_h = sh["h_att"] * sa["a_def"] * avg_h
    lam_a = sa["a_att"] * sh["h_def"] * avg_a
    g = np.arange(MAX_GOALS + 1)
    P = np.outer(poisson.pmf(g, lam_h), poisson.pmf(g, lam_a))
    P /= P.sum()
    i, j = np.indices(P.shape)
    best = np.unravel_index(P.argmax(), P.shape)
    return {
        "Ev xG": round(lam_h, 2),
        "Dep xG": round(lam_a, 2),
        "MS1 %": round(np.tril(P, -1).sum() * 100),
        "MSX %": round(np.trace(P) * 100),
        "MS2 %": round(np.triu(P, 1).sum() * 100),
        "2.5 Üst %": round(P[(i + j) > 2].sum() * 100),
        "KG Var %": round(P[(i >= 1) & (j >= 1)].sum() * 100),
        "En olası skor": f"{best[0]}-{best[1]}",
    }


st.title("⚽ Haftalık Maç Analiz Paneli")
st.caption("Veri: football-data.co.uk • Model: Poisson gol modeli • Her 6 saatte otomatik yenilenir")

league = st.selectbox("Lig", list(LEAGUES))
code = LEAGUES[league]

results = load_results(code)
if results.empty:
    st.error("Sonuç verisi alınamadı.")
    st.stop()

try:
    fx = load_csv(f"{BASE}/fixtures.csv")
    fx = fx[fx["Div"] == code].copy()
    fx["Date"] = pd.to_datetime(fx["Date"], dayfirst=True, errors="coerce")
    fx = fx.dropna(subset=["Date"]).sort_values("Date")
except Exception:
    fx = pd.DataFrame()

if fx.empty:
    st.info("Bu lig için yaklaşan maç bulunamadı (ara verilmiş olabilir).")
    st.stop()

s, avg_h, avg_a = strengths(results)
rows = []
for _, m in fx.iterrows():
    p = predict(m["HomeTeam"], m["AwayTeam"], s, avg_h, avg_a)
    rows.append({"Tarih": m["Date"].strftime("%d.%m.%Y"),
                 "Maç": f'{m["HomeTeam"]} - {m["AwayTeam"]}', **p})

table = pd.DataFrame(rows)
st.dataframe(table, use_container_width=True, hide_index=True)

st.download_button("CSV indir", table.to_csv(index=False).encode("utf-8"),
                   f"{code}_analiz.csv", "text/csv")
st.warning("Bu panel olasılık hesaplar, kazanç garantisi vermez. Sorumlu oynayın.")
