import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dateutil import parser

# --------- page config ---------
st.set_page_config(page_title="BetSync Prototype", page_icon="📈", layout="wide")
st.title("BetSync Prototype — Limit Risk Dashboard")

st.markdown("""
Upload a CSV with columns (case-sensitive):  
`Book, Sport, MarketType, OddsPlaced, ClosingOdds, Stake, BetTime, EventTime, Result`

- **OddsPlaced/ClosingOdds** can be **American** (+110 / -120) or **Decimal** (2.10 / 1.83)  
- **BetTime/EventTime** should be parseable datetimes (e.g., `2025-10-10 14:05:00`)
""")

# --------- helpers ---------
def american_to_decimal(x):
    """Convert American odds to Decimal. If already decimal, return float."""
    s = str(x).strip()
    try:
        v = float(s)
    except:
        return np.nan

    # Decimal range heuristic
    if 1.01 <= v <= 100.0:
        return v

    # American conversion
    # allow + sign without casting issues
    if s.startswith("+"):
        try:
            a = float(s[1:])
        except:
            return np.nan
        return 1.0 + (a / 100.0)

    # negative american
    if v < 0:
        return 1.0 + (100.0 / abs(v))

    # large positive number like 110 (treated as +110)
    if v >= 100:
        return 1.0 + (v / 100.0)

    return np.nan

def american_or_decimal(x):
    s = str(x).strip()
    # negative American explicitly
    if s.startswith('-'):
        try:
            a = float(s)
            return 1.0 + (100.0 / abs(a))
        except:
            return np.nan
    return american_to_decimal(s)

def parse_dt(x):
    try:
        return parser.parse(str(x))
    except:
        return pd.NaT

def herfindahl(series):
    counts = series.value_counts(dropna=False)
    p = counts / counts.sum()
    return (p ** 2).sum()

def clamp01(v):
    return max(0.0, min(1.0, float(v)))

# --------- uploader + sample ---------
file = st.file_uploader("Upload bet history CSV", type=["csv"])

with st.expander("Need a sample CSV? Click to copy"):
    st.code("""Book,Sport,MarketType,OddsPlaced,ClosingOdds,Stake,BetTime,EventTime,Result
Bet99,NBA,PlayerPoints,+110,+100,50,2025-10-10 13:00:00,2025-10-10 19:30:00,W
Bet99,NHL,Moneyline,-120,-115,55,2025-10-10 14:05:00,2025-10-10 20:00:00,L
FanDuel,NBA,AltSpread,2.05,1.98,50,2025-10-09 10:00:00,2025-10-10 19:30:00,W
Bet365,NFL,Totals,1.91,1.90,60,2025-10-08 09:45:00,2025-10-12 13:00:00,L
BetMGM,NHL,ShotsOnGoal,+125,+120,50,2025-10-11 12:00:00,2025-10-11 19:00:00,W
""", language="csv")

if not file:
    st.info("Upload a CSV to see your dashboard.")
    st.stop()

# --------- load & clean ---------
df = pd.read_csv(file)

required = ["Book","Sport","MarketType","OddsPlaced","ClosingOdds","Stake","BetTime","EventTime"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

df["OddsPlaced_dec"] = df["OddsPlaced"].apply(american_or_decimal)
df["ClosingOdds_dec"] = df["ClosingOdds"].apply(american_or_decimal)
df["BetTime_dt"] = df["BetTime"].apply(parse_dt)
df["EventTime_dt"] = df["EventTime"].apply(parse_dt)
df["LeadHours"] = (df["EventTime_dt"] - df["BetTime_dt"]).dt.total_seconds() / 3600.0
df["CLV_pct"] = (df["ClosingOdds_dec"] - df["OddsPlaced_dec"]) / df["OddsPlaced_dec"] * 100.0

df = df.dropna(subset=["OddsPlaced_dec","ClosingOdds_dec","Stake","LeadHours","CLV_pct"])
if df.empty:
    st.error("No valid rows after cleaning. Check your CSV formatting.")
    st.stop()

# --------- metrics ---------
avg_clv = df["CLV_pct"].mean()
pos_clv_rate = (df["CLV_pct"] > 0).mean()                  # fraction beating the close
stake_cv = (df["Stake"].std() / (df["Stake"].mean() + 1e-9)) * 100.0
market_hhi = herfindahl(df["MarketType"])
book_hhi = herfindahl(df["Book"])
lead_mean = df["LeadHours"].mean()
lead_std = df["LeadHours"].std()

# normalized feature "risks" (0..1) — simple heuristics you can tune later
clv_risk        = clamp01((avg_clv - 1.0) / 4.0)       # 1% ok; 5%+ looks sharp
posclv_risk     = clamp01((pos_clv_rate - 0.55) / 0.25)# >55% beating close becomes suspicious
stake_risk      = clamp01((12.0 - min(stake_cv, 30.0)) / 12.0)  # low variance = robotic
market_risk     = clamp01((market_hhi - 0.20) / 0.60)  # high concentration = risk
book_risk       = clamp01((book_hhi - 0.25) / 0.60)
lead_mean_risk  = clamp01((lead_mean - 12.0) / 48.0)   # early average = model-y
lead_std_risk   = clamp01((6.0 - min(lead_std, 24.0)) / 6.0)    # consistent timing = model-y

weights = {
    "clv": 0.28, "posclv": 0.12, "stake": 0.16,
    "market": 0.14, "book": 0.10, "lead_mean": 0.10, "lead_std": 0.10
}
risk_01 = (
    weights["clv"]*clv_risk +
    weights["posclv"]*posclv_risk +
    weights["stake"]*stake_risk +
    weights["market"]*market_risk +
    weights["book"]*book_risk +
    weights["lead_mean"]*lead_mean_risk +
    weights["lead_std"]*lead_std_risk
)
risk_score = round(100.0 * risk_01, 1)

# --------- top KPIs ---------
col1, col2, col3 = st.columns(3)
with col1:
    badge = "🟢"
    if risk_score >= 66: badge = "🔴"
    elif risk_score >= 40: badge = "🟠"
    st.metric("Limit Risk Score", f"{badge} {risk_score}/100")
with col2:
    st.metric("Avg CLV", f"{avg_clv:.2f}%")
with col3:
    st.metric("% Bets Beating Close", f"{pos_clv_rate*100:.1f}%")

st.divider()

# --------- charts ---------
c1, c2 = st.columns(2)
with c1:
    st.subheader("CLV (%) Distribution")
    fig, ax = plt.subplots()
    ax.hist(df["CLV_pct"], bins=20)
    ax.set_xlabel("CLV (%)"); ax.set_ylabel("Count")
    st.pyplot(fig)

with c2:
    st.subheader("Stake Distribution")
    fig2, ax2 = plt.subplots()
    ax2.hist(df["Stake"], bins=20)
    ax2.set_xlabel("Stake"); ax2.set_ylabel("Count")
    st.pyplot(fig2)

c3, c4 = st.columns(2)
with c3:
    st.subheader("Lead Time (hours) Distribution")
    fig3, ax3 = plt.subplots()
    ax3.hist(df["LeadHours"], bins=20)
    ax3.set_xlabel("Hours between bet & event"); ax3.set_ylabel("Count")
    st.pyplot(fig3)

with c4:
    st.subheader("Markets Hit (Top 10)")
    top_markets = df["MarketType"].value_counts().head(10)
    fig4, ax4 = plt.subplots()
    top_markets.plot(kind="bar", ax=ax4)
    ax4.set_ylabel("Bets")
    st.pyplot(fig4)

st.divider()

# --------- recommendations ---------
st.subheader("Recommendations (auto-generated)")
recs = []
if clv_risk > 0.6:
    recs.append("High positive CLV: mix in later bets or smaller edges to look less sharp.")
if posclv_risk > 0.6:
    recs.append("Large share beating the close: add some neutral/coin-flip markets.")
if stake_risk > 0.6:
    recs.append("Stake sizes too consistent: vary stakes ±10–25% around your base.")
if market_risk > 0.6:
    recs.append("Market concentration high: add 2–3 different markets or sports weekly.")
if book_risk > 0.6:
    recs.append("Book concentration high: spread action across additional legal books.")
if lead_mean_risk > 0.6:
    recs.append("You bet very early on average: add some closer-to-start bets.")
if lead_std_risk > 0.6:
    recs.append("Bet timing is very consistent: randomize time-of-day you place bets.")

if not recs:
    st.success("Profile looks reasonably recreational. Keep rotating markets, stakes, and timing.")
else:
    for r in recs:
        st.write(f"• {r}")

st.caption("Prototype only. Risk logic will be tuned with real data.")
