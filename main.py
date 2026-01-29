# main.py
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
from scipy.stats import spearmanr
from datetime import datetime, timedelta
import requests
import json
import os
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIG & SECRETS
# ============================================================
FRED_KEY = os.environ.get("FRED_API_KEY")
TG_TOKEN = os.environ.get("TG_TOKEN")
TG_CHAT_ID = os.environ.get("TG_CHAT_ID")

LOOKBACK_YEARS = 4
TOP_N = 20
Z_THRESH = 0.5
IC_WINDOW = 12

# 날짜 설정
END_DATE = datetime.today().strftime('%Y-%m-%d')
START_DATE = (datetime.today() - timedelta(days=365 * LOOKBACK_YEARS)).strftime('%Y-%m-%d')

print(f"🚀 Titan V6.9 Auto-Pilot Started ({END_DATE})")

# ============================================================
# 2. DATA LOADING
# ============================================================
print("🔄 [Step 1] Loading Data...")

# (1) Tickers
try:
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df_univ = table[0]
    TICKERS = df_univ['Symbol'].str.replace('.', '-').tolist()
    SECTOR_MAP = df_univ.set_index('Symbol')['GICS Sector'].to_dict()
    SECTORS = df_univ['GICS Sector'].unique()
except Exception as e:
    print(f"⚠️ Wiki Load Error: {e}")
    # Fallback (Demo)
    TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "TSLA", "META", "JPM", "V", "JNJ"]
    SECTOR_MAP = {t: "Tech" for t in TICKERS}
    SECTORS = ["Tech"]

# (2) Prices
price = yf.download(TICKERS, start=START_DATE, end=END_DATE, progress=False)['Close']
# Ensure simple columns
if isinstance(price.columns, pd.MultiIndex):
    price.columns = price.columns.get_level_values(-1)
    
price_m = price.resample('ME').last()
ret_m = price_m.pct_change()

# (3) Macro (FRED)
fred = Fred(api_key=FRED_KEY)
rate = fred.get_series("DGS10").resample('ME').last().ffill()
cpi = fred.get_series("CPIAUCSL").pct_change(12).resample('ME').last().ffill()
macro = pd.concat([rate, cpi], axis=1).dropna()
macro.columns = ["RATE", "CPI"]

# ============================================================
# 3. MACRO REGIME & IC CALCULATION
# ============================================================
print("🌍 [Step 2] Analyzing Macro & IC...")

# (A) Regime Z-Score
window = 36
macro_z = (macro - macro.rolling(window).mean()) / macro.rolling(window).std()
current_z = macro_z.iloc[-1]

regime_status = "Normal"
if (current_z['RATE'] > Z_THRESH) and (current_z['CPI'] > Z_THRESH):
    regime_status = "Stagflation"
elif (current_z['RATE'] < -Z_THRESH):
    regime_status = "Recession"
elif (current_z['RATE'] > Z_THRESH) and (current_z['CPI'] < 0):
    regime_status = "Overheat"

# (B) Factor Calculation
val = 1 / price_m
mom = price_m.pct_change(12) - price_m.pct_change(1)
lowvol = -ret_m.rolling(12).std()
qual = ret_m.rolling(12).mean() / ret_m.rolling(12).std()

factors = {"VALUE": val, "MOM": mom, "LOWVOL": lowvol, "QUALITY": qual}

# (C) Historical IC Calculation (For Report)
ic_data = {}
for name, f_df in factors.items():
    res = []
    # Rank for correlation
    rk = f_df.rank(axis=1, pct=True)
    for t in range(len(rk)-1):
        x = rk.iloc[t]
        y = ret_m.iloc[t+1]
        # Valid data check
        mask = x.notna() & y.notna()
        if mask.sum() > 10:
            res.append(spearmanr(x[mask], y[mask])[0])
        else:
            res.append(0)
    ic_data[name] = pd.Series(res, index=rk.index[:-1])

df_ic = pd.DataFrame(ic_data)
# Latest IC (Rolling 12M Mean)
latest_ic = df_ic.rolling(IC_WINDOW).mean().iloc[-1]

# ============================================================
# 4. SCORING & SELECTION
# ============================================================
print("⚙️ [Step 3] Selecting Stocks...")

# Determine Weights based on Regime
w = {'VALUE':0.25, 'MOM':0.25, 'LOWVOL':0.25, 'QUALITY':0.25}

if regime_status == "Stagflation":
    w = {'VALUE':0.3, 'MOM':0.0, 'LOWVOL':0.35, 'QUALITY':0.35}
elif regime_status == "Recession":
    w = {'VALUE':0.1, 'MOM':0.0, 'LOWVOL':0.4, 'QUALITY':0.5}
elif regime_status == "Overheat":
    w = {'VALUE':0.5, 'MOM':0.3, 'LOWVOL':0.1, 'QUALITY':0.1}

# Calculate Score
def get_rank(df_factor):
    raw = df_factor.iloc[-1]
    z = pd.Series(np.nan, index=raw.index)
    for sector in SECTORS:
        secs = [t for t in TICKERS if SECTOR_MAP.get(t)==sector]
        valid = [t for t in secs if t in raw.index]
        if not valid: continue
        z[valid] = raw[valid].rank(pct=True)
    return z

total_score = sum(get_rank(factors[k]) * v for k, v in w.items())

# Full Score DataFrame
df_full = pd.DataFrame({
    'Score': total_score,
    'Price': price_m.iloc[-1],
    'Sector': [SECTOR_MAP.get(t, 'N/A') for t in total_score.index]
}).dropna().sort_values('Score', ascending=False)

# Top N Selection
buy_list = df_full.head(TOP_N).copy()
buy_list['Weight'] = 1 / TOP_N

# ============================================================
# 5. EXPORT & WEB DATA
# ============================================================
print("💾 [Step 4] Saving Files...")

# (1) CSVs
today_str = datetime.today().strftime('%Y-%m-%d')
buy_list.to_csv(f"Buy_List.csv")
df_full.to_csv(f"Full_Scores.csv")
df_ic.to_csv(f"IC_Factors.csv")

# (2) JSON for Web Dashboard
web_data = {
    "date": today_str,
    "regime": {
        "status": regime_status,
        "rate_z": round(current_z['RATE'], 2),
        "cpi_z": round(current_z['CPI'], 2)
    },
    "weights": w,
    "ic": latest_ic.to_dict(),
    "portfolio": []
}

for t in buy_list.index:
    row = buy_list.loc[t]
    web_data["portfolio"].append({
        "ticker": t,
        "sector": row['Sector'],
        "score": round(row['Score'], 4),
        "price": round(row['Price'], 2),
        "weight": round(row['Weight'], 4)
    })

with open("dashboard_data.json", "w") as f:
    json.dump(web_data, f, indent=4)

# ============================================================
# 6. TELEGRAM NOTIFICATION
# ============================================================
print("📨 [Step 5] Sending Telegram...")

def send_file(path, caption):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendDocument"
    with open(path, "rb") as f:
        requests.post(url, data={"chat_id": TG_CHAT_ID, "caption": caption}, files={"document": f})

# Summary Message
msg = f"""
🚀 *Titan V6.9 Monthly Rebalance* ({today_str})

🌍 *Regime*: {regime_status}
📊 *Macro Z*: Rate {current_z['RATE']:.2f} | CPI {current_z['CPI']:.2f}

⚖️ *Active Weights*:
Value: {w['VALUE']:.1f} | Mom: {w['MOM']:.1f}
Vol: {w['LOWVOL']:.1f} | Qual: {w['QUALITY']:.1f}

✅ *Top 3 Picks*:
1. {buy_list.index[0]} ({buy_list.iloc[0]['Sector']})
2. {buy_list.index[1]} ({buy_list.iloc[1]['Sector']})
3. {buy_list.index[2]} ({buy_list.iloc[2]['Sector']})

🔗 *Dashboard*: [View Full Report](https://{os.environ.get('GITHUB_REPOSITORY_OWNER')}.github.io/{os.environ.get('GITHUB_REPOSITORY').split('/')[-1]}/)
"""

requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage", 
              data={"chat_id": TG_CHAT_ID, "text": msg, "parse_mode": "Markdown"})

send_file("Buy_List.csv", "📋 Final Buy List")
send_file("IC_Factors.csv", "📈 Factor IC History")
send_file("Full_Scores.csv", "💯 All Stock Scores")

print("✅ All Done.")
