import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
from scipy.stats import spearmanr
import requests
import os
import warnings
import time
from datetime import datetime
import json

warnings.filterwarnings("ignore")

# ============================================================
# 1. USER CONFIG (절대 수정 금지: 검증된 로직)
# ============================================================
# ⚠️ SEC 크롤링을 위해 본인 이메일을 입력해야 차단되지 않습니다.
SEC_USER_AGENT = "borodin21651@gmail.com" 

# 환경변수 로드
FRED_KEY = os.getenv("FRED_KEY")
TG_TOKEN = os.getenv("TG_TOKEN")
TG_CHAT_ID = os.getenv("TG_CHAT_ID")

START_DATE = "2010-01-01"
END_DATE = pd.Timestamp.now().strftime('%Y-%m-%d')
REBAL_FREQ = "ME" # 월간 리밸런싱 (Titan V6.9 핵심)
SEC_FILE = "sec_facts_bt_ready_15yr.csv"

# ============================================================
# 2. SEC DATA SCRAPER (Robust)
# ============================================================
def fetch_sec_eps(tickers):
    print(f"📡 [SEC Scraper] Fetching EPS data for {len(tickers)} tickers...")
    all_data = []
    headers = {"User-Agent": SEC_USER_AGENT}
    
    try:
        cik_map_url = "https://www.sec.gov/files/company_tickers.json"
        cik_data = requests.get(cik_map_url, headers=headers).json()
        cik_df = pd.DataFrame.from_dict(cik_data, orient='index')
        cik_df['cik_str'] = cik_df['cik_str'].astype(str).str.zfill(10)
        ticker_to_cik = dict(zip(cik_df['ticker'], cik_df['cik_str']))
    except Exception as e:
        print(f"❌ CIK Load Failed: {e}"); return None

    for i, t in enumerate(tickers):
        clean_t = t.replace("-", "")
        cik = ticker_to_cik.get(clean_t)
        if not cik: continue
        
        try:
            url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
            resp = requests.get(url, headers=headers)
            
            if resp.status_code == 200:
                data = resp.json()
                concepts = data.get('facts', {}).get('us-gaap', {})
                eps_facts = concepts.get('EarningsPerShareBasic', {}).get('units', {}).get('USD/shares', [])
                
                for entry in eps_facts:
                    if entry.get('form') in ['10-K', '10-Q']:
                        all_data.append({
                            'Ticker': t,
                            'Year': int(str(entry['end'])[:4]),
                            'EPS': entry['val'],
                            'End': entry['end']
                        })
            time.sleep(0.11) 
            if i % 50 == 0: print(f"   Progress: {i}/{len(tickers)}...")
        except: continue
        
    df = pd.DataFrame(all_data)
    if not df.empty:
        df['End'] = pd.to_datetime(df['End'])
        df = df.sort_values('End').drop_duplicates(subset=['Ticker', 'Year'], keep='last')
        df.to_csv(SEC_FILE, index=False)
        print("✅ SEC Data Saved.")
        return df
    return None

# ============================================================
# 3. UNIVERSE & PRICE LOADING
# ============================================================
def get_sp500_tickers():
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {'User-Agent': 'Mozilla/5.0'}
        df = pd.read_html(requests.get(url, headers=headers).text)[0]
        df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
        return df
    except:
        return pd.DataFrame({'Symbol': ["AAPL", "MSFT", "GOOG"], 'GICS Sector': ['Tech']*3})

print("🔄 Step 1: Loading Market Data...")
univ = get_sp500_tickers()
TICKERS = univ["Symbol"].tolist()
SECTOR_MAP = univ.set_index("Symbol")["GICS Sector"].to_dict()

# Load Price
price = yf.download(TICKERS, start=START_DATE, end=END_DATE, progress=False)["Close"]
if isinstance(price.columns, pd.MultiIndex):
    price.columns = price.columns.get_level_values(-1)

price = price.resample(REBAL_FREQ).last()
ret = price.pct_change(fill_method=None)

# Load SEC
if not os.path.exists(SEC_FILE):
    sec_df = fetch_sec_eps(TICKERS)
else:
    print(f"✅ Found existing {SEC_FILE}. Loading...")
    sec_df = pd.read_csv(SEC_FILE)

# Fundamental Processing
if sec_df is not None:
    sec_df["Ticker"] = sec_df["Ticker"].astype(str)
    eps_annual = sec_df.pivot(index='Year', columns='Ticker', values='EPS')
    eps_monthly = pd.DataFrame(np.nan, index=price.index, columns=price.columns)
    
    for year in eps_annual.index:
        start_m, end_m = f"{year + 1}-05-01", f"{year + 2}-04-30"
        period = eps_monthly.index.intersection(pd.date_range(start_m, end_m, freq='ME'))
        if not period.empty:
            y_eps = eps_annual.loc[year].reindex(eps_monthly.columns)
            eps_monthly.loc[period] = np.tile(y_eps.values, (len(period), 1))
    eps_monthly = eps_monthly.ffill()
else:
    eps_monthly = None

# ============================================================
# 4. MACRO & FACTOR ENGINE
# ============================================================
print("🌍 Step 2: Macro Regime (Z-Score)...")
fred = Fred(api_key=FRED_KEY)
rate = fred.get_series("DGS10").resample(REBAL_FREQ).last().ffill()
cpi = fred.get_series("CPIAUCSL").pct_change(12).resample(REBAL_FREQ).last().ffill()
macro_raw = pd.concat([rate, cpi], axis=1).reindex(ret.index).ffill()
macro_raw.columns = ["RATE", "CPI"]

# Z-Score (36M Rolling)
macro_z = (macro_raw - macro_raw.rolling(36).mean()) / macro_raw.rolling(36).std()
macro_z = macro_z.fillna(0)

print("⚙️ Step 3: Factor Engineering...")
if eps_monthly is not None:
    raw_ep = eps_monthly / price
    raw_ep = raw_ep.fillna(1.0 / price)
else:
    raw_ep = 1.0 / price

# Sector-Neutral Value
value_z = pd.DataFrame(np.nan, index=raw_ep.index, columns=raw_ep.columns)
for dt in raw_ep.index:
    for s in univ["GICS Sector"].unique():
        secs = [t for t in TICKERS if SECTOR_MAP.get(t) == s and t in raw_ep.columns]
        if len(secs) > 2:
            row = raw_ep.loc[dt, secs]
            if row.std() > 0:
                value_z.loc[dt, secs] = (row - row.mean()) / row.std()
            else:
                value_z.loc[dt, secs] = 0
VALUE = value_z.fillna(0)

MOM = price.pct_change(12) - price.pct_change(1)
LOWVOL = -ret.rolling(12).std()
QUALITY = ret.rolling(12).mean() / ret.rolling(12).std()
FACTORS = {"VALUE": VALUE, "MOM": MOM, "LOWVOL": LOWVOL, "QUALITY": QUALITY}

# ============================================================
# 5. DYNAMIC WEIGHTING & PORTFOLIO
# ============================================================
print("🧠 Step 4: Titan Regime Interaction...")
# IC Calculation
ic_raw = {}
for name, f_df in FACTORS.items():
    ic_list = []
    rk = f_df.rank(axis=1, pct=True)
    for t in range(len(rk)-1):
        x = rk.iloc[t]
        y = ret.iloc[t+1]
        valid = x.notna() & y.notna()
        if valid.sum() > 10:
            ic_list.append(spearmanr(x[valid], y[valid])[0])
        else:
            ic_list.append(0)
    ic_raw[name] = pd.Series(ic_list, index=rk.index[:-1])

ic_df = pd.DataFrame(ic_raw).rolling(12).mean().fillna(0)
mult = pd.DataFrame(1.0, index=ic_df.index, columns=ic_df.columns)

# Titan V6.9 Logic
Z_THRESH = 0.5
for t in mult.index:
    r_z, c_z = macro_z.loc[t, "RATE"], macro_z.loc[t, "CPI"]
    if (r_z > Z_THRESH) and (c_z > Z_THRESH): mult.loc[t, :] = [1.2, 0.0, 1.5, 1.5]
    elif (r_z < -Z_THRESH): mult.loc[t, :] = [0.5, 0.0, 1.0, 1.5]
    elif (r_z > Z_THRESH) and (c_z < 0): mult.loc[t, ["VALUE", "MOM"]] = [1.5, 0.8]

w_final = (ic_df * mult).clip(lower=0)
w_final = w_final.div(w_final.sum(axis=1).replace(0, 1), axis=0).fillna(0)

# Build Portfolio & 100% Allocation Logic
score = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
for name, f_df in FACTORS.items():
    if name in w_final.columns:
        score += f_df.rank(axis=1, pct=True).multiply(w_final[name], axis=0)

alloc = pd.DataFrame(0.0, index=score.index, columns=score.columns)

# Allocation Fix: Ensure 100% Invested
for t in score.index:
    daily_picks = []
    for s in univ["GICS Sector"].unique():
        cols = [c for c in score.columns if SECTOR_MAP.get(c) == s]
        sub = score.loc[t, cols].dropna()
        if not sub.empty:
            picks = sub[sub >= sub.quantile(1 - 0.2)].index.tolist()
            daily_picks.extend(picks)
    if daily_picks:
        alloc.loc[t, daily_picks] = 1.0 / len(daily_picks)

# Backtest Stats Calculation
port_ret = (alloc.shift(1).fillna(0) * ret).sum(axis=1)
vol_lev = (0.12 / (port_ret.rolling(6).std() * np.sqrt(12))).clip(0, 1.2).shift(1).fillna(1.0)
final_ret = port_ret * vol_lev

def stats(r):
    c = (1+r).cumprod(); ann = 12
    return c.iloc[-1]**(1/(len(r)/ann))-1, (c/c.cummax()-1).min(), (r.mean()*ann-0.02)/(r.std()*np.sqrt(ann))

s_p = stats(final_ret)
msg = f"🚀 Titan V6.9.1 Live\nCAGR: {s_p[0]:.2%}\nMDD: {s_p[1]:.2%}\nSharpe: {s_p[2]:.2f}"
print(msg)

# ============================================================
# 7. FILE MANAGEMENT & WEB DASHBOARD DATA
# ============================================================
print("💾 Step 5: Archiving Results & Generating Dashboard Data...")

today = datetime.now()
month_str = today.strftime("%Y-%m")      # 예: 2026-01
date_str = today.strftime("%Y-%m-%d")    # 예: 2026-01-30

# [중요] 최신 Buy List 생성 (비중 0 초과 종목만)
latest_weights = alloc.iloc[-1]
active_weights = latest_weights[latest_weights > 0]
latest_prices_series = price.iloc[-1] 

buy_list = pd.DataFrame({
    'Ticker': active_weights.index,
    'Weight': active_weights.values,
    'Sector': [SECTOR_MAP.get(t, 'N/A') for t in active_weights.index],
    'Score': [score.iloc[-1][t] for t in active_weights.index],
    'Price': [latest_prices_series.get(t, 0.0) for t in active_weights.index]
}).sort_values('Weight', ascending=False)

# 📂 폴더 생성 (results/2026-01/)
base_dir = f"results/{month_str}"
os.makedirs(base_dir, exist_ok=True)

# 📄 파일 경로 정의
buy_list_path = f"{base_dir}/Buy_List_{date_str}.csv"
scores_path = f"{base_dir}/Full_Scores_{date_str}.csv"
ic_path = f"{base_dir}/IC_Factors_{date_str}.csv"
weights_path = f"{base_dir}/Weights_{date_str}.csv"
returns_path = f"{base_dir}/Returns_{date_str}.csv"

# 💾 CSV 파일 저장 (기록용)
buy_list.to_csv(buy_list_path, index=False)
score.to_csv(scores_path)
ic_df.to_csv(ic_path)
w_final.to_csv(weights_path)
final_ret.to_csv(returns_path)

# ------------------------------------------------------------
# 🌍 JSON Data Generation (For HTML Dashboard)
# ------------------------------------------------------------
# 1. 현재 국면 정보
last_r_z = macro_z.iloc[-1]['RATE']
last_c_z = macro_z.iloc[-1]['CPI']

if (last_r_z > Z_THRESH) and (last_c_z > Z_THRESH): current_regime = "Stagflation"
elif (last_r_z < -Z_THRESH): current_regime = "Recession"
elif (last_r_z > Z_THRESH) and (last_c_z < 0): current_regime = "Overheat"
else: current_regime = "Normal"

# 2. 포트폴리오 리스트 변환
portfolio_json = []
for idx, row in buy_list.iterrows():
    portfolio_json.append({
        "ticker": row['Ticker'],
        "sector": row['Sector'],
        "price": row['Price'],
        "weight": row['Weight']
    })

# 3. 팩터 비중
current_factor_weights = w_final.iloc[-1].to_dict()

# 4. JSON 생성
web_data = {
    "date": date_str,
    "regime": {
        "status": current_regime,
        "rate_z": round(last_r_z, 2),
        "cpi_z": round(last_c_z, 2)
    },
    "weights": current_factor_weights,
    "portfolio": portfolio_json
}

# [🚨 핵심 수정] 웹사이트가 읽을 수 있게 Root 폴더에도 저장
with open("dashboard_data.json", "w") as f:
    json.dump(web_data, f, indent=4)

# 기록용 폴더에도 저장
with open(f"{base_dir}/dashboard_data.json", "w") as f:
    json.dump(web_data, f, indent=4)

print("✅ Dashboard JSON generated.")

# ------------------------------------------------------------
# 📨 Telegram Notification
# ------------------------------------------------------------
def send_file(path, caption):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendDocument"
    try:
        with open(path, "rb") as f:
            requests.post(url, data={"chat_id": TG_CHAT_ID, "caption": caption}, files={"document": f})
    except Exception as e:
        print(f"❌ File upload failed ({path}): {e}")

if TG_TOKEN and TG_CHAT_ID:
    # 메시지 전송
    requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage", 
                  data={"chat_id": TG_CHAT_ID, "text": msg, "parse_mode": "Markdown"})

    # 주요 파일 전송
    send_file(buy_list_path, f"📋 {month_str} Buy List (Final)")
    send_file(scores_path, f"💯 {month_str} Total Scores")
    send_file(ic_path, f"📈 {month_str} IC Factors")
    
    print(f"✅ Records saved to {base_dir} and sent to Telegram.")
else:
    print(f"⚠️ Telegram Token not found. Files saved locally.")
