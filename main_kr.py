import pandas as pd
import numpy as np
import yfinance as yf
import requests
from scipy.stats import spearmanr
import os
import warnings
import json
import FinanceDataReader as fdr
from datetime import datetime
import time # sleep용

warnings.filterwarnings("ignore")

# ============================================================
# 1. USER CONFIG
# ============================================================
START_DATE = "2000-01-01"
REBAL_FREQ = "ME"
ECOS_KEY = os.getenv("ECOS_KEY", "N671R802ZP944AEQ5J53") 
CACHE_FILE = "krx_yfinance_cache.pkl"

# 매크로 및 전략 설정
Z_WINDOW = 36
Z_THRESH = 0.5
TOP_PCT = 0.2

# 텔레그램 설정
TG_TOKEN = os.getenv("TG_TOKEN")
TG_CHAT_ID = os.getenv("TG_CHAT_ID")

print("🚀 Titan V7.7 (KOSPI) Started...")

# ============================================================
# 2. MACRO DATA LOADING (ECOS)
# ============================================================
def fetch_ecos_long(stat_code, item_code, start_date):
    full_series = []
    periods = [("200001", "200912"), ("201001", "201912"), ("202001", "202612")]
    if start_date < "200001":
        periods.insert(0, ("198001", "199912"))

    for s, e in periods:
        url = f"https://ecos.bok.or.kr/api/StatisticSearch/{ECOS_KEY}/json/kr/1/500/{stat_code}/M/{s}/{e}/{item_code}/"
        try:
            res = requests.get(url).json()
            if 'StatisticSearch' in res:
                df = pd.DataFrame(res['StatisticSearch']['row'])
                full_series.append(df)
        except: continue

    if not full_series: return pd.Series(dtype=float)

    final_df = pd.concat(full_series)
    final_df['DATE'] = pd.to_datetime(final_df['TIME'], format='%Y%m') + pd.offsets.MonthEnd(0)
    final_df['VALUE'] = pd.to_numeric(final_df['DATA_VALUE'])
    return final_df.drop_duplicates('DATE').set_index('DATE')['VALUE'].sort_index()

rate_series = fetch_ecos_long("721Y001", "5050000", "200001")
cpi_series = fetch_ecos_long("901Y009", "0", "198001")

# ============================================================
# 3. PRICE & SECTOR LOADING (Modified)
# ============================================================
print("🔄 Loading Stock Data...")

# 1. KRX 전체 리스트 조회 (종목명 매핑용)
df_krx = fdr.StockListing('KRX')

# [중요] 나중에 JSON 생성 시 사용할 '티커:종목명' 딕셔너리 생성
# 예: {'005930': '삼성전자', '000660': 'SK하이닉스'}
NAME_MAP = df_krx.set_index('Code')['Name'].to_dict()

# 2. KOSPI 시총 상위 200개 선정
df_kospi = df_krx[df_krx['Market'] == 'KOSPI'].sort_values('Marcap', ascending=False).head(200)
tickers = [f"{code}.KS" for code in df_kospi['Code']]

# [수정] 섹터 정보: yfinance 우선 -> 실패 시 FDR 정보 사용 (하이브리드)
print(f"   - Fetching Sectors for {len(tickers)} tickers...")
sector_map = {}

# FDR 섹터 정보 백업 (yfinance 실패 대비)
fdr_sectors = df_kospi.set_index('Code')['Sector'].to_dict()

for i, t in enumerate(tickers):
    pure_code = t.replace('.KS', '') # 005930.KS -> 005930
    
    try:
        # 1순위: yfinance (글로벌 표준 섹터명)
        info = yf.Ticker(t).info
        sec = info.get('sector', 'Unknown')
        
        # yfinance가 비어있거나 Unknown이면 FDR(한국표준) 사용
        if sec == 'Unknown' or sec is None:
            sec = fdr_sectors.get(pure_code, 'Unknown')
            
        sector_map[t] = sec
    except:
        # 에러 시 FDR 정보 사용
        sector_map[t] = fdr_sectors.get(pure_code, 'Unknown')
    
    # 진행률 표시
    if i % 50 == 0: print(f"     ... {i}/{len(tickers)}")
    time.sleep(0.05) # 차단 방지용 미세 딜레이

# Price Download
price = yf.download(tickers, start=START_DATE, progress=False)['Close']
if isinstance(price.columns, pd.MultiIndex):
    price.columns = price.columns.get_level_values(-1)

price = price.resample(REBAL_FREQ).last().ffill()
ret = price.pct_change()

# Macro Align
rate = rate_series.reindex(price.index).ffill()
cpi = cpi_series.pct_change(12).reindex(price.index).ffill()

macro_raw = pd.concat([rate, cpi], axis=1).ffill()
macro_raw.columns = ["RATE", "CPI"]
macro_z = (macro_raw - macro_raw.rolling(Z_WINDOW).mean()) / macro_raw.rolling(Z_WINDOW).std()
macro_z = macro_z.fillna(0)

# ============================================================
# 4. STRATEGY ENGINE
# ============================================================
print("⚙️ Calculating Factors...")

sectors = list(set(sector_map.values()))
raw_val = 1 / price
value_z = pd.DataFrame(np.nan, index=raw_val.index, columns=raw_val.columns)

# 최근 1년치만 계산 (속도 최적화)
for dt in raw_val.index[-13:]:
    for s in sectors:
        codes = [c for c, sec in sector_map.items() if sec == s and c in raw_val.columns]
        if len(codes) > 2:
            row = raw_val.loc[dt, codes]
            value_z.loc[dt, codes] = (row - row.mean()) / (row.std() if row.std() > 0 else 1)

FACTORS = {
    "VALUE": value_z.fillna(0),
    "MOM": (price.pct_change(12) - price.pct_change(1)).fillna(0),
    "LOWVOL": (-ret.rolling(12).std()).fillna(0),
    "QUALITY": (ret.rolling(12).mean() / ret.rolling(12).std()).fillna(0)
}

# IC & Weights
ic_raw = {}
for name, f_df in FACTORS.items():
    res = []
    rk = f_df.rank(axis=1, pct=True)
    for t in range(len(rk) - 1):
        x, y = rk.iloc[t], ret.iloc[t + 1]
        mask = x.notna() & y.notna()
        res.append(spearmanr(x[mask], y[mask])[0] if mask.sum() > 10 else 0)
    ic_raw[name] = pd.Series(res, index=rk.index[:-1])

ic_df = pd.DataFrame(ic_raw).rolling(12).mean().fillna(0)
mult = pd.DataFrame(1.0, index=ic_df.index, columns=ic_df.columns)

for t in mult.index:
    try:
        r_z, c_z = macro_z.loc[t, "RATE"], macro_z.loc[t, "CPI"]
        if (r_z > Z_THRESH) and (c_z > Z_THRESH): 
            mult.loc[t, "MOM"] = 0.0
            mult.loc[t, ["VALUE", "LOWVOL", "QUALITY"]] *= 1.5
        elif (r_z < -Z_THRESH): 
            mult.loc[t, ["MOM", "VALUE"]] *= 0.5
            mult.loc[t, ["QUALITY", "LOWVOL"]] *= 1.8
    except: continue

w_final = (ic_df * mult).clip(lower=0)
w_final = w_final.div(w_final.sum(axis=1).replace(0, 1), axis=0)
w_final = w_final.reindex(ret.index).ffill() # Forward Fill for Live

# ============================================================
# 5. LIVE SELECTION & JSON GEN
# ============================================================
print("💾 Generating Live Data...")

last_idx = price.index[-1]
latest_weights = w_final.iloc[-1]

final_score_series = pd.Series(0.0, index=price.columns)
for name in FACTORS:
    val_series = FACTORS[name].iloc[-1]
    valid_rank = val_series.rank(pct=True, ascending=True).fillna(0.5)
    weight = latest_weights[name]
    final_score_series += valid_rank * weight

# 상위 종목 선정
candidates = final_score_series.sort_values(ascending=False).head(int(len(final_score_series) * TOP_PCT))
latest_prices = price.iloc[-1]

# Buy List DF
buy_list = pd.DataFrame({
    'Ticker': candidates.index,
    'Score': candidates.values,
    'Sector': [sector_map.get(t, 'Unknown') for t in candidates.index],
    'Price': [latest_prices.get(t, 0) for t in candidates.index],
    'Weight': [1.0/len(candidates)] * len(candidates) 
})

# JSON 생성 (대시보드용)
last_r, last_c = macro_z.iloc[-1]['RATE'], macro_z.iloc[-1]['CPI']
regime = "Normal"
if (last_r > Z_THRESH) and (last_c > Z_THRESH): regime = "Stagflation"
elif (last_r < -Z_THRESH): regime = "Recession"
elif (last_r > Z_THRESH) and (last_c < 0): regime = "Overheat"

web_data = {
    "date": last_idx.strftime('%Y-%m-%d'),
    "regime": {"status": regime, "rate_z": round(last_r, 2), "cpi_z": round(last_c, 2)},
    "weights": latest_weights.to_dict(),
    "portfolio": []
}

for _, row in buy_list.iterrows():
    web_data["portfolio"].append({
        "ticker": row['Ticker'].replace(".KS", ""),
        "sector": row['Sector'],
        "price": row['Price'],
        "weight": row['Weight']
    })

# JSON 저장 (한국용 파일명: dashboard_data_kr.json)
with open("dashboard_data_kr.json", "w", encoding='utf-8') as f:
    json.dump(web_data, f, indent=4, ensure_ascii=False)

print("✅ KR Dashboard JSON Saved.")

# 텔레그램 전송
if TG_TOKEN and TG_CHAT_ID:
    msg = f"🇰🇷 Titan V7.7 KOSPI Update\nRegime: {regime}\nTop Pick: {buy_list.iloc[0]['Ticker']}"
    requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage", data={"chat_id": TG_CHAT_ID, "text": msg})
