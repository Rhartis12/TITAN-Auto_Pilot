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
try:
    df_krx = fdr.StockListing('KRX')
    # '티커:종목명' 딕셔너리 (예: '005930': '삼성전자')
    NAME_MAP = df_krx.set_index('Code')['Name'].to_dict()
    # FDR 섹터 정보 백업 (yfinance 실패 시 최후의 수단)
    fdr_sectors = df_krx.set_index('Code')['Sector'].to_dict()
except Exception as e:
    print(f"⚠️ KRX Listing Error: {e}")
    NAME_MAP = {}
    fdr_sectors = {}

# 2. KOSPI 시총 상위 200개 선정 (FDR 사용)
# KOSPI 데이터가 없으면 전체에서 필터링
df_kospi = df_krx[df_krx['Market'] == 'KOSPI'].sort_values('Marcap', ascending=False).head(200)
tickers = [f"{code}.KS" for code in df_kospi['Code']]

# [수정] yfinance 섹터 강제 조회 로직
print(f"   - Fetching Sectors for {len(tickers)} tickers from yfinance...")
sector_map = {}

for i, t in enumerate(tickers):
    pure_code = t.replace('.KS', '')
    
    # 1순위: yfinance (글로벌 표준 섹터명)
    yf_sector = None
    try:
        ticker_obj = yf.Ticker(t)
        # fast_info는 네트워크 요청을 줄이고 더 빠름 (최신 yfinance 기능)
        # 하지만 섹터 정보는 여전히 .info에 있을 수 있음. 우선순위 체크.
        
        # 1. .info 접근 (네트워크 요청 발생)
        info = ticker_obj.info
        yf_sector = info.get('sector')
        
    except Exception:
        yf_sector = None

    # 2. 결과 처리
    if yf_sector and yf_sector != "Unknown":
        # yfinance 성공
        sector_map[t] = yf_sector
        # 로그가 너무 많으면 주석 처리하세요
        # print(f"     [YF] {t}: {yf_sector}") 
    else:
        # yfinance 실패 시 -> FDR 정보 사용 (백업)
        fdr_sec = fdr_sectors.get(pure_code, 'Unknown')
        sector_map[t] = fdr_sec
        print(f"     ⚠️ [FDR Fallback] {t}: YF failed -> Using {fdr_sec}")

    # 차단 방지 딜레이 (yfinance 연속 호출 시 필수)
    if i % 10 == 0: 
        print(f"     ... Processed {i}/{len(tickers)}")
    time.sleep(0.2) # 딜레이를 조금 더 주어(0.2초) 안정성 확보


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
