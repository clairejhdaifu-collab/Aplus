"""시세 데이터 수집 + 로컬 캐시.

Coinbase 공개 캔들 API(키 불필요)에서 일봉을 받아 data/ 폴더에 CSV로 저장한다.
이미 받은 구간은 다시 받지 않고, 새로 생긴 캔들만 이어 붙인다.
아직 끝나지 않은(진행 중인) 캔들은 저장하지 않는다 — 미래 정보를 쓰는 실수를 막기 위해서.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

COINBASE_URL = "https://api.exchange.coinbase.com/products/{symbol}/candles"
GRANULARITY = {"1d": 86400, "6h": 21600, "1h": 3600}
MAX_CANDLES_PER_CALL = 300
COLUMNS = ["open", "high", "low", "close", "volume"]


def cache_path(symbol: str, timeframe: str) -> Path:
    return DATA_DIR / f"{symbol}_{timeframe}.csv"


def load_cached(symbol: str, timeframe: str) -> pd.DataFrame:
    path = cache_path(symbol, timeframe)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS, index=pd.DatetimeIndex([], tz="UTC", name="time"))
    df = pd.read_csv(path, index_col="time", parse_dates=["time"])
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df[COLUMNS].astype(float)


def _fetch_coinbase(symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    gran = GRANULARITY[timeframe]
    step = timedelta(seconds=gran * MAX_CANDLES_PER_CALL)
    rows: list[list[float]] = []
    cursor = start
    session = requests.Session()
    session.headers["User-Agent"] = "aplus-crypto-bot/1.0"
    while cursor < end:
        chunk_end = min(cursor + step, end)
        params = {"granularity": gran, "start": cursor.isoformat(), "end": chunk_end.isoformat()}
        for attempt in range(5):
            resp = session.get(COINBASE_URL.format(symbol=symbol), params=params, timeout=30)
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            rows.extend(resp.json())
            break
        else:
            raise RuntimeError(f"Coinbase rate limit: {symbol} {cursor}")
        cursor = chunk_end
        time.sleep(0.35)  # 공개 API 요청 한도 존중

    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    # Coinbase 형식: [time, low, high, open, close, volume]
    df = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.drop_duplicates("time").set_index("time").sort_index()
    return df[COLUMNS].astype(float)


def drop_unfinished(df: pd.DataFrame, timeframe: str, now: datetime | None = None) -> pd.DataFrame:
    now = now or datetime.now(timezone.utc)
    bar = pd.Timedelta(seconds=GRANULARITY[timeframe])
    return df[df.index + bar <= pd.Timestamp(now)]


def update(symbol: str, timeframe: str = "1d", history_start: str = "2017-01-01") -> pd.DataFrame:
    """캐시를 최신 상태로 맞추고 완성된 캔들 전체를 돌려준다."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cached = load_cached(symbol, timeframe)
    now = datetime.now(timezone.utc)
    if len(cached):
        start = cached.index[-1].to_pydatetime() + timedelta(seconds=GRANULARITY[timeframe])
    else:
        start = datetime.fromisoformat(history_start).replace(tzinfo=timezone.utc)

    if start < now:
        fresh = _fetch_coinbase(symbol, timeframe, start, now)
        merged = pd.concat([cached, fresh])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    else:
        merged = cached

    merged = drop_unfinished(merged[COLUMNS].astype(float), timeframe, now)
    merged.index.name = "time"
    merged.to_csv(cache_path(symbol, timeframe))
    return merged


def latest_price(symbol: str) -> float:
    resp = requests.get(f"https://api.exchange.coinbase.com/products/{symbol}/ticker", timeout=15)
    resp.raise_for_status()
    return float(resp.json()["price"])
