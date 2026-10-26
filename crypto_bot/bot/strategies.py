"""지표 기반 매매 전략 모음.

모든 전략은 롱(매수) 전용이다: 1 = 코인 보유, 0 = 현금(USD) 보유.
공매도·레버리지는 쓰지 않는다 — "덜 잃는" 목표에 맞추기 위해서.
각 전략은 그날 종가까지의 정보만 사용한다 (미래 정보 사용 없음).
"""
from __future__ import annotations

import itertools
from typing import Callable

import numpy as np
import pandas as pd

from . import indicators as ind


def hold_between(entry: pd.Series, exit_: pd.Series) -> pd.Series:
    """entry 가 뜨면 들어가고 exit 이 뜨면 나오는 상태 신호. 같은 날 둘 다면 exit 우선."""
    sig = pd.Series(np.nan, index=entry.index)
    sig[entry.fillna(False).astype(bool)] = 1.0
    sig[exit_.fillna(False).astype(bool)] = 0.0
    return sig.ffill().fillna(0.0)


def _flag(cond: pd.Series) -> pd.Series:
    return cond.fillna(False).astype(float)


def sma_cross(df, fast, slow):
    c = df["close"]
    return _flag(ind.sma(c, fast) > ind.sma(c, slow))


def ema_cross(df, fast, slow):
    c = df["close"]
    return _flag(ind.ema(c, fast) > ind.ema(c, slow))


def trend_filter(df, n):
    c = df["close"]
    return _flag(c > ind.sma(c, n))


def macd_trend(df, fast, slow, signal, above_zero):
    line, sig, _ = ind.macd(df["close"], fast, slow, signal)
    cond = line > sig
    if above_zero:
        cond &= line > 0
    return _flag(cond)


def rsi_momentum(df, n, enter, leave):
    r = ind.rsi(df["close"], n)
    return hold_between(r > enter, r < leave)


def rsi_dip_in_uptrend(df, n, buy_below, sell_above):
    c = df["close"]
    r = ind.rsi(c, n)
    up = c > ind.sma(c, 200)
    return hold_between((r < buy_below) & up, (r > sell_above) | ~up)


def bollinger_revert(df, n, k):
    c = df["close"]
    lo, mid, _ = ind.bollinger(c, n, k)
    up = c > ind.sma(c, 200)
    return hold_between((c < lo) & up, (c > mid) | ~up)


def donchian_breakout(df, entry_n, exit_n):
    c = df["close"]
    _, hi = ind.donchian(df, entry_n)
    lo, _ = ind.donchian(df, exit_n)
    return hold_between(c > hi, c < lo)


def supertrend(df, n, mult):
    return ind.supertrend(df, n, mult)


def adx_trend(df, threshold, n):
    c = df["close"]
    a = ind.adx(df, 14)
    return hold_between((a > threshold) & (c > ind.ema(c, n)), c < ind.ema(c, n))


def ts_momentum(df, lookback):
    c = df["close"]
    return _flag(c / c.shift(lookback) - 1 > 0)


def buy_and_hold(df):
    return pd.Series(1.0, index=df.index)


def _grid(**axes):
    keys = list(axes)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(axes[k] for k in keys))]


# 이름 → (함수, 파라미터 후보 목록)
STRATEGIES: dict[str, tuple[Callable[..., pd.Series], list[dict]]] = {
    "sma_cross": (sma_cross, [p for p in _grid(fast=[10, 20, 50], slow=[100, 150, 200]) if p["fast"] < p["slow"]]),
    "ema_cross": (ema_cross, _grid(fast=[12, 21, 30], slow=[50, 100, 150])),
    "trend_filter": (trend_filter, _grid(n=[50, 100, 150, 200])),
    "macd_trend": (macd_trend, [
        dict(fast=f, slow=s, signal=g, above_zero=z)
        for (f, s, g) in [(12, 26, 9), (8, 21, 5), (19, 39, 9)] for z in (False, True)
    ]),
    "rsi_momentum": (rsi_momentum, [dict(n=14, enter=55, leave=45), dict(n=14, enter=60, leave=40),
                                    dict(n=21, enter=55, leave=45), dict(n=21, enter=60, leave=45)]),
    "rsi_dip_in_uptrend": (rsi_dip_in_uptrend, _grid(n=[14], buy_below=[30, 40], sell_above=[55, 70])),
    "bollinger_revert": (bollinger_revert, _grid(n=[20], k=[2.0, 2.5])),
    "donchian_breakout": (donchian_breakout, _grid(entry_n=[20, 55], exit_n=[10, 20])),
    "supertrend": (supertrend, _grid(n=[10, 14], mult=[2.0, 3.0, 4.0])),
    "adx_trend": (adx_trend, _grid(threshold=[20, 25], n=[50, 100])),
    "ts_momentum": (ts_momentum, _grid(lookback=[30, 60, 90, 180])),
}

BASELINE = "buy_and_hold"


def signal(df: pd.DataFrame, name: str, params: dict) -> pd.Series:
    if name == BASELINE:
        return buy_and_hold(df)
    fn, _ = STRATEGIES[name]
    return fn(df, **params)


def param_key(params: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in params.items())
