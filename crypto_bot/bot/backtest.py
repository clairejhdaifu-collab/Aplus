"""백테스트 엔진 + 리스크 관리 + 성과 지표.

타이밍 규칙 (미래 정보 사용 금지):
  t일 종가까지 보고 목표 비중을 정하고 → t일 종가 직후 매매 → t+1일 수익률부터 반영.
비용: 매매한 비중만큼 (수수료 + 슬리피지)를 차감.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import indicators as ind

PERIODS_PER_YEAR = 365  # 코인은 365일 거래


@dataclass
class RiskParams:
    target_annual_vol: float = 0.35
    max_exposure: float = 1.0
    atr_stop_mult: float = 3.0
    min_rebalance_fraction: float = 0.05

    @classmethod
    def from_config(cls, cfg: dict) -> "RiskParams":
        r = cfg.get("risk", {})
        return cls(
            target_annual_vol=r.get("target_annual_vol", 0.35),
            max_exposure=r.get("max_exposure", 1.0),
            atr_stop_mult=r.get("atr_stop_mult", 3.0),
            min_rebalance_fraction=r.get("min_rebalance_fraction", 0.05),
        )


def apply_risk(df: pd.DataFrame, raw_signal: pd.Series, risk: RiskParams) -> pd.Series:
    """전략 신호(0/1)를 실제 보유 비중(0~max_exposure)으로 바꾼다.

    1) 변동성 타깃: 시장이 거칠수록 비중을 줄인다 (target_vol / 최근 변동성).
    2) ATR 트레일링 스톱: 보유 중 최고 종가에서 ATR×배수 만큼 빠지면 즉시 현금화,
       전략 신호가 한 번 꺼졌다가 다시 켜져야 재진입.
    3) 작은 비중 변화(min_rebalance_fraction 미만)는 무시 → 쓸데없는 수수료 방지.
    """
    close = df["close"].to_numpy()
    sig = raw_signal.reindex(df.index).fillna(0).to_numpy()
    vol = ind.realized_vol(df["close"], 30, PERIODS_PER_YEAR).to_numpy()
    a = ind.atr(df, 14).to_numpy()

    out = np.zeros(len(df))
    pos = 0.0
    peak = np.nan
    stopped = False
    for i in range(len(df)):
        if sig[i] <= 0:
            stopped = False
            target = 0.0
        elif stopped:
            target = 0.0
        else:
            if np.isnan(vol[i]) or vol[i] <= 0:
                target = 0.0  # 데이터가 부족한 초기 구간은 쉰다
            else:
                target = min(risk.max_exposure, risk.target_annual_vol / vol[i]) * sig[i]

        # 트레일링 스톱 점검 (보유 중일 때만)
        if pos > 0 and target > 0 and risk.atr_stop_mult > 0 and not np.isnan(a[i]):
            peak = close[i] if np.isnan(peak) else max(peak, close[i])
            if close[i] < peak - risk.atr_stop_mult * a[i]:
                target = 0.0
                stopped = True

        if target == 0.0 or abs(target - pos) >= risk.min_rebalance_fraction:
            if pos == 0.0 and target > 0:
                peak = close[i]
            pos = target
        if pos == 0.0:
            peak = np.nan
        out[i] = pos
    return pd.Series(out, index=df.index)


def run(df: pd.DataFrame, exposure: pd.Series, cost_rate: float) -> pd.Series:
    """일별 전략 수익률을 돌려준다."""
    asset_ret = df["close"].pct_change().fillna(0.0)
    held = exposure.shift(1).fillna(0.0)
    turnover = exposure.diff().abs().fillna(exposure.abs())
    cost = (turnover * cost_rate).shift(1).fillna(0.0)
    return held * asset_ret - cost


def metrics(returns: pd.Series, exposure: pd.Series | None = None) -> dict:
    r = returns.fillna(0.0)
    n = len(r)
    if n < 2:
        return {}
    equity = (1 + r).cumprod()
    years = n / PERIODS_PER_YEAR
    total = float(equity.iloc[-1] - 1)
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if equity.iloc[-1] > 0 else -1.0
    dd = equity / equity.cummax() - 1
    max_dd = float(-dd.min())
    ulcer = float(np.sqrt((dd ** 2).mean()))
    std = r.std()
    downside = r[r < 0].std()
    monthly = (1 + r).resample("ME").prod() - 1 if isinstance(r.index, pd.DatetimeIndex) else r
    out = {
        "total_return": total,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "ulcer_index": ulcer,
        "martin": cagr / max(ulcer, 0.01),
        "calmar": cagr / max(max_dd, 0.01),
        "sharpe": float(r.mean() / std * np.sqrt(PERIODS_PER_YEAR)) if std > 0 else 0.0,
        "sortino": float(r.mean() / downside * np.sqrt(PERIODS_PER_YEAR)) if downside and downside > 0 else 0.0,
        "worst_day": float(r.min()),
        "worst_month": float(monthly.min()) if len(monthly) else 0.0,
        "days": n,
    }
    if exposure is not None:
        e = exposure.reindex(r.index).fillna(0)
        out["time_in_market"] = float((e > 0).mean())
        out["trades"] = int(((e > 0) & (e.shift(1).fillna(0) == 0)).sum())
    return out


def score(m: dict, objective: str = "martin", max_dd_limit: float = 0.35,
          min_trades_per_year: float = 0.0) -> float:
    """'덜 잃는' 점수. 높을수록 좋다.

    기본 목표 martin = 연수익률 ÷ 얼서지수(하락의 깊이와 기간을 함께 반영하는 고통지수).
    - 최대낙폭이 한도를 넘으면 큰 감점.
    - 수익이 마이너스면 점수는 그 손실 자체 (적게 잃을수록 덜 나쁨).
    """
    if not m:
        return -np.inf
    if m["cagr"] <= 0:
        s = m["cagr"]
    else:
        s = m.get(objective, m["martin"])
    if m["max_drawdown"] > max_dd_limit:
        s -= 10 * (m["max_drawdown"] - max_dd_limit) + 1
    years = m["days"] / PERIODS_PER_YEAR
    if min_trades_per_year and "trades" in m and m["trades"] < min_trades_per_year * years * 0.5:
        s -= 0.5
    return float(s)
