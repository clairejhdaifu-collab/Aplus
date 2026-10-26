"""코인 로테이션 연구: 여러 코인 중 '지금 가장 힘센 코인'에 올라타는 전략.

모멘텀 로테이션 (제안 버전):
  주 1회, 최근 lookback 일 수익률 ÷ 변동성(위험 대비 추세 강도)이 가장 높은 상위 N개 코인을 보유.
  단, 그 코인이 하락추세(종가 < 이동평균 또는 수익률 ≤ 0)면 그 자리는 현금.
  보유 중 하락추세로 꺾이면 리밸런싱 날을 기다리지 않고 바로 현금화.
  비중은 변동성 타깃으로 조절 (거친 코인일수록 적게).

최대 변동성 매일 픽 (원래 아이디어, 비교용):
  매일, 최근 7일 수익률이 플러스인 코인 중 30일 변동성이 가장 큰 코인 하나에 전액.

둘 다 기존 시스템과 똑같은 비용·워크포워드 검증·덜 잃는 점수로 평가한다.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import backtest as bt
from . import learn


def panel(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    closes = pd.concat({s: df["close"] for s, df in data.items()}, axis=1).sort_index()
    full = pd.date_range(closes.index[0], closes.index[-1], freq="D", tz="UTC")
    return closes.reindex(full)  # 상장 전·데이터 공백은 NaN 그대로 (거래 불가)


def _vol(closes: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    return np.log(closes).diff().rolling(n, min_periods=n).std() * np.sqrt(bt.PERIODS_PER_YEAR)


def momentum_weights(closes: pd.DataFrame, lookback: int, top_n: int, filter_n: int,
                     target_vol: float, rebalance_days: int = 7, band: float = 0.05) -> pd.DataFrame:
    mom = closes / closes.shift(lookback) - 1
    sma = closes.rolling(filter_n, min_periods=filter_n).mean()
    vol = _vol(closes)
    ok = (closes > sma) & (mom > 0) & vol.notna() & (vol > 0)
    score = (mom / vol).where(ok)
    size = (target_vol / vol).clip(upper=1.0) / top_n

    W = np.zeros(closes.shape)
    w = np.zeros(closes.shape[1])
    okv, scv, szv = ok.to_numpy(), score.to_numpy(), size.to_numpy()
    for i in range(len(closes)):
        w = np.where(okv[i], w, 0.0)  # 매일: 하락추세로 꺾인 코인은 즉시 현금화
        if i % rebalance_days == 0:
            s = np.nan_to_num(scv[i], nan=-np.inf)
            pick = [j for j in np.argsort(-s)[:top_n] if np.isfinite(s[j])]
            new = np.zeros_like(w)
            new[pick] = szv[i][pick]
            # 계속 뽑힌 코인은 비중 차이가 작으면 그대로 둠 (수수료 절약)
            keep = (w > 0) & (new > 0) & (np.abs(new - w) < band)
            w = np.where(keep, w, new)
        W[i] = w
    return pd.DataFrame(W, index=closes.index, columns=closes.columns)


def max_vol_daily_weights(closes: pd.DataFrame) -> pd.DataFrame:
    vol = _vol(closes)
    mom7 = closes / closes.shift(7) - 1
    cand = vol.where(mom7 > 0)
    W = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    best = cand.dropna(how="all").idxmax(axis=1)
    for t, s in best.items():
        W.at[t, s] = 1.0
    return W


def buy_and_hold_equal(closes: pd.DataFrame) -> pd.DataFrame:
    avail = closes.notna()
    return avail.div(avail.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def run_portfolio(closes: pd.DataFrame, weights: pd.DataFrame, cost_rate: float) -> pd.Series:
    rets = closes.pct_change(fill_method=None).fillna(0.0)
    held = weights.shift(1).fillna(0.0)
    turnover = weights.diff().abs().fillna(weights.abs()).sum(axis=1)
    return (held * rets).sum(axis=1) - (turnover * cost_rate).shift(1).fillna(0.0)


def grid() -> list[dict]:
    return [dict(lookback=lb, top_n=n, filter_n=f)
            for lb, n, f in itertools.product([30, 60, 90], [1, 2, 3], [50, 100])]


def walk_forward(closes: pd.DataFrame, cfg: dict) -> dict:
    lc = cfg["learning"]
    cost = cfg["costs"]["fee_rate"] + cfg["costs"]["slippage_rate"]
    tv = cfg["risk"]["target_annual_vol"]
    rb = cfg.get("rotation", {}).get("rebalance_days", 7)
    args = (lc.get("objective", "martin"), lc["max_drawdown_limit"], lc.get("min_trades_per_year", 0))

    cands = []
    for p in grid():
        W = momentum_weights(closes, p["lookback"], p["top_n"], p["filter_n"], tv, rb)
        cands.append((p, W, run_portfolio(closes, W, cost)))

    first = 200
    starts = list(range(first + lc["train_days"], len(closes), lc["test_days"]))
    parts, picks = [], []
    for s in starts:
        hi = min(s + lc["test_days"], len(closes))
        lo = s - lc["train_days"]
        sc = [bt.score(bt.metrics(r.iloc[lo:s]), *args) for _, _, r in cands]
        j = int(np.argmax(sc))
        parts.append(cands[j][1].iloc[s:hi])
        picks.append(cands[j][0])
    oos_idx = closes.index[starts[0]:]
    W_oos = pd.concat(parts)
    res = {"momentum_rotation": run_portfolio(closes.loc[oos_idx], W_oos, cost)}
    res["max_vol_daily"] = run_portfolio(closes.loc[oos_idx], max_vol_daily_weights(closes).loc[oos_idx], cost)
    res["equal_weight_hold"] = run_portfolio(closes.loc[oos_idx], buy_and_hold_equal(closes).loc[oos_idx], cost)

    lo = max(0, len(closes) - lc["train_days"])
    sc = [bt.score(bt.metrics(r.iloc[lo:]), *args) for _, _, r in cands]
    latest = cands[int(np.argmax(sc))]
    turnover = {k: float(v.diff().abs().sum(axis=1).sum() / (len(v) / 365))
                for k, v in [("momentum_rotation", W_oos), ("max_vol_daily", max_vol_daily_weights(closes).loc[oos_idx])]}
    return {"returns": res, "picks": picks, "latest_params": latest[0],
            "latest_weights": latest[1].iloc[-1], "turnover_per_year": turnover, "oos_index": oos_idx}


def current_system_returns(data: dict[str, pd.DataFrame], cfg: dict, index: pd.DatetimeIndex) -> pd.Series:
    """기존 BTC/ETH 시스템(종목별 OOS 1위 전략, 균등분할)의 검증 수익률."""
    risk = bt.RiskParams.from_config(cfg)
    cost = cfg["costs"]["fee_rate"] + cfg["costs"]["slippage_rate"]
    rets = []
    for sym in ["BTC-USD", "ETH-USD"]:
        wf = learn.walk_forward(data[sym], learn.build_candidates(data[sym], risk), cost, cfg["learning"])
        res = {f: v for f, v in wf["results"].items() if f != "buy_and_hold"}
        best = max(res, key=lambda f: res[f]["score"])
        rets.append(res[best]["returns"] if res[best]["score"] > 0 else res[best]["returns"] * 0)
    return pd.concat(rets, axis=1).reindex(index).fillna(0.0).mean(axis=1)


NAMES = {
    "momentum_rotation": "모멘텀 로테이션 (제안)",
    "max_vol_daily": "최대 변동성 매일 픽 (원래 아이디어)",
    "current_system": "현재 시스템 (BTC·ETH)",
    "equal_weight_hold": "전 코인 균등 보유",
    "btc_hold": "BTC 그냥 보유",
}


def report(data: dict[str, pd.DataFrame], cfg: dict) -> tuple[str, dict]:
    closes = panel(data)
    wf = walk_forward(closes, cfg)
    idx = wf["oos_index"]
    rets = dict(wf["returns"])
    rets["current_system"] = current_system_returns(data, cfg, idx)
    cost = cfg["costs"]["fee_rate"] + cfg["costs"]["slippage_rate"]
    btc = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    btc["BTC-USD"] = 1.0
    rets["btc_hold"] = run_portfolio(closes.loc[idx], btc.loc[idx], cost)

    lc = cfg["learning"]
    args = (lc.get("objective", "martin"), lc["max_drawdown_limit"], 0)
    rows = []
    for k, r in rets.items():
        m = bt.metrics(r)
        rows.append((bt.score(m, *args), k, m))
    rows.sort(key=lambda x: x[0], reverse=True)

    def pct(x):
        return f"{x * 100:+.1f}%"

    out = [f"# 코인 로테이션 백테스트 ({str(idx[0].date())} ~ {str(idx[-1].date())}, 처음 보는 구간)", "",
           f"- 후보 코인: {', '.join(s.replace('-USD', '') for s in closes.columns)}",
           f"- 비용: 매매 1회당 {cost:.1%} (Robinhood 실제 스프레드 약 1.9% 왕복 반영)",
           "- 점수 = 연수익률 ÷ 얼서지수 (덜 잃는 점수, 최대낙폭 35% 초과 시 감점)", "",
           "| 순위 | 전략 | 점수 | 연수익률 | 최대낙폭 | 최악의 달 | 최악의 날 | 샤프 |",
           "|---|---|---|---|---|---|---|---|"]
    for i, (sc, k, m) in enumerate(rows, 1):
        out.append(f"| {i} | {NAMES[k]} | {sc:.2f} | {pct(m['cagr'])} | {pct(-m['max_drawdown'])} | "
                   f"{pct(m['worst_month'])} | {pct(m['worst_day'])} | {m['sharpe']:.2f} |")
    t = wf["turnover_per_year"]
    out += ["", "## 매매 빈도 (1년간 사고판 금액 ÷ 자산)",
            f"- 모멘텀 로테이션: {t['momentum_rotation']:.0f}배 → 비용 약 {t['momentum_rotation'] * cost:.0%}/년",
            f"- 최대 변동성 매일 픽: {t['max_vol_daily']:.0f}배 → 비용 약 {t['max_vol_daily'] * cost:.0%}/년", "",
            "## 모멘텀 로테이션: 지금 쓸 설정과 오늘의 보유",
            f"- 설정: {wf['latest_params']}"]
    held = {s: w for s, w in wf["latest_weights"].items() if w > 0}
    out.append("- 오늘: " + (", ".join(f"{s} {w:.0%}" for s, w in held.items()) if held else "전액 현금"))
    out += ["", "> 주의: 후보 코인은 '오늘 살아남은' 코인이라 과거 성과가 실제보다 좋게 나올 수 있습니다 (생존 편향).",
            "> LUNA·FTT 처럼 사라진 코인은 목록에 없습니다.", ""]
    summary = {k: {"score": sc, **{x: m[x] for x in ("cagr", "max_drawdown", "worst_month")}} for sc, k, m in rows}
    return "\n".join(out), summary
