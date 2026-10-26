"""합성 데이터로 엔진을 검증한다 (네트워크 불필요).  실행: python -m pytest crypto_bot/tests"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot import backtest as bt  # noqa: E402
from bot import execution as ex  # noqa: E402
from bot import learn  # noqa: E402
from bot import strategies as st  # noqa: E402

CFG = {
    "costs": {"fee_rate": 0.005, "slippage_rate": 0.001},
    "learning": {"train_days": 730, "test_days": 90, "objective": "martin", "min_trades_per_year": 1,
                 "max_drawdown_limit": 0.35, "switch_margin": 0.1, "ensemble_top_k": 3},
    "risk": {"target_annual_vol": 0.35, "max_exposure": 1.0, "atr_stop_mult": 3.0,
             "kill_switch_drawdown": 0.25, "min_rebalance_fraction": 0.05},
    "live": {"max_order_usd": 200},
}


def synthetic(n=2000, seed=0):
    """상승장/하락장이 번갈아 나오는 가짜 코인 가격."""
    rng = np.random.default_rng(seed)
    drift = np.where((np.arange(n) // 250) % 2 == 0, 0.003, -0.003)
    r = drift + rng.normal(0, 0.035, n)
    close = 100 * np.exp(np.cumsum(r))
    idx = pd.date_range("2018-01-01", periods=n, freq="D", tz="UTC")
    high = close * (1 + np.abs(rng.normal(0, 0.01, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.01, n)))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"open": open_, "high": np.maximum(high, close), "low": np.minimum(low, close),
                         "close": close, "volume": 1.0}, index=idx)


def test_no_lookahead():
    """과거만 잘라 계산한 비중 == 전체로 계산한 비중 (미래 데이터가 새지 않음)."""
    df = synthetic()
    risk = bt.RiskParams()
    cut = 1500
    for name, (_, grid) in st.STRATEGIES.items():
        p = grid[0]
        full = bt.apply_risk(df, st.signal(df, name, p), risk)
        part = bt.apply_risk(df.iloc[:cut], st.signal(df.iloc[:cut], name, p), risk)
        assert np.allclose(full.iloc[:cut].to_numpy(), part.to_numpy()), name


def test_costs_reduce_returns():
    df = synthetic()
    exp = pd.Series(np.tile([0.0, 1.0], len(df) // 2), index=df.index)
    assert bt.run(df, exp, 0.01).sum() < bt.run(df, exp, 0.0).sum()


def test_cash_is_zero_risk():
    df = synthetic()
    m = bt.metrics(bt.run(df, pd.Series(0.0, index=df.index), 0.006))
    assert m["max_drawdown"] == 0 and m["total_return"] == 0


def test_walk_forward_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(learn, "STATE_DIR", tmp_path)
    monkeypatch.setattr(learn, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(learn, "ACTIVE_PATH", tmp_path / "active.json")
    data = {"AAA-USD": synthetic(seed=1), "BBB-USD": synthetic(seed=2)}
    active = learn.run_learning(CFG, data)
    for sym, p in active["symbols"].items():
        assert p["name"] in list(st.STRATEGIES) + [learn.VOL_HOLD, "ensemble", "cash"]
    # 덜 잃기: 선택 전략의 최대낙폭이 단순보유보다 작아야 한다
    assert active["portfolio_oos"]["max_drawdown"] < active["portfolio_buy_and_hold"]["max_drawdown"]
    assert (tmp_path / "learn_report.md").exists()


def _signal(exp_btc, exp_eth):
    return {"generated_at": "x", "weight_per_symbol": 0.5,
            "symbols": {"BTC-USD": {"target_exposure": exp_btc}, "ETH-USD": {"target_exposure": exp_eth}}}


def test_plan_buys_within_budget():
    ledger = ex.load_ledger(Path("/nonexistent"), 200)
    plan = ex.plan_orders(_signal(1.0, 0.5), ledger, {}, {"BTC-USD": 100000, "ETH-USD": 4000}, CFG,
                          available_cash=200)
    spend = sum(o["dollar_amount"] for o in plan["orders"])
    assert all(o["side"] == "buy" for o in plan["orders"])
    assert spend <= 200 and spend > 140


def test_plan_respects_actual_buying_power():
    ledger = ex.load_ledger(Path("/nonexistent"), 200)
    plan = ex.plan_orders(_signal(1.0, 1.0), ledger, {}, {"BTC-USD": 100000, "ETH-USD": 4000}, CFG,
                          available_cash=10)
    assert sum(o["dollar_amount"] for o in plan["orders"]) <= 10


def test_full_exit_and_kill_switch():
    ledger = ex.load_ledger(Path("/nonexistent"), 200)
    ledger.update(cash=0.0, peak_value=200)
    holdings = {"BTC-USD": 0.001, "ETH-USD": 0.02}
    prices = {"BTC-USD": 70000, "ETH-USD": 3000}  # 가치 130 → 고점대비 -35%
    plan = ex.plan_orders(_signal(1.0, 1.0), ledger, holdings, prices, CFG)
    assert plan["halted"]
    assert {o["symbol"] for o in plan["orders"]} == {"BTC-USD", "ETH-USD"}
    assert all(o["side"] == "sell" and "quantity" in o for o in plan["orders"])
