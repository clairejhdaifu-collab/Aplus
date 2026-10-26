"""오늘의 목표 비중 계산 + 리밸런싱 주문 계획 + 장부(ledger) 관리.

같은 주문 계획 로직을 모의매매(paper)와 Robinhood 실매매가 함께 쓴다.
Robinhood 쪽은 Claude 가 MCP 로 잔고/시세를 읽어 이 모듈에 넣고, 나온 주문만 그대로 실행한다.
→ 매매 판단은 항상 코드(규칙)가 하고, 사람/AI 의 즉흥 판단이 끼어들지 않는다.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import backtest as bt
from .learn import STATE_DIR, load_active, member_exposure

SIGNAL_PATH = STATE_DIR / "signal.json"
MIN_ORDER_USD = 1.0


def compute_signal(cfg: dict, data: dict[str, pd.DataFrame]) -> dict:
    risk = bt.RiskParams.from_config(cfg)
    active = load_active().get("symbols", {})
    out = {}
    for sym, df in data.items():
        pick = active.get(sym, {"name": "cash", "members": []})
        exp = member_exposure(df, pick.get("members", []), risk)
        out[sym] = {
            "strategy": pick.get("name", "cash"),
            "target_exposure": round(float(exp.iloc[-1]), 4),
            "previous_exposure": round(float(exp.iloc[-2]), 4) if len(exp) > 1 else 0.0,
            "bar_time": str(df.index[-1]),
            "close": float(df["close"].iloc[-1]),
        }
    sig = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "weight_per_symbol": round(1 / max(len(out), 1), 6), "symbols": out}
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    SIGNAL_PATH.write_text(json.dumps(sig, indent=2))
    return sig


def load_ledger(path: Path, budget: float) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"budget": budget, "cash": budget, "peak_value": budget, "halted": False,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "history": []}


def save_ledger(path: Path, ledger: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, indent=2))


def plan_orders(signal: dict, ledger: dict, holdings: dict[str, float], prices: dict[str, float],
                cfg: dict, available_cash: float | None = None) -> dict:
    """목표 비중에 맞추는 주문 목록을 만든다.

    holdings: {"BTC-USD": 보유수량}, prices: {"BTC-USD": 현재가}
    available_cash: 계좌에서 실제로 쓸 수 있는 현금 (장부 현금보다 적으면 그만큼만 매수)
    """
    risk = cfg["risk"]
    w = signal["weight_per_symbol"]
    pos_value = {s: holdings.get(s, 0.0) * prices[s] for s in signal["symbols"]}
    value = ledger["cash"] + sum(pos_value.values())

    notes = []
    peak = max(ledger.get("peak_value", value), value)
    drawdown = 1 - value / peak if peak > 0 else 0.0
    if not ledger.get("halted") and drawdown >= risk["kill_switch_drawdown"]:
        ledger["halted"] = True
        notes.append(f"킬스위치 발동: 고점 대비 -{drawdown:.1%} → 전량 현금화 후 정지")
    if ledger.get("halted"):
        notes.append("정지 상태: 모든 목표 비중 0 (재개하려면 장부의 halted 를 false 로)")

    sells, buys = [], []
    band = risk["min_rebalance_fraction"] * value
    for s, info in signal["symbols"].items():
        target_exp = 0.0 if ledger.get("halted") else info["target_exposure"]
        target = value * w * target_exp
        diff = target - pos_value[s]
        if target_exp == 0 and holdings.get(s, 0) > 0:
            sells.append({"symbol": s, "side": "sell", "quantity": holdings[s],
                          "est_usd": round(pos_value[s], 2), "reason": "목표 0 → 전량 매도"})
        elif diff <= -max(band, MIN_ORDER_USD):
            sells.append({"symbol": s, "side": "sell", "dollar_amount": round(-diff, 2),
                          "est_usd": round(-diff, 2), "reason": f"비중 축소 → {target_exp:.0%}"})
        elif diff >= max(band, MIN_ORDER_USD):
            buys.append({"symbol": s, "side": "buy", "dollar_amount": round(diff, 2),
                         "est_usd": round(diff, 2), "reason": f"비중 확대 → {target_exp:.0%}"})

    cash_after_sells = ledger["cash"] + sum(o["est_usd"] for o in sells)
    spendable = cash_after_sells if available_cash is None else min(cash_after_sells, available_cash + sum(o["est_usd"] for o in sells))
    spendable *= 0.98  # 가격 변동·수수료 여유분
    max_order = cfg.get("live", {}).get("max_order_usd", float("inf"))
    final_buys = []
    for o in buys:
        amt = min(o["dollar_amount"], spendable, max_order)
        if amt >= MIN_ORDER_USD:
            o["dollar_amount"] = o["est_usd"] = round(amt, 2)
            final_buys.append(o)
            spendable -= amt
    return {"value": round(value, 2), "peak_value": round(peak, 2), "drawdown": round(drawdown, 4),
            "orders": sells + final_buys, "notes": notes, "halted": ledger.get("halted", False)}


def record_fills(ledger: dict, fills: list[dict], value_after: float | None = None) -> dict:
    """체결 결과로 장부 현금을 갱신. fill = {symbol, side, usd (체결금액, 수수료 포함 실지출/실수령)}"""
    for f in fills:
        if f["side"] == "buy":
            ledger["cash"] -= f["usd"]
        else:
            ledger["cash"] += f["usd"]
    ledger["cash"] = round(max(ledger["cash"], 0.0), 2)
    if value_after is not None:
        ledger["peak_value"] = round(max(ledger.get("peak_value", value_after), value_after), 2)
    ledger["history"].append({"time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                              "fills": fills, "value_after": value_after})
    ledger["history"] = ledger["history"][-500:]
    return ledger
