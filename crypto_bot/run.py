"""명령 모음.

  python crypto_bot/run.py learn     # 데이터 갱신 → 워크포워드 학습 → 전략 선택 → 리포트
  python crypto_bot/run.py signal    # 데이터 갱신 → 오늘의 목표 비중 (state/signal.json)
  python crypto_bot/run.py paper     # signal + 모의매매 1회 (state/paper_ledger.json)
  python crypto_bot/run.py rh-plan  --input snapshot.json   # Robinhood 주문 계획 (실행은 Claude 가 MCP 로)
  python crypto_bot/run.py rh-record --input fills.json     # Robinhood 체결 결과를 장부에 기록
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from bot import data as data_mod  # noqa: E402
from bot import execution as ex  # noqa: E402
from bot import learn  # noqa: E402

CONFIG_PATH = ROOT / "config.json"
PAPER_LEDGER = learn.STATE_DIR / "paper_ledger.json"
RH_LEDGER = learn.STATE_DIR / "robinhood_ledger.json"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def load_data(cfg: dict) -> dict:
    return {s: data_mod.update(s, cfg["timeframe"], cfg["history_start"]) for s in cfg["symbols"]}


def cmd_learn(cfg):
    active = learn.run_learning(cfg, load_data(cfg))
    for sym, p in active["symbols"].items():
        m = p.get("oos_metrics") or {}
        print(f"{sym}: {p['name']}  score={p['score']:.2f}  CAGR={m.get('cagr', 0):+.1%}  MDD={m.get('max_drawdown', 0):.1%}")
    po, bh = active["portfolio_oos"], active["portfolio_buy_and_hold"]
    print(f"PORTFOLIO OOS: CAGR={po['cagr']:+.1%} MDD={po['max_drawdown']:.1%} | buy&hold CAGR={bh['cagr']:+.1%} MDD={bh['max_drawdown']:.1%}")


def cmd_signal(cfg):
    sig = ex.compute_signal(cfg, load_data(cfg))
    print(json.dumps(sig, indent=2))
    return sig


def cmd_paper(cfg):
    sig = cmd_signal(cfg)
    fee = cfg["costs"]["fee_rate"] + cfg["costs"]["slippage_rate"]
    ledger = ex.load_ledger(PAPER_LEDGER, cfg["paper"]["starting_cash"])
    ledger.setdefault("holdings", {})
    if ledger["history"] and ledger["history"][-1].get("bar_time") == next(iter(sig["symbols"].values()))["bar_time"]:
        print("이미 이 캔들로 모의매매 완료 — 건너뜀")
        return
    prices = {s: v["close"] for s, v in sig["symbols"].items()}
    plan = ex.plan_orders(sig, ledger, ledger["holdings"], prices, cfg)
    fills = []
    for o in plan["orders"]:
        s, px = o["symbol"], prices[o["symbol"]]
        if o["side"] == "sell":
            qty = o["quantity"] if "quantity" in o else o["dollar_amount"] / px
            qty = min(qty, ledger["holdings"].get(s, 0.0))
            usd = qty * px * (1 - fee)
            ledger["holdings"][s] = ledger["holdings"].get(s, 0.0) - qty
        else:
            usd = o["dollar_amount"]
            qty = usd * (1 - fee) / px
            ledger["holdings"][s] = ledger["holdings"].get(s, 0.0) + qty
        fills.append({"symbol": s, "side": o["side"], "usd": round(usd, 2), "qty": qty, "price": px})
    value = ledger["cash"] + sum(f["usd"] if f["side"] == "sell" else -f["usd"] for f in fills) + \
        sum(q * prices[s] for s, q in ledger["holdings"].items())
    ex.record_fills(ledger, fills, round(value, 2))
    ledger["history"][-1]["bar_time"] = next(iter(sig["symbols"].values()))["bar_time"]
    ex.save_ledger(PAPER_LEDGER, ledger)
    print(json.dumps({"plan": plan, "fills": fills, "value_after": round(value, 2)}, indent=2, default=float))


def cmd_rh_plan(cfg, input_path):
    """snapshot = {"holdings": {"BTC-USD": qty}, "prices": {"BTC-USD": px}, "buying_power": 10.0}"""
    snap = json.loads(Path(input_path).read_text())
    sig = json.loads(ex.SIGNAL_PATH.read_text())
    ledger = ex.load_ledger(RH_LEDGER, cfg["robinhood"]["budget_usd"])
    plan = ex.plan_orders(sig, ledger, snap["holdings"], snap["prices"], cfg, snap.get("buying_power"))
    ex.save_ledger(RH_LEDGER, ledger)  # 킬스위치 상태 저장
    plan["signal_generated_at"] = sig["generated_at"]
    print(json.dumps(plan, indent=2, default=float))


def cmd_rh_record(cfg, input_path):
    """fills = {"fills": [{"symbol","side","usd","qty","price","order_id"}], "value_after": 201.3}"""
    payload = json.loads(Path(input_path).read_text())
    ledger = ex.load_ledger(RH_LEDGER, cfg["robinhood"]["budget_usd"])
    ex.record_fills(ledger, payload["fills"], payload.get("value_after"))
    ex.save_ledger(RH_LEDGER, ledger)
    print(json.dumps({"cash": ledger["cash"], "peak_value": ledger["peak_value"], "halted": ledger["halted"]}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["learn", "signal", "paper", "rh-plan", "rh-record"])
    ap.add_argument("--input")
    args = ap.parse_args()
    cfg = load_config()
    if args.command == "learn":
        cmd_learn(cfg)
    elif args.command == "signal":
        cmd_signal(cfg)
    elif args.command == "paper":
        cmd_paper(cfg)
    elif args.command == "rh-plan":
        cmd_rh_plan(cfg, args.input)
    elif args.command == "rh-record":
        cmd_rh_record(cfg, args.input)


if __name__ == "__main__":
    main()
