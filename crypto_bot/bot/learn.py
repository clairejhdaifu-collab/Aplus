"""워크포워드 학습: '덜 잃는' 전략을 고르고, 주기적으로 다시 고른다.

방식 (과최적화 방지):
  과거 2년(train)으로 각 전략의 최적 파라미터를 고르고 → 바로 다음 3개월(test)에 적용.
  이 창을 3개월씩 밀면서 끝까지 반복하면, 모든 test 구간은 "고를 때 못 본 데이터"가 된다.
  이렇게 이어 붙인 test 성과(OOS: out-of-sample)로 전략들을 비교해 챔피언을 정한다.
  후보: 개별 전략 11종 + 변동성조절 보유 + 상위 K개 앙상블(평균). 기준선: 단순 보유(buy & hold).
  챔피언 점수가 0 이하면 '현금 보유'가 정답 — 잃지 않는 것이 목표이므로.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest as bt
from . import strategies as st

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "state"
REPORT_DIR = ROOT / "reports"
ACTIVE_PATH = STATE_DIR / "active_strategy.json"

VOL_HOLD = "vol_target_hold"  # 항상 보유하되 변동성 타깃·트레일링 스톱만 적용


def build_candidates(df: pd.DataFrame, risk: bt.RiskParams) -> dict[str, dict[str, tuple[dict, pd.Series]]]:
    """{전략명: {파라미터키: (파라미터, 보유비중 시계열)}}"""
    out: dict[str, dict[str, tuple[dict, pd.Series]]] = {}
    for name, (_, grid) in st.STRATEGIES.items():
        out[name] = {}
        for params in grid:
            exp = bt.apply_risk(df, st.signal(df, name, params), risk)
            out[name][st.param_key(params)] = (params, exp)
    out[VOL_HOLD] = {"": ({}, bt.apply_risk(df, st.buy_and_hold(df), risk))}
    return out


def _fold_starts(n: int, first_valid: int, train: int, test: int) -> list[int]:
    return list(range(first_valid + train, n, test))


def walk_forward(df: pd.DataFrame, cands, cost_rate: float, lcfg: dict) -> dict:
    train, test = lcfg["train_days"], lcfg["test_days"]
    k = lcfg.get("ensemble_top_k", 3)
    obj = lcfg.get("objective", "martin")
    dd_lim = lcfg.get("max_drawdown_limit", 0.35)
    mtpy = lcfg.get("min_trades_per_year", 0)

    # 각 후보의 전체기간 일수익률 (비중이 인과적이므로 슬라이스해도 미래정보 없음)
    rets = {f: {p: bt.run(df, e, cost_rate) for p, (_, e) in grid.items()} for f, grid in cands.items()}

    first_valid = 200  # 장기 지표(200일선 등) 준비 기간
    starts = _fold_starts(len(df), first_valid, train, test)
    if not starts:
        raise ValueError("데이터가 너무 짧아 워크포워드를 할 수 없습니다")

    fam_parts: dict[str, list[pd.Series]] = {f: [] for f in cands}
    ens_parts: list[pd.Series] = []
    picks: list[dict] = []

    def train_score(f, p, lo, hi):
        sl = slice(lo, hi)
        return bt.score(bt.metrics(rets[f][p].iloc[sl], cands[f][p][1].iloc[sl]), obj, dd_lim, mtpy)

    for s in starts:
        lo, hi = s - train, min(s + test, len(df))
        best_by_fam = {}
        for f, grid in cands.items():
            scored = [(train_score(f, p, lo, s), p) for p in grid]
            sc, p = max(scored, key=lambda x: x[0])
            best_by_fam[f] = (sc, p)
            fam_parts[f].append(cands[f][p][1].iloc[s:hi])
        top = sorted(best_by_fam.items(), key=lambda kv: kv[1][0], reverse=True)[:k]
        ens_parts.append(pd.concat([cands[f][p][1].iloc[s:hi] for f, (_, p) in top], axis=1).mean(axis=1))
        picks.append({"test_start": str(df.index[s].date()), "ensemble": [f for f, _ in top]})

    oos_index = df.index[starts[0]:]
    results = {}
    for f, parts in fam_parts.items():
        exp = pd.concat(parts)
        r = bt.run(df.loc[oos_index], exp, cost_rate)
        m = bt.metrics(r, exp)
        results[f] = {"metrics": m, "score": bt.score(m, obj, dd_lim, mtpy), "returns": r}
    ens = pd.concat(ens_parts)
    r = bt.run(df.loc[oos_index], ens, cost_rate)
    m = bt.metrics(r, ens)
    results["ensemble"] = {"metrics": m, "score": bt.score(m, obj, dd_lim, mtpy), "returns": r}

    hold = pd.Series(1.0, index=oos_index)
    r = bt.run(df.loc[oos_index], hold, cost_rate)
    m = bt.metrics(r, hold)
    results[st.BASELINE] = {"metrics": m, "score": bt.score(m, obj, dd_lim, mtpy), "returns": r}

    # 최신 2년으로 '지금 쓸' 파라미터 확정
    lo = max(0, len(df) - train)
    latest = {}
    for f, grid in cands.items():
        scored = [(train_score(f, p, lo, len(df)), p) for p in grid]
        sc, p = max(scored, key=lambda x: x[0])
        latest[f] = {"params": cands[f][p][0], "train_score": sc}

    return {"results": results, "latest": latest, "picks": picks,
            "oos_start": str(oos_index[0].date()), "oos_end": str(oos_index[-1].date())}


def choose(wf: dict, previous: dict | None, lcfg: dict) -> dict:
    """챔피언 선정 (+ 잦은 교체를 막는 여유폭)."""
    k = lcfg.get("ensemble_top_k", 3)
    margin = lcfg.get("switch_margin", 0.10)
    res = {f: v for f, v in wf["results"].items() if f != st.BASELINE}
    best = max(res, key=lambda f: res[f]["score"])

    prev_name = (previous or {}).get("name")
    kept = False
    if prev_name in res and prev_name != best:
        prev_score = res[prev_name]["score"]
        if prev_score > 0 and res[best]["score"] < prev_score * (1 + margin):
            best, kept = prev_name, True

    if res[best]["score"] <= 0:
        return {"name": "cash", "kind": "cash", "members": [], "score": 0.0, "kept_previous": False,
                "reason": "모든 전략의 검증 점수가 0 이하 → 현금 보유가 가장 덜 잃음"}

    if best == "ensemble":
        ranked = sorted(wf["latest"].items(), key=lambda kv: kv[1]["train_score"], reverse=True)[:k]
        members = [{"strategy": f, "params": v["params"]} for f, v in ranked]
        kind = "ensemble"
    else:
        members = [{"strategy": best, "params": wf["latest"][best]["params"]}]
        kind = "single"
    return {"name": best, "kind": kind, "members": members, "score": res[best]["score"],
            "kept_previous": kept, "reason": "워크포워드 검증(OOS) 점수 1위" + (" (기존 전략 유지: 차이가 여유폭 이내)" if kept else "")}


def member_exposure(df: pd.DataFrame, members: list[dict], risk: bt.RiskParams) -> pd.Series:
    if not members:
        return pd.Series(0.0, index=df.index)
    exps = []
    for m in members:
        if m["strategy"] == VOL_HOLD:
            sig = st.buy_and_hold(df)
        else:
            sig = st.signal(df, m["strategy"], m["params"])
        exps.append(bt.apply_risk(df, sig, risk))
    return pd.concat(exps, axis=1).mean(axis=1)


def load_active() -> dict:
    if ACTIVE_PATH.exists():
        return json.loads(ACTIVE_PATH.read_text())
    return {}


def _pct(x: float) -> str:
    return f"{x * 100:+.1f}%" if x == x else "n/a"


def write_report(all_wf: dict[str, dict], active: dict, cfg: dict) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"# 학습 리포트 ({now})", "",
             f"- 목표: **덜 잃기** (점수 = 연수익률 ÷ 얼서지수, 최대낙폭 한도 {cfg['learning']['max_drawdown_limit']:.0%})",
             f"- 검증: 워크포워드 (학습 {cfg['learning']['train_days']}일 → 다음 {cfg['learning']['test_days']}일 적용, 반복)",
             f"- 비용 가정: 매매 1회당 수수료 {cfg['costs']['fee_rate']:.2%} + 슬리피지 {cfg['costs']['slippage_rate']:.2%}",
             ""]
    for sym, wf in all_wf.items():
        a = active[sym]
        lines += [f"## {sym}  (검증 구간 {wf['oos_start']} ~ {wf['oos_end']})", "",
                  f"**선택: `{a['name']}`** — {a['reason']}", ""]
        if a["members"]:
            for m in a["members"]:
                lines.append(f"- {m['strategy']} {st.param_key(m['params'])}")
            lines.append("")
        lines += ["| 순위 | 전략 | 점수 | 연수익률 | 최대낙폭 | 최악의 달 | 보유시간 | 진입횟수 |",
                  "|---|---|---|---|---|---|---|---|"]
        ranked = sorted(wf["results"].items(), key=lambda kv: kv[1]["score"], reverse=True)
        for i, (f, v) in enumerate(ranked, 1):
            m = v["metrics"]
            tag = " (기준선)" if f == st.BASELINE else ""
            lines.append(f"| {i} | {f}{tag} | {v['score']:.2f} | {_pct(m['cagr'])} | {_pct(-m['max_drawdown'])} | "
                         f"{_pct(m['worst_month'])} | {m.get('time_in_market', 1):.0%} | {m.get('trades', '-')} |")
        lines.append("")
    lines += ["> 과거 성과는 미래 수익을 보장하지 않습니다. 검증 구간 성과도 전략 '선택' 과정에서 약간 낙관적으로 나올 수 있습니다.", ""]
    return "\n".join(lines)


def run_learning(cfg: dict, data: dict[str, pd.DataFrame]) -> dict:
    risk = bt.RiskParams.from_config(cfg)
    cost = cfg["costs"]["fee_rate"] + cfg["costs"]["slippage_rate"]
    previous = load_active().get("symbols", {})
    all_wf, chosen = {}, {}
    for sym, df in data.items():
        cands = build_candidates(df, risk)
        wf = walk_forward(df, cands, cost, cfg["learning"])
        pick = choose(wf, previous.get(sym), cfg["learning"])
        pick["oos_metrics"] = wf["results"].get(pick["name"], {}).get("metrics", {})
        pick["baseline_metrics"] = wf["results"][st.BASELINE]["metrics"]
        all_wf[sym], chosen[sym] = wf, pick

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    active = {"updated_at": now, "symbols": chosen}
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ACTIVE_PATH.write_text(json.dumps(active, indent=2, ensure_ascii=False, default=float))
    (REPORT_DIR / "learn_report.md").write_text(write_report(all_wf, chosen, cfg))

    hist = REPORT_DIR / "selection_history.csv"
    new = not hist.exists()
    with hist.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["time", "symbol", "chosen", "score", "oos_cagr", "oos_max_drawdown", "kept_previous"])
        for sym, p in chosen.items():
            m = p.get("oos_metrics") or {}
            w.writerow([now, sym, p["name"], round(p["score"], 4), round(m.get("cagr", 0), 4),
                        round(m.get("max_drawdown", 0), 4), p["kept_previous"]])

    # 포트폴리오(종목 균등분할) OOS 곡선 요약
    port = pd.concat([all_wf[s]["results"][chosen[s]["name"]]["returns"] if chosen[s]["name"] != "cash"
                      else pd.Series(0.0, index=all_wf[s]["results"][st.BASELINE]["returns"].index)
                      for s in chosen], axis=1).fillna(0).mean(axis=1)
    hold = pd.concat([all_wf[s]["results"][st.BASELINE]["returns"] for s in chosen], axis=1).fillna(0).mean(axis=1)
    active["portfolio_oos"] = bt.metrics(port)
    active["portfolio_buy_and_hold"] = bt.metrics(hold)
    po, bh = active["portfolio_oos"], active["portfolio_buy_and_hold"]
    summary = ["## 포트폴리오 요약 (종목 균등분할, 검증구간)", "",
               "| | 연수익률 | 최대낙폭 | 최악의 달 | 샤프 |", "|---|---|---|---|---|",
               f"| 이 시스템 | {_pct(po['cagr'])} | {_pct(-po['max_drawdown'])} | {_pct(po['worst_month'])} | {po['sharpe']:.2f} |",
               f"| 그냥 보유 | {_pct(bh['cagr'])} | {_pct(-bh['max_drawdown'])} | {_pct(bh['worst_month'])} | {bh['sharpe']:.2f} |", ""]
    report = REPORT_DIR / "learn_report.md"
    body = report.read_text().split("\n")
    report.write_text("\n".join(body[:6] + summary + body[6:]))
    ACTIVE_PATH.write_text(json.dumps(active, indent=2, ensure_ascii=False, default=float))
    return active
