# Robinhood 일일 자동매매 절차 (Claude 루틴용)

매일 01:00 UTC(캘리포니아 오후 6시 / 겨울 오후 5시)에 Claude 루틴이 이 절차를 그대로 수행한다.
**매매 판단은 코드(`run.py rh-plan`)가 한다. Claude 는 데이터를 옮기고 나온 주문만 실행한다.**

## 절대 규칙
- 계좌: Robinhood 의 AI 거래 허용 계좌(별명 `MCP`) 하나만 쓴다. 다른 계좌는 건드리지 않는다.
- 종목: `config.json` 의 `robinhood.symbols_allowed` (BTC-USD, ETH-USD) 만.
- 주문: `rh-plan` 출력의 `orders` 그대로만. 즉흥 주문·추가 주문·옵션·주식 금지.
- 예산: 봇이 굴리는 돈은 `robinhood.budget_usd`($200)와 그 손익뿐. 계좌의 다른 현금·주식은 손대지 않는다.
- 신호가 오늘 것이 아니거나, 무엇이든 예상과 다르면 → **매매하지 않고** 사용자에게 알린다.
- 킬스위치(고점 대비 -25%)가 걸리면 전량 매도 후 정지. 사용자가 직접 재개해야 한다.

## 단계
1. `git pull origin claude/stock-trading-automation-system-6spbe2`
2. `crypto_bot/state/signal.json` 의 `generated_at` 이 오늘(UTC)인지 확인.
   - 아니면 `crypto_bot/state/tick.txt` 에 현재시각을 쓰고 커밋·푸시 → GitHub Actions 가 신호를 만든다.
     최대 20분 기다렸다가 다시 pull. 그래도 없으면 매매 없이 종료하고 알린다.
3. Robinhood 조회 (MCP):
   - `get_accounts` → 거래 가능 계좌의 `rhs_account_number`
   - `get_portfolio` → `crypto_buying_power.buying_power`
   - `get_crypto_positions` → BTC, ETH 의 `quantity_transferable`
   - `get_crypto_quotes(["BTC-USD","ETH-USD"])` → `mark_price`
4. `crypto_bot/snapshot.json` 작성 후 `python crypto_bot/run.py rh-plan --input crypto_bot/snapshot.json`
   ```json
   {"holdings": {"BTC-USD": 0.0, "ETH-USD": 0.0},
    "prices": {"BTC-USD": 0.0, "ETH-USD": 0.0},
    "buying_power": 0.0}
   ```
5. `orders` 가 비어 있으면 매매 없음. 있으면 매도 먼저, 매수 나중에, 각각:
   `preview_crypto_order` → 문제 없으면 `place_crypto_order` (type `market`, 새 UUID `ref_id`).
   - `quantity` 가 있으면 수량 주문(전량 매도), 아니면 `dollar_amount` 주문.
6. `get_crypto_orders` 로 체결 확인 → `crypto_bot/fills.json` 작성 후
   `python crypto_bot/run.py rh-record --input crypto_bot/fills.json`
   ```json
   {"fills": [{"symbol": "BTC-USD", "side": "buy", "usd": 98.0, "qty": 0.00098, "price": 100000, "order_id": "..."}],
    "value_after": 200.0}
   ```
   `value_after` = 장부 현금 + 보유코인 평가액.
7. `crypto_bot/reports/robinhood_trades.md` 에 한 줄 기록 → 커밋·푸시.
8. 사용자에게 한두 줄 요약 (매매 내역, 봇 자산 평가액, 고점 대비 낙폭).
