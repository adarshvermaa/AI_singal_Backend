import time
import logging
import uuid
from typing import Optional
from fastapi import APIRouter, Request, Header, HTTPException
from app.config import get_settings
from app.api.schemas import (
    WebhookTradePayload, WebhookLogItem, WebhookConfigResponse,
    TradingModeEnum,
)
from app.exchanges.base import OrderSide, OrderType, ExchangeError
from app.core.indicator_engine import IndicatorEngine
from app.core.signal_generator import RuleBasedSignalGenerator
from app.core.risk_manager import RiskManager
from app.core.order_executor import OrderExecutor
from app.api.routes.ws import manager as ws_manager

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory circular log buffer (no database storage)
_WEBHOOK_LOGS: list[WebhookLogItem] = []
_MAX_LOGS = 50


def _add_log(item: WebhookLogItem):
    _WEBHOOK_LOGS.insert(0, item)
    if len(_WEBHOOK_LOGS) > _MAX_LOGS:
        _WEBHOOK_LOGS.pop()


@router.post("/trade")
async def receive_webhook_trade(
    request: Request,
    payload: WebhookTradePayload,
    x_webhook_secret: Optional[str] = Header(None),
):
    """
    Primary Inbound Webhook Endpoint for TradingView Alerts and CoinDCX triggers.
    
    Accepts JSON alert payload:
    - Validates security token
    - Optionally passes through AI Smart Sentinel (Ensemble check)
    - Automatically executes on CoinDCX (Spot, Margin, or Futures)
    - Broadcasts live event to frontend chart via WebSocket
    """
    settings = get_settings()
    query_secret = request.query_params.get("secret")
    provided_secret = x_webhook_secret or payload.secret or query_secret

    # Extract user's CoinDCX token and webhook ID if configured
    coindcx_token = "jSsqRkGH5aQDvuD01JLoV1jDo"
    if getattr(settings, "coindcx_webhook_url", None):
        coindcx_token = settings.coindcx_webhook_url.rstrip("/").split("/")[-1]

    coindcx_webhook_id = getattr(settings, "coindcx_webhook_id", "d536c102-6568-4f5e-8297-5ef23f194ace")

    valid_secrets = {
        settings.webhook_secret_key,
        coindcx_token,
        "jSsqRkGH5aQDvuD01JLoV1jDo",
        coindcx_webhook_id,
        "d536c102-6568-4f5e-8297-5ef23f194ace",
    }

    # 1. Security Check — HMAC signature verification for CoinDCX direct webhooks
    hmac_signature = request.headers.get("x-webhook-signature") or request.headers.get("x-coindcx-signature")
    if hmac_signature:
        import hmac as hmac_mod
        import hashlib
        raw_body = await request.body()
        secret_bytes = settings.webhook_secret_key.encode('utf-8')
        expected_sig = hmac_mod.new(secret_bytes, raw_body, hashlib.sha256).hexdigest()
        if not hmac_mod.compare_digest(expected_sig, hmac_signature):
            log = WebhookLogItem(
                id=str(uuid.uuid4())[:8],
                timestamp=int(time.time() * 1000),
                symbol=payload.symbol or "UNKNOWN",
                action=payload.action or "unknown",
                trading_mode=payload.trading_mode.value,
                status="invalid_hmac",
                message="Unauthorized: HMAC signature verification failed.",
            )
            _add_log(log)
            raise HTTPException(status_code=401, detail="Invalid HMAC signature")
    elif not provided_secret or provided_secret not in valid_secrets:
        log = WebhookLogItem(
            id=str(uuid.uuid4())[:8],
            timestamp=int(time.time() * 1000),
            symbol=payload.symbol or "UNKNOWN",
            action=payload.action or "unknown",
            trading_mode=payload.trading_mode.value,
            status="invalid_secret",
            message="Unauthorized: Webhook secret does not match configured key or CoinDCX token.",
        )
        _add_log(log)
        raise HTTPException(status_code=401, detail="Invalid webhook secret token")

    factory = request.app.state.exchange_factory
    try:
        exch = factory.get_or_raise("coindcx")
    except Exception as e:
        log = WebhookLogItem(
            id=str(uuid.uuid4())[:8],
            timestamp=int(time.time() * 1000),
            symbol=payload.symbol,
            action=payload.action,
            trading_mode=payload.trading_mode.value,
            status="failed",
            message=f"Exchange connection error: {str(e)}",
        )
        _add_log(log)
        raise HTTPException(status_code=500, detail=str(e))

    action_lower = payload.action.lower()

    # 2. Handle Position Close
    if action_lower == "close":
        try:
            positions = await exch.get_futures_positions()
            target_pos = next((p for p in positions if p.pair == payload.symbol), None)
            if not target_pos:
                msg = f"No active futures position found for {payload.symbol} to close."
                log = WebhookLogItem(
                    id=str(uuid.uuid4())[:8],
                    timestamp=int(time.time() * 1000),
                    symbol=payload.symbol,
                    action=payload.action,
                    trading_mode=payload.trading_mode.value,
                    status="failed",
                    message=msg,
                )
                _add_log(log)
                return {"status": "failed", "message": msg}

            success = await exch.exit_futures_position(target_pos.id)
            status_str = "executed" if success else "failed"
            log = WebhookLogItem(
                id=str(uuid.uuid4())[:8],
                timestamp=int(time.time() * 1000),
                symbol=payload.symbol,
                action="close",
                trading_mode=payload.trading_mode.value,
                status=status_str,
                order_id=target_pos.id,
                message=f"Position {target_pos.id} closed on CoinDCX",
            )
            _add_log(log)

            await ws_manager.broadcast_all({
                "type": "webhook_event",
                "data": log.dict(),
            })

            return {"status": status_str, "position_id": target_pos.id}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Close position failed: {str(e)}")

    # 3. Optional: AI Smart Sentinel Filter
    should_verify_ai = (
        payload.require_ai_confirmation
        if payload.require_ai_confirmation is not None
        else settings.webhook_ai_filter_enabled
    )

    ai_verdict = None
    ai_conf = None

    if should_verify_ai and action_lower in ("buy", "sell"):
        try:
            # Fetch recent candles for fast AI check
            candles = await exch.get_futures_candles(payload.symbol, "15m", limit=150)
            if candles and len(candles) >= 30:
                engine = IndicatorEngine()
                df = engine.candles_to_dataframe(candles)
                df = engine.compute_all(df)
                
                sig_gen = RuleBasedSignalGenerator()
                sig_result = sig_gen.generate(df)
                
                ai_verdict = sig_result.direction.value
                ai_conf = sig_result.confidence

                expected_direction = "BUY" if action_lower == "buy" else "SELL"

                # Check if AI agrees
                if ai_verdict != expected_direction or ai_conf < settings.webhook_min_confidence:
                    rejection_msg = (
                        f"AI Smart Sentinel protected capital: Alert signaled {expected_direction}, "
                        f"but AI model evaluated {ai_verdict} ({(ai_conf * 100):.1f}% confidence)."
                    )
                    log = WebhookLogItem(
                        id=str(uuid.uuid4())[:8],
                        timestamp=int(time.time() * 1000),
                        symbol=payload.symbol,
                        action=payload.action,
                        trading_mode=payload.trading_mode.value,
                        status="rejected_by_ai",
                        ai_verdict=ai_verdict,
                        ai_confidence=ai_conf,
                        message=rejection_msg,
                    )
                    _add_log(log)

                    await ws_manager.broadcast_all({
                        "type": "webhook_event",
                        "data": log.dict(),
                    })

                    return {
                        "status": "rejected_by_ai",
                        "message": rejection_msg,
                        "ai_verdict": ai_verdict,
                        "ai_confidence": ai_conf,
                    }
        except Exception as e:
            logger.warning(f"AI Sentinel check skipped due to error: {e}")

    # 4. Execute Trade via OrderExecutor
    try:
        risk_mgr = RiskManager()

        # Determine price & sizing if not explicitly supplied
        quantity = payload.quantity
        calc_price = payload.price
        if not calc_price:
            try:
                ticker = await exch.get_ticker(payload.symbol)
                calc_price = ticker.last_price
            except Exception:
                calc_price = 0.0

        if (not quantity or quantity <= 0) and payload.auto_risk:
            try:
                balances = await exch.get_balances()
                usdt_bal = next((b for b in balances if b.currency.upper() in ('USDT', 'INR', 'USD')), None)
                avail = usdt_bal.available if usdt_bal else 100.0
            except Exception:
                avail = 100.0

            if payload.stop_loss and calc_price and calc_price > 0:
                quantity = risk_mgr.calculate_position_size(
                    balance=avail,
                    entry_price=calc_price,
                    stop_loss=payload.stop_loss,
                    max_risk_pct=0.02,
                    leverage=payload.leverage or 3.0,
                )
            elif calc_price and calc_price > 0:
                quantity = (avail * 0.05) / calc_price
            else:
                quantity = 0.001

        if not quantity or quantity <= 0:
            quantity = 0.001

        quantity = round(quantity, 4)

        executor = OrderExecutor(exch, risk_mgr)
        order_res = await executor.execute_trade(
            pair=payload.symbol,
            side=action_lower,
            quantity=quantity,
            price=payload.price,
            leverage=payload.leverage or 3.0,
            stop_loss=payload.stop_loss,
            take_profit=payload.take_profit,
            trading_mode=payload.trading_mode.value,
        )

        status_str = "executed" if order_res and getattr(order_res, "status", None) in ("filled", "placed", "executed", "open") else "failed"
        order_id = getattr(order_res, "order_id", None) if order_res else None

        log = WebhookLogItem(
            id=str(uuid.uuid4())[:8],
            timestamp=int(time.time() * 1000),
            symbol=payload.symbol,
            action=payload.action,
            trading_mode=payload.trading_mode.value,
            status=status_str,
            ai_verdict=ai_verdict,
            ai_confidence=ai_conf,
            order_id=order_id,
            message=f"Order {status_str} on CoinDCX: {payload.symbol} {action_lower.upper()} qty={quantity}",
        )
        _add_log(log)

        # Broadcast live event to frontend chart
        await ws_manager.broadcast_all({
            "type": "webhook_event",
            "data": log.dict(),
        })

        return {
            "status": status_str,
            "order_id": order_id,
            "symbol": payload.symbol,
            "side": action_lower,
            "quantity": quantity,
            "price": payload.price or calc_price,
            "trading_mode": payload.trading_mode.value,
            "message": log.message,
        }

    except ExchangeError as e:
        log = WebhookLogItem(
            id=str(uuid.uuid4())[:8],
            timestamp=int(time.time() * 1000),
            symbol=payload.symbol,
            action=payload.action,
            trading_mode=payload.trading_mode.value,
            status="failed",
            message=f"Exchange execution error: {e.message}",
        )
        _add_log(log)
        return {"status": "error", "message": e.message}
    except Exception as e:
        log = WebhookLogItem(
            id=str(uuid.uuid4())[:8],
            timestamp=int(time.time() * 1000),
            symbol=payload.symbol,
            action=payload.action,
            trading_mode=payload.trading_mode.value,
            status="failed",
            message=f"Execution error: {str(e)}",
        )
        _add_log(log)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs", response_model=list[WebhookLogItem])
async def get_webhook_logs():
    """Retrieve recent in-memory webhook execution logs."""
    return _WEBHOOK_LOGS


@router.get("/config", response_model=WebhookConfigResponse)
async def get_webhook_config(request: Request):
    """Get webhook URL, authorization secret, and pre-formatted TradingView alert templates."""
    settings = get_settings()
    host = request.headers.get("host", "localhost:8000")
    forwarded_proto = request.headers.get("x-forwarded-proto")
    scheme = forwarded_proto if forwarded_proto else ("https" if request.url.is_secure else "http")
    webhook_url = f"{scheme}://{host}/api/webhook/trade"

    coindcx_id = getattr(settings, "coindcx_webhook_id", "d536c102-6568-4f5e-8297-5ef23f194ace")
    templates = {
        "coindcx_official_dynamic": {
            "side": "{{strategy.order.action}}",
            "price": "{{close}}",
            "symbol": "{{ticker}}",
            "order_type": "market_order",
            "total_quantity": "0.001",
            "margin_currency_short_name": "USDT",
            "webhook_id": coindcx_id,
        },
        "coindcx_official_static_buy": {
            "side": "buy",
            "price": "67000",
            "symbol": "B-BTC_USDT",
            "order_type": "market_order",
            "total_quantity": "0.001",
            "margin_currency_short_name": "USDT",
            "webhook_id": coindcx_id,
        },
        "coindcx_official_static_sell": {
            "side": "sell",
            "price": "67000",
            "symbol": "B-BTC_USDT",
            "order_type": "market_order",
            "total_quantity": "0.001",
            "margin_currency_short_name": "USDT",
            "webhook_id": coindcx_id,
        },
        "coindcx_native_futures_buy": {
            "pair": "B-BTC_USDT",
            "side": "buy",
            "order_type": "market_order",
            "total_quantity": 0.001,
            "leverage": 3,
        },
        "coindcx_native_futures_sell": {
            "pair": "B-BTC_USDT",
            "side": "sell",
            "order_type": "market_order",
            "total_quantity": 0.001,
            "leverage": 3,
        },
        "futures_long_3x": {
            "secret": settings.webhook_secret_key,
            "symbol": "B-BTC_USDT",
            "action": "buy",
            "trading_mode": "futures",
            "leverage": 3.0,
            "auto_risk": True,
            "stop_loss": 67800,
            "take_profit": 69500,
            "require_ai_confirmation": True,
        },
        "futures_short_3x": {
            "secret": settings.webhook_secret_key,
            "symbol": "B-BTC_USDT",
            "action": "sell",
            "trading_mode": "futures",
            "leverage": 3.0,
            "auto_risk": True,
            "stop_loss": 69200,
            "take_profit": 67400,
            "require_ai_confirmation": True,
        },
        "close_position": {
            "secret": settings.webhook_secret_key,
            "symbol": "B-BTC_USDT",
            "action": "close",
            "trading_mode": "futures",
        },
        "spot_market_buy": {
            "secret": settings.webhook_secret_key,
            "symbol": "B-BTC_USDT",
            "action": "buy",
            "trading_mode": "spot",
            "order_type": "market_order",
            "auto_risk": True,
        },
    }

    return WebhookConfigResponse(
        webhook_url=webhook_url,
        coindcx_webhook_url=getattr(settings, "coindcx_webhook_url", None),
        secret_key=settings.webhook_secret_key,
        ai_filter_enabled=settings.webhook_ai_filter_enabled,
        templates=templates,
    )
