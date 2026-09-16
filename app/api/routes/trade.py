import logging
from fastapi import APIRouter, Request, HTTPException
from app.api.schemas import (
    ExecuteTradeRequest, ExecuteTradeResponse,
    ClosePositionRequest, TradingModeEnum,
)
from app.exchanges.base import OrderSide, OrderType, ExchangeError
from app.core.currency import currency_converter

logger = logging.getLogger(__name__)
router = APIRouter()


def get_symbol_baseline_price(symbol: str) -> float:
    """Return fallback typical USD price for a cryptocurrency symbol."""
    upper = symbol.upper()
    if 'BTC' in upper:
        return 68500.0
    elif 'ETH' in upper:
        return 3520.0
    elif 'SOL' in upper:
        return 184.0
    elif 'DOGE' in upper:
        return 0.1284
    elif 'XRP' in upper:
        return 0.584
    elif any(k in upper for k in ('PEPE', 'SHIB', 'BONK', 'FLOKI')):
        return 0.0000185
    return 100.0


def format_order_quantity(symbol: str, raw_qty: float) -> float:
    """
    Format quantity to conform to exchange lot sizes and step precision.
    Prevents 'Invalid quantity' exchange rejection on CoinDCX / exchanges.
    """
    upper = symbol.upper()

    if 'BTC' in upper:
        min_qty, step, dec = 0.001, 0.001, 3
    elif 'ETH' in upper:
        min_qty, step, dec = 0.01, 0.01, 2
    elif 'SOL' in upper or 'BNB' in upper:
        min_qty, step, dec = 0.1, 0.1, 1
    elif any(k in upper for k in ('DOGE', 'XRP', 'ADA', 'TRX', 'MATIC', 'POL', 'LTC')):
        min_qty, step, dec = 1.0, 1.0, 0
    elif any(k in upper for k in ('PEPE', 'SHIB', 'BONK', 'FLOKI')):
        min_qty, step, dec = 10000.0, 1000.0, 0
    else:
        min_qty, step, dec = 0.1, 0.01, 2

    if not raw_qty or raw_qty <= 0:
        return min_qty

    steps_count = round(raw_qty / step)
    quantized = steps_count * step
    final_qty = max(min_qty, round(quantized, dec))
    if dec == 0:
        return float(int(final_qty))
    return round(final_qty, dec)


@router.post("/execute", response_model=ExecuteTradeResponse)
async def execute_trade(
    request: Request,
    body: ExecuteTradeRequest,
):
    """
    Execute a trade on the connected exchange.
    Supports spot, margin, and futures orders with dual currency (USD & INR) capital conversion.
    """
    factory = request.app.state.exchange_factory
    try:
        exch = factory.get_or_raise(body.exchange)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    try:
        side = OrderSide.BUY if body.side.lower() == 'buy' else OrderSide.SELL
        order_type = OrderType(body.order_type)

        # 1. Determine Quote Currency (USDT vs INR) and live conversion rate
        upper_sym = body.symbol.upper()
        is_inr_pair = upper_sym.endswith('INR') or '_INR' in upper_sym
        quote_currency = 'INR' if is_inr_pair else 'USDT'

        usd_inr_rate = await currency_converter.get_usd_inr_rate()
        if not usd_inr_rate or usd_inr_rate <= 0:
            usd_inr_rate = 99.95

        # 2. Position sizing & quantity formatting
        quantity = body.quantity
        if body.auto_size or quantity is None or quantity <= 0:
            balances = []
            try:
                balances = await exch.get_balances()
            except Exception as be:
                logger.warning(f"Could not fetch balances for trade sizing ({be}), using paper capital")

            # Calculate total available capital converted into the pair's quote currency
            total_available_quote = 0.0
            for b in balances:
                if b.available and b.available > 0:
                    curr = b.currency.upper()
                    if curr in ('USDT', 'USD'):
                        if quote_currency == 'USDT':
                            total_available_quote += b.available
                        else:  # quote is INR
                            total_available_quote += b.available * usd_inr_rate
                    elif curr == 'INR':
                        if quote_currency == 'INR':
                            total_available_quote += b.available
                        else:  # quote is USDT
                            total_available_quote += b.available / usd_inr_rate

            # If user has zero live balance or in paper/demo mode, ensure safe trading capital
            if total_available_quote <= 0:
                total_available_quote = 100000.0 if quote_currency == 'INR' else 1000.0

            # Determine price in quote currency
            calc_price = body.price
            if not calc_price or calc_price <= 0:
                try:
                    ticker = await exch.get_ticker(body.symbol)
                    calc_price = ticker.last_price
                except Exception:
                    calc_price = 0.0

            if not calc_price or calc_price <= 0:
                base_usd = get_symbol_baseline_price(body.symbol)
                calc_price = base_usd * usd_inr_rate if quote_currency == 'INR' else base_usd

            # Risk calculation: 2% of available capital in quote currency
            risk_amount = total_available_quote * 0.02
            raw_quantity = 0.0

            if body.stop_loss and abs(calc_price - body.stop_loss) > 0:
                risk_per_unit = abs(calc_price - body.stop_loss)
                raw_quantity = risk_amount / risk_per_unit
            else:
                # Default: 5% of capital with leverage
                lev = max(1.0, float(body.leverage or 1.0))
                raw_quantity = (total_available_quote * 0.05 * lev) / calc_price

            quantity = format_order_quantity(body.symbol, raw_quantity)
        else:
            # Custom quantity passed from frontend -> sanitize precision
            quantity = format_order_quantity(body.symbol, float(quantity))

        if not quantity or quantity <= 0:
            quantity = format_order_quantity(body.symbol, 0.001)

        result = None
        
        if body.trading_mode == TradingModeEnum.MARGIN:
            result = await exch.place_margin_order(
                pair=body.symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=body.price,
                leverage=body.leverage,
                stop_loss=body.stop_loss,
                take_profit=body.take_profit,
            )
        elif body.trading_mode == TradingModeEnum.FUTURES:
            # Set leverage first (best-effort)
            try:
                await exch.set_futures_leverage(body.symbol, body.leverage)
            except Exception as le:
                logger.debug(f"Setting leverage failed ({le}), proceeding to place order")
            result = await exch.place_futures_order(
                pair=body.symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=body.price,
                leverage=body.leverage,
            )
            # Set TP/SL if provided
            if result and (body.stop_loss or body.take_profit):
                try:
                    positions = await exch.get_futures_positions()
                    pos = next(
                        (p for p in positions if p.pair == body.symbol),
                        None,
                    )
                    if pos:
                        await exch.set_futures_tp_sl(
                            pos.id, body.take_profit, body.stop_loss
                        )
                except Exception:
                    pass  # TP/SL is best-effort
        else:  # SPOT
            result = await exch.place_spot_order(
                pair=body.symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=body.price,
            )
        
        # Check if order was rejected by exchange
        order_status = getattr(result, 'status', 'executed')
        if order_status in ('rejected', 'failed', 'error'):
            raw_err = (result.raw or {}) if hasattr(result, 'raw') and isinstance(result.raw, dict) else {}
            msg = raw_err.get('message', raw_err.get('error', f"Order {order_status} by exchange"))
            return ExecuteTradeResponse(
                status="rejected",
                order_id=result.order_id if result else None,
                exchange=body.exchange,
                trading_mode=body.trading_mode.value,
                side=body.side,
                quantity=quantity,
                price=body.price,
                leverage=body.leverage,
                stop_loss=body.stop_loss,
                take_profit=body.take_profit,
                message=f"Order rejected: {msg}",
            )

        return ExecuteTradeResponse(
            status="executed",
            order_id=result.order_id if result else None,
            exchange=body.exchange,
            trading_mode=body.trading_mode.value,
            side=body.side,
            quantity=quantity,
            price=body.price,
            leverage=body.leverage,
            stop_loss=body.stop_loss,
            take_profit=body.take_profit,
            message=f"Order placed successfully on {body.exchange}",
        )
    
    except ExchangeError as e:
        return ExecuteTradeResponse(
            status="error",
            exchange=body.exchange,
            trading_mode=body.trading_mode.value,
            side=body.side,
            quantity=body.quantity or 0,
            message=f"Exchange error: {e.message}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trade execution failed: {str(e)}")


@router.post("/close")
async def close_position(
    request: Request,
    body: ClosePositionRequest,
):
    """Close an active position."""
    factory = request.app.state.exchange_factory
    try:
        exch = factory.get_or_raise(body.exchange)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    try:
        if body.trading_mode == TradingModeEnum.MARGIN:
            success = await exch.exit_margin_position(body.position_id)
        else:
            success = await exch.exit_futures_position(body.position_id)
        
        return {
            "status": "closed" if success else "failed",
            "position_id": body.position_id,
            "exchange": body.exchange,
        }
    except ExchangeError as e:
        raise HTTPException(status_code=400, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/active-orders")
async def get_active_orders(
    request: Request,
    exchange: str = "coindcx",
    trading_mode: str = "futures",
    pair: str = None,
):
    """Get all active orders."""
    factory = request.app.state.exchange_factory
    try:
        exch = factory.get_or_raise(exchange)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    try:
        if trading_mode == "margin":
            orders = await exch.get_margin_orders(pair)
        elif trading_mode == "futures":
            orders = await exch.futures_trading.get_active_orders(pair)
        else:
            orders = await exch.get_active_spot_orders(pair)
        
        return {
            "exchange": exchange,
            "trading_mode": trading_mode,
            "count": len(orders),
            "orders": [
                {
                    "order_id": o.order_id,
                    "status": o.status,
                    "side": o.side,
                    "order_type": o.order_type,
                    "price": o.price,
                    "quantity": o.quantity,
                    "filled_quantity": o.filled_quantity,
                }
                for o in orders
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
