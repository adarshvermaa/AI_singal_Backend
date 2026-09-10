from fastapi import APIRouter, Request, HTTPException
from app.api.schemas import (
    ExecuteTradeRequest, ExecuteTradeResponse,
    ClosePositionRequest, TradingModeEnum,
)
from app.exchanges.base import OrderSide, OrderType, ExchangeError

router = APIRouter()


@router.post("/execute", response_model=ExecuteTradeResponse)
async def execute_trade(
    request: Request,
    body: ExecuteTradeRequest,
):
    """
    Execute a trade on the connected exchange.
    Supports spot, margin, and futures orders.
    """
    factory = request.app.state.exchange_factory
    try:
        exch = factory.get_or_raise(body.exchange)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    try:
        side = OrderSide.BUY if body.side.lower() == 'buy' else OrderSide.SELL
        order_type = OrderType(body.order_type)
        
        # Auto-calculate position size if needed
        quantity = body.quantity
        if body.auto_size and quantity is None:
            # Get USDT balance and calculate 2% risk
            balances = await exch.get_balances()
            usdt_balance = next(
                (b for b in balances if b.currency.upper() in ('USDT', 'INR')),
                None,
            )
            if not usdt_balance:
                return ExecuteTradeResponse(
                    status="rejected",
                    exchange=body.exchange,
                    trading_mode=body.trading_mode.value,
                    side=body.side,
                    quantity=0,
                    message="No USDT balance found",
                )
            # Risk 2% of available balance
            risk_amount = usdt_balance.available * 0.02
            calc_price = body.price
            if not calc_price or calc_price <= 0:
                try:
                    ticker = await exch.get_ticker(body.symbol)
                    calc_price = ticker.last_price
                except Exception:
                    calc_price = 0.0

            if body.stop_loss and calc_price and calc_price > 0:
                risk_per_unit = abs(calc_price - body.stop_loss)
                if risk_per_unit > 0:
                    quantity = risk_amount / risk_per_unit
            elif calc_price and calc_price > 0:
                # Default: use 5% of balance
                quantity = (usdt_balance.available * 0.05) / calc_price
            else:
                return ExecuteTradeResponse(
                    status="rejected",
                    exchange=body.exchange,
                    trading_mode=body.trading_mode.value,
                    side=body.side,
                    quantity=0,
                    message="Cannot calculate position size without price",
                )
        
        if not quantity or quantity <= 0:
            return ExecuteTradeResponse(
                status="rejected",
                exchange=body.exchange,
                trading_mode=body.trading_mode.value,
                side=body.side,
                quantity=0,
                message="Invalid quantity",
            )
        
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
            # Set leverage first
            await exch.set_futures_leverage(body.symbol, body.leverage)
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
