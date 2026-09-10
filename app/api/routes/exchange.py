import logging
from fastapi import APIRouter, Request, HTTPException, Query
from app.api.schemas import (
    ConnectExchangeRequest, ConnectExchangeResponse,
    BalancesResponse, BalanceItem,
    PositionsResponse, PositionItem,
)
from app.exchanges.factory import ExchangeFactory

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/connect", response_model=ConnectExchangeResponse)
async def connect_exchange(
    request: Request,
    body: ConnectExchangeRequest,
):
    """
    Connect to an exchange with API credentials.
    This verifies the credentials and stores the connection.
    """
    factory: ExchangeFactory = request.app.state.exchange_factory
    try:
        exchange = factory.create(
            body.exchange,
            api_key=body.api_key,
            api_secret=body.api_secret,
        )
        await exchange.initialize()
        
        # Verify credentials by fetching user info
        user_name = None
        try:
            info = await exchange.get_user_info()
            user_name = info.get('first_name', info.get('name', 'Trader'))
        except Exception as e:
            logger.warning(f"CoinDCX authentication failed during connect: {e}")
            return ConnectExchangeResponse(
                exchange=body.exchange,
                status="failed",
                message=f"Authentication failed: {str(e)}",
            )
        
        return ConnectExchangeResponse(
            exchange=body.exchange,
            status="connected",
            message=f"Successfully connected to {body.exchange}",
            user_name=user_name,
            is_demo=getattr(exchange, 'is_demo', False),
        )
    except Exception as e:
        return ConnectExchangeResponse(
            exchange=body.exchange,
            status="failed",
            message=f"Failed to connect: {str(e)}",
        )


@router.get("/status")
async def exchange_status(request: Request, exchange: str = Query(default="coindcx")):
    """Get connected exchanges status."""
    factory: ExchangeFactory = request.app.state.exchange_factory
    try:
        exch = factory.get_or_raise(exchange)
    except Exception:
        exch = factory.get(exchange)

    is_authenticated = False
    is_demo = getattr(exch, 'is_demo', False) if exch else False
    user_name = None
    msg = None

    if exch:
        is_authenticated = getattr(exch, 'is_authenticated', False)
        if not is_authenticated and exch.auth.api_key and exch.auth.api_secret and not is_demo:
            try:
                info = await exch.get_user_info()
                if info and ('first_name' in info or 'email' in info or 'name' in info):
                    is_authenticated = True
                    user_name = info.get('first_name', info.get('name', 'Trader'))
            except Exception as e:
                msg = str(e)
        elif is_demo:
            is_authenticated = True
            user_name = "Demo Trader (Paper)"

    return {
        "connected": factory.connected_exchanges,
        "authenticated": is_authenticated,
        "is_demo": is_demo,
        "user_name": user_name,
        "message": msg,
        "supported": ExchangeFactory.supported_exchanges(),
    }


@router.get("/balances", response_model=BalancesResponse)
async def get_balances(
    request: Request,
    exchange: str = Query(default="coindcx"),
):
    """Get wallet balances from connected exchange."""
    factory: ExchangeFactory = request.app.state.exchange_factory
    try:
        exch = factory.get_or_raise(exchange)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    try:
        balances = await exch.get_balances()
        items = [
            BalanceItem(
                currency=b.currency,
                available=b.available,
                locked=b.locked,
                total=b.total,
            )
            for b in balances
        ]
        return BalancesResponse(
            exchange=exchange,
            balances=items,
            authenticated=getattr(exch, 'is_authenticated', True),
            message="Balances fetched successfully",
        )
    except Exception as e:
        logger.warning(f"Failed to get balances from {exchange}: {e}")
        return BalancesResponse(
            exchange=exchange,
            balances=[],
            authenticated=False,
            message=str(e),
        )


@router.get("/positions", response_model=PositionsResponse)
async def get_positions(
    request: Request,
    exchange: str = Query(default="coindcx"),
):
    """Get active positions from connected exchange."""
    factory: ExchangeFactory = request.app.state.exchange_factory
    try:
        exch = factory.get_or_raise(exchange)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    try:
        positions = await exch.get_futures_positions()
        items = [
            PositionItem(
                id=p.id,
                pair=p.pair,
                side=p.side,
                quantity=p.quantity,
                entry_price=p.entry_price,
                mark_price=p.mark_price,
                leverage=p.leverage,
                unrealized_pnl=p.unrealized_pnl,
                liquidation_price=p.liquidation_price,
                take_profit=p.take_profit,
                stop_loss=p.stop_loss,
            )
            for p in positions
        ]
        return PositionsResponse(
            exchange=exchange,
            positions=items,
            authenticated=getattr(exch, 'is_authenticated', True),
            message=f"{len(items)} active positions",
        )
    except Exception as e:
        logger.warning(f"Failed to get positions from {exchange}: {e}")
        return PositionsResponse(
            exchange=exchange,
            positions=[],
            authenticated=False,
            message=str(e),
        )


@router.post("/disconnect")
async def disconnect_exchange(
    request: Request,
    exchange: str = Query(default="coindcx"),
):
    """Disconnect from an exchange."""
    factory: ExchangeFactory = request.app.state.exchange_factory
    exch = factory.get(exchange)
    if exch:
        await exch.close()
        factory._instances.pop(exchange, None)
        return {"status": "disconnected", "exchange": exchange}
    return {"status": "not_connected", "exchange": exchange}
