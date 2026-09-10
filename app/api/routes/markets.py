import logging
from fastapi import APIRouter, Request, Query, HTTPException
from app.api.schemas import MarketsResponse, MarketItem

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/markets", response_model=MarketsResponse)
async def get_markets(
    request: Request,
    exchange: str = Query(default="coindcx", description="Exchange name"),
    quote: str = Query(default=None, description="Filter by quote currency e.g. USDT"),
    search: str = Query(default=None, description="Search by pair name"),
):
    """List all available trading pairs from an exchange."""
    factory = request.app.state.exchange_factory
    try:
        exch = factory.get_or_raise(exchange)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    try:
        markets = await exch.get_markets()
        tickers = {}
        try:
            all_tickers = await exch.get_all_tickers()
            for t in all_tickers:
                tickers[t.pair] = t
        except Exception as te:
            logger.warning(f"Could not load tickers: {te}")
        
        items = []
        for m in markets:
            if not m.is_active:
                continue
            if quote and m.quote_currency.upper() != quote.upper():
                continue
            if search and search.upper() not in m.pair.upper():
                continue
            
            clean_market = m.pair.split('-', 1)[1].replace('_', '') if '-' in m.pair else m.pair
            ticker = tickers.get(clean_market) or tickers.get(m.pair)
            items.append(MarketItem(
                pair=m.pair,
                base_currency=m.base_currency,
                quote_currency=m.quote_currency,
                last_price=ticker.last_price if ticker else None,
                change_24h=ticker.change_24h if ticker else None,
                volume_24h=ticker.volume_24h if ticker else None,
                max_leverage=m.max_leverage or 1.0,
                trading_modes=[mode.value for mode in m.trading_modes],
                is_active=m.is_active,
            ))
        
        return MarketsResponse(
            exchange=exchange,
            count=len(items),
            markets=sorted(items, key=lambda x: x.volume_24h or 0, reverse=True),
        )
    except Exception as e:
        logger.error(f"Failed to fetch markets: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch markets: {str(e)}")


@router.get("/markets/futures")
async def get_futures_markets(
    request: Request,
    exchange: str = Query(default="coindcx"),
):
    """List all active futures instruments."""
    factory = request.app.state.exchange_factory
    try:
        exch = factory.get_or_raise(exchange)
        instruments = await exch.get_futures_instruments()
        return {"exchange": exchange, "count": len(instruments), "instruments": instruments}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/markets/candles")
async def get_candles(
    request: Request,
    symbol: str = Query(..., description="Pair e.g. B-BTC_USDT"),
    interval: str = Query(default="15m", description="Timeframe e.g. 15m, 1h, 4h"),
    limit: int = Query(default=300, ge=10, le=1000),
    trading_mode: str = Query(default="futures"),
    exchange: str = Query(default="coindcx"),
):
    """Fetch live historical candlestick data for the TradingView chart."""
    factory = request.app.state.exchange_factory
    try:
        exch = factory.get_or_raise(exchange)
        candles = []
        if trading_mode.lower() == "futures":
            try:
                candles = await exch.get_futures_candles(symbol, interval, limit=limit)
            except Exception as fe:
                logger.warning(f"Futures candles failed ({fe}), falling back to spot")
                candles = await exch.get_candles(symbol, interval, limit=limit)
        else:
            candles = await exch.get_candles(symbol, interval, limit=limit)

        if not candles and trading_mode.lower() == "futures":
            candles = await exch.get_candles(symbol, interval, limit=limit)
        
        return {
            "symbol": symbol,
            "interval": interval,
            "count": len(candles),
            "candles": [
                {
                    "time": c.timestamp if c.timestamp < 10000000000 else int(c.timestamp / 1000),
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                }
                for c in candles
            ],
        }
    except Exception as e:
        logger.error(f"Failed to fetch candles: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch candles: {str(e)}")

