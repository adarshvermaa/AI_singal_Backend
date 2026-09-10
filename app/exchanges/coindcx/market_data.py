import logging
from typing import Optional
import httpx
import time

from app.exchanges.base import (
    Candle, OrderBook, OrderBookEntry, Trade, Ticker, MarketInfo,
    FuturesInfo, TradingMode, ExchangeError,
)
from app.exchanges.coindcx import constants as C

logger = logging.getLogger(__name__)


def safe_float(val, default: float = 0.0) -> float:
    """Safely convert any value to float without raising TypeError or ValueError."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def safe_int(val, default: int = 0) -> int:
    """Safely convert any value to int without raising TypeError or ValueError."""
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


class CoinDCXMarketData:
    """Public market data methods for CoinDCX."""
    
    def __init__(self, client: httpx.AsyncClient):
        self.client = client
        self._markets_cache: list[MarketInfo] = []
        self._cache_time: float = 0
        self._cache_ttl: float = 300  # 5 minutes
    
    async def get_candles(
        self, pair: str, interval: str, limit: int = 500
    ) -> list[Candle]:
        """Fetch spot OHLCV candles from CoinDCX."""
        coindcx_interval = C.SPOT_INTERVALS.get(interval, interval)
        resp = await self.client.get(
            C.CANDLES_URL,
            params={'pair': pair, 'interval': coindcx_interval, 'limit': limit},
        )
        self._check_response(resp, 'get_candles')
        data = resp.json()
        if not isinstance(data, list):
            return []
        candles = [
            Candle(
                timestamp=safe_int(c.get('time') or c.get('t')),
                open=safe_float(c.get('open') or c.get('o')),
                high=safe_float(c.get('high') or c.get('h')),
                low=safe_float(c.get('low') or c.get('l')),
                close=safe_float(c.get('close') or c.get('c')),
                volume=safe_float(c.get('volume') or c.get('v')),
            )
            for c in data
            if isinstance(c, dict)
        ]
        return candles[-limit:] if len(candles) > limit else candles
    
    async def get_futures_candles(
        self, pair: str, interval: str, limit: int = 500,
        from_ts: Optional[int] = None, to_ts: Optional[int] = None,
    ) -> list[Candle]:
        """Fetch futures OHLCV candles from CoinDCX with spot fallback."""
        resolution = C.FUTURES_INTERVALS.get(interval, interval)
        if to_ts is None:
            to_ts = int(time.time())
        if from_ts is None:
            # Calculate from_ts based on interval and limit
            interval_seconds = self._interval_to_seconds(interval)
            from_ts = to_ts - (limit * interval_seconds)
        
        try:
            resp = await self.client.get(
                C.FUTURES_CANDLES_URL,
                params={
                    'pair': pair,
                    'resolution': resolution,
                    'from': from_ts,
                    'to': to_ts,
                    'pcode': 'f',
                },
            )
            self._check_response(resp, 'get_futures_candles')
            data = resp.json()
            candle_list = data.get('data', data) if isinstance(data, dict) else data
            if not isinstance(candle_list, list) or len(candle_list) == 0:
                return await self.get_candles(pair, interval, limit)
            return [
                Candle(
                    timestamp=safe_int(c.get('time') or c.get('t')),
                    open=safe_float(c.get('open') or c.get('o')),
                    high=safe_float(c.get('high') or c.get('h')),
                    low=safe_float(c.get('low') or c.get('l')),
                    close=safe_float(c.get('close') or c.get('c')),
                    volume=safe_float(c.get('volume') or c.get('v')),
                )
                for c in candle_list
                if isinstance(c, dict)
            ]
        except Exception as e:
            logger.debug(f"Futures candles fetch failed ({e}), falling back to spot candles")
            return await self.get_candles(pair, interval, limit)
    
    async def get_orderbook(self, pair: str, depth: int = 20) -> OrderBook:
        """Fetch spot order book from CoinDCX."""
        resp = await self.client.get(
            C.ORDERBOOK_URL,
            params={'pair': pair},
        )
        self._check_response(resp, 'get_orderbook')
        data = resp.json()

        raw_bids = data.get('bids', {})
        raw_asks = data.get('asks', {})

        bids: list[OrderBookEntry] = []
        if isinstance(raw_bids, dict):
            for p, q in raw_bids.items():
                try:
                    bids.append(OrderBookEntry(price=float(p), quantity=float(q)))
                except (ValueError, TypeError):
                    continue
        elif isinstance(raw_bids, list):
            for b in raw_bids:
                try:
                    if isinstance(b, (list, tuple)) and len(b) >= 2:
                        bids.append(OrderBookEntry(price=float(b[0]), quantity=float(b[1])))
                    elif isinstance(b, dict):
                        bids.append(OrderBookEntry(
                            price=float(b.get('price', b.get('p', 0))),
                            quantity=float(b.get('quantity', b.get('q', 0)))
                        ))
                except (ValueError, TypeError):
                    continue

        bids.sort(key=lambda x: x.price, reverse=True)
        if depth and depth > 0:
            bids = bids[:depth]

        asks: list[OrderBookEntry] = []
        if isinstance(raw_asks, dict):
            for p, q in raw_asks.items():
                try:
                    asks.append(OrderBookEntry(price=float(p), quantity=float(q)))
                except (ValueError, TypeError):
                    continue
        elif isinstance(raw_asks, list):
            for a in raw_asks:
                try:
                    if isinstance(a, (list, tuple)) and len(a) >= 2:
                        asks.append(OrderBookEntry(price=float(a[0]), quantity=float(a[1])))
                    elif isinstance(a, dict):
                        asks.append(OrderBookEntry(
                            price=float(a.get('price', a.get('p', 0))),
                            quantity=float(a.get('quantity', a.get('q', 0)))
                        ))
                except (ValueError, TypeError):
                    continue

        asks.sort(key=lambda x: x.price)
        if depth and depth > 0:
            asks = asks[:depth]

        return OrderBook(
            bids=bids,
            asks=asks,
            timestamp=int(time.time() * 1000),
        )

    async def get_futures_orderbook(
        self, pair: str, depth: int = 20
    ) -> OrderBook:
        """Fetch futures order book with graceful spot fallback."""
        pair_clean = pair
        if not pair_clean.endswith('-futures'):
            pair_clean = f"{pair_clean}-futures"
        url = f"{C.FUTURES_ORDERBOOK_URL}/{pair_clean}/{depth}"
        try:
            resp = await self.client.get(url)
            self._check_response(resp, 'get_futures_orderbook')
            data = resp.json()
            raw_bids = data.get('bids', [])
            raw_asks = data.get('asks', [])

            bids: list[OrderBookEntry] = []
            for b in raw_bids:
                try:
                    if isinstance(b, dict):
                        bids.append(OrderBookEntry(price=float(b.get('price', 0)), quantity=float(b.get('quantity', 0))))
                    elif isinstance(b, (list, tuple)) and len(b) >= 2:
                        bids.append(OrderBookEntry(price=float(b[0]), quantity=float(b[1])))
                except (ValueError, TypeError):
                    continue

            asks: list[OrderBookEntry] = []
            for a in raw_asks:
                try:
                    if isinstance(a, dict):
                        asks.append(OrderBookEntry(price=float(a.get('price', 0)), quantity=float(a.get('quantity', 0))))
                    elif isinstance(a, (list, tuple)) and len(a) >= 2:
                        asks.append(OrderBookEntry(price=float(a[0]), quantity=float(a[1])))
                except (ValueError, TypeError):
                    continue

            return OrderBook(
                bids=bids[:depth],
                asks=asks[:depth],
                timestamp=int(time.time() * 1000),
            )
        except Exception as e:
            logger.debug(f"Futures orderbook fetch failed ({e}), falling back to spot orderbook")
            return await self.get_orderbook(pair, depth=depth)
    
    async def get_ticker(self, pair: str) -> Ticker:
        """Fetch 24h ticker for a specific pair."""
        tickers = await self.get_all_tickers()
        # pair format: B-BTC_USDT → market: BTCUSDT
        market_name = pair.split('-', 1)[1].replace('_', '') if '-' in pair else pair
        for t in tickers:
            if t.pair == market_name or t.pair == pair:
                return t
        raise ExchangeError(f"Ticker not found for pair: {pair}")
    
    async def get_all_tickers(self) -> list[Ticker]:
        """Fetch 24h ticker data for all pairs."""
        resp = await self.client.get(C.TICKER_URL)
        self._check_response(resp, 'get_all_tickers')
        data = resp.json()
        if not isinstance(data, list):
            return []
        tickers: list[Ticker] = []
        for t in data:
            if not isinstance(t, dict):
                continue
            tickers.append(
                Ticker(
                    pair=t.get('market', ''),
                    last_price=safe_float(t.get('last_price')),
                    high_24h=safe_float(t.get('high')),
                    low_24h=safe_float(t.get('low')),
                    volume_24h=safe_float(t.get('volume')),
                    change_24h=safe_float(t.get('change_24_hour')),
                    bid=safe_float(t.get('bid')),
                    ask=safe_float(t.get('ask')),
                    timestamp=safe_int(t.get('timestamp')),
                )
            )
        return tickers

    async def get_recent_trades(
        self, pair: str, limit: int = 50
    ) -> list[Trade]:
        """Fetch recent trades."""
        resp = await self.client.get(
            C.TRADE_HISTORY_URL,
            params={'pair': pair, 'limit': limit},
        )
        self._check_response(resp, 'get_recent_trades')
        data = resp.json()
        if not isinstance(data, list):
            return []
        trades: list[Trade] = []
        for t in data:
            if not isinstance(t, dict):
                continue
            trades.append(
                Trade(
                    price=safe_float(t.get('p') or t.get('price')),
                    quantity=safe_float(t.get('q') or t.get('quantity')),
                    is_buyer_maker=bool(t.get('m') or t.get('is_buyer_maker', False)),
                    timestamp=safe_int(t.get('T') or t.get('timestamp')),
                )
            )
        return trades

    async def get_markets(self) -> list[MarketInfo]:
        """Fetch all market details. Uses 5-min cache."""
        if self._markets_cache and (time.time() - self._cache_time) < self._cache_ttl:
            return self._markets_cache
        
        resp = await self.client.get(C.MARKETS_DETAILS_URL)
        self._check_response(resp, 'get_markets')
        data = resp.json()
        if not isinstance(data, list):
            return []

        markets = []
        for m in data:
            if not isinstance(m, dict):
                continue
            # Determine supported trading modes
            modes = [TradingMode.SPOT]
            max_lev = safe_float(m.get('max_leverage'), 1.0)
            if max_lev > 1.0:
                modes.append(TradingMode.MARGIN)
            
            markets.append(MarketInfo(
                pair=m.get('pair', ''),
                base_currency=m.get('target_currency_short_name') or m.get('base_currency_short_name', ''),
                quote_currency=m.get('base_currency_short_name') or m.get('target_currency_short_name', ''),
                min_quantity=safe_float(m.get('min_quantity'), 0.0),
                max_quantity=safe_float(m.get('max_quantity'), 999999.0),
                min_price=safe_float(m.get('min_price'), 0.0),
                max_price=safe_float(m.get('max_price'), 999999.0),
                min_notional=safe_float(m.get('min_notional'), 0.0),
                step_size=safe_float(m.get('step') or m.get('step_size'), 0.001),
                tick_size=safe_float(m.get('min_price_step') or m.get('tick_size'), 0.01),
                max_leverage=max_lev,
                trading_modes=modes,
                is_active=m.get('status', 'active') == 'active',
            ))
        
        self._markets_cache = markets
        self._cache_time = time.time()
        return markets

    async def get_futures_instruments(self) -> list[dict]:
        """List all active futures instruments."""
        resp = await self.client.get(
            C.FUTURES_INSTRUMENTS_URL,
            params={'margin_currency_short_name[]': 'USDT'},
        )
        self._check_response(resp, 'get_futures_instruments')
        return resp.json()

    async def get_futures_info(self, pair: str) -> FuturesInfo:
        """Fetch futures-specific data: funding rate, mark price, OI, L/S ratio."""
        # Get real-time prices (funding rate, mark price)
        rt_data = {}
        try:
            rt_resp = await self.client.get(C.FUTURES_RT_PRICES_URL)
            if rt_resp.status_code == 200:
                rt_data = rt_resp.json()
            else:
                logger.debug(f"Futures RT prices returned HTTP {rt_resp.status_code}")
        except Exception as e:
            logger.debug(f"Failed to fetch futures RT prices: {e}")
        
        # Find our pair in the response
        pair_key = pair.replace('B-', '').replace('_', '')
        pair_data = None
        if isinstance(rt_data, dict):
            pair_data = rt_data.get(pair_key, rt_data.get(pair, {}))
        elif isinstance(rt_data, list):
            pair_data = next(
                (p for p in rt_data if p.get('pair', '') == pair or p.get('symbol', '') == pair_key),
                {}
            )
        
        if not pair_data:
            pair_data = {}
        
        # Get pair stats (long/short ratio)
        stats_data = {}
        try:
            stats_resp = await self.client.get(
                C.FUTURES_STATS_URL, params={'pair': pair}
            )
            if stats_resp.status_code == 200:
                stats_data = stats_resp.json()
        except Exception as e:
            logger.warning(f"Failed to fetch futures stats: {e}")
        
        return FuturesInfo(
            pair=pair,
            funding_rate=safe_float(pair_data.get('fr') or pair_data.get('funding_rate')),
            mark_price=safe_float(pair_data.get('mp') or pair_data.get('mark_price')),
            index_price=safe_float(pair_data.get('ip') or pair_data.get('index_price')),
            open_interest=safe_float(pair_data.get('oi') or pair_data.get('open_interest')),
            volume_24h=safe_float(pair_data.get('v') or pair_data.get('volume_24h')),
            long_short_ratio=safe_float(stats_data.get('long_short_ratio'), 1.0),
            next_funding_time=safe_int(pair_data.get('nft') or pair_data.get('next_funding_time')),
        )
    
    def _check_response(self, resp: httpx.Response, method: str):
        """Check HTTP response for errors."""
        if resp.status_code != 200:
            try:
                error_data = resp.json()
            except Exception:
                error_data = {'message': resp.text}
            raise ExchangeError(
                f"CoinDCX {method} failed: {error_data}",
                code=resp.status_code,
                raw=error_data,
            )
    
    @staticmethod
    def _interval_to_seconds(interval: str) -> int:
        """Convert interval string to seconds."""
        multipliers = {'m': 60, 'h': 3600, 'd': 86400, 'w': 604800}
        unit = interval[-1].lower()
        value = int(interval[:-1])
        return value * multipliers.get(unit, 60)
