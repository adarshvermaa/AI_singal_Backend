import logging
import time
from typing import Optional
import httpx

from app.exchanges.base import (
    ExchangeBase, Candle, OrderBook, Trade, Ticker, MarketInfo,
    FuturesInfo, Balance, OrderSide, OrderType, MarginType,
    OrderResult, Position, ExchangeError,
)
from app.exchanges.coindcx.auth import CoinDCXAuth
from app.exchanges.coindcx.market_data import CoinDCXMarketData
from app.exchanges.coindcx.spot import CoinDCXSpot
from app.exchanges.coindcx.margin import CoinDCXMargin
from app.exchanges.coindcx.futures import CoinDCXFutures
from app.exchanges.coindcx import constants as C

logger = logging.getLogger(__name__)


class CoinDCXExchange(ExchangeBase):
    """
    CoinDCX exchange adapter.
    Composes: MarketData + Spot + Margin + Futures sub-modules.
    Implements the full ExchangeBase interface.
    """
    
    name = 'coindcx'
    
    def __init__(self, api_key: str, api_secret: str, **kwargs):
        self.api_key = api_key
        self.api_secret = api_secret
        import os
        self.is_demo = (
            api_key.lower().strip() in ('demo', 'paper', 'test', 'simulated')
            or os.environ.get('DEMO_MODE', '').lower() in ('true', '1')
        )
        self.client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
            headers={'User-Agent': 'AI-Signal-Engine/1.0'},
        )
        self.auth = CoinDCXAuth(api_key, api_secret)
        self.is_authenticated: bool = True if self.is_demo else False
        
        # Interactive Paper Trading state for Demo Mode
        self._demo_balances: dict[str, Balance] = {
            'USDT': Balance(currency='USDT', available=10000.0, locked=0.0, total=10000.0),
            'INR': Balance(currency='INR', available=50000.0, locked=0.0, total=50000.0),
            'BTC': Balance(currency='BTC', available=0.25, locked=0.0, total=0.25),
        }
        self._demo_positions: list[Position] = [
            Position(
                id='pos_demo_btc_01',
                pair='B-BTC_USDT',
                side='long',
                quantity=0.05,
                entry_price=79200.0,
                mark_price=79520.0,
                leverage=5.0,
                unrealized_pnl=16.0,
                liquidation_price=63500.0,
                take_profit=82000.0,
                stop_loss=78000.0,
            )
        ]
        
        # Compose sub-modules
        self.market_data = CoinDCXMarketData(self.client)
        self.spot_trading = CoinDCXSpot(self.client, self.auth)
        self.margin_trading = CoinDCXMargin(self.client, self.auth)
        self.futures_trading = CoinDCXFutures(self.client, self.auth)
    
    async def initialize(self) -> None:
        """Verify connection by fetching user info."""
        try:
            info = await self.get_user_info()
            if info and ('first_name' in info or 'email' in info or 'name' in info):
                self.is_authenticated = True
                logger.info(f"CoinDCX connected. User: {info.get('first_name', 'Unknown')}")
            else:
                self.is_authenticated = False
        except Exception as e:
            self.is_authenticated = False
            logger.warning(f"CoinDCX init warning (public APIs still work): {e}")
    
    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()
    
    # === MARKET DATA (delegates to CoinDCXMarketData) ===
    
    async def get_candles(self, pair: str, interval: str, limit: int = 500) -> list[Candle]:
        return await self.market_data.get_candles(pair, interval, limit)
    
    async def get_orderbook(self, pair: str, depth: int = 20) -> OrderBook:
        return await self.market_data.get_orderbook(pair, depth)
    
    async def get_ticker(self, pair: str) -> Ticker:
        return await self.market_data.get_ticker(pair)
    
    async def get_all_tickers(self) -> list[Ticker]:
        return await self.market_data.get_all_tickers()
    
    async def get_recent_trades(self, pair: str, limit: int = 50) -> list[Trade]:
        return await self.market_data.get_recent_trades(pair, limit)
    
    async def get_markets(self) -> list[MarketInfo]:
        return await self.market_data.get_markets()
    
    async def get_futures_candles(
        self, pair: str, interval: str, limit: int = 500,
        from_ts: Optional[int] = None, to_ts: Optional[int] = None
    ) -> list[Candle]:
        return await self.market_data.get_futures_candles(pair, interval, limit, from_ts, to_ts)
    
    async def get_futures_orderbook(self, pair: str, depth: int = 20) -> OrderBook:
        return await self.market_data.get_futures_orderbook(pair, depth)
    
    async def get_futures_info(self, pair: str) -> FuturesInfo:
        return await self.market_data.get_futures_info(pair)
    
    async def get_futures_instruments(self) -> list[dict]:
        return await self.market_data.get_futures_instruments()
    
    # === ACCOUNT ===
    
    async def get_balances(self) -> list[Balance]:
        if self.is_demo:
            self.is_authenticated = True
            return list(self._demo_balances.values())

        if not self.auth.api_key or not self.auth.api_secret:
            return []
        body = {}
        json_body, headers = self.auth.prepare_body(body)
        resp = await self.client.post(
            C.BALANCES_URL, content=json_body, headers=headers
        )
        if resp.status_code == 401:
            self.is_authenticated = False
            raise ExchangeError("Invalid CoinDCX API credentials (HTTP 401). Please check API key and secret.")
        if resp.status_code == 422:
            error = resp.json() if resp.text else {}
            raise ExchangeError(f"Validation error: {error.get('message', resp.text)}")
        if resp.status_code == 429:
            raise ExchangeError("Rate limit exceeded. Please wait before retrying.")
        if resp.status_code != 200:
            raise ExchangeError(f"Failed to get balances (HTTP {resp.status_code}): {resp.text}")
        
        self.is_authenticated = True
        data = resp.json()
        if not isinstance(data, list):
            if isinstance(data, dict) and 'balances' in data:
                data = data['balances']
            else:
                data = []

        balances: list[Balance] = []
        currencies_seen = set()
        for b in data:
            if not isinstance(b, dict):
                continue
            curr = (b.get('currency') or '').upper()
            avail = float(b.get('balance') or 0)
            locked = float(b.get('locked_balance') or 0)
            if avail > 0 or locked > 0 or curr in ('USDT', 'INR', 'BTC'):
                currencies_seen.add(curr)
                balances.append(
                    Balance(
                        currency=curr,
                        available=avail,
                        locked=locked,
                        total=avail + locked,
                    )
                )

        # Guarantee USDT and INR are always present if response was valid
        if 'USDT' not in currencies_seen:
            balances.insert(0, Balance(currency='USDT', available=0.0, locked=0.0, total=0.0))
        if 'INR' not in currencies_seen:
            balances.append(Balance(currency='INR', available=0.0, locked=0.0, total=0.0))

        return balances

    async def get_user_info(self) -> dict:
        if self.is_demo:
            self.is_authenticated = True
            return {'first_name': 'Demo Trader', 'name': 'Demo Trader (Paper)', 'email': 'demo@antigravity.ai'}

        if not self.auth.api_key or not self.auth.api_secret:
            return {}
        body = {}
        json_body, headers = self.auth.prepare_body(body)
        resp = await self.client.post(
            C.USER_INFO_URL, content=json_body, headers=headers
        )
        if resp.status_code == 401:
            self.is_authenticated = False
            raise ExchangeError("Invalid CoinDCX API credentials (HTTP 401)")
        if resp.status_code == 422:
            error = resp.json() if resp.text else {}
            raise ExchangeError(f"Validation error: {error.get('message', resp.text)}")
        if resp.status_code == 429:
            raise ExchangeError("Rate limit exceeded. Please wait before retrying.")
        if resp.status_code != 200:
            raise ExchangeError(f"Failed to get user info: {resp.text}")
        data = resp.json()
        self.is_authenticated = True
        return data
    
    # === SPOT TRADING (delegates to CoinDCXSpot) ===
    
    async def place_spot_order(
        self, pair: str, side: OrderSide, order_type: OrderType,
        quantity: float, price: Optional[float] = None,
    ) -> OrderResult:
        if self.is_demo:
            import uuid
            side_str = side.value if hasattr(side, 'value') else str(side)
            type_str = order_type.value if hasattr(order_type, 'value') else str(order_type)
            exec_price = price or 79500.0
            total_cost = quantity * exec_price
            
            # Update demo USDT balance
            if side_str.lower() == 'buy' and 'USDT' in self._demo_balances:
                avail = self._demo_balances['USDT'].available
                self._demo_balances['USDT'].available = max(0.0, avail - total_cost)
                self._demo_balances['USDT'].total = self._demo_balances['USDT'].available + self._demo_balances['USDT'].locked

            return OrderResult(
                order_id=f"demo_spot_{uuid.uuid4().hex[:8]}",
                status="filled",
                side=side_str,
                order_type=type_str,
                price=exec_price,
                quantity=quantity,
                filled_quantity=quantity,
                average_price=exec_price,
                exchange="coindcx",
                trading_mode="spot",
                pair=pair,
                raw={"demo": True, "message": "Demo paper spot order executed successfully"}
            )
        return await self.spot_trading.place_order(pair, side, order_type, quantity, price)
    
    async def cancel_spot_order(self, order_id: str) -> bool:
        if self.is_demo:
            return True
        return await self.spot_trading.cancel_order(order_id)
    
    async def get_spot_order_status(self, order_id: str) -> OrderResult:
        return await self.spot_trading.get_order_status(order_id)
    
    async def get_active_spot_orders(self, pair: Optional[str] = None) -> list[OrderResult]:
        return await self.spot_trading.get_active_orders(pair)
    
    # === MARGIN TRADING (delegates to CoinDCXMargin) ===
    
    async def place_margin_order(
        self, pair: str, side: OrderSide, order_type: OrderType,
        quantity: float, price: Optional[float] = None,
        leverage: float = 1.0, stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None, trailing_sl: bool = False,
    ) -> OrderResult:
        if self.is_demo:
            import uuid
            side_str = side.value if hasattr(side, 'value') else str(side)
            type_str = order_type.value if hasattr(order_type, 'value') else str(order_type)
            exec_price = price or 79500.0
            return OrderResult(
                order_id=f"demo_margin_{uuid.uuid4().hex[:8]}",
                status="filled",
                side=side_str,
                order_type=type_str,
                price=exec_price,
                quantity=quantity,
                filled_quantity=quantity,
                average_price=exec_price,
                exchange="coindcx",
                trading_mode="margin",
                pair=pair,
                raw={"demo": True, "message": "Demo paper margin order executed successfully"}
            )
        return await self.margin_trading.place_order(
            pair, side, order_type, quantity, price,
            leverage, stop_loss, take_profit, trailing_sl,
        )
    
    async def edit_margin_stop_loss(self, order_id: str, stop_loss: float) -> bool:
        return await self.margin_trading.edit_stop_loss(order_id, stop_loss)
    
    async def edit_margin_take_profit(self, order_id: str, take_profit: float) -> bool:
        return await self.margin_trading.edit_take_profit(order_id, take_profit)
    
    async def exit_margin_position(self, order_id: str) -> bool:
        return await self.margin_trading.exit_position(order_id)
    
    async def cancel_margin_order(self, order_id: str) -> bool:
        return await self.margin_trading.cancel_order(order_id)
    
    async def get_margin_orders(self, pair: Optional[str] = None) -> list[OrderResult]:
        return await self.margin_trading.fetch_orders(pair)
    
    async def add_margin(self, order_id: str, amount: float) -> bool:
        return await self.margin_trading.add_margin(order_id, amount)
    
    async def remove_margin(self, order_id: str, amount: float) -> bool:
        return await self.margin_trading.remove_margin(order_id, amount)
    
    # === FUTURES TRADING (delegates to CoinDCXFutures) ===
    
    async def place_futures_order(
        self, pair: str, side: OrderSide, order_type: OrderType,
        quantity: float, price: Optional[float] = None,
        leverage: float = 1.0, stop_price: Optional[float] = None,
        margin_type: MarginType = MarginType.ISOLATED,
    ) -> OrderResult:
        if self.is_demo:
            import uuid
            side_str = side.value if hasattr(side, 'value') else str(side)
            type_str = order_type.value if hasattr(order_type, 'value') else str(order_type)
            exec_price = price or 79500.0
            margin_req = (quantity * exec_price) / max(1.0, leverage)
            
            # Deduct margin from available USDT
            if 'USDT' in self._demo_balances:
                avail = self._demo_balances['USDT'].available
                self._demo_balances['USDT'].available = max(0.0, avail - margin_req)
                self._demo_balances['USDT'].locked += margin_req
                self._demo_balances['USDT'].total = self._demo_balances['USDT'].available + self._demo_balances['USDT'].locked
            
            # Add to demo positions
            pos_id = f"pos_demo_{uuid.uuid4().hex[:6]}"
            is_buy = side_str.lower() in ('buy', 'long')
            liq_factor = 0.8 / max(1.0, leverage)
            liq_price = exec_price * (1 - liq_factor) if is_buy else exec_price * (1 + liq_factor)
            
            new_pos = Position(
                id=pos_id,
                pair=pair,
                side="long" if is_buy else "short",
                quantity=quantity,
                entry_price=exec_price,
                mark_price=exec_price,
                leverage=leverage,
                unrealized_pnl=0.0,
                liquidation_price=round(liq_price, 2),
                take_profit=0.0,
                stop_loss=stop_price or 0.0,
                timestamp=int(time.time() * 1000),
            )
            self._demo_positions.insert(0, new_pos)
            
            return OrderResult(
                order_id=f"demo_ord_{uuid.uuid4().hex[:8]}",
                status="filled",
                side=side_str,
                order_type=type_str,
                price=exec_price,
                quantity=quantity,
                filled_quantity=quantity,
                average_price=exec_price,
                exchange="coindcx",
                trading_mode="futures",
                pair=pair,
                raw={"demo": True, "message": "Demo paper order executed successfully"}
            )
        return await self.futures_trading.place_order(
            pair, side, order_type, quantity, price,
            leverage, stop_price, margin_type,
        )
    
    async def cancel_futures_order(self, order_id: str) -> bool:
        if self.is_demo:
            return True
        return await self.futures_trading.cancel_order(order_id)
    
    async def get_futures_positions(self) -> list[Position]:
        if self.is_demo:
            # Update mark price and unrealized pnl from live ticker if available
            for p in self._demo_positions:
                try:
                    t = await self.get_ticker(p.pair)
                    if t and t.last_price > 0:
                        p.mark_price = t.last_price
                        diff = (p.mark_price - p.entry_price) if p.side == "long" else (p.entry_price - p.mark_price)
                        p.unrealized_pnl = round(diff * p.quantity, 2)
                except Exception:
                    pass
            return list(self._demo_positions)
        return await self.futures_trading.get_positions()
    
    async def set_futures_leverage(self, pair: str, leverage: float) -> bool:
        if self.is_demo:
            return True
        return await self.futures_trading.set_leverage(pair, leverage)
    
    async def set_futures_tp_sl(
        self, position_id: str, take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
    ) -> bool:
        if self.is_demo:
            target = next((p for p in self._demo_positions if p.id == position_id), None)
            if target:
                if take_profit:
                    target.take_profit = take_profit
                if stop_loss:
                    target.stop_loss = stop_loss
            return True
        return await self.futures_trading.set_tp_sl(position_id, take_profit, stop_loss)
    
    async def exit_futures_position(self, position_id: str) -> bool:
        if self.is_demo:
            target = next((p for p in self._demo_positions if p.id == position_id), None)
            if target:
                self._demo_positions.remove(target)
                # Release locked margin back
                margin_req = (target.quantity * target.entry_price) / max(1.0, target.leverage)
                if 'USDT' in self._demo_balances:
                    self._demo_balances['USDT'].locked = max(0.0, self._demo_balances['USDT'].locked - margin_req)
                    self._demo_balances['USDT'].available += margin_req + target.unrealized_pnl
                    self._demo_balances['USDT'].total = self._demo_balances['USDT'].available + self._demo_balances['USDT'].locked
            return True
        return await self.futures_trading.exit_position(position_id)
    
    async def add_futures_margin(self, position_id: str, amount: float) -> bool:
        return await self.futures_trading.add_margin(position_id, amount)
    
    async def transfer_funds(
        self, amount: float, currency: str = 'USDT', to_futures: bool = True
    ) -> bool:
        return await self.futures_trading.transfer_funds(amount, currency, to_futures)
