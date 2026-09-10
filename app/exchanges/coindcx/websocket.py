import logging
import asyncio
from typing import Callable, Optional
import socketio

from app.exchanges.coindcx.auth import CoinDCXAuth
from app.exchanges.coindcx import constants as C

logger = logging.getLogger(__name__)


class CoinDCXWebSocket:
    """
    CoinDCX real-time WebSocket handler using socket.io.
    
    Spot socket: wss://stream-spot.coindcx.com
    Futures socket: wss://stream.coindcx.com
    
    Public channels (no auth):
      - {pair}_{interval}: candlestick updates
      - {pair}@orderbook@{depth}: order book snapshots/updates
      - {pair}@trades: new trades
      - currentPrices@spot@1s: all spot prices every 1s
    
    Private channels (auth required):
      - coindcx: balance-update, order-update, trade-update
      
    Futures channels:
      - {pair}_{resolution}@futures: candlestick updates
      - {pair}@orderbook@{depth}@futures: order book
      - {pair}@trades@futures: new trades
      - coindcx: df-position-update, df-order-update, df-balance-update
    """
    
    def __init__(self, auth: Optional[CoinDCXAuth] = None):
        self.auth = auth
        self._spot_sio: Optional[socketio.AsyncClient] = None
        self._futures_sio: Optional[socketio.AsyncClient] = None
        self._callbacks: dict[str, list[Callable]] = {}
        self._connected_spot = False
        self._connected_futures = False
        self._ping_task: Optional[asyncio.Task] = None
    
    # === CONNECTION ===
    
    async def connect_spot(self) -> None:
        """Connect to CoinDCX spot WebSocket."""
        self._spot_sio = socketio.AsyncClient(
            reconnection=True,
            reconnection_attempts=10,
            reconnection_delay=1,
            logger=False,
        )
        self._setup_spot_handlers()
        await self._spot_sio.connect(
            C.SPOT_WS_URL,
            transports=['websocket'],
        )
        self._connected_spot = True
        logger.info("Connected to CoinDCX Spot WebSocket")
    
    async def connect_futures(self) -> None:
        """Connect to CoinDCX futures WebSocket."""
        self._futures_sio = socketio.AsyncClient(
            reconnection=True,
            reconnection_attempts=10,
            reconnection_delay=1,
            logger=False,
        )
        self._setup_futures_handlers()
        await self._futures_sio.connect(
            C.FUTURES_WS_URL,
            transports=['websocket'],
        )
        self._connected_futures = True
        logger.info("Connected to CoinDCX Futures WebSocket")
    
    async def disconnect(self) -> None:
        """Disconnect all WebSocket connections."""
        if self._ping_task:
            self._ping_task.cancel()
        if self._spot_sio and self._connected_spot:
            await self._spot_sio.disconnect()
            self._connected_spot = False
        if self._futures_sio and self._connected_futures:
            await self._futures_sio.disconnect()
            self._connected_futures = False
        logger.info("Disconnected from CoinDCX WebSockets")
    
    # === SPOT SUBSCRIPTIONS ===
    
    async def subscribe_spot_candles(
        self, pair: str, interval: str, callback: Callable
    ) -> None:
        """Subscribe to spot candlestick updates."""
        if not self._connected_spot:
            await self.connect_spot()
        channel = f"{pair}_{interval}"
        self._register_callback('candlestick', callback)
        await self._spot_sio.emit('join', {'channelName': channel})
        logger.info(f"Subscribed to spot candles: {channel}")
    
    async def subscribe_spot_orderbook(
        self, pair: str, depth: int = 20, callback: Callable = None
    ) -> None:
        """Subscribe to spot order book updates."""
        if not self._connected_spot:
            await self.connect_spot()
        channel = f"{pair}@orderbook@{depth}"
        if callback:
            self._register_callback('depth-snapshot', callback)
            self._register_callback('depth-update', callback)
        await self._spot_sio.emit('join', {'channelName': channel})
        logger.info(f"Subscribed to spot orderbook: {channel}")
    
    async def subscribe_spot_trades(
        self, pair: str, callback: Callable = None
    ) -> None:
        """Subscribe to spot new trades."""
        if not self._connected_spot:
            await self.connect_spot()
        channel = f"{pair}@trades"
        if callback:
            self._register_callback('new-trade', callback)
        await self._spot_sio.emit('join', {'channelName': channel})
        logger.info(f"Subscribed to spot trades: {channel}")
    
    async def subscribe_spot_prices(
        self, callback: Callable = None
    ) -> None:
        """Subscribe to all spot prices (every 1s or 10s)."""
        if not self._connected_spot:
            await self.connect_spot()
        channel = 'currentPrices@spot@1s'
        if callback:
            self._register_callback('currentPrices@spot#update', callback)
        await self._spot_sio.emit('join', {'channelName': channel})
        logger.info("Subscribed to spot prices")
    
    # === FUTURES SUBSCRIPTIONS ===
    
    async def subscribe_futures_candles(
        self, pair: str, resolution: str, callback: Callable = None
    ) -> None:
        """Subscribe to futures candlestick updates."""
        if not self._connected_futures:
            await self.connect_futures()
        channel = f"{pair}_{resolution}@futures"
        if callback:
            self._register_callback('candlestick', callback)
        await self._futures_sio.emit('join', {'channelName': channel})
        logger.info(f"Subscribed to futures candles: {channel}")
    
    async def subscribe_futures_orderbook(
        self, pair: str, depth: int = 20, callback: Callable = None
    ) -> None:
        """Subscribe to futures order book."""
        if not self._connected_futures:
            await self.connect_futures()
        channel = f"{pair}@orderbook@{depth}@futures"
        if callback:
            self._register_callback('depth-snapshot', callback)
            self._register_callback('depth-update', callback)
        await self._futures_sio.emit('join', {'channelName': channel})
        logger.info(f"Subscribed to futures orderbook: {channel}")
    
    async def subscribe_futures_trades(
        self, pair: str, callback: Callable = None
    ) -> None:
        """Subscribe to futures new trades."""
        if not self._connected_futures:
            await self.connect_futures()
        channel = f"{pair}@trades@futures"
        if callback:
            self._register_callback('new-trade', callback)
        await self._futures_sio.emit('join', {'channelName': channel})
        logger.info(f"Subscribed to futures trades: {channel}")
    
    # === PRIVATE CHANNELS (Auth Required) ===
    
    async def subscribe_private(
        self,
        on_order_update: Optional[Callable] = None,
        on_trade_update: Optional[Callable] = None,
        on_balance_update: Optional[Callable] = None,
        on_position_update: Optional[Callable] = None,
    ) -> None:
        """Subscribe to private channels (requires auth)."""
        if not self.auth:
            raise ValueError("Auth required for private channels")
        
        # Connect both if needed
        if not self._connected_spot:
            await self.connect_spot()
        
        # Register callbacks
        if on_order_update:
            self._register_callback('order-update', on_order_update)
            self._register_callback('df-order-update', on_order_update)
        if on_trade_update:
            self._register_callback('trade-update', on_trade_update)
            self._register_callback('df-trade-update', on_trade_update)
        if on_balance_update:
            self._register_callback('balance-update', on_balance_update)
            self._register_callback('df-balance-update', on_balance_update)
        if on_position_update:
            self._register_callback('df-position-update', on_position_update)
        
        # Auth and join private channel
        auth_payload = {
            'channelName': 'coindcx',
            'authSignature': self.auth.sign({}),
            'apiKey': self.auth.api_key,
        }
        await self._spot_sio.emit('join', auth_payload)
        # Also authenticate on futures socket for position/order updates
        if not self._connected_futures:
            await self.connect_futures()
        await self._futures_sio.emit('join', auth_payload)
        logger.info("Subscribed to spot and futures private channels")
        
        # Start ping to keep alive (every 25 seconds)
        self._ping_task = asyncio.create_task(self._ping_loop())
    
    # === INTERNAL ===
    
    def _register_callback(self, event: str, callback: Callable) -> None:
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)
    
    async def _dispatch(self, event: str, data) -> None:
        """Dispatch event to registered callbacks."""
        callbacks = self._callbacks.get(event, [])
        for cb in callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(data)
                else:
                    cb(data)
            except Exception as e:
                logger.error(f"WebSocket callback error for {event}: {e}")
    
    def _setup_spot_handlers(self) -> None:
        """Register socket.io event handlers for spot."""
        events = [
            'candlestick', 'depth-snapshot', 'depth-update',
            'new-trade', 'currentPrices@spot#update',
            'order-update', 'trade-update', 'balance-update',
        ]
        for event in events:
            self._spot_sio.on(event, lambda data, e=event: asyncio.create_task(self._dispatch(e, data)))
        self._spot_sio.on('connect', lambda: logger.info('Spot WS connected'))
        self._spot_sio.on('disconnect', lambda: [logger.info('Spot WS disconnected'), setattr(self, '_connected_spot', False)])
    
    def _setup_futures_handlers(self) -> None:
        """Register socket.io event handlers for futures."""
        events = [
            'candlestick', 'depth-snapshot', 'depth-update',
            'new-trade', 'currentPrices@futures#update',
            'df-position-update', 'df-order-update',
            'df-trade-update', 'df-balance-update',
        ]
        for event in events:
            self._futures_sio.on(event, lambda data, e=event: asyncio.create_task(self._dispatch(e, data)))
        self._futures_sio.on('connect', lambda: logger.info('Futures WS connected'))
        self._futures_sio.on('disconnect', lambda: [logger.info('Futures WS disconnected'), setattr(self, '_connected_futures', False)])
    
    async def _ping_loop(self) -> None:
        """Send ping every 25 seconds to keep connection alive."""
        while True:
            try:
                await asyncio.sleep(25)
                if self._connected_spot and self._spot_sio:
                    await self._spot_sio.emit('ping')
                if self._connected_futures and self._futures_sio:
                    await self._futures_sio.emit('ping')
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Ping error: {e}")
