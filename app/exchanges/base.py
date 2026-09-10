from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    LIMIT = "limit_order"
    MARKET = "market_order"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"
    STOP_MARKET = "stop_market"
    TAKE_PROFIT_MARKET = "take_profit_market"


class TradingMode(str, Enum):
    SPOT = "spot"
    MARGIN = "margin"
    FUTURES = "futures"


class MarginType(str, Enum):
    ISOLATED = "isolated"
    CROSS = "cross"


@dataclass
class Candle:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class OrderBookEntry:
    price: float
    quantity: float

    def __getitem__(self, idx: int) -> float:
        if idx == 0:
            return self.price
        elif idx == 1:
            return self.quantity
        raise IndexError(f"OrderBookEntry index out of range: {idx}")

    def __iter__(self):
        yield self.price
        yield self.quantity


@dataclass
class OrderBook:
    bids: list[OrderBookEntry]
    asks: list[OrderBookEntry]
    timestamp: int = 0


@dataclass
class Trade:
    price: float
    quantity: float
    is_buyer_maker: bool  # True = seller aggressive, False = buyer aggressive
    timestamp: int = 0


@dataclass
class Ticker:
    pair: str
    last_price: float
    high_24h: float
    low_24h: float
    volume_24h: float
    change_24h: float
    bid: float = 0.0
    ask: float = 0.0
    timestamp: int = 0


@dataclass
class MarketInfo:
    pair: str  # e.g., "B-BTC_USDT"
    base_currency: str  # e.g., "BTC"
    quote_currency: str  # e.g., "USDT"
    min_quantity: float
    max_quantity: float
    min_price: float
    max_price: float
    min_notional: float
    step_size: float  # quantity precision
    tick_size: float  # price precision
    max_leverage: float
    trading_modes: list[TradingMode] = field(default_factory=list)
    is_active: bool = True


@dataclass
class OrderResult:
    order_id: str
    status: str
    side: str
    order_type: str
    price: float
    quantity: float
    filled_quantity: float = 0.0
    average_price: float = 0.0
    exchange: str = ""
    trading_mode: str = ""
    pair: str = ""
    timestamp: int = 0
    raw: dict = field(default_factory=dict)
    raw_response: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.order_id


@dataclass
class Position:
    id: str
    pair: str
    side: str  # "long" or "short"
    quantity: float
    entry_price: float
    mark_price: float
    leverage: float
    unrealized_pnl: float
    liquidation_price: float
    margin_type: str = "isolated"  # "isolated" or "cross"
    take_profit: float = 0.0
    stop_loss: float = 0.0
    timestamp: int = 0
    raw: dict = field(default_factory=dict)


@dataclass
class Balance:
    currency: str
    available: float
    locked: float
    total: float


@dataclass
class FuturesInfo:
    pair: str
    funding_rate: float
    mark_price: float
    index_price: float
    open_interest: float
    volume_24h: float
    long_short_ratio: float = 0.0
    next_funding_time: int = 0


class ExchangeError(Exception):
    """Base exception for exchange errors."""
    def __init__(self, message: str, code: int = 0, raw: dict = None):
        self.message = message
        self.code = code
        self.raw = raw or {}
        super().__init__(message)


class AuthenticationError(ExchangeError):
    pass


class InsufficientFundsError(ExchangeError):
    pass


class OrderError(ExchangeError):
    pass


class RateLimitError(ExchangeError):
    pass


class ExchangeBase(ABC):
    """
    Abstract base class for all exchange adapters.
    
    Every exchange (CoinDCX, WazirX, Binance, etc.) implements this interface.
    The AI engine, API routes, and frontend NEVER talk to exchange-specific code directly.
    
    To add a new exchange:
    1. Create a new directory: app/exchanges/newexchange/
    2. Create adapter.py that extends ExchangeBase
    3. Implement all abstract methods
    4. Register in ExchangeFactory
    """
    
    name: str = "base"
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the exchange connection (load markets, etc.)."""
        ...
    
    @abstractmethod
    async def close(self) -> None:
        """Cleanup resources (close HTTP client, disconnect websocket)."""
        ...
    
    # ==================== MARKET DATA (Public, No Auth) ====================
    
    @abstractmethod
    async def get_candles(
        self, pair: str, interval: str, limit: int = 500
    ) -> list[Candle]:
        """Fetch OHLCV candle data."""
        ...
    
    @abstractmethod
    async def get_orderbook(
        self, pair: str, depth: int = 20
    ) -> OrderBook:
        """Fetch order book (bids and asks)."""
        ...
    
    @abstractmethod
    async def get_ticker(self, pair: str) -> Ticker:
        """Fetch 24h ticker data for a pair."""
        ...
    
    @abstractmethod
    async def get_all_tickers(self) -> list[Ticker]:
        """Fetch 24h ticker data for all pairs."""
        ...
    
    @abstractmethod
    async def get_recent_trades(
        self, pair: str, limit: int = 50
    ) -> list[Trade]:
        """Fetch recent trades."""
        ...
    
    @abstractmethod
    async def get_markets(self) -> list[MarketInfo]:
        """Fetch all available trading pairs with their details."""
        ...
    
    # ==================== FUTURES DATA ====================
    
    @abstractmethod
    async def get_futures_candles(
        self, pair: str, interval: str, limit: int = 500,
        from_ts: Optional[int] = None, to_ts: Optional[int] = None
    ) -> list[Candle]:
        """Fetch futures OHLCV candle data."""
        ...
    
    @abstractmethod
    async def get_futures_orderbook(
        self, pair: str, depth: int = 20
    ) -> OrderBook:
        """Fetch futures order book."""
        ...
    
    @abstractmethod
    async def get_futures_info(self, pair: str) -> FuturesInfo:
        """Fetch futures-specific data: funding rate, mark price, OI, L/S ratio."""
        ...
    
    @abstractmethod
    async def get_futures_instruments(self) -> list[dict]:
        """List all active futures instruments."""
        ...
    
    # ==================== ACCOUNT (Authenticated) ====================
    
    @abstractmethod
    async def get_balances(self) -> list[Balance]:
        """Fetch wallet balances."""
        ...
    
    @abstractmethod
    async def get_user_info(self) -> dict:
        """Fetch user account information."""
        ...
    
    # ==================== SPOT TRADING ====================
    
    @abstractmethod
    async def place_spot_order(
        self,
        pair: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
    ) -> OrderResult:
        """Place a spot order."""
        ...
    
    @abstractmethod
    async def cancel_spot_order(self, order_id: str) -> bool:
        """Cancel a spot order."""
        ...
    
    @abstractmethod
    async def get_spot_order_status(self, order_id: str) -> OrderResult:
        """Get spot order status."""
        ...
    
    @abstractmethod
    async def get_active_spot_orders(
        self, pair: Optional[str] = None
    ) -> list[OrderResult]:
        """Get all active spot orders."""
        ...
    
    # ==================== MARGIN TRADING ====================
    
    @abstractmethod
    async def place_margin_order(
        self,
        pair: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        leverage: float = 1.0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_sl: bool = False,
    ) -> OrderResult:
        """Place a margin order with optional SL/TP."""
        ...
    
    @abstractmethod
    async def edit_margin_stop_loss(
        self, order_id: str, stop_loss: float
    ) -> bool:
        """Edit the stop loss of a margin order."""
        ...
    
    @abstractmethod
    async def edit_margin_take_profit(
        self, order_id: str, take_profit: float
    ) -> bool:
        """Edit the take profit of a margin order."""
        ...
    
    @abstractmethod
    async def exit_margin_position(self, order_id: str) -> bool:
        """Exit/close a margin position."""
        ...
    
    @abstractmethod
    async def cancel_margin_order(self, order_id: str) -> bool:
        """Cancel a margin order."""
        ...
    
    @abstractmethod
    async def get_margin_orders(
        self, pair: Optional[str] = None
    ) -> list[OrderResult]:
        """Fetch margin orders."""
        ...
    
    @abstractmethod
    async def add_margin(self, order_id: str, amount: float) -> bool:
        """Add margin to a position."""
        ...
    
    @abstractmethod
    async def remove_margin(self, order_id: str, amount: float) -> bool:
        """Remove margin from a position."""
        ...
    
    # ==================== FUTURES TRADING ====================
    
    @abstractmethod
    async def place_futures_order(
        self,
        pair: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        leverage: float = 1.0,
        stop_price: Optional[float] = None,
        margin_type: MarginType = MarginType.ISOLATED,
    ) -> OrderResult:
        """Place a futures order."""
        ...
    
    @abstractmethod
    async def cancel_futures_order(self, order_id: str) -> bool:
        """Cancel a futures order."""
        ...
    
    @abstractmethod
    async def get_futures_positions(self) -> list[Position]:
        """Get all active futures positions."""
        ...
    
    @abstractmethod
    async def set_futures_leverage(
        self, pair: str, leverage: float
    ) -> bool:
        """Set leverage for a futures pair."""
        ...
    
    @abstractmethod
    async def set_futures_tp_sl(
        self,
        position_id: str,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
    ) -> bool:
        """Set take profit and/or stop loss for a futures position."""
        ...
    
    @abstractmethod
    async def exit_futures_position(
        self, position_id: str
    ) -> bool:
        """Exit/close a futures position."""
        ...
    
    @abstractmethod
    async def add_futures_margin(
        self, position_id: str, amount: float
    ) -> bool:
        """Add margin to a futures position."""
        ...
    
    @abstractmethod
    async def transfer_funds(
        self, amount: float, currency: str, to_futures: bool = True
    ) -> bool:
        """Transfer funds between spot and futures wallet."""
        ...
