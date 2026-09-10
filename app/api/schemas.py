from pydantic import BaseModel, Field, model_validator
from typing import Optional, Any
from enum import Enum


# === Enums ===

class TradingModeEnum(str, Enum):
    SPOT = "spot"
    MARGIN = "margin"
    FUTURES = "futures"


class SignalDirection(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


# === Analysis ===

class AnalyzeRequest(BaseModel):
    """Request to analyze a trading pair."""
    symbol: str = Field(..., description="Trading pair e.g. B-BTC_USDT", examples=["B-BTC_USDT"])
    timeframe: str = Field(default="15m", description="Chart timeframe", examples=["1m", "5m", "15m", "1h", "4h", "1d"])
    exchange: str = Field(default="coindcx", description="Exchange to fetch data from")
    trading_mode: TradingModeEnum = Field(default=TradingModeEnum.FUTURES, description="Trading mode")


class IndicatorValues(BaseModel):
    """Current indicator values for display."""
    rsi_14: Optional[float] = None
    rsi_7: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    ema_9: Optional[float] = None
    ema_21: Optional[float] = None
    ema_50: Optional[float] = None
    ema_200: Optional[float] = None
    adx: Optional[float] = None
    atr_14: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_percent: Optional[float] = None
    stoch_k: Optional[float] = None
    stoch_d: Optional[float] = None
    obv: Optional[float] = None
    vwap: Optional[float] = None
    cmf: Optional[float] = None
    mfi: Optional[float] = None
    volume_sma_ratio: Optional[float] = None
    supertrend: Optional[float] = None
    supertrend_direction: Optional[int] = None


class ChartLevel(BaseModel):
    """A horizontal level to draw on chart."""
    price: float
    label: str
    color: str
    line_style: str = "solid"  # solid, dashed, dotted
    line_width: int = 1


class ChartPattern(BaseModel):
    """A pattern detected on chart."""
    name: str  # e.g., "Bullish Engulfing", "Double Bottom"
    start_index: int
    end_index: int
    confidence: float  # 0-1
    direction: str  # "bullish" or "bearish"


class ChartZone(BaseModel):
    """A zone (price range) to draw on chart."""
    upper: float
    lower: float
    label: str
    color: str
    opacity: float = 0.2


class ChartOverlays(BaseModel):
    """All overlays to draw on the TradingView chart."""
    support_levels: list[ChartLevel] = []
    resistance_levels: list[ChartLevel] = []
    entry_line: Optional[ChartLevel] = None
    stop_loss_line: Optional[ChartLevel] = None
    take_profit_lines: list[ChartLevel] = []
    patterns: list[ChartPattern] = []
    fvg_zones: list[ChartZone] = []  # Fair Value Gaps
    supply_demand_zones: list[ChartZone] = []


class ReasoningItem(BaseModel):
    """A single reasoning item explaining WHY."""
    category: str  # e.g., "Trend", "Momentum", "Volume"
    signal: str  # "bullish", "bearish", "neutral"
    strength: float  # 0-1
    description: str  # Human-readable explanation
    indicators: list[str] = []  # Which indicators support this


class TradeLevels(BaseModel):
    """Entry, SL, TP levels for the trade."""
    entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: Optional[float] = None
    take_profit_3: Optional[float] = None
    risk_reward_ratio: float
    position_size_suggestion: Optional[float] = None  # in quote currency
    leverage_suggestion: Optional[float] = None


class ModelVote(BaseModel):
    """Individual model prediction."""
    model_name: str
    direction: SignalDirection
    buy_prob: float
    hold_prob: float
    sell_prob: float


class SignalResult(BaseModel):
    """The AI signal result."""
    direction: SignalDirection
    confidence: float = Field(..., ge=0, le=1)
    model_agreement: str  # "UNANIMOUS", "MAJORITY", "SPLIT"
    model_votes: list[ModelVote] = []


class AnalyzeResponse(BaseModel):
    """Complete analysis response sent to frontend."""
    symbol: str
    timeframe: str
    exchange: str
    timestamp: int
    current_price: float
    
    # Signal
    signal: SignalResult
    
    # Levels
    levels: Optional[TradeLevels] = None
    
    # Reasoning
    reasoning: list[ReasoningItem] = []
    
    # Chart overlays
    overlays: ChartOverlays = ChartOverlays()
    
    # Indicator values for display
    indicators: IndicatorValues = IndicatorValues()
    
    # Market context
    market_regime: str = "unknown"  # "trending", "ranging", "volatile"
    funding_rate: Optional[float] = None
    long_short_ratio: Optional[float] = None
    volume_24h: Optional[float] = None


# === Exchange ===

class ConnectExchangeRequest(BaseModel):
    """Request to connect an exchange."""
    exchange: str = Field(..., description="Exchange name", examples=["coindcx"])
    api_key: str = Field(..., description="API key")
    api_secret: str = Field(..., description="API secret")


class ConnectExchangeResponse(BaseModel):
    """Response after connecting exchange."""
    exchange: str
    status: str  # "connected", "failed"
    message: str
    user_name: Optional[str] = None
    is_demo: bool = False


class BalanceItem(BaseModel):
    """Single currency balance."""
    currency: str
    available: float
    locked: float
    total: float


class BalancesResponse(BaseModel):
    """Wallet balances response."""
    exchange: str
    balances: list[BalanceItem]
    total_usdt_value: Optional[float] = None
    authenticated: bool = True
    message: Optional[str] = None


class PositionItem(BaseModel):
    """Single position."""
    id: str
    pair: str
    side: str
    quantity: float
    entry_price: float
    mark_price: float
    leverage: float
    unrealized_pnl: float
    liquidation_price: float
    take_profit: float = 0
    stop_loss: float = 0


class PositionsResponse(BaseModel):
    """Active positions response."""
    exchange: str
    positions: list[PositionItem]
    authenticated: bool = True
    message: Optional[str] = None


# === Trading ===

class ExecuteTradeRequest(BaseModel):
    """Request to execute a trade based on signal."""
    exchange: str = Field(default="coindcx")
    symbol: str = Field(..., description="Trading pair", examples=["B-BTC_USDT"])
    side: str = Field(..., description="buy or sell")
    trading_mode: TradingModeEnum = Field(default=TradingModeEnum.MARGIN)
    order_type: str = Field(default="market_order")
    quantity: Optional[float] = None  # If None, auto-calculate
    price: Optional[float] = None  # For limit orders
    leverage: float = Field(default=3.0, ge=1, le=25)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    auto_size: bool = Field(default=True, description="Auto-calculate position size based on risk")


class ExecuteTradeResponse(BaseModel):
    """Response after trade execution."""
    status: str  # "executed", "rejected", "error"
    order_id: Optional[str] = None
    exchange: str
    trading_mode: str
    side: str
    quantity: float
    price: Optional[float] = None
    leverage: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    message: str = ""


class ClosePositionRequest(BaseModel):
    """Request to close a position."""
    exchange: str = Field(default="coindcx")
    position_id: str
    trading_mode: TradingModeEnum = Field(default=TradingModeEnum.FUTURES)


# === Markets ===

class MarketItem(BaseModel):
    """Single market/trading pair info."""
    pair: str
    base_currency: str
    quote_currency: str
    last_price: Optional[float] = None
    change_24h: Optional[float] = None
    volume_24h: Optional[float] = None
    max_leverage: float = 1.0
    trading_modes: list[str] = []
    is_active: bool = True


class MarketsResponse(BaseModel):
    """Available markets response."""
    exchange: str
    count: int
    markets: list[MarketItem]


# === Health ===

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str = "1.0.0"
    exchange_connections: list[str] = []
    models_loaded: bool = False


# === Error ===

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: Optional[str] = None
    code: int = 400


# === Webhook ===

class WebhookTradePayload(BaseModel):
    """
    Flexible payload received from TradingView, CoinDCX webhook alerts, or external bots.
    Accepts both standard format and CoinDCX native callback format.
    """
    secret: Optional[str] = Field(default=None, description="Webhook authorization token")
    webhook_id: Optional[str] = Field(default=None, description="CoinDCX official webhook ID")
    symbol: Optional[str] = Field(default="B-BTC_USDT", description="Pair e.g. B-BTC_USDT or BTCUSDT")
    pair: Optional[str] = Field(default=None, description="CoinDCX alias for symbol")
    action: Optional[str] = Field(default="buy", description="buy, sell, close")
    side: Optional[str] = Field(default=None, description="CoinDCX alias for action")
    trading_mode: TradingModeEnum = Field(default=TradingModeEnum.FUTURES)
    order_type: str = Field(default="market_order")
    quantity: Optional[float] = None
    total_quantity: Optional[Any] = Field(default=None, description="CoinDCX alias for quantity")
    price: Optional[Any] = None
    leverage: Optional[Any] = Field(default=3.0)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    auto_risk: bool = Field(default=True, description="Auto-calculate size based on 2% risk")
    require_ai_confirmation: Optional[bool] = Field(default=None, description="Require AI ensemble to agree")
    margin_currency_short_name: Optional[str] = Field(default="USDT", description="INR or USDT margin")
    order: Optional[dict] = Field(default=None, description="Nested order object from CoinDCX/TradingView")

    @model_validator(mode="before")
    @classmethod
    def normalize_payload(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        # 1. Unpack nested order if provided (e.g. {"order": {"pair": "...", "side": "..."}})
        order_dict = data.get("order")
        if isinstance(order_dict, dict):
            for k, v in order_dict.items():
                if k not in data or data[k] is None:
                    data[k] = v

        # 2. Webhook ID / Secret mapping
        if ("secret" not in data or not data["secret"]) and "webhook_id" in data and data["webhook_id"]:
            data["secret"] = data["webhook_id"]

        # 3. Pair -> Symbol
        if ("symbol" not in data or not data["symbol"]) and "pair" in data and data["pair"]:
            data["symbol"] = data["pair"]

        # 4. Side -> Action
        if ("action" not in data or not data["action"]) and "side" in data and data["side"]:
            data["action"] = data["side"]

        # 5. Quantity / Total Quantity string or float parsing
        raw_qty = data.get("quantity") if data.get("quantity") is not None else data.get("total_quantity")
        if raw_qty is not None:
            if isinstance(raw_qty, (int, float)):
                data["quantity"] = float(raw_qty)
            elif isinstance(raw_qty, str):
                try:
                    data["quantity"] = float(raw_qty.strip())
                except (ValueError, TypeError):
                    # Placeholder string like "Enter total quantity" or empty
                    data["quantity"] = None

        # 6. Price string or float parsing
        raw_price = data.get("price")
        if raw_price is not None:
            if isinstance(raw_price, (int, float)):
                data["price"] = float(raw_price)
            elif isinstance(raw_price, str):
                try:
                    data["price"] = float(raw_price.strip())
                except (ValueError, TypeError):
                    data["price"] = None

        # 7. Leverage string or float parsing
        raw_lev = data.get("leverage")
        if raw_lev is not None:
            if isinstance(raw_lev, (int, float)):
                data["leverage"] = float(raw_lev)
            elif isinstance(raw_lev, str):
                try:
                    data["leverage"] = float(raw_lev.strip())
                except (ValueError, TypeError):
                    data["leverage"] = 3.0

        # 8. Order type normalization
        ot = data.get("order_type")
        if ot and isinstance(ot, str):
            ot_lower = ot.strip().lower()
            if "limit" in ot_lower and "market" not in ot_lower:
                data["order_type"] = "limit_order"
            else:
                data["order_type"] = "market_order"

        # 9. Normalize symbol name (e.g. BTCUSDT -> B-BTC_USDT)
        sym = data.get("symbol")
        if sym and isinstance(sym, str):
            sym = sym.strip().upper()
            if not sym.startswith("B-") and "_" not in sym:
                if sym.endswith("USDT"):
                    sym = f"B-{sym[:-4]}_USDT"
                elif sym.endswith("INR"):
                    sym = f"B-{sym[:-3]}_INR"
            data["symbol"] = sym

        return data


class WebhookLogItem(BaseModel):
    """In-memory record of a processed webhook alert."""
    id: str
    timestamp: int
    symbol: str
    action: str
    trading_mode: str
    status: str  # "executed", "rejected_by_ai", "invalid_secret", "failed"
    ai_verdict: Optional[str] = None
    ai_confidence: Optional[float] = None
    order_id: Optional[str] = None
    message: str


class WebhookConfigResponse(BaseModel):
    """Webhook connection and template configuration."""
    webhook_url: str
    coindcx_webhook_url: Optional[str] = None
    secret_key: str
    ai_filter_enabled: bool
    templates: dict[str, dict]


