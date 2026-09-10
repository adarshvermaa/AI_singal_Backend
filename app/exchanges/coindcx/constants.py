# Base URLs
BASE_URL = "https://api.coindcx.com"
PUBLIC_URL = "https://public.coindcx.com"
SPOT_WS_URL = "wss://stream-spot.coindcx.com"
FUTURES_WS_URL = "wss://stream.coindcx.com"

# Public endpoints
TICKER_URL = f"{BASE_URL}/exchange/ticker"
MARKETS_URL = f"{BASE_URL}/exchange/v1/markets"
MARKETS_DETAILS_URL = f"{BASE_URL}/exchange/v1/markets_details"
CANDLES_URL = f"{BASE_URL}/market_data/candles"
ORDERBOOK_URL = f"{BASE_URL}/market_data/orderbook"
TRADE_HISTORY_URL = f"{BASE_URL}/market_data/trade_history"

# Futures public endpoints
FUTURES_INSTRUMENTS_URL = f"{BASE_URL}/exchange/v1/derivatives/futures/data/active_instruments"
FUTURES_INSTRUMENT_URL = f"{BASE_URL}/exchange/v1/derivatives/futures/data/instrument"
FUTURES_CANDLES_URL = f"{PUBLIC_URL}/market_data/candlesticks"
FUTURES_ORDERBOOK_URL = f"{PUBLIC_URL}/market_data/v3/orderbook"
FUTURES_RT_PRICES_URL = f"{PUBLIC_URL}/market_data/v3/current_prices/futures/rt"
FUTURES_TRADES_URL = f"{BASE_URL}/exchange/v1/derivatives/futures/data/trades"
FUTURES_STATS_URL = f"{BASE_URL}/api/v1/derivatives/futures/data/stats"

# Authenticated spot endpoints
BALANCES_URL = f"{BASE_URL}/exchange/v1/users/balances"
USER_INFO_URL = f"{BASE_URL}/exchange/v1/users/info"
SPOT_CREATE_ORDER_URL = f"{BASE_URL}/exchange/v1/orders/create"
SPOT_CANCEL_ORDER_URL = f"{BASE_URL}/exchange/v1/orders/cancel"
SPOT_ORDER_STATUS_URL = f"{BASE_URL}/exchange/v1/orders/status"
SPOT_ACTIVE_ORDERS_URL = f"{BASE_URL}/exchange/v1/orders/active_orders"
SPOT_CANCEL_ALL_URL = f"{BASE_URL}/exchange/v1/orders/cancel_all"
SPOT_EDIT_PRICE_URL = f"{BASE_URL}/exchange/v1/orders/edit"
SPOT_TRADE_HISTORY_URL = f"{BASE_URL}/exchange/v1/orders/trade_history"

# Authenticated margin endpoints
MARGIN_CREATE_URL = f"{BASE_URL}/exchange/v1/margin/create"
MARGIN_CANCEL_URL = f"{BASE_URL}/exchange/v1/margin/cancel"
MARGIN_EXIT_URL = f"{BASE_URL}/exchange/v1/margin/exit"
MARGIN_EDIT_SL_URL = f"{BASE_URL}/exchange/v1/margin/edit_sl"
MARGIN_EDIT_TARGET_URL = f"{BASE_URL}/exchange/v1/margin/edit_target"
MARGIN_ADD_MARGIN_URL = f"{BASE_URL}/exchange/v1/margin/add_margin"
MARGIN_REMOVE_MARGIN_URL = f"{BASE_URL}/exchange/v1/margin/remove_margin"
MARGIN_FETCH_ORDERS_URL = f"{BASE_URL}/exchange/v1/margin/fetch_orders"

# Authenticated futures endpoints
FUTURES_CREATE_ORDER_URL = f"{BASE_URL}/exchange/v1/derivatives/futures/orders/create"
FUTURES_CANCEL_ORDER_URL = f"{BASE_URL}/exchange/v1/derivatives/futures/orders/cancel"
FUTURES_CANCEL_ALL_URL = f"{BASE_URL}/exchange/v1/derivatives/futures/orders/cancel_all"
FUTURES_EDIT_ORDER_URL = f"{BASE_URL}/exchange/v1/derivatives/futures/orders/edit"
FUTURES_POSITIONS_URL = f"{BASE_URL}/exchange/v1/derivatives/futures/positions"
FUTURES_UPDATE_LEVERAGE_URL = f"{BASE_URL}/exchange/v1/derivatives/futures/positions/update_leverage"
FUTURES_EXIT_POSITION_URL = f"{BASE_URL}/exchange/v1/derivatives/futures/positions/exit"
FUTURES_ADD_MARGIN_URL = f"{BASE_URL}/exchange/v1/derivatives/futures/positions/add_margin"
FUTURES_REMOVE_MARGIN_URL = f"{BASE_URL}/exchange/v1/derivatives/futures/positions/remove_margin"
FUTURES_CREATE_TPSL_URL = f"{BASE_URL}/exchange/v1/derivatives/futures/positions/create_tpsl"
FUTURES_WALLET_TRANSFER_URL = f"{BASE_URL}/exchange/v1/derivatives/futures/wallets/transfer"
FUTURES_ACTIVE_ORDERS_URL = f"{BASE_URL}/exchange/v1/derivatives/futures/orders/active_orders"
FUTURES_EXECUTED_ORDERS_URL = f"{BASE_URL}/exchange/v1/derivatives/futures/orders/executed_orders"
FUTURES_CROSS_MARGIN_URL = f"{BASE_URL}/exchange/v1/derivatives/futures/positions/cross_margin"

# Rate limits (requests per 60 seconds)
RATE_LIMIT_CREATE_ORDER = 2000
RATE_LIMIT_CANCEL_ALL = 30
RATE_LIMIT_EDIT_ORDER = 1800
RATE_LIMIT_DEFAULT = 1000

# Interval mappings
# CoinDCX Spot candle intervals
SPOT_INTERVALS = {
    '1m': '1m',
    '5m': '5m',
    '15m': '15m',
    '30m': '30m',
    '1h': '1h',
    '2h': '2h',
    '4h': '4h',
    '1d': '1d',
    '1w': '1w',
}

# CoinDCX Futures candle resolutions
FUTURES_INTERVALS = {
    '1m': '1',
    '5m': '5',
    '15m': '15',
    '30m': '30',
    '1h': '60',
    '2h': '120',
    '4h': '240',
    '1d': '1D',
    '1w': '1W',
}

# Default ecode for margin
DEFAULT_ECODE = 'B'
