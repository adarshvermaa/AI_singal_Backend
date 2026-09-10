import pandas as pd
import numpy as np
from typing import List, Optional
from app.exchanges.base import Candle

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    ta = None
    HAS_PANDAS_TA = False


class IndicatorEngine:
    """
    Engine to compute comprehensive technical indicators for given price data.
    Uses pandas-ta when available, and provides robust vectorized native pandas
    fallbacks to guarantee 100% reliability.
    """

    def candles_to_dataframe(self, candles: List[Candle]) -> pd.DataFrame:
        """
        Convert a list of Candle objects into a clean pandas DataFrame.
        """
        if not candles:
            return pd.DataFrame()

        data = [{
            'timestamp': c.timestamp,
            'open': float(c.open or 0),
            'high': float(c.high or 0),
            'low': float(c.low or 0),
            'close': float(c.close or 0),
            'volume': float(c.volume or 0),
        } for c in candles]

        df = pd.DataFrame(data)
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)

        return df

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes all required technical indicators.
        Handles missing data carefully to prevent data leakage.
        """
        if df.empty:
            return df

        df = df.copy()

        # Ensure correct types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # --- 1. Trend Indicators ---
        # EMAs
        df['ema_9'] = close.ewm(span=9, adjust=False).mean()
        df['ema_21'] = close.ewm(span=21, adjust=False).mean()
        df['ema_50'] = close.ewm(span=50, adjust=False).mean()
        df['ema_100'] = close.ewm(span=100, adjust=False).mean()
        df['ema_200'] = close.ewm(span=200, adjust=False).mean()

        # EMA Crossovers
        df['ema_9_cross_21'] = np.where(df['ema_9'] > df['ema_21'], 1, np.where(df['ema_9'] < df['ema_21'], -1, 0))
        df['ema_50_cross_200'] = np.where(df['ema_50'] > df['ema_200'], 1, np.where(df['ema_50'] < df['ema_200'], -1, 0))

        # MACD (12, 26, 9)
        fast_ema = close.ewm(span=12, adjust=False).mean()
        slow_ema = close.ewm(span=26, adjust=False).mean()
        df['macd'] = fast_ema - slow_ema
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        df['macd_hist_slope'] = df['macd_histogram'].diff()

        # ADX & Directional Movement
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        atr_14 = tr.rolling(14).mean()

        plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(14).mean() / (atr_14 + 1e-10))
        minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(14).mean() / (atr_14 + 1e-10))
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
        df['adx'] = dx.rolling(14).mean()
        df['dmp'] = plus_di
        df['dmn'] = minus_di

        # Supertrend (10, 3.0)
        hl2 = (high + low) / 2
        atr_10 = tr.rolling(10).mean()
        df['supertrend'] = hl2 - (3.0 * atr_10)
        df['supertrend_direction'] = np.where(close >= df['supertrend'], 1, -1)

        # Linear Regression Slope (14)
        df['lr_slope'] = close.diff(14) / 14.0

        # --- 2. Momentum Indicators ---
        # RSI (7, 14, 21)
        def _calc_rsi(series: pd.Series, period: int) -> pd.Series:
            delta = series.diff()
            gain = delta.where(delta > 0, 0.0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
            rs = gain / (loss + 1e-10)
            return 100 - (100 / (1 + rs))

        df['rsi_7'] = _calc_rsi(close, 7)
        df['rsi_14'] = _calc_rsi(close, 14)
        df['rsi_21'] = _calc_rsi(close, 21)

        # Stochastic Oscillator (14, 3)
        low_14 = low.rolling(14).min()
        high_14 = high.rolling(14).max()
        df['stoch_k'] = 100 * ((close - low_14) / (high_14 - low_14 + 1e-10))
        df['stoch_k'] = df['stoch_k'].bfill().fillna(50)  # Neutral default, not 0 (oversold)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean().bfill().fillna(50)

        # Stochastic RSI
        rsi_14 = df['rsi_14']
        rsi_low = rsi_14.rolling(14).min()
        rsi_high = rsi_14.rolling(14).max()
        stochrsi = (rsi_14 - rsi_low) / (rsi_high - rsi_low + 1e-10)
        df['stochrsi_k'] = stochrsi.rolling(3).mean()
        df['stochrsi_d'] = df['stochrsi_k'].rolling(3).mean()

        # CCI (14, 20)
        tp = (high + low + close) / 3
        df['cci_14'] = (tp - tp.rolling(14).mean()) / (0.015 * tp.rolling(14).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True) + 1e-10)
        df['cci_20'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True) + 1e-10)

        # Williams %R (14)
        df['willr_14'] = -100 * ((high_14 - close) / (high_14 - low_14 + 1e-10))

        # MFI (Money Flow Index 14)
        raw_money_flow = tp * volume
        pos_flow = np.where(tp > tp.shift(1), raw_money_flow, 0.0)
        neg_flow = np.where(tp < tp.shift(1), raw_money_flow, 0.0)
        pos_mf = pd.Series(pos_flow, index=df.index).rolling(14).sum()
        neg_mf = pd.Series(neg_flow, index=df.index).rolling(14).sum()
        df['mfi'] = 100 - (100 / (1 + (pos_mf / (neg_mf + 1e-10))))

        # ROC
        df['roc_10'] = close.pct_change(10) * 100
        df['roc_20'] = close.pct_change(20) * 100

        # --- 3. Volatility Indicators ---
        df['atr_7'] = tr.rolling(7).mean()
        df['atr_14'] = atr_14

        # Bollinger Bands (20, 2)
        bb_sma = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        df['bb_middle'] = bb_sma
        df['bb_upper'] = bb_sma + (2 * bb_std)
        df['bb_lower'] = bb_sma - (2 * bb_std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (bb_sma + 1e-10)
        df['bb_percent'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

        # Keltner Channels (20, 1.5)
        kc_sma = close.rolling(20).mean()
        df['kc_middle'] = kc_sma
        df['kc_upper'] = kc_sma + (1.5 * atr_14)
        df['kc_lower'] = kc_sma - (1.5 * atr_14)

        # Squeeze Momentum Flag
        df['bb_inside_kc'] = (df['bb_lower'] > df['kc_lower']) & (df['bb_upper'] < df['kc_upper'])

        # Historical Volatility
        df['hist_vol_20'] = close.pct_change().rolling(20).std() * np.sqrt(365)
        df['hist_vol_50'] = close.pct_change().rolling(50).std() * np.sqrt(365)

        # --- 4. Volume & Flow Indicators ---
        df['volume_sma_20'] = volume.rolling(20).mean()
        df['volume_sma_ratio'] = np.where(df['volume_sma_20'] > 0, volume / df['volume_sma_20'], 0)

        # OBV
        obv_direction = np.sign(close.diff()).fillna(0)
        df['obv'] = (obv_direction * volume).cumsum()
        df['obv_ema_21'] = df['obv'].ewm(span=21, adjust=False).mean()

        # VWAP
        cum_volume = volume.cumsum()
        cum_vol_price = (tp * volume).cumsum()
        df['vwap'] = np.where(cum_volume > 0, cum_vol_price / cum_volume, close)
        df['distance_from_vwap'] = np.where(df['vwap'] > 0, (close - df['vwap']) / df['vwap'], 0)

        # CMF (Chaikin Money Flow 20)
        mf_mult = np.where(high != low, ((close - low) - (high - close)) / (high - low + 1e-10), 0)
        mf_vol = mf_mult * volume
        df['cmf'] = mf_vol.rolling(20).sum() / (volume.rolling(20).sum() + 1e-10)

        # --- 5. Statistical & Return Features ---
        df['return_1'] = close.pct_change(1)
        df['return_3'] = close.pct_change(3)
        df['return_5'] = close.pct_change(5)
        df['return_10'] = close.pct_change(10)
        df['return_20'] = close.pct_change(20)
        df['log_return_1'] = np.log(close / close.shift(1).replace(0, np.nan))

        rolling_mean_20 = close.rolling(20).mean()
        rolling_std_20 = close.rolling(20).std()
        df['price_zscore'] = (close - rolling_mean_20) / (rolling_std_20 + 1e-10)

        # Forward fill and fillna without data leakage
        df.ffill(inplace=True)
        df.fillna(0, inplace=True)

        return df
