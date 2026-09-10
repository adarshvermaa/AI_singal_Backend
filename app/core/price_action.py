import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from scipy.signal import argrelextrema

class PriceActionEngine:
    """
    Engine to detect price action patterns, market structure, support/resistance, and fair value gaps.
    """

    def detect_candlestick_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """
        Identifies key candlestick patterns.
        """
        patterns = []
        if df.empty or len(df) < 3:
            return patterns

        opens = df['open'].to_numpy()
        highs = df['high'].to_numpy()
        lows = df['low'].to_numpy()
        closes = df['close'].to_numpy()
        indices = df.index

        for i in range(2, len(df)):
            idx = indices[i]
            c_open, c_high, c_low, c_close = opens[i], highs[i], lows[i], closes[i]
            prev_open, prev_high, prev_low, prev_close = opens[i-1], highs[i-1], lows[i-1], closes[i-1]
            prev2_open, prev2_close = opens[i-2], closes[i-2]
            
            body = abs(c_close - c_open)
            total_range = c_high - c_low
            if total_range == 0:
                total_range = 1e-10

            upper_wick = c_high - max(c_open, c_close)
            lower_wick = min(c_open, c_close) - c_low
            
            is_bullish = c_close > c_open
            is_bearish = c_close < c_open

            # Doji
            if body / total_range < 0.1:
                if lower_wick > total_range * 0.6:
                    patterns.append({'index': idx, 'name': 'Dragonfly Doji', 'direction': 'bullish', 'confidence': 0.7})
                elif upper_wick > total_range * 0.6:
                    patterns.append({'index': idx, 'name': 'Gravestone Doji', 'direction': 'bearish', 'confidence': 0.7})
                else:
                    patterns.append({'index': idx, 'name': 'Doji', 'direction': 'neutral', 'confidence': 0.5})
            
            # Hammer / Inverted Hammer / Pinbar
            elif body / total_range < 0.3:
                if lower_wick > total_range * 0.6 and is_bullish:
                    patterns.append({'index': idx, 'name': 'Hammer / Bullish Pinbar', 'direction': 'bullish', 'confidence': 0.8})
                elif upper_wick > total_range * 0.6 and is_bearish:
                    patterns.append({'index': idx, 'name': 'Inverted Hammer / Bearish Pinbar', 'direction': 'bearish', 'confidence': 0.8})
            
            # Engulfing
            prev_body = abs(prev_close - prev_open)
            prev_is_bullish = prev_close > prev_open
            prev_is_bearish = prev_close < prev_open

            if is_bullish and prev_is_bearish and c_close > prev_open and c_open < prev_close:
                patterns.append({'index': idx, 'name': 'Bullish Engulfing', 'direction': 'bullish', 'confidence': 0.85})
            elif is_bearish and prev_is_bullish and c_close < prev_open and c_open > prev_close:
                patterns.append({'index': idx, 'name': 'Bearish Engulfing', 'direction': 'bearish', 'confidence': 0.85})

            # Morning / Evening Star (simplified 3-candle)
            prev2_is_bearish = prev2_close < prev2_open
            prev2_is_bullish = prev2_close > prev2_open
            
            if prev2_is_bearish and (prev_body / (prev_high - prev_low + 1e-10) < 0.3) and is_bullish and c_close > (prev2_open + prev2_close)/2:
                patterns.append({'index': idx, 'name': 'Morning Star', 'direction': 'bullish', 'confidence': 0.85})
            elif prev2_is_bullish and (prev_body / (prev_high - prev_low + 1e-10) < 0.3) and is_bearish and c_close < (prev2_open + prev2_close)/2:
                patterns.append({'index': idx, 'name': 'Evening Star', 'direction': 'bearish', 'confidence': 0.85})

        return patterns

    def detect_support_resistance(self, df: pd.DataFrame, window: int = 15, max_levels: int = 5) -> Tuple[List[float], List[float]]:
        """
        Finds key horizontal support & resistance pivot points using local minima and maxima.
        """
        if df.empty or len(df) < window * 2:
            return [], []

        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values

        # Local maxima (Resistance)
        local_max = argrelextrema(highs, np.greater, order=window)[0]
        # Local minima (Support)
        local_min = argrelextrema(lows, np.less, order=window)[0]

        resistances = highs[local_max].tolist()
        supports = lows[local_min].tolist()

        # Cluster nearby levels (within 0.5%)
        def cluster_levels(levels: List[float], threshold: float = 0.005) -> List[float]:
            levels.sort(reverse=True)
            clustered = []
            while levels:
                current = levels.pop(0)
                cluster = [current]
                i = 0
                while i < len(levels):
                    if abs(current - levels[i]) / current <= threshold:
                        cluster.append(levels.pop(i))
                    else:
                        i += 1
                clustered.append(sum(cluster) / len(cluster))
            return clustered

        supports = cluster_levels(supports)
        resistances = cluster_levels(resistances)

        # Sort and return top levels
        supports.sort()
        resistances.sort(reverse=True)

        return supports[:max_levels], resistances[:max_levels]

    def detect_fair_value_gaps(self, df: pd.DataFrame, min_gap_pct: float = 0.001) -> List[Dict]:
        """
        Identifies 3-candle Fair Value Gaps (FVG) / Imbalances.
        """
        fvgs = []
        if df.empty or len(df) < 3:
            return fvgs

        for i in range(2, len(df)):
            c1 = df.iloc[i-2]
            c2 = df.iloc[i-1] # The candle creating the imbalance
            c3 = df.iloc[i]

            # Bullish FVG: C1 High < C3 Low
            if c1['high'] < c3['low']:
                gap_size = (c3['low'] - c1['high']) / c1['high']
                if gap_size >= min_gap_pct:
                    fvgs.append({
                        'start_index': df.index[i-1],
                        'end_index': df.index[i],
                        'type': 'bullish_fvg',
                        'lower': c1['high'],
                        'upper': c3['low']
                    })

            # Bearish FVG: C1 Low > C3 High
            if c1['low'] > c3['high']:
                gap_size = (c1['low'] - c3['high']) / c3['high']
                if gap_size >= min_gap_pct:
                    fvgs.append({
                        'start_index': df.index[i-1],
                        'end_index': df.index[i],
                        'type': 'bearish_fvg',
                        'lower': c3['high'],
                        'upper': c1['low']
                    })

        return fvgs

    def detect_market_structure(self, df: pd.DataFrame) -> Dict:
        """
        Calculates HH, HL, LH, LL to classify market structure.
        """
        if df.empty or len(df) < 10:
            return {"trend": "unknown"}

        # Simply use last 10-20 periods to check recent pivots
        window = 5
        highs = df['high'].values
        lows = df['low'].values
        
        local_max = argrelextrema(highs, np.greater, order=window)[0]
        local_min = argrelextrema(lows, np.less, order=window)[0]

        if len(local_max) < 2 or len(local_min) < 2:
            return {"trend": "ranging"}

        last_2_highs = highs[local_max[-2:]]
        last_2_lows = lows[local_min[-2:]]

        hh = last_2_highs[1] > last_2_highs[0]
        lh = last_2_highs[1] < last_2_highs[0]
        
        hl = last_2_lows[1] > last_2_lows[0]
        ll = last_2_lows[1] < last_2_lows[0]

        if hh and hl:
            trend = "bullish_trend"
        elif lh and ll:
            trend = "bearish_trend"
        else:
            trend = "ranging"

        return {
            "trend": trend,
            "last_high": float(last_2_highs[1]),
            "last_low": float(last_2_lows[1])
        }
