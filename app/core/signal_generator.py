import pandas as pd
import numpy as np
from typing import Optional, List, Tuple

from app.api.schemas import SignalDirection, SignalResult, ModelVote, ReasoningItem
from app.exchanges.base import OrderBook, Trade, FuturesInfo

class RuleBasedSignalGenerator:
    """
    Generates trading signals based on technical analysis rules across 6 dimensions.
    """

    def __init__(self):
        self.reasoning: list[ReasoningItem] = []
        self.market_regime: str = "unknown"
    def generate(
        self,
        df: pd.DataFrame,
        orderbook: Optional[OrderBook] = None,
        trades: Optional[List[Trade]] = None,
        futures_info: Optional[FuturesInfo] = None
    ) -> SignalResult:
        # Default scores
        trend_score = 0.0
        momentum_score = 0.0
        volume_score = 0.0
        ob_score = 0.0
        futures_score = 0.0
        pa_score = 0.0
        
        self.reasoning = []
        
        # 1. Trend & Technical Indicators
        if len(df) >= 30:
            last = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else last

            # EMA alignment
            ema_9 = last.get('ema_9', 0)
            ema_21 = last.get('ema_21', 0)
            ema_50 = last.get('ema_50', 0)
            ema_200 = last.get('ema_200', 0)

            if ema_9 and ema_21 and ema_50:
                if ema_9 > ema_21 > ema_50:
                    if len(df) >= 200 and ema_200 and ema_50 > ema_200:
                        trend_score += 0.5
                        self._add_reasoning("Trend", "bullish", 0.85, "Full Bullish EMA alignment (9 > 21 > 50 > 200)", ["EMA"])
                    else:
                        trend_score += 0.35
                        self._add_reasoning("Trend", "bullish", 0.70, "Bullish EMA structure (9 > 21 > 50)", ["EMA"])
                elif ema_9 < ema_21 < ema_50:
                    if len(df) >= 200 and ema_200 and ema_50 < ema_200:
                        trend_score -= 0.5
                        self._add_reasoning("Trend", "bearish", 0.85, "Full Bearish EMA alignment (9 < 21 < 50 < 200)", ["EMA"])
                    else:
                        trend_score -= 0.35
                        self._add_reasoning("Trend", "bearish", 0.70, "Bearish EMA structure (9 < 21 < 50)", ["EMA"])

            # MACD
            macd_hist = last.get('macd_histogram', 0)
            prev_macd_hist = prev.get('macd_histogram', 0)
            macd_val = last.get('macd', 0)
            macd_sig = last.get('macd_signal', 0)
            prev_macd = prev.get('macd', 0)
            prev_sig = prev.get('macd_signal', 0)

            if macd_val > macd_sig and prev_macd <= prev_sig:
                trend_score += 0.3
                self._add_reasoning("Trend", "bullish", 0.75, "Bullish MACD crossover confirmed", ["MACD"])
            elif macd_val < macd_sig and prev_macd >= prev_sig:
                trend_score -= 0.3
                self._add_reasoning("Trend", "bearish", 0.75, "Bearish MACD crossover confirmed", ["MACD"])
            elif macd_hist > 0 and macd_hist > prev_macd_hist:
                trend_score += 0.2
            elif macd_hist < 0 and macd_hist < prev_macd_hist:
                trend_score -= 0.2

            # Supertrend
            st_dir = last.get('supertrend_direction', 0)
            if st_dir == 1:
                trend_score += 0.25
                self._add_reasoning("Trend", "bullish", 0.75, "Supertrend Bullish trend confirmation", ["Supertrend"])
            elif st_dir == -1:
                trend_score -= 0.25
                self._add_reasoning("Trend", "bearish", 0.75, "Supertrend Bearish trend confirmation", ["Supertrend"])

            trend_score = max(-1.0, min(1.0, trend_score))

            # 2. Momentum Score (RSI)
            rsi = last.get('rsi_14', 50)
            if rsi < 32:
                momentum_score += 0.8
                self._add_reasoning("Momentum", "bullish", 0.85, f"RSI severely oversold at {rsi:.1f} (Mean-reversion bounce expected)", ["RSI"])
            elif rsi < 42:
                momentum_score += 0.4
                self._add_reasoning("Momentum", "bullish", 0.65, f"RSI recovering from oversold zone at {rsi:.1f}", ["RSI"])
            elif rsi > 68:
                momentum_score -= 0.8
                self._add_reasoning("Momentum", "bearish", 0.85, f"RSI overbought at {rsi:.1f} (Exhaustion risk)", ["RSI"])
            elif rsi > 58:
                momentum_score -= 0.4
                self._add_reasoning("Momentum", "bearish", 0.65, f"RSI elevated in upper distribution band at {rsi:.1f}", ["RSI"])
            elif 45 <= rsi <= 55:
                self._add_reasoning("Momentum", "neutral", 0.20, f"RSI balanced at {rsi:.1f}", ["RSI"])

            # 3. Volume & Flow Score
            vol_ratio = last.get('volume_sma_ratio', 1.0)
            close_p = float(last.get('close', 0))
            open_p = float(last.get('open', 0))
            if vol_ratio > 1.3:
                if close_p > open_p:
                    volume_score += 0.5
                    self._add_reasoning("Volume", "bullish", 0.75, f"Bullish volume expansion ({vol_ratio:.1f}x average)", ["Volume"])
                else:
                    volume_score -= 0.5
                    self._add_reasoning("Volume", "bearish", 0.75, f"Bearish volume distribution ({vol_ratio:.1f}x average)", ["Volume"])

            cmf = last.get('cmf', 0)
            if cmf > 0.08:
                volume_score += 0.4
            elif cmf < -0.08:
                volume_score -= 0.4

            volume_score = max(-1.0, min(1.0, volume_score))

            # 4. Price Action & Candlestick Structure
            high_p = float(last.get('high', 0))
            low_p = float(last.get('low', 0))
            prev_high = float(prev.get('high', 0))
            prev_low = float(prev.get('low', 0))
            prev_close = float(prev.get('close', 0))
            prev_open = float(prev.get('open', 0))
            body_size = abs(close_p - open_p)
            candle_range = max(1e-6, high_p - low_p)

            # Higher High & Higher Low
            if high_p > prev_high and low_p > prev_low:
                pa_score += 0.4
                self._add_reasoning("Price Action", "bullish", 0.65, "Higher High and Higher Low swing structure", ["Price Action"])
            elif high_p < prev_high and low_p < prev_low:
                pa_score -= 0.4
                self._add_reasoning("Price Action", "bearish", 0.65, "Lower High and Lower Low swing breakdown", ["Price Action"])

            # Bullish Hammer / Pinbar
            lower_wick = min(open_p, close_p) - low_p
            if lower_wick > 2.0 * body_size and (close_p - low_p) / candle_range > 0.65:
                pa_score += 0.4
                self._add_reasoning("Price Action", "bullish", 0.70, "Bullish rejection pinbar at support", ["Candlestick"])

            # Bearish Shooting Star / Inverted Pinbar
            upper_wick = high_p - max(open_p, close_p)
            if upper_wick > 2.0 * body_size and (high_p - close_p) / candle_range > 0.65:
                pa_score -= 0.4
                self._add_reasoning("Price Action", "bearish", 0.70, "Bearish rejection pinbar at resistance", ["Candlestick"])

            pa_score = max(-1.0, min(1.0, pa_score))

        # 5. Orderbook & Taker Flow Score
        if orderbook:
            bids_sum = sum(getattr(b, 'quantity', b[1] if isinstance(b, (list, tuple)) else 0) for b in orderbook.bids)
            asks_sum = sum(getattr(a, 'quantity', a[1] if isinstance(a, (list, tuple)) else 0) for a in orderbook.asks)
            if asks_sum > 0:
                imbalance = bids_sum / asks_sum
                if imbalance > 1.25:
                    ob_score += 0.5
                    self._add_reasoning("Microstructure", "bullish", 0.65, f"Bid book depth dominance ({imbalance:.2f}x)", ["Orderbook"])
                elif imbalance < (1 / 1.25):
                    ob_score -= 0.5
                    self._add_reasoning("Microstructure", "bearish", 0.65, f"Ask book depth resistance ({(1 / imbalance):.2f}x)", ["Orderbook"])

        if trades:
            agg_buys = sum(t.quantity for t in trades if not t.is_buyer_maker)
            agg_sells = sum(t.quantity for t in trades if t.is_buyer_maker)
            total_flow = agg_buys + agg_sells
            if total_flow > 0:
                buy_ratio = agg_buys / total_flow
                if buy_ratio > 0.58:
                    ob_score += 0.4
                elif buy_ratio < 0.42:
                    ob_score -= 0.4

        ob_score = max(-1.0, min(1.0, ob_score))

        # 6. Futures Sentiment Score
        if futures_info:
            funding = futures_info.funding_rate
            if funding < -0.0003:  # Negative funding
                futures_score += 0.5
                self._add_reasoning("Futures", "bullish", 0.60, "Negative funding indicates short squeeze potential", ["Funding Rate"])
            elif funding > 0.0005:
                futures_score -= 0.5
                self._add_reasoning("Futures", "bearish", 0.60, "Elevated positive funding indicates long crowdedness", ["Funding Rate"])

            ls_ratio = futures_info.long_short_ratio
            if ls_ratio > 1.5:
                futures_score -= 0.4
            elif 0 < ls_ratio < 0.7:
                futures_score += 0.4

        futures_score = max(-1.0, min(1.0, futures_score))

        # Composite institutional calculation
        composite = (
            trend_score * 0.30 +
            momentum_score * 0.25 +
            volume_score * 0.15 +
            ob_score * 0.15 +
            futures_score * 0.05 +
            pa_score * 0.10
        )

        if composite >= 0.18:
            direction = SignalDirection.BUY
            confidence = min(0.96, max(0.68, 0.62 + abs(composite) * 0.38))
        elif composite <= -0.18:
            direction = SignalDirection.SELL
            confidence = min(0.96, max(0.68, 0.62 + abs(composite) * 0.38))
        else:
            direction = SignalDirection.HOLD
            confidence = min(0.50, max(0.25, abs(composite) * 1.5))

        model_votes = [
            ModelVote(
                model_name="Trend Engine",
                direction=SignalDirection.BUY if trend_score > 0.15 else SignalDirection.SELL if trend_score < -0.15 else SignalDirection.HOLD,
                buy_prob=max(0.0, trend_score),
                sell_prob=max(0.0, -trend_score),
                hold_prob=1.0 - abs(trend_score),
            ),
            ModelVote(
                model_name="Momentum Oscillator",
                direction=SignalDirection.BUY if momentum_score > 0.15 else SignalDirection.SELL if momentum_score < -0.15 else SignalDirection.HOLD,
                buy_prob=max(0.0, momentum_score),
                sell_prob=max(0.0, -momentum_score),
                hold_prob=1.0 - abs(momentum_score),
            ),
            ModelVote(
                model_name="Volume Flow",
                direction=SignalDirection.BUY if volume_score > 0.15 else SignalDirection.SELL if volume_score < -0.15 else SignalDirection.HOLD,
                buy_prob=max(0.0, volume_score),
                sell_prob=max(0.0, -volume_score),
                hold_prob=1.0 - abs(volume_score),
            ),
            ModelVote(
                model_name="Price Action",
                direction=SignalDirection.BUY if pa_score > 0.15 else SignalDirection.SELL if pa_score < -0.15 else SignalDirection.HOLD,
                buy_prob=max(0.0, pa_score),
                sell_prob=max(0.0, -pa_score),
                hold_prob=1.0 - abs(pa_score),
            ),
        ]

        votes = [v.direction for v in model_votes]
        matching_votes = votes.count(direction) if direction != SignalDirection.HOLD else votes.count(SignalDirection.HOLD)
        if matching_votes >= 3:
            agreement = "UNANIMOUS" if matching_votes == 4 else "MAJORITY"
        else:
            agreement = "MAJORITY" if matching_votes >= 2 else "SPLIT"

        self.market_regime = "unknown"
        if len(df) > 0:
            last = df.iloc[-1]
            adx = last.get('adx', 0)
            if adx > 25:
                self.market_regime = "trending"
            elif adx < 20:
                self.market_regime = "ranging"
            else:
                self.market_regime = "volatile"

        return SignalResult(
            direction=direction,
            confidence=round(confidence, 2),
            model_agreement=agreement,
            model_votes=model_votes
        )

    def _add_reasoning(self, category: str, signal: str, strength: float, description: str, indicators: List[str]):
        self.reasoning.append(ReasoningItem(
            category=category,
            signal=signal,
            strength=strength,
            description=description,
            indicators=indicators
        ))

    def get_reasoning(self) -> List[ReasoningItem]:
        return getattr(self, 'reasoning', [])

    def get_market_regime(self) -> str:
        return getattr(self, 'market_regime', "unknown")
