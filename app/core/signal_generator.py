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
        
        # 1. Trend Score
        if len(df) > 200:
            last = df.iloc[-1]
            # EMA alignment
            ema_9 = last.get('ema_9', 0)
            ema_21 = last.get('ema_21', 0)
            ema_50 = last.get('ema_50', 0)
            ema_200 = last.get('ema_200', 0)
            
            if ema_9 > ema_21 > ema_50 > ema_200:
                trend_score += 0.5
                self._add_reasoning("Trend", "bullish", 0.8, "Perfect bullish EMA alignment (9 > 21 > 50 > 200)", ["EMA"])
            elif ema_9 < ema_21 < ema_50 < ema_200:
                trend_score -= 0.5
                self._add_reasoning("Trend", "bearish", 0.8, "Perfect bearish EMA alignment (9 < 21 < 50 < 200)", ["EMA"])
            
            # MACD
            macd_hist = last.get('macd_histogram', 0)
            prev_macd_hist = df.iloc[-2].get('macd_histogram', 0) if len(df) > 1 else 0
            if macd_hist > 0 and macd_hist > prev_macd_hist:
                trend_score += 0.3
            elif macd_hist < 0 and macd_hist < prev_macd_hist:
                trend_score -= 0.3
                
            # Supertrend
            st_dir = last.get('supertrend_direction', 0)
            if st_dir == 1:
                trend_score += 0.2
            elif st_dir == -1:
                trend_score -= 0.2
                
            # Normalize trend_score
            trend_score = max(-1.0, min(1.0, trend_score))
            
            # 2. Momentum Score
            rsi = last.get('rsi_14', 50)
            if rsi < 30:
                momentum_score += 0.8
                self._add_reasoning("Momentum", "bullish", 0.8, f"RSI is oversold at {rsi:.1f}", ["RSI"])
            elif rsi > 70:
                momentum_score -= 0.8
                self._add_reasoning("Momentum", "bearish", 0.8, f"RSI is overbought at {rsi:.1f}", ["RSI"])
            elif 45 <= rsi <= 55:
                self._add_reasoning("Momentum", "neutral", 0.2, f"RSI is neutral at {rsi:.1f}", ["RSI"])
                
            # 3. Volume & Flow Score
            vol_ratio = last.get('volume_sma_ratio', 1.0)
            if vol_ratio > 1.5:
                if last['close'] > last['open']:
                    volume_score += 0.5
                    self._add_reasoning("Volume", "bullish", 0.7, "Bullish volume spike detected", ["Volume"])
                else:
                    volume_score -= 0.5
                    self._add_reasoning("Volume", "bearish", 0.7, "Bearish volume spike detected", ["Volume"])
                    
            cmf = last.get('cmf', 0)
            if cmf > 0.1:
                volume_score += 0.5
            elif cmf < -0.1:
                volume_score -= 0.5
                
            volume_score = max(-1.0, min(1.0, volume_score))
            
            # 6. Price Action
            # Placeholder for price action heuristics
            pa_score = 0.0 
            
        # 4. Orderbook & Taker Flow Score
        if orderbook:
            bids_sum = sum(getattr(b, 'quantity', b[1] if isinstance(b, (list, tuple)) else 0) for b in orderbook.bids)
            asks_sum = sum(getattr(a, 'quantity', a[1] if isinstance(a, (list, tuple)) else 0) for a in orderbook.asks)
            if asks_sum > 0:
                imbalance = bids_sum / asks_sum
                if imbalance > 1.3:
                    ob_score += 0.5
                    self._add_reasoning("Microstructure", "bullish", 0.6, f"Bid depth imbalance ({imbalance:.2f}x)", ["Orderbook"])
                elif imbalance < (1/1.3):
                    ob_score -= 0.5
                    self._add_reasoning("Microstructure", "bearish", 0.6, f"Ask depth imbalance ({(1/imbalance):.2f}x)", ["Orderbook"])
        
        if trades:
            agg_buys = sum(t.quantity for t in trades if not t.is_buyer_maker)
            agg_sells = sum(t.quantity for t in trades if t.is_buyer_maker)
            total_flow = agg_buys + agg_sells
            if total_flow > 0:
                buy_ratio = agg_buys / total_flow
                if buy_ratio > 0.6:
                    ob_score += 0.5
                elif buy_ratio < 0.4:
                    ob_score -= 0.5
                    
        ob_score = max(-1.0, min(1.0, ob_score))
        
        # 5. Futures Sentiment Score
        if futures_info:
            funding = futures_info.funding_rate
            if funding < -0.0005:  # -0.05%
                futures_score += 0.5
                self._add_reasoning("Futures", "bullish", 0.6, "Negative funding indicates short squeeze potential", ["Funding Rate"])
            elif funding > 0.0005:
                futures_score -= 0.5
                self._add_reasoning("Futures", "bearish", 0.6, "High positive funding indicates overleveraged longs", ["Funding Rate"])
                
            ls_ratio = futures_info.long_short_ratio
            if ls_ratio > 1.5:
                futures_score -= 0.5
            elif ls_ratio > 0 and ls_ratio < 0.7:
                futures_score += 0.5
                
        futures_score = max(-1.0, min(1.0, futures_score))
        
        # Composite calculation
        composite = (
            trend_score * 0.3 +
            momentum_score * 0.2 +
            volume_score * 0.15 +
            ob_score * 0.15 +
            futures_score * 0.1 +
            pa_score * 0.1
        )
        
        confidence = abs(composite)
        
        if composite > 0.25 and confidence >= 0.4:  # Adjusted confidence for demonstration
            direction = SignalDirection.BUY
        elif composite < -0.25 and confidence >= 0.4:
            direction = SignalDirection.SELL
        else:
            direction = SignalDirection.HOLD
            
        model_votes = [
            ModelVote(model_name="Trend Model", direction=SignalDirection.BUY if trend_score > 0.2 else SignalDirection.SELL if trend_score < -0.2 else SignalDirection.HOLD, buy_prob=max(0, trend_score), sell_prob=max(0, -trend_score), hold_prob=1-abs(trend_score)),
            ModelVote(model_name="Momentum Model", direction=SignalDirection.BUY if momentum_score > 0.2 else SignalDirection.SELL if momentum_score < -0.2 else SignalDirection.HOLD, buy_prob=max(0, momentum_score), sell_prob=max(0, -momentum_score), hold_prob=1-abs(momentum_score)),
            ModelVote(model_name="Flow Model", direction=SignalDirection.BUY if volume_score > 0.2 else SignalDirection.SELL if volume_score < -0.2 else SignalDirection.HOLD, buy_prob=max(0, volume_score), sell_prob=max(0, -volume_score), hold_prob=1-abs(volume_score)),
        ]
        
        votes = [v.direction for v in model_votes]
        unique_votes = set(votes)
        if len(unique_votes) == 1:
            agreement = "UNANIMOUS"
        elif list(votes).count(direction) >= 2:
            agreement = "MAJORITY"
        else:
            agreement = "SPLIT"
            
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
            confidence=confidence,
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
