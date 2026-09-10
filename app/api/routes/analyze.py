import time
import logging
import pandas as pd
from fastapi import APIRouter, Request, HTTPException
from app.api.schemas import (
    AnalyzeRequest, AnalyzeResponse, SignalResult, SignalDirection,
    TradeLevels, IndicatorValues, ChartOverlays, ChartLevel,
    ReasoningItem, ModelVote, TradingModeEnum,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_pair(
    request: Request,
    body: AnalyzeRequest,
):
    """
    Analyze a trading pair and generate AI signal.
    
    This is the MAIN endpoint:
    1. Fetches live candles from CoinDCX
    2. Computes 150+ technical indicators
    3. Runs 3 AI models (XGBoost + LSTM + LightGBM)
    4. Returns signal + reasoning + chart overlays
    """
    factory = request.app.state.exchange_factory
    try:
        exch = factory.get_or_raise(body.exchange)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    try:
        # STEP 1: Fetch live market data (no storage)
        logger.info(f"Analyzing {body.symbol} on {body.timeframe}")
        
        # Fetch candles (multi-timeframe)
        if body.trading_mode == TradingModeEnum.FUTURES:
            candles = await exch.get_futures_candles(
                body.symbol, body.timeframe, limit=500
            )
        else:
            candles = await exch.get_candles(
                body.symbol, body.timeframe, limit=500
            )
        
        if not candles or len(candles) < 50:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient candle data. Got {len(candles) if candles else 0}, need at least 50."
            )
        
        current_price = candles[-1].close
        
        # Fetch additional data for AI
        orderbook = None
        trades = None
        futures_info = None
        ticker = None
        
        try:
            if body.trading_mode == 'futures':
                orderbook = await exch.get_futures_orderbook(body.symbol, depth=50)
            else:
                orderbook = await exch.get_orderbook(body.symbol, depth=50)
        except Exception as e:
            logger.warning(f"Failed to fetch orderbook: {e}")
        
        try:
            trades = await exch.get_recent_trades(body.symbol, limit=200)
        except Exception as e:
            logger.warning(f"Failed to fetch trades: {e}")
        
        try:
            futures_info = await exch.get_futures_info(body.symbol)
        except Exception as e:
            logger.warning(f"Failed to fetch futures info: {e}")
        
        try:
            ticker = await exch.get_ticker(body.symbol)
        except Exception as e:
            logger.warning(f"Failed to fetch ticker: {e}")
        
        # STEP 2: Compute indicators
        # TODO: Use IndicatorEngine when available
        # For now, compute basic indicators inline
        from app.core.indicator_engine import IndicatorEngine
        indicator_engine = IndicatorEngine()
        df = indicator_engine.candles_to_dataframe(candles)
        df = indicator_engine.compute_all(df)
        
        # Get latest indicator values
        latest = df.iloc[-1]
        indicators = IndicatorValues(
            rsi_14=_safe_float(latest, 'rsi_14'),
            rsi_7=_safe_float(latest, 'rsi_7'),
            macd=_safe_float(latest, 'macd'),
            macd_signal=_safe_float(latest, 'macd_signal'),
            macd_histogram=_safe_float(latest, 'macd_histogram'),
            ema_9=_safe_float(latest, 'ema_9'),
            ema_21=_safe_float(latest, 'ema_21'),
            ema_50=_safe_float(latest, 'ema_50'),
            ema_200=_safe_float(latest, 'ema_200'),
            adx=_safe_float(latest, 'adx'),
            atr_14=_safe_float(latest, 'atr_14'),
            bb_upper=_safe_float(latest, 'bb_upper'),
            bb_middle=_safe_float(latest, 'bb_middle'),
            bb_lower=_safe_float(latest, 'bb_lower'),
            bb_percent=_safe_float(latest, 'bb_percent'),
            stoch_k=_safe_float(latest, 'stoch_k'),
            stoch_d=_safe_float(latest, 'stoch_d'),
            obv=_safe_float(latest, 'obv'),
            vwap=_safe_float(latest, 'vwap'),
            cmf=_safe_float(latest, 'cmf'),
            mfi=_safe_float(latest, 'mfi'),
            volume_sma_ratio=_safe_float(latest, 'volume_sma_ratio'),
            supertrend=_safe_float(latest, 'supertrend'),
            supertrend_direction=int(latest.get('supertrend_direction', 0)) if ('supertrend_direction' in latest.index and pd.notna(latest.get('supertrend_direction'))) else 0,
        )
        
        # STEP 3: Generate signal
        # TODO: Use ML ensemble when trained models are available
        # For now, use rule-based signal from indicators
        from app.core.signal_generator import RuleBasedSignalGenerator
        signal_gen = RuleBasedSignalGenerator()
        signal_result = signal_gen.generate(
            df=df,
            orderbook=orderbook,
            trades=trades,
            futures_info=futures_info,
        )
        
        # STEP 4: Calculate levels
        levels = None
        if signal_result.direction != SignalDirection.HOLD:
            atr = indicators.atr_14 or (current_price * 0.01)
            if signal_result.direction == SignalDirection.BUY:
                sl = current_price - (1.5 * atr)
                tp1 = current_price + (2.0 * atr)
                tp2 = current_price + (3.0 * atr)
                tp3 = current_price + (4.5 * atr)
            else:  # SELL
                sl = current_price + (1.5 * atr)
                tp1 = current_price - (2.0 * atr)
                tp2 = current_price - (3.0 * atr)
                tp3 = current_price - (4.5 * atr)
            
            risk = abs(current_price - sl)
            reward = abs(tp1 - current_price)
            rr = reward / risk if risk > 0 else 0
            
            levels = TradeLevels(
                entry=current_price,
                stop_loss=round(sl, 2),
                take_profit_1=round(tp1, 2),
                take_profit_2=round(tp2, 2),
                take_profit_3=round(tp3, 2),
                risk_reward_ratio=round(rr, 2),
                leverage_suggestion=3.0,
            )
        
        # STEP 5: Build chart overlays
        overlays = ChartOverlays()
        if levels:
            overlays.entry_line = ChartLevel(
                price=levels.entry, label="Entry",
                color="#2196F3", line_style="solid", line_width=2,
            )
            overlays.stop_loss_line = ChartLevel(
                price=levels.stop_loss, label="Stop Loss",
                color="#F44336", line_style="dashed", line_width=2,
            )
            overlays.take_profit_lines = [
                ChartLevel(
                    price=levels.take_profit_1, label="TP1",
                    color="#4CAF50", line_style="dashed", line_width=1,
                ),
            ]
            if levels.take_profit_2:
                overlays.take_profit_lines.append(
                    ChartLevel(
                        price=levels.take_profit_2, label="TP2",
                        color="#4CAF50", line_style="dotted", line_width=1,
                    )
                )
        
        # STEP 6: Build response
        return AnalyzeResponse(
            symbol=body.symbol,
            timeframe=body.timeframe,
            exchange=body.exchange,
            timestamp=int(time.time() * 1000),
            current_price=current_price,
            signal=signal_result,
            levels=levels,
            reasoning=signal_gen.get_reasoning(),
            overlays=overlays,
            indicators=indicators,
            market_regime=signal_gen.get_market_regime(),
            funding_rate=futures_info.funding_rate if futures_info else None,
            long_short_ratio=futures_info.long_short_ratio if futures_info else None,
            volume_24h=ticker.volume_24h if ticker else None,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


def _safe_float(series, key: str) -> float | None:
    """Safely extract float from pandas series."""
    try:
        if key in series.index:
            val = series[key]
            if val is not None and not (isinstance(val, float) and val != val):  # NaN check
                return round(float(val), 6)
    except Exception:
        pass
    return None
