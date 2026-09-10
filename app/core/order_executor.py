import logging
from typing import Optional, Dict, Any
from app.exchanges.base import ExchangeBase, OrderSide, OrderType, OrderResult, MarginType
from app.core.risk_manager import RiskManager

logger = logging.getLogger(__name__)

class OrderExecutor:
    """
    Executes trades on exchanges based on AI signals.
    """
    def __init__(self, exchange: ExchangeBase, risk_manager: RiskManager):
        self.exchange = exchange
        self.risk_manager = risk_manager

    async def execute_trade(
        self,
        pair: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        leverage: float = 1.0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trading_mode: str = "futures"
    ) -> OrderResult:
        """
        Executes a trade after running pre-trade checks.
        """
        
        # Risk Manager Check
        signal_mock = {"confidence": 0.8}  # Ideally pass full signal
        approved, msg = self.risk_manager.approve_trade(signal=signal_mock)
        if not approved:
            raise ValueError(f"Trade not approved by Risk Manager: {msg}")

        order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
        order_type = OrderType.LIMIT if price else OrderType.MARKET
        
        if trading_mode == "margin":
            result = await self.exchange.place_margin_order(
                pair=pair,
                side=order_side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                leverage=leverage,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            return result
            
        elif trading_mode == "futures":
            result = await self.exchange.place_futures_order(
                pair=pair,
                side=order_side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                leverage=leverage,
                stop_price=stop_loss,
                margin_type=MarginType.ISOLATED
            )
            
            # Set Take Profit and Stop Loss separately if not supported in initial order
            if result and (take_profit or stop_loss):
                target_id = getattr(result, 'order_id', getattr(result, 'id', ''))
                if target_id:
                    try:
                        await self.exchange.set_futures_tp_sl(
                            position_id=target_id,
                            take_profit=take_profit,
                            stop_loss=stop_loss
                        )
                    except Exception as e:
                        logger.critical(f"FAILED to set TP/SL for position {target_id} — POSITION IS UNPROTECTED: {e}")
                    
            return result
            
        else:
            # Spot
            result = await self.exchange.place_spot_order(
                pair=pair,
                side=order_side,
                order_type=order_type,
                quantity=quantity,
                price=price
            )
            return result
