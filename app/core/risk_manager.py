from typing import Tuple, Dict, Any

class RiskManager:
    """
    Handles position sizing, dynamic SL/TP calculation and pre-trade risk checks.
    """
    
    def calculate_position_size(
        self, balance: float, entry_price: float, stop_loss: float, max_risk_pct: float = 0.02, leverage: float = 1.0
    ) -> float:
        """
        Calculate position size such that loss at stop_loss == balance * max_risk_pct.
        """
        risk_amount = balance * max_risk_pct
        price_diff = abs(entry_price - stop_loss)
        
        if price_diff <= 0:
            return 0.0
            
        # size * price_diff = risk_amount
        size = risk_amount / price_diff
        return size

    def calculate_dynamic_levels(
        self, entry_price: float, atr: float, direction: str, risk_reward: float = 2.0
    ) -> Dict[str, float]:
        """
        Calculates dynamic Stop Loss and Take Profit levels based on ATR.
        """
        levels = {}
        
        if direction.lower() == 'buy' or direction.lower() == 'long':
            levels['stop_loss'] = entry_price - (1.5 * atr)
            levels['take_profit_1'] = entry_price + (2.0 * atr)
            levels['take_profit_2'] = entry_price + (3.0 * atr)
            levels['take_profit_3'] = entry_price + (4.5 * atr)
        else:
            levels['stop_loss'] = entry_price + (1.5 * atr)
            levels['take_profit_1'] = entry_price - (2.0 * atr)
            levels['take_profit_2'] = entry_price - (3.0 * atr)
            levels['take_profit_3'] = entry_price - (4.5 * atr)
            
        return levels

    def approve_trade(
        self, signal: dict, current_positions_count: int = 0, max_positions: int = 3
    ) -> Tuple[bool, str]:
        """
        Perform pre-trade risk validation.
        """
        if current_positions_count >= max_positions:
            return False, f"Max positions reached ({max_positions})"
            
        confidence = signal.get("confidence", 0.0)
        if confidence < 0.5:
            return False, f"Signal confidence too low ({confidence:.2f} < 0.50)"
        elif confidence < 0.65:
            import logging
            logging.getLogger(__name__).warning(
                f"Low confidence signal ({confidence:.2f} < 0.65) — proceeding with caution"
            )
            
        return True, "Trade approved"
