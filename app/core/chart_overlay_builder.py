import pandas as pd
from typing import Optional, List, Dict, Any
from app.api.schemas import ChartOverlays, ChartLevel, ChartPattern, ChartZone, TradeLevels

class ChartOverlayBuilder:
    """
    Constructs TradingView Lightweight Charts overlays like Support/Resistance, FVG, Patterns.
    """

    def build_overlays(
        self,
        df: pd.DataFrame,
        levels: Optional[TradeLevels],
        supports: List[float],
        resistances: List[float],
        fvgs: List[Dict[str, Any]],
        patterns: List[Dict[str, Any]]
    ) -> ChartOverlays:
        overlays = ChartOverlays()
        
        # 1. Trade Levels
        if levels:
            overlays.entry_line = ChartLevel(
                price=levels.entry, label="Entry", color="#2962FF", line_style="dashed", line_width=2
            )
            overlays.stop_loss_line = ChartLevel(
                price=levels.stop_loss, label="SL", color="#FF5252", line_style="solid", line_width=2
            )
            
            tps = []
            if levels.take_profit_1:
                tps.append(ChartLevel(price=levels.take_profit_1, label="TP1", color="#00E676", line_style="solid", line_width=2))
            if levels.take_profit_2:
                tps.append(ChartLevel(price=levels.take_profit_2, label="TP2", color="#00C853", line_style="solid", line_width=2))
            if levels.take_profit_3:
                tps.append(ChartLevel(price=levels.take_profit_3, label="TP3", color="#00B0FF", line_style="solid", line_width=2))
                
            overlays.take_profit_lines = tps
            
        # 2. Support / Resistance
        for s in supports:
            overlays.support_levels.append(
                ChartLevel(price=s, label="Support", color="#4CAF50", line_style="dotted", line_width=1)
            )
            
        for r in resistances:
            overlays.resistance_levels.append(
                ChartLevel(price=r, label="Resistance", color="#FF5252", line_style="dotted", line_width=1)
            )
            
        # 3. Fair Value Gaps
        for f in fvgs:
            overlays.fvg_zones.append(
                ChartZone(
                    upper=f.get('upper', 0.0),
                    lower=f.get('lower', 0.0),
                    label="FVG",
                    color="#FFD600" if f.get('type') == 'bullish' else "#AA00FF",
                    opacity=0.2
                )
            )
            
        # 4. Patterns
        for p in patterns:
            overlays.patterns.append(
                ChartPattern(
                    name=p.get('name', 'Pattern'),
                    start_index=p.get('start_index', 0),
                    end_index=p.get('end_index', 0),
                    confidence=p.get('confidence', 0.8),
                    direction=p.get('direction', 'neutral')
                )
            )

        return overlays
