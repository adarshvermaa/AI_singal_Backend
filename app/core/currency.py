import time
import logging
from typing import Optional
import httpx

logger = logging.getLogger(__name__)


class CurrencyConverter:
    """
    Handles live USD/USDT to INR currency conversion and dual currency formatting.
    Caches the live CoinDCX USDTINR market price with fallback.
    """

    def __init__(self):
        self._usd_inr_rate = 99.95
        self._last_fetched = 0.0
        self._cache_ttl = 300.0  # 5 minutes cache

    async def get_usd_inr_rate(self) -> float:
        now = time.time()
        if now - self._last_fetched < self._cache_ttl and self._usd_inr_rate > 0:
            return self._usd_inr_rate

        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                res = await client.get("https://api.coindcx.com/exchange/ticker")
                if res.status_code == 200:
                    tickers = res.json()
                    for t in tickers:
                        if t.get("market") == "USDTINR":
                            price = float(t.get("last_price", 0))
                            if price > 50:
                                self._usd_inr_rate = price
                                self._last_fetched = now
                                break
        except Exception as e:
            logger.debug(f"USDT/INR rate fetch failed, using fallback {self._usd_inr_rate}: {e}")

        return self._usd_inr_rate

    @staticmethod
    def format_inr(amount: float) -> str:
        """Format number into Indian Rupee style (e.g. ₹77,17,515.05)."""
        if amount is None or amount == 0:
            return "₹0.00"

        if amount >= 100:
            s = f"{amount:.2f}"
            int_part, dec_part = s.split(".")
            if len(int_part) <= 3:
                res = int_part
            else:
                last_three = int_part[-3:]
                other = int_part[:-3]
                groups = []
                while other:
                    groups.insert(0, other[-2:])
                    other = other[:-2]
                res = ",".join(groups) + "," + last_three
            return f"₹{res}.{dec_part}" if dec_part != "00" else f"₹{res}"
        elif amount > 0:
            return f"₹{amount:.2f}"
        else:
            return f"₹{amount:.2f}"

    @staticmethod
    def format_usd(amount: float) -> str:
        """Format number into USD style (e.g. $77,198.31)."""
        if amount is None or amount == 0:
            return "$0.00"
        if amount >= 1000:
            return f"${amount:,.2f}"
        elif amount >= 1:
            return f"${amount:,.2f}"
        else:
            return f"${amount:,.4f}"

    def format_dual(self, usd_amount: float, rate: Optional[float] = None) -> str:
        """Format both Dollar and Rupee side by side (e.g. $77,198.31 | ₹77,17,515.05)."""
        r = rate or self._usd_inr_rate
        inr_amount = usd_amount * r
        return f"{self.format_usd(usd_amount)} | {self.format_inr(inr_amount)}"


currency_converter = CurrencyConverter()
