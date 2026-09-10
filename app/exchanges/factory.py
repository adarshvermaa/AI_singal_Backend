from typing import Optional
from app.exchanges.base import ExchangeBase, ExchangeError


class ExchangeFactory:
    """
    Registry and factory for exchange adapters.
    
    Usage:
        factory = ExchangeFactory()
        exchange = factory.create("coindcx", api_key="...", api_secret="...")
        await exchange.initialize()
    """
    
    _registry: dict[str, type[ExchangeBase]] = {}
    _instances: dict[str, ExchangeBase] = {}
    
    def __init__(self):
        self._instances = {}
        self._register_defaults()
    
    def _register_defaults(self):
        """Register built-in exchange adapters."""
        from app.exchanges.coindcx.adapter import CoinDCXExchange
        self._registry["coindcx"] = CoinDCXExchange
        # Future: self._registry["wazirx"] = WazirXExchange
        # Future: self._registry["binance"] = BinanceExchange
    
    @classmethod
    def register(cls, name: str, adapter_class: type[ExchangeBase]):
        """Register a new exchange adapter."""
        cls._registry[name.lower()] = adapter_class
    
    def create(
        self, name: str, api_key: str, api_secret: str, **kwargs
    ) -> ExchangeBase:
        """Create an exchange instance. Reuses existing if same name."""
        name = name.lower()
        if name not in self._registry:
            available = ", ".join(self._registry.keys())
            raise ExchangeError(
                f"Exchange '{name}' not supported. Available: {available}"
            )
        
        # Create new instance
        instance = self._registry[name](
            api_key=api_key, api_secret=api_secret, **kwargs
        )
        self._instances[name] = instance
        return instance
    
    def get(self, name: str) -> Optional[ExchangeBase]:
        """Get an existing exchange instance."""
        return self._instances.get(name.lower())
    
    def get_or_raise(self, name: str) -> ExchangeBase:
        """Get an existing exchange instance or auto-create for public data."""
        exchange = self.get(name)
        if not exchange:
            if name.lower() in self._registry:
                from app.config import get_settings
                settings = get_settings()
                return self.create(
                    name,
                    api_key=settings.coindcx_api_key,
                    api_secret=settings.coindcx_api_secret,
                )
            raise ExchangeError(
                f"Exchange '{name}' not supported. Available: {list(self._registry.keys())}"
            )
        return exchange
    
    @property
    def connected_exchanges(self) -> list[str]:
        """List connected exchange names."""
        return list(self._instances.keys())
    
    @staticmethod
    def supported_exchanges() -> list[dict]:
        """List all supported exchanges with info."""
        return [
            {
                "name": "coindcx",
                "display_name": "CoinDCX",
                "supports": ["spot", "margin", "futures"],
                "status": "active",
            },
            {
                "name": "wazirx",
                "display_name": "WazirX",
                "supports": ["spot"],
                "status": "coming_soon",
            },
            {
                "name": "binance",
                "display_name": "Binance",
                "supports": ["spot", "margin", "futures"],
                "status": "coming_soon",
            },
        ]
    
    async def close_all(self):
        """Close all exchange connections."""
        for exchange in self._instances.values():
            await exchange.close()
        self._instances.clear()
