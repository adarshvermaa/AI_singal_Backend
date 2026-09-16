from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # Server
    app_name: str = "AI Signal Engine"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]
    
    # CoinDCX
    coindcx_api_key: str = ""
    coindcx_api_secret: str = ""
    
    # Trading
    default_leverage: float = 3.0
    max_risk_per_trade: float = 0.02  # 2% of portfolio
    min_confidence: float = 0.70
    default_timeframe: str = "15m"
    
    # Model paths
    model_dir: str = "app/ml/pretrained"

    # Webhooks
    webhook_secret_key: str = "coindcx_secret_key_8899"
    webhook_ai_filter_enabled: bool = False
    webhook_min_confidence: float = 0.70
    coindcx_webhook_url: str = "https://api.coindcx.com/callbacks/v1/derivatives/futures/order/jSsqRkGH5aQDvuD01JLoV1jDo"
    coindcx_webhook_id: str = "d536c102-6568-4f5e-8297-5ef23f194ace"
    
    # Telegram Bot
    telegram_bot_token: str = "8978992155:AAF6k9I0k61hYI97TfuuKDVAq_5_pGT7Rf0"
    telegram_chat_id: str = ""
    telegram_auto_send: bool = False
    telegram_min_confidence: float = 0.70
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache
def get_settings() -> Settings:
    return Settings()
