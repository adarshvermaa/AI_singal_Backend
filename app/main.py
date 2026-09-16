from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.config import get_settings
from app.api.routes import analyze, exchange, trade, markets, health, ws, webhook, telegram
from app.exchanges.factory import ExchangeFactory

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load ML models, initialize exchange factory, start WebSocket streaming."""
    settings = get_settings()
    logger.info("Starting AI Signal Engine...")
    
    # Initialize exchange factory (global registry)
    app.state.exchange_factory = ExchangeFactory()
    
    # Start live ticker WebSocket background stream task
    import asyncio
    stream_task = asyncio.create_task(ws.manager.start_background_stream(app))

    logger.info("Ready for real-time market data streaming and inference.")
    
    yield
    
    # Cleanup
    logger.info("Shutting down...")
    stream_task.cancel()
    if hasattr(app.state, 'exchange_factory'):
        await app.state.exchange_factory.close_all()

app = FastAPI(
    title="AI Crypto Signal Engine",
    description="AI-powered crypto trading signal generator with CoinDCX integration",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(markets.router, prefix="/api", tags=["Markets"])
app.include_router(analyze.router, prefix="/api", tags=["Analysis"])
app.include_router(exchange.router, prefix="/api/exchange", tags=["Exchange"])
app.include_router(trade.router, prefix="/api/trade", tags=["Trading"])
app.include_router(webhook.router, prefix="/api/webhook", tags=["Webhook"])
app.include_router(ws.router, prefix="/api", tags=["WebSocket"])
app.include_router(telegram.router, prefix="/api/telegram", tags=["Telegram"])
