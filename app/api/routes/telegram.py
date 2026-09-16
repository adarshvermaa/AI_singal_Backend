import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from app.config import get_settings
from app.core.telegram_service import telegram_service

logger = logging.getLogger(__name__)
router = APIRouter()


# === Schemas ===

class TelegramConfigRequest(BaseModel):
    chat_id: Optional[str] = Field(None, description="Target Channel Username or Chat ID (e.g. @my_channel or -100xxx)")
    auto_send: Optional[bool] = Field(None, description="Automatically broadcast signals >= min_confidence")
    min_confidence: Optional[float] = Field(None, ge=0.5, le=1.0, description="Minimum confidence threshold for auto-send")
    bot_token: Optional[str] = Field(None, description="Custom Bot Token override if needed")


class SendSignalRequest(BaseModel):
    symbol: str = Field(..., description="Pair e.g. B-BTC_USDT")
    direction: str = Field(..., description="BUY or SELL")
    price: float = Field(..., description="Current price")
    confidence: float = Field(default=0.80, description="Confidence 0-1")
    sl: Optional[float] = None
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    tp3: Optional[float] = None
    timeframe: str = Field(default="15m")
    trading_mode: str = Field(default="futures")
    chat_id: Optional[str] = None
    model_agreement: Optional[str] = None
    rsi: Optional[float] = None
    supertrend: Optional[str] = None
    regime: Optional[str] = None


class TestMessageRequest(BaseModel):
    chat_id: Optional[str] = None


# === Routes ===

@router.get("/status")
async def telegram_status():
    """Get current Telegram bot connection status."""
    settings = get_settings()
    bot_info = await telegram_service.get_me()
    return {
        "configured": telegram_service.is_configured,
        "bot_info": bot_info.get("result") if bot_info.get("ok") else None,
        "chat_id": settings.telegram_chat_id,
        "auto_send": settings.telegram_auto_send,
        "min_confidence": settings.telegram_min_confidence,
    }


@router.get("/detect-chat")
async def detect_telegram_chat():
    """Auto-detect recent chats/channels where the bot was invited or received messages."""
    chats = await telegram_service.detect_recent_chats()
    return {
        "count": len(chats),
        "chats": chats,
        "instruction": "Send /start to @alphx_signal_bot or add it as Admin to your Channel to auto-detect."
    }


@router.post("/config")
async def update_telegram_config(body: TelegramConfigRequest):
    """Update Telegram target chat ID and automated broadcasting settings."""
    settings = get_settings()

    if body.chat_id is not None:
        settings.telegram_chat_id = body.chat_id.strip()
    if body.auto_send is not None:
        settings.telegram_auto_send = body.auto_send
    if body.min_confidence is not None:
        settings.telegram_min_confidence = body.min_confidence
    if body.bot_token:
        settings.telegram_bot_token = body.bot_token.strip()
        telegram_service.bot_token = settings.telegram_bot_token
        telegram_service.base_url = f"https://api.telegram.org/bot{telegram_service.bot_token}"

    return {
        "status": "updated",
        "chat_id": settings.telegram_chat_id,
        "auto_send": settings.telegram_auto_send,
        "min_confidence": settings.telegram_min_confidence,
    }


def _validate_target_chat(chat_id: str):
    bot_usernames = {"@alphx_signal_bot", "alphx_signal_bot", "8978992155"}
    if chat_id.strip().lower() in bot_usernames:
        raise HTTPException(
            status_code=400,
            detail=(
                "You entered the bot's username (@alphx_signal_bot). "
                "A Telegram bot cannot send messages to itself! "
                "Please enter YOUR channel username (e.g. @your_channel) where you added the bot as an Admin, "
                "or your personal Chat ID."
            )
        )


@router.post("/send")
async def send_signal_to_telegram(body: SendSignalRequest):
    """
    Broadcast an AI trade signal to Telegram.
    Can be called manually from the UI or automatically by the signal engine.
    """
    settings = get_settings()
    target_chat = body.chat_id or settings.telegram_chat_id
    if not target_chat:
        raise HTTPException(
            status_code=400,
            detail="No target Telegram chat_id or channel specified. Please configure chat_id first."
        )

    _validate_target_chat(target_chat)

    res = await telegram_service.send_signal_message(
        chat_id=target_chat,
        symbol=body.symbol,
        direction=body.direction,
        price=body.price,
        confidence=body.confidence,
        sl=body.sl,
        tp1=body.tp1,
        tp2=body.tp2,
        tp3=body.tp3,
        timeframe=body.timeframe,
        trading_mode=body.trading_mode,
        model_agreement=body.model_agreement,
        rsi=body.rsi,
        supertrend=body.supertrend,
        regime=body.regime,
        bypass_cooldown=True,  # Manual/explicit send always bypasses cooldown
    )

    if not res.get("ok"):
        error_msg = res.get("description") or res.get("error") or "Failed to send Telegram message"
        if "bot can't send messages to the bot" in error_msg.lower():
            error_msg = (
                "The bot cannot send messages to itself. Please enter YOUR channel handle "
                "(e.g. @your_channel) or your personal Chat ID."
            )
        elif "chat not found" in error_msg.lower():
            error_msg = (
                f"Channel or chat '{target_chat}' not found. Ensure the channel exists, is public, "
                f"or that the bot has been added as an Administrator."
            )
        raise HTTPException(status_code=400, detail=error_msg)

    return {
        "status": "sent",
        "message_id": res.get("result", {}).get("message_id"),
        "chat": res.get("result", {}).get("chat", {}).get("title") or target_chat,
    }


@router.post("/test")
async def test_telegram_message(body: TestMessageRequest):
    """Send a verification test ping to the specified or configured Telegram chat."""
    settings = get_settings()
    target_chat = body.chat_id or settings.telegram_chat_id
    if not target_chat:
        raise HTTPException(
            status_code=400,
            detail="Please specify or configure a target chat_id (e.g. @your_channel or -100xxxxxxx)"
        )

    _validate_target_chat(target_chat)

    test_text = (
        "🚀 <b>ALPHX QUANT TERMINAL | TELEGRAM CONNECTED</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "✅ <b>Status:</b> Live & Connected\n"
        "🤖 <b>Bot:</b> @alphx_signal_bot\n"
        "⚡ <b>Signal Broadcasting:</b> Active\n\n"
        "<i>This is a verification test message. Algorithmic trade signals will be posted here in real-time.</i>"
    )

    res = await telegram_service.send_message(
        chat_id=target_chat,
        text=test_text,
        parse_mode="HTML",
    )

    if not res.get("ok"):
        error_msg = res.get("description") or res.get("error") or "Failed to deliver test message"
        if "bot can't send messages to the bot" in error_msg.lower():
            error_msg = (
                "The bot cannot send messages to itself. Please enter YOUR channel handle "
                "(e.g. @your_channel) or your personal Chat ID."
            )
        elif "chat not found" in error_msg.lower():
            error_msg = (
                f"Channel or chat '{target_chat}' not found. Ensure the channel exists, is public, "
                f"or that the bot has been added as an Administrator."
            )
        raise HTTPException(status_code=400, detail=error_msg)

    return {
        "status": "delivered",
        "message_id": res.get("result", {}).get("message_id"),
        "chat": res.get("result", {}).get("chat", {}).get("title") or target_chat,
    }
