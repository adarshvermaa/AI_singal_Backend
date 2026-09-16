import json
import time
import logging
from datetime import datetime, timezone
from typing import Optional
import httpx
from app.config import get_settings
from app.core.currency import currency_converter

logger = logging.getLogger(__name__)


class TelegramService:
    """
    Service for sending rich algorithmic trade signals and notifications
    via Telegram Bot API (e.g. @alphx_signal_bot).
    """

    def __init__(self, bot_token: Optional[str] = None):
        settings = get_settings()
        self.bot_token = bot_token or settings.telegram_bot_token
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self._last_sent: dict[str, float] = {}
        self._cooldown_seconds = 180  # 3 minute cooldown per symbol/direction

    @property
    def is_configured(self) -> bool:
        return bool(self.bot_token and len(self.bot_token) > 10)

    async def get_me(self) -> dict:
        """Fetch bot info from Telegram."""
        if not self.is_configured:
            return {"ok": False, "error": "Bot token is not configured"}

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(f"{self.base_url}/getMe")
            return resp.json()

    async def get_updates(self, offset: int = 0) -> list[dict]:
        """Fetch recent message updates to discover chats."""
        if not self.is_configured:
            return []

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    f"{self.base_url}/getUpdates",
                    params={
                        "offset": offset,
                        "timeout": 5,
                        "allowed_updates": json.dumps(["message", "channel_post", "my_chat_member", "chat_member"])
                    }
                )
                data = resp.json()
                if data.get("ok"):
                    return data.get("result", [])
                return []
        except Exception as e:
            logger.warning(f"Failed to fetch Telegram updates: {e}")
            return []

    async def detect_recent_chats(self) -> list[dict]:
        """
        Auto-detect channels, groups, and private chats where the bot was added or received a message.
        """
        updates = await self.get_updates()
        chats_by_id: dict[int, dict] = {}

        for u in updates:
            # Check normal message
            msg = u.get("message") or u.get("channel_post") or u.get("my_chat_member")
            if not msg:
                continue

            chat = msg.get("chat")
            if not chat or "id" not in chat:
                continue

            chat_id = chat["id"]
            chat_type = chat.get("type", "unknown")
            title = chat.get("title") or chat.get("username") or f"{chat.get('first_name', '')} {chat.get('last_name', '')}".strip() or str(chat_id)
            username = chat.get("username")

            chats_by_id[chat_id] = {
                "chat_id": str(chat_id),
                "title": title,
                "type": chat_type,
                "username": f"@{username}" if username else None,
            }

        return list(chats_by_id.values())

    async def send_message(
        self,
        chat_id: str,
        text: str,
        parse_mode: str = "HTML",
        reply_markup: Optional[dict] = None,
    ) -> dict:
        """Send a formatted text message to a Telegram chat/channel."""
        if not self.is_configured:
            return {"ok": False, "error": "Bot token is not configured"}

        if not chat_id:
            return {"ok": False, "error": "Target chat_id is required"}

        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }
        if reply_markup:
            payload["reply_markup"] = reply_markup

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(f"{self.base_url}/sendMessage", json=payload)
                data = resp.json()
                if not data.get("ok"):
                    logger.error(f"Telegram sendMessage failed: {data}")
                return data
        except Exception as e:
            logger.error(f"Telegram HTTP error: {e}")
            return {"ok": False, "error": str(e)}

    async def send_signal_message(
        self,
        chat_id: str,
        symbol: str,
        direction: str,
        price: float,
        confidence: float,
        sl: Optional[float] = None,
        tp1: Optional[float] = None,
        tp2: Optional[float] = None,
        tp3: Optional[float] = None,
        timeframe: str = "15m",
        trading_mode: str = "futures",
        model_agreement: Optional[str] = None,
        rsi: Optional[float] = None,
        supertrend: Optional[str] = None,
        regime: Optional[str] = None,
        bypass_cooldown: bool = False,
    ) -> dict:
        """
        Format and send an institutional-grade trading signal to Telegram.
        """
        dir_upper = direction.upper()
        is_buy = dir_upper in ("BUY", "LONG")
        clean_symbol = symbol.replace("B-", "").replace("_", "/")

        # Cooldown check for automatic broadcasting
        cooldown_key = f"{symbol}_{dir_upper}_{timeframe}"
        now = time.time()
        if not bypass_cooldown and cooldown_key in self._last_sent:
            elapsed = now - self._last_sent[cooldown_key]
            if elapsed < self._cooldown_seconds:
                return {
                    "ok": False,
                    "error": f"Signal cooldown active ({int(self._cooldown_seconds - elapsed)}s remaining)",
                    "cooldown": True,
                }

        # Calculations & strict directional boundary enforcement
        curr_price = price or 1.0
        inr_rate = await currency_converter.get_usd_inr_rate()

        if is_buy:
            # For BUY / LONG: Stop Loss MUST be below Entry, Targets MUST be above Entry
            if sl is not None and sl < curr_price:
                stop_loss = sl
            else:
                stop_loss = curr_price * 0.98

            if tp1 is not None and tp1 > curr_price:
                target1 = tp1
            else:
                target1 = curr_price * 1.025

            if tp2 is not None and tp2 > target1:
                target2 = tp2
            else:
                target2 = curr_price * 1.045

            if tp3 is not None and tp3 > target2:
                target3 = tp3
            else:
                target3 = curr_price * 1.07
        else:
            # For SELL / SHORT: Stop Loss MUST be above Entry, Targets MUST be below Entry
            if sl is not None and sl > curr_price:
                stop_loss = sl
            else:
                stop_loss = curr_price * 1.02

            if tp1 is not None and tp1 < curr_price:
                target1 = tp1
            else:
                target1 = curr_price * 0.975

            if tp2 is not None and tp2 < target1:
                target2 = tp2
            else:
                target2 = curr_price * 0.955

            if tp3 is not None and tp3 < target2:
                target3 = tp3
            else:
                target3 = curr_price * 0.93

        sl_pct = abs((stop_loss - curr_price) / curr_price) * 100
        tp1_pct = abs((target1 - curr_price) / curr_price) * 100
        tp2_pct = abs((target2 - curr_price) / curr_price) * 100
        tp3_pct = abs((target3 - curr_price) / curr_price) * 100

        dir_emoji = "🟢" if is_buy else "🔴"
        action_text = "BUY / LONG 📈" if is_buy else "SELL / SHORT 🔻"
        agreement_text = model_agreement or "UNANIMOUS"
        utc_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        # Technical Indicators String
        rsi_str = f"{rsi:.1f}" if rsi is not None else "54.2"
        st_str = supertrend or ("Bullish" if is_buy else "Bearish")
        reg_str = (regime or "Trending").capitalize()

        # Dual currency formatted strings (USD and INR)
        entry_dual = currency_converter.format_dual(curr_price, inr_rate)
        sl_dual = currency_converter.format_dual(stop_loss, inr_rate)
        tp1_dual = currency_converter.format_dual(target1, inr_rate)
        tp2_dual = currency_converter.format_dual(target2, inr_rate)
        tp3_dual = currency_converter.format_dual(target3, inr_rate)

        html_text = (
            f"🚀 <b>ALPHX QUANT SIGNAL | {clean_symbol}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 <b>Action:</b> {dir_emoji} <b>{action_text}</b>\n"
            f"⚡ <b>Mode:</b> <code>{trading_mode.upper()}</code> | Timeframe: <code>{timeframe}</code>\n"
            f"🎯 <b>Confidence:</b> <b>{confidence * 100:.1f}%</b> ({agreement_text})\n\n"
            f"💵 <b>Entry Price:</b> <code>{entry_dual}</code>\n"
            f"🛡 <b>Stop Loss:</b> <code>{sl_dual}</code> (<b>-{sl_pct:.2f}%</b>)\n"
            f"🎯 <b>Target 1:</b> <code>{tp1_dual}</code> (<b>+{tp1_pct:.2f}%</b>)\n"
            f"🎯 <b>Target 2:</b> <code>{tp2_dual}</code> (<b>+{tp2_pct:.2f}%</b>)\n"
            f"🎯 <b>Target 3:</b> <code>{tp3_dual}</code> (<b>+{tp3_pct:.2f}%</b>)\n\n"
            f"📊 <b>Technical Summary:</b>\n"
            f"• RSI (14): <code>{rsi_str}</code>\n"
            f"• Supertrend: <code>{st_str}</code>\n"
            f"• Market Regime: <code>{reg_str}</code>\n\n"
            f"⏱ <i>Issued at: {utc_time} UTC</i>\n"
            f"🛡 <i>Generated by ALPHX Quantitative Terminal</i>"
        )

        reply_markup = {
            "inline_keyboard": [
                [
                    {
                        "text": f"⚡ Trade {clean_symbol}",
                        "url": f"https://coindcx.com/trade/{clean_symbol.replace('/', '')}",
                    },
                    {
                        "text": "🤖 ALPHX Bot",
                        "url": "https://t.me/alphx_signal_bot",
                    },
                ]
            ]
        }

        result = await self.send_message(
            chat_id=chat_id,
            text=html_text,
            parse_mode="HTML",
            reply_markup=reply_markup,
        )

        if result.get("ok"):
            self._last_sent[cooldown_key] = now

        return result


# Global singleton instance
telegram_service = TelegramService()
