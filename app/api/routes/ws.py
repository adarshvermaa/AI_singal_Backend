import asyncio
import json
import logging
import time
from typing import Dict, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)
router = APIRouter()


class ConnectionManager:
    """Manages active frontend WebSocket client connections and channel subscriptions."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []
        # Channel name -> Set of WebSockets subscribed
        self.subscriptions: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Frontend WebSocket client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        for channel, subscribers in list(self.subscriptions.items()):
            subscribers.discard(websocket)
            if not subscribers:
                del self.subscriptions[channel]
        logger.info(f"Frontend WebSocket client disconnected. Remaining: {len(self.active_connections)}")

    def subscribe(self, websocket: WebSocket, channel: str):
        if channel not in self.subscriptions:
            self.subscriptions[channel] = set()
        self.subscriptions[channel].add(websocket)

    def unsubscribe(self, websocket: WebSocket, channel: str):
        if channel in self.subscriptions:
            self.subscriptions[channel].discard(websocket)

    async def broadcast_to_channel(self, channel: str, message: dict):
        if channel in self.subscriptions:
            dead_sockets = []
            for ws in self.subscriptions[channel]:
                try:
                    await ws.send_json(message)
                except Exception:
                    dead_sockets.append(ws)
            for ws in dead_sockets:
                self.disconnect(ws)

    async def broadcast_all(self, message: dict):
        dead_sockets = []
        for ws in self.active_connections:
            try:
                await ws.send_json(message)
            except Exception:
                dead_sockets.append(ws)
        for ws in dead_sockets:
            self.disconnect(ws)

    async def start_background_stream(self, app):
        """Continuously streams real-time prices for active ticker subscriptions."""
        logger.info("Starting background WebSocket ticker streaming task...")
        while True:
            try:
                await asyncio.sleep(2.5)
                ticker_channels = [c for c in list(self.subscriptions.keys()) if c.startswith("ticker:")]
                if not ticker_channels:
                    continue

                factory = getattr(app.state, "exchange_factory", None)
                if not factory:
                    continue

                try:
                    exch = factory.get_or_raise("coindcx")
                    all_tickers = await exch.get_all_tickers()
                    tickers_map = {t.pair: t for t in all_tickers}

                    for ch in ticker_channels:
                        sym = ch.split("ticker:", 1)[1]
                        clean_sym = sym.split('-', 1)[1].replace('_', '') if '-' in sym else sym
                        t = tickers_map.get(clean_sym) or tickers_map.get(sym)
                        if t and t.last_price:
                            await self.broadcast_to_channel(ch, {
                                "type": "price",
                                "symbol": sym,
                                "data": {
                                    "price": t.last_price,
                                    "change_24h": t.change_24h,
                                    "high_24h": t.high_24h,
                                    "low_24h": t.low_24h,
                                    "volume_24h": t.volume_24h,
                                    "timestamp": t.timestamp or int(time.time() * 1000),
                                }
                            })
                except Exception as e:
                    logger.debug(f"Live ticker stream cycle error: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in background ticker streaming loop: {e}")
                await asyncio.sleep(5)


manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Real-time streaming WebSocket endpoint for frontend clients.
    """
    await manager.connect(websocket)
    try:
        while True:
            raw_text = await websocket.receive_text()
            try:
                payload = json.loads(raw_text)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
                continue

            action = payload.get("action")
            channel = payload.get("channel")

            if action == "ping":
                await websocket.send_json({"type": "pong", "timestamp": int(time.time() * 1000)})
            elif action == "subscribe" and channel:
                manager.subscribe(websocket, channel)
                await websocket.send_json({"type": "subscribed", "channel": channel})

                # Instant snapshot for ticker channel
                if channel.startswith("ticker:"):
                    sym = channel.split("ticker:", 1)[1]
                    try:
                        factory = getattr(websocket.app.state, "exchange_factory", None)
                        if factory:
                            exch = factory.get_or_raise("coindcx")
                            clean_sym = sym.split('-', 1)[1].replace('_', '') if '-' in sym else sym
                            t = await exch.get_ticker(sym)
                            if t and t.last_price:
                                await websocket.send_json({
                                    "type": "price",
                                    "symbol": sym,
                                    "data": {
                                        "price": t.last_price,
                                        "change_24h": t.change_24h,
                                        "timestamp": t.timestamp or int(time.time() * 1000),
                                    }
                                })
                    except Exception:
                        pass

            elif action == "unsubscribe" and channel:
                manager.unsubscribe(websocket, channel)
                await websocket.send_json({"type": "unsubscribed", "channel": channel})
            else:
                await websocket.send_json({"error": f"Unknown action: {action}"})

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)
