import logging
import json
from typing import Optional
import httpx

from app.exchanges.base import (
    OrderSide, OrderType, OrderResult, ExchangeError, OrderError,
)
from app.exchanges.coindcx.auth import CoinDCXAuth
from app.exchanges.coindcx import constants as C

logger = logging.getLogger(__name__)


class CoinDCXSpot:
    """CoinDCX spot trading operations."""

    def __init__(self, client: httpx.AsyncClient, auth: CoinDCXAuth):
        self.client = client
        self.auth = auth

    async def _auth_post(self, url: str, body: dict) -> dict:
        """Make authenticated POST request."""
        json_body, headers = self.auth.prepare_body(body)
        resp = await self.client.post(url, content=json_body, headers=headers)
        if resp.status_code != 200:
            error = resp.json() if resp.text else {}
            raise OrderError(f"CoinDCX spot error: {error}", code=resp.status_code, raw=error)
        return resp.json()

    async def place_order(
        self,
        pair: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
    ) -> OrderResult:
        """
        Place a spot order on CoinDCX.
        POST /exchange/v1/orders/create
        
        Required fields: side, order_type, market, total_quantity
        For limit orders: price_per_unit is required
        
        market format: "BTCUSDT" (not "B-BTC_USDT")
        """
        # Convert pair format: "B-BTC_USDT" -> "BTCUSDT"
        market = pair.split('-', 1)[1].replace('_', '') if '-' in pair else pair

        body = {
            'side': side.value,
            'order_type': order_type.value,
            'market': market,
            'total_quantity': quantity,
        }
        if price is not None and order_type != OrderType.MARKET:
            body['price_per_unit'] = price

        result = await self._auth_post(C.SPOT_CREATE_ORDER_URL, body)
        order = result[0] if isinstance(result, list) else result
        
        return OrderResult(
            order_id=str(order.get('id', '')),
            status=order.get('status', 'open'),
            side=side.value,
            order_type=order_type.value,
            price=price or float(order.get('price_per_unit', 0)),
            quantity=quantity,
            filled_quantity=float(order.get('filled_quantity', 0)),
            average_price=float(order.get('avg_price', 0)),
            exchange='coindcx',
            trading_mode='spot',
            raw=order,
        )

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a spot order. POST /exchange/v1/orders/cancel"""
        result = await self._auth_post(
            C.SPOT_CANCEL_ORDER_URL, {'id': order_id}
        )
        return True

    async def get_order_status(self, order_id: str) -> OrderResult:
        """Get status of a spot order. POST /exchange/v1/orders/status"""
        result = await self._auth_post(
            C.SPOT_ORDER_STATUS_URL, {'id': order_id}
        )
        return OrderResult(
            order_id=str(result.get('id', order_id)),
            status=result.get('status', 'unknown'),
            side=result.get('side', ''),
            order_type=result.get('order_type', ''),
            price=float(result.get('price_per_unit', 0)),
            quantity=float(result.get('total_quantity', 0)),
            filled_quantity=float(result.get('filled_quantity', 0)),
            average_price=float(result.get('avg_price', 0)),
            exchange='coindcx',
            trading_mode='spot',
            raw=result,
        )

    async def get_active_orders(
        self, pair: Optional[str] = None
    ) -> list[OrderResult]:
        """Get active spot orders. POST /exchange/v1/orders/active_orders"""
        body = {}
        if pair:
            market = pair.split('-', 1)[1].replace('_', '') if '-' in pair else pair
            body['market'] = market
        body['side'] = 'buy'  # fetch both sides
        
        results = []
        for side in ['buy', 'sell']:
            body_copy = {**body, 'side': side}
            try:
                data = await self._auth_post(C.SPOT_ACTIVE_ORDERS_URL, body_copy)
                orders = data.get('orders', data) if isinstance(data, dict) else data
                if isinstance(orders, list):
                    for o in orders:
                        results.append(OrderResult(
                            order_id=str(o.get('id', '')),
                            status=o.get('status', 'open'),
                            side=o.get('side', side),
                            order_type=o.get('order_type', ''),
                            price=float(o.get('price_per_unit', 0)),
                            quantity=float(o.get('total_quantity', 0)),
                            filled_quantity=float(o.get('filled_quantity', 0)),
                            average_price=float(o.get('avg_price', 0)),
                            exchange='coindcx',
                            trading_mode='spot',
                            raw=o,
                        ))
            except ExchangeError:
                pass
        return results

    async def cancel_all_orders(
        self, pair: Optional[str] = None
    ) -> bool:
        """Cancel all active spot orders. POST /exchange/v1/orders/cancel_all"""
        body = {}
        if pair:
            market = pair.split('-', 1)[1].replace('_', '') if '-' in pair else pair
            body['market'] = market
        body['side'] = 'buy'
        
        for side in ['buy', 'sell']:
            body_copy = {**body, 'side': side}
            try:
                await self._auth_post(C.SPOT_CANCEL_ALL_URL, body_copy)
            except ExchangeError:
                pass
        return True
