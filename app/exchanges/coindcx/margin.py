import logging
from typing import Optional
import httpx

from app.exchanges.base import (
    OrderSide, OrderType, OrderResult, ExchangeError, OrderError,
)
from app.exchanges.coindcx.auth import CoinDCXAuth
from app.exchanges.coindcx import constants as C

logger = logging.getLogger(__name__)


class CoinDCXMargin:
    """CoinDCX margin trading operations."""

    def __init__(self, client: httpx.AsyncClient, auth: CoinDCXAuth):
        self.client = client
        self.auth = auth

    async def _auth_post(self, url: str, body: dict) -> dict:
        """Make authenticated POST request."""
        json_body, headers = self.auth.prepare_body(body)
        resp = await self.client.post(url, content=json_body, headers=headers)
        if resp.status_code != 200:
            error = resp.json() if resp.text else {}
            raise OrderError(f"CoinDCX margin error: {error}", code=resp.status_code, raw=error)
        return resp.json()

    async def place_order(
        self,
        pair: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        leverage: float = 1.0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_sl: bool = False,
    ) -> OrderResult:
        """
        Place a margin order on CoinDCX.
        POST /exchange/v1/margin/create
        
        Supports: limit_order, market_order, stop_limit, take_profit
        ecode must be 'B' for all margin orders.
        """
        market = pair.split('-', 1)[1].replace('_', '') if '-' in pair else pair

        body = {
            'side': side.value,
            'order_type': order_type.value,
            'market': market,
            'quantity': quantity,
            'leverage': leverage,
            'ecode': C.DEFAULT_ECODE,
        }
        if price is not None and order_type != OrderType.MARKET:
            body['price'] = price
        if stop_loss is not None:
            body['sl_price'] = stop_loss
        if take_profit is not None:
            body['target_price'] = take_profit
        if trailing_sl:
            body['trailing_sl'] = True

        result = await self._auth_post(C.MARGIN_CREATE_URL, body)
        order = result[0] if isinstance(result, list) else result

        return OrderResult(
            order_id=str(order.get('id', '')),
            status=order.get('status', 'open'),
            side=side.value,
            order_type=order_type.value,
            price=price or float(order.get('price', 0)),
            quantity=quantity,
            filled_quantity=float(order.get('filled_quantity', 0)),
            average_price=float(order.get('avg_price', 0)),
            exchange='coindcx',
            trading_mode='margin',
            raw=order,
        )

    async def edit_stop_loss(self, order_id: str, stop_loss: float) -> bool:
        """Edit SL of a margin order. POST /exchange/v1/margin/edit_sl"""
        result = await self._auth_post(
            C.MARGIN_EDIT_SL_URL,
            {'id': order_id, 'sl_price': stop_loss},
        )
        return True

    async def edit_take_profit(self, order_id: str, take_profit: float) -> bool:
        """Edit TP of a margin order. POST /exchange/v1/margin/edit_target"""
        result = await self._auth_post(
            C.MARGIN_EDIT_TARGET_URL,
            {'id': order_id, 'target_price': take_profit},
        )
        return True

    async def exit_position(self, order_id: str) -> bool:
        """Exit/close a margin position. POST /exchange/v1/margin/exit"""
        result = await self._auth_post(
            C.MARGIN_EXIT_URL, {'id': order_id}
        )
        return True

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a margin order. POST /exchange/v1/margin/cancel"""
        result = await self._auth_post(
            C.MARGIN_CANCEL_URL, {'id': order_id}
        )
        return True

    async def fetch_orders(
        self, pair: Optional[str] = None
    ) -> list[OrderResult]:
        """Fetch margin orders. POST /exchange/v1/margin/fetch_orders"""
        body = {}
        if pair:
            market = pair.split('-', 1)[1].replace('_', '') if '-' in pair else pair
            body['market'] = market

        data = await self._auth_post(C.MARGIN_FETCH_ORDERS_URL, body)
        orders = data if isinstance(data, list) else data.get('orders', [])
        return [
            OrderResult(
                order_id=str(o.get('id', '')),
                status=o.get('status', ''),
                side=o.get('side', ''),
                order_type=o.get('order_type', ''),
                price=float(o.get('price', 0)),
                quantity=float(o.get('quantity', 0)),
                filled_quantity=float(o.get('filled_quantity', 0)),
                average_price=float(o.get('avg_price', 0)),
                exchange='coindcx',
                trading_mode='margin',
                raw=o,
            )
            for o in orders
        ]

    async def add_margin(self, order_id: str, amount: float) -> bool:
        """Add margin to a position. POST /exchange/v1/margin/add_margin"""
        result = await self._auth_post(
            C.MARGIN_ADD_MARGIN_URL,
            {'id': order_id, 'amount': amount},
        )
        return True

    async def remove_margin(self, order_id: str, amount: float) -> bool:
        """Remove margin from a position. POST /exchange/v1/margin/remove_margin"""
        result = await self._auth_post(
            C.MARGIN_REMOVE_MARGIN_URL,
            {'id': order_id, 'amount': amount},
        )
        return True
