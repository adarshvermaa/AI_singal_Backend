import logging
from typing import Optional
import httpx

from app.exchanges.base import (
    OrderSide, OrderType, MarginType, OrderResult, Position,
    ExchangeError, OrderError,
)
from app.exchanges.coindcx.auth import CoinDCXAuth
from app.exchanges.coindcx import constants as C

logger = logging.getLogger(__name__)


class CoinDCXFutures:
    """CoinDCX futures/derivatives trading operations."""

    def __init__(self, client: httpx.AsyncClient, auth: CoinDCXAuth):
        self.client = client
        self.auth = auth

    async def _auth_post(self, url: str, body: dict) -> dict:
        """Make authenticated POST request for CoinDCX futures (requires epoch seconds)."""
        json_body, headers = self.auth.prepare_body(body)
        resp = await self.client.post(url, content=json_body, headers=headers)
        if resp.status_code != 200:
            error = resp.json() if resp.text else {}
            raise OrderError(f"CoinDCX futures error: {error}", code=resp.status_code, raw=error)
        return resp.json()

    async def place_order(
        self,
        pair: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        leverage: float = 1.0,
        stop_price: Optional[float] = None,
        margin_type: MarginType = MarginType.ISOLATED,
    ) -> OrderResult:
        """
        Place a futures order on CoinDCX.
        POST /exchange/v1/derivatives/futures/orders/create
        
        Order types: limit_order, market_order, stop_limit, stop_market,
                     take_profit_limit, take_profit_market
        """
        order_body = {
            'side': side.value,
            'pair': pair,
            'order_type': order_type.value,
            'total_quantity': quantity,
            'leverage': str(int(leverage)),
            'notification': 'email_notification',
        }
        if price is not None and order_type not in (
            OrderType.MARKET, OrderType.STOP_MARKET, OrderType.TAKE_PROFIT_MARKET
        ):
            order_body['price'] = str(price)
        if stop_price is not None:
            order_body['stop_price'] = str(stop_price)
        if order_type not in (OrderType.MARKET, OrderType.STOP_MARKET, OrderType.TAKE_PROFIT_MARKET):
            order_body['time_in_force'] = 'good_till_cancel'

        result = await self._auth_post(
            C.FUTURES_CREATE_ORDER_URL, order_body
        )
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
            trading_mode='futures',
            raw=order,
        )

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a futures order."""
        await self._auth_post(
            C.FUTURES_CANCEL_ORDER_URL, {'id': order_id}
        )
        return True

    async def cancel_all_orders(self, pair: str) -> bool:
        """Cancel all active futures orders for a pair."""
        await self._auth_post(
            C.FUTURES_CANCEL_ALL_URL, {'pair': pair}
        )
        return True

    async def get_positions(self) -> list[Position]:
        """Get all active futures positions."""
        if not self.auth.api_key or not self.auth.api_secret:
            return []

        try:
            data = await self._auth_post(
                C.FUTURES_POSITIONS_URL,
                {
                    'page': '1',
                    'size': '100',
                    'margin_currency_short_name': ['USDT', 'INR'],
                },
            )
        except Exception as e:
            logger.warning(f"Error calling CoinDCX futures positions endpoint: {e}")
            return []

        if isinstance(data, list):
            positions_list = data
        elif isinstance(data, dict):
            positions_list = data.get('data', data.get('positions', []))
            if not isinstance(positions_list, list):
                positions_list = []
        else:
            positions_list = []

        positions: list[Position] = []
        for p in positions_list:
            try:
                active_pos = float(p.get('active_pos') or p.get('quantity') or 0)
                if active_pos == 0:
                    continue
                positions.append(
                    Position(
                        id=str(p.get('id', '')),
                        pair=p.get('pair', ''),
                        side='long' if active_pos > 0 else 'short',
                        quantity=abs(active_pos),
                        entry_price=float(p.get('avg_price') or p.get('entry_price') or 0),
                        mark_price=float(p.get('mark_price') or 0),
                        leverage=float(p.get('leverage') or 1),
                        unrealized_pnl=float(p.get('unrealised_pnl') or p.get('pnl') or 0),
                        liquidation_price=float(p.get('liquidation_price') or 0),
                        margin_type=p.get('margin_mode', 'isolated'),
                        take_profit=float(p.get('take_profit_price') or 0),
                        stop_loss=float(p.get('stop_loss_price') or 0),
                        raw=p,
                    )
                )
            except Exception as e:
                logger.debug(f"Skipping malformed position entry: {e}")
                continue

        return positions

    async def set_leverage(self, pair: str, leverage: float) -> bool:
        """Set leverage for a futures pair."""
        await self._auth_post(
            C.FUTURES_UPDATE_LEVERAGE_URL,
            {'pair': pair, 'leverage': str(int(leverage))},
        )
        return True

    async def set_tp_sl(
        self,
        position_id: str,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
    ) -> bool:
        """
        Set take profit and/or stop loss for a futures position.
        POST /exchange/v1/derivatives/futures/positions/create_tpsl
        """
        body = {'id': position_id}
        if take_profit is not None:
            body['take_profit'] = {
                'stop_price': str(take_profit),
                'order_type': 'take_profit_market',
            }
        if stop_loss is not None:
            body['stop_loss'] = {
                'stop_price': str(stop_loss),
                'order_type': 'stop_market',
            }
        await self._auth_post(C.FUTURES_CREATE_TPSL_URL, body)
        return True

    async def exit_position(self, position_id: str) -> bool:
        """Exit/close a futures position."""
        await self._auth_post(
            C.FUTURES_EXIT_POSITION_URL, {'id': position_id}
        )
        return True

    async def add_margin(self, position_id: str, amount: float) -> bool:
        """Add margin to a futures position."""
        await self._auth_post(
            C.FUTURES_ADD_MARGIN_URL,
            {'id': position_id, 'amount': str(amount)},
        )
        return True

    async def remove_margin(self, position_id: str, amount: float) -> bool:
        """Remove margin from a futures position."""
        await self._auth_post(
            C.FUTURES_REMOVE_MARGIN_URL,
            {'id': position_id, 'amount': str(amount)},
        )
        return True

    async def transfer_funds(
        self, amount: float, currency: str = 'USDT', to_futures: bool = True
    ) -> bool:
        """Transfer funds between spot and futures wallet."""
        await self._auth_post(
            C.FUTURES_WALLET_TRANSFER_URL,
            {
                'transfer_type': 'deposit' if to_futures else 'withdraw',
                'amount': amount,
                'currency_short_name': currency,
            },
        )
        return True

    async def get_active_orders(
        self, pair: Optional[str] = None
    ) -> list[OrderResult]:
        """Get active futures orders."""
        body = {}
        if pair:
            body['pair'] = pair
        data = await self._auth_post(C.FUTURES_ACTIVE_ORDERS_URL, body)
        orders_list = data if isinstance(data, list) else data.get('orders', [])
        return [
            OrderResult(
                order_id=str(o.get('id', '')),
                status=o.get('status', 'open'),
                side=o.get('side', ''),
                order_type=o.get('order_type', ''),
                price=float(o.get('price', 0)),
                quantity=float(o.get('total_quantity', 0)),
                filled_quantity=float(o.get('filled_quantity', 0)),
                average_price=float(o.get('avg_price', 0)),
                exchange='coindcx',
                trading_mode='futures',
                raw=o,
            )
            for o in orders_list
        ]

    async def get_cross_margin_details(self) -> dict:
        """Get cross margin details for futures."""
        return await self._auth_post(C.FUTURES_CROSS_MARGIN_URL, {})
