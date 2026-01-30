"""
═══════════════════════════════════════════════════════════════
⚡ PART 3: SMART EXECUTION ENGINE - FIXED & COMPLETED FOR PYTHON 3.8
═══════════════════════════════════════════════════════════════
- Hoàn thiện phần bị truncated
- Thêm rounding chính xác theo Binance rules
- Thêm retry & error handling cho place order
- Hoàn thiện TWAP, Iceberg, emergency close
"""

import time
import logging
import numpy as np
import uuid
from decimal import Decimal, ROUND_DOWN
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

from binance.client import Client
from binance.exceptions import BinanceAPIException

from error_handling import handle_errors, OrderExecutionError, APIError

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    STOP_LIMIT = "STOP_LIMIT"
    POST_ONLY = "POST_ONLY"
    IOC = "IOC"
    FOK = "FOK"

class ExecutionStrategy(Enum):
    AGGRESSIVE = "AGGRESSIVE"
    PASSIVE = "PASSIVE"
    TWAP = "TWAP"
    VWAP = "VWAP"
    ICEBERG = "ICEBERG"
    ADAPTIVE = "ADAPTIVE"

class OrderState(Enum):
    PENDING = "PENDING"
    VALIDATING = "VALIDATING"
    SENDING = "SENDING"
    SENT = "SENT"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"

@dataclass
class Order:
    symbol: str
    side: str  # BUY / SELL
    order_type: OrderType
    quantity: float
    
    price: Optional[float] = None
    stop_price: Optional[float] = None
    reduce_only: bool = False
    post_only: bool = False
    time_in_force: str = "GTC"
    
    execution_strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE
    
    state: OrderState = OrderState.PENDING
    order_id: Optional[int] = None
    client_order_id: Optional[str] = None
    
    filled_qty: float = 0.0
    avg_price: float = 0.0
    cumulative_quote_qty: float = 0.0
    
    created_at: datetime = None
    sent_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    
    latency_ms: float = 0.0
    slippage_pct: float = 0.0
    expected_price: float = 0.0
    fees_paid: float = 0.0
    
    error_message: str = ""
    retry_count: int = 0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.client_order_id is None:
            # CIO-grade uniqueness: avoid duplicate clientOrderId across threads/restarts
            self.client_order_id = f"{self.symbol.replace('/', '')}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:10]}"

@dataclass
class Position:
    symbol: str
    side: str  # LONG / SHORT
    entry_price: float
    quantity: float
    leverage: int = 10
    
    stop_loss: float = 0.0
    take_profit: float = 0.0
    trailing_stop_pct: float = 0.0
    breakeven_trigger_pct: float = 0.0
    
    entry_time: datetime = None
    highest_price: float = 0.0
    lowest_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    risk_amount: float = 0.0
    risk_pct: float = 0.0
    max_adverse_excursion: float = 0.0
    max_favorable_excursion: float = 0.0
    
    entry_reason: str = ""
    regime: str = ""
    confidence: float = 0.0

    # Pyramiding (add-to-winner)
    pyramid_count: int = 0
    last_add_ts: float = 0.0
    
    def __post_init__(self):
        if self.entry_time is None:
            self.entry_time = datetime.now()
        self.highest_price = self.entry_price
        self.lowest_price = self.entry_price

class SmartExecutionEngine:
    """Smart execution engine with coin-specific optimization & advanced strategies"""
    
    def __init__(self, client: Client, market_data, default_leverage: int = 10, force_cross_margin: bool = True):
        self.client = client
        self.market_data = market_data

        self.active_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.positions: Dict[str, Position] = {}

        # Live trading defaults (can be overridden by TradingSystem)
        try:
            self.default_leverage = int(default_leverage or 10)
        except Exception:
            self.default_leverage = 10
        self.force_cross_margin = bool(force_cross_margin)
        

        # Cache exchange info (giảm gọi API, tăng tốc)
        self._symbol_info_cache: Dict[str, Dict] = {}
        self._symbol_info_cache_ts: float = 0.0
        self._symbol_info_cache_ttl: float = 30.0 * 60.0  # 30 phút
        self._margin_leverage_cache = {}  # symbol -> {'cross': bool, 'lev': int}
        
        self.latency_tracker = []
        self.slippage_tracker = []
        self.fill_rate = 0.0
        self.total_fees = 0.0
        
        self.max_retries = 5
        self.retry_delays = [0.1, 0.5, 1.0, 2.0, 5.0]
        
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        self.maker_fee = 0.0002  # 0.02%
        self.taker_fee = 0.0004  # 0.04%
        
        # Coin-specific execution params
        self.coin_params = {
            'BTCUSDT': {'slippage_tolerance': 0.02, 'urgency': 'medium', 'min_qty_step': 0.001},
            'ETHUSDT': {'slippage_tolerance': 0.03, 'urgency': 'medium', 'min_qty_step': 0.001},
            'SOLUSDT': {'slippage_tolerance': 0.05, 'urgency': 'low', 'min_qty_step': 0.1},
            'XRPUSDT': {'slippage_tolerance': 0.05, 'urgency': 'low', 'min_qty_step': 10.0}
        }
        
        logging.info("⚡ Smart Execution Engine initialized")

    def _ensure_cross_and_leverage(self, symbol: str, leverage: Optional[int] = None):
        """Best-effort safety for Binance USDT-M futures.

        - Ensure CROSS margin type (if enabled)
        - Ensure leverage (best-effort)

        NOTE: Binance may return errors like "No need to change margin type";
        treat them as success.
        """
        lev = None
        try:
            lev = int(leverage or self.default_leverage or 10)
        except Exception:
            lev = int(self.default_leverage or 10)

        cache = self._margin_leverage_cache.get(symbol, {})
        need_cross = self.force_cross_margin and not cache.get('cross', False)
        need_lev = (cache.get('lev') != lev)

        if not (need_cross or need_lev):
            return

        # CROSS margin
        if need_cross:
            try:
                self.client.futures_change_margin_type(symbol=symbol, marginType='CROSSED')
                cache['cross'] = True
            except Exception as e:
                msg = str(e)
                # Common benign cases
                if 'No need to change margin type' in msg or 'margin type' in msg:
                    cache['cross'] = True
                else:
                    logging.warning(f"[LIVE] Could not set CROSS margin for {symbol}: {e}")

        # Leverage
        if need_lev:
            try:
                self.client.futures_change_leverage(symbol=symbol, leverage=lev)
                cache['lev'] = lev
            except Exception as e:
                logging.warning(f"[LIVE] Could not set leverage={lev} for {symbol}: {e}")

        self._margin_leverage_cache[symbol] = cache

    def _wait_futures_fill(self, symbol: str, order_id: int, timeout_s: float = 8.0, poll_s: float = 0.25) -> Dict:
        """Poll futures order until FILLED / PARTIALLY_FILLED / CANCELED / REJECTED.

        Returns raw order dict from Binance.
        """
        t0 = time.time()
        last = None
        while (time.time() - t0) < timeout_s:
            try:
                last = self.client.futures_get_order(symbol=symbol, orderId=order_id)
                status = str(last.get('status', '')).upper()
                if status in ('FILLED', 'PARTIALLY_FILLED', 'CANCELED', 'REJECTED', 'EXPIRED'):
                    return last
            except Exception:
                # transient; retry
                pass
            time.sleep(poll_s)
        return last or {}

    def reconcile_live_position(self, symbol: str, expected_side: Optional[str] = None) -> Dict:
        """Fetch current futures position for symbol (best-effort).

        Returns dict with fields: positionAmt, entryPrice, unRealizedProfit, side.
        """
        try:
            pos_list = self.client.futures_position_information(symbol=symbol)
            if not pos_list:
                return {}
            p = pos_list[0]
            amt = float(p.get('positionAmt', 0) or 0)
            side = 'LONG' if amt > 0 else ('SHORT' if amt < 0 else 'FLAT')
            if expected_side and side not in ('FLAT', str(expected_side).upper()):
                # side mismatch
                return dict(p, _side=side, _mismatch=True)
            return dict(p, _side=side)
        except Exception as e:
            logging.warning(f"[LIVE] reconcile position failed for {symbol}: {e}")
            return {}

    def close_live_position(self, symbol: str, qty: Optional[float] = None, timeout_s: float = 10.0) -> Dict:
        """Close an existing Binance USDT-M futures position using MARKET reduceOnly.

        CIO-grade behavior:
        - Fetch current position from exchange (exchange is truth)
        - Send opposite MARKET order with reduceOnly=True
        - Wait for order fill best-effort
        - Reconcile position again; if still open beyond tolerance -> raise

        Returns: {'symbol','orderId','executedQty','avgPrice','status','beforeAmt','afterAmt'}
        """
        if not symbol:
            raise OrderExecutionError('Missing symbol')

        p = self.reconcile_live_position(symbol)
        try:
            amt0 = float(p.get('positionAmt', 0.0) or 0.0)
        except Exception:
            amt0 = 0.0
        if amt0 == 0.0:
            return {'symbol': symbol, 'status': 'FLAT', 'beforeAmt': 0.0, 'afterAmt': 0.0}

        side = 'SELL' if amt0 > 0 else 'BUY'
        q = abs(amt0)
        if qty is not None:
            try:
                q = min(q, abs(float(qty)))
            except Exception:
                pass
        if q <= 0:
            return {'symbol': symbol, 'status': 'FLAT', 'beforeAmt': float(amt0), 'afterAmt': float(amt0)}

        # Best-effort: ensure CROSS/leverage cache is warm
        try:
            self._ensure_cross_and_leverage(symbol)
        except Exception:
            pass

        o = self.place_order(symbol=symbol, side=side, quantity=float(q), order_type=OrderType.MARKET, reduce_only=True)
        if o is None or o.order_id is None:
            raise OrderExecutionError(f"Failed to send reduceOnly close for {symbol}")

        raw = {}
        try:
            raw = self._wait_futures_fill(symbol, int(o.order_id), timeout_s=timeout_s, poll_s=0.25) or {}
        except Exception:
            raw = {}

        try:
            executed = float(raw.get('executedQty', 0) or 0)
        except Exception:
            executed = 0.0
        try:
            avgp = float(raw.get('avgPrice', 0) or 0)
        except Exception:
            avgp = 0.0
        status = str(raw.get('status', '') or '').upper() or 'UNKNOWN'

        # Reconcile after close
        p1 = self.reconcile_live_position(symbol)
        try:
            amt1 = float(p1.get('positionAmt', 0.0) or 0.0)
        except Exception:
            amt1 = 0.0

        # Tolerance: after a reduceOnly close, position should move toward 0.
        # For partial closes, amt1 can be non-zero.
        if qty is None:
            # full close expected
            if abs(amt1) > 1e-12:
                raise OrderExecutionError(f"Close verify failed for {symbol}: before={amt0} after={amt1} status={status}")
        else:
            # partial close expected: ensure absolute position decreased
            if abs(amt1) >= abs(amt0) - 1e-12:
                raise OrderExecutionError(f"Partial close verify failed for {symbol}: before={amt0} after={amt1} status={status}")

        return {
            'symbol': symbol,
            'orderId': int(o.order_id),
            'executedQty': executed,
            'avgPrice': avgp,
            'status': status,
            'beforeAmt': float(amt0),
            'afterAmt': float(amt1),
        }
    
    def _calculate_latency(self, start_time: float) -> float:
        return (time.time() - start_time) * 1000


    def _get_best_bid_ask_safe(self, symbol: str) -> Dict:
        """Best-effort best bid/ask snapshot from MarketData.

        Expected keys from market_data.get_best_bid_ask(symbol):
        - bid, ask, mid, spread_pct
        """
        try:
            if self.market_data is None:
                return {}
            ba = self.market_data.get_best_bid_ask(symbol)
            if isinstance(ba, dict):
                return ba
        except Exception:
            pass
        # Fallback to mid
        try:
            mid = float(self.market_data.get_current_price(symbol) or 0.0)
            if mid > 0:
                return {'bid': mid, 'ask': mid, 'mid': mid, 'spread_pct': 0.0}
        except Exception:
            pass
        return {}

    def _live_entry_post_only(self, symbol: str, side: str, qty: float, timeout_s: float = 4.0) -> Tuple[Optional[Order], Dict]:
        """Try to open position using a post-only LIMIT (maker) to reduce fees.

        Returns (order, raw_futures_order_dict).
        On failure or not-filled, returns (None, {}). Caller may fall back to MARKET.
        """
        ba = self._get_best_bid_ask_safe(symbol)
        bid = float(ba.get('bid', 0.0) or 0.0)
        ask = float(ba.get('ask', 0.0) or 0.0)
        if bid <= 0 or ask <= 0:
            return None, {}

        side_u = str(side).upper()
        if side_u == 'BUY':
            price = bid  # to ensure maker (<= best bid)
        else:
            price = ask  # to ensure maker (>= best ask)

        # Place post-only LIMIT
        o = self.place_order(
            symbol=symbol,
            side=side_u,
            quantity=qty,
            price=price,
            order_type=OrderType.LIMIT,
            post_only=True,
            reduce_only=False,
        )
        if o is None or o.order_id is None:
            return None, {}

        raw = self._wait_futures_fill(symbol, int(o.order_id), timeout_s=float(timeout_s), poll_s=0.25)
        st = str((raw or {}).get('status', '')).upper()
        if st == 'FILLED':
            try:
                o.filled_qty = float(raw.get('executedQty', 0.0) or 0.0)
            except Exception:
                pass
            try:
                o.avg_price = float(raw.get('avgPrice', 0.0) or 0.0)
            except Exception:
                pass
            return o, raw or {}

        # Not filled -> cancel to avoid hanging orders
        try:
            self.cancel_order(symbol, int(o.order_id))
        except Exception:
            pass
        return None, raw or {}

    def _place_live_brackets(self, symbol: str, position_side: str, qty: float, stop_price: float, take_profit: float):
        """Best-effort protective orders on Binance Futures.

        - Stop loss: STOP_MARKET reduceOnly using MARK_PRICE trigger
        - Take profit: LIMIT reduceOnly (GTC)

        Notes:
        - We keep it minimal to avoid breaking on accounts that disallow certain params.
        - Caller is responsible for cancelling these when position closes.
        """
        if qty <= 0:
            return
        pos_side = str(position_side).upper()
        if pos_side not in ('LONG', 'SHORT'):
            return
        close_side = 'SELL' if pos_side == 'LONG' else 'BUY'

        # STOP_MARKET (reduceOnly)
        try:
            if stop_price and stop_price > 0:
                self.place_order(
                    symbol=symbol,
                    side=close_side,
                    quantity=qty,
                    stop_price=float(stop_price),
                    order_type=OrderType.STOP_MARKET,
                    reduce_only=True,
                    extra_params={'workingType': 'MARK_PRICE'}
                )
        except Exception as e:
            logging.warning(f"[LIVE] place SL failed for {symbol}: {e}")

        # TAKE PROFIT LIMIT (reduceOnly)
        try:
            if take_profit and take_profit > 0:
                self.place_order(
                    symbol=symbol,
                    side=close_side,
                    quantity=qty,
                    price=float(take_profit),
                    order_type=OrderType.LIMIT,
                    reduce_only=True,
                    post_only=False,
                )
        except Exception as e:
            logging.warning(f"[LIVE] place TP failed for {symbol}: {e}")

    # ===================== HIGH-LEVEL TRADE API (PAPER/LIVE) =====================

    def execute_trade(self, signal, mode: str = 'PAPER'):
        """Thực thi trade từ TradingSignal.

        - PAPER: mô phỏng mở position, KHÔNG gửi lệnh lên Binance.
        - LIVE : gửi MARKET order (futures) nếu có API keys; nếu fail sẽ raise.

        Yêu cầu signal có các thuộc tính: symbol, action(LONG/SHORT), entry_price,
        stop_loss, take_profit. Quantity ưu tiên signal.position_qty.
        """
        symbol = getattr(signal, 'symbol', None) or getattr(signal, 'ticker', None)
        if not symbol:
            raise ValueError('Signal missing symbol')

        side = getattr(signal, 'action', None) or getattr(signal, 'side', None)
        if side not in ('LONG', 'SHORT'):
            raise ValueError(f'Invalid signal side: {side}')

        qty = float(getattr(signal, 'position_qty', 0.0) or 0.0)
        if qty <= 0:
            # fallback: try to infer a small qty that will be rounded by Binance rules
            qty = 0.01

        entry_price = float(getattr(signal, 'entry_price', 0.0) or 0.0)
        sl = float(getattr(signal, 'stop_loss', 0.0) or 0.0)
        tp = float(getattr(signal, 'take_profit', 0.0) or 0.0)

        mode_u = str(mode or 'PAPER').upper()

        # ---------------------------
        # PAPER MODE (SIMULATION)
        # ---------------------------
        if mode_u == 'PAPER':
            # Nếu đã có position: cho phép pyramiding (add-to-winner) khi signal cho phép
            if symbol in self.positions:
                pos = self.positions.get(symbol)
                allow_add = bool(getattr(signal, 'allow_add', False))
                if allow_add and pos is not None and str(getattr(pos, 'side', '')).upper() == side:
                    try:
                        add_qty = float(getattr(signal, 'position_qty', 0.0) or 0.0)
                    except Exception:
                        add_qty = 0.0
                    if add_qty <= 0:
                        logging.info(f"[PAPER] Skip add {symbol}: add_qty<=0")
                        return

                    fill_price = self._paper_fill_price(symbol, side, entry_price)

                    # Keep existing brackets, but re-align to fill just in case
                    sl2, tp2 = self._align_brackets_with_fill(
                        side=side,
                        fill_price=float(fill_price),
                        signal_entry=float(pos.entry_price or fill_price),
                        stop_loss=float(getattr(pos, 'stop_loss', sl) or sl),
                        take_profit=float(getattr(pos, 'take_profit', tp) or tp),
                    )

                    old_qty = float(getattr(pos, 'quantity', 0.0) or 0.0)
                    new_qty = old_qty + float(add_qty)
                    if new_qty <= 0:
                        return
                    # Weighted average entry
                    new_entry = (float(pos.entry_price) * old_qty + float(fill_price) * float(add_qty)) / new_qty
                    pos.entry_price = float(new_entry)
                    pos.quantity = float(new_qty)
                    pos.stop_loss = float(sl2)
                    pos.take_profit = float(tp2)
                    try:
                        pos.pyramid_count = int(getattr(pos, 'pyramid_count', 0) or 0) + 1
                        pos.last_add_ts = float(time.time())
                    except Exception:
                        pass

                    logging.info(
                        f"[PAPER] ADD {side} {symbol} @ {fill_price:.4f} add_qty={add_qty:.8f} new_qty={new_qty:.8f} avg_entry={pos.entry_price:.4f} SL={pos.stop_loss:.4f} TP={pos.take_profit:.4f}"
                    )
                    return

                logging.info(f"[PAPER] Skip {symbol}: position already open")
                return

            # Paper fill model: pessimistic spread + slippage to be closer to LIVE
            fill_price = self._paper_fill_price(symbol, side, entry_price)

            # IMPORTANT: SL/TP in signal are typically computed from signal.entry_price.
            # When paper fill deviates (spread/slippage), brackets can become invalid
            # (e.g., LONG but TP < fill). Re-align brackets to the actual fill while
            # preserving the intended distance/ratios.
            sl, tp = self._align_brackets_with_fill(
                side=side,
                fill_price=float(fill_price),
                signal_entry=float(entry_price),
                stop_loss=float(sl),
                take_profit=float(tp),
            )

            self.positions[symbol] = Position(
                symbol=symbol,
                side=side,
                entry_price=fill_price,
                quantity=qty,
                stop_loss=sl,
                take_profit=tp,
                entry_reason=str(getattr(signal, 'reason', '')),
                confidence=float(getattr(signal, 'confidence', 0.0) or 0.0),
            )
            try:
                self.positions[symbol].mode = 'paper'
            except Exception:
                pass

            # Persist initial risk so RR/BE/Trailing doesn't get distorted when SL moves
            try:
                pos0 = self.positions.get(symbol)
                if pos0 is not None:
                    pos0.initial_stop_loss = float(getattr(pos0, 'stop_loss', 0.0) or 0.0)
                    pos0.initial_take_profit = float(getattr(pos0, 'take_profit', 0.0) or 0.0)
                    pos0.initial_risk = abs(float(getattr(pos0, 'entry_price', 0.0) or 0.0) - float(getattr(pos0, 'initial_stop_loss', 0.0) or 0.0))
            except Exception:
                pass

            # Attach extra context for self-training / analytics (safe even if Position doesn't define these fields)
            try:
                self.positions[symbol].regime = str(getattr(signal, 'regime', '') or '')
                self.positions[symbol].confidence = float(getattr(signal, 'confidence', 0.0) or 0.0)
                self.positions[symbol].entry_reason = str(getattr(signal, 'reason', '') or getattr(signal, 'entry_reason', '') or '')
                self.positions[symbol].features = getattr(signal, 'features', {}) or {}
            except Exception:
                pass

            logging.info(
                f"[PAPER] Open {side} {symbol} @ {fill_price:.4f} qty={qty:.8f} SL={sl:.4f} TP={tp:.4f}"
            )
            return

        # ---------------------------
        # LIVE MODE (BINANCE FUTURES)
        # ---------------------------
        # Map LONG/SHORT -> BUY/SELL
        order_side = 'BUY' if side == 'LONG' else 'SELL'

        # If position already exists in LIVE snapshot: allow pyramiding when requested
        if symbol in self.positions and bool(getattr(signal, 'allow_add', False)):
            pos0 = self.positions.get(symbol)
            if pos0 is not None and str(getattr(pos0, 'side', '')).upper() == side:
                # We will execute an additional market entry and then refresh brackets
                try:
                    add_qty = float(getattr(signal, 'position_qty', 0.0) or 0.0)
                except Exception:
                    add_qty = 0.0
                if add_qty > 0:
                    try:
                        self._ensure_cross_and_leverage(symbol, leverage=int(getattr(signal, 'leverage', 0) or 0) or None)
                    except Exception:
                        pass

                    # Execute add using MARKET for certainty (trend add-to-winner)
                    o_add = self.place_order(
                        symbol=symbol,
                        side=order_side,
                        order_type=OrderType.MARKET,
                        quantity=float(add_qty),
                        price=None,
                        stop_price=None,
                        reduce_only=False,
                        post_only=False,
                    )
                    if o_add is None or getattr(o_add, 'order_id', None) is None:
                        raise OrderExecutionError(f"LIVE add failed for {symbol}")

                    # Reconcile to get new avg entry/size
                    rp = self.reconcile_live_position(symbol, expected_side=side)
                    try:
                        amt = float(rp.get('positionAmt', 0.0) or 0.0)
                    except Exception:
                        amt = 0.0
                    if amt == 0.0:
                        raise OrderExecutionError(f"LIVE add opened no position for {symbol}")

                    try:
                        entry_px = float(rp.get('entryPrice', 0.0) or 0.0)
                    except Exception:
                        entry_px = 0.0
                    if entry_px <= 0:
                        entry_px = float(getattr(pos0, 'entry_price', 0.0) or 0.0)

                    # Keep existing SL/TP (structural) but align to new avg entry if needed
                    sl_keep = float(getattr(pos0, 'stop_loss', sl) or sl)
                    tp_keep = float(getattr(pos0, 'take_profit', tp) or tp)
                    sl2, tp2 = self._align_brackets_with_fill(
                        side=side,
                        fill_price=float(entry_px),
                        signal_entry=float(getattr(pos0, 'entry_price', entry_px) or entry_px),
                        stop_loss=float(sl_keep),
                        take_profit=float(tp_keep),
                    )

                    # Update snapshot
                    pos0.entry_price = float(entry_px)
                    pos0.quantity = float(abs(amt))
                    pos0.stop_loss = float(sl2)
                    pos0.take_profit = float(tp2)
                    try:
                        pos0.pyramid_count = int(getattr(pos0, 'pyramid_count', 0) or 0) + 1
                        pos0.last_add_ts = float(time.time())
                    except Exception:
                        pass

                    # Refresh protective orders for full size
                    try:
                        self.cancel_all_orders(symbol)
                    except Exception:
                        pass
                    try:
                        self._place_live_brackets(symbol, side, float(abs(amt)), float(sl2), float(tp2))
                    except Exception:
                        pass

                    logging.info(
                        f"[LIVE] ADD {side} {symbol} add_qty={float(add_qty):.8f} new_qty={float(abs(amt)):.8f} avg_entry={float(entry_px):.4f}"
                    )
                    return

        # Best-effort: enforce CROSS margin + leverage before placing order
        try:
            lev = int(getattr(signal, 'leverage', 0) or 0)
        except Exception:
            lev = 0
        try:
            self._ensure_cross_and_leverage(symbol, leverage=(lev if lev > 0 else None))
        except Exception:
            pass

        # Maker-first entry to reduce fees (CIO-grade cost control)
        # 1) Try post-only LIMIT at best bid/ask (maker)
        order = None
        raw_fill = {}
        try:
            o2, raw2 = self._live_entry_post_only(symbol, order_side, qty, timeout_s=float(getattr(signal, 'maker_timeout_s', 4.0) or 4.0))
            if o2 is not None:
                order = o2
                raw_fill = raw2 or {}
        except Exception as e:
            logging.warning(f"[LIVE] maker entry failed for {symbol}: {e}")

        # 2) Fallback to MARKET if not filled (configurable)
        if order is None:
            allow_fallback = True
            try:
                allow_fallback = bool(getattr(signal, 'maker_fallback_market', True))
            except Exception:
                allow_fallback = True
            if not allow_fallback:
                raise OrderExecutionError(f"Maker entry not filled and fallback disabled for {symbol}")
            order = self.place_order(symbol=symbol, side=order_side, quantity=qty, order_type=OrderType.MARKET)
            if order is None:
                raise OrderExecutionError(f"Failed to place order for {symbol}")


        # Reconcile fill (avgPrice, executedQty) to avoid ghost positions
        try:
            raw = self._wait_futures_fill(symbol, int(order.order_id), timeout_s=8.0, poll_s=0.25)
            if raw:
                try:
                    executed = float(raw.get('executedQty', 0) or 0)
                except Exception:
                    executed = 0.0
                try:
                    avgp = float(raw.get('avgPrice', 0) or 0)
                except Exception:
                    avgp = 0.0

                if executed > 0:
                    order.filled_qty = executed
                if avgp > 0:
                    order.avg_price = avgp

                st = str(raw.get('status', '')).upper()
                if st in ('CANCELED', 'REJECTED', 'EXPIRED'):
                    raise OrderExecutionError(f"Order {order.order_id} {st} for {symbol}")
        except Exception as e:
            logging.warning(f"[LIVE] Fill reconcile warning for {symbol} (order_id={order.order_id}): {e}")

        # Position reconciliation: verify Binance position direction & size after MARKET order
        try:
            pos = self.reconcile_live_position(symbol, expected_side=side)
            amt = float(pos.get('positionAmt', 0.0) or 0.0)
            if amt == 0.0:
                raise OrderExecutionError(f"No live position opened for {symbol} after order_id={order.order_id}")
            if side == 'LONG' and amt < 0:
                raise OrderExecutionError(f"Side mismatch: expected LONG but Binance positionAmt={amt} for {symbol}")
            if side == 'SHORT' and amt > 0:
                raise OrderExecutionError(f"Side mismatch: expected SHORT but Binance positionAmt={amt} for {symbol}")
        except Exception as e:
            # Strong safety: if we can't confirm the position, raise to allow outer layer to freeze live
            raise OrderExecutionError(str(e))

        # Align brackets to *actual* fill for LIVE as well (avoid TP<fill / SL wrong side)
        try:
            live_fill = float(getattr(order, 'avg_price', 0.0) or 0.0)
        except Exception:
            live_fill = 0.0
        if live_fill <= 0:
            live_fill = float(entry_price or 0.0)
        if live_fill > 0:
            sl, tp = self._align_brackets_with_fill(
                side=side,
                fill_price=float(live_fill),
                signal_entry=float(entry_price or live_fill),
                stop_loss=float(sl),
                take_profit=float(tp),
            )

        # Snapshot for dashboard (LIVE positions should be reconciled elsewhere periodically)
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                side=side,
                entry_price=live_fill if live_fill > 0 else (entry_price if entry_price > 0 else (order.avg_price or 0.0)),
                quantity=float(order.filled_qty or qty),
                stop_loss=sl,
                take_profit=tp,
                entry_reason=str(getattr(signal, 'reason', '')),
                confidence=float(getattr(signal, 'confidence', 0.0) or 0.0),
            )
            try:
                self.positions[symbol].mode = 'live'
            except Exception:
                pass

        # Persist initial risk so RR/BE/Trailing doesn't get distorted when SL moves
            try:
                pos0 = self.positions.get(symbol)
                if pos0 is not None:
                    pos0.initial_stop_loss = float(getattr(pos0, 'stop_loss', 0.0) or 0.0)
                    pos0.initial_take_profit = float(getattr(pos0, 'take_profit', 0.0) or 0.0)
                    pos0.initial_risk = abs(float(getattr(pos0, 'entry_price', 0.0) or 0.0) - float(getattr(pos0, 'initial_stop_loss', 0.0) or 0.0))
            except Exception:
                pass
            except Exception:
                pass

        # Best-effort reconcile actual position side
        try:
            rp = self.reconcile_live_position(symbol, expected_side=side)
            if rp.get('_mismatch'):
                logging.error(f"[LIVE] Position side mismatch after order for {symbol}: expected={side} actual={rp.get('_side')}")
        except Exception:
            pass

        # Place protective orders (reduceOnly) best-effort
        try:
            q_br = float(order.filled_qty or qty)
        except Exception:
            q_br = float(qty)
        try:
            self._place_live_brackets(symbol, side, q_br, sl, tp)
        except Exception:
            pass

        logging.info(f"[LIVE] Sent {order_side} {symbol} qty={float(order.filled_qty or qty):.8f} (order_id={order.order_id})")

    def _paper_fill_price(self, symbol: str, side: str, ref_price: float) -> float:
        """Best-effort paper fill model: spread + slippage.

        NOTE: This intentionally makes paper results *worse* (more realistic) than using close/last.
        """
        try:
            mid = float(getattr(self.market_data, 'get_current_price')(symbol) or 0.0)
            if mid <= 0:
                mid = float(ref_price or 0.0)
        except Exception:
            mid = float(ref_price or 0.0)
        if mid <= 0:
            return float(ref_price or 0.0)

        # NOTE: previous default tolerance (0.04) implied ~0.4% slip and could
        # invert TP/SL in PAPER. For top futures pairs, use a tighter (still
        # pessimistic) model.
        p = self.coin_params.get(symbol, {'slippage_tolerance': 0.004})
        tol = float(p.get('slippage_tolerance', 0.004) or 0.004)

        # Spread & slippage estimates (very simple, but stable)
        # Keep it pessimistic but not extreme
        spread_pct = min(max(tol * 0.05, 0.00002), 0.00025)
        slip_pct = min(max(tol * 0.10, 0.00005), 0.0012)

        side_u = str(side or 'LONG').upper()
        if side_u in ('LONG', 'BUY'):
            return mid * (1.0 + spread_pct / 2.0 + slip_pct)
        return mid * (1.0 - spread_pct / 2.0 - slip_pct)

    def _align_brackets_with_fill(
        self,
        side: str,
        fill_price: float,
        signal_entry: float,
        stop_loss: float,
        take_profit: float,
    ):
        """Realign SL/TP to the actual fill.

        Goal: guarantee monotonic brackets:
          - LONG : SL < fill < TP
          - SHORT: TP < fill < SL

        Preserve the intended distance ratios from the original signal entry where possible.
        """
        side_u = str(side or '').upper()
        fp = float(fill_price or 0.0)
        se = float(signal_entry or 0.0)
        sl = float(stop_loss or 0.0)
        tp = float(take_profit or 0.0)

        if fp <= 0:
            return sl, tp

        # If signal_entry is missing, treat fill as entry
        if se <= 0:
            se = fp

        # Compute offsets from the signal entry (fallback to small percent)
        if side_u == 'LONG':
            sl_off = max(0.0002 * se, se - sl) if sl > 0 else 0.0002 * se
            tp_off = max(0.0004 * se, tp - se) if tp > 0 else 0.0004 * se
            sl_new = fp - sl_off
            tp_new = fp + tp_off
            # Guard: ensure correct ordering
            if sl_new >= fp:
                sl_new = fp * (1.0 - 0.00025)
            if tp_new <= fp:
                tp_new = fp * (1.0 + 0.00035)
            return float(sl_new), float(tp_new)

        # SHORT
        sl_off = max(0.0002 * se, sl - se) if sl > 0 else 0.0002 * se
        tp_off = max(0.0004 * se, se - tp) if tp > 0 else 0.0004 * se
        sl_new = fp + sl_off
        tp_new = fp - tp_off
        if sl_new <= fp:
            sl_new = fp * (1.0 + 0.00025)
        if tp_new >= fp:
            tp_new = fp * (1.0 - 0.00035)
        return float(sl_new), float(tp_new)

    # ===================== PnL & PAPER CLOSE (FIXED) =====================

    def _estimate_roundtrip_fee(self, entry_price: float, exit_price: float, qty: float, taker: bool = True) -> float:
        """Estimate futures round-trip fee in USDT.

        Binance USDT-M futures fee is charged on notional (price*qty) per fill.
        This estimate is used for DEMO/PAPER accounting and dashboard.
        """
        fee_rate = self.taker_fee if taker else self.maker_fee
        notional = (float(entry_price) + float(exit_price)) * float(qty)
        return max(0.0, notional * float(fee_rate))

    def calculate_pnl_usdt(self, side: str, entry_price: float, exit_price: float, qty: float) -> float:
        """Correct PnL for USDT-M futures (DO NOT multiply by leverage).

        - LONG : (exit - entry) * qty
        - SHORT: (entry - exit) * qty
        """
        entry_price = float(entry_price)
        exit_price = float(exit_price)
        qty = float(qty)
        if qty <= 0:
            return 0.0
        if str(side).upper() == 'SHORT':
            return (entry_price - exit_price) * qty
        return (exit_price - entry_price) * qty

    def close_paper_position(self, symbol: str, exit_price: float, reason: str = 'CLOSE') -> float:
        """Close PAPER position and return pnl_net.

        IMPORTANT FIX:
        - PnL must NOT be multiplied by leverage.
        """
        pos = self.positions.get(symbol)
        if not pos:
            return 0.0

        exit_price = float(exit_price)
        pnl_gross = self.calculate_pnl_usdt(pos.side, pos.entry_price, exit_price, pos.quantity)
        fee = self._estimate_roundtrip_fee(pos.entry_price, exit_price, pos.quantity, taker=True)
        pnl_net = pnl_gross - fee

        try:
            pos.realized_pnl += pnl_net
        except Exception:
            pass

        # remove position
        try:
            del self.positions[symbol]
        except Exception:
            self.positions.pop(symbol, None)

        logging.info(
            f"[PAPER] Close {symbol} {reason} qty={pos.quantity:.8f} pnl_net=${pnl_net:.2f} (fee~${fee:.2f})"
        )
        
        # Feed closed trade to brain (self-training) if available
        try:
            brain = getattr(self, 'brain', None)
            if brain is not None:
                trade_rec = {
                    "timestamp": datetime.now().isoformat(),
                    "symbol": str(pos.symbol),
                    "side": str(pos.side),
                    "entry_price": float(pos.entry_price),
                    "exit_price": float(exit_price),
                    "quantity": float(pos.quantity),
                    "pnl": float(pnl_net),
                    "pnl_pct": float(pnl_net) / max(1e-9, float(pos.entry_price) * float(pos.quantity)),
                    "duration_seconds": int((datetime.now() - (pos.entry_time or datetime.now())).total_seconds()),
                    "exit_reason": str(reason),
                    "regime": str(getattr(pos, "regime", "") or ""),
                    "confidence": float(getattr(pos, "confidence", 0.0) or 0.0),
                    "features": getattr(pos, "features", {}) or {}
                }
                try:
                    brain.log_trade(trade_rec)
                except Exception:
                    pass
                try:
                    brain.add_trade_to_buffer(trade_rec)
                except Exception:
                    pass
        except Exception:
            pass

        return pnl_net

    def close_all(self):
        """Đóng tất cả positions trong bộ nhớ (paper)."""
        self.positions.clear()
    
    @handle_errors(retry=3, retry_delay=1.0, exponential_backoff=True)
    def _get_symbol_info(self, symbol: str) -> Dict:
        """Get symbol info including filters (stepSize/tickSize/minQty/minNotional).

        - Caches futures_exchange_info() để giảm gọi API và tăng tốc.
        - Trả về stepSize/tickSize và các min constraints để chuẩn hoá order.
        """
        import time
        now = time.time()
        if self._symbol_info_cache and (now - float(self._symbol_info_cache_ts or 0.0) < float(self._symbol_info_cache_ttl or 1800.0)):
            if symbol in self._symbol_info_cache:
                return self._symbol_info_cache[symbol]

        info = self.client.futures_exchange_info()

        cache: Dict[str, Dict] = {}
        for s in info.get('symbols', []):
            try:
                sym = s.get('symbol')
                if not sym:
                    continue
                filters = {f.get('filterType'): f for f in (s.get('filters') or [])}
                lot = filters.get('LOT_SIZE', {}) or {}
                pricef = filters.get('PRICE_FILTER', {}) or {}
                min_not = filters.get('MIN_NOTIONAL', {}) or {}

                cache[str(sym)] = {
                    'minQty': float(lot.get('minQty', 0.0) or 0.0),
                    'stepSize': float(lot.get('stepSize', 0.0) or 0.0),
                    'tickSize': float(pricef.get('tickSize', 0.0) or 0.0),
                    'minNotional': float(min_not.get('notional', min_not.get('minNotional', 0.0)) or 0.0),
                }
            except Exception:
                continue

        self._symbol_info_cache = cache
        self._symbol_info_cache_ts = now

        if symbol in cache:
            return cache[symbol]
        raise ValueError(f"Không tìm thấy thông tin symbol {symbol}")
    
    def _round_quantity(self, symbol: str, quantity: float) -> float:
        """Round quantity theo stepSize của Binance"""
        info = self._get_symbol_info(symbol)
        step_size = info['stepSize']
        min_qty = info['minQty']
        
        # Sử dụng Decimal để tránh floating point error
        qty_dec = Decimal(str(quantity))
        step_dec = Decimal(str(step_size))
        
        rounded = (qty_dec // step_dec) * step_dec
        if rounded < Decimal(str(min_qty)):
            return 0.0
        
        return float(rounded)
    
    def _round_price(self, symbol: str, price: float) -> float:
        """Round price theo tickSize của Binance (ROUND_DOWN)."""
        info = self._get_symbol_info(symbol)
        tick = float(info.get('tickSize') or 0.0)
        if tick <= 0:
            return float(price)
        p_dec = Decimal(str(price))
        t_dec = Decimal(str(tick))
        rounded = (p_dec // t_dec) * t_dec
        return float(rounded)

    def _fmt_qty(self, symbol: str, quantity: float) -> str:
        q = self._round_quantity(symbol, quantity)
        if q <= 0:
            return "0"
        # Không ép precision theo count('0'); dùng normalize theo stepSize
        step = self._get_symbol_info(symbol).get('stepSize') or 0.0
        if step and step > 0:
            # số chữ số thập phân từ stepSize
            s = str(step)
            decs = 0
            if '.' in s:
                decs = len(s.split('.')[-1].rstrip('0'))
            return f"{q:.{decs}f}"
        return str(q)

    def _fmt_price(self, symbol: str, price: float) -> str:
        p = self._round_price(symbol, price)
        tick = self._get_symbol_info(symbol).get('tickSize') or 0.0
        if tick and tick > 0:
            s = str(tick)
            decs = 0
            if '.' in s:
                decs = len(s.split('.')[-1].rstrip('0'))
            return f"{p:.{decs}f}"
        # fallback
        return str(p)

    @handle_errors(retry=3, context="Place order")
    def place_order(self, symbol: str, side: str, quantity: float,
                   price: Optional[float] = None,
                   stop_price: Optional[float] = None,
                   order_type: OrderType = OrderType.MARKET,
                   reduce_only: bool = False, post_only: bool = False,
                   extra_params: Optional[Dict] = None) -> Optional[Order]:
        """Place order with validation & retry"""
        start_time = time.time()
        
        # Validate & round quantity
        quantity = self._round_quantity(symbol, quantity)
        if quantity <= 0:
            raise OrderExecutionError(f"Quantity sau rounding = 0 cho {symbol}")


        # MinNotional guard (tránh lệnh quá nhỏ bị reject)
        try:
            info = self._get_symbol_info(symbol)
            min_notional = float(info.get('minNotional') or 0.0)
            if min_notional > 0:
                ref_price = price if price is not None else (stop_price if stop_price is not None else None)
                if ref_price is None and self.market_data is not None:
                    try:
                        ref_price = float(self.market_data.get_current_price(symbol))
                    except Exception:
                        ref_price = None
                if ref_price is not None and (float(quantity) * float(ref_price) < float(min_notional)):
                    raise OrderExecutionError(
                        f"Notional quá nhỏ cho {symbol}: qty={quantity} * price={ref_price} < minNotional={min_notional}"
                    )
        except OrderExecutionError:
            raise
        except Exception:
            # Nếu không lấy được info/price thì bỏ qua guard này
            pass

        
        order = Order(
            symbol=symbol,
            side=side.upper(),
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            reduce_only=reduce_only,
            post_only=post_only
        )
        
        params = {
            'symbol': symbol,
            'side': side.upper(),
            'type': order_type.value,
            'quantity': self._fmt_qty(symbol, float(quantity)),
            'newClientOrderId': order.client_order_id,
            'reduceOnly': reduce_only
        }
        
        if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            if price is None:
                raise ValueError("Price required for LIMIT/STOP_LIMIT")
            params['price'] = self._fmt_price(symbol, float(price))
            params['timeInForce'] = "GTC"
        
        if order_type in [OrderType.STOP_MARKET, OrderType.STOP_LIMIT]:
            if order.stop_price is None:
                raise ValueError("Stop price required")
            params['stopPrice'] = self._fmt_price(symbol, float(order.stop_price))
        
        if post_only:
            params['timeInForce'] = "GTX"  # Post-only mode

        # Allow caller to pass through Binance futures_create_order params (e.g., workingType=MARK_PRICE)
        if extra_params and isinstance(extra_params, dict):
            try:
                for k, v in extra_params.items():
                    if v is not None and k not in params:
                        params[k] = v
            except Exception:
                pass
        
        try:
            order.sent_at = datetime.now()
            response = self.client.futures_create_order(**params)
            
            order.order_id = response['orderId']
            order.state = OrderState.SENT
            order.latency_ms = self._calculate_latency(start_time)
            
            self.active_orders[order.client_order_id] = order
            self.order_history.append(order)
            
            logging.info(f"Order placed: {side} {quantity} {symbol} @ {price or 'MARKET'} - ID: {order.order_id}")
            return order
            
        except BinanceAPIException as e:
            order.state = OrderState.FAILED
            order.error_message = str(e)
            raise OrderExecutionError(f"Binance order failed: {e.message} (code {e.code})")
    
    def cancel_order(self, symbol: str, order_id: int) -> bool:
        """Cancel existing order"""
        try:
            self.client.futures_cancel_order(symbol=symbol, orderId=order_id)
            if order_id in [o.order_id for o in self.active_orders.values()]:
                for key, ord in list(self.active_orders.items()):
                    if ord.order_id == order_id:
                        ord.state = OrderState.CANCELLED
                        del self.active_orders[key]
            return True
        except Exception as e:
            logging.error(f"Cancel order failed: {e}")
            return False
    
    def cancel_all_orders(self, symbol: Optional[str] = None):
        """Cancel all open orders (for symbol or all)"""
        try:
            if symbol:
                self.client.futures_cancel_all_open_orders(symbol=symbol)
            else:
                self.client.futures_cancel_all_open_orders()
            self.active_orders.clear()
            logging.info("All orders cancelled")
        except Exception as e:
            logging.error(f"Cancel all failed: {e}")
    
    def emergency_close_all(self):
        """Emergency close all positions - MARKET reduce-only"""
        logging.warning("🚨 EMERGENCY CLOSE ALL POSITIONS TRIGGERED")
        
        try:
            positions = self.client.futures_position_information()
            
            for pos in positions:
                amt = float(pos['positionAmt'])
                if amt == 0:
                    continue
                
                symbol = pos['symbol']
                quantity = abs(amt)
                side = 'SELL' if amt > 0 else 'BUY'  # Close opposite
                
                logging.warning(f"Closing {symbol} {side} {quantity}")
                
                try:
                    self.place_order(
                        symbol=symbol,
                        side=side,
                        quantity=quantity,
                        order_type=OrderType.MARKET,
                        reduce_only=True
                    )
                except Exception as e:
                    logging.error(f"Failed to close {symbol}: {e}")
            
            self.cancel_all_orders()
            logging.warning("🚨 Emergency close hoàn tất")
            return True
            
        except Exception as e:
            logging.critical(f"Emergency close failed: {e}")
            return False
    
    def modify_stop_loss(self, symbol: str, old_order_id: int, new_stop_price: float) -> Order:
        """Modify existing stop loss order"""
        try:
            # Cancel old stop loss
            self.client.futures_cancel_order(symbol=symbol, orderId=old_order_id)
            
            # Get current position
            pos_info = self.client.futures_position_information(symbol=symbol)
            for pos in pos_info:
                amt = float(pos['positionAmt'])
                if amt == 0:
                    continue
                
                quantity = abs(amt)
                side = 'SELL' if amt > 0 else 'BUY'
                
                # Place new stop loss
                return self.place_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    stop_price=new_stop_price,
                    order_type=OrderType.STOP_MARKET,
                    reduce_only=True
                )
            
            raise ValueError(f"Không tìm thấy position cho {symbol}")
            
        except Exception as e:
            logging.error(f"Modify stop loss failed: {e}")
            raise OrderExecutionError("Modify SL failed") from e
    
    def execute_twap(self, symbol: str, side: str, total_quantity: float, 
                    duration_minutes: int, num_orders: int = None) -> List[Order]:
        """TWAP - Time Weighted Average Price execution"""
        if num_orders is None:
            num_orders = max(3, duration_minutes // 2)
        
        quantity_per_order = total_quantity / num_orders
        interval_seconds = (duration_minutes * 60) / num_orders
        
        orders = []
        logging.info(f"🔄 TWAP execution: {total_quantity} {symbol} trong {duration_minutes} phút, {num_orders} orders")
        
        for i in range(num_orders):
            try:
                order = Order(
                    symbol=symbol,
                    side=side,
                    order_type=OrderType.LIMIT,
                    quantity=quantity_per_order,
                    execution_strategy=ExecutionStrategy.PASSIVE
                )
                
                filled_order = self.place_order_passive(order)
                if filled_order:
                    orders.append(filled_order)
                
                if i < num_orders - 1:
                    time.sleep(interval_seconds)
                    
            except Exception as e:
                logging.error(f"TWAP slice {i+1} failed: {e}")
                break
        
        total_filled = sum(o.filled_qty for o in orders)
        total_value = sum(o.avg_price * o.filled_qty for o in orders if o.filled_qty > 0)
        avg_price = total_value / total_filled if total_filled > 0 else 0
        
        logging.info(f"✅ TWAP hoàn tất: {total_filled}/{total_quantity} @ avg {avg_price:.4f}")
        return orders
    
    def place_order_passive(self, order: Order) -> Order:
        """Place passive LIMIT order (maker)"""
        current_price = self.market_data.get_current_price(order.symbol)
        if current_price <= 0:
            raise ValueError("Không lấy được giá hiện tại")
        
        # Đặt giá tốt hơn một chút để làm maker
        if order.side == 'BUY':
            order.price = current_price * 0.9995  # Bid thấp hơn 0.05%
        else:
            order.price = current_price * 1.0005  # Ask cao hơn 0.05%
        
        return self.place_order(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=order.price,
            order_type=OrderType.LIMIT,
            post_only=True
        )
    
    def execute_iceberg(self, symbol: str, side: str, total_quantity: float, 
                       visible_quantity: float) -> List[Order]:
        """Iceberg execution - ẩn kích thước thật"""
        orders = []
        remaining = total_quantity
        
        logging.info(f"🧊 Iceberg: {total_quantity} {symbol} với visible {visible_quantity}")
        
        while remaining > 0:
            slice_qty = min(visible_quantity, remaining)
            slice_qty = self._round_quantity(symbol, slice_qty)
            
            if slice_qty <= 0:
                break
            
            try:
                order = Order(
                    symbol=symbol,
                    side=side,
                    order_type=OrderType.LIMIT,
                    quantity=slice_qty,
                    execution_strategy=ExecutionStrategy.PASSIVE
                )
                
                filled_order = self.place_order_passive(order)
                if filled_order:
                    orders.append(filled_order)
                    remaining -= filled_order.filled_qty
                
                time.sleep(1)  # Delay nhỏ giữa các slice
                
            except Exception as e:
                logging.error(f"Iceberg slice failed: {e}")
                break
        
        total_filled = sum(o.filled_qty for o in orders)
        logging.info(f"✅ Iceberg hoàn tất: {total_filled}/{total_quantity}")
        return orders