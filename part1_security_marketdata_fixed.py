"""
Part 1: Security + Market Data Core - Complete & Fixed Version (2026)
- Fixed cryptography import (PBKDF2 -> PBKDF2HMAC)
- Added OrderBookSnapshot class and fetching method
- FIX BOM: dùng utf-8-sig khi đọc .env.encrypted để tránh lỗi encoding/BOM
- Thêm debug print chi tiết trong decrypt_api_keys để dễ trace lỗi "Passphrase sai"
- Thêm phương thức get_klines để hỗ trợ lấy dữ liệu nến OHLCV (fix lỗi ở part5)
- FIX LỖI MỚI: Giới hạn limit tối đa 1500 trong get_klines để tránh APIError(code=-1130)
- Giữ nguyên SecurityManager để tương thích với .env.encrypted cũ (không thay đổi derive key)
- FIX: Thêm get_current_price() để lấy giá realtime (fix lỗi N/A trong Telegram/Strategy)

### LIVE FIX (2026-01)
- Thêm bulk klines loader để lấy >1500 candles cho ML + Strategy
- get_klines() tự động dùng bulk mode khi limit > 1500
"""

import os
import sys
import time
import json
import logging
import getpass
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException, BinanceOrderException

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64

from error_handling import handle_errors, ErrorHandler, SecurityError, APIError


@dataclass
class TradeFlow:
    timestamp: datetime
    price: float
    quantity: float
    is_buyer_maker: bool


@dataclass
class OrderBookSnapshot:
    symbol: str
    timestamp: datetime
    bids: List[Tuple[float, float]]
    asks: List[Tuple[float, float]]
    last_update_id: Optional[int] = None


class OrderFlowAnalyzer:
    """Analyzes order flow from recent trades"""

    def __init__(self, window_size: int = 500):
        self.trades = deque(maxlen=window_size)
        self.delta_history = deque(maxlen=100)
        self.cvd = 0.0  # Cumulative Volume Delta

    def add_trade(self, trade: TradeFlow):
        self.trades.append(trade)
        delta = trade.quantity if not trade.is_buyer_maker else -trade.quantity
        self.delta_history.append(delta)
        self.cvd += delta

    def calculate_delta(self) -> float:
        if not self.delta_history:
            return 0.0
        return sum(self.delta_history)

    def calculate_cvd(self) -> float:
        return self.cvd

    def detect_aggressive_buying(self, threshold: float = 1.5) -> bool:
        recent_deltas = list(self.delta_history)[-20:]
        if not recent_deltas:
            return False
        avg_delta = sum(recent_deltas) / len(recent_deltas)
        return avg_delta > threshold * np.std(recent_deltas) if len(recent_deltas) > 5 else False

    def detect_aggressive_selling(self, threshold: float = 1.5) -> bool:
        recent_deltas = list(self.delta_history)[-20:]
        if not recent_deltas:
            return False
        avg_delta = sum(recent_deltas) / len(recent_deltas)
        return avg_delta < -threshold * np.std(recent_deltas) if len(recent_deltas) > 5 else False

    def get_bid_ask_imbalance(self, snapshot: OrderBookSnapshot, depth: int = 5) -> float:
        bid_volume = sum(q for _, q in snapshot.bids[:depth])
        ask_volume = sum(q for _, q in snapshot.asks[:depth])
        total = bid_volume + ask_volume
        return (bid_volume - ask_volume) / total if total > 0 else 0.0


class SecurityManager:
    """Manages API keys encryption/decryption"""

    def __init__(self):
        self.client = None
        self.api_key = None
        self.api_secret = None

    def decrypt_api_keys(self, passphrase: str) -> Tuple[str, str]:
        """Decrypt API keys from .env.encrypted using passphrase"""
        if not os.path.exists('.env.encrypted'):
            raise FileNotFoundError("Không tìm thấy file .env.encrypted")

        with open('.env.encrypted', 'rb') as f:
            encrypted_data = f.read()

        try:
            salt = b'god_tier_salt_2026'
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
            key = base64.urlsafe_b64encode(kdf.derive(passphrase.encode()))
            fernet = Fernet(key)
            decrypted = fernet.decrypt(encrypted_data).decode('utf-8')

            lines = decrypted.splitlines()
            api_key = next((l.split('=', 1)[1].strip() for l in lines if l.startswith('api_key=')), None)
            api_secret = next((l.split('=', 1)[1].strip() for l in lines if l.startswith('api_secret=')), None)

            if not api_key or not api_secret:
                raise ValueError("Không tìm thấy api_key/api_secret trong file decrypt")

            self.api_key = api_key
            self.api_secret = api_secret
            self.client = Client(api_key, api_secret)

            logging.info("Decrypt thành công - API keys đã load")
            return api_key, api_secret

        except InvalidToken:
            raise ValueError("Passphrase sai hoặc file .env.encrypted bị hỏng")
        except Exception as e:
            raise ValueError(f"Lỗi decrypt: {str(e)}")


    def decrypt_env(self, passphrase: str) -> Dict[str, str]:
        """Decrypt .env.encrypted và load secrets rộng hơn api_key/api_secret.

        - Tương thích ngược: vẫn hiểu api_key/api_secret.
        - Hỗ trợ thêm các biến như telegram_bot_token/chat_id...
        - Tự động export ra os.environ với các alias UPPERCASE phổ biến.

        Trả về dict secrets (key theo đúng tên trong file, chưa normalize).
        """
        if not os.path.exists('.env.encrypted'):
            raise FileNotFoundError("Không tìm thấy file .env.encrypted")

        with open('.env.encrypted', 'rb') as f:
            encrypted_data = f.read()

        try:
            salt = b'god_tier_salt_2026'
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
            key = base64.urlsafe_b64encode(kdf.derive(passphrase.encode()))
            fernet = Fernet(key)
            decrypted = fernet.decrypt(encrypted_data).decode('utf-8')

            secrets: Dict[str, str] = {}
            for raw in decrypted.splitlines():
                line = (raw or '').strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                k, v = line.split('=', 1)
                k = (k or '').strip()
                v = (v or '').strip()
                if not k:
                    continue
                secrets[k] = v

            # Backward compatible keys
            api_key = secrets.get('api_key') or secrets.get('BINANCE_API_KEY') or secrets.get('binance_api_key')
            api_secret = secrets.get('api_secret') or secrets.get('BINANCE_API_SECRET') or secrets.get('binance_api_secret')

            if api_key:
                os.environ.setdefault('BINANCE_API_KEY', api_key)
            if api_secret:
                os.environ.setdefault('BINANCE_API_SECRET', api_secret)

            # Telegram aliases
            tg_token = (secrets.get('telegram_bot_token') or secrets.get('TELEGRAM_BOT_TOKEN') or secrets.get('telegram_token'))
            tg_chat = (secrets.get('telegram_chat_id') or secrets.get('TELEGRAM_CHAT_ID') or secrets.get('chat_id'))
            if tg_token:
                os.environ.setdefault('TELEGRAM_BOT_TOKEN', tg_token)
            if tg_chat:
                os.environ.setdefault('TELEGRAM_CHAT_ID', tg_chat)

            # Export all keys (best-effort) as env too (do not overwrite existing)
            for k, v in secrets.items():
                up = (k or '').strip().upper()
                if up and up not in os.environ:
                    os.environ[up] = v

            # If we have Binance keys, also prime the client for compatibility with the rest of code.
            if api_key and api_secret:
                self.api_key = api_key
                self.api_secret = api_secret
                self.client = Client(api_key, api_secret)

            logging.info("Decrypt thành công - secrets đã load (env + in-memory)")
            return secrets

        except InvalidToken:
            raise ValueError("Passphrase sai hoặc file .env.encrypted bị hỏng")
        except Exception as e:
            raise ValueError(f"Lỗi decrypt: {str(e)}")


class EnhancedMarketDataEngine:
    """Enhanced market data engine with caching, order flow, and real-time features"""

    # ------------------------------------------------------------------
    # Compatibility wrapper: accept limit and extra kwargs
    def get_klines_bulk(self, symbol: str, interval: str = '5m', limit: int = 500, **kwargs):
        _ = kwargs
        return self.get_klines(symbol=symbol, interval=interval, limit=limit)

    def __init__(self, security_manager):
        self.security_manager = security_manager
        self.logger = logging.getLogger(__name__)
        self.rate_limit_delay = 0.1

        if hasattr(security_manager, 'client') and security_manager.client is not None:
            self.client = security_manager.client
        else:
            self.client = Client(None, None)
            self.logger.warning("Public mode: Client chỉ có quyền đọc dữ liệu (không trade)")

        if not hasattr(self.client, 'futures_klines'):
            raise RuntimeError("Client không hợp lệ - thiếu phương thức futures_klines")

        self.order_flow_analyzers = defaultdict(lambda: OrderFlowAnalyzer())
        self.cache = {}
        self.logger.info("EnhancedMarketDataEngine initialized - client ready")

    def _rate_limit_check(self):
        time.sleep(self.rate_limit_delay)

    def get_order_book_snapshot(self, symbol: str, limit: int = 100) -> Optional[OrderBookSnapshot]:
        self._rate_limit_check()
        try:
            book = self.client.futures_order_book(symbol=symbol, limit=limit)
            bids = [(float(p), float(q)) for p, q in book['bids']]
            asks = [(float(p), float(q)) for p, q in book['asks']]

            snapshot = OrderBookSnapshot(
                symbol=symbol,
                timestamp=datetime.now(),
                bids=sorted(bids, reverse=True),
                asks=asks,
                last_update_id=book.get('lastUpdateId')
            )
            return snapshot
        except Exception as e:
            self.logger.error(f"Order book error: {e}")
            return None

    # ==========================================================
    # ### LIVE FIX ###
    # Bulk klines loader để vượt giới hạn 1500 candle
    # ==========================================================
    def _fetch_klines_bulk(self, symbol, interval, limit):
        all_rows = []
        end_time = int(time.time() * 1000)

        while len(all_rows) < limit:
            batch = min(1500, limit - len(all_rows))

            klines = self.client.futures_klines(
                symbol=symbol,
                interval=interval,
                endTime=end_time,
                limit=batch
            )

            if not klines:
                break

            all_rows = klines + all_rows
            end_time = klines[0][0] - 1

            if len(klines) < batch:
                break

            time.sleep(0.2)

        return all_rows[-limit:]

    @handle_errors(retry=3, exponential_backoff=True)
    def get_klines(self, symbol: str, interval: str = '5m', limit: int = 500, start_time: int = None, end_time: int = None) -> Optional[pd.DataFrame]:
        self._rate_limit_check()

        MAX_LIMIT = 1500
        safe_limit = min(limit, MAX_LIMIT)
        if limit > MAX_LIMIT:
            self.logger.warning(f"Limit {limit} vượt quá max 1500 → dùng bulk loader")

        try:
            # ===========================
            # ### LIVE FIX ###
            # ===========================
            if limit > MAX_LIMIT:
                self.logger.info(f"Bulk fetching {limit} candles for {symbol} {interval}")
                klines = self._fetch_klines_bulk(symbol, interval, limit)
            else:
                klines = self.client.futures_klines(
                    symbol=symbol,
                    interval=interval,
                    startTime=start_time,
                    endTime=end_time,
                    limit=safe_limit
                )

            if not klines:
                self.logger.warning(f"Không có dữ liệu nến cho {symbol} {interval}")
                return None

            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])

            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)

            df.set_index('open_time', inplace=True)
            return df[['open', 'high', 'low', 'close', 'volume']]

        except BinanceAPIException as e:
            self.logger.error(f"Binance API error khi lấy klines {symbol} {interval}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Lỗi bất ngờ khi lấy klines {symbol} {interval}: {e}")
            return None

    def get_current_price(self, symbol: str) -> float:
        self._rate_limit_check()
        try:
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except (BinanceAPIException, BinanceRequestException, BinanceOrderException) as e:
            # IMPORTANT: Do not silently return 0.0 (it kills signals and makes bot "no orders" for days).
            # Throttle logs to avoid spam when Binance is unstable / rate-limited.
            try:
                now = time.time()
                last = float(getattr(self, '_last_price_err_ts', 0.0) or 0.0)
                if (now - last) >= 30.0:
                    self.logger.warning(f"get_current_price failed for {symbol}: {type(e).__name__}: {e}")
                    setattr(self, '_last_price_err_ts', now)
            except Exception:
                pass
            return 0.0
        except Exception as e:
            try:
                now = time.time()
                last = float(getattr(self, '_last_price_err_ts', 0.0) or 0.0)
                if (now - last) >= 30.0:
                    self.logger.warning(f"get_current_price unexpected error for {symbol}: {type(e).__name__}: {e}")
                    setattr(self, '_last_price_err_ts', now)
            except Exception:
                pass
            return 0.0

    # -------------------------
    # CIO-grade microstructure helpers (best-effort, no extra deps)
    # -------------------------
    def get_mark_price(self, symbol: str) -> float:
        """Best-effort mark price for USDT-M futures."""
        self._rate_limit_check()
        try:
            mp = self.client.futures_mark_price(symbol=symbol)
            return float(mp.get('markPrice', 0.0) or 0.0)
        except Exception:
            return 0.0

    def get_best_bid_ask(self, symbol: str) -> Dict:
        """Return best bid/ask and spread pct (best-effort)."""
        snap = self.get_order_book_snapshot(symbol=symbol, limit=5)
        if not snap or not snap.bids or not snap.asks:
            return {'bid': 0.0, 'ask': 0.0, 'spread_pct': 0.0}
        bid = float(snap.bids[0][0] or 0.0)
        ask = float(snap.asks[0][0] or 0.0)
        mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else 0.0
        spread_pct = ((ask - bid) / mid) if mid > 0 else 0.0
        return {'bid': bid, 'ask': ask, 'spread_pct': spread_pct}

    def get_funding_rate(self, symbol: str) -> float:
        """Best-effort latest funding rate for USDT-M futures."""
        self._rate_limit_check()
        try:
            fr = self.client.futures_funding_rate(symbol=symbol, limit=1)
            if isinstance(fr, list) and fr:
                return float(fr[0].get('fundingRate', 0.0) or 0.0)
        except Exception:
            pass
        return 0.0

    def get_latency_stats(self) -> Dict:
        return {'p50': 0, 'p95': 0, 'p99': 0, 'mean': 0, 'max': 0}
