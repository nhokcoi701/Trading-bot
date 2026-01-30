"""part7_backtesting_fixed.py

FUND-GRADE Backtesting Engine (Binance Futures USD-M oriented) – Python 3.8

Key upgrades vs previous version
- ❌ Removed proxy/synthetic candles (previously biased results).
- ❌ Removed Martingale / DCA / Grid tricks (tail-risk, non-fund-grade).
- ✅ Strictly causal bar-by-bar simulation (only past data visible).
- ✅ Cost model: taker fee + volatility-based slippage (pessimistic by default).
- ✅ Optional walk-forward (train/test) with simple purge+embargo to reduce leakage.
- ✅ Data integrity checks (monotonic timestamps, duplicates, OHLC sanity).

Notes
- This backtester is meant to validate *robustness*, not to inflate PnL.
- For Binance USD-M futures, market orders are almost always taker.

Interface compatibility
- Keeps class name `Backtester` and method `run_backtest(start_date, end_date, symbols)`.
- Expects `market_data.get_klines(symbol, interval, start_time, end_time, limit)` returning DataFrame
  indexed by datetime with columns: open, high, low, close, (optional volume).
- Expects `strategy.extract_features(symbol, df=past_df)` and `strategy.generate_signals(symbol, features)`.

"""

import logging
import math
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd


@dataclass
class BacktestTrade:
    entry_time: datetime
    exit_time: datetime
    symbol: str
    side: str  # LONG / SHORT
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    commission: float
    slippage: float
    hold_time: timedelta
    exit_reason: str
    mae: float = 0.0
    mfe: float = 0.0


class Backtester:
    """Fund-grade backtester with conservative assumptions."""

    def __init__(self, market_data, strategy, brain, config: Dict):
        self.market_data = market_data
        self.strategy = strategy
        self.brain = brain
        self.config = config or {}

        bt = (self.config.get('backtesting') or {})
        self.initial_equity = float(bt.get('initial_equity', 10000.0) or 10000.0)

        # Binance USD-M typical fee tiers vary; use conservative defaults.
        # Market order => taker in most cases.
        self.taker_fee = float(bt.get('taker_fee', bt.get('commission', 0.0004)) or 0.0004)
        self.maker_fee = float(bt.get('maker_fee', 0.0002) or 0.0002)

        # Base slippage is additional to volatility-based component.
        self.slippage_base = float(bt.get('slippage', 0.0006) or 0.0006)

        # Timeframe
        self.interval = str(bt.get('interval', '5m') or '5m')

        # Risk/position sizing (simple, robust)
        self.risk_per_trade = float(bt.get('risk_per_trade', 0.005) or 0.005)  # 0.5% per trade
        self.max_leverage = int(bt.get('max_leverage', 5) or 5)
        self.min_notional = float(bt.get('min_notional', 20.0) or 20.0)

        # Trade frequency guard
        self.cooldown_bars = int(bt.get('cooldown_bars', 3) or 3)

        # Walk-forward
        self.walk_forward = bool(bt.get('walk_forward', False))
        self.train_period_days = int(bt.get('train_period_days', 90) or 90)
        self.test_period_days = int(bt.get('test_period_days', 30) or 30)
        self.embargo_bars = int(bt.get('embargo_bars', 5) or 5)

        logging.info(
            f"[BT] Fund-grade backtester initialized | interval={self.interval} initial=${self.initial_equity:,.2f} "
            f"taker_fee={self.taker_fee:.4%} slippage_base={self.slippage_base:.4%} walk_forward={self.walk_forward}"
        )

    # --------------------------
    # Data loading & validation
    # --------------------------

    def _load_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        if end_dt <= start_dt:
            return pd.DataFrame()

        all_chunks: List[pd.DataFrame] = []
        current = start_dt

        # Binance limit is usually 1000 klines per call; chunk conservatively.
        # Using time windows instead of just limit makes it independent of interval.
        chunk_days = 15

        while current < end_dt:
            chunk_start = current
            chunk_end = min(current + timedelta(days=chunk_days), end_dt)

            start_ts = int(chunk_start.timestamp() * 1000)
            end_ts = int(chunk_end.timestamp() * 1000)

            try:
                df_chunk = self.market_data.get_klines(
                    symbol=symbol,
                    interval=self.interval,
                    start_time=start_ts,
                    end_time=end_ts,
                    limit=1000,
                )
            except Exception as e:
                logging.error(f"[BT] get_klines failed for {symbol} {chunk_start.date()}..{chunk_end.date()}: {e}")
                df_chunk = None

            if df_chunk is None or getattr(df_chunk, 'empty', True):
                # Fund-grade behavior: never fabricate candles.
                logging.warning(f"[BT] Missing data for {symbol} {chunk_start.date()}..{chunk_end.date()} – skipping chunk")
                current = chunk_end
                continue

            all_chunks.append(df_chunk)
            current = chunk_end

        if not all_chunks:
            return pd.DataFrame()

        df = pd.concat(all_chunks)
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                pass

        df = df[~df.index.duplicated(keep='last')]
        df = df.sort_index()

        # Basic OHLC sanity
        for c in ('open', 'high', 'low', 'close'):
            if c not in df.columns:
                raise ValueError(f"[BT] Missing column '{c}' for {symbol}")

        # Drop rows with non-positive close
        df = df[df['close'] > 0]

        # Fix obviously bad candles (high < max(open,close) etc.) by dropping
        bad = (
            (df['high'] < df[['open', 'close']].max(axis=1)) |
            (df['low'] > df[['open', 'close']].min(axis=1)) |
            (df['high'] <= 0) |
            (df['low'] <= 0)
        )
        if bad.any():
            n_bad = int(bad.sum())
            logging.warning(f"[BT] Dropping {n_bad} bad candles for {symbol}")
            df = df[~bad]

        if len(df) < 200:
            logging.warning(f"[BT] Not enough candles for {symbol}: {len(df)}")

        return df

    # --------------------------
    # Cost & sizing models
    # --------------------------

    def _estimate_slippage_pct(self, ohlc_row: pd.Series) -> float:
        """Pessimistic slippage proxy using intrabar range.

        - Base slippage: config
        - Volatility component: k * (high-low)/close
        """
        try:
            h = float(ohlc_row['high'])
            l = float(ohlc_row['low'])
            c = float(ohlc_row['close'])
            if c <= 0:
                return self.slippage_base
            intrabar = max(0.0, (h - l) / c)
            # k chosen to be conservative for crypto.
            return min(0.02, max(self.slippage_base, self.slippage_base + 0.25 * intrabar))
        except Exception:
            return self.slippage_base

    def _compute_position_qty(self, equity: float, entry: float, stop: float, leverage: int = 1) -> float:
        """Risk-based sizing using stop distance.

        qty = (equity * risk_per_trade) / |entry-stop|
        Then clamp by notional & leverage.
        """
        entry = float(entry or 0.0)
        stop = float(stop or 0.0)
        if entry <= 0 or stop <= 0:
            return 0.0

        stop_dist = abs(entry - stop)
        if stop_dist <= 0:
            return 0.0

        risk_dollars = max(0.0, equity * self.risk_per_trade)
        raw_qty = risk_dollars / stop_dist

        lev = int(max(1, min(self.max_leverage, leverage or 1)))
        max_notional = equity * lev
        qty = min(raw_qty, max_notional / entry)

        # Notional floor
        if qty * entry < self.min_notional:
            return 0.0

        return float(qty)

    # --------------------------
    # Simulation
    # --------------------------

    def run_backtest(self, start_date: str, end_date: str, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        for symbol in (symbols or []):
            try:
                df = self._load_historical_data(symbol, start_date, end_date)
                if df is None or df.empty:
                    logging.warning(f"[BT] No data for {symbol} – skipping")
                    continue

                if self.walk_forward:
                    results[symbol] = self._run_walk_forward(symbol, df)
                else:
                    results[symbol] = self._run_single_pass(symbol, df)

            except Exception as e:
                logging.error(f"[BT] Backtest failed for {symbol}: {e}", exc_info=True)

        return results

    def _run_walk_forward(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Simple walk-forward: [train -> embargo -> test] rolling windows.

        - We do not re-train models here; we only allow the strategy/brain to be updated
          using train portion, then evaluate test portion.
        - Embargo bars reduce leakage around boundaries.
        """
        train_len = self._days_to_bars(df, self.train_period_days)
        test_len = self._days_to_bars(df, self.test_period_days)

        if train_len <= 0 or test_len <= 0:
            return self._run_single_pass(symbol, df)

        start = 0
        all_trades: List[BacktestTrade] = []
        equity = self.initial_equity
        equity_curve = [equity]
        curve_times = [df.index[0]]

        while True:
            train_end = start + train_len
            test_start = train_end + self.embargo_bars
            test_end = test_start + test_len
            if test_end >= len(df):
                break

            train_df = df.iloc[start:train_end].copy()
            test_df = df.iloc[test_start:test_end].copy()

            # Allow brain to see *train* results only (best-effort)
            try:
                if hasattr(self.brain, 'reset_session'):
                    self.brain.reset_session()
            except Exception:
                pass

            # Optional: pre-warm by running through train without executing trades
            # (we just build any internal indicators, if strategy/brain caches anything)
            try:
                _ = self._iterate_features_only(symbol, train_df)
            except Exception:
                pass

            sim = self._simulate(symbol, test_df, starting_equity=equity)
            equity = float(sim.get('final_equity', equity) or equity)
            all_trades.extend(sim.get('trades', []) or [])

            # Extend curve (append last point of each test window)
            equity_curve.append(equity)
            curve_times.append(test_df.index[-1])

            start = test_end

        return self._summarize(symbol, all_trades, equity_curve, curve_times)

    def _run_single_pass(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        sim = self._simulate(symbol, df, starting_equity=self.initial_equity)
        return self._summarize(symbol, sim.get('trades', []) or [], sim.get('equity_curve', []) or [], sim.get('timestamps', []) or [])

    def _iterate_features_only(self, symbol: str, df: pd.DataFrame) -> None:
        for i in range(200, len(df)):
            past = df.iloc[: i + 1]
            _ = self.strategy.extract_features(symbol, df=past)

    def _simulate(self, symbol: str, df: pd.DataFrame, starting_equity: float) -> Dict[str, Any]:
        equity = float(starting_equity)
        trades: List[BacktestTrade] = []
        equity_curve = [equity]
        timestamps = [df.index[0]]

        position: Optional[Dict[str, Any]] = None
        last_entry_i = -10_000

        # Warmup for indicators
        warmup = 200
        if len(df) < warmup + 10:
            return {
                'trades': trades,
                'equity_curve': equity_curve,
                'timestamps': timestamps,
                'final_equity': equity,
            }

        for i in range(warmup, len(df)):
            ts = df.index[i]
            row = df.iloc[i]
            c = float(row['close'])
            h = float(row['high'])
            l = float(row['low'])

            past = df.iloc[: i + 1]

            # Generate features/signal causally
            try:
                features = self.strategy.extract_features(symbol, df=past) or {}
            except Exception as e:
                logging.debug(f"[BT] extract_features error {symbol} @ {ts}: {e}")
                features = {}

            try:
                signal = self.strategy.generate_signals(symbol, features)
            except Exception as e:
                logging.debug(f"[BT] generate_signals error {symbol} @ {ts}: {e}")
                signal = None

            # ---------------- exit logic (intrabar SL/TP) ----------------
            if position is not None:
                side = position['side']
                entry = float(position['entry_price'])
                qty = float(position['quantity'])
                sl = float(position['sl'])
                tp = float(position['tp'])

                # Track MAE/MFE
                if side == 'LONG':
                    position['mae'] = min(position.get('mae', 0.0), l - entry)
                    position['mfe'] = max(position.get('mfe', 0.0), h - entry)
                else:
                    position['mae'] = min(position.get('mae', 0.0), entry - h)
                    position['mfe'] = max(position.get('mfe', 0.0), entry - l)

                exit_price = None
                exit_reason = None

                # Conservative ordering: assume SL hits before TP if both inside same candle.
                if side == 'LONG':
                    if l <= sl:
                        exit_price = sl
                        exit_reason = 'SL'
                    elif h >= tp:
                        exit_price = tp
                        exit_reason = 'TP'
                else:
                    if h >= sl:
                        exit_price = sl
                        exit_reason = 'SL'
                    elif l <= tp:
                        exit_price = tp
                        exit_reason = 'TP'

                # Optional time-stop
                max_hold = int(self.config.get('backtesting', {}).get('max_hold_bars', 720) or 720)
                if exit_price is None and (i - position['entry_i']) >= max_hold:
                    exit_price = c
                    exit_reason = 'TIME'

                if exit_price is not None:
                    slip = self._estimate_slippage_pct(row)
                    # Market exit -> pay slippage unfavorably
                    if side == 'LONG':
                        exit_fill = float(exit_price) * (1.0 - slip)
                        gross = (exit_fill - entry) * qty
                    else:
                        exit_fill = float(exit_price) * (1.0 + slip)
                        gross = (entry - exit_fill) * qty

                    # Fees: entry + exit
                    fee = (entry * qty + exit_fill * qty) * self.taker_fee
                    net = gross - fee

                    equity += net

                    trades.append(
                        BacktestTrade(
                            entry_time=position['entry_time'],
                            exit_time=ts,
                            symbol=symbol,
                            side=side,
                            entry_price=entry,
                            exit_price=float(exit_fill),
                            quantity=qty,
                            pnl=float(net),
                            pnl_pct=float(net) / max(1e-12, (entry * qty)) * 100.0,
                            commission=float(fee),
                            slippage=float(slip),
                            hold_time=ts - position['entry_time'],
                            exit_reason=str(exit_reason),
                            mae=float(position.get('mae', 0.0)),
                            mfe=float(position.get('mfe', 0.0)),
                        )
                    )

                    position = None

            # ---------------- entry logic ----------------
            if position is None and signal is not None:
                if (i - last_entry_i) < self.cooldown_bars:
                    pass
                else:
                    action = getattr(signal, 'action', None)
                    conf = float(getattr(signal, 'confidence', 0.0) or 0.0)
                    if action in ('LONG', 'SHORT') and conf >= 0.55:
                        entry_price_ref = float(getattr(signal, 'entry_price', 0.0) or c)
                        sl = float(getattr(signal, 'stop_loss', 0.0) or 0.0)
                        tp = float(getattr(signal, 'take_profit', 0.0) or 0.0)

                        # If strategy doesn't provide SL/TP, derive from ATR (conservative)
                        atr = float(features.get('atr', 0.0) or 0.0)
                        if atr <= 0:
                            atr = max(1e-12, c * 0.002)  # 0.2% fallback

                        if sl <= 0 or tp <= 0:
                            if action == 'LONG':
                                sl = entry_price_ref - 2.0 * atr
                                tp = entry_price_ref + 3.0 * atr
                            else:
                                sl = entry_price_ref + 2.0 * atr
                                tp = entry_price_ref - 3.0 * atr

                        lev = int(getattr(signal, 'leverage', 1) or 1)
                        qty = self._compute_position_qty(equity, entry_price_ref, sl, leverage=lev)
                        if qty > 0:
                            slip = self._estimate_slippage_pct(row)
                            # Market entry -> pay slippage unfavorably
                            if action == 'LONG':
                                entry_fill = entry_price_ref * (1.0 + slip)
                            else:
                                entry_fill = entry_price_ref * (1.0 - slip)

                            position = {
                                'entry_time': ts,
                                'entry_i': i,
                                'side': action,
                                'entry_price': float(entry_fill),
                                'quantity': float(qty),
                                'sl': float(sl),
                                'tp': float(tp),
                                'mae': 0.0,
                                'mfe': 0.0,
                            }
                            last_entry_i = i

            equity_curve.append(equity)
            timestamps.append(ts)

        return {
            'trades': trades,
            'equity_curve': equity_curve,
            'timestamps': timestamps,
            'final_equity': equity,
        }

    # --------------------------
    # Summaries
    # --------------------------

    def _days_to_bars(self, df: pd.DataFrame, days: int) -> int:
        if days <= 0 or df is None or df.empty:
            return 0
        # Rough conversion based on median delta
        try:
            deltas = df.index.to_series().diff().dropna()
            med = deltas.median()
            if pd.isna(med) or med.total_seconds() <= 0:
                return 0
            bars_per_day = int(round(86400.0 / med.total_seconds()))
            return max(1, int(days) * bars_per_day)
        except Exception:
            return 0

    def _summarize(self, symbol: str, trades: List[BacktestTrade], equity_curve: List[float], timestamps: List[datetime]) -> Dict[str, Any]:
        trades = trades or []
        if not equity_curve:
            equity_curve = [self.initial_equity]
        final_equity = float(equity_curve[-1])

        n = len(trades)
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]

        win_rate = (len(wins) / n) if n > 0 else 0.0
        gross_profit = float(sum(t.pnl for t in wins)) if wins else 0.0
        gross_loss = float(-sum(t.pnl for t in losses)) if losses else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 1e-12 else (math.inf if gross_profit > 0 else 0.0)

        # Equity stats
        eq = np.array(equity_curve, dtype=float)
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak)
        dd_pct = np.where(peak > 0, dd / peak, 0.0)
        max_dd_pct = float(np.min(dd_pct)) if len(dd_pct) else 0.0

        # Return series (simple)
        rets = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
        vol = float(np.std(rets) * math.sqrt(365.0)) if len(rets) > 2 else 0.0
        sharpe = float((np.mean(rets) / (np.std(rets) + 1e-12)) * math.sqrt(365.0)) if len(rets) > 2 else 0.0

        return {
            'symbol': symbol,
            'initial_equity': self.initial_equity,
            'final_equity': final_equity,
            'net_pnl': final_equity - self.initial_equity,
            'net_pnl_pct': (final_equity / self.initial_equity - 1.0) * 100.0 if self.initial_equity > 0 else 0.0,
            'trades': trades,
            'num_trades': n,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown_pct': max_dd_pct * 100.0,
            'volatility_ann': vol,
            'sharpe_ann': sharpe,
            'equity_curve': equity_curve,
            'timestamps': timestamps,
        }
