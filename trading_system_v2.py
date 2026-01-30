"""
Main Trading System V2.0 - COMPLETE FIXED VERSION (2026)
- Fixed public mode fallback
- Override client an toàn
- Improved passphrase handling
- THÊM: Logic đầy đủ trong start() để generate signal và execute
- THÊM: Logging chi tiết khi không có signal
- THÊM: Kiểm tra volatility trước khi scan
- THÊM: Lock cho equity/positions
- THÊM: Telegram hai chiều (nhận lệnh /status /pause /resume /close_all /report)
- FIX: extract_features() chỉ nhận symbol, không truyền df thừa
- FIX MỚI: Gộp signal + PnL giả lập thành 1 thông báo cho mỗi mode
- FIX: SL/TP hợp lý, exit giả lập khi hit SL/TP
- FIX: 1 coin chỉ 1 lệnh (chờ đóng mới mở mới)
- FIX: BTC leader trước, altcoin theo vol tốt (check last_btc_signal_time, vol >0.001)
- TỐI ƯU: Giảm scan_interval=60s để tăng tần suất lệnh
- NÂNG CẤP PNL: Thêm DCA (mua thêm nếu giá giảm 2%), arbitrage basic (if price diff >0.5%), EMA crossover filter
"""

import os
import sys
import time
import json
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import getpass
import signal
import threading
import urllib.request
import urllib.parse
import uuid


class _FileLock:
    """Simple cross-platform lock using a lock file (no extra deps).

    - Creates <path>.lock atomically (O_EXCL).
    - Retries for a short period.
    """

    def __init__(self, lock_path: str, timeout_s: float = 5.0, poll_s: float = 0.05):
        self.lock_path = lock_path
        self.timeout_s = timeout_s
        self.poll_s = poll_s
        self._fd = None

    def acquire(self):
        start = time.time()
        while True:
            try:
                self._fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.write(self._fd, str(os.getpid()).encode('utf-8', errors='ignore'))
                return
            except FileExistsError:
                if (time.time() - start) >= self.timeout_s:
                    raise TimeoutError(f"Lock timeout: {self.lock_path}")
                time.sleep(self.poll_s)

    def release(self):
        try:
            if self._fd is not None:
                os.close(self._fd)
        except Exception:
            pass
        self._fd = None
        try:
            if os.path.exists(self.lock_path):
                os.remove(self.lock_path)
        except Exception:
            pass

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()



class ConfigReloader:
    """Hot reload config.json when file mtime changes (no extra deps)."""

    def __init__(self, path: str):
        self.path = path
        self.last_mtime = 0.0

    def maybe_reload(self):
        try:
            st = os.stat(self.path)
            mtime = float(st.st_mtime)
        except Exception:
            return None
        if mtime <= float(self.last_mtime):
            return None
        try:
            with open(self.path, 'r', encoding='utf-8-sig') as f:
                cfg = json.load(f)
            self.last_mtime = mtime
            return cfg
        except Exception:
            logging.exception('[CONFIG] reload failed')
            return None


class CircuitBreaker:
    """Live safety circuit breaker.

    Trips live kill-switch if:
    - too many consecutive order/API errors
    - repeated reconcile failures (ghost position risk)
    - extreme order latency

    The breaker is intentionally conservative.
    """

    def __init__(self, cfg: dict):
        self.update_config(cfg)
        self.consec_order_errors = 0
        self.consec_reconcile_fail = 0
        self.last_trip_reason = ''
        self.tripped = False
        self.trip_ts = 0.0

    def update_config(self, cfg: dict):
        cb = (cfg or {}).get('circuit_breakers', {}) if isinstance(cfg, dict) else {}
        self.max_consec_order_errors = int(cb.get('max_consecutive_order_errors', 3) or 3)
        self.max_consec_reconcile_fail = int(cb.get('max_consecutive_reconcile_fail', 2) or 2)
        self.max_order_latency_ms = float(cb.get('max_order_latency_ms', 2500) or 2500)
        self.cooldown_seconds = int(cb.get('cooldown_seconds', 600) or 600)

    def record_order_ok(self):
        self.consec_order_errors = 0

    def record_order_error(self, reason: str):
        self.consec_order_errors += 1
        self.last_trip_reason = reason or 'ORDER_ERROR'

    def record_reconcile_ok(self):
        self.consec_reconcile_fail = 0

    def record_reconcile_fail(self, reason: str):
        self.consec_reconcile_fail += 1
        self.last_trip_reason = reason or 'RECONCILE_FAIL'

    def record_latency(self, latency_ms: float):
        try:
            if float(latency_ms) > float(self.max_order_latency_ms):
                self.last_trip_reason = f'LATENCY_MS>{self.max_order_latency_ms}'
                self.consec_order_errors += 1
        except Exception:
            pass

    def should_trip(self) -> bool:
        if self.consec_order_errors >= self.max_consec_order_errors:
            return True
        if self.consec_reconcile_fail >= self.max_consec_reconcile_fail:
            return True
        return False

    def trip(self, reason: str):
        self.tripped = True
        self.trip_ts = time.time()
        self.last_trip_reason = reason or self.last_trip_reason or 'TRIPPED'

    def is_in_cooldown(self) -> bool:
        if not self.tripped:
            return False
        return (time.time() - float(self.trip_ts)) < float(self.cooldown_seconds)


class MicrostructureGuard:
    """Pre-trade microstructure safety checks (CIO-grade best-effort).

    Rationale: Futures market can become toxic (spread spikes, mark/last divergence,
    volatility shocks). Entering with MARKET in those moments is a common source
    of slippage and stop-hunts.

    This guard is intentionally conservative: when in doubt -> block entry.
    """

    def __init__(self, cfg: dict):
        self.update_config(cfg)
        self._last_ok = {}

    def update_config(self, cfg: dict):
        ms = (cfg or {}).get('microstructure', {}) if isinstance(cfg, dict) else {}
        # Spread limits (pct)
        self.max_spread_pct = float(ms.get('max_spread_pct', 0.0018) or 0.0018)  # 0.18%
        # Mark/last divergence (pct)
        self.max_mark_last_div_pct = float(ms.get('max_mark_last_div_pct', 0.0025) or 0.0025)  # 0.25%
        # Volatility shock gate (5m return abs)
        self.max_abs_ret_5m = float(ms.get('max_abs_ret_5m', 0.012) or 0.012)  # 1.2%
        # Cooldown per-symbol after block
        self.symbol_cooldown_s = int(ms.get('symbol_cooldown_s', 90) or 90)

    def _cooldown_active(self, symbol: str) -> bool:
        ts = float(self._last_ok.get(symbol, 0.0) or 0.0)
        return (time.time() - ts) < float(self.symbol_cooldown_s)

    def check(self, symbol: str, market_data, features: dict) -> Tuple[bool, str, dict]:
        if not symbol or market_data is None:
            return False, 'MS_NO_DATA', {}

        # Simple per-symbol cooldown to avoid hammering during toxic conditions
        if self._cooldown_active(symbol):
            return False, 'MS_COOLDOWN', {}

        diag = {}
        # Spread check
        try:
            ba = market_data.get_best_bid_ask(symbol)
            spread_pct = float(ba.get('spread_pct', 0.0) or 0.0)
            diag['spread_pct'] = spread_pct
            if spread_pct > self.max_spread_pct:
                return False, 'MS_SPREAD', diag
        except Exception:
            return False, 'MS_SPREAD_ERR', diag

        # Mark/last divergence
        try:
            last_px = float(market_data.get_current_price(symbol) or 0.0)
            mark_px = float(getattr(market_data, 'get_mark_price')(symbol) or 0.0)
            if last_px > 0 and mark_px > 0:
                div = abs(mark_px - last_px) / last_px
            else:
                div = 0.0
            diag['mark_last_div_pct'] = div
            if div > self.max_mark_last_div_pct:
                return False, 'MS_MARK_LAST', diag
        except Exception:
            return False, 'MS_MARK_LAST_ERR', diag

        # Volatility shock (5m abs return)
        try:
            ret5 = float((features or {}).get('return_5m', 0.0) or 0.0)
            diag['abs_ret_5m'] = abs(ret5)
            if abs(ret5) > self.max_abs_ret_5m:
                return False, 'MS_VOL_SHOCK', diag
        except Exception:
            pass

        # OK
        self._last_ok[symbol] = time.time()
        return True, 'OK', diag


class RiskGovernor:
    """Independent runtime risk governor.

    CIO-grade rule: strategy may be good, but execution/risk must never explode.
    Governor clamps position sizing and leverage *without changing your base config*.
    """

    def __init__(self, cfg: dict):
        self.update_config(cfg)
        self.last_scale = 1.0

    def update_config(self, cfg: dict):
        gov = (cfg or {}).get('governor', {}) if isinstance(cfg, dict) else {}
        self.enabled = bool(gov.get('enabled', True))
        self.max_leverage_cap = int(gov.get('max_leverage_cap', 20) or 20)
        self.min_leverage_cap = int(gov.get('min_leverage_cap', 5) or 5)
        self.dd_soft_pct = float(gov.get('dd_soft_pct', 0.08) or 0.08)   # 8%
        self.dd_hard_pct = float(gov.get('dd_hard_pct', 0.15) or 0.15)   # 15%
        self.loss_streak_soft = int(gov.get('loss_streak_soft', 3) or 3)
        self.loss_streak_hard = int(gov.get('loss_streak_hard', 5) or 5)
        self.vol_shock_scale = float(gov.get('vol_shock_scale', 0.6) or 0.6)

    def compute(self, risk_metrics: dict, features: dict) -> Tuple[float, int, str]:
        """Return (qty_scale, leverage_cap, reason)."""
        if not self.enabled:
            return 1.0, self.max_leverage_cap, 'GOV_OFF'

        dd = 0.0
        loss_streak = 0
        state = ''
        try:
            dd = float(risk_metrics.get('drawdown_pct', 0.0) or 0.0)
            loss_streak = int(risk_metrics.get('loss_streak', 0) or 0)
            state = str(risk_metrics.get('state', '') or '')
        except Exception:
            pass

        # Base scale
        scale = 1.0
        lev_cap = self.max_leverage_cap
        reason = 'GOV_OK'

        # Risk-state clamps
        if state in ('CRITICAL', 'DEAD'):
            return 0.0, self.min_leverage_cap, 'GOV_STATE_CRITICAL'
        if state == 'DANGER':
            scale *= 0.35
            lev_cap = max(self.min_leverage_cap, int(self.max_leverage_cap * 0.5))
            reason = 'GOV_STATE_DANGER'
        elif state == 'CAUTION':
            scale *= 0.65
            lev_cap = max(self.min_leverage_cap, int(self.max_leverage_cap * 0.75))
            reason = 'GOV_STATE_CAUTION'

        # Drawdown clamps
        if dd >= self.dd_hard_pct:
            scale *= 0.25
            lev_cap = self.min_leverage_cap
            reason = 'GOV_DD_HARD'
        elif dd >= self.dd_soft_pct:
            scale *= 0.60
            lev_cap = max(self.min_leverage_cap, int(self.max_leverage_cap * 0.7))
            reason = 'GOV_DD_SOFT'

        # Loss streak clamps
        if loss_streak >= self.loss_streak_hard:
            scale *= 0.25
            lev_cap = self.min_leverage_cap
            reason = 'GOV_STREAK_HARD'
        elif loss_streak >= self.loss_streak_soft:
            scale *= 0.65
            lev_cap = max(self.min_leverage_cap, int(self.max_leverage_cap * 0.7))
            reason = 'GOV_STREAK_SOFT'

        # Volatility shock clamp (use return_5m)
        try:
            ret5 = abs(float((features or {}).get('return_5m', 0.0) or 0.0))
            if ret5 > 0.012:
                scale *= float(self.vol_shock_scale)
                reason = 'GOV_VOL_SHOCK'
        except Exception:
            pass

        scale = max(0.0, min(1.0, float(scale)))
        lev_cap = int(max(self.min_leverage_cap, min(self.max_leverage_cap, int(lev_cap))))
        self.last_scale = scale
        return scale, lev_cap, reason


class MetricsEngine:
    """Runtime metrics with CIO-grade auto-de-risk / auto-pause.

    Keeps rolling windows (in-memory) and periodically persists a snapshot to AccountStateManager.
    This is intentionally lightweight (no extra deps).
    """

    def __init__(self, cfg: dict):
        self.update_config(cfg)
        self._orders = []   # list of dict
        self._trades = []   # list of dict
        self._last_persist = 0.0
        self._bad_streak = 0

    def update_config(self, cfg: dict):
        m = (cfg or {}).get('metrics', {}) if isinstance(cfg, dict) else {}
        self.enabled = bool(m.get('enabled', True))
        self.window_orders = int(m.get('window_orders', 50) or 50)
        self.window_trades = int(m.get('window_trades', 40) or 40)
        self.persist_every_s = int(m.get('persist_every_seconds', 120) or 120)

        a = (cfg or {}).get('alerts', {}) if isinstance(cfg, dict) else {}
        # thresholds for auto-pause
        self.max_avg_slippage_pct = float(a.get('max_avg_slippage_pct', 0.0045) or 0.0045)  # 0.45%
        self.max_p95_latency_ms = float(a.get('max_p95_latency_ms', 3500) or 3500)
        self.max_reject_rate = float(a.get('max_reject_rate', 0.35) or 0.35)
        self.max_bad_streak = int(a.get('max_bad_streak', 2) or 2)

    def record_order(self, symbol: str, latency_ms: float, slippage_pct: float, ok: bool, meta: dict = None):
        if not self.enabled:
            return
        self._orders.append({
            'ts': time.time(),
            'symbol': symbol,
            'latency_ms': float(latency_ms or 0.0),
            'slippage_pct': float(slippage_pct or 0.0),
            'ok': bool(ok),
            'meta': meta or {},
        })
        if len(self._orders) > self.window_orders:
            self._orders = self._orders[-self.window_orders:]

    def record_trade_close(self, symbol: str, pnl_net: float, reason: str, meta: dict = None):
        if not self.enabled:
            return
        self._trades.append({
            'ts': time.time(),
            'symbol': symbol,
            'pnl_net': float(pnl_net or 0.0),
            'reason': str(reason or ''),
            'meta': meta or {},
        })
        if len(self._trades) > self.window_trades:
            self._trades = self._trades[-self.window_trades:]

    def snapshot(self) -> dict:
        # Reject rate
        orders = list(self._orders)
        n = len(orders)
        reject_rate = 0.0
        lat = []
        slip = []
        if n > 0:
            rej = sum(1 for o in orders if not o.get('ok', False))
            reject_rate = rej / float(n)
            lat = [float(o.get('latency_ms', 0.0) or 0.0) for o in orders]
            slip = [float(o.get('slippage_pct', 0.0) or 0.0) for o in orders]
        lat_sorted = sorted(lat)
        def _p95(xs):
            if not xs:
                return 0.0
            k = int(max(0, min(len(xs) - 1, round(0.95 * (len(xs) - 1)))))
            return float(xs[k])

        avg_lat = (sum(lat) / len(lat)) if lat else 0.0
        p95_lat = _p95(lat_sorted)
        avg_slip = (sum(slip) / len(slip)) if slip else 0.0

        # Trade expectancy
        trades = list(self._trades)
        tn = len(trades)
        win = sum(1 for t in trades if float(t.get('pnl_net', 0.0) or 0.0) > 0)
        loss = sum(1 for t in trades if float(t.get('pnl_net', 0.0) or 0.0) < 0)
        avg_pnl = (sum(float(t.get('pnl_net', 0.0) or 0.0) for t in trades) / tn) if tn else 0.0

        return {
            'orders_n': n,
            'reject_rate': float(reject_rate),
            'avg_latency_ms': float(avg_lat),
            'p95_latency_ms': float(p95_lat),
            'avg_slippage_pct': float(avg_slip),
            'trades_n': tn,
            'win_n': int(win),
            'loss_n': int(loss),
            'avg_pnl_net': float(avg_pnl),
            'ts': datetime.now().isoformat(),
        }

    def check_and_maybe_trip(self) -> Tuple[bool, str, dict]:
        """Return (trip?, reason, snapshot)."""
        snap = self.snapshot()
        if not self.enabled:
            return False, 'METRICS_OFF', snap

        bad = False
        reason = ''
        if snap.get('avg_slippage_pct', 0.0) > self.max_avg_slippage_pct and snap.get('orders_n', 0) >= 10:
            bad = True
            reason = 'METRIC_SLIPPAGE'
        if snap.get('p95_latency_ms', 0.0) > self.max_p95_latency_ms and snap.get('orders_n', 0) >= 10:
            bad = True
            reason = 'METRIC_LATENCY'
        if snap.get('reject_rate', 0.0) > self.max_reject_rate and snap.get('orders_n', 0) >= 10:
            bad = True
            reason = 'METRIC_REJECT'

        if bad:
            self._bad_streak += 1
        else:
            self._bad_streak = 0

        if self._bad_streak >= self.max_bad_streak:
            return True, reason or 'METRIC_BAD_STREAK', snap
        return False, 'OK', snap

    def maybe_persist(self, state_mgr: 'AccountStateManager', modes: List[str]):
        if not self.enabled:
            return
        now = time.time()
        if (now - float(self._last_persist or 0.0)) < float(self.persist_every_s):
            return
        self._last_persist = now
        snap = self.snapshot()
        for m in modes:
            try:
                state_mgr.apply_metrics(m, snap)
            except Exception:
                pass


class AccountStateManager:
    """Single source of truth for balances/equity per mode.

    v10 upgrades:
    - atomic persist + lock
    - event-sourcing (append-only jsonl)
    - safe reconciliation hook for LIVE
    """

    def __init__(self, base_dir: str, initial_equity: float = 1000.0):
        self.base_dir = base_dir
        self.initial_equity = float(initial_equity or 1000.0)
        self._mem = {}
        self._mem_lock = threading.Lock()

    def _state_path(self, mode: str) -> str:
        return os.path.join(self.base_dir, f'state_{mode}.json')

    def _events_path(self, mode: str) -> str:
        return os.path.join(self.base_dir, f'events_{mode}.jsonl')

    def _lock_path(self, mode: str) -> str:
        return os.path.join(self.base_dir, f'state_{mode}.lock')

    def _default_state(self) -> Dict:
        eq = self.initial_equity
        return {
            'balance': eq,
            'equity': eq,
            'realized_pnl': 0.0,
            'unrealized_pnl': 0.0,
            'peak_equity': eq,
            # Persisted pause across restarts (fund-grade safety)
            'system_paused': False,
            'system_pause_reason': '',
            'updated_at': datetime.now().isoformat(),
        }

    def load_all(self, modes: List[str]):
        for m in modes:
            self.get(m)  # lazy load

    def get(self, mode: str) -> Dict:
        with self._mem_lock:
            if mode in self._mem:
                return dict(self._mem[mode])

        path = self._state_path(mode)
        st = None
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    st = json.load(f)
        except Exception:
            logging.exception(f"Failed to load state: {path}")

        if not isinstance(st, dict):
            st = self._default_state()

        for k in ['balance', 'equity', 'realized_pnl', 'unrealized_pnl', 'peak_equity']:
            try:
                st[k] = float(st.get(k, 0.0) or 0.0)
            except Exception:
                st[k] = 0.0
        # Fund-grade: persisted pause flag
        if 'system_paused' not in st:
            st['system_paused'] = False
        if 'system_pause_reason' not in st:
            st['system_pause_reason'] = ''


        with self._mem_lock:
            self._mem[mode] = dict(st)
        return dict(st)

    def _save_atomic_locked(self, mode: str, st: Dict):
        lock_path = self._lock_path(mode)
        with _FileLock(lock_path, timeout_s=5.0):
            st = dict(st)
            st['updated_at'] = datetime.now().isoformat()
            try:
                st['peak_equity'] = max(float(st.get('peak_equity', st.get('equity', 0.0) or 0.0)), float(st.get('equity', 0.0) or 0.0))
            except Exception:
                pass

            tmp = self._state_path(mode) + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(st, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self._state_path(mode))

    def set(self, mode: str, **kwargs):
        st = self.get(mode)
        for k, v in kwargs.items():
            st[k] = v
        with self._mem_lock:
            self._mem[mode] = dict(st)
        self._save_atomic_locked(mode, st)
        return dict(st)

    def append_event(self, mode: str, event_type: str, payload: Dict):
        evt = {
            'ts': datetime.now().isoformat(),
            'mode': mode,
            'type': event_type,
            'id': payload.get('id') or uuid.uuid4().hex,
            'payload': payload,
        }
        # append-only; best-effort
        try:
            with open(self._events_path(mode), 'a', encoding='utf-8') as f:
                f.write(json.dumps(evt, ensure_ascii=False) + "\n")
        except Exception:
            logging.exception("Failed to append event")

    def apply_unrealized(self, mode: str, unrealized_pnl: float):
        st = self.get(mode)
        st['unrealized_pnl'] = float(unrealized_pnl or 0.0)
        st['equity'] = float(st.get('balance', 0.0) or 0.0) + float(st['unrealized_pnl'])
        with self._mem_lock:
            self._mem[mode] = dict(st)
        self._save_atomic_locked(mode, st)
        return dict(st)

    def apply_trade_close(self, mode: str, pnl_net: float, meta: Dict):
        st = self.get(mode)
        pnl_net = float(pnl_net or 0.0)
        st['balance'] = float(st.get('balance', 0.0) or 0.0) + pnl_net
        st['realized_pnl'] = float(st.get('realized_pnl', 0.0) or 0.0) + pnl_net
        st['equity'] = float(st['balance']) + float(st.get('unrealized_pnl', 0.0) or 0.0)
        with self._mem_lock:
            self._mem[mode] = dict(st)
        self._save_atomic_locked(mode, st)
        self.append_event(mode, str(meta.get('event_type','CLOSE')), dict(meta, pnl_net=pnl_net, balance=st['balance'], equity=st['equity']))
        return dict(st)

    def apply_metrics(self, mode: str, metrics: Dict):
        """Persist lightweight runtime metrics (best-effort).

        Stored under state['metrics'] and also appended as an event.
        """
        st = self.get(mode)
        try:
            st['metrics'] = dict(metrics or {})
        except Exception:
            st['metrics'] = {}
        with self._mem_lock:
            self._mem[mode] = dict(st)
        self._save_atomic_locked(mode, st)
        try:
            self.append_event(mode, 'METRICS', {'metrics': st.get('metrics', {})})
        except Exception:
            pass
        return dict(st)


class BanditTuner:
    """Lightweight self-evolution with guardrails (no extra deps).

    Chooses among a few conservative parameter profiles using UCB1.
    Reward = realized pnl net (after fees) with a drawdown penalty.
    """

    def __init__(self, base_dir: str, state_mgr: AccountStateManager, strategy_engine):
        self.base_dir = base_dir
        self.state_mgr = state_mgr
        self.strategy_engine = strategy_engine
        self.path = os.path.join(base_dir, 'tuner_state.json')
        self.active_profile = None
        self.last_ts = None

        # Conservative profiles: trade quality vs frequency
        # Runtime profiles (AdaptiveEngine rotates/chooses). We add a frequency-first
        # profile to meet the user's target of ~5-15+ trades/day.
        self.profiles = [
            {'name': 'hyper', 'min_confidence': 0.20, 'min_volatility': 0.00030, 'rr_min': 1.05,
             'compression_atr_ratio_max': 0.95, 'allow_range_mode': True,
             'cost_gate_reward_mult': 1.12, 'cost_gate_risk_mult': 0.85, 'cost_gate_slip_cushion': 0.00012,
             'follower_signal_age_min': {'SOLUSDT': 90, 'XRPUSDT': 120, 'ETHUSDT': 120}},
            {'name': 'aggressive', 'min_confidence': 0.24, 'min_volatility': 0.00035, 'rr_min': 1.10,
             'compression_atr_ratio_max': 0.92, 'allow_range_mode': True,
             'cost_gate_reward_mult': 1.15, 'cost_gate_risk_mult': 0.90, 'cost_gate_slip_cushion': 0.00015},
            {'name': 'balanced', 'min_confidence': 0.30, 'min_volatility': 0.00045, 'rr_min': 1.20,
             'compression_atr_ratio_max': 0.88, 'allow_range_mode': True,
             'cost_gate_reward_mult': 1.22, 'cost_gate_risk_mult': 0.95, 'cost_gate_slip_cushion': 0.00018},
            {'name': 'selective', 'min_confidence': 0.40, 'min_volatility': 0.00065, 'rr_min': 1.35,
             'compression_atr_ratio_max': 0.80, 'allow_range_mode': False,
             'cost_gate_reward_mult': 1.35, 'cost_gate_risk_mult': 1.00, 'cost_gate_slip_cushion': 0.00022},
        ]
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path, 'r', encoding='utf-8') as f:
                    st = json.load(f)
                if isinstance(st, dict):
                    self.active_profile = st.get('active_profile')
                    self.last_ts = st.get('last_ts')
                    self.stats = st.get('stats') or {}
                    return
        except Exception:
            pass
        self.stats = {p['name']: {'n': 0, 'sum': 0.0} for p in self.profiles}

    def _save(self):
        try:
            tmp = self.path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump({'active_profile': self.active_profile, 'last_ts': self.last_ts, 'stats': self.stats}, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except Exception:
            pass

    def _ucb_score(self, name: str, total_n: int) -> float:
        s = self.stats.get(name, {'n': 0, 'sum': 0.0})
        n = int(s.get('n', 0) or 0)
        mean = (float(s.get('sum', 0.0) or 0.0) / n) if n > 0 else 0.0
        if n == 0:
            return 1e9
        import math
        return mean + 0.5 * math.sqrt(math.log(max(1, total_n)) / n)

    def choose_and_apply(self):
        total_n = sum(int(self.stats.get(p['name'], {}).get('n', 0) or 0) for p in self.profiles) + 1
        best = None
        best_score = -1e18
        for p in self.profiles:
            sc = self._ucb_score(p['name'], total_n)
            if sc > best_score:
                best_score = sc
                best = p
        if not best:
            best = self.profiles[0]
        self.active_profile = best['name']
        # Apply full profile to strategy engine (frequency/quality gates).
        # set_runtime_params silently ignores unknown attributes.
        try:
            self.strategy_engine.set_runtime_params(
                min_confidence=best.get('min_confidence'),
                min_volatility=best.get('min_volatility'),
                rr_min=best.get('rr_min'),
                compression_atr_ratio_max=best.get('compression_atr_ratio_max'),
                allow_range_mode=best.get('allow_range_mode'),
                cost_gate_reward_mult=best.get('cost_gate_reward_mult'),
                cost_gate_risk_mult=best.get('cost_gate_risk_mult'),
                cost_gate_slip_cushion=best.get('cost_gate_slip_cushion'),
                follower_signal_age_min=best.get('follower_signal_age_min'),
            )
        except Exception:
            pass
        self._save()
        return best

    def observe_reward(self, mode: str = 'paper'):
        # Reward: change in realized pnl since last observation (best-effort)
        try:
            st = self.state_mgr.get(mode)
            realized = float(st.get('realized_pnl', 0.0) or 0.0)
        except Exception:
            return

        key = f'last_realized_{mode}'
        last = float(self.stats.get('_meta', {}).get(key, 0.0) or 0.0)
        delta = realized - last

        # drawdown penalty (encourage stability)
        try:
            peak = float(st.get('peak_equity', st.get('equity', 0.0)) or 0.0)
            eq = float(st.get('equity', 0.0) or 0.0)
            dd = max(0.0, peak - eq)
        except Exception:
            dd = 0.0
        reward = float(delta) - 0.2 * float(dd)

        if self.active_profile:
            s = self.stats.get(self.active_profile, {'n': 0, 'sum': 0.0})
            s['n'] = int(s.get('n', 0) or 0) + 1
            s['sum'] = float(s.get('sum', 0.0) or 0.0) + reward
            self.stats[self.active_profile] = s

        meta = self.stats.get('_meta', {})
        meta[key] = realized
        self.stats['_meta'] = meta
        self._save()

    def maybe_rotate(self, interval_min: int = 30):
        now = datetime.now()
        if self.last_ts:
            try:
                last = datetime.fromisoformat(self.last_ts)
                if (now - last).total_seconds() < interval_min * 60:
                    return
            except Exception:
                pass
        # observe reward from previous profile before switching
        self.observe_reward('paper')
        prof = self.choose_and_apply()
        self.last_ts = now.isoformat()
        self._save()
        return prof


class AdaptiveEngine:
    """CIO-grade adaptive controller (no extra deps).

    Goals:
    - Increase trade frequency when market allows (without destroying slippage).
    - Adapt parameters across regimes (trend vs range) using guardrailed bandit.
    - Auto-rollback when the chosen profile degrades expectancy or ops quality.

    This engine applies *runtime overrides* only (no secret in config; no permanent mutations).
    """

    def __init__(self, base_dir: str, config: dict, state_mgr: 'AccountStateManager',
                 strategy_engine, risk_guard, micro_guard: 'MicrostructureGuard',
                 governor: 'RiskGovernor', metrics: 'MetricsEngine'):
        self.base_dir = base_dir
        self.config = config if isinstance(config, dict) else {}
        self.state_mgr = state_mgr
        self.strategy_engine = strategy_engine
        self.risk_guard = risk_guard
        self.micro = micro_guard
        self.governor = governor
        self.metrics = metrics

        self.path = os.path.join(base_dir, 'adaptive_state.json')
        self.last_ts = None
        self.active_profile = None
        self.prev_profile = None
        self.meta_tune = {}
        self._load()

        ae = (self.config.get('adaptive_engine', {}) or {}) if isinstance(self.config, dict) else {}
        self.enabled = bool(ae.get('enabled', True))
        self.learn_mode = str(ae.get('learn_mode', 'paper') or 'paper')  # paper|live
        self.rotate_minutes = int(ae.get('rotate_minutes', 15) or 15)
        self.min_orders_for_ops = int(ae.get('min_orders_for_ops', 10) or 10)

        # Profile definitions (can be overridden by config)
        self.profiles = ae.get('profiles')
        if not isinstance(self.profiles, list) or not self.profiles:
            self.profiles = self._default_profiles()

        # UCB stats
        self.stats = getattr(self, 'stats', None)
        if not isinstance(self.stats, dict) or not self.stats:
            self.stats = {p['name']: {'n': 0, 'sum': 0.0} for p in self.profiles}

        # Rollback policy
        rb = (ae.get('rollback', {}) or {})
        self.rollback_min_trades = int(rb.get('min_trades', 10) or 10)
        self.rollback_bad_expectancy = float(rb.get('bad_expectancy', 0.0) or 0.0)
        self.rollback_max_reject = float(rb.get('max_reject_rate', 0.40) or 0.40)
        self.rollback_max_slip = float(rb.get('max_avg_slippage_pct', 0.0060) or 0.0060)

        # Apply initial profile
        try:
            if self.active_profile:
                self.apply(self.active_profile)
            else:
                self.choose_and_apply()

            # Apply any meta-tuning overrides loaded from state
            try:
                self._apply_meta_tune()
            except Exception:
                pass
        except Exception:
            pass

    def _default_profiles(self) -> List[dict]:
        # Trend profiles: higher RR, stricter filters.
        # Range profiles: more trades; lower RR; still BTC-led + microstructure guarded.
        return [
            {
                'name': 'trend_strict',
                'strategy': {
                    'min_confidence': 0.46,
                    'rr_min': 1.55,
                    'compression_atr_ratio_max': 0.80,
                    'allow_range_mode': False,
                    'follower_signal_age_min': {'SOLUSDT': 14, 'XRPUSDT': 18, 'ETHUSDT': 22},
                    # BTC leader thresholds (strict): follow only when BTC strength is clearly strong
                    'follower_requirements': {
                        'SOLUSDT': {'min_btc_strength': 0.68, 'max_range_atr': 1.6, 'max_ext_ma50': 0.020},
                        'XRPUSDT': {'min_btc_strength': 0.62, 'max_range_atr': 1.8, 'max_ext_ma50': 0.018},
                        'ETHUSDT': {'min_btc_strength': 0.58, 'max_range_atr': 2.0, 'max_ext_ma50': 0.015},
                    }
                },
                'risk': {'risk_scale': 0.95},
                'micro': {'max_spread_pct': 0.0022, 'max_mark_last_div_pct': 0.0030, 'max_abs_ret_5m': 0.013, 'symbol_cooldown_s': 75},
                'governor': {'max_leverage_cap': 18}
            },
            {
                'name': 'trend_balanced',
                'strategy': {
                    'min_confidence': 0.40,
                    'rr_min': 1.40,
                    'compression_atr_ratio_max': 0.83,
                    'allow_range_mode': True,
                    'range_rsi_oversold': 32,
                    'range_rsi_overbought': 68,
                    'range_dev_ma20_min': 0.0016,
                    'follower_signal_age_min': {'SOLUSDT': 16, 'XRPUSDT': 20, 'ETHUSDT': 25},
                    # Slightly looser BTC strength so system can trade more when ops quality is OK
                    'follower_requirements': {
                        # FREQ-FIRST override: the logs show BTC strength often ~0.25-0.40 in chop.
                        # Keeping 0.55-0.62 starves followers for hours.
                        'SOLUSDT': {'min_btc_strength': 0.38, 'max_range_atr': 1.9, 'max_ext_ma50': 0.020},
                        'XRPUSDT': {'min_btc_strength': 0.36, 'max_range_atr': 2.1, 'max_ext_ma50': 0.018},
                        'ETHUSDT': {'min_btc_strength': 0.34, 'max_range_atr': 2.3, 'max_ext_ma50': 0.016},
                    }
                },
                'risk': {'risk_scale': 1.00},
                'micro': {'max_spread_pct': 0.0026, 'max_mark_last_div_pct': 0.0035, 'max_abs_ret_5m': 0.016, 'symbol_cooldown_s': 60},
                'governor': {'max_leverage_cap': 22}
            },
            {
                'name': 'range_balanced',
                'strategy': {
                    'min_confidence': 0.36,
                    'rr_min': 1.25,
                    'compression_atr_ratio_max': 0.88,
                    'allow_range_mode': True,
                    'range_rsi_oversold': 30,
                    'range_rsi_overbought': 70,
                    'range_dev_ma20_min': 0.0014,
                    'follower_signal_age_min': {'SOLUSDT': 18, 'XRPUSDT': 22, 'ETHUSDT': 28},
                    # Range: more trades, BTC strength gate loosened but still bounded
                    'follower_requirements': {
                        'SOLUSDT': {'min_btc_strength': 0.34, 'max_range_atr': 2.1, 'max_ext_ma50': 0.023},
                        'XRPUSDT': {'min_btc_strength': 0.32, 'max_range_atr': 2.2, 'max_ext_ma50': 0.021},
                        'ETHUSDT': {'min_btc_strength': 0.30, 'max_range_atr': 2.4, 'max_ext_ma50': 0.018},
                    }
                },
                'risk': {'risk_scale': 1.08},
                'micro': {'max_spread_pct': 0.0030, 'max_mark_last_div_pct': 0.0040, 'max_abs_ret_5m': 0.018, 'symbol_cooldown_s': 55},
                'governor': {'max_leverage_cap': 24}
            },
            {
                'name': 'range_aggressive',
                'strategy': {
                    # FREQ-FIRST: lower threshold to reach 5-15+ trades/day when market is not dead.
                    'min_confidence': 0.26,
                    'rr_min': 1.10,
                    'compression_atr_ratio_max': 0.92,
                    'allow_range_mode': True,
                    # More permissive range triggers to reach the target trade frequency.
                    'range_rsi_oversold': 38,
                    'range_rsi_overbought': 62,
                    'range_dev_ma20_min': 0.0009,
                    'follower_signal_age_min': {'SOLUSDT': 20, 'XRPUSDT': 25, 'ETHUSDT': 30},
                    'follower_requirements': {
                        # NOTE: These are the most important knobs for trade frequency.
                        # The logs show BTC strength often sits around 0.25-0.40 in chop.
                        # If we keep 0.50-0.62, the system will starve.
                        'SOLUSDT': {'min_btc_strength': 0.30, 'max_range_atr': 2.2, 'max_ext_ma50': 0.026},
                        'XRPUSDT': {'min_btc_strength': 0.28, 'max_range_atr': 2.3, 'max_ext_ma50': 0.023},
                        'ETHUSDT': {'min_btc_strength': 0.26, 'max_range_atr': 2.5, 'max_ext_ma50': 0.020},
                    }
                },
                'risk': {'risk_scale': 1.15},
                'micro': {'max_spread_pct': 0.0032, 'max_mark_last_div_pct': 0.0044, 'max_abs_ret_5m': 0.019, 'symbol_cooldown_s': 50},
                'governor': {'max_leverage_cap': 25}
            }
        ]

    def _load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path, 'r', encoding='utf-8') as f:
                    st = json.load(f)
                if isinstance(st, dict):
                    self.active_profile = st.get('active_profile')
                    self.prev_profile = st.get('prev_profile')
                    self.last_ts = st.get('last_ts')
                    self.stats = st.get('stats') or {}
                    self.meta_tune = st.get('meta_tune') or {}
        except Exception:
            pass

    def _save(self):
        try:
            tmp = self.path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump({
                    'active_profile': self.active_profile,
                    'prev_profile': self.prev_profile,
                    'last_ts': self.last_ts,
                    'stats': self.stats,
                    'meta_tune': self.meta_tune,
                }, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except Exception:
            pass

    def _apply_meta_tune(self):
        """Apply meta-tuning overrides (safe bounds, runtime-only).

        Meta-tuning is a small adaptive layer on top of profiles:
        - nudges min_confidence to control trade frequency vs quality
        - nudges rr_min to control expectancy

        This helps the system self-adjust without changing folder structure or installing deps.
        """
        mt = self.meta_tune or {}
        if not isinstance(mt, dict):
            return

        # Read current runtime values from strategy (fallback defaults)
        try:
            cur_min_conf = float(getattr(self.strategy_engine, 'min_confidence', 0.35) or 0.35)
        except Exception:
            cur_min_conf = 0.35
        try:
            cur_rr = float(getattr(self.strategy_engine, 'rr_min', 1.35) or 1.35)
        except Exception:
            cur_rr = 1.35

        # Apply stored overrides if present
        new_min_conf = float(mt.get('min_confidence', cur_min_conf) or cur_min_conf)
        new_rr = float(mt.get('rr_min', cur_rr) or cur_rr)

        # Safety bounds
        # Allow lower confidence in FREQ-FIRST mode to avoid starvation.
        new_min_conf = max(0.22, min(0.55, new_min_conf))
        new_rr = max(1.10, min(1.85, new_rr))

        try:
            self.strategy_engine.set_runtime_params(min_confidence=new_min_conf, rr_min=new_rr)
        except Exception:
            pass

    def _meta_tune_step(self):
        """Update meta-tune knobs based on recent trades & ops quality (guardrailed).

        - If quality is poor (winrate/expectancy), tighten min_confidence and raise rr_min.
        - If system is too quiet but ops quality is OK, loosen min_confidence slightly.

        NOTE: This does not guarantee profits; it only adapts thresholds within safe bounds.
        """
        snap = {}
        try:
            snap = self.metrics.snapshot() if self.metrics is not None else {}
        except Exception:
            snap = {}

        trades_n = int(snap.get('trades_n', 0) or 0)
        win_n = int(snap.get('win_n', 0) or 0)
        loss_n = int(snap.get('loss_n', 0) or 0)
        avg_pnl = float(snap.get('avg_pnl_net', 0.0) or 0.0)
        reject_rate = float(snap.get('reject_rate', 0.0) or 0.0)
        avg_slip = float(snap.get('avg_slippage_pct', 0.0) or 0.0)
        orders_n = int(snap.get('orders_n', 0) or 0)

        # Current runtime
        try:
            cur_min_conf = float(getattr(self.strategy_engine, 'min_confidence', 0.35) or 0.35)
        except Exception:
            cur_min_conf = 0.35
        try:
            cur_rr = float(getattr(self.strategy_engine, 'rr_min', 1.35) or 1.35)
        except Exception:
            cur_rr = 1.35

        # If ops are degraded, do not loosen gates
        ops_ok = True
        if orders_n >= self.min_orders_for_ops:
            if reject_rate > 0.30:
                ops_ok = False
            if avg_slip > 0.0050:
                ops_ok = False

        new_min_conf = cur_min_conf
        new_rr = cur_rr

        # Quality signal
        winrate = (win_n / float(max(1, trades_n))) if trades_n > 0 else 0.0

        if trades_n >= 12:
            # Enough samples to judge quality
            if (avg_pnl < 0.0) or (winrate < 0.45) or (loss_n >= win_n + 2):
                # tighten
                new_min_conf = min(0.55, new_min_conf + 0.01)
                new_rr = min(1.85, new_rr + 0.04)
            elif (avg_pnl > 0.0) and (winrate > 0.55) and ops_ok:
                # loosen slightly to increase frequency (still bounded)
                new_min_conf = max(0.30, new_min_conf - 0.01)
                new_rr = max(1.10, new_rr - 0.03)
        else:
            # Not enough trades: if too quiet but ops quality is OK, loosen a bit.
            if trades_n <= 1 and orders_n >= self.min_orders_for_ops and ops_ok:
                new_min_conf = max(0.30, new_min_conf - 0.01)

        # Apply & persist if changed
        new_min_conf = max(0.30, min(0.55, float(new_min_conf)))
        new_rr = max(1.10, min(1.85, float(new_rr)))

        changed = (abs(new_min_conf - cur_min_conf) > 1e-12) or (abs(new_rr - cur_rr) > 1e-12)
        if changed:
            self.meta_tune = dict(self.meta_tune or {})
            self.meta_tune['min_confidence'] = float(new_min_conf)
            self.meta_tune['rr_min'] = float(new_rr)
            self._apply_meta_tune()
            self._save()

    def _ucb_score(self, name: str, total_n: int) -> float:
        s = self.stats.get(name, {'n': 0, 'sum': 0.0})
        n = int(s.get('n', 0) or 0)
        mean = (float(s.get('sum', 0.0) or 0.0) / n) if n > 0 else 0.0
        if n == 0:
            return 1e9
        import math
        return mean + 0.45 * math.sqrt(math.log(max(1, total_n)) / n)

    def _get_profile(self, name: str) -> Optional[dict]:
        for p in self.profiles:
            if p.get('name') == name:
                return p
        return None

    def apply(self, profile_name: str) -> Optional[dict]:
        p = self._get_profile(profile_name)
        if not p:
            return None
        self.prev_profile = self.active_profile
        self.active_profile = p.get('name')

        # Strategy runtime params
        try:
            sp = p.get('strategy', {}) or {}
            self.strategy_engine.set_runtime_params(**sp)
        except Exception:
            pass

        # Risk runtime scale + optional min_conf overrides
        try:
            rs = float((p.get('risk', {}) or {}).get('risk_scale', 1.0) or 1.0)
            if self.risk_guard is not None and hasattr(self.risk_guard, 'set_runtime_risk_scale'):
                self.risk_guard.set_runtime_risk_scale(rs)
        except Exception:
            pass

        # Microstructure guard runtime limits
        try:
            ms = p.get('micro', {}) or {}
            if self.micro is not None and hasattr(self.micro, 'update_config') and ms:
                self.micro.update_config({'microstructure': ms})
        except Exception:
            pass

        # Governor runtime limits
        try:
            gv = p.get('governor', {}) or {}
            if self.governor is not None and hasattr(self.governor, 'update_config') and gv:
                self.governor.update_config({'governor': gv})
        except Exception:
            pass

        self._save()
        return p

    def choose_and_apply(self) -> dict:
        total_n = sum(int(self.stats.get(p['name'], {}).get('n', 0) or 0) for p in self.profiles) + 1
        best = None
        best_score = -1e18
        for p in self.profiles:
            sc = self._ucb_score(p['name'], total_n)
            if sc > best_score:
                best_score = sc
                best = p
        if not best:
            best = self.profiles[0]
        self.apply(best['name'])
        return best

    def _reward(self, mode: str) -> float:
        # Reward = realized pnl delta - ops penalties (slippage/reject/latency) - drawdown penalty
        try:
            st = self.state_mgr.get(mode)
            realized = float(st.get('realized_pnl', 0.0) or 0.0)
            peak = float(st.get('peak_equity', st.get('equity', 0.0)) or 0.0)
            eq = float(st.get('equity', 0.0) or 0.0)
            dd = max(0.0, peak - eq)
        except Exception:
            return 0.0

        meta = self.stats.get('_meta', {})
        last_key = f'last_realized_{mode}'
        last_realized = float(meta.get(last_key, 0.0) or 0.0)
        delta = realized - last_realized

        ops_pen = 0.0
        try:
            snap = self.metrics.snapshot() if self.metrics is not None else {}
            if snap.get('orders_n', 0) >= self.min_orders_for_ops:
                ops_pen += 120.0 * max(0.0, float(snap.get('avg_slippage_pct', 0.0) or 0.0) - 0.0040)
                ops_pen += 40.0 * max(0.0, float(snap.get('reject_rate', 0.0) or 0.0) - 0.25)
                ops_pen += 0.02 * max(0.0, float(snap.get('p95_latency_ms', 0.0) or 0.0) - 3000.0)
        except Exception:
            pass

        reward = float(delta) - 0.18 * float(dd) - float(ops_pen)
        meta[last_key] = realized
        self.stats['_meta'] = meta
        return float(reward)

    def observe_and_update(self, mode: str):
        if not self.active_profile:
            return
        r = self._reward(mode)
        s = self.stats.get(self.active_profile, {'n': 0, 'sum': 0.0})
        s['n'] = int(s.get('n', 0) or 0) + 1
        s['sum'] = float(s.get('sum', 0.0) or 0.0) + float(r)
        self.stats[self.active_profile] = s
        self._save()

    def _should_rollback(self) -> bool:
        try:
            snap = self.metrics.snapshot() if self.metrics is not None else {}
            # If operations degrade badly, rollback to previous profile
            if snap.get('orders_n', 0) >= self.rollback_min_trades:
                if float(snap.get('reject_rate', 0.0) or 0.0) > self.rollback_max_reject:
                    return True
                if float(snap.get('avg_slippage_pct', 0.0) or 0.0) > self.rollback_max_slip:
                    return True
                if float(snap.get('avg_pnl_net', 0.0) or 0.0) < self.rollback_bad_expectancy:
                    return True
        except Exception:
            return False
        return False

    def maybe_step(self):
        if not self.enabled:
            return
        now = datetime.now()
        if self.last_ts:
            try:
                last = datetime.fromisoformat(self.last_ts)
                if (now - last).total_seconds() < float(self.rotate_minutes) * 60.0:
                    # still check rollback quickly
                    if self.prev_profile and self._should_rollback():
                        self.apply(self.prev_profile)
                    return
            except Exception:
                pass

        mode = self.learn_mode if self.learn_mode in ('paper', 'live', 'shadow', 'demo') else 'paper'
        try:
            self.observe_and_update(mode)
        except Exception:
            pass

        # rollback if current profile is harming execution quality
        if self.prev_profile and self._should_rollback():
            self.apply(self.prev_profile)
            self.last_ts = now.isoformat()
            self._save()
            return

        try:
            self.choose_and_apply()
        except Exception:
            pass

        # Meta-tune gates (min_confidence / rr_min) based on recent performance.
        # This keeps the system adaptive across regimes without new dependencies.
        try:
            self._meta_tune_step()
        except Exception:
            pass
        self.last_ts = now.isoformat()
        self._save()


try:
    # Core modules (bắt buộc)
    from part1_security_marketdata_fixed import (
        SecurityManager,
        EnhancedMarketDataEngine,
        OrderFlowAnalyzer,
        OrderBookSnapshot
    )
    from part2_risk_management_fixed import (
        AdvancedRiskGuard,
        RiskMode,
        RiskState
    )
    from part3_smart_execution_fixed import (
        SmartExecutionEngine,
        ExecutionStrategy,
        Order,
        Position,
        OrderType
    )
    from part4_ml_brain_fixed import AdvancedTradingBrain
    from part5_strategy_engine_fixed import (
        CompleteStrategyEngine,
        RegimeType,
        TradingSignal
    )
    from part6_telegram_dashboard_fixed import TelegramReporter
    from error_handling import handle_errors, ErrorHandler, TradingSystemError
except ImportError as e:
    print(f"⚠️ Critical import error: {e}")
    print("Đảm bảo tất cả các file part*.py đều nằm cùng thư mục!")
    sys.exit(1)

# Telegram commands (tùy chọn).
# Chuẩn hỗ trợ: python-telegram-bot v20+ (đúng với piplist của hệ thống).
# Nếu không cài telegram, hệ thống vẫn chạy và chỉ gửi báo cáo bằng TelegramReporter (requests).
try:
    from telegram.ext import ApplicationBuilder, CommandHandler
    from telegram import Update
except Exception:
    ApplicationBuilder = None
    CommandHandler = None
    Update = None

from binance.client import Client
from binance.exceptions import BinanceAPIException

class TradingSystemV2:
    """Main God-Tier Trading System V2.0 - Fixed & Completed Version"""

    def _persist_modes(self):
        try:
            if hasattr(self, 'brain') and hasattr(self.brain, 'set_kv'):
                self.brain.set_kv('mode_state', json.dumps(self.modes))
        except Exception:
            pass

    def _set_mode(self, name: str, value: bool):
        if name in self.modes:
            self.modes[name] = bool(value)
            self._persist_modes()

    def _modes_summary_line(self) -> str:
        return (
            f"demo {'ON' if self.modes.get('demo') else 'OFF'} | "
            f"shadow {'ON' if self.modes.get('shadow') else 'OFF'} | "
            f"paper {'ON' if self.modes.get('paper') else 'OFF'} | "
            f"live {'ON' if self.modes.get('live') else 'OFF'}"
        )
    def _set_system_paused(self, paused: bool, reason: str = ''):
        """Persist pause flag across restarts (applies to all local modes).

        This is intentionally conservative: if the system is paused manually or
        auto-paused due to degraded execution quality, it stays paused until
        user explicitly resumes.
        """
        try:
            p = bool(paused)
        except Exception:
            p = False
        r = str(reason or '')[:200]
        self.running = (not p)
        try:
            if getattr(self, 'state', None) is None:
                return
            for m in ['demo','shadow','paper','live']:
                try:
                    st = self.state.get(m)
                    st['system_paused'] = p
                    st['system_pause_reason'] = r
                    self.state.set(m, system_paused=p, system_pause_reason=r)
                except Exception:
                    continue
        except Exception:
            pass





    # ===================== ACCOUNT STATE (LIVE-LIKE) =====================

    def _init_accounts(self):
        """Khởi tạo & load account state cho từng mode.

        - Demo/Shadow/Paper: persist local JSON → không reset khi restart.
        - Live: sẽ sync từ Binance futures (nếu có quyền), fallback local.
        """
        base = os.path.dirname(os.path.abspath(__file__))
        init_eq = float(self.config.get('initial_equity', 1000.0) or 1000.0)
        self.state = AccountStateManager(base_dir=base, initial_equity=init_eq)
        self.state.load_all(['demo', 'shadow', 'paper', 'live'])

    def _boot_reconcile_restore(self):
        """Boot-time reconciliation & restore (exchange is truth).

        CIO-grade behavior:
        - On startup, reconcile open positions from Binance futures.
        - Import any open positions into self.execution.positions to allow exits to work.
        - If live is enabled but we cannot reconcile -> trip kill-switch (safe default).
        """
        try:
            if bool(getattr(self, 'is_public_mode', False)):
                return
            if not bool(self.modes.get('live', False)):
                return
            # Even if live_control is off, we still reconcile to display truth.
            positions = self.client.futures_position_information()
            imported = 0
            for p in positions:
                try:
                    sym = str(p.get('symbol') or '')
                    if sym not in (self.symbols or []):
                        continue
                    amt = float(p.get('positionAmt', 0.0) or 0.0)
                    if amt == 0.0:
                        continue
                    side = 'LONG' if amt > 0 else 'SHORT'
                    qty = abs(amt)
                    ep = float(p.get('entryPrice', 0.0) or 0.0)
                    if ep <= 0:
                        # fall back to last price if entryPrice missing
                        try:
                            ep = float(self.market_data.get_current_price(sym) or 0.0)
                        except Exception:
                            ep = 0.0
                    pos = Position(
                        symbol=sym,
                        side=side,
                        entry_price=ep,
                        quantity=qty,
                        stop_loss=0.0,
                        take_profit=0.0,
                        entry_time=datetime.now(),
                    )
                    try:
                        pos.mode = 'live'
                    except Exception:
                        pass
                    self.execution.positions[sym] = pos
                    imported += 1
                except Exception:
                    continue
            if imported > 0:
                logging.warning(f"[BOOT] Imported {imported} live positions from exchange")
        except Exception as e:
            logging.error(f"[BOOT] reconcile/restore failed: {e}")
            try:
                self._trip_live_killswitch(f'boot_reconcile_failed: {e}')
            except Exception:
                pass

    def _persist_account(self, mode: str):
        # v10: persisted automatically by AccountStateManager on updates.
        try:
            if hasattr(self, 'state'):
                self.state.get(mode)
        except Exception:
            pass

    def _apply_accounts_to_telegram(self):
        """Đồng bộ số dư hiển thị Telegram theo account state (không dùng mặc định 1000)."""
        tg = getattr(self, 'telegram', None)
        if not tg or not hasattr(self, 'state'):
            return
        try:
            for mode in ['demo','shadow','paper','live']:
                st = self.state.get(mode)
                tg.mode_balances[mode] = float(st.get('balance', tg.mode_balances.get(mode, 1000.0)) or 0.0)
        except Exception:
            logging.exception('Failed to apply account balances to Telegram')

    def _sync_live_account(self):
        """Sync LIVE equity/balance từ Binance futures.

        Nếu public mode hoặc API không hỗ trợ futures → giữ nguyên state local.
        """
        if not hasattr(self, 'state'):
            return
        try:
            # python-binance futures
            acct = self.client.futures_account()
            # totalWalletBalance, totalUnrealizedProfit
            wallet = float(acct.get('totalWalletBalance', 0.0) or 0.0)
            upnl = float(acct.get('totalUnrealizedProfit', 0.0) or 0.0)
            equity = wallet + upnl
            self.state.set('live', balance=wallet, unrealized_pnl=upnl, equity=equity)
            # event
            self.state.append_event('live', 'SYNC', {'wallet': wallet, 'unrealized_pnl': upnl, 'equity': equity})
        except Exception:
            # Không spam log nếu chưa có futures permission
            return

    def _primary_mode_for_sizing(self) -> str:
        # Ưu tiên sizing theo mode mạnh nhất đang bật
        for m in ['live','paper','demo','shadow']:
            if bool(self.modes.get(m, False)):
                return m
        return 'paper'

    def _fee_rate(self) -> float:
        # đồng bộ fee với telegram reporter / execution
        try:
            tg = getattr(self, 'telegram', None)
            if tg and hasattr(tg, 'fee_rate'):
                return float(getattr(tg, 'fee_rate'))
        except Exception:
            pass
        return 0.0004


    def _breakeven_buffer(self, entry: float, qty: float) -> float:
        """Breakeven buffer in price units.
        Includes round-trip fees, slippage cushion, and an optional minimum net profit (USDT)
        converted to price distance by qty.
        """
        try:
            cfg = self.config if isinstance(getattr(self, 'config', None), dict) else {}
            exit_cfg = self._get_exit_cfg()
            slip = float(cfg.get('slippage_pct', 0.0005) or 0.0005)
            fee_rt = float(self._fee_rate() or 0.0) * 2.0
            mult = float(exit_cfg.get('breakeven_buffer_mult', 1.0) or 1.0)
            base = float(entry) * (fee_rt + slip) * mult
            min_net = float(exit_cfg.get('min_close_profit_net_usdt', 0.0) or 0.0)
            extra = (min_net / float(qty)) if (min_net > 0.0 and qty and qty > 0.0) else 0.0
            return max(0.0, base + extra)
        except Exception:
            return float(entry) * float(self._fee_rate() or 0.0) * 2.0
    def _calc_futures_pnl_net(self, pos: Position, exit_price: float) -> tuple:
        """Tính PnL (USDT-M Futures) + fee ước tính.

        ✅ QUAN TRỌNG: PnL danh nghĩa KHÔNG nhân leverage.
        - LONG : (exit - entry) * qty
        - SHORT: (entry - exit) * qty

        Leverage chỉ ảnh hưởng margin/liquidation, không làm PnL danh nghĩa lớn hơn.

        Returns: (pnl_gross, fee_est, pnl_net)
        """
        qty = float(getattr(pos, 'quantity', 0.0) or 0.0)
        entry = float(getattr(pos, 'entry_price', 0.0) or 0.0)
        exit_price = float(exit_price or 0.0)
        if qty <= 0 or entry <= 0 or exit_price <= 0:
            return 0.0, 0.0, 0.0

        side = str(getattr(pos, 'side', 'LONG') or 'LONG').upper()
        if side == 'SHORT':
            pnl_gross = (entry - exit_price) * qty
        else:
            pnl_gross = (exit_price - entry) * qty

        # Fee futures tính theo notional mỗi fill; round-trip = (entry + exit) * qty * fee_rate
        fee_rate = float(self._fee_rate() or 0.0)
        fee = (entry + exit_price) * qty * fee_rate
        pnl_net = pnl_gross - fee
        return pnl_gross, fee, pnl_net

    def __init__(self, config_path: str = 'config.json'):
        print("🚀 Khởi tạo God-Tier Trading System V2.0 (Fixed & Full Logic)...")
        
        try:
            with open(config_path, 'r', encoding='utf-8-sig') as f:
                self.config = json.load(f)
            self.config_path = config_path
            self._config_reloader = ConfigReloader(config_path)
            self.circuit = CircuitBreaker(self.config)
            self.micro = MicrostructureGuard(self.config)
            self.governor = RiskGovernor(self.config)
            self.metrics = MetricsEngine(self.config)
            self._cooldown = {}
            self._last_close = {}
        except Exception as e:
            print(f"❌ Lỗi load config.json: {e}")
            sys.exit(1)
        
        self.modes = self.config.get('modes', {
            'demo': True,
            'shadow': False,
            'paper': False,
            'live': False
        })
        
        
        # Load/persist balances so PnL & balance don't reset on restart
        self._init_accounts()
        # Restore persisted pause flag (fund-grade safety)
        try:
            stp = self.state.get('paper')
            if bool(stp.get('system_paused', False)):
                self.running = False
            else:
                self.running = True
        except Exception:
            self.running = True

        self.symbols = self.config.get('symbols', ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT'])
        if not self.symbols:
            print("❌ Danh sách symbols trống!")
            sys.exit(1)
        
        self.security = SecurityManager()
        print("\n🔑 Nhập passphrase cho .env.encrypted (Enter để public mode):")
        passphrase = getpass.getpass("").strip()
        
        api_key = None
        api_secret = None
        is_public_mode = False
        
        if passphrase:
            try:
                # v10/secure: decrypt toàn bộ secrets từ .env.encrypted và export ra env
                _secrets = self.security.decrypt_env(passphrase)
                api_key = os.getenv('BINANCE_API_KEY')
                api_secret = os.getenv('BINANCE_API_SECRET')
                if api_key and api_secret:
                    print("✓ Giải mã thành công! Sử dụng Binance API keys thật.")
                else:
                    raise ValueError("Decrypt OK nhưng thiếu BINANCE_API_KEY/BINANCE_API_SECRET")
            except Exception as e:
                print(f"✗ Giải mã thất bại: {e}")
                print("→ Chuyển sang PUBLIC MODE")
                is_public_mode = True
        else:
            print("→ Không nhập passphrase → PUBLIC MODE")
            is_public_mode = True
        
        if is_public_mode:
            self.client = Client(None, None)
            logging.warning("PUBLIC MODE: Chỉ đọc dữ liệu")
        else:
            self.client = Client(api_key, api_secret)

        # Persist mode flag for later safety checks
        self.is_public_mode = bool(is_public_mode)
        
        try:
            self.market_data = EnhancedMarketDataEngine(self.security)
            self.market_data.client = self.client
            
            test = self.client.get_server_time()
            print("✓ Test kết nối Binance OK")
        except Exception as e:
            print(f"❌ Lỗi kết nối Binance: {e}")
            sys.exit(1)

        self.risk_guard = AdvancedRiskGuard(self.config)
        # FIX: SmartExecutionEngine expects (client, market_data). Passing in reverse order
        # will break paper/live execution (and can make system appear "không có lệnh").
        # Live execution defaults (CROSS + per-symbol leverage)
        try:
            lev = int(self.config.get('default_leverage', 10) or 10)
        except Exception:
            lev = 10
        self.execution = SmartExecutionEngine(self.client, self.market_data, default_leverage=lev, force_cross_margin=True)
        self.brain = AdvancedTradingBrain(config=self.config)
        self.strategy = CompleteStrategyEngine(self.market_data, self.brain, self.config)

        # CIO-grade: boot-time reconcile/restore so live exits work after restart
        try:
            self._boot_reconcile_restore()
        except Exception:
            pass

        # CIO-grade: AdaptiveEngine (guardrailed, rollback, ops-aware)
        # This supersedes the older BanditTuner.
        self.tuner = None
        try:
            ae_cfg = (self.config.get('adaptive_engine', {}) or {}) if isinstance(self.config, dict) else {}
            if bool(ae_cfg.get('enabled', True)):
                base = os.path.dirname(os.path.abspath(__file__))
                self.adaptive = AdaptiveEngine(
                    base_dir=base,
                    config=self.config,
                    state_mgr=self.state,
                    strategy_engine=self.strategy,
                    risk_guard=self.risk_guard,
                    micro_guard=getattr(self, 'micro', None),
                    governor=getattr(self, 'governor', None),
                    metrics=getattr(self, 'metrics', None),
                )
                # apply last or best profile at startup
                if not self.adaptive.active_profile:
                    self.adaptive.choose_and_apply()
                else:
                    self.adaptive.apply(self.adaptive.active_profile)
            else:
                self.adaptive = None
        except Exception:
            self.adaptive = None

        # Keep persisted pause state (do not force-run on restart)
        self.running = bool(getattr(self, "running", True))

        # Throttle logs for "no trade" reasons so user can debug why bot is quiet
        # without spamming console/logfile.
        self._no_trade_log_ts = {}

        self._tg_report_stop = False
        self._tg_report_thread = threading.Thread(target=self._telegram_report_loop, daemon=True, name="TgReportLoop")
        self._tg_report_thread.start()
        self.scan_interval_seconds = int((self.config.get('scan_interval_seconds', 25) if isinstance(self.config, dict) else 25) or 25)
        self.equity_lock = threading.Lock()
        self.positions = {}
        # v10: separate per-mode positions (prevents mixing demo/shadow state)
        self.demo_positions = {}
        self.shadow_positions = {}

        # Performance safety: auto de-risk / auto-pause when system is losing
        self._perf_pause_until = 0.0
        self._perf_forced_safe_mode = False
        self._perf_last_tune_ts = 0.0
        self._pyramiding_forced_off = False

        self.last_btc_signal_time = None  # Để check BTC lead

        # Telegram
        tg_config = self.config.get('telegram', {})
        self.telegram = None
        self.telegram_updater = None

        if tg_config.get('enabled', False):
            try:
                # Secrets-first: lấy token/chat_id từ ENV (không lưu trong config)
                tg_env = (tg_config.get('env') or {}) if isinstance(tg_config, dict) else {}
                token_env_key = str(tg_env.get('bot_token') or 'TELEGRAM_BOT_TOKEN')
                chat_env_key = str(tg_env.get('chat_id') or 'TELEGRAM_CHAT_ID')
                tg_token = os.getenv(token_env_key) or tg_config.get('bot_token')
                tg_chat_id = os.getenv(chat_env_key) or tg_config.get('chat_id')

                if not tg_token or not tg_chat_id:
                    raise ValueError(
                        "Thiếu Telegram secrets. Hãy set ENV TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID "
                        "hoặc cấu hình telegram.env trong config để trỏ tới tên biến môi trường."
                    )

                self.telegram = TelegramReporter(
                    bot_token=tg_token,
                    chat_id=tg_chat_id,
                    market_data=self.market_data
                )
                print("√ Telegram reporter initialized")

                # Ping ngay để biết chắc token/chat_id OK (nếu sai sẽ log rõ)
                try:
                    self.telegram.ping()
                except Exception:
                    pass

                # Sync balances to Telegram so dashboard uses real persisted balances
                try:
                    self._sync_live_account()
                except Exception:
                    pass
                self._apply_accounts_to_telegram()

                # Sync Telegram displayed balances from persisted account states
                self._apply_accounts_to_telegram()

                # Telegram hai chiều (commands): dùng Simple Polling (getUpdates) mặc định.
                # Lý do: python-telegram-bot v20 chạy asyncio có thể "im lặng" nếu event loop/thread lỗi,
                # khiến user thấy bot không phản hồi lệnh.
                # Simple polling hoạt động ổn định, không phụ thuộc version thư viện.
                commands_backend = str(tg_config.get('commands_backend') or 'simple').lower()

                if commands_backend == 'simple':
                    try:
                        self._start_simple_tg_commands(tg_config)
                        print("Telegram commands đã kích hoạt (simple polling)")
                    except Exception as e_cmd:
                        logging.error(f"Không khởi động được Telegram commands (simple): {e_cmd}")
                        print("⚠️ Telegram hai chiều không hoạt động")
                    # Không chạy python-telegram-bot backend nữa để tránh double polling.
                    self._tg_app = None
                    self.telegram_updater = None
                    self._tg_app_thread = None
                    # Short-circuit để bỏ qua phần python-telegram-bot bên dưới
                    raise StopIteration("simple telegram commands enabled")
                else:
                    # Nếu user thật sự muốn dùng python-telegram-bot, set telegram.commands_backend='ptb'
                    # (giữ nguyên code tương thích phía dưới)

                    # Telegram hai chiều: ưu tiên tương thích python-telegram-bot v20+ (pip list của bạn: 20.0)
                    # Nếu không khả dụng thì mới fallback Updater (v13 trở xuống) hoặc simple polling.

                    self._tg_use_context = True
                    self._tg_app = None
                    self._tg_app_thread = None

                # Nếu user thật sự muốn dùng python-telegram-bot, set telegram.commands_backend='ptb'
                # (giữ nguyên code tương thích phía dưới)

                # Telegram hai chiều: ưu tiên tương thích python-telegram-bot v20+ (pip list của bạn: 20.0)
                # Nếu không khả dụng thì mới fallback Updater (v13 trở xuống) hoặc simple polling.

                self._tg_use_context = True
                self._tg_app = None
                self._tg_app_thread = None

                if ApplicationBuilder is not None and CommandHandler is not None:
                    # PTB v20+: ApplicationBuilder
                    try:
                        app = ApplicationBuilder().token(tg_token).build()
                        app.add_handler(CommandHandler('status', self._tg_status_ctx))
                        app.add_handler(CommandHandler('pause', self._tg_pause_ctx))
                        app.add_handler(CommandHandler('resume', self._tg_resume_ctx))
                        app.add_handler(CommandHandler('close_all', self._tg_close_all_ctx))
                        app.add_handler(CommandHandler('toggle_demo', self._tg_toggle_demo_ctx))
                        app.add_handler(CommandHandler('toggle_shadow', self._tg_toggle_shadow_ctx))
                        app.add_handler(CommandHandler('toggle_paper', self._tg_toggle_paper_ctx))
                        app.add_handler(CommandHandler('toggle_live', self._tg_toggle_live_ctx))
                        app.add_handler(CommandHandler('report', self._tg_report_ctx))

                        self._tg_app = app

                        def _run_app_polling():
                            try:
                                # PTB v20 chạy trên asyncio. Khi chạy trong thread riêng,
                                # Python 3.8 sẽ không tự tạo event loop -> RuntimeError.
                                # Vì vậy tạo loop riêng cho thread này.
                                import asyncio
                                try:
                                    loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(loop)
                                except Exception:
                                    # Nếu không tạo được loop thì vẫn thử chạy bình thường
                                    loop = None

                                # drop_pending_updates=True để không xử lý backlog cũ
                                self._tg_app.run_polling(drop_pending_updates=True)

                                # Dọn loop nếu còn mở (phòng trường hợp run_polling không đóng)
                                try:
                                    if loop is not None and not loop.is_closed():
                                        loop.close()
                                except Exception:
                                    pass
                            except Exception:
                                logging.exception("Telegram Application polling crashed")

                        self._tg_app_thread = threading.Thread(
                            target=_run_app_polling,
                            daemon=True,
                            name="TgAppPolling"
                        )
                        self._tg_app_thread.start()
                        print("Telegram commands đã kích hoạt (PTB v20 Application)")
                    except Exception as _e_app:
                        raise _e_app
                else:
                    # PTB v20+ is required (piplist pinned). Legacy Updater is intentionally disabled
                    # to avoid silent incompatibilities across versions.
                    raise ImportError('python-telegram-bot v20+ required; legacy Updater disabled')

            except Exception as e:
                # StopIteration dùng để short-circuit sau khi bật simple commands thành công
                if isinstance(e, StopIteration):
                    pass
                else:
                    logging.error(f"Không khởi động được Telegram commands: {e}")
                    # Fallback: nếu python-telegram-bot lỗi init -> bật simple polling
                    try:
                        self._start_simple_tg_commands(tg_config)
                        print("Telegram commands đã kích hoạt (simple polling)")
                    except Exception as e2:
                        logging.error(f"Không khởi động được Telegram commands (simple): {e2}")
                        print("⚠️ Telegram hai chiều không hoạt động")

        print("\n" + "="*70)
        print("HỆ THỐNG KHỞI ĐỘNG THÀNH CÔNG")
        if self.telegram_updater or getattr(self, '_tg_app', None) is not None:
            print("Telegram commands đã kích hoạt")
        print("="*70 + "\n")

    # ===================== TELEGRAM PERIODIC REPORT LOOP =====================


    def _post_trade_performance_tune(self):
        """Auto-fix: when trades are all losing, immediately tighten filters and cut risk.

        This does NOT guarantee profit, but it prevents the common failure mode:
        loose signals + big leverage => rapid equity bleed.
        """
        try:
            now = time.time()
        except Exception:
            return

        # Throttle
        try:
            if (now - float(getattr(self, '_perf_last_tune_ts', 0.0) or 0.0)) < 8.0:
                return
            self._perf_last_tune_ts = float(now)
        except Exception:
            pass

        # Recent realized PnL window
        try:
            hist = list(getattr(self.risk_guard, 'trade_history', []) or [])
        except Exception:
            hist = []
        if len(hist) < 6:
            return

        win = 0
        loss = 0
        pnl_sum = 0.0
        pnl_pos = 0.0
        pnl_neg = 0.0
        window = hist[-25:]
        for t in window:
            try:
                p = float(t.get('pnl', 0.0) or 0.0)
            except Exception:
                p = 0.0
            pnl_sum += p
            if p > 0:
                win += 1
                pnl_pos += p
            elif p < 0:
                loss += 1
                pnl_neg += abs(p)

        n = len(window)
        winrate = (win / float(n)) if n else 0.0
        avg_pnl = (pnl_sum / float(n)) if n else 0.0
        pf = (pnl_pos / pnl_neg) if pnl_neg > 0 else (float('inf') if pnl_pos > 0 else 0.0)

        # If the system is bleeding, switch to SAFE mode: fewer trades, higher quality
        bleeding = (n >= 10 and (winrate < 0.40 or pf < 0.80) and avg_pnl < 0)
        severe = (n >= 12 and winrate < 0.35 and avg_pnl < 0)

        if bleeding:
            try:
                self._pyramiding_forced_off = True
            except Exception:
                pass
            # De-risk fast
            try:
                self.risk_guard.set_runtime_risk_scale(0.30 if severe else 0.40)
            except Exception:
                pass
            # Tighten signal quality gates (trend-only, no range-revert)
            try:
                self.strategy.set_runtime_params(
                    min_confidence=0.58 if severe else 0.54,
                    allow_range_mode=False,
                    cost_gate_reward_mult=1.05,
                    cost_gate_risk_mult=0.70,
                )
            except Exception:
                pass
            # Short cooldown to stop revenge-trading
            try:
                self._perf_pause_until = max(float(getattr(self, '_perf_pause_until', 0.0) or 0.0), now + (120.0 if severe else 60.0))
                self._perf_forced_safe_mode = True
            except Exception:
                pass
        else:
            # Recover gradually
            try:
                if bool(getattr(self, '_perf_forced_safe_mode', False)) and n >= 12 and winrate >= 0.52 and avg_pnl >= 0:
                    self.risk_guard.set_runtime_risk_scale(0.75)
                    self.strategy.set_runtime_params(min_confidence=0.26, allow_range_mode=False)
                    self._perf_forced_safe_mode = False
                    self._pyramiding_forced_off = False
            except Exception:
                pass

    def _telegram_report_loop(self):
        """Gửi báo cáo định kỳ để biết bot còn sống + tóm tắt trạng thái.

        - Không crash nếu Telegram tắt.
        - Chạy nhẹ nhàng (sleep), tránh spam.
        - Nếu Telegram bật: gửi report mỗi 15 phút, và ping nhanh khi mới khởi động.
        """

        # Đợi hệ thống init xong Telegram (thread start sớm hơn phần init Telegram)
        time.sleep(3)
        last_sent = 0.0
        interval = 15 * 60  # 15 phút

        while not getattr(self, '_tg_report_stop', False):
            try:
                tg = getattr(self, 'telegram', None)
                if tg is not None:
                    now = time.time()
                    if last_sent == 0.0 or (now - last_sent) >= interval:
                        metrics = None
                        try:
                            metrics = self.risk_guard.get_metrics()
                        except Exception:
                            metrics = {}

                        # Gửi status ngắn gọn
                        try:
                            # keep balances in sync
                            try:
                                self._sync_live_account()
                            except Exception:
                                pass
                            self._apply_accounts_to_telegram()
                            pos_by_mode = self._positions_by_mode()
                            tg.report_dashboard(metrics, self.modes, pos_by_mode)
                        except Exception:
                            # Không để thread chết vì lỗi Telegram
                            logging.exception("Telegram periodic report failed")

                        last_sent = now

            except Exception:
                logging.exception("_telegram_report_loop crashed")

            # Sleep nhỏ để có thể stop nhanh
            for _ in range(30):
                if getattr(self, '_tg_report_stop', False):
                    break
                time.sleep(1)

    # ===================== TELEGRAM COMMAND HANDLERS =====================

    def _positions_by_mode(self):
        """Snapshot positions theo từng mode để dùng cho Telegram dashboard."""
        demo_pos = dict(getattr(self, 'demo_positions', {}) or {})
        shadow_pos = dict(getattr(self, 'shadow_positions', {}) or {})

        try:
            exec_pos = dict(getattr(getattr(self, 'execution', None), 'positions', {}) or {})
        except Exception:
            exec_pos = {}

        paper_pos = {}
        live_pos = {}
        for sym, pos in (exec_pos or {}).items():
            m = str(getattr(pos, 'mode', 'paper') or 'paper')
            if m == 'live':
                live_pos[sym] = pos
            else:
                paper_pos[sym] = pos

        return {
            'demo': demo_pos,
            'shadow': shadow_pos,
            'paper': paper_pos,
            'live': live_pos,
        }

    # ===================== TELEGRAM COMMANDS (COMPAT WRAPPERS) (COMPAT WRAPPERS) =====================

        def _tg_reply(self, update, text: str):
            """Reply helper compatible across PTB versions."""
            try:
                if update is not None and getattr(update, 'message', None) is not None:
                    r = update.message.reply_text(text)
                    # PTB v13 returns None; PTB v20+ may return coroutine.
                    try:
                        import inspect
                        import asyncio
                        if inspect.isawaitable(r):
                            try:
                                loop = asyncio.get_running_loop()
                                loop.create_task(r)
                            except RuntimeError:
                                loop = asyncio.new_event_loop()
                                try:
                                    asyncio.set_event_loop(loop)
                                    loop.run_until_complete(r)
                                finally:
                                    try:
                                        loop.close()
                                    except Exception:
                                        pass
                    except Exception:
                        pass
            except Exception:
                pass

# ===================== TELEGRAM SIMPLE COMMANDS (NO python-telegram-bot) =====================

    def _start_simple_tg_commands(self, tg_config: Dict):
        """Fallback cho telegram 2 chiều khi python-telegram-bot quá cũ / lỗi init.

        Dùng polling trực tiếp Telegram Bot API (getUpdates) nên:
        - Không cần thay pip list
        - Không phụ thuộc Updater/Dispatcher
        """
        tg_env = (tg_config.get('env') or {}) if isinstance(tg_config, dict) else {}
        token_env_key = str(tg_env.get('bot_token') or 'TELEGRAM_BOT_TOKEN')
        chat_env_key = str(tg_env.get('chat_id') or 'TELEGRAM_CHAT_ID')
        token = os.getenv(token_env_key) or tg_config.get('bot_token')
        chat_id = str(os.getenv(chat_env_key) or tg_config.get('chat_id', '')).strip()
        if not token or not chat_id:
            raise ValueError('Thiếu bot_token hoặc chat_id cho Telegram simple commands')

        self._simple_tg_token = token
        self._simple_tg_chat_id = chat_id
        self._simple_tg_offset = 0
        self._simple_tg_stop = False

        # Network tuning (some ISPs block/slow api.telegram.org)
        try:
            self._tg_connect_timeout = float(os.getenv('TELEGRAM_CONNECT_TIMEOUT', '10') or 7)
        except Exception:
            self._tg_connect_timeout = 7.0
        try:
            self._tg_read_timeout = float(os.getenv('TELEGRAM_READ_TIMEOUT', '40') or 25)
        except Exception:
            self._tg_read_timeout = 25.0
        self._tg_proxy = str(os.getenv('TELEGRAM_PROXY') or '').strip()

        # Clamp telegram timeouts
        try:
            self._tg_connect_timeout = max(5.0, float(self._tg_connect_timeout))
        except Exception:
            self._tg_connect_timeout = 10.0
        try:
            self._tg_read_timeout = max(20.0, float(self._tg_read_timeout))
        except Exception:
            self._tg_read_timeout = 40.0
        self._tg_last_log_ts = 0.0

        # Build a reusable opener so we can support proxies without extra dependencies
        try:
            handlers = []
            if self._tg_proxy:
                handlers.append(urllib.request.ProxyHandler({'http': self._tg_proxy, 'https': self._tg_proxy}))
            self._tg_opener = urllib.request.build_opener(*handlers)
        except Exception:
            self._tg_opener = None

        t = threading.Thread(target=self._simple_tg_poll_loop, daemon=True, name='TgSimplePoll')
        t.start()
        self._simple_tg_thread = t

        # Ping nhỏ để xác nhận đã bật
        try:
            self._simple_tg_send("✅ Telegram commands (simple polling) đã bật. Gõ /status hoặc /report")
        except Exception:
            pass

    def _simple_tg_api(self, method: str, params: Dict) -> Dict:
        """Simple Telegram API helper (urllib) with circuit-breaker.

        IMPORTANT:
        - Telegram/network issues must NOT impact trading loop.
        - When errors repeat, temporarily disable calls to avoid blocking scan loop.
        """
        try:
            now_ts = time.time()
            disabled_until = float(getattr(self, '_tg_disabled_until', 0.0) or 0.0)
            if disabled_until and now_ts < disabled_until:
                return {'ok': False, 'skipped': True, 'reason': 'tg_disabled'}

            url = f"https://api.telegram.org/bot{self._simple_tg_token}/{method}"
            data = urllib.parse.urlencode(params).encode('utf-8')
            req = urllib.request.Request(url, data=data, method='POST')

            # Keep connect timeout short, read timeout a bit longer (getUpdates is long-poll)
            # urllib only supports a single timeout; we compromise with a safe upper bound.
            timeout = float(getattr(self, '_tg_read_timeout', 25.0) or 25.0)
            opener = getattr(self, '_tg_opener', None)
            if opener is None:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    raw = resp.read().decode('utf-8', errors='ignore')
            else:
                with opener.open(req, timeout=timeout) as resp:
                    raw = resp.read().decode('utf-8', errors='ignore')

            try:
                payload = json.loads(raw)
            except Exception:
                payload = {'ok': False, 'raw': raw}

            # reset fail counter on success
            if payload.get('ok'):
                try:
                    setattr(self, '_tg_fail_count', 0)
                except Exception:
                    pass
            return payload

        except Exception as e:
            # increment failure count; if repeated, disable for a while
            try:
                fc = int(getattr(self, '_tg_fail_count', 0) or 0) + 1
                setattr(self, '_tg_fail_count', fc)
                if fc >= 3:
                    # disable 3 minutes
                    setattr(self, '_tg_disabled_until', time.time() + 180)
            except Exception:
                pass
            # Avoid spamming logs when network blocks Telegram
            try:
                now_ts = time.time()
                last = float(getattr(self, '_tg_last_log_ts', 0.0) or 0.0)
                if now_ts - last > 60.0:
                    setattr(self, '_tg_last_log_ts', now_ts)
                    logging.error(f"Telegram API error ({method}): {e}")
            except Exception:
                pass
            return {'ok': False, 'error': str(e)}

    def _simple_tg_send(self, text: str):
        # Dùng HTML tương thích (giống TelegramReporter)
        return self._simple_tg_api('sendMessage', {
            'chat_id': self._simple_tg_chat_id,
            'text': text,
            'parse_mode': 'HTML'
        })

    def _simple_tg_poll_loop(self):
        # Long-polling nhẹ (timeout 25s) để phản hồi nhanh mà không spam request
        # Có backoff khi bị reset kết nối / lỗi mạng (WinError 10054, timeout, ...)
        backoff = 1.0
        while not getattr(self, '_simple_tg_stop', False):
            try:
                res = self._simple_tg_api('getUpdates', {
                    'timeout': 25,
                    'offset': self._simple_tg_offset,
                    'allowed_updates': 'message'
                })
                if not res.get('ok'):
                    time.sleep(min(10.0, backoff))
                    backoff = min(20.0, backoff * 1.7)
                    continue

                for upd in res.get('result', []) or []:
                    try:
                        self._simple_tg_offset = int(upd.get('update_id', 0)) + 1
                    except Exception:
                        pass

                    msg = upd.get('message') or {}
                    chat = msg.get('chat') or {}
                    chat_id = str(chat.get('id', ''))
                    if chat_id != str(self._simple_tg_chat_id):
                        continue

                    text = (msg.get('text') or '').strip()
                    if not text:
                        continue
                    self._simple_tg_handle_text(text)

                # success -> reset backoff
                backoff = 1.0

            except Exception:
                logging.exception('TgSimplePoll loop error')
                time.sleep(min(10.0, backoff))
                backoff = min(20.0, backoff * 1.7)

    def _simple_tg_handle_text(self, text: str):
        cmd = text.split()[0].lower()

        if cmd in ['/start', '/help']:
            self._simple_tg_send(
                "<b>📌 Lệnh Telegram</b>\n"
                "• /status — Dashboard 4 mode\n"
                "• /report — Dashboard (force)\n"
                "• /pause — Tạm dừng scan\n"
                "• /resume — Tiếp tục scan\n"
                "• /close_all — Đóng tất cả (paper/live)\n"
                "• /toggle_demo /toggle_shadow /toggle_paper /toggle_live\n"
            )
            return

        if cmd == '/status':
            try:
                if not self.telegram:
                    self._simple_tg_send('Telegram reporter chưa bật.')
                    return
                metrics = self.risk_guard.get_metrics()
                pos_by_mode = self._positions_by_mode()
                self.telegram.report_dashboard(metrics, self.modes, pos_by_mode, force_send=True)
                self._simple_tg_send('✅ Đã gửi dashboard trạng thái!')
            except Exception as e:
                self._simple_tg_send(f'❌ Lỗi /status: {e}')
            return

        if cmd == '/report':
            try:
                if not self.telegram:
                    self._simple_tg_send('Telegram reporter chưa bật.')
                    return

                metrics = self.risk_guard.get_metrics()
                # /report = hiệu suất (đọc trades_*.jsonl + events_*.jsonl), KHÔNG giống /status
                # report_detailed tự load file theo mode (trong root hoặc thư mục data/)
                self.telegram.report_detailed(
                    modes=self.modes,
                    positions=self.positions,
                    risk_metrics=metrics,
                    trades=[]
                )
                self._simple_tg_send('✅ Đã gửi báo cáo hiệu suất!')
            except Exception as e:
                self._simple_tg_send(f'❌ Lỗi /report: {e}')
            return

        if cmd == '/pause':
            self.running = False
            self._simple_tg_send('🛑 Hệ thống đã tạm dừng')
            return

        if cmd == '/resume':
            self.running = True
            self._simple_tg_send('▶️ Hệ thống tiếp tục')
            return

        if cmd == '/close_all':
            try:
                self._close_all_positions()
                self._simple_tg_send('✅ Đã đóng tất cả vị thế (nếu có).')
            except Exception as e:
                self._simple_tg_send(f'❌ Lỗi /close_all: {e}')
            return

        if cmd in ['/toggle_demo', '/toggle_shadow', '/toggle_paper', '/toggle_live']:
            m = cmd.replace('/toggle_', '')
            try:
                self.modes[m] = not bool(self.modes.get(m, False))
                self._simple_tg_send(f"✅ {m.upper()} = {'ON' if self.modes[m] else 'OFF'}")
            except Exception as e:
                self._simple_tg_send(f'❌ Lỗi {cmd}: {e}')
            return

        # Unknown
        self._simple_tg_send('Không hiểu lệnh. Gõ /help')

    # ---- Context-based callbacks: (update, context)
    def _tg_status_ctx(self, update, context):
        return self._tg_status_impl(update)

    def _tg_report_ctx(self, update, context):
        return self._tg_report_impl(update)

    def _tg_pause_ctx(self, update, context):
        return self._tg_pause_impl(update)

    def _tg_resume_ctx(self, update, context):
        return self._tg_resume_impl(update)

    def _tg_close_all_ctx(self, update, context):
        return self._tg_close_all_impl(update)

    def _tg_toggle_demo_ctx(self, update, context):
        return self._tg_toggle_demo_impl(update)

    def _tg_toggle_shadow_ctx(self, update, context):
        return self._tg_toggle_shadow_impl(update)

    def _tg_toggle_paper_ctx(self, update, context):
        return self._tg_toggle_paper_impl(update)

    def _tg_toggle_live_ctx(self, update, context):
        return self._tg_toggle_live_impl(update)

    # ---- Pre-context callbacks: (bot, update)
    def _tg_status_nctx(self, bot, update):
        return self._tg_status_impl(update)

    def _tg_report_nctx(self, bot, update):
        return self._tg_report_impl(update)

    def _tg_pause_nctx(self, bot, update):
        return self._tg_pause_impl(update)

    def _tg_resume_nctx(self, bot, update):
        return self._tg_resume_impl(update)

    def _tg_close_all_nctx(self, bot, update):
        return self._tg_close_all_impl(update)

    def _tg_toggle_demo_nctx(self, bot, update):
        return self._tg_toggle_demo_impl(update)

    def _tg_toggle_shadow_nctx(self, bot, update):
        return self._tg_toggle_shadow_impl(update)

    def _tg_toggle_paper_nctx(self, bot, update):
        return self._tg_toggle_paper_impl(update)

    def _tg_toggle_live_nctx(self, bot, update):
        return self._tg_toggle_live_impl(update)

    # ---- Implementations (update only)
    def _tg_status_impl(self, update):
        if not self.telegram:
            self._tg_reply(update, 'Telegram reporter chưa bật.')
            return
        try:
            self._sync_live_account()
        except Exception:
            pass
        self._apply_accounts_to_telegram()
        metrics = self.risk_guard.get_metrics()
        pos_by_mode = self._positions_by_mode()
        self.telegram.report_dashboard(metrics, self.modes, pos_by_mode, force_send=True)
        self._tg_reply(update, '✅ Đã gửi dashboard trạng thái!')

    def _tg_report_impl(self, update):
        if not self.telegram:
            self._tg_reply(update, 'Telegram reporter chưa bật.')
            return
        try:
            self._sync_live_account()
        except Exception:
            pass
        self._apply_accounts_to_telegram()
        metrics = self.risk_guard.get_metrics()
        pos_by_mode = self._positions_by_mode()
        self.telegram.report_dashboard(metrics, self.modes, pos_by_mode, force_send=True)
        self._tg_reply(update, '✅ Đã gửi dashboard chi tiết!')

    def _tg_pause_impl(self, update):
        self._set_system_paused(True, reason='MANUAL_PAUSE')
        self._tg_reply(update, '🛑 Hệ thống đã tạm dừng')

    def _tg_resume_impl(self, update):
        self._set_system_paused(False, reason='')
        self._tg_reply(update, '▶️ Hệ thống tiếp tục')

    def _tg_close_all_impl(self, update):
        with self.equity_lock:
            self.positions.clear()
            self.shadow_positions.clear()
            try:
                if getattr(self, 'execution', None) is not None:
                    getattr(self.execution, 'positions', {}).clear()
            except Exception:
                pass
        self._tg_reply(update, '🔄 Đã đóng tất cả positions')

    def _tg_toggle_demo_impl(self, update):
        self.modes['demo'] = not self.modes.get('demo', False)
        self._tg_reply(update, f"DEMO mode: {'BẬT' if self.modes['demo'] else 'TẮT'}")

    def _tg_toggle_shadow_impl(self, update):
        self.modes['shadow'] = not self.modes.get('shadow', False)
        self._tg_reply(update, f"SHADOW mode: {'BẬT' if self.modes['shadow'] else 'TẮT'}")

    def _tg_toggle_paper_impl(self, update):
        self.modes['paper'] = not self.modes.get('paper', False)
        self._tg_reply(update, f"PAPER mode: {'BẬT' if self.modes['paper'] else 'TẮT'}")

    def _tg_toggle_live_impl(self, update):
        self.modes['live'] = not self.modes.get('live', False)
        self._tg_reply(update, f"LIVE mode: {'BẬT' if self.modes['live'] else 'TẮT'}")

    def has_recent_btc_signal(self):
        if self.last_btc_signal_time and (datetime.now() - self.last_btc_signal_time) < timedelta(minutes=30):
            return True
        return False

    def _save_config_atomic(self):
        """Persist current config back to config_path (atomic)."""
        try:
            path = getattr(self, 'config_path', 'config.json')
            tmp = path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        except Exception:
            logging.exception('[CONFIG] save failed')

    def _maybe_reload_config(self):
        """Hot reload config and propagate a safe subset."""
        try:
            new_cfg = getattr(self, '_config_reloader', None)
            new_cfg = new_cfg.maybe_reload() if new_cfg else None
        except Exception:
            new_cfg = None
        if not isinstance(new_cfg, dict):
            return
        self.config = new_cfg
        # Modes
        try:
            if isinstance(new_cfg.get('modes'), dict):
                self.modes.update({k: bool(v) for k, v in new_cfg.get('modes', {}).items()})
        except Exception:
            pass
        # Risk
        try:
            if getattr(self, 'risk_guard', None) is not None and hasattr(self.risk_guard, 'update_config'):
                self.risk_guard.update_config(new_cfg)
        except Exception:
            pass
        # Circuit
        try:
            if getattr(self, 'circuit', None) is not None:
                self.circuit.update_config(new_cfg)
        except Exception:
            pass
        # Microstructure
        try:
            if getattr(self, 'micro', None) is not None:
                self.micro.update_config(new_cfg)
        except Exception:
            pass
        # Governor
        try:
            if getattr(self, 'governor', None) is not None:
                self.governor.update_config(new_cfg)
        except Exception:
            pass
        # Metrics
        try:
            if getattr(self, 'metrics', None) is not None:
                self.metrics.update_config(new_cfg)
        except Exception:
            pass
        logging.info('[CONFIG] hot reloaded')

    def _trip_live_killswitch(self, reason: str):
        """Disable LIVE via config kill-switch (persistent)."""
        try:
            self.circuit.trip(reason or 'TRIPPED')
        except Exception:
            pass
        try:
            live_ctl = self.config.get('live_control', {}) if isinstance(self.config, dict) else {}
            if not isinstance(live_ctl, dict):
                live_ctl = {}
            live_ctl['enabled'] = False
            self.config['live_control'] = live_ctl
            if isinstance(self.config.get('modes'), dict):
                self.config['modes']['live'] = False
            self.modes['live'] = False
            self._save_config_atomic()
            logging.error(f"[LIVE][KILL] disabled: {reason}")
            try:
                if getattr(self, 'telegram', None):
                    self.telegram.send_message(f"LIVE disabled: {reason}")
            except Exception:
                pass
        except Exception:
            logging.exception('[LIVE][KILL] failed')

    def _periodic_live_reconcile(self):
        """Periodic live reconcile (account + positions)."""
        if not self.modes.get('live', False):
            return
        live_ctl = self.config.get('live_control', {}) if isinstance(self.config, dict) else {}
        if not bool((live_ctl or {}).get('enabled', False)):
            return
        freq = int((self.config.get('circuit_breakers', {}) or {}).get('reconcile_every_seconds', 60) or 60)
        now = time.time()
        if (now - float(getattr(self, '_last_live_reconcile', 0.0) or 0.0)) < freq:
            return
        self._last_live_reconcile = now

        # Sync wallet/upnl
        try:
            self._sync_live_account()
        except Exception:
            pass

        # Reconcile open positions
        try:
            exec_pos = getattr(self.execution, 'positions', {}) or {}
            for sym, pos in list(exec_pos.items()):
                side = str(getattr(pos, 'side', '') or '').upper()
                rp = self.execution.reconcile_live_position(sym, expected_side=side)
                try:
                    amt = float(rp.get('positionAmt', 0.0) or 0.0) if rp else 0.0
                except Exception:
                    amt = 0.0
                if amt == 0.0:
                    self.circuit.record_reconcile_fail(f'position_missing:{sym}')
                elif rp.get('_mismatch'):
                    self.circuit.record_reconcile_fail(f'position_side_mismatch:{sym}')
                else:
                    self.circuit.record_reconcile_ok()

            if self.circuit.should_trip():
                self._trip_live_killswitch(f'reconcile: {self.circuit.last_trip_reason}')
        except Exception as e:
            self.circuit.record_reconcile_fail(str(e))
            if self.circuit.should_trip():
                self._trip_live_killswitch(f'reconcile_exception: {self.circuit.last_trip_reason}')



    def start(self):
        print(f"▶️ Bắt đầu chạy | Scan mỗi {self.scan_interval_seconds}s\n")
        try:
            while self.running:
                scan_started = time.time()

                # Performance auto-pause: if recent trades are losing, pause entries for a cooldown
                try:
                    if float(getattr(self, '_perf_pause_until', 0.0) or 0.0) > time.time():
                        remaining = int(getattr(self, '_perf_pause_until', 0.0) - time.time())
                        if remaining > 0:
                            logging.warning("PERF_PAUSE active for ~%ss | reason: %s" % (remaining, str(getattr(self, '_perf_pause_reason', '') or '')))
                        time.sleep(min(self.scan_interval_seconds, 10))
                        continue
                except Exception:
                    pass

                # Hot reload config (safe subset)
                try:
                    self._maybe_reload_config()
                except Exception:
                    pass

                # CIO-grade adaptation: ops-aware bandit + rollback
                try:
                    if getattr(self, 'adaptive', None) is not None:
                        self.adaptive.maybe_step()
                except Exception:
                    pass
                stats = {
                    'scanned': 0,
                    'no_data': 0,
                    'low_vol': 0,
                    'no_signal': 0,
                    'risk_reject': 0,
                    'has_position': 0,
                    'follower_gate': 0,
                    'opened_demo': 0,
                    'opened_shadow': 0,
                    'opened_paper': 0,
                    'opened_live': 0,
                    'errors': 0,
                }

                # BTC leader context (strict gating for followers)
                btc_ctx = {}
                try:
                    btc_df = self.market_data.get_klines('BTCUSDT', '5m', limit=200)
                    btc_feat = self.strategy.extract_features('BTCUSDT', df=btc_df) if btc_df is not None else self.strategy.extract_features('BTCUSDT')
                    # Compute an explicit BTC signal for context (does NOT force an order)
                    btc_sig = self.strategy.generate_signals('BTCUSDT', btc_feat) if btc_feat else None
                    btc_ctx = self.strategy.update_btc_context(btc_feat, btc_sig) if btc_feat else {}
                except Exception:
                    btc_ctx = {}

                # Portfolio caps (CIO-grade): prevent opening 4 correlated positions at once.
                # Default is conservative to reduce correlation blow-ups while still allowing
                # frequent trading.
                caps = (self.config.get('portfolio_caps', {}) or self.config.get('portfolio', {})) if isinstance(self.config, dict) else {}
                try:
                    max_total_pos = int(caps.get('max_total_positions', 2) or 2)
                except Exception:
                    max_total_pos = 2
                try:
                    max_entries_scan = int(caps.get('max_entries_per_scan', 2) or 2)
                except Exception:
                    max_entries_scan = 2
                try:
                    max_alt_pos = int(caps.get('max_alt_positions', 1) or 1)
                except Exception:
                    max_alt_pos = 1
                try:
                    min_entry_spacing_s = float(caps.get('min_entry_spacing_s', 15.0) or 15.0)
                except Exception:
                    min_entry_spacing_s = 15.0

                opened_this_scan = 0
                opened_alt_this_scan = 0
                last_entry_ts = float(getattr(self, '_last_entry_ts', 0.0) or 0.0)

                for symbol in self.symbols:
                    stats['scanned'] += 1
                    try:
                        df = self.market_data.get_klines(symbol, '5m', limit=200)
                        if df is None or len(df) < 50:
                            stats['no_data'] += 1
                            continue

                        # Reuse the fetched dataframe to avoid double API calls
                        # and keep features consistent with the volatility gate.
                        features = self.strategy.extract_features(symbol, df=df)
                        volatility = float(features.get('volatility', 0.0) or 0.0)
                        # Use strategy runtime volatility gate (adaptive engine may tune it)
                        try:
                            if hasattr(self.strategy, 'get_min_volatility'):
                                min_vol = float(self.strategy.get_min_volatility(symbol, features) or 0.0006)
                            else:
                                min_vol = float(getattr(self.strategy, 'min_volatility', 0.0006) or 0.0006)
                        except Exception:
                            min_vol = 0.0006
                        if volatility < min_vol:
                            stats['low_vol'] += 1
                            continue

                        sig = self.strategy.generate_signals(symbol, features)
                        if not sig:
                            # Attribute some no-signal cases to follower gating for better visibility
                            try:
                                lr = getattr(self.strategy, 'last_reject', {}) or {}
                                if lr.get('symbol') == symbol and lr.get('code') in ('BTC_GATE', 'WICK_FILTER', 'EXT_FILTER'):
                                    stats['follower_gate'] += 1

                                # Also surface the *reason* periodically (throttled) so you can debug
                                # why the system is not placing orders.
                                if lr.get('symbol') == symbol and lr.get('code'):
                                    now_ts = time.time()
                                    last_ts = float((self._no_trade_log_ts or {}).get(symbol, 0.0) or 0.0)
                                    # Log at most once every 3 minutes per symbol
                                    if (now_ts - last_ts) >= 180.0:
                                        self._no_trade_log_ts[symbol] = now_ts
                                        logging.info(f"NO_TRADE {symbol}: {lr.get('code')} | {lr.get('reason')}")
                            except Exception:
                                pass
                            stats['no_signal'] += 1
                            continue

                        logging.info(f"Signal: {sig.action} {symbol} @ {sig.entry_price:.2f} | Conf {sig.confidence:.3f}")

                        with self.equity_lock:
                            # Use mode-specific equity for sizing (live-like).
                            primary = self._primary_mode_for_sizing()
                            try:
                                if hasattr(self, 'state'):
                                    self.risk_guard.equity = float(self.state.get(primary).get('equity', self.risk_guard.equity) or self.risk_guard.equity)
                            except Exception:
                                pass
                            # ===== V7: provide runtime snapshots for anti-churn & correlation cluster caps =====
                            try:
                                exec_pos_snapshot = getattr(self.execution, 'positions', {}) or {}
                                self.risk_guard.set_positions_snapshot(exec_pos_snapshot)
                            except Exception:
                                exec_pos_snapshot = {}

                            try:
                                if df is not None and 'close' in df.columns:
                                    self.risk_guard.update_returns_cache(symbol, df['close'].astype(float).values)
                                    mid_px = float(df['close'].astype(float).iloc[-1])
                                    # ATR%% proxy (14): volatility for anti-churn latency penalty
                                    vol_pct = 0.0
                                    try:
                                        if all(c in df.columns for c in ['high','low','close']):
                                            h = df['high'].astype(float)
                                            lo = df['low'].astype(float)
                                            cl = df['close'].astype(float)
                                            prev = cl.shift(1)
                                            tr = (h - lo).abs()
                                            tr = np.maximum(tr, (h - prev).abs())
                                            tr = np.maximum(tr, (lo - prev).abs())
                                            atr = tr.rolling(14).mean().iloc[-1]
                                            if mid_px > 0 and atr is not None and np.isfinite(atr):
                                                vol_pct = float(atr / mid_px)
                                    except Exception:
                                        vol_pct = 0.0
                                    # spread estimate (bps) from config cap
                                    ms = self.config.get('microstructure', {}) if isinstance(self.config, dict) else {}
                                    max_spread_pct = float(ms.get('max_spread_pct', 0.0015) or 0.0015)
                                    spread_bps_est = float(max(1.0, min(12.0, max_spread_pct * 10000.0 * 0.35)))
                                    self.risk_guard.update_market_snapshot(symbol, mid=mid_px, spread_bps=spread_bps_est, vol_pct=vol_pct)
                            except Exception:
                                pass

                            if not self.risk_guard.validate_entry(symbol, sig):
                                stats['risk_reject'] += 1
                                continue

                            exec_pos = getattr(self.execution, 'positions', {}) or {}
                            # If there is an existing position, we normally skip. However, when
                            # pyramiding is enabled we may add-to-winner on strong trend.
                            is_add = False
                            pyr_cfg = self.config.get('pyramiding', {}) if isinstance(self.config, dict) else {}
                            pyr_enabled = bool(pyr_cfg.get('enabled', False))
                            # Performance/risk clamp: disable pyramiding when system is in drawdown or auto-derisk is active
                            try:
                                if bool(getattr(self, '_pyramiding_forced_off', False)):
                                    pyr_enabled = False
                                rs = getattr(getattr(self, 'risk_guard', None), 'state', None)
                                if rs is not None and str(getattr(rs, 'value', rs)) in ('CAUTION','DANGER','CRITICAL','DEAD'):
                                    pyr_enabled = False
                            except Exception:
                                pass
                            if symbol in exec_pos:
                                try:
                                    pos0 = exec_pos.get(symbol)
                                    if pyr_enabled and pos0 is not None and str(getattr(pos0, 'side', '')).upper() == str(sig.action).upper():
                                        # Compute RR in favor vs initial risk
                                        cur_px = float(features.get('close', 0.0) or 0.0)
                                        if cur_px <= 0:
                                            try:
                                                cur_px = float(self.market_data.get_current_price(symbol) or 0.0)
                                            except Exception:
                                                cur_px = 0.0
                                        entry_px = float(getattr(pos0, 'entry_price', 0.0) or 0.0)
                                        sl_px = float(getattr(pos0, 'stop_loss', 0.0) or 0.0)
                                        denom = 0.0
                                        if str(sig.action).upper() == 'LONG':
                                            denom = max(entry_px - sl_px, 1e-12)
                                            rr_fav = (cur_px - entry_px) / denom
                                        else:
                                            denom = max(sl_px - entry_px, 1e-12)
                                            rr_fav = (entry_px - cur_px) / denom

                                        add_trigger_rr = float(pyr_cfg.get('add_trigger_rr', 0.9) or 0.9)
                                        max_adds = int(pyr_cfg.get('max_adds', 2) or 2)
                                        add_size_pct = float(pyr_cfg.get('add_size_pct', 0.45) or 0.45)
                                        min_add_interval_s = float(pyr_cfg.get('min_add_interval_s', 120) or 120)
                                        ts_now = float(time.time())
                                        last_add = float(getattr(pos0, 'last_add_ts', 0.0) or 0.0)
                                        add_count = int(getattr(pos0, 'pyramid_count', 0) or 0)

                                        # Extra quality gate: only add when trend is strong (avoid adding in chop)
                                        ts_min = float(pyr_cfg.get('only_if_trend_strength_min', 0.0018) or 0.0018)
                                        trend_strength = float(features.get('trend_strength', 0.0) or 0.0)
                                        strong_trend = abs(trend_strength) >= ts_min
                                        dir_ok = (trend_strength > 0 and str(sig.action).upper() == 'LONG') or (trend_strength < 0 and str(sig.action).upper() == 'SHORT')

                                        if rr_fav >= add_trigger_rr and add_count < max_adds and (ts_now - last_add) >= min_add_interval_s and strong_trend and dir_ok:
                                            # Prepare an ADD signal using existing brackets
                                            is_add = True
                                            try:
                                                setattr(sig, 'allow_add', True)
                                            except Exception:
                                                pass
                                            try:
                                                setattr(sig, 'stop_loss', float(sl_px))
                                                setattr(sig, 'take_profit', float(getattr(pos0, 'take_profit', sig.take_profit) or sig.take_profit))
                                            except Exception:
                                                pass
                                            try:
                                                add_qty = float(getattr(pos0, 'quantity', 0.0) or 0.0) * float(add_size_pct)
                                                setattr(sig, 'position_qty', float(add_qty))
                                            except Exception:
                                                pass
                                            try:
                                                sig.reason = f"{getattr(sig, 'reason', '')} | PYRAMID_ADD rr={rr_fav:.2f}"
                                            except Exception:
                                                pass
                                        else:
                                            stats['has_position'] += 1
                                            continue
                                    else:
                                        stats['has_position'] += 1
                                        continue
                                except Exception:
                                    stats['has_position'] += 1
                                    continue

                            if (not is_add) and (symbol in getattr(self, 'demo_positions', {}) or symbol in getattr(self, 'shadow_positions', {})):
                                # Optional pyramiding for demo/shadow modes
                                if pyr_enabled:
                                    try:
                                        posd = None
                                        if symbol in getattr(self, 'demo_positions', {}):
                                            posd = getattr(self, 'demo_positions', {}).get(symbol)
                                        elif symbol in getattr(self, 'shadow_positions', {}):
                                            posd = getattr(self, 'shadow_positions', {}).get(symbol)
                                        if posd is not None and str(getattr(posd, 'side', '')).upper() == str(sig.action).upper():
                                            cur_px = float(features.get('close', 0.0) or 0.0)
                                            entry_px = float(getattr(posd, 'entry_price', 0.0) or 0.0)
                                            sl_px = float(getattr(posd, 'stop_loss', 0.0) or 0.0)
                                            if str(sig.action).upper() == 'LONG':
                                                denom = max(entry_px - sl_px, 1e-12)
                                                rr_fav = (cur_px - entry_px) / denom
                                            else:
                                                denom = max(sl_px - entry_px, 1e-12)
                                                rr_fav = (entry_px - cur_px) / denom

                                            add_trigger_rr = float(pyr_cfg.get('add_trigger_rr', 0.9) or 0.9)
                                            max_adds = int(pyr_cfg.get('max_adds', 2) or 2)
                                            add_size_pct = float(pyr_cfg.get('add_size_pct', 0.45) or 0.45)
                                            min_add_interval_s = float(pyr_cfg.get('min_add_interval_s', 120) or 120)
                                            ts_now = float(time.time())
                                            last_add = float(getattr(posd, 'last_add_ts', 0.0) or 0.0)
                                            add_count = int(getattr(posd, 'pyramid_count', 0) or 0)
                                            ts_min = float(pyr_cfg.get('only_if_trend_strength_min', 0.0018) or 0.0018)
                                            trend_strength = float(features.get('trend_strength', 0.0) or 0.0)
                                            strong_trend = abs(trend_strength) >= ts_min
                                            dir_ok = (trend_strength > 0 and str(sig.action).upper() == 'LONG') or (trend_strength < 0 and str(sig.action).upper() == 'SHORT')

                                            if rr_fav >= add_trigger_rr and add_count < max_adds and (ts_now - last_add) >= min_add_interval_s and strong_trend and dir_ok:
                                                is_add = True
                                                try:
                                                    setattr(sig, 'allow_add', True)
                                                    setattr(sig, 'stop_loss', float(sl_px))
                                                    setattr(sig, 'take_profit', float(getattr(posd, 'take_profit', sig.take_profit) or sig.take_profit))
                                                    add_qty = float(getattr(posd, 'quantity', 0.0) or 0.0) * float(add_size_pct)
                                                    setattr(sig, 'position_qty', float(add_qty))
                                                    sig.reason = f"{getattr(sig, 'reason', '')} | PYRAMID_ADD rr={rr_fav:.2f}"
                                                except Exception:
                                                    pass
                                                # Apply immediately to demo/shadow positions (simulate add)
                                                try:
                                                    add_qty2 = float(getattr(sig, 'position_qty', 0.0) or 0.0)
                                                    if add_qty2 > 0:
                                                        new_qty = float(getattr(posd, 'quantity', 0.0) or 0.0) + add_qty2
                                                        if new_qty > 0:
                                                            posd.entry_price = (float(getattr(posd, 'entry_price', 0.0) or 0.0) * float(getattr(posd, 'quantity', 0.0) or 0.0) + float(cur_px) * add_qty2) / new_qty
                                                            posd.quantity = float(new_qty)
                                                            posd.pyramid_count = int(getattr(posd, 'pyramid_count', 0) or 0) + 1
                                                            posd.last_add_ts = float(time.time())
                                                except Exception:
                                                    pass

                                    except Exception:
                                        pass
                                if not is_add:
                                    stats['has_position'] += 1
                                    continue
                            profile = self.risk_guard.coin_profiles.get(symbol)
                            if getattr(profile, 'role', 'FOLLOWER') == 'FOLLOWER':
                                # Follower gating is enforced inside StrategyEngine via BTC leader context.
                                # This block remains only to keep stats readable when BTC has no direction.
                                try:
                                    if not (btc_ctx or {}).get('direction'):
                                        stats['follower_gate'] += 1
                                except Exception:
                                    pass

                            if symbol == 'BTCUSDT':
                                self.last_btc_signal_time = datetime.now()

                            qty = float(getattr(sig, 'position_qty', 0.0) or 0.0)
                            if qty <= 0:
                                # safety: không có qty hợp lệ
                                stats['risk_reject'] += 1
                                continue

                            # CIO-grade runtime risk governor: clamp qty + leverage cap (no config mutation)
                            try:
                                rm = {}
                                if hasattr(self.risk_guard, 'get_metrics'):
                                    _m = self.risk_guard.get_metrics()  # dataclass or dict
                                    if isinstance(_m, dict):
                                        rm = _m
                                    else:
                                        # dataclass -> dict-ish
                                        rm = getattr(_m, '__dict__', {}) or {}
                                scale, lev_cap, gov_reason = self.governor.compute(rm, features)
                                if scale <= 0.0:
                                    stats['risk_reject'] += 1
                                    logging.warning(f"[GOV] Block {symbol}: {gov_reason}")
                                    continue
                                qty = qty * float(scale)
                                try:
                                    setattr(sig, 'position_qty', float(qty))
                                except Exception:
                                    pass

                                # === Sync entry/SL/TP across DEMO/SHADOW/PAPER (use same paper fill model) ===
                                try:
                                    if getattr(self, 'execution', None) is not None and hasattr(self.execution, '_paper_fill_price'):
                                        ref_px = 0.0
                                        try:
                                            ref_px = float(self.market_data.get_current_price(symbol) or 0.0)
                                        except Exception:
                                            ref_px = 0.0
                                        if ref_px <= 0:
                                            try:
                                                ref_px = float(getattr(sig, 'entry_price', 0.0) or 0.0)
                                            except Exception:
                                                ref_px = 0.0
                                        side_u = str(getattr(sig, 'action', '') or '').upper()
                                        fill_px = float(self.execution._paper_fill_price(symbol, side_u, float(ref_px or 0.0)) or 0.0)
                                        if fill_px > 0:
                                            try:
                                                sl2, tp2 = self.execution._align_brackets_with_fill(
                                                    side=side_u,
                                                    fill_price=float(fill_px),
                                                    signal_entry=float(getattr(sig, 'entry_price', 0.0) or fill_px),
                                                    stop_loss=float(getattr(sig, 'stop_loss', 0.0) or 0.0),
                                                    take_profit=float(getattr(sig, 'take_profit', 0.0) or 0.0),
                                                )
                                            except Exception:
                                                sl2, tp2 = float(getattr(sig, 'stop_loss', 0.0) or 0.0), float(getattr(sig, 'take_profit', 0.0) or 0.0)
                                            try:
                                                sig.entry_price = float(fill_px)
                                                sig.stop_loss = float(sl2)
                                                sig.take_profit = float(tp2)
                                            except Exception:
                                                pass
                                except Exception:
                                    pass
                                # cap leverage for LIVE executor
                                try:
                                    base_lev = int(self.config.get('default_leverage', 10) or 10)
                                    setattr(sig, 'leverage', int(min(base_lev, int(lev_cap))))
                                except Exception:
                                    pass

                            except Exception:
                                pass
                            if qty <= 0:
                                stats['risk_reject'] += 1
                                continue

                            # ---------------------------
                            # Portfolio-level entry caps
                            # ---------------------------
                            try:
                                total_open = 0
                                # IMPORTANT: count UNIQUE symbols across modes (demo/shadow/paper/live)
                                # so multi-mode does NOT consume portfolio caps 3x.
                                open_syms = set()
                                try:
                                    open_syms |= set((getattr(self, 'demo_positions', {}) or {}).keys())
                                except Exception:
                                    pass
                                try:
                                    open_syms |= set((getattr(self, 'shadow_positions', {}) or {}).keys())
                                except Exception:
                                    pass
                                try:
                                    open_syms |= set((getattr(getattr(self, 'execution', None), 'positions', {}) or {}).keys())
                                except Exception:
                                    pass
                                total_open = len(open_syms)
                            except Exception:
                                total_open = 0

                            # Enforce spacing between entries (avoid 4 entries in same second)
                            try:
                                now_ts = time.time()
                            except Exception:
                                now_ts = 0.0
                            if now_ts > 0 and last_entry_ts > 0 and (now_ts - last_entry_ts) < float(min_entry_spacing_s):
                                stats['risk_reject'] += 1
                                logging.info(f"[PORTFOLIO] Skip {symbol}: entry spacing {now_ts-last_entry_ts:.1f}s < {min_entry_spacing_s:.1f}s")
                                continue

                            if opened_this_scan >= max_entries_scan:
                                stats['risk_reject'] += 1
                                logging.info(f"[PORTFOLIO] Skip {symbol}: max_entries_per_scan={max_entries_scan}")
                                continue

                            if (not is_add) and total_open >= max_total_pos:
                                stats['risk_reject'] += 1
                                logging.info(f"[PORTFOLIO] Skip {symbol}: max_total_positions={max_total_pos}")
                                continue

                            # Correlation control: prefer BTC, allow at most N ALTs per scan.
                            is_alt = (str(symbol).upper() != 'BTCUSDT')
                            if (not is_add) and is_alt and opened_alt_this_scan >= max_alt_pos:
                                stats['follower_gate'] += 1
                                logging.info(f"[PORTFOLIO] Skip {symbol}: max_alt_positions={max_alt_pos}")
                                continue

                            did_open = False

                            if self.modes.get('demo', False):
                                # Live-like fill + bracket alignment (shared with PAPER)
                                try:
                                    fill_price = self.execution._paper_fill_price(symbol, sig.action, float(getattr(sig, 'entry_price', 0.0) or 0.0))
                                    sl2, tp2 = self.execution._align_brackets_with_fill(
                                        side=str(sig.action),
                                        fill_price=float(fill_price),
                                        signal_entry=float(getattr(sig, 'entry_price', 0.0) or 0.0),
                                        stop_loss=float(getattr(sig, 'stop_loss', 0.0) or 0.0),
                                        take_profit=float(getattr(sig, 'take_profit', 0.0) or 0.0),
                                    )
                                except Exception:
                                    fill_price = float(getattr(sig, 'entry_price', 0.0) or 0.0)
                                    sl2 = float(getattr(sig, 'stop_loss', 0.0) or 0.0)
                                    tp2 = float(getattr(sig, 'take_profit', 0.0) or 0.0)

                                # Ensure TP exists (never SL-only)
                                if not tp2 or float(tp2) <= 0:
                                    try:
                                        risk = abs(float(fill_price) - float(sl2))
                                        rr_min = float(self.config.get('rr_min', 1.5) or 1.5) if isinstance(self.config, dict) else 1.5
                                    except Exception:
                                        risk = abs(float(fill_price) - float(sl2))
                                        rr_min = 1.5
                                    if risk > 0:
                                        tp2 = float(fill_price) + (risk * rr_min if str(sig.action).upper() == 'LONG' else -risk * rr_min)

                                self.demo_positions[symbol] = Position(
                                    symbol=symbol,
                                    side=sig.action,
                                    entry_price=float(fill_price),
                                    quantity=float(qty),
                                    stop_loss=float(sl2),
                                    take_profit=float(tp2),
                                    entry_time=datetime.now()
                                )
                                # Seed scaling / analytics fields
                                try:
                                    pos0 = self.demo_positions.get(symbol)
                                    if pos0 is not None:
                                        pos0.initial_qty = float(getattr(pos0, 'quantity', 0.0) or 0.0)
                                        pos0.initial_entry = float(getattr(pos0, 'entry_price', 0.0) or 0.0)
                                        pos0.pyramid_count = int(getattr(pos0, 'pyramid_count', 0) or 0)
                                        pos0.last_add_ts = float(time.time())
                                except Exception:
                                    pass
                                try:
                                    pos = self.demo_positions.get(symbol)
                                    if pos is not None:
                                        pos.leverage = int(self.config.get('default_leverage', getattr(pos, 'leverage', 10)) or getattr(pos, 'leverage', 10) or 10)
                                        pos.confidence = float(getattr(sig, 'confidence', 0.0) or 0.0)
                                        pos.regime = str(getattr(sig, 'regime', '') or '')
                                        pos.entry_reason = str(getattr(sig, 'reason', '') or '')
                                        setattr(pos, 'features', getattr(sig, 'features', {}) or {})
                                except Exception:
                                    pass
                                try:
                                    self.demo_positions[symbol].mode = 'demo'
                                except Exception:
                                    pass
                                # Persist initial risk for RR/BE/Trailing logic (prevents thin wins due to tight initial SL)
                                try:
                                    p0 = self.demo_positions.get(symbol)
                                    if p0 is not None:
                                        p0.initial_stop_loss = float(getattr(p0, 'stop_loss', 0.0) or 0.0)
                                        p0.initial_take_profit = float(getattr(p0, 'take_profit', 0.0) or 0.0)
                                        p0.initial_risk = abs(float(getattr(p0, 'entry_price', 0.0) or 0.0) - float(getattr(p0, 'initial_stop_loss', 0.0) or 0.0))
                                except Exception:
                                    pass
                                stats['opened_demo'] += 1
                                did_open = True
                                logging.info(f"[DEMO] Open {symbol} {sig.action} qty={qty:.8f} entry={sig.entry_price:.2f}")

                                # Persist OPEN event + feed runtime metrics (so /report shows real orders/trades)
                                try:
                                    if hasattr(self, 'state'):
                                        self.state.append_event('demo', 'OPEN', {
                                            'symbol': symbol,
                                            'side': str(sig.action),
                                            'entry': float(getattr(sig, 'entry_price', 0.0) or 0.0),
                                            'qty': float(qty or 0.0),
                                            'sl': float(getattr(sig, 'stop_loss', 0.0) or 0.0),
                                            'tp': float(getattr(sig, 'take_profit', 0.0) or 0.0),
                                            'conf': float(getattr(sig, 'confidence', 0.0) or 0.0),
                                        })
                                except Exception:
                                    pass
                                try:
                                    if getattr(self, 'metrics', None) is not None:
                                        self.metrics.record_order(symbol, 0.0, 0.0, True, meta={'mode': 'demo', 'sim': True})
                                except Exception:
                                    pass
                                try:
                                    # baseline for scaling management (pyramiding/partials)
                                    self.demo_positions[symbol].initial_qty = float(qty or 0.0)
                                    self.demo_positions[symbol].pyramid_count = int(getattr(self.demo_positions[symbol], 'pyramid_count', 0) or 0)
                                    self.demo_positions[symbol].last_add_ts = float(time.time())
                                except Exception:
                                    pass
                            if self.modes.get('shadow', False):
                                # Live-like fill + bracket alignment (shared with PAPER)
                                try:
                                    fill_price = self.execution._paper_fill_price(symbol, sig.action, float(getattr(sig, 'entry_price', 0.0) or 0.0))
                                    sl2, tp2 = self.execution._align_brackets_with_fill(
                                        side=str(sig.action),
                                        fill_price=float(fill_price),
                                        signal_entry=float(getattr(sig, 'entry_price', 0.0) or 0.0),
                                        stop_loss=float(getattr(sig, 'stop_loss', 0.0) or 0.0),
                                        take_profit=float(getattr(sig, 'take_profit', 0.0) or 0.0),
                                    )
                                except Exception:
                                    fill_price = float(getattr(sig, 'entry_price', 0.0) or 0.0)
                                    sl2 = float(getattr(sig, 'stop_loss', 0.0) or 0.0)
                                    tp2 = float(getattr(sig, 'take_profit', 0.0) or 0.0)

                                # Ensure TP exists (never SL-only)
                                if not tp2 or float(tp2) <= 0:
                                    try:
                                        risk = abs(float(fill_price) - float(sl2))
                                        rr_min = float(self.config.get('rr_min', 1.5) or 1.5) if isinstance(self.config, dict) else 1.5
                                    except Exception:
                                        risk = abs(float(fill_price) - float(sl2))
                                        rr_min = 1.5
                                    if risk > 0:
                                        tp2 = float(fill_price) + (risk * rr_min if str(sig.action).upper() == 'LONG' else -risk * rr_min)

                                self.shadow_positions[symbol] = Position(
                                    symbol=symbol,
                                    side=sig.action,
                                    entry_price=float(fill_price),
                                    quantity=float(qty),
                                    stop_loss=float(sl2),
                                    take_profit=float(tp2),
                                    entry_time=datetime.now()
                                )
                                try:
                                    pos0 = self.shadow_positions.get(symbol)
                                    if pos0 is not None:
                                        pos0.initial_qty = float(getattr(pos0, 'quantity', 0.0) or 0.0)
                                        pos0.initial_entry = float(getattr(pos0, 'entry_price', 0.0) or 0.0)
                                        pos0.pyramid_count = int(getattr(pos0, 'pyramid_count', 0) or 0)
                                        pos0.last_add_ts = float(time.time())
                                except Exception:
                                    pass
                                try:
                                    pos = self.shadow_positions.get(symbol)
                                    if pos is not None:
                                        pos.leverage = int(self.config.get('default_leverage', getattr(pos, 'leverage', 10)) or getattr(pos, 'leverage', 10) or 10)
                                        pos.confidence = float(getattr(sig, 'confidence', 0.0) or 0.0)
                                        pos.regime = str(getattr(sig, 'regime', '') or '')
                                        pos.entry_reason = str(getattr(sig, 'reason', '') or '')
                                        setattr(pos, 'features', getattr(sig, 'features', {}) or {})
                                except Exception:
                                    pass
                                try:
                                    self.shadow_positions[symbol].mode = 'shadow'
                                except Exception:
                                    pass
                                # Persist initial risk for RR/BE/Trailing logic (prevents thin wins due to tight initial SL)
                                try:
                                    p0 = self.shadow_positions.get(symbol)
                                    if p0 is not None:
                                        p0.initial_stop_loss = float(getattr(p0, 'stop_loss', 0.0) or 0.0)
                                        p0.initial_take_profit = float(getattr(p0, 'take_profit', 0.0) or 0.0)
                                        p0.initial_risk = abs(float(getattr(p0, 'entry_price', 0.0) or 0.0) - float(getattr(p0, 'initial_stop_loss', 0.0) or 0.0))
                                except Exception:
                                    pass
                                stats['opened_shadow'] += 1
                                did_open = True
                                logging.info(f"[SHADOW] Open {symbol} {sig.action} qty={qty:.8f} entry={sig.entry_price:.2f}")

                                try:
                                    if hasattr(self, 'state'):
                                        self.state.append_event('shadow', 'OPEN', {
                                            'symbol': symbol,
                                            'side': str(sig.action),
                                            'entry': float(getattr(sig, 'entry_price', 0.0) or 0.0),
                                            'qty': float(qty or 0.0),
                                            'sl': float(getattr(sig, 'stop_loss', 0.0) or 0.0),
                                            'tp': float(getattr(sig, 'take_profit', 0.0) or 0.0),
                                            'conf': float(getattr(sig, 'confidence', 0.0) or 0.0),
                                        })
                                except Exception:
                                    pass
                                try:
                                    if getattr(self, 'metrics', None) is not None:
                                        self.metrics.record_order(symbol, 0.0, 0.0, True, meta={'mode': 'shadow', 'sim': True})
                                except Exception:
                                    pass

                                    self.shadow_positions[symbol].initial_qty = float(qty or 0.0)
                                    self.shadow_positions[symbol].pyramid_count = int(getattr(self.shadow_positions[symbol], 'pyramid_count', 0) or 0)
                                    self.shadow_positions[symbol].last_add_ts = float(time.time())
                                except Exception:
                                    pass
                            if self.modes.get('paper', False):
                                ok = self.execution.execute_trade(sig, mode='PAPER')
                                if ok is None:
                                    ok = True
                                if ok:
                                    stats['opened_paper'] += 1
                                    try:
                                        ppos = getattr(self.execution, 'positions', {}) or {}
                                        p = ppos.get(symbol)
                                        try:
                                            if p is not None:
                                                p.confidence = float(getattr(sig, 'confidence', 0.0) or 0.0)
                                                p.regime = str(getattr(sig, 'regime', '') or '')
                                                p.entry_reason = str(getattr(sig, 'reason', '') or '')
                                                setattr(p, 'features', getattr(sig, 'features', {}) or {})
                                        except Exception:
                                            pass
                                        p_entry = float(getattr(p, 'entry_price', 0.0) or getattr(sig, 'entry_price', 0.0) or 0.0)
                                    except Exception:
                                        p_entry = float(getattr(sig, 'entry_price', 0.0) or 0.0)
                                    logging.info(f"[PAPER] Open {symbol} {sig.action} qty={qty:.8f} entry={p_entry:.2f}")
                                    did_open = True

                                    try:
                                        if hasattr(self, 'state'):
                                            self.state.append_event('paper', 'OPEN', {
                                                'symbol': symbol,
                                                'side': str(sig.action),
                                                'entry': float(p_entry or 0.0),
                                                'qty': float(qty or 0.0),
                                                'sl': float(getattr(sig, 'stop_loss', 0.0) or 0.0),
                                                'tp': float(getattr(sig, 'take_profit', 0.0) or 0.0),
                                                'conf': float(getattr(sig, 'confidence', 0.0) or 0.0),
                                            })
                                    except Exception:
                                        pass
                                    try:
                                        if getattr(self, 'metrics', None) is not None:
                                            self.metrics.record_order(symbol, 0.0, 0.0, True, meta={'mode': 'paper', 'sim': True})
                                    except Exception:
                                        pass
                                    try:
                                        # attach baseline info to the paper position object
                                        ppos = getattr(self.execution, 'positions', {}) or {}
                                        p = ppos.get(symbol)
                                        if p is not None:
                                            p.initial_qty = float(getattr(p, 'quantity', qty) or qty)
                                            p.pyramid_count = int(getattr(p, 'pyramid_count', 0) or 0)
                                            p.last_add_ts = float(time.time())
                                    except Exception:
                                        pass
                                    # Seed initial sizing fields for paper position too
                                    try:
                                        ppos = getattr(self.execution, 'positions', {}) or {}
                                        p = ppos.get(symbol)
                                        if p is not None:
                                            p.initial_qty = float(getattr(p, 'quantity', 0.0) or 0.0)
                                            p.initial_entry = float(getattr(p, 'entry_price', 0.0) or 0.0)
                                            p.pyramid_count = int(getattr(p, 'pyramid_count', 0) or 0)
                                            p.last_add_ts = float(time.time())
                                    except Exception:
                                        pass

                            if self.modes.get('live', False):
                                # Live kill-switch is controlled via config (NOT env)
                                live_ctl = self.config.get('live_control', {}) if isinstance(self.config, dict) else {}
                                live_enabled = bool(live_ctl.get('enabled', False))
                                if self.is_public_mode:
                                    live_enabled = False
                                if getattr(self, 'circuit', None) is not None and self.circuit.is_in_cooldown():
                                    live_enabled = False
                                    logging.warning("[LIVE] cooldown active; skip new orders")
                                if not live_enabled:
                                    stats['risk_reject'] += 1
                                    logging.warning(f"[LIVE] Skipped {symbol}: live_control.enabled=false or circuit cooldown")
                                else:
                                    # CIO-grade microstructure guard: block MARKET entries in toxic microstructure
                                    try:
                                        ok_ms, ms_code, ms_diag = self.micro.check(symbol, self.market_data, features)
                                        if not ok_ms:
                                            stats['risk_reject'] += 1
                                            logging.warning(f"[MS] Block {symbol}: {ms_code} {ms_diag}")
                                            continue
                                    except Exception:
                                        stats['risk_reject'] += 1
                                        logging.warning(f"[MS] Block {symbol}: MS_EXCEPTION")
                                        continue
                                    try:
                                        self.execution.execute_trade(sig, mode='LIVE')
                                        stats['opened_live'] += 1
                                        did_open = True
                                        # latency metric (best-effort)
                                        try:
                                            # last order object is stored in execution.order_history
                                            oh = getattr(self.execution, 'order_history', []) or []
                                            if oh:
                                                lat = float(getattr(oh[-1], 'latency_ms', 0.0) or 0.0)
                                                self.circuit.record_latency(lat)
                                                # runtime metrics
                                                try:
                                                    avgp = float(getattr(oh[-1], 'avg_price', 0.0) or 0.0)
                                                except Exception:
                                                    avgp = 0.0
                                                ep = float(getattr(sig, 'entry_price', 0.0) or 0.0)
                                                slip = abs(avgp - ep) / ep if (avgp > 0 and ep > 0) else 0.0
                                                try:
                                                    if getattr(self, 'metrics', None) is not None:
                                                        self.metrics.record_order(symbol, lat, slip, True, meta={'mode': 'live'})
                                                except Exception:
                                                    pass
                                        except Exception:
                                            pass
                                        self.circuit.record_order_ok()
                                        logging.info(f"[LIVE] Submit {symbol} {sig.action} qty={qty:.8f} entry={sig.entry_price:.2f}")
                                    except Exception as e:
                                        stats['errors'] += 1
                                        self.circuit.record_order_error(str(e))
                                        try:
                                            if getattr(self, 'metrics', None) is not None:
                                                self.metrics.record_order(symbol, 0.0, 0.0, False, meta={'mode': 'live', 'err': str(e)})
                                        except Exception:
                                            pass
                                        logging.error(f"[LIVE] Order failed for {symbol}: {e}", exc_info=True)
                                        if self.circuit.should_trip():
                                            self._trip_live_killswitch(f"order_errors: {self.circuit.last_trip_reason}")

                            # Update per-scan entry counters (only if at least one mode opened)
                            if did_open:
                                opened_this_scan += 1
                                if symbol != 'BTCUSDT':
                                    opened_alt_this_scan += 1
                                last_entry_ts = time.time()
                                try:
                                    setattr(self, '_last_entry_ts', float(last_entry_ts))
                                except Exception:
                                    pass
                                try:
                                    setattr(self.strategy, '_last_trade_ts', time.time())
                                except Exception:
                                    pass


                    except Exception as e:
                        stats['errors'] += 1
                        logging.error(f"Error {symbol}: {e}", exc_info=True)
                # Monitor exits 1 lan/scan (tranh O(n^2))
                try:
                    self._monitor_exits()
                except Exception:
                    logging.exception('monitor exits failed')

                # Live periodic reconcile & safety
                try:
                    self._periodic_live_reconcile()
                except Exception:
                    pass

                # Runtime metrics: persist + auto-pause if execution quality degrades
                try:
                    if getattr(self, 'metrics', None) is not None and hasattr(self, 'state'):
                        self.metrics.maybe_persist(self.state, ['demo','shadow','paper','live'])
                        trip, reason, snap = self.metrics.check_and_maybe_trip()
                        if trip and bool(self.modes.get('live', False)):
                            self._trip_live_killswitch(f'metrics_trip: {reason}')
                            try:
                                if getattr(self, 'telegram', None) is not None:
                                    self.telegram.send_message(f"⚠️ LIVE auto-pause: {reason}\n{snap}")
                            except Exception:
                                pass

                except Exception:
                    pass
                elapsed = time.time() - scan_started
                logging.info(
                    "SCAN summary: scanned=%d no_data=%d low_vol=%d no_signal=%d risk_reject=%d has_pos=%d follower_gate=%d "
                    "opened(demo=%d shadow=%d paper=%d live=%d) errors=%d elapsed=%.1fs",
                    stats['scanned'], stats['no_data'], stats['low_vol'], stats['no_signal'], stats['risk_reject'],
                    stats['has_position'], stats['follower_gate'], stats['opened_demo'], stats['opened_shadow'],
                    stats['opened_paper'], stats['opened_live'], stats['errors'], elapsed
                )


                # Feed scan stats to RiskGuard so it can self-tune anti-churn in paper/demo/shadow
                try:
                    if getattr(self, 'risk_guard', None) is not None and hasattr(self.risk_guard, 'note_scan'):
                        self.risk_guard.note_scan(stats)
                except Exception:
                    pass

                time.sleep(self.scan_interval_seconds)

        except KeyboardInterrupt:
            self.stop()

    def _monitor_exits(self):
        """Exit engine (DEMO/SHADOW + PAPER).

        Mục tiêu live-like:
        - Không chỉ SL/TP cứng: thêm Breakeven, Partial TP, Trailing, Time-stop.
        - Update balance/pnl theo mode và persist để không reset.
        """

        cfg_exit = self.config.get('exit_engine', {}) if isinstance(getattr(self, 'config', {}), dict) else {}
        be_rr = float(cfg_exit.get('breakeven_rr', 0.8) or 0.8)
        tp1_rr = float(cfg_exit.get('tp1_rr', 1.0) or 1.0)
        tp1_pct = float(cfg_exit.get('tp1_close_pct', 0.4) or 0.4)  # 40% close
        trail_start_rr = float(cfg_exit.get('trailing_start_rr', 1.2) or 1.2)
        trail_pct = float(cfg_exit.get('trailing_pct', 0.003) or 0.003)  # 0.3%
        time_stop_min = int(cfg_exit.get('time_stop_minutes', 90) or 90)
        time_stop_rr_min = float(cfg_exit.get('time_stop_min_rr', 0.25) or 0.25)

        # Deep performance tuning:
        # - early loss cut reduces tail losses (so 5W/1L stays profitable)
        # - runner mode removes hard TP after TP1 so big winners can run
        early_cut_rr = float(cfg_exit.get('early_cut_rr', -1.25) or -1.25)
        early_cut_pct = float(cfg_exit.get('early_cut_close_pct', 0.35) or 0.35)
        early_cut_min_age = float(cfg_exit.get('early_cut_min_age_minutes', 12) or 12)
        loss_cap_remaining_r = float(cfg_exit.get('loss_cap_remaining_r', 0.5) or 0.5)

        # --- Position dict routing (fix bug: deleting wrong dict) ---
        def _pos_dict_for_mode(mode: str):
            ml = str(mode or '').lower()
            if ml == 'demo':
                return self.demo_positions
            if ml == 'shadow':
                return self.shadow_positions
            if ml == 'live':
                return getattr(self, 'live_positions', {}) or {}
            # PAPER positions are tracked inside execution engine as well, but we also keep mirror dict `self.positions`
            # for stats/reporting. Prefer `self.positions` if present.
            return getattr(self, 'positions', {}) or {}

        def _del_pos(mode: str, sym: str):
            try:
                d = _pos_dict_for_mode(mode)
                if isinstance(d, dict) and sym in d:
                    del d[sym]
            except Exception:
                pass


        # --- Smart TIME_STOP (mode-agnostic, prevents repeated partial fee-bleed) ---
        def _smart_time_stop(mode: str, sym: str, pos, price: float, rr: float, entry: float, sl_dist: float):
            try:
                # age minutes
                try:
                    age_min = (datetime.now() - (getattr(pos, 'entry_time', None) or datetime.now())).total_seconds() / 60.0
                except Exception:
                    age_min = 0.0

                # regime-aware sideway detection
                try:
                    feat_ts = _get_features(sym) or {}
                    atr_ratio = float(feat_ts.get('atr_ratio', 1.0) or 1.0)
                    tr_abs = abs(float(feat_ts.get('trend_strength', 0.0) or 0.0))
                    is_sideway = (atr_ratio <= sideway_atr_ratio_max) and (tr_abs <= sideway_trend_abs_max)
                except Exception:
                    is_sideway = False

                ts_limit = float(time_stop_sideway_min if is_sideway else time_stop_trend_min)

                # quick exits
                if age_min < ts_limit:
                    return False
                if rr >= time_stop_rr_min:
                    return False

                # Do NOT time-stop close positions that are net-positive beyond costs.
                # This prevents the common failure: tiny green gets cut -> win mỏng, lose dày.
                try:
                    q_tmp2 = float(getattr(pos, 'quantity', 0.0) or 0.0)
                    if q_tmp2 > 0:
                        _, _, pnl_net2 = self._calc_futures_pnl_net(pos, float(price))
                        cost2 = _roundtrip_cost_usdt(float(entry), q_tmp2)
                        min_net2 = float((self._get_exit_cfg().get('min_close_profit_net_usdt', 2.0)) or 2.0)

                        # Optional: disable time-stop closes while trade is net-negative (prevents lose dày via TIME_STOP)
                        try:
                            if bool((self._get_exit_cfg().get('disable_time_stop_when_negative', False))):
                                if float(pnl_net2 or 0.0) <= 0:
                                    pos.time_stop_snooze_until = datetime.now() + timedelta(minutes=6.0 if is_sideway else 12.0)
                                    return False
                        except Exception:
                            pass

                        # Do NOT time-stop close if net-positive beyond costs, or if already net-positive (snooze)
                        if float(pnl_net2 or 0.0) >= max(min_net2, 0.80 * float(cost2)):
                            return False
                        if float(pnl_net2 or 0.0) > 0:
                            pos.time_stop_snooze_until = datetime.now() + timedelta(minutes=6.0 if is_sideway else 12.0)
                            return False
                except Exception:
                    pass

                # snooze guard
                try:
                    su = getattr(pos, 'time_stop_snooze_until', None)
                    if su is not None and datetime.now() < su:
                        return False
                except Exception:
                    pass

                # compute thesis invalidation + chop
                try:
                    feat = _get_features(sym) or {}
                    htf_dir = int(feat.get('htf_dir', 0) or 0)
                    htf_strength = float(feat.get('htf_strength', 0.0) or 0.0)
                    trend_strength = float(feat.get('trend_strength', 0.0) or 0.0)
                    chop = float(feat.get('chop', 50.0) or 50.0)
                except Exception:
                    htf_dir, htf_strength, trend_strength, chop = 0, 0.0, 0.0, 50.0

                side_u = str(getattr(pos, 'side', '') or '').upper()
                invalid = False
                try:
                    if side_u == 'LONG':
                        invalid = (htf_dir < 0 and htf_strength >= 0.55) or (trend_strength < -0.0018)
                    else:
                        invalid = (htf_dir > 0 and htf_strength >= 0.55) or (trend_strength > 0.0018)
                except Exception:
                    invalid = False

                # Full close on invalidation or deep chop
                if invalid or (is_sideway and chop >= 58.0):
                    # If configured, do not force-close time-stop while net-negative (let SL manage risk)
                    try:
                        if bool((self._get_exit_cfg().get('disable_time_stop_when_negative', False))):
                            _, _, pnl_net_ts = self._calc_futures_pnl_net(pos, float(price))
                            if float(pnl_net_ts or 0.0) <= 0:
                                try:
                                    pos.time_stop_snooze_until = datetime.now() + timedelta(minutes=6.0 if is_sideway else 12.0)
                                except Exception:
                                    pass
                                return False
                    except Exception:
                        pass
                    _apply_close(mode, sym, pos, price, 'TIME_STOP')
                    # delete from mode dict
                    try:
                        _del_pos(mode, sym)
                    except Exception:
                        pass
                    # delete paper/live executor position if present
                    try:
                        ml = str(mode or '').lower()
                        if ml == 'paper' and hasattr(self, 'execution') and hasattr(self.execution, 'positions'):
                            if sym in self.execution.positions:
                                del self.execution.positions[sym]
                        if ml == 'live' and hasattr(self, 'execution') and hasattr(self.execution, 'positions_live'):
                            if sym in self.execution.positions_live:
                                del self.execution.positions_live[sym]
                    except Exception:
                        pass
                    return True

                # Otherwise: ONLY ONE partial time-stop per position + fee-band gate
                if bool(getattr(pos, 'time_stop_partial_done', False)):
                    try:
                        pos.time_stop_snooze_until = datetime.now() + timedelta(minutes=6.0 if is_sideway else 12.0)
                    except Exception:
                        pass
                    return False

                # Fee band gate: don't pay fees to close tiny moves
                try:
                    q_tmp = float(getattr(pos, 'quantity', 0.0) or 0.0)
                    if q_tmp <= 0:
                        return False
                    if side_u == 'LONG':
                        ug = (float(price) - float(entry)) * q_tmp
                    else:
                        ug = (float(entry) - float(price)) * q_tmp
                    fee_rt = 2.0 * float(getattr(self, 'fee_rate', 0.0004) or 0.0004)
                    est_fee = fee_rt * float(price) * q_tmp
                    if abs(ug) <= float(est_fee) * 1.2:
                        pos.time_stop_snooze_until = datetime.now() + timedelta(minutes=6.0 if is_sideway else 12.0)
                        return False
                except Exception:
                    pass

                # Partial close size (meaningful only)
                try:
                    q0 = float(getattr(pos, 'quantity', 0.0) or 0.0)
                    if q0 <= 0:
                        return False
                    part_pct = 0.35 if is_sideway else 0.25
                    q_close = max(0.0, min(q0 * part_pct, q0 - 1e-12))

                    # minimum notional guard
                    try:
                        min_notional = float(((self.config.get('futures', {}) or {}).get('min_notional_usdt', 0.0)) or 0.0)
                    except Exception:
                        min_notional = 0.0
                    if min_notional > 0 and (q_close * float(price)) < (0.6 * min_notional):
                        # too small: either keep (cheaper) or close full if clearly losing beyond fee band
                        pos.time_stop_snooze_until = datetime.now() + timedelta(minutes=6.0 if is_sideway else 12.0)
                        return False

                    if q_close > 0.0 and q_close < q0:
                        # Guard: never time-stop partial at a net loss (fee-bleed). Let SL/TP handle it.
                        try:
                            _, _, _pnl_net_ts = self._calc_futures_pnl_net(pos, float(price))
                            if float(_pnl_net_ts or 0.0) <= 0.0:
                                try:
                                    pos.time_stop_snooze_until = datetime.now() + timedelta(minutes=6.0 if is_sideway else 12.0)
                                except Exception:
                                    pass
                                return False
                        except Exception:
                            pass
                        _apply_close(mode, sym, pos, price, 'TIME_STOP_PARTIAL', qty_close=q_close)
                        try:
                            pos.quantity = float(q0 - q_close)
                        except Exception:
                            pass
                        try:
                            pos.time_stop_partial_done = True
                        except Exception:
                            pass
                        try:
                            pos.time_stop_snooze_until = datetime.now() + timedelta(minutes=12.0)
                        except Exception:
                            pass
                        # move SL toward BE to stop bleeding
                        try:
                            if sl_dist > 0:
                                be_buf_r = float(self._get_exit_cfg().get('be_buffer_r', 0.12) or 0.12)
                                if side_u == 'LONG':
                                    pos.stop_loss = max(float(getattr(pos, 'stop_loss', 0.0) or 0.0), float(entry) + float(be_buf_r) * float(sl_dist))
                                else:
                                    pos.stop_loss = min(float(getattr(pos, 'stop_loss', entry) or entry), float(entry) - float(be_buf_r) * float(sl_dist))
                        except Exception:
                            pass
                except Exception:
                    pass
                return False
            except Exception:
                return False
        # --- Fee / cost model for scratch/BE decisions (Binance USDT-M futures) ---
        try:
            fee_rate = float(self.config.get('fee_rate', 0.0004) or 0.0004)  # taker default; maker lower
        except Exception:
            fee_rate = 0.0004
        try:
            slip_pct = float(self.config.get('slippage_pct', 0.0006) or 0.0006)
        except Exception:
            slip_pct = 0.0006
        try:
            spread_buf = float((self.config.get('microstructure', {}) or {}).get('spread_cost_buffer', 0.00015) or 0.00015)
        except Exception:
            spread_buf = 0.00015

        def _roundtrip_cost_usdt(entry_px: float, qty: float):
            notional = max(0.0, float(entry_px) * float(qty))
            # Fee charged per side, plus expected slippage (2 sides) and spread buffer.
            return notional * ((fee_rate * 2.0) + (slip_pct * 2.0) + spread_buf)

        # --- Regime-aware time-stop (sideway vs trend) ---
        regime_cfg = cfg_exit.get('regime', {}) if isinstance(cfg_exit, dict) else {}
        time_stop_sideway_min = float(regime_cfg.get('time_stop_sideway_minutes', 10) or 10)
        time_stop_trend_min   = float(regime_cfg.get('time_stop_trend_minutes', time_stop_min) or time_stop_min)
        sideway_atr_ratio_max = float(regime_cfg.get('sideway_atr_ratio_max', 0.92) or 0.92)
        sideway_trend_abs_max = float(regime_cfg.get('sideway_trend_abs_max', 0.012) or 0.012)

        # --- Reversal rescue (controlled hedging) ---
        rescue_cfg = cfg_exit.get('reversal_rescue', {}) if isinstance(cfg_exit, dict) else {}
        rescue_enabled = bool(rescue_cfg.get('enabled', True))
        rescue_max_total_pos = int(rescue_cfg.get('max_total_positions', 4) or 4)  # allow adding 2 ALTs in rescue, but size is small
        rescue_add_qty_frac  = float(rescue_cfg.get('hedge_qty_frac', 0.22) or 0.22)
        rescue_trigger_rr    = float(rescue_cfg.get('trigger_rr', -0.55) or -0.55)
        rescue_min_age_min   = float(rescue_cfg.get('min_age_minutes', 2.0) or 2.0)
        rescue_max_age_min   = float(rescue_cfg.get('max_age_minutes', 25.0) or 25.0)
        rescue_opposite_conf = float(rescue_cfg.get('opposite_signal_min_conf', 0.62) or 0.62)
        rescue_widen_sl_mult = float(rescue_cfg.get('widen_old_sl_mult', 1.22) or 1.22)
        rescue_widen_sl_cap_pct = float(rescue_cfg.get('widen_old_sl_cap_pct', 0.012) or 0.012)  # cap widen to 1.2% price to avoid disaster
        rescue_scratch_fee_mult = float(rescue_cfg.get('scratch_close_fee_mult', 1.1) or 1.1)
        rescue_cooldown_s = int(rescue_cfg.get('cooldown_seconds', 420) or 420)

        if not hasattr(self, '_last_rescue_ts'):
            self._last_rescue_ts = 0.0

        # Add-to-winner (pyramiding) — increases expectancy while keeping initial risk fixed.
        # Only enabled for DEMO/SHADOW/PAPER by default.
        pyr_cfg = cfg_exit.get('pyramiding', {}) if isinstance(cfg_exit, dict) else {}
        pyr_enabled = bool(pyr_cfg.get('enabled', True))
        pyr_max_adds = int(pyr_cfg.get('max_adds', 1) or 1)
        pyr_start_rr = float(pyr_cfg.get('start_rr', 0.8) or 0.8)
        pyr_step_rr = float(pyr_cfg.get('step_rr', 0.9) or 0.9)
        pyr_add_frac = float(pyr_cfg.get('add_qty_frac_of_initial', 0.35) or 0.35)
        pyr_min_age_min = float(pyr_cfg.get('min_age_minutes', 8.0) or 8.0)
        pyr_min_gap_s = int(pyr_cfg.get('min_gap_seconds', 240) or 240)

        def _maybe_pyramid(mode: str, sym: str, pos, price: float, rr: float, features: dict):
            if not pyr_enabled:
                return
            mode_l = str(mode or '').lower()
            if mode_l not in ('demo', 'shadow', 'paper'):
                return

            try:
                pc = int(getattr(pos, 'pyramid_count', 0) or 0)
            except Exception:
                pc = 0
            if pc >= pyr_max_adds:
                return

            # RR-based step ladder
            if rr < (pyr_start_rr + (pyr_step_rr * float(pc))):
                return

            try:
                age_min = (datetime.now() - (getattr(pos, 'entry_time', None) or datetime.now())).total_seconds() / 60.0
            except Exception:
                age_min = 0.0
            if age_min < pyr_min_age_min:
                return

            try:
                last_add = float(getattr(pos, 'last_add_ts', 0.0) or 0.0)
            except Exception:
                last_add = 0.0
            if time.time() - last_add < pyr_min_gap_s:
                return

            # Micro-filter: only add when trend agrees (avoid adding into chop)
            try:
                ts = float((features or {}).get('trend_strength', 0.0) or 0.0)
                macd = float((features or {}).get('macd', 0.0) or 0.0)
                macd_sig = float((features or {}).get('macd_signal', 0.0) or 0.0)
                side = str(getattr(pos, 'side', '') or '').upper()
                ok = (ts > 0 and macd > macd_sig) if side == 'LONG' else (ts < 0 and macd < macd_sig)
                if not ok:
                    return
            except Exception:
                pass

            try:
                init_qty = float(getattr(pos, 'initial_qty', 0.0) or 0.0)
                cur_qty = float(getattr(pos, 'quantity', 0.0) or 0.0)
            except Exception:
                return
            if init_qty <= 0 or cur_qty <= 0:
                return

            add_qty = float(init_qty) * float(pyr_add_frac)
            if add_qty <= 0:
                return

            # PAPER: use executor to update avg entry/qty; DEMO/SHADOW: update local Position.
            if mode_l == 'paper':
                try:
                    class _Sig:
                        pass
                    s = _Sig()
                    s.symbol = sym
                    s.action = str(getattr(pos, 'side', ''))
                    s.entry_price = float(price)
                    s.stop_loss = float(getattr(pos, 'stop_loss', 0.0) or 0.0)
                    s.take_profit = float(getattr(pos, 'take_profit', 0.0) or 0.0)
                    s.position_qty = float(add_qty)
                    s.allow_add = True
                    s.confidence = float(getattr(pos, 'confidence', 0.0) or 0.0)
                    s.reason = 'PYRAMID'
                    self.execution.execute_trade(s, mode='PAPER')
                    _journal('paper', 'PYRAMID', sym, {'add_qty': float(add_qty), 'price': float(price), 'rr': float(rr)})
                except Exception:
                    return
            else:
                try:
                    # weighted average entry
                    new_qty = float(cur_qty) + float(add_qty)
                    new_entry = (float(getattr(pos, 'entry_price', price)) * float(cur_qty) + float(price) * float(add_qty)) / max(new_qty, 1e-12)
                    pos.entry_price = float(new_entry)
                    pos.quantity = float(new_qty)
                    pos.pyramid_count = int(pc) + 1
                    pos.last_add_ts = float(time.time())
                    _journal(mode_l, 'PYRAMID', sym, {'add_qty': float(add_qty), 'price': float(price), 'new_qty': float(new_qty), 'new_entry': float(new_entry), 'rr': float(rr)})
                except Exception:
                    return

            try:
                setattr(pos, 'last_add_ts', float(time.time()))
                setattr(pos, 'pyramid_count', int(pc) + 1)
            except Exception:
                pass

        runner_disable_hard_tp = bool(cfg_exit.get('runner_disable_hard_tp_after_tp1', True))
        trail_use_atr = bool(cfg_exit.get('trail_use_atr', True))
        trail_atr_mult = float(cfg_exit.get('trail_atr_mult', 0.9) or 0.9)
        trail_atr_mult_after_tp1 = float(cfg_exit.get('trail_atr_mult_after_tp1', 1.05) or 1.05)

        _feat_cache = {}

        def _get_features(sym: str):
            try:
                if sym in _feat_cache:
                    return _feat_cache[sym]
                df = self.market_data.get_klines(sym, '5m', limit=200)
                feat = self.strategy.extract_features(sym, df=df) if df is not None else self.strategy.extract_features(sym)
                _feat_cache[sym] = feat or {}
                return _feat_cache[sym]
            except Exception:
                _feat_cache[sym] = {}
                return {}

        def _trail_dist(sym: str, px: float, tp1_done: bool) -> float:
            try:
                base = float(px) * float(trail_pct)
                if not trail_use_atr:
                    return max(0.0, base)
                feat = _get_features(sym)
                atr = float((feat or {}).get('atr', 0.0) or 0.0)
                mult = float(trail_atr_mult_after_tp1 if tp1_done else trail_atr_mult)
                return max(0.0, base, atr * mult)
            except Exception:
                return max(0.0, float(px) * float(trail_pct))

        def _journal(mode: str, event: str, sym: str, data: dict):
            try:
                base = os.path.dirname(os.path.abspath(__file__))
                path = os.path.join(base, f'trades_{mode}.jsonl')
                payload = {'ts': datetime.now().isoformat(), 'mode': mode, 'event': event, 'symbol': sym}
                payload.update(data or {})
                with open(path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            except Exception:
                pass

        # Idempotency guard to prevent duplicate CLOSE events in the same scan loop
        closed_this_scan = set()

        def _apply_close(mode: str, sym: str, pos: Position, exit_price: float, reason: str, qty_close: float = None):
            qty_close = float(qty_close if qty_close is not None else getattr(pos, 'quantity', 0.0) or 0.0)
            q0_at_call = float(getattr(pos, 'quantity', 0.0) or 0.0)
            if qty_close <= 0:
                return

            # Deduplicate (same scan) - protects against double-processing demo/shadow, re-entrant calls, etc.
            try:
                et = getattr(pos, 'entry_time', None)
                et_s = et.isoformat() if hasattr(et, 'isoformat') else str(et)
            except Exception:
                et_s = ''
            key = (
                str(mode), str(sym), str(reason), str(getattr(pos, 'side', '')), et_s,
                float(getattr(pos, 'entry_price', 0.0) or 0.0),
                float(qty_close or 0.0)
            )
            if key in closed_this_scan:
                return
            closed_this_scan.add(key)
            # clone-like pnl on partial
            tmp = Position(
                symbol=pos.symbol,
                side=pos.side,
                entry_price=pos.entry_price,
                quantity=qty_close,
                leverage=int(getattr(pos, 'leverage', 1) or 1),
                stop_loss=float(getattr(pos, 'stop_loss', 0.0) or 0.0),
                take_profit=float(getattr(pos, 'take_profit', 0.0) or 0.0),
                entry_time=getattr(pos, 'entry_time', None) or datetime.now(),
            )
            pnl_gross, fee, pnl_net = self._calc_futures_pnl_net(tmp, exit_price)
            # Persist closed trade into the ML brain for self-training / self-evolve.
            # Critical link: without this, retraining never triggers and the brain never adapts.
            try:
                feat_closed = getattr(pos, 'features', None) or {}
                if not isinstance(feat_closed, dict) or not feat_closed:
                    try:
                        feat_closed = _get_features(sym) or {}
                    except Exception:
                        feat_closed = {}
                # Duration
                try:
                    dur_s = (datetime.now() - (getattr(pos, 'entry_time', None) or datetime.now())).total_seconds()
                except Exception:
                    dur_s = 0.0
                # Notional-based pct (approx) - stable across lever/qty
                try:
                    lev = int(getattr(pos, 'leverage', 1) or 1)
                    notional = abs(float(entry) * float(qty_close) * float(max(1, lev)))
                    pnl_pct = float(pnl_net) / max(notional, 1e-9)
                except Exception:
                    pnl_pct = 0.0

                trade_row = {
                    'timestamp': datetime.now().isoformat(),
                    'symbol': str(sym),
                    'side': str(getattr(pos, 'side', '') or ''),
                    'entry_price': float(entry),
                    'exit_price': float(exit_price),
                    'quantity': float(qty_close),
                    'pnl': float(pnl_net),
                    'pnl_pct': float(pnl_pct),
                    'duration_seconds': float(dur_s),
                    'exit_reason': str(reason),
                    'regime': str(getattr(pos, 'regime', '') or ''),
                    'confidence': float(getattr(pos, 'confidence', 0.0) or 0.0),
                    'features': feat_closed,
                }
                brain = getattr(self, 'brain', None)
                if brain is not None:
                    if hasattr(brain, 'log_trade'):
                        brain.log_trade(trade_row)
                    try:
                        if hasattr(brain, 'update_edge'):
                            brain.update_edge(trade_row)
                        if hasattr(brain, 'update_policy'):
                            brain.update_policy(trade_row)
                    except Exception:
                        pass

                    if hasattr(brain, 'add_trade_to_buffer'):
                        brain.add_trade_to_buffer(trade_row)
            except Exception:
                pass


            # update account state (single source of truth)
            try:
                if hasattr(self, 'state'):
                    self.state.apply_trade_close(mode, pnl_net, {
                        'event_type': ('PARTIAL' if float(qty_close) < float(q0_at_call) - 1e-12 else 'CLOSE'),
                        'symbol': sym,
                        'reason': reason,
                        'side': pos.side,
                        'entry': float(pos.entry_price),
                        'exit': float(exit_price),
                        'qty': float(qty_close),
                        'lev': int(getattr(pos, 'leverage', 1) or 1),
                        'pnl_gross': float(pnl_gross),
                        'fee': float(fee),
                    })
            
                # record last close for cooldown / no-flip control (mode-agnostic)
                try:
                    if not hasattr(self, '_last_close'):
                        self._last_close = {}
                    self._last_close[str(sym)] = {
                        'ts': datetime.now().timestamp(),
                        'side': str(getattr(pos, 'side', '')),
                        'reason': str(reason),
                        'pnl_net': float(pnl_net),
                        'qty': float(qty_close),
                        'exit': float(exit_price),
                    }
                except Exception:
                    pass

            except Exception:
                logging.exception('Failed to update account state on close')

            # update risk guard (global) to reflect realized pnl
            try:
                trade_id = f"live:{sym}:{getattr(pos,'entry_time',None)}:{reason}:{float(exit_px):.8f}:{float(executed_qty):.8f}"
                self.risk_guard.update_after_trade(sym, pnl_net, fee_est=fee, trade_id=trade_id)
            except Exception:
                pass

            try:
                self._post_trade_performance_tune()
            except Exception:
                pass

            _journal(mode, 'CLOSE', sym, {
                'reason': reason,
                'side': pos.side,
                'entry': float(pos.entry_price),
                'exit': float(exit_price),
                'qty': float(qty_close),
                'lev': int(getattr(pos, 'leverage', 1) or 1),
                'pnl_gross': float(pnl_gross),
                'fee': float(fee),
                'pnl_net': float(pnl_net),
            })

            logging.info(f"[{mode.upper()}] Close {sym} {reason} qty={qty_close:.8f} pnl_net=${pnl_net:.2f} (fee~${fee:.2f})")

            # --- State update: reduce/remove position to prevent duplicate CLOSE loops ---
            try:
                new_qty = float(getattr(pos, 'quantity', 0.0) or 0.0) - float(qty_close or 0.0)
                # Treat as full close if qty is effectively zero OR we closed >= original qty at call
                if new_qty <= 1e-12 or float(q0_at_call) <= float(qty_close) + 1e-12:
                    try:
                        pos.quantity = 0.0
                        pos.is_closed = True
                        pos.exit_time = datetime.now()
                    except Exception:
                        pass
                    m0 = str(mode or '').lower()
                    if m0 == 'paper':
                        try:
                            ppos = getattr(self.execution, 'positions', None)
                            if isinstance(ppos, dict):
                                ppos.pop(sym, None)
                        except Exception:
                            pass
                    elif m0 == 'demo':
                        try:
                            self.demo_positions.pop(sym, None)
                        except Exception:
                            pass
                    elif m0 == 'shadow':
                        try:
                            self.shadow_positions.pop(sym, None)
                        except Exception:
                            pass
                else:
                    try:
                        pos.quantity = float(new_qty)
                    except Exception:
                        pass
            except Exception:
                pass


        def _apply_close_live(sym: str, pos: Position, reason: str, qty_close: float = None):
            """Close LIVE position on-exchange and journal realized PnL.

            - Uses reduceOnly MARKET close via SmartExecutionEngine.close_live_position
            - Verifies position decreases / goes flat
            """
            qty_close = float(qty_close if qty_close is not None else getattr(pos, 'quantity', 0.0) or 0.0)
            q0_at_call = float(getattr(pos, 'quantity', 0.0) or 0.0)
            if qty_close <= 0:
                return False
            try:
                res = self.execution.close_live_position(sym, qty=qty_close if qty_close < float(getattr(pos, 'quantity', qty_close) or qty_close) else None)
            except Exception as e:
                self.circuit.record_order_error(f'live_close:{e}')
                if self.circuit.should_trip():
                    self._trip_live_killswitch(f'live_close_error: {self.circuit.last_trip_reason}')
                raise

            try:
                exit_px = float(res.get('avgPrice', 0.0) or 0.0)
            except Exception:
                exit_px = 0.0
            if exit_px <= 0:
                # fallback to last price if avg missing
                try:
                    exit_px = float(self.market_data.get_current_price(sym) or 0.0)
                except Exception:
                    exit_px = 0.0

            try:
                executed_qty = float(res.get('executedQty', 0.0) or 0.0)
            except Exception:
                executed_qty = qty_close
            if executed_qty <= 0:
                executed_qty = qty_close

            # Compute pnl on executed quantity
            tmp = Position(
                symbol=pos.symbol,
                side=pos.side,
                entry_price=float(getattr(pos, 'entry_price', 0.0) or 0.0),
                quantity=float(executed_qty),
                leverage=int(getattr(pos, 'leverage', 1) or 1),
                stop_loss=float(getattr(pos, 'stop_loss', 0.0) or 0.0),
                take_profit=float(getattr(pos, 'take_profit', 0.0) or 0.0),
                entry_time=getattr(pos, 'entry_time', None) or datetime.now(),
            )
            pnl_gross, fee, pnl_net = self._calc_futures_pnl_net(tmp, exit_px)

            try:
                if hasattr(self, 'state'):
                    self.state.apply_trade_close('live', pnl_net, {
                        'symbol': sym,
                        'reason': reason,
                        'side': pos.side,
                        'entry': float(getattr(pos, 'entry_price', 0.0) or 0.0),
                        'exit': float(exit_px),
                        'qty': float(executed_qty),
                        'lev': int(getattr(pos, 'leverage', 1) or 1),
                        'pnl_gross': float(pnl_gross),
                        'fee': float(fee),
                        'orderId': int(res.get('orderId', 0) or 0),
                    })
            except Exception:
                logging.exception('Failed to update live state on close')

            try:
                self.risk_guard.update_after_trade(sym, pnl_net)
            except Exception:
                pass

            try:
                self._post_trade_performance_tune()
            except Exception:
                pass

            try:
                if getattr(self, 'metrics', None) is not None:
                    self.metrics.record_trade_close(sym, pnl_net, reason, meta={'mode': 'live'})
            except Exception:
                pass

            _journal('live', 'CLOSE', sym, {
                'reason': reason,
                'side': pos.side,
                'entry': float(getattr(pos, 'entry_price', 0.0) or 0.0),
                'exit': float(exit_px),
                'qty': float(executed_qty),
                'pnl_gross': float(pnl_gross),
                'fee': float(fee),
                'pnl_net': float(pnl_net),
                'orderId': int(res.get('orderId', 0) or 0),
            })

            logging.warning(f"[LIVE] Close {sym} {reason} qty={executed_qty:.8f} pnl_net=${pnl_net:.2f}")
            return True

        unrealized_sum = {'demo': 0.0, 'shadow': 0.0, 'paper': 0.0}

        def _update_unrealized(mode: str, sym: str, pos: Position, price: float):
            _, _, pnl_net = self._calc_futures_pnl_net(pos, price)
            if mode in unrealized_sum:
                unrealized_sum[mode] += float(pnl_net or 0.0)

        # v10: unrealized is accumulated in-memory per scan and then written once via AccountStateManager.

        # ===== DEMO positions =====
        demo_items = list((getattr(self, 'demo_positions', {}) or {}).items())
        for sym, pos in demo_items:
            try:
                price = float(self.market_data.get_current_price(sym) or 0.0)
            except Exception:
                price = 0.0
            if price <= 0:
                continue

            mode = 'demo'

            # track MAE/MFE
            try:
                pos.highest_price = max(float(getattr(pos, 'highest_price', pos.entry_price) or pos.entry_price), price)
                pos.lowest_price = min(float(getattr(pos, 'lowest_price', pos.entry_price) or pos.entry_price), price)
            except Exception:
                pass

            sl = float(getattr(pos, 'stop_loss', 0.0) or 0.0)
            entry = float(getattr(pos, 'entry_price', 0.0) or 0.0)
            qty = float(getattr(pos, 'quantity', 0.0) or 0.0)
            if entry <= 0 or qty <= 0:
                continue

            init_risk = float(getattr(pos, 'initial_risk', 0.0) or 0.0)
            sl_dist = init_risk if init_risk > 0 else (abs(entry - sl) if sl else 0.0)
            # unrealized update
            _update_unrealized(mode, sym, pos, price)

            # RR estimation
            rr = 0.0
            if sl_dist > 0:
                if str(pos.side).upper() == 'LONG':
                    rr = (price - entry) / sl_dist
                else:
                    rr = (entry - price) / sl_dist


            # Reversal-take: lock profits before the market flips.
            # Track best favorable price and close if pullback exceeds threshold after reaching min RR.
            try:
                cfg_exit = self.config.get('exit_engine', {}) if isinstance(getattr(self, 'config', {}), dict) else {}
                if bool(cfg_exit.get('reversal_take_enabled', True)):
                    rt_min_rr = float(cfg_exit.get('reversal_take_min_rr', 0.9) or 0.9)
                    rt_pull_r = float(cfg_exit.get('reversal_take_pullback_r', 0.55) or 0.55)
                    if sl_dist > 0 and rr >= rt_min_rr:
                        side_u = str(getattr(pos, 'side', '') or '').upper()
                        best = float(getattr(pos, 'best_favorable_price', entry) or entry)
                        if side_u == 'LONG':
                            best = max(best, float(price))
                            pb_r = (best - float(price)) / float(sl_dist)
                        else:
                            best = min(best, float(price))
                            pb_r = (float(price) - best) / float(sl_dist)
                        pos.best_favorable_price = float(best)
                        if pb_r >= rt_pull_r:
                            _apply_close(mode, sym, pos, float(price), 'REVERSAL_TAKE')
                            try:
                                _del_pos(mode, sym)
                            except Exception:
                                pass
                            continue
            except Exception:
                pass


            try:
                feat0 = _get_features(sym)
                _maybe_pyramid('paper', sym, pos, price, rr, feat0)
            except Exception:
                pass

            # Add-to-winner when the move proves itself (expectancy booster)
            try:
                feat = _get_features(sym)
                _maybe_pyramid(mode, sym, pos, price, rr, feat)
            except Exception:
                pass
            # time stop (smart, mode-agnostic)
            if _smart_time_stop(mode, sym, pos, price, rr, entry, sl_dist):
                continue

                # Otherwise: reduce exposure to avoid fee-death, keep runner.
                try:
                    q0 = float(getattr(pos, 'quantity', 0.0) or 0.0)
                    if q0 > 0:
                        # partial size depends on regime
                        part_pct = 0.35 if is_sideway else 0.25
                        q_close = max(0.0, min(q0 * part_pct, q0 - 1e-12))
                        # only do partial if meaningful
                        if q_close > 0.0 and q_close < q0:
                            # Guard: never time-stop partial at a net loss (fee-bleed). Let SL/TP handle it.
                            try:
                                _, _, _pnl_net_ts = self._calc_futures_pnl_net(pos, float(price))
                                if float(_pnl_net_ts or 0.0) <= 0.0:
                                    try:
                                        pos.time_stop_snooze_until = datetime.now() + timedelta(minutes=6.0 if is_sideway else 12.0)
                                    except Exception:
                                        pass
                                    return False
                            except Exception:
                                pass
                            _apply_close(mode, sym, pos, price, 'TIME_STOP_PARTIAL', qty_close=q_close)
                            try:
                                pos.quantity = float(q0 - q_close)
                            except Exception:
                                pass
                            # protect stop: move toward BE to avoid bleeding
                            try:
                                if sl_dist > 0:
                                    be_buf_r = float((self._get_exit_cfg().get('be_buffer_r', 0.12)) or 0.12)
                                    if side_u == 'LONG':
                                        pos.stop_loss = max(float(getattr(pos,'stop_loss',0.0) or 0.0), float(entry) + float(be_buf_r) * float(sl_dist))
                                    else:
                                        pos.stop_loss = min(float(getattr(pos,'stop_loss',entry) or entry), float(entry) - float(be_buf_r) * float(sl_dist))
                            except Exception:
                                pass
                except Exception:
                    pass

            # early loss cut (reduce tail losses): cut partial when thesis is likely invalidated.
            # early loss cut: ONLY cut when the thesis is likely invalidated (avoid "right direction but got cut").
            # We require multi-signal invalidation + minimum adverse excursion. This reduces churn and fee-death.
            invalidated = False
            try:
                feat = _get_features(sym) or {}
                htf_dir = int(feat.get('htf_dir', 0) or 0)
                htf_strength = float(feat.get('htf_strength', 0.0) or 0.0)
                trend_strength = float(feat.get('trend_strength', 0.0) or 0.0)
                macd = float(feat.get('macd', 0.0) or 0.0)
                macd_sig = float(feat.get('macd_signal', 0.0) or 0.0)
                ma20 = float(feat.get('ma20', entry) or entry)
                rsi = float(feat.get('rsi', 50.0) or 50.0)
                side = str(getattr(pos, 'side', '') or '').upper()

                if side == 'LONG':
                    invalidated = (
                        (htf_dir < 0 and htf_strength >= 0.55) or
                        (trend_strength < -0.0015) or
                        (macd < macd_sig and rsi < 48.0)
                    ) and (price < ma20)
                else:
                    invalidated = (
                        (htf_dir > 0 and htf_strength >= 0.55) or
                        (trend_strength > 0.0015) or
                        (macd > macd_sig and rsi > 52.0)
                    ) and (price > ma20)
            except Exception:
                invalidated = False

            # Fee-aware adverse excursion gate: don't early-cut unless price moved enough to matter (avoid churn/fees)
            fee_gate_ok = False
            try:
                fee_rt = 2.0 * float(getattr(self, 'fee_rate', 0.0004) or 0.0004)
            except Exception:
                fee_rt = 0.0004 * 2.0
            try:
                slip = float(getattr(self, 'slippage_pct', 0.0006) or 0.0006)
            except Exception:
                slip = 0.0006
            try:
                cost_pct = fee_rt + (2.0 * slip)
                adverse_pct = abs(float(price) - float(entry)) / max(float(entry), 1e-12)
                # require: at least ~1.25x friction AND close to -1R move
                fee_gate_ok = (adverse_pct >= (1.25 * cost_pct))
            except Exception:
                fee_gate_ok = False

            if sl_dist > 0 and invalidated and fee_gate_ok and rr <= float(early_cut_rr) and not bool(getattr(pos, 'loss_cut_done', False)) and age_min >= early_cut_min_age:
                close_qty = qty * max(0.0, min(1.0, early_cut_pct))
                close_qty = float(close_qty)
                if close_qty > 0.0 and close_qty < qty:
                    _apply_close(mode, sym, pos, price, 'EARLY_CUT_PARTIAL', qty_close=close_qty)
                    pos.quantity = qty - close_qty
                    pos.loss_cut_done = True
                    # cap remaining worst-case loss to loss_cap_remaining_r * R by tightening SL on the remainder
                    try:
                        if str(pos.side).upper() == 'LONG':
                            pos.stop_loss = max(float(pos.stop_loss or 0.0), float(entry) - float(loss_cap_remaining_r) * float(sl_dist))
                        else:
                            pos.stop_loss = min(float(pos.stop_loss or entry), float(entry) + float(loss_cap_remaining_r) * float(sl_dist))
                    except Exception:
                        pass
                    _journal(mode, 'PARTIAL', sym, {'reason': 'EARLY_CUT_PARTIAL', 'remaining_qty': float(getattr(pos,'quantity',0.0) or 0.0)})
                    qty = float(getattr(pos, 'quantity', qty) or qty)
                else:
                    _apply_close(mode, sym, pos, price, 'EARLY_CUT')
                    try:
                        _del_pos(mode, sym)
                    except Exception:
                        pass
                    continue

            # partial TP
            if sl_dist > 0 and rr >= tp1_rr and not bool(getattr(pos, 'tp1_done', False)):
                close_qty = qty * max(0.0, min(1.0, tp1_pct))
                if close_qty > 0 and close_qty < qty:
                    _apply_close(mode, sym, pos, price, 'TP1_PARTIAL', qty_close=close_qty)
                    pos.quantity = qty - close_qty
                    pos.tp1_done = True
                    _journal(mode, 'PARTIAL', sym, {'remaining_qty': float(pos.quantity)})
                    # runner mode: remove the hard TP after TP1 so winners can run with trailing.
                    if runner_disable_hard_tp:
                        try:
                            pos.take_profit = 0.0
                        except Exception:
                            pass


            # breakeven
            if sl_dist > 0 and rr >= be_rr and not bool(getattr(pos, 'be_done', False)):
                # move SL to entry +/- small buffer (fees)
                buf = self._breakeven_buffer(entry, qty)
                if str(pos.side).upper() == 'LONG':
                    pos.stop_loss = max(pos.stop_loss or 0.0, entry + buf)
                else:
                    pos.stop_loss = min(pos.stop_loss or entry, entry - buf)
                pos.be_done = True

            # trailing (runner-friendly): use ATR-based distance (wider when needed), not only percent.
            if sl_dist > 0 and rr >= trail_start_rr:
                side = str(pos.side).upper()
                td = _trail_dist(sym, price, bool(getattr(pos, 'tp1_done', False)))
                if side == 'LONG':
                    trail_sl = float(getattr(pos, 'highest_price', price) or price) - td
                    pos.stop_loss = max(float(pos.stop_loss or 0.0), float(trail_sl))
                else:
                    trail_sl = float(getattr(pos, 'lowest_price', price) or price) + td
                    # for short, SL should move down (more favorable), so take min
                    pos.stop_loss = min(float(pos.stop_loss or entry), float(trail_sl))

            # hard SL/TP hit
            hit = None
            tp = float(getattr(pos, 'take_profit', 0.0) or 0.0)
            side = str(pos.side).upper()
            if side == 'LONG':
                if pos.stop_loss and price <= float(pos.stop_loss):
                    # Stop-hunt defense: during initial arming window, ignore wick touches for LOSS SL
                    ignore_sl = False
                    try:
                        shd = (self.config.get('stop_hunt_defense', {}) or {}) if isinstance(getattr(self, 'config', None), dict) else {}
                        arm_min = float(shd.get('arm_minutes', 1.2) or 1.2)
                        extra_atr = float(shd.get('extra_atr', 0.25) or 0.25)
                        age_m = (datetime.now() - (getattr(pos, 'entry_time', None) or datetime.now())).total_seconds() / 60.0
                        if age_m < arm_min:
                            feat_sl = _get_features(sym) or {}
                            atr = float(feat_sl.get('atr', feat_sl.get('atr_14', 0.0)) or 0.0)
                            if atr > 0:
                                slp = float(pos.stop_loss)
                                if slp < float(entry) and float(price) > (slp - extra_atr * atr):
                                    ignore_sl = True
                    except Exception:
                        ignore_sl = False

                    if not ignore_sl:
                        # if SL has moved to >= entry, it's a protective stop (BE/trailing), not a true loss-cut
                        tag_sl = 'PROTECT_STOP' if float(pos.stop_loss) >= entry else 'SL'
                        hit = (tag_sl, float(pos.stop_loss))
                elif tp and price >= tp:
                    hit = ('TP', tp)
            else:
                if pos.stop_loss and price >= float(pos.stop_loss):
                    # Stop-hunt defense: during initial arming window, ignore wick touches for LOSS SL
                    ignore_sl = False
                    try:
                        shd = (self.config.get('stop_hunt_defense', {}) or {}) if isinstance(getattr(self, 'config', None), dict) else {}
                        arm_min = float(shd.get('arm_minutes', 1.2) or 1.2)
                        extra_atr = float(shd.get('extra_atr', 0.25) or 0.25)
                        age_m = (datetime.now() - (getattr(pos, 'entry_time', None) or datetime.now())).total_seconds() / 60.0
                        if age_m < arm_min:
                            feat_sl = _get_features(sym) or {}
                            atr = float(feat_sl.get('atr', feat_sl.get('atr_14', 0.0)) or 0.0)
                            if atr > 0:
                                slp = float(pos.stop_loss)
                                if slp > float(entry) and float(price) < (slp + extra_atr * atr):
                                    ignore_sl = True
                    except Exception:
                        ignore_sl = False

                    if not ignore_sl:
                        tag_sl = 'PROTECT_STOP' if float(pos.stop_loss) <= entry else 'SL'
                        hit = (tag_sl, float(pos.stop_loss))
                elif tp and price <= tp:
                    hit = ('TP', tp)

            if hit:
                tag, exit_px = hit
                _apply_close(mode, sym, pos, exit_px, tag)
                try:
                    _del_pos(mode, sym)
                except Exception:
                    pass

        # ===== SHADOW positions =====
        shadow_items = list((getattr(self, 'shadow_positions', {}) or {}).items())
        for sym, pos in shadow_items:
            try:
                price = float(self.market_data.get_current_price(sym) or 0.0)
            except Exception:
                price = 0.0
            if price <= 0:
                continue

            mode = 'shadow'

            # track MAE/MFE
            try:
                pos.highest_price = max(float(getattr(pos, 'highest_price', pos.entry_price) or pos.entry_price), price)
                pos.lowest_price = min(float(getattr(pos, 'lowest_price', pos.entry_price) or pos.entry_price), price)
            except Exception:
                pass

            sl = float(getattr(pos, 'stop_loss', 0.0) or 0.0)
            entry = float(getattr(pos, 'entry_price', 0.0) or 0.0)
            qty = float(getattr(pos, 'quantity', 0.0) or 0.0)
            if entry <= 0 or qty <= 0:
                continue

            init_risk = float(getattr(pos, 'initial_risk', 0.0) or 0.0)
            sl_dist = init_risk if init_risk > 0 else (abs(entry - sl) if sl else 0.0)
            _update_unrealized(mode, sym, pos, price)

            rr = 0.0
            if sl_dist > 0:
                if str(pos.side).upper() == 'LONG':
                    rr = (price - entry) / sl_dist
                else:
                    rr = (entry - price) / sl_dist

            # Reversal-take: lock profits before the market flips.
            # Track best favorable price and close if pullback exceeds threshold after reaching min RR.
            try:
                cfg_exit = self.config.get('exit_engine', {}) if isinstance(getattr(self, 'config', {}), dict) else {}
                if bool(cfg_exit.get('reversal_take_enabled', True)):
                    rt_min_rr = float(cfg_exit.get('reversal_take_min_rr', 0.9) or 0.9)
                    rt_pull_r = float(cfg_exit.get('reversal_take_pullback_r', 0.55) or 0.55)
                    if sl_dist > 0 and rr >= rt_min_rr:
                        side_u = str(getattr(pos, 'side', '') or '').upper()
                        best = float(getattr(pos, 'best_favorable_price', entry) or entry)
                        if side_u == 'LONG':
                            best = max(best, float(price))
                            pb_r = (best - float(price)) / float(sl_dist)
                        else:
                            best = min(best, float(price))
                            pb_r = (float(price) - best) / float(sl_dist)
                        pos.best_favorable_price = float(best)
                        if pb_r >= rt_pull_r:
                            _apply_close(mode, sym, pos, float(price), 'REVERSAL_TAKE')
                            try:
                                _del_pos(mode, sym)
                            except Exception:
                                pass
                            continue
            except Exception:
                pass



            # Add-to-winner (shadow)
            try:
                feat0 = _get_features(sym)
                _maybe_pyramid(mode, sym, pos, price, rr, feat0)
            except Exception:
                pass


            # Add-to-winner when move proves itself (shadow)
            try:
                feat0 = _get_features(sym)
                _maybe_pyramid(mode, sym, pos, price, rr, feat0)
            except Exception:
                pass
            # time stop (smart, mode-agnostic)
            if _smart_time_stop(mode, sym, pos, price, rr, entry, sl_dist):
                continue

            # partial TP
            if sl_dist > 0 and rr >= tp1_rr and not bool(getattr(pos, 'tp1_done', False)):
                close_qty = qty * max(0.0, min(1.0, tp1_pct))
                if close_qty > 0 and close_qty < qty:
                    _apply_close(mode, sym, pos, price, 'TP1_PARTIAL', qty_close=close_qty)
                    pos.quantity = qty - close_qty
                    pos.tp1_done = True
                    _journal(mode, 'PARTIAL', sym, {'remaining_qty': float(pos.quantity)})
                    # runner mode: remove the hard TP after TP1 so winners can run with trailing.
                    if runner_disable_hard_tp:
                        try:
                            pos.take_profit = 0.0
                        except Exception:
                            pass


            # breakeven
            if sl_dist > 0 and rr >= be_rr and not bool(getattr(pos, 'be_done', False)):
                buf = self._breakeven_buffer(entry, qty)
                if str(pos.side).upper() == 'LONG':
                    pos.stop_loss = max(pos.stop_loss or 0.0, entry + buf)
                else:
                    pos.stop_loss = min(pos.stop_loss or entry, entry - buf)
                pos.be_done = True

            # trailing
            if sl_dist > 0 and rr >= trail_start_rr:
                side = str(pos.side).upper()
                if side == 'LONG':
                    trail_sl = float(getattr(pos, 'highest_price', price) or price) * (1.0 - trail_pct)
                    pos.stop_loss = max(float(pos.stop_loss or 0.0), trail_sl)
                else:
                    trail_sl = float(getattr(pos, 'lowest_price', price) or price) * (1.0 + trail_pct)
                    pos.stop_loss = min(float(pos.stop_loss or entry), trail_sl)

            # hard SL/TP hit
            hit = None
            tp = float(getattr(pos, 'take_profit', 0.0) or 0.0)
            side = str(pos.side).upper()
            if side == 'LONG':
                if pos.stop_loss and price <= float(pos.stop_loss):
                    # Stop-hunt defense: during initial arming window, ignore wick touches for LOSS SL
                    ignore_sl = False
                    try:
                        shd = (self.config.get('stop_hunt_defense', {}) or {}) if isinstance(getattr(self, 'config', None), dict) else {}
                        arm_min = float(shd.get('arm_minutes', 1.2) or 1.2)
                        extra_atr = float(shd.get('extra_atr', 0.25) or 0.25)
                        age_m = (datetime.now() - (getattr(pos, 'entry_time', None) or datetime.now())).total_seconds() / 60.0
                        if age_m < arm_min:
                            feat_sl = _get_features(sym) or {}
                            atr = float(feat_sl.get('atr', feat_sl.get('atr_14', 0.0)) or 0.0)
                            if atr > 0:
                                slp = float(pos.stop_loss)
                                if slp < float(entry) and float(price) > (slp - extra_atr * atr):
                                    ignore_sl = True
                    except Exception:
                        ignore_sl = False

                    if not ignore_sl:
                        # if SL has moved to >= entry, it's a protective stop (BE/trailing), not a true loss-cut
                        tag_sl = 'PROTECT_STOP' if float(pos.stop_loss) >= entry else 'SL'
                        hit = (tag_sl, float(pos.stop_loss))
                elif tp and price >= tp:
                    hit = ('TP', tp)
            else:
                if pos.stop_loss and price >= float(pos.stop_loss):
                    # Stop-hunt defense: during initial arming window, ignore wick touches for LOSS SL
                    ignore_sl = False
                    try:
                        shd = (self.config.get('stop_hunt_defense', {}) or {}) if isinstance(getattr(self, 'config', None), dict) else {}
                        arm_min = float(shd.get('arm_minutes', 1.2) or 1.2)
                        extra_atr = float(shd.get('extra_atr', 0.25) or 0.25)
                        age_m = (datetime.now() - (getattr(pos, 'entry_time', None) or datetime.now())).total_seconds() / 60.0
                        if age_m < arm_min:
                            feat_sl = _get_features(sym) or {}
                            atr = float(feat_sl.get('atr', feat_sl.get('atr_14', 0.0)) or 0.0)
                            if atr > 0:
                                slp = float(pos.stop_loss)
                                if slp > float(entry) and float(price) < (slp + extra_atr * atr):
                                    ignore_sl = True
                    except Exception:
                        ignore_sl = False

                    if not ignore_sl:
                        tag_sl = 'PROTECT_STOP' if float(pos.stop_loss) <= entry else 'SL'
                        hit = (tag_sl, float(pos.stop_loss))
                elif tp and price <= tp:
                    hit = ('TP', tp)

            if hit:
                tag, exit_px = hit
                _apply_close(mode, sym, pos, exit_px, tag)
                try:
                    del self.shadow_positions[sym]
                except Exception:
                    pass

        # ===== PAPER positions =====
        paper_pos = getattr(self.execution, 'positions', {}) or {}
        for sym, pos in list(paper_pos.items()):
            try:
                price = float(self.market_data.get_current_price(sym) or 0.0)
            except Exception:
                price = 0.0
            if price <= 0:
                continue

            mode = str(getattr(pos, 'mode', 'paper') or 'paper')
            mode = 'paper' if mode not in ['paper','live'] else mode
            if mode != 'paper':
                continue

            # update MFE/MAE
            try:
                pos.highest_price = max(float(getattr(pos, 'highest_price', pos.entry_price) or pos.entry_price), price)
                pos.lowest_price = min(float(getattr(pos, 'lowest_price', pos.entry_price) or pos.entry_price), price)
            except Exception:
                pass

            sl = float(getattr(pos, 'stop_loss', 0.0) or 0.0)
            entry = float(getattr(pos, 'entry_price', 0.0) or 0.0)
            qty = float(getattr(pos, 'quantity', 0.0) or 0.0)
            if entry <= 0 or qty <= 0:
                continue

            init_risk = float(getattr(pos, 'initial_risk', 0.0) or 0.0)
            sl_dist = init_risk if init_risk > 0 else (abs(entry - sl) if sl else 0.0)
            _update_unrealized('paper', sym, pos, price)

            rr = 0.0
            if sl_dist > 0:
                if str(pos.side).upper() == 'LONG':
                    rr = (price - entry) / sl_dist
                else:
                    rr = (entry - price) / sl_dist

            # Reversal-take: lock profits before the market flips.
            # Track best favorable price and close if pullback exceeds threshold after reaching min RR.
            try:
                cfg_exit = self.config.get('exit_engine', {}) if isinstance(getattr(self, 'config', {}), dict) else {}
                if bool(cfg_exit.get('reversal_take_enabled', True)):
                    rt_min_rr = float(cfg_exit.get('reversal_take_min_rr', 0.9) or 0.9)
                    rt_pull_r = float(cfg_exit.get('reversal_take_pullback_r', 0.55) or 0.55)
                    if sl_dist > 0 and rr >= rt_min_rr:
                        side_u = str(getattr(pos, 'side', '') or '').upper()
                        best = float(getattr(pos, 'best_favorable_price', entry) or entry)
                        if side_u == 'LONG':
                            best = max(best, float(price))
                            pb_r = (best - float(price)) / float(sl_dist)
                        else:
                            best = min(best, float(price))
                            pb_r = (float(price) - best) / float(sl_dist)
                        pos.best_favorable_price = float(best)
                        if pb_r >= rt_pull_r:
                            _apply_close(mode, sym, pos, float(price), 'REVERSAL_TAKE')
                            try:
                                _del_pos(mode, sym)
                            except Exception:
                                pass
                            continue
            except Exception:
                pass



            # Add-to-winner (paper)
            try:
                feat0 = _get_features(sym)
                _maybe_pyramid(mode, sym, pos, price, rr, feat0)
            except Exception:
                pass
            # time stop (smart, mode-agnostic)
            if _smart_time_stop(mode, sym, pos, price, rr, entry, sl_dist):
                continue

            # partial TP
            if sl_dist > 0 and rr >= tp1_rr and not bool(getattr(pos, 'tp1_done', False)):
                close_qty = qty * max(0.0, min(1.0, tp1_pct))
                if close_qty > 0 and close_qty < qty:
                    _apply_close('paper', sym, pos, price, 'TP1_PARTIAL', qty_close=close_qty)
                    pos.quantity = qty - close_qty
                    pos.tp1_done = True
                    # runner mode: remove the hard TP after TP1 so winners can run with trailing.
                    if runner_disable_hard_tp:
                        try:
                            pos.take_profit = 0.0
                        except Exception:
                            pass


            # breakeven
            if sl_dist > 0 and rr >= be_rr and not bool(getattr(pos, 'be_done', False)):
                buf = self._breakeven_buffer(entry, qty)
                if str(pos.side).upper() == 'LONG':
                    pos.stop_loss = max(float(pos.stop_loss or 0.0), entry + buf)
                else:
                    pos.stop_loss = min(float(pos.stop_loss or entry), entry - buf)
                pos.be_done = True

            # trailing
            if sl_dist > 0 and rr >= trail_start_rr:
                side = str(pos.side).upper()
                if side == 'LONG':
                    trail_sl = float(getattr(pos, 'highest_price', price) or price) * (1.0 - trail_pct)
                    pos.stop_loss = max(float(pos.stop_loss or 0.0), trail_sl)
                else:
                    trail_sl = float(getattr(pos, 'lowest_price', price) or price) * (1.0 + trail_pct)
                    pos.stop_loss = min(float(pos.stop_loss or entry), trail_sl)

            # hard SL/TP hit
            hit = None
            tp = float(getattr(pos, 'take_profit', 0.0) or 0.0)
            side = str(pos.side).upper()
            if side == 'LONG':
                if pos.stop_loss and price <= float(pos.stop_loss):
                    # Stop-hunt defense: during initial arming window, ignore wick touches for LOSS SL
                    ignore_sl = False
                    try:
                        shd = (self.config.get('stop_hunt_defense', {}) or {}) if isinstance(getattr(self, 'config', None), dict) else {}
                        arm_min = float(shd.get('arm_minutes', 1.2) or 1.2)
                        extra_atr = float(shd.get('extra_atr', 0.25) or 0.25)
                        age_m = (datetime.now() - (getattr(pos, 'entry_time', None) or datetime.now())).total_seconds() / 60.0
                        if age_m < arm_min:
                            feat_sl = _get_features(sym) or {}
                            atr = float(feat_sl.get('atr', feat_sl.get('atr_14', 0.0)) or 0.0)
                            if atr > 0:
                                slp = float(pos.stop_loss)
                                if slp < float(entry) and float(price) > (slp - extra_atr * atr):
                                    ignore_sl = True
                    except Exception:
                        ignore_sl = False

                    if not ignore_sl:
                        # if SL has moved to >= entry, it's a protective stop (BE/trailing), not a true loss-cut
                        tag_sl = 'PROTECT_STOP' if float(pos.stop_loss) >= entry else 'SL'
                        hit = (tag_sl, float(pos.stop_loss))
                elif tp and price >= tp:
                    hit = ('TP', tp)
            else:
                if pos.stop_loss and price >= float(pos.stop_loss):
                    # Stop-hunt defense: during initial arming window, ignore wick touches for LOSS SL
                    ignore_sl = False
                    try:
                        shd = (self.config.get('stop_hunt_defense', {}) or {}) if isinstance(getattr(self, 'config', None), dict) else {}
                        arm_min = float(shd.get('arm_minutes', 1.2) or 1.2)
                        extra_atr = float(shd.get('extra_atr', 0.25) or 0.25)
                        age_m = (datetime.now() - (getattr(pos, 'entry_time', None) or datetime.now())).total_seconds() / 60.0
                        if age_m < arm_min:
                            feat_sl = _get_features(sym) or {}
                            atr = float(feat_sl.get('atr', feat_sl.get('atr_14', 0.0)) or 0.0)
                            if atr > 0:
                                slp = float(pos.stop_loss)
                                if slp > float(entry) and float(price) < (slp + extra_atr * atr):
                                    ignore_sl = True
                    except Exception:
                        ignore_sl = False

                    if not ignore_sl:
                        tag_sl = 'PROTECT_STOP' if float(pos.stop_loss) <= entry else 'SL'
                        hit = (tag_sl, float(pos.stop_loss))
                elif tp and price <= tp:
                    hit = ('TP', tp)

            if hit:
                tag, exit_px = hit
                _apply_close('paper', sym, pos, exit_px, tag)
                try:
                    del self.execution.positions[sym]
                except Exception:
                    pass

        # ===== LIVE positions (on-exchange exit engine) =====
        live_pos = getattr(self.execution, 'positions', {}) or {}
        for sym, pos in list(live_pos.items()):
            try:
                mode = str(getattr(pos, 'mode', 'paper') or 'paper')
            except Exception:
                mode = 'paper'
            if mode != 'live':
                continue

            # Live kill-switch check
            live_ctl = self.config.get('live_control', {}) if isinstance(self.config, dict) else {}
            live_enabled = bool(live_ctl.get('enabled', False)) and not bool(getattr(self, 'is_public_mode', False))
            if getattr(self, 'circuit', None) is not None and self.circuit.is_in_cooldown():
                live_enabled = False
            if not live_enabled:
                # Still sync account, but do not auto-close if live disabled (operator decision)
                continue

            try:
                price = float(self.market_data.get_current_price(sym) or 0.0)
            except Exception:
                price = 0.0
            if price <= 0:
                continue

            # update MFE/MAE (local)
            try:
                pos.highest_price = max(float(getattr(pos, 'highest_price', pos.entry_price) or pos.entry_price), price)
                pos.lowest_price = min(float(getattr(pos, 'lowest_price', pos.entry_price) or pos.entry_price), price)
            except Exception:
                pass

            sl = float(getattr(pos, 'stop_loss', 0.0) or 0.0)
            entry = float(getattr(pos, 'entry_price', 0.0) or 0.0)
            qty = float(getattr(pos, 'quantity', 0.0) or 0.0)
            if entry <= 0 or qty <= 0:
                continue

            init_risk = float(getattr(pos, 'initial_risk', 0.0) or 0.0)
            sl_dist = init_risk if init_risk > 0 else (abs(entry - sl) if sl else 0.0)
            rr = 0.0
            if sl_dist > 0:
                if str(pos.side).upper() == 'LONG':
                    rr = (price - entry) / sl_dist

            # Reversal-take: lock profits before the market flips.
            # Track best favorable price and close if pullback exceeds threshold after reaching min RR.
            try:
                cfg_exit = self.config.get('exit_engine', {}) if isinstance(getattr(self, 'config', {}), dict) else {}
                if bool(cfg_exit.get('reversal_take_enabled', True)):
                    rt_min_rr = float(cfg_exit.get('reversal_take_min_rr', 0.9) or 0.9)
                    rt_pull_r = float(cfg_exit.get('reversal_take_pullback_r', 0.55) or 0.55)
                    if sl_dist > 0 and rr >= rt_min_rr:
                        side_u = str(getattr(pos, 'side', '') or '').upper()
                        best = float(getattr(pos, 'best_favorable_price', entry) or entry)
                        if side_u == 'LONG':
                            best = max(best, float(price))
                            pb_r = (best - float(price)) / float(sl_dist)
                        else:
                            best = min(best, float(price))
                            pb_r = (float(price) - best) / float(sl_dist)
                        pos.best_favorable_price = float(best)
                        if pb_r >= rt_pull_r:
                            _apply_close(mode, sym, pos, float(price), 'REVERSAL_TAKE')
                            try:
                                _del_pos(mode, sym)
                            except Exception:
                                pass
                            continue
            except Exception:
                pass



            # time stop (close on-exchange)
            try:
                age_min = (datetime.now() - (getattr(pos, 'entry_time', None) or datetime.now())).total_seconds() / 60.0
            except Exception:
                age_min = 0.0
            # time stop (regime-aware)
            try:
                feat_ts = _get_features(sym) or {}
                atr_ratio = float(feat_ts.get('atr_ratio', 1.0) or 1.0)
                tr_abs = abs(float(feat_ts.get('trend_strength', 0.0) or 0.0))
                is_sideway = (atr_ratio <= sideway_atr_ratio_max) and (tr_abs <= sideway_trend_abs_max)
            except Exception:
                is_sideway = False
            ts_limit = float(time_stop_sideway_min if is_sideway else time_stop_trend_min)
            # time stop (smart, mode-agnostic)
            if _smart_time_stop('live', sym, pos, price, rr, entry, sl_dist):
                continue

            # partial TP1 (close portion)
            # early loss cut (reduce tail losses) - partially close to protect equity.
            fee_gate_ok_live = False
            try:
                fee_rt = 2.0 * float(getattr(self, 'fee_rate', 0.0004) or 0.0004)
            except Exception:
                fee_rt = 0.0008
            try:
                slip = float(getattr(self, 'slippage_pct', 0.0006) or 0.0006)
            except Exception:
                slip = 0.0006
            try:
                cost_pct = fee_rt + (2.0 * slip)
                adverse_pct = abs(float(price) - float(entry)) / max(float(entry), 1e-12)
                fee_gate_ok_live = (adverse_pct >= (1.25 * cost_pct))
            except Exception:
                fee_gate_ok_live = False

            if sl_dist > 0 and fee_gate_ok_live and rr <= early_cut_rr and not bool(getattr(pos, 'loss_cut_done', False)) and age_min >= early_cut_min_age:
                close_qty = qty * max(0.0, min(1.0, early_cut_pct))
                close_qty = float(close_qty)
                if close_qty > 0.0 and close_qty < qty:
                    _apply_close_live(sym, pos, 'EARLY_CUT_PARTIAL', qty_close=close_qty)
                    try:
                        pos.quantity = max(0.0, float(qty) - float(close_qty))
                    except Exception:
                        pass
                    pos.loss_cut_done = True
                    try:
                        if str(pos.side).upper() == 'LONG':
                            pos.stop_loss = max(float(pos.stop_loss or 0.0), float(entry) - float(loss_cap_remaining_r) * float(sl_dist))
                        else:
                            pos.stop_loss = min(float(pos.stop_loss or entry), float(entry) + float(loss_cap_remaining_r) * float(sl_dist))
                    except Exception:
                        pass
                    qty = float(getattr(pos, 'quantity', qty) or qty)
                else:
                    _apply_close_live(sym, pos, 'EARLY_CUT')
                    try:
                        del self.execution.positions[sym]
                    except Exception:
                        pass
                    continue

            # partial TP1 (close portion)
            if sl_dist > 0 and rr >= tp1_rr and not bool(getattr(pos, 'tp1_done', False)):
                close_qty = qty * max(0.0, min(1.0, tp1_pct))
                if close_qty > 0 and close_qty < qty:
                    _apply_close_live(sym, pos, 'TP1_PARTIAL', qty_close=close_qty)
                    try:
                        pos.quantity = max(0.0, qty - close_qty)
                    except Exception:
                        pass
                    pos.tp1_done = True
                    # runner mode: remove the hard TP after TP1 so winners can run with trailing.
                    if runner_disable_hard_tp:
                        try:
                            pos.take_profit = 0.0
                        except Exception:
                            pass


            # breakeven/trailing are local guides; for MARKET exits we simply close when thresholds hit.

            # hard SL/TP hit (close on-exchange)
            hit = None
            tp = float(getattr(pos, 'take_profit', 0.0) or 0.0)
            side = str(pos.side).upper()
            if side == 'LONG':
                if sl and price <= float(sl):
                    hit = ('SL',)
                elif tp and price >= tp:
                    hit = ('TP',)
            else:
                if sl and price >= float(sl):
                    hit = ('SL',)
                elif tp and price <= tp:
                    hit = ('TP',)
            if hit:
                tag = hit[0]
                _apply_close_live(sym, pos, tag)
                try:
                    del self.execution.positions[sym]
                except Exception:
                    pass

        # finalize unrealized per mode
        try:
            if hasattr(self, 'state'):
                for m in ['demo','shadow','paper']:
                    self.state.apply_unrealized(m, float(unrealized_sum.get(m, 0.0) or 0.0))
        except Exception:
            logging.exception('Failed to finalize unrealized')

        # --- Reversal rescue & scratch flatten (DEMO/SHADOW/PAPER) ---
        try:
            if rescue_enabled and (time.time() - float(getattr(self, '_last_rescue_ts', 0.0) or 0.0)) >= float(rescue_cooldown_s):
                # Build per-mode position views
                def _iter_mode_positions(mode: str):
                    ml = str(mode or '').lower()
                    if ml == 'paper':
                        # Prefer execution engine positions for PAPER
                        d = getattr(getattr(self, 'execution_engine', None), 'positions', None) or getattr(self, 'positions', {}) or {}
                        return list(d.items())
                    d = _pos_dict_for_mode(ml)
                    return list(d.items()) if isinstance(d, dict) else []

                # Scratch-close all ONLY when we are truly near break-even (after costs) AND positions are mature.
                # Goal: avoid rapid churn (fee burn) and avoid triggering risk-state escalation from tiny scratches.
                for ml in ('demo', 'shadow', 'paper'):
                    items = _iter_mode_positions(ml)
                    if not items:
                        continue
                    gross = 0.0
                    cost = 0.0
                    now_ts = time.time()
                    # config-driven guards
                    try:
                        min_pos = int(rescue_cfg.get('scratch_all_min_positions', 3) or 3)
                    except Exception:
                        min_pos = 3
                    strict = bool(rescue_cfg.get('scratch_all_strict', True))
                    try:
                        min_age_sec = float(rescue_cfg.get('scratch_all_min_age_sec', 180.0) or 180.0)
                    except Exception:
                        min_age_sec = 180.0
                    try:
                        fee_band_mult = float(rescue_cfg.get('scratch_all_fee_band_mult', 0.8) or 0.8)
                    except Exception:
                        fee_band_mult = 0.8
                    try:
                        winner_guard_mult = float(rescue_cfg.get('scratch_all_winner_guard_fee_mult', 0.6) or 0.6)
                    except Exception:
                        winner_guard_mult = 0.6

                    if strict and len(items) < min_pos:
                        continue

                    oldest_age = 0.0
                    best_unreal = -1e18
                    for sym, pos in items:
                        try:
                            gross += float(getattr(pos, 'unrealized_pnl', 0.0) or 0.0)
                            cost += _roundtrip_cost_usdt(float(getattr(pos, 'entry_price', 0.0) or 0.0), float(getattr(pos, 'quantity', 0.0) or 0.0))
                            best_unreal = max(best_unreal, float(getattr(pos, 'unrealized_pnl', 0.0) or 0.0))
                            # age guard if available
                            ets = getattr(pos, 'entry_ts', None)
                            if ets is None:
                                ets = getattr(pos, 'open_time', None)
                            if ets is not None:
                                try:
                                    age = max(0.0, now_ts - float(ets))
                                    oldest_age = max(oldest_age, age)
                                except Exception:
                                    pass
                        except Exception:
                            pass

                    if cost <= 0:
                        continue
                    net = gross - cost
                    # Require positions to be held for a while (no 1-5s scratch closes).
                    if oldest_age < min_age_sec:
                        continue
                    # Only flatten if net is within a tight band around 0 (after costs).
                    if abs(net) > (cost * fee_band_mult):
                        continue
                    # Don't flatten if we have a meaningful winner (let TP work).
                    if best_unreal > (cost * winner_guard_mult):
                        continue

                    for sym, pos in list(items):
                        try:
                            px = float(self.market_data.get_current_price(sym) or 0.0)
                        except Exception:
                            px = 0.0
                        if px > 0:
                            _apply_close(ml, sym, pos, px, 'SCRATCH_ALL')
                            _del_pos(ml, sym)
                # Rescue trigger: if BTC + 1 ALT are both losing and a strong opposite swing appears,
                # open up to 2 remaining ALTs opposite direction with small size; widen old SL slightly (capped).
                # Universe config may be dict(listed under tickers/symbols) or a raw list/tuple of symbols
                symbols_cfg = None
                try:
                    cfg = getattr(self, 'config', {}) or {}
                    if isinstance(cfg, dict):
                        symbols_cfg = cfg.get('symbols')
                except Exception:
                    symbols_cfg = None

                universe = ['BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT']
                try:
                    if isinstance(symbols_cfg, dict):
                        universe = list(symbols_cfg.get('tickers') or symbols_cfg.get('symbols') or universe)
                    elif isinstance(symbols_cfg, (list, tuple, set)):
                        universe = list(symbols_cfg) or universe
                    elif isinstance(symbols_cfg, str):
                        u = [x.strip().upper() for x in symbols_cfg.split(',') if x.strip()]
                        universe = u or universe
                except Exception:
                    universe = ['BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT']
                btc_sym = None
                for s in universe:
                    if str(s).upper().startswith('BTC'):
                        btc_sym = s
                        break
                for ml in ('demo','shadow','paper'):
                    items = _iter_mode_positions(ml)
                    if len(items) < 2:
                        continue
                    if len(items) > 2:
                        continue  # normal mode: not rescue
                    syms_open = [s for s,_ in items]
                    if btc_sym and btc_sym not in syms_open:
                        continue
                    # determine common direction and loss severity
                    sides = [str(getattr(p,'side','')).upper() for _,p in items]
                    if not sides or any(s not in ('LONG','SHORT') for s in sides):
                        continue
                    if len(set(sides)) != 1:
                        continue
                    common_side = sides[0]
                    opp_side = 'SHORT' if common_side == 'LONG' else 'LONG'

                    # check both are losing (RR) and within age window
                    ok = True
                    for sym, pos in items:
                        try:
                            price = float(self.market_data.get_current_price(sym) or 0.0)
                        except Exception:
                            price = 0.0
                        entry = float(getattr(pos,'entry_price',0.0) or 0.0)
                        sl = float(getattr(pos,'stop_loss',0.0) or 0.0)
                        sl_dist = abs(entry - sl) if (entry>0 and sl>0) else 0.0
                        rr = 0.0
                        if sl_dist > 0 and price > 0:
                            rr = (price-entry)/sl_dist if common_side=='LONG' else (entry-price)/sl_dist
                        try:
                            age_min = (datetime.now() - (getattr(pos, 'entry_time', None) or datetime.now())).total_seconds()/60.0
                        except Exception:
                            age_min = 0.0
                        if not (age_min >= rescue_min_age_min and age_min <= rescue_max_age_min and rr <= rescue_trigger_rr):
                            ok = False
                            break
                    if not ok:
                        continue

                    # score remaining ALTs for opposite swing
                    candidates = []
                    for sym in universe:
                        if sym in syms_open:
                            continue
                        feat = _get_features(sym) or {}
                        htf_dir = int(feat.get('htf_dir', 0) or 0)
                        htf_strength = float(feat.get('htf_strength', 0.0) or 0.0)
                        trend_strength = float(feat.get('trend_strength', 0.0) or 0.0)
                        mom = float(feat.get('momentum', 0.0) or feat.get('mom', 0.0) or 0.0)
                        # Require opposite HTF bias + momentum in that direction
                        opp_dir = -1 if opp_side=='SHORT' else 1
                        if htf_dir != opp_dir:
                            continue
                        if abs(htf_strength) < 0.45:
                            continue
                        if (opp_dir==1 and mom <= 0) or (opp_dir==-1 and mom >= 0):
                            continue
                        # approximate "confidence"
                        conf = 0.5 + min(0.49, abs(htf_strength)*0.35 + abs(trend_strength)*8.0 + min(0.2, abs(mom)*5.0))
                        if conf < rescue_opposite_conf:
                            continue
                        score = (abs(htf_strength) * (1.0 + abs(trend_strength)*10.0) * (1.0 + min(1.0, abs(mom)*3.0)))
                        candidates.append((score, conf, sym, feat))
                    candidates.sort(reverse=True, key=lambda x: x[0])

                    # open up to 2 candidates, but cap total positions
                    n_open_allowed = max(0, int(rescue_max_total_pos) - len(items))
                    n_to_open = min(2, n_open_allowed, len(candidates))
                    if n_to_open <= 0:
                        continue

                    # widen old SL slightly (capped)
                    for sym, pos in items:
                        try:
                            entry = float(getattr(pos,'entry_price',0.0) or 0.0)
                            sl = float(getattr(pos,'stop_loss',0.0) or 0.0)
                            if entry>0 and sl>0:
                                dist = abs(entry-sl)
                                new_dist = dist * float(rescue_widen_sl_mult)
                                cap = entry * float(rescue_widen_sl_cap_pct)
                                new_dist = min(new_dist, cap)
                                if common_side=='LONG':
                                    new_sl = entry - new_dist
                                else:
                                    new_sl = entry + new_dist
                                # apply
                                if ml == 'paper' and hasattr(self, 'execution_engine'):
                                    try:
                                        self.execution_engine.modify_stop_loss(sym, new_sl, mode='PAPER')
                                    except Exception:
                                        pass
                                try:
                                    setattr(pos, 'stop_loss', float(new_sl))
                                except Exception:
                                    pass
                        except Exception:
                            pass

                    # open hedge(s)
                    for k in range(n_to_open):
                        _, conf, sym, feat = candidates[k]
                        try:
                            entry_px = float(self.market_data.get_current_price(sym) or 0.0)
                        except Exception:
                            entry_px = 0.0
                        if entry_px <= 0:
                            continue
                        atr = float(feat.get('atr', 0.0) or 0.0)
                        if atr <= 0:
                            atr = max(1e-6, entry_px * 0.0025)
                        sl_dist = atr * 1.55
                        tp_dist = atr * 2.10
                        if opp_side == 'LONG':
                            sl = entry_px - sl_dist
                            tp = entry_px + tp_dist
                        else:
                            sl = entry_px + sl_dist
                            tp = entry_px - tp_dist

                        # Determine qty: fraction of initial risk size (use risk guard sizing helper if available)
                        qty = 0.0
                        try:
                            base_qty = float(getattr(items[0][1], 'quantity', 0.0) or 0.0)
                            qty = max(0.0, base_qty * float(rescue_add_qty_frac))
                        except Exception:
                            qty = 0.0
                        # if we can't infer qty, skip
                        if qty <= 0:
                            continue

                        class _Sig:  # lightweight signal container
                            pass
                        sig = _Sig()
                        sig.symbol = sym
                        sig.action = opp_side
                        sig.entry_price = entry_px
                        sig.stop_loss = sl
                        sig.take_profit = tp
                        sig.position_qty = qty
                        sig.confidence = conf
                        sig.reason = 'REVERSAL_RESCUE'
                        sig.regime = 'rescue'

                        if ml == 'paper' and hasattr(self, 'execution_engine'):
                            self.execution_engine.execute_trade(sig, mode='PAPER')
                        else:
                            # simulate open in DEMO/SHADOW using Position dataclass
                            try:
                                posn = Position(symbol=sym, side=opp_side, entry_price=entry_px, quantity=qty, stop_loss=sl, take_profit=tp,
                                                entry_reason='REVERSAL_RESCUE', regime='rescue', confidence=float(conf))
                                d = _pos_dict_for_mode(ml)
                                if isinstance(d, dict):
                                    d[sym] = posn
                            except Exception:
                                pass

                    self._last_rescue_ts = time.time()
        except Exception:
            logging.exception('reversal rescue failed')

        # live sync (so Telegram balance matches reality)
        self._sync_live_account()
        self._apply_accounts_to_telegram()

    def stop(self):
        self.running = False
        if self.telegram:
            self.telegram.send_message("🛑 Hệ thống dừng an toàn")
        if self.telegram_updater:
            self.telegram_updater.stop()
        # PTB v20 Application polling (daemon thread) - best-effort stop
        try:
            app = getattr(self, '_tg_app', None)
            if app is not None:
                # Some PTB versions expose sync stop(); in newer versions it may be coroutine.
                try:
                    app.stop()
                except TypeError:
                    pass
                except Exception:
                    pass
        except Exception:
            pass
        # Simple polling loop
        try:
            setattr(self, '_simple_tg_stop', True)
        except Exception:
            pass
        print("🛑 Hệ thống dừng")
        sys.exit(0)

    def _monitor_loop(self):
        while self.running:
            try:
                if self.telegram:
                    metrics = self.risk_guard.get_metrics()
                    pos_by_mode = self._positions_by_mode()
                    self.telegram.report_dashboard(metrics, self.modes, pos_by_mode, force_send=True)
                time.sleep(300)
            except Exception as e:
                logging.error(f"Monitor error: {e}")
                time.sleep(60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler('trading_v2.log', encoding='utf-8')],
        force=True
    )
    
    try:
        system = TradingSystemV2()
        system.start()
        
        print("DEBUG: Hệ thống đang chạy liên tục. Nhấn Ctrl+C để dừng.")
        while True:
            time.sleep(3600)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Đã hủy bởi người dùng")
        if 'system' in locals():
            system.stop()
        sys.exit(0)
    except Exception as e:
        print(f"Critical error: {e}")
        logging.critical(f"Startup failed: {e}", exc_info=True)
        sys.exit(1)


# ===================== TELEGRAM POLLING FIX (PTB v20, Python 3.8) =====================
# NOTE: This definition intentionally overrides any previous _run_app_polling
# to avoid indentation or event-loop issues.

def _run_app_polling(self):
    """Chạy Telegram polling ổn định cho PTB v20+ trong thread riêng (Python 3.8).

    - Không dùng HTTPXRequest (tránh phụ thuộc/khác phiên bản).
    - Tự backoff nếu polling crash.
    """
    import time
    import logging
    try:
        from telegram import Update
    except Exception:
        Update = None

    backoff = 1.0
    while True:
        try:
            if not getattr(self, '_tg_app', None):
                logging.error("Telegram app is not initialized; polling skipped")
                return

            # stop_signals=None để không bắt signal khi chạy trong thread
            kwargs = {
                'drop_pending_updates': True,
                'stop_signals': None,
            }
            # allowed_updates giúp nhận command ổn định
            if Update is not None:
                kwargs['allowed_updates'] = Update.ALL_TYPES

            self._tg_app.run_polling(**kwargs)
            backoff = 1.0
        except Exception:
            logging.exception("Telegram Application polling crashed; retrying...")
            time.sleep(min(backoff, 30.0))
            backoff = min(backoff * 2.0, 30.0)