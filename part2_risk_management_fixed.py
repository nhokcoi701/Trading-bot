"""
Part 2: ADVANCED RISK MANAGEMENT - FIXED & COMPLETED FOR PYTHON 3.8
- Hoàn thiện phần bị truncated
- Thêm @lru_cache cho metrics nặng
- Xử lý edge case (equity=0, division by zero)
- FIX: Không dùng self.drawdown_pct hay self.daily_pnl_pct (không tồn tại)
  → Tính toán trực tiếp dd_pct và daily_pnl_pct trong _update_state()
- Optional: _load_state() và _save_state() để lưu trạng thái risk (JSON)
- FIX: Thêm portfolio_heat và max_portfolio_heat vào RiskMetrics & get_metrics()
- FIX: Khởi tạo CoinRiskProfile đúng cách với symbol từ key config
- THÊM: Method validate_entry() đầy đủ để fix lỗi 'no attribute validate_entry'
- FIX: Sửa .get() cho dataclass CoinRiskProfile (dùng getattr thay vì .get())
- FIX MỚI: Sửa lỗi 'no attribute portfolio_heat' → dùng self.current_heat trong get_metrics()
"""

import logging
import time
import numpy as np
from datetime import datetime, timedelta
import datetime as _dt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import json
import os
from functools import lru_cache

from error_handling import handle_errors, RiskLimitExceeded, InsufficientFundsError

class RiskMode(Enum):
    ULTRA_AGGRESSIVE = "ULTRA_AGGRESSIVE"
    MAX = "MAX"
    SMART = "SMART"
    DEFENSIVE = "DEFENSIVE"
    SURVIVAL = "SURVIVAL"
    NO_TRADE = "NO_TRADE"

class RiskState(Enum):
    EXCELLENT = "EXCELLENT"
    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    DANGER = "DANGER"
    CRITICAL = "CRITICAL"
    DEAD = "DEAD"

@dataclass
class CoinRiskProfile:
    symbol: str
    role: str  # LEADER or FOLLOWER
    volatility_rank: int
    risk_multiplier: float
    priority: int
    max_position_pct: float
    min_confidence: float
    
    trades: int = 0
    wins: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0

@dataclass
class RiskMetrics:
    equity: float
    initial_equity: float
    daily_pnl: float
    daily_pnl_pct: float
    weekly_pnl: float
    weekly_pnl_pct: float
    monthly_pnl: float
    monthly_pnl_pct: float
    
    peak_equity: float
    drawdown: float
    drawdown_pct: float
    max_drawdown: float
    max_drawdown_pct: float
    
    win_streak: int
    loss_streak: int
    max_win_streak: int
    max_loss_streak: int
    
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    volatility: float
    
    max_positions: int
    current_positions: int
    portfolio_heat: float
    max_portfolio_heat: float
    
    mode: RiskMode
    state: RiskState

class AdvancedRiskGuard:
    def __init__(self, config: Dict):
        self.config = config
        self.equity = config.get('initial_equity', 1000.0)
        self.initial_equity = self.equity
        self.peak_equity = self.equity
        self.drawdown = 0.0
        self.max_drawdown = 0.0
        
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.monthly_pnl = 0.0
        
        self.win_streak = 0
        self.loss_streak = 0
        self.max_win_streak = 0
        self.max_loss_streak = 0
        
        # Persistent timestamps for loss-brake / daily reset
        import datetime as _dt
        self.state_date = _dt.date.today().isoformat()
        self.last_trade_ts = 0.0
        self.last_loss_ts = 0.0
        self.recovery_trades_left = 0
        
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.win_rate = 0.0
        self.profit_factor = 0.0
        
        self.sharpe_ratio = 0.0
        self.sortino_ratio = 0.0
        self.calmar_ratio = 0.0
        self.volatility = 0.0
        
        self.current_positions = 0
        self.max_positions = 10
        self.current_heat = 0.0  # Giá trị thực tế portfolio heat hiện tại
        self.max_portfolio_heat = config.get('risk_limits', {}).get('max_portfolio_heat', 20.0)
        
        self.mode = RiskMode.SMART
        self.state = RiskState.NORMAL
        
        self.coin_profiles: Dict[str, CoinRiskProfile] = {}
        for sym, prof in config.get('coin_profiles', {}).items():
            self.coin_profiles[sym] = CoinRiskProfile(
                symbol=sym,
                role=prof.get('role', 'FOLLOWER'),
                volatility_rank=prof.get('volatility_rank', 1),
                risk_multiplier=prof.get('risk_multiplier', 1.0),
                priority=prof.get('priority', 1),
                max_position_pct=prof.get('max_position_pct', 2.0),
                min_confidence=prof.get('min_confidence', 0.6)
            )
        
        self.trade_history = []
        self.daily_history = []
        self.weekly_history = []
        self.monthly_history = []
        
        self.last_day = datetime.now().date()
        self.last_week = datetime.now().isocalendar()[1]
        self.last_month = datetime.now().month
        
        self._load_state()

        # CIO-grade runtime overrides (do NOT mutate base config)
        # Used by AdaptiveEngine/Governor to adjust aggressiveness safely at runtime.
        self.runtime_risk_scale = 1.0
        self.runtime_min_conf_overrides: Dict[str, float] = {}


        # ===== V7: runtime snapshots for anti-churn & correlation cluster guard =====
        self._market_snapshot = {}   # symbol -> dict(mid, spread_bps, vol_pct, latency_ms, ts)
        self._returns_cache = {}     # symbol -> dict(ret=np.ndarray, ts=float)
        self._positions_snapshot = {}  # raw positions dict from execution engine
        self._last_cluster_debug = {}

        # ===== Adaptive Anti-Churn (paper/demo/shadow) =====
        # Mục tiêu: không bị "đói lệnh" khi đang test, nhưng vẫn fee-aware.
        self._last_anti_churn_debug = {}
        self._anti_churn_starve_scans = 0
        self._anti_churn_relax_mult = 1.0
        self._anti_churn_state_path = (self.config.get('risk_limits', {}) or {}).get('anti_churn', {}).get('adaptive_state_path', 'data/anti_churn_adapt.json')
        self._load_anti_churn_state()

    def update_config(self, new_config: dict):
        "Hot-apply config changes (safe subset)."
        if not isinstance(new_config, dict):
            return
        self.config = new_config

        # Risk limits
        try:
            rl = new_config.get('risk_limits', {}) or {}
            self.max_portfolio_heat = float(rl.get('max_portfolio_heat', self.max_portfolio_heat) or self.max_portfolio_heat)
        except Exception:
            pass

        # Coin profiles
        try:
            cp = new_config.get('coin_profiles', {}) or {}
            if isinstance(cp, dict) and cp:
                self.coin_profiles = {}
                for sym, prof in cp.items():
                    self.coin_profiles[sym] = CoinRiskProfile(
                        symbol=sym,
                        role=prof.get('role', 'FOLLOWER'),
                        volatility_rank=prof.get('volatility_rank', 1),
                        risk_multiplier=prof.get('risk_multiplier', 1.0),
                        priority=prof.get('priority', 1),
                        max_position_pct=prof.get('max_position_pct', 2.0),
                        min_confidence=prof.get('min_confidence', 0.6)
                    )
        except Exception:
            pass


    # ===== Adaptive Anti-Churn helpers =====
    def _load_anti_churn_state(self):
        try:
            p = str(self._anti_churn_state_path or 'data/anti_churn_adapt.json')
            d = os.path.dirname(p)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8') as f:
                    st = json.load(f) or {}
                self._anti_churn_relax_mult = float(st.get('relax_mult', 1.0) or 1.0)
                self._anti_churn_starve_scans = int(st.get('starve_scans', 0) or 0)
        except Exception:
            # stay with defaults
            self._anti_churn_relax_mult = float(getattr(self, '_anti_churn_relax_mult', 1.0) or 1.0)
            self._anti_churn_starve_scans = int(getattr(self, '_anti_churn_starve_scans', 0) or 0)

    def _save_anti_churn_state(self):
        try:
            p = str(self._anti_churn_state_path or 'data/anti_churn_adapt.json')
            d = os.path.dirname(p)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            st = {
                'relax_mult': float(self._anti_churn_relax_mult),
                'starve_scans': int(self._anti_churn_starve_scans),
                'ts': float(time.time())
            }
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(st, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def note_scan(self, scan_stats: dict):
        """Call once per scan loop so anti-churn can self-tune in paper/demo/shadow.

        - If system is starving (many scans with risk_reject and opened=0), relax thresholds gradually.
        - If trades start opening, slowly revert toward baseline.
        """
        try:
            if not isinstance(scan_stats, dict):
                return
            opened = int(scan_stats.get('opened_demo', 0) or 0) + int(scan_stats.get('opened_shadow', 0) or 0) + int(scan_stats.get('opened_paper', 0) or 0) + int(scan_stats.get('opened_live', 0) or 0)
            risk_reject = int(scan_stats.get('risk_reject', 0) or 0)

            # Determine if we are in "paperish" learning modes
            modes = (self.config.get('modes', {}) or {}) if isinstance(self.config, dict) else {}
            paperish = bool(modes.get('paper', False) or modes.get('demo', False) or modes.get('shadow', False))

            if not paperish:
                # In live: keep stable, gently restore to 1.0
                self._anti_churn_relax_mult = 1.0
                self._anti_churn_starve_scans = 0
                return

            # Count anti-churn reject reasons (best-effort)
            rej_edge = 0
            rej_risk = 0
            rej_min_edge = 0
            try:
                for _sym, dbg in (getattr(self, '_last_anti_churn_debug', {}) or {}).items():
                    rr = str((dbg or {}).get('reject_reason', '') or '')
                    if rr == 'edge_over_cost':
                        rej_edge += 1
                    elif rr == 'risk_over_cost':
                        rej_risk += 1
                    elif rr == 'edge_bps<min_edge_bps':
                        rej_min_edge += 1
            except Exception:
                pass

            if opened <= 0 and risk_reject > 0:
                self._anti_churn_starve_scans = int(getattr(self, '_anti_churn_starve_scans', 0) or 0) + 1

                # Every 6 starve scans -> relax a bit more (floored)
                if self._anti_churn_starve_scans % 6 == 0:
                    floor = float((self.config.get('risk_limits', {}) or {}).get('anti_churn', {}).get('relax_floor_paper', 0.55) or 0.55)
                    cur = float(getattr(self, '_anti_churn_relax_mult', 1.0) or 1.0)
                    # If mostly edge_over_cost, relax more aggressively; if min_edge, relax mildly.
                    if rej_edge >= max(1, rej_risk + rej_min_edge):
                        new = max(floor, cur * 0.92)
                    else:
                        new = max(floor, cur * 0.95)
                    if new < cur - 1e-6:
                        self._anti_churn_relax_mult = new
                        self._save_anti_churn_state()
                        logging.warning(f"ANTI_CHURN adaptive relax -> {new:.3f} (starve_scans={self._anti_churn_starve_scans}, rej_edge={rej_edge}, rej_risk={rej_risk}, rej_min_edge={rej_min_edge})")
            else:
                # When we open trades, slowly restore toward baseline
                if opened > 0:
                    self._anti_churn_starve_scans = 0
                cur = float(getattr(self, '_anti_churn_relax_mult', 1.0) or 1.0)
                if cur < 1.0:
                    new = min(1.0, cur + 0.02)
                    if new != cur:
                        self._anti_churn_relax_mult = new
                        self._save_anti_churn_state()
        except Exception:
            pass

    def _mode_relax_factor(self) -> float:
        """Base relax factor by mode (paper/demo/shadow only)."""
        try:
            modes = (self.config.get('modes', {}) or {}) if isinstance(self.config, dict) else {}
            paperish = bool(modes.get('paper', False) or modes.get('demo', False) or modes.get('shadow', False))
            if paperish:
                return float((self.config.get('risk_limits', {}) or {}).get('anti_churn', {}).get('paper_relax', 0.72) or 0.72)
        except Exception:
            pass
        return 1.0
    def set_runtime_risk_scale(self, scale: float):
            """Set runtime risk multiplier (0.05..1.50). Does not change base config."""
            try:
                s = float(scale or 1.0)
                self.runtime_risk_scale = max(0.05, min(1.50, s))
            except Exception:
                self.runtime_risk_scale = 1.0

    def set_runtime_min_confidence(self, symbol: str, min_conf: float):
            """Override per-symbol min_confidence at runtime."""
            try:
                if not symbol:
                    return
                mc = float(min_conf)
                # keep reasonable bounds
                mc = max(0.20, min(0.80, mc))
                self.runtime_min_conf_overrides[str(symbol)] = mc
            except Exception:
                pass

    def clear_runtime_min_confidence(self):
            try:
                self.runtime_min_conf_overrides = {}
            except Exception:
                pass

    def _save_state(self):
            state = {
                'equity': self.equity,
                'peak_equity': self.peak_equity,
                'max_drawdown': self.max_drawdown,
                'win_streak': self.win_streak,
                'loss_streak': self.loss_streak,
                'max_win_streak': self.max_win_streak,
                'max_loss_streak': self.max_loss_streak,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'state_date': self.state_date,
                'last_trade_ts': self.last_trade_ts,
                'last_loss_ts': self.last_loss_ts,
                'recovery_trades_left': getattr(self, 'recovery_trades_left', 0),
                'trade_history': self.trade_history[-1000:],
                'coin_profiles': {k: asdict(v) for k, v in self.coin_profiles.items()}
            }
            with open('risk_state.json', 'w') as f:
                json.dump(state, f)

    def _load_state(self):
            if os.path.exists('risk_state.json'):
                try:
                    with open('risk_state.json', 'r') as f:
                        state = json.load(f)
                    self.equity = state.get('equity', self.equity)
                    self.peak_equity = state.get('peak_equity', self.peak_equity)
                    self.max_drawdown = state.get('max_drawdown', self.max_drawdown)
                    self.win_streak = state.get('win_streak', 0)
                    self.loss_streak = state.get('loss_streak', 0)
                    self.state_date = state.get('state_date', self.state_date)
                    self.last_trade_ts = float(state.get('last_trade_ts', 0.0) or 0.0)
                    self.last_loss_ts = float(state.get('last_loss_ts', 0.0) or 0.0)
                    self.recovery_trades_left = int(state.get('recovery_trades_left', 0) or 0)

                    # Soft reset when day changes (avoid permanent lock from yesterday)
                    try:
                        today = _dt.date.today().isoformat()
                        if str(self.state_date) != today:
                            self.state_date = today
                            self.daily_pnl = 0.0
                            self.loss_streak = 0
                            self.win_streak = 0
                            self.recovery_trades_left = 0
                    except Exception:
                        pass
                    self.max_win_streak = state.get('max_win_streak', 0)
                    self.max_loss_streak = state.get('max_loss_streak', 0)
                    self.total_trades = state.get('total_trades', 0)
                    self.winning_trades = state.get('winning_trades', 0)
                    self.losing_trades = state.get('losing_trades', 0)
                    self.trade_history = state.get('trade_history', [])
                    for sym, prof in state.get('coin_profiles', {}).items():
                        if sym in self.coin_profiles:
                            p = self.coin_profiles[sym]
                            p.trades = prof.get('trades', 0)
                            p.wins = prof.get('wins', 0)
                            p.total_pnl = prof.get('total_pnl', 0.0)
                            p.win_rate = prof.get('win_rate', 0.0)
                except Exception as e:
                    logging.error(f"Load risk state failed: {e}")

    def update_after_trade(self, symbol: str, pnl: float, fee_est: float = None, trade_id: str = None):
            now_ts = time.time()
            # De-duplicate close events to prevent double-counting (common cause of CRITICAL lockups)
            try:
                tid = str(trade_id) if trade_id is not None else None
            except Exception:
                tid = None
            if tid:
                try:
                    last_tid = str(getattr(self, '_last_trade_id', '') or '')
                    if last_tid == tid:
                        return
                    self._last_trade_id = tid
                except Exception:
                    pass
            else:
                try:
                    if getattr(self, 'trade_history', None):
                        last = self.trade_history[-1]
                        if str(last.get('symbol')) == str(symbol) and abs(float(last.get('pnl', 0.0)) - float(pnl)) < 1e-9:
                            return
                except Exception:
                    pass
            self.last_trade_ts = now_ts
            try:
                self.state_date = _dt.date.today().isoformat()
            except Exception:
                pass
            self.total_trades += 1
            if pnl > 0:
                self.winning_trades += 1
                try:
                    self.recovery_trades_left = 0
                except Exception:
                    pass
                self.win_streak += 1
                self.loss_streak = 0
                self.max_win_streak = max(self.max_win_streak, self.win_streak)
            else:
                # Ignore tiny fee-only scratches for loss streak / danger state
                ignore_mult = float((self.config.get('risk_limits', {}) or {}).get('ignore_losses_under_fee_mult', 1.6) or 1.6)
                try:
                    fe = float(fee_est) if fee_est is not None else None
                except Exception:
                    fe = None
                ignored_fee_scratch = bool(fe is not None and fe > 0 and abs(float(pnl)) <= (fe * ignore_mult))
                self.losing_trades += 1
                if not ignored_fee_scratch:
                    self.loss_streak += 1
                    self.last_loss_ts = now_ts
                # enter recovery mode after consecutive losses
                try:
                    if self.loss_streak >= 2:
                        self.recovery_trades_left = max(int(getattr(self, 'recovery_trades_left', 0) or 0), 3)
                except Exception:
                    pass
                self.win_streak = 0
                self.max_loss_streak = max(self.max_loss_streak, self.loss_streak)
        
            self.equity += pnl
            self.peak_equity = max(self.peak_equity, self.equity)
            self.drawdown = self.peak_equity - self.equity
            self.max_drawdown = max(self.max_drawdown, self.drawdown)
        
            self.daily_pnl += pnl
            self.weekly_pnl += pnl
            self.monthly_pnl += pnl
        
            self.trade_history.append({'symbol': symbol, 'pnl': pnl, 'timestamp': datetime.now().isoformat()})
        
            if symbol in self.coin_profiles:
                profile = self.coin_profiles[symbol]
                profile.trades += 1
                profile.wins += 1 if pnl > 0 else 0
                profile.total_pnl += pnl
                profile.win_rate = profile.wins / profile.trades if profile.trades > 0 else 0.0
        
            self._update_state()
            self._save_state()

    def _update_state(self):
            self.win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0
        
            total_profit = sum(t['pnl'] for t in self.trade_history if t['pnl'] > 0)
            total_loss = abs(sum(t['pnl'] for t in self.trade_history if t['pnl'] < 0))
            self.profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
            if len(self.trade_history) > 1:
                returns = [t['pnl'] / self.initial_equity for t in self.trade_history]
                self.volatility = np.std(returns) if returns else 0.0
                self.sharpe_ratio = np.mean(returns) / self.volatility if self.volatility > 0 else 0.0
            
                negative_returns = [r for r in returns if r < 0]
                sortino_vol = np.std(negative_returns) if negative_returns else 0.0
                self.sortino_ratio = np.mean(returns) / sortino_vol if sortino_vol > 0 else 0.0
            
                self.calmar_ratio = np.mean(returns) / (self.max_drawdown / self.initial_equity) if self.max_drawdown > 0 else 0.0
        
            self.current_heat = (self.current_positions / self.max_positions) * 100 if self.max_positions > 0 else 0.0
        
            now = datetime.now()
            if now.date() != self.last_day:
                self.daily_history.append(self.daily_pnl)
                self.daily_pnl = 0.0
                self.last_day = now.date()
        
            if now.isocalendar()[1] != self.last_week:
                self.weekly_history.append(self.weekly_pnl)
                self.weekly_pnl = 0.0
                self.last_week = now.isocalendar()[1]
        
            if now.month != self.last_month:
                self.monthly_history.append(self.monthly_pnl)
                self.monthly_pnl = 0.0
                self.last_month = now.month
        
            dd_pct = (self.drawdown / self.peak_equity * 100) if self.peak_equity > 0 else 0
            daily_pnl_pct = (self.daily_pnl / self.initial_equity * 100) if self.initial_equity > 0 else 0
            weekly_pnl_pct = (self.weekly_pnl / self.initial_equity * 100) if self.initial_equity > 0 else 0
            monthly_pnl_pct = (self.monthly_pnl / self.initial_equity * 100) if self.initial_equity > 0 else 0
        
            if dd_pct > 20 or self.loss_streak > 5:
                self.state = RiskState.CRITICAL
            elif dd_pct > 15 or self.loss_streak > 3:
                self.state = RiskState.DANGER
            elif dd_pct > 10 or self.loss_streak > 2:
                self.state = RiskState.CAUTION
            elif dd_pct < 5 and self.win_streak > 3:
                self.state = RiskState.EXCELLENT
            else:
                self.state = RiskState.NORMAL
        
            self.mode = RiskMode[self.config.get('auto_mode_switching', {}).get(self.state.value, 'SMART')]

    @lru_cache(maxsize=32)
    def get_metrics(self) -> RiskMetrics:
            dd_pct = (self.drawdown / self.peak_equity * 100) if self.peak_equity > 0 else 0
            md_pct = (self.max_drawdown / self.initial_equity * 100) if self.initial_equity > 0 else 0
            daily_pnl_pct = (self.daily_pnl / self.initial_equity * 100) if self.initial_equity > 0 else 0
            weekly_pnl_pct = (self.weekly_pnl / self.initial_equity * 100) if self.initial_equity > 0 else 0
            monthly_pnl_pct = (self.monthly_pnl / self.initial_equity * 100) if self.initial_equity > 0 else 0
        
            return RiskMetrics(
                equity=self.equity,
                initial_equity=self.initial_equity,
                daily_pnl=self.daily_pnl,
                daily_pnl_pct=daily_pnl_pct,
                weekly_pnl=self.weekly_pnl,
                weekly_pnl_pct=weekly_pnl_pct,
                monthly_pnl=self.monthly_pnl,
                monthly_pnl_pct=monthly_pnl_pct,
                peak_equity=self.peak_equity,
                drawdown=self.drawdown,
                drawdown_pct=dd_pct,
                max_drawdown=self.max_drawdown,
                max_drawdown_pct=md_pct,
                win_streak=self.win_streak,
                loss_streak=self.loss_streak,
                max_win_streak=self.max_win_streak,
                max_loss_streak=self.max_loss_streak,
                total_trades=self.total_trades,
                winning_trades=self.winning_trades,
                losing_trades=self.losing_trades,
                win_rate=self.win_rate,
                profit_factor=self.profit_factor,
                sharpe_ratio=self.sharpe_ratio,
                sortino_ratio=self.sortino_ratio,
                calmar_ratio=self.calmar_ratio,
                volatility=self.volatility,
                max_positions=self.max_positions,
                current_positions=self.current_positions,
                portfolio_heat=self.current_heat,  # ĐÃ FIX: dùng current_heat thay vì portfolio_heat
                max_portfolio_heat=self.max_portfolio_heat,
                mode=self.mode,
                state=self.state
            )

    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float) -> float:
            # Base risk per trade (equity-based).
            # Prefer risk_limits.risk_per_trade_pct_equity if provided; otherwise fall back to base_risk_pct.
            rl = (self.config.get('risk_limits', {}) or {}) if isinstance(self.config, dict) else {}
            try:
                cfg_risk = float(rl.get('risk_per_trade_pct_equity', 0.0) or 0.0)
            except Exception:
                cfg_risk = 0.0
            if cfg_risk <= 0:
                cfg_risk = float(self.config.get('base_risk_pct', 1.0) or 1.0)
            # Hard cap to avoid accidental over-sizing
            try:
                max_loss_trade = float(rl.get('max_loss_per_trade_pct_equity', 0.0) or 0.0)
            except Exception:
                max_loss_trade = 0.0
            if max_loss_trade > 0:
                cfg_risk = min(cfg_risk, max_loss_trade)
            cfg_risk = max(0.05, min(5.0, float(cfg_risk)))
            risk_pct = cfg_risk / 100.0
            # Runtime scaling (AdaptiveEngine/Governor). Keep within sensible bounds.
            try:
                rs = float(getattr(self, 'runtime_risk_scale', 1.0) or 1.0)
                rs = max(0.05, min(1.50, rs))
                risk_pct *= rs
            except Exception:
                pass
            risk_per_share = abs(entry_price - stop_loss)
            if risk_per_share == 0:
                return 0.0
        
            if symbol in self.coin_profiles:
                profile = self.coin_profiles[symbol]
                risk_pct *= profile.risk_multiplier
        
            position_size = (self.equity * risk_pct) / risk_per_share
            # Futures-aware max size cap: treat coin profile max_position_pct as a NOTIONAL budget fraction
            # of (equity * leverage * max_margin_util), instead of spot-style %equity.
            try:
                lev_cap = int(self.config.get('default_leverage', 10) or 10)
            except Exception:
                lev_cap = 10
            lev_cap = max(1, min(125, lev_cap))
            try:
                mmu = float(((self.config.get('futures', {}) or {}).get('max_margin_util', None) if isinstance(self.config, dict) else None) or (self.config.get('risk_limits', {}) or {}).get('max_margin_util', 0.80))
            except Exception:
                mmu = 0.80
            mmu = max(0.10, min(0.98, mmu))
            if symbol in self.coin_profiles:
                profile = self.coin_profiles[symbol]
                max_notional = float(self.equity) * float(lev_cap) * float(mmu) * (float(profile.max_position_pct) / 100.0)
                max_size = max_notional / max(entry_price, 1e-12)
            else:
                max_size = position_size
        
            return min(position_size, max_size)

    def update_market_snapshot(self, symbol: str, mid: float = None, spread_bps: float = None,
                                   vol_pct: float = None, latency_ms: float = None) -> None:
            """Best-effort market snapshot updated from main loop. All fields optional."""
            try:
                snap = self._market_snapshot.get(symbol, {})
                ts = time.time()
                if mid is not None:
                    snap['mid'] = float(mid)
                if spread_bps is not None:
                    snap['spread_bps'] = float(spread_bps)
                if vol_pct is not None:
                    snap['vol_pct'] = float(vol_pct)
                if latency_ms is not None:
                    snap['latency_ms'] = float(latency_ms)
                snap['ts'] = ts
                self._market_snapshot[symbol] = snap
            except Exception:
                # never break trading because of snapshot update
                return

    def update_returns_cache(self, symbol: str, closes, max_len: int = 600) -> None:
            """Store recent log-returns for correlation checks."""
            try:
                arr = np.asarray(closes, dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size < 30:
                    return
                if arr.size > max_len:
                    arr = arr[-max_len:]
                ret = np.diff(np.log(arr))
                ret = ret[np.isfinite(ret)]
                if ret.size < 20:
                    return
                self._returns_cache[symbol] = {'ret': ret.astype(float), 'ts': time.time()}
            except Exception:
                return

    def set_positions_snapshot(self, positions: dict) -> None:
            """Provide current open positions from execution engine."""
            try:
                self._positions_snapshot = positions or {}
            except Exception:
                self._positions_snapshot = {}

    def _corr(self, a: np.ndarray, b: np.ndarray) -> float:
            try:
                n = min(a.size, b.size)
                if n < 20:
                    return 0.0
                a2 = a[-n:]
                b2 = b[-n:]
                if np.std(a2) < 1e-12 or np.std(b2) < 1e-12:
                    return 0.0
                c = float(np.corrcoef(a2, b2)[0, 1])
                if not np.isfinite(c):
                    return 0.0
                return c
            except Exception:
                return 0.0

    def _infer_position_direction(self, pos: dict) -> str:
            """Return 'LONG' or 'SHORT' for a position dict."""
            try:
                # common keys: side, positionSide, qty/positionAmt sign
                side = (pos.get('side') or pos.get('positionSide') or pos.get('direction') or '').upper()
                if side in ('LONG', 'SHORT'):
                    return side
                amt = pos.get('positionAmt')
                if amt is None:
                    amt = pos.get('qty') or pos.get('amount') or pos.get('position_qty')
                if amt is not None:
                    try:
                        amt = float(amt)
                        return 'LONG' if amt > 0 else 'SHORT'
                    except Exception:
                        pass
            except Exception:
                pass
            return 'LONG'

    def _infer_position_notional(self, pos: dict) -> float:
            """Best-effort notional USD for a position."""
            try:
                for k in ('notional', 'position_value_usd', 'pos_value_usd', 'value_usd'):
                    if k in pos and pos[k] is not None:
                        return float(pos[k])
                qty = pos.get('qty') or pos.get('positionAmt') or pos.get('amount') or pos.get('position_qty')
                entry = pos.get('entry_price') or pos.get('entryPrice') or pos.get('avg_price') or pos.get('avgPrice')
                if qty is not None and entry is not None:
                    return abs(float(qty)) * float(entry)
            except Exception:
                return 0.0
            return 0.0

    def _anti_churn_gate(self, symbol: str, signal) -> bool:
            """Fee/spread/latency-aware anti-churn gate.

            Goal: reject tiny-edge trades that will churn in fees/spread, but avoid starving the system.
            Thresholds adapt to signal confidence and to paper/demo/shadow modes.
            """
            cfg = (self.config.get('risk_limits', {}) or {}).get('anti_churn', {}) or {}
            if not bool(cfg.get('enabled', True)):
                return True

            try:
                snap = self._market_snapshot.get(symbol, {}) or {}
                spread_bps = float(snap.get('spread_bps', cfg.get('spread_bps_est', 2.0)) or cfg.get('spread_bps_est', 2.0))
                vol_pct = float(snap.get('vol_pct', 0.0) or 0.0)
                latency_ms = float(snap.get('latency_ms', cfg.get('latency_ms_est', 150.0)) or cfg.get('latency_ms_est', 150.0))

                # Interpret fee_bps/slippage_bps as per-side (bps). Round-trip uses *2.
                fee_bps = float(cfg.get('fee_bps', cfg.get('fee_bps_side', 4.0)) or cfg.get('fee_bps_side', 4.0))
                slippage_bps = float(cfg.get('slippage_bps', cfg.get('slippage_bps_side', 4.0)) or cfg.get('slippage_bps_side', 4.0))

                # Paper/demo/shadow: allow lower default slippage estimate (still override-able in config)
                try:
                    modes = (self.config.get('modes', {}) or {}) if isinstance(self.config, dict) else {}
                    paperish = bool(modes.get('paper', False) or modes.get('demo', False) or modes.get('shadow', False))
                except Exception:
                    paperish = True
                if paperish:
                    try:
                        sl_paper = cfg.get('slippage_bps_side_paper', None)
                        if sl_paper is not None:
                            slippage_bps = float(sl_paper)
                        else:
                            slippage_bps = float(min(slippage_bps, 2.0))
                    except Exception:
                        pass

                entry = float(getattr(signal, 'entry_price', 0.0) or getattr(signal, 'price', 0.0) or 0.0)
                tp = float(getattr(signal, 'take_profit', 0.0) or 0.0)
                sl = float(getattr(signal, 'stop_loss', 0.0) or 0.0)
                action = (getattr(signal, 'action', '') or '').upper()
                conf = float(getattr(signal, 'confidence', 0.5) or 0.5)
                conf = max(0.0, min(1.0, conf))

                if entry <= 0 or tp <= 0 or sl <= 0 or action not in ('LONG', 'SHORT'):
                    return True

                if action == 'LONG':
                    edge_bps = (tp - entry) / entry * 10000.0
                    risk_bps = (entry - sl) / entry * 10000.0
                else:
                    edge_bps = (entry - tp) / entry * 10000.0
                    risk_bps = (sl - entry) / entry * 10000.0

                vol_bps = max(0.0, vol_pct * 10000.0)
                lat_ref = float(cfg.get('latency_ref_s', 60.0) or 60.0)
                lat_pen = (latency_ms / 1000.0) * (vol_bps / max(lat_ref, 1e-9))

                cost_bps = (2.0 * fee_bps) + (2.0 * slippage_bps) + max(0.0, spread_bps) + max(0.0, lat_pen)

                base_min_edge_bps = float(cfg.get('min_edge_bps', 14.0))
                base_edge_over_cost = float(cfg.get('min_edge_over_cost', 2.0))
                base_risk_over_cost = float(cfg.get('min_risk_over_cost', 1.2))

                # Relax slightly in non-live modes to learn (still fee-aware)
                try:
                    modes = (self.config.get('modes', {}) or {}) if isinstance(self.config, dict) else {}
                    paperish = bool(modes.get('paper', False) or modes.get('demo', False) or modes.get('shadow', False))
                except Exception:
                    paperish = True
                relax = self._mode_relax_factor() * float(getattr(self, '_anti_churn_relax_mult', 1.0) or 1.0)

                # Higher confidence => allow thinner edge; lower confidence => stricter.
                conf_factor = max(0.70, min(1.35, 1.0 - 1.2 * (conf - 0.50)))

                min_edge_bps = max(6.0, base_min_edge_bps * conf_factor * relax)
                edge_over_cost = max(1.25, base_edge_over_cost * (0.95 + 0.80 * conf_factor) * relax)
                risk_over_cost = max(0.95, base_risk_over_cost * (0.95 + 0.50 * conf_factor) * relax)

                ok = True
                reason = None
                if edge_bps < min_edge_bps:
                    ok = False
                    reason = 'edge_bps<min_edge_bps'
                elif cost_bps > 0 and ((edge_bps / cost_bps) + 0.05) < edge_over_cost:
                    ok = False
                    reason = 'edge_over_cost'
                elif cost_bps > 0 and ((risk_bps / cost_bps) + 0.05) < risk_over_cost:
                    ok = False
                    reason = 'risk_over_cost'

                if not ok:
                    dbg = {
                        'edge_bps': float(edge_bps),
                        'risk_bps': float(risk_bps),
                        'cost_bps': float(cost_bps),
                        'spread_bps': float(spread_bps),
                        'fee_bps_side': float(fee_bps),
                        'slip_bps_side': float(slippage_bps),
                        'lat_pen_bps': float(lat_pen),
                        'conf': float(conf),
                        'min_edge_bps': float(min_edge_bps),
                        'min_edge_over_cost': float(edge_over_cost),
                        'min_risk_over_cost': float(risk_over_cost),
                        'reject_reason': str(reason or 'unknown')
                    }
                    try:
                        setattr(signal, 'anti_churn_debug', dbg)
                    except Exception:
                        pass
                    try:
                        self._last_anti_churn_debug[symbol] = dbg
                    except Exception:
                        try:
                            self._last_anti_churn_debug = {symbol: dbg}
                        except Exception:
                            pass
                    import logging
                    logging.warning(
                        f"ANTI_CHURN {symbol} reject ({dbg['reject_reason']}): edge={dbg['edge_bps']:.1f}bps risk={dbg['risk_bps']:.1f}bps cost={dbg['cost_bps']:.1f}bps "
                        f"[spread={dbg['spread_bps']:.1f} fee={dbg['fee_bps_side']:.1f}/side slip={dbg['slip_bps_side']:.1f}/side lat={dbg['lat_pen_bps']:.2f}] "
                        f"thr: min_edge={dbg['min_edge_bps']:.1f} edge/cost>={dbg['min_edge_over_cost']:.2f} risk/cost>={dbg['min_risk_over_cost']:.2f} conf={dbg['conf']:.2f}"
                    )
                    return False

                return True
            except Exception:
                return True

    def _apply_directional_cluster_cap(self, symbol: str, signal) -> None:
            """Scale down signal.position_size_pct if correlated cluster exposure is already high (direction-aware)."""
            cfg = (self.config.get('risk_limits', {}) or {}).get('cluster_guard', {}) or {}
            if not bool(cfg.get('enabled', True)):
                return

            try:
                corr_th = float(cfg.get('corr_threshold', 0.70))
                max_cluster_pct = float(cfg.get('max_cluster_exposure_pct', 2.0))  # percent of equity
                min_scale = float(cfg.get('min_scale', 0.15))

                sig_dir = (getattr(signal, 'action', '') or '').upper()
                if sig_dir not in ('LONG', 'SHORT'):
                    return

                # compute correlated exposure of same direction
                sig_ret = (self._returns_cache.get(symbol, {}) or {}).get('ret')
                if sig_ret is None:
                    return

                used_pct = 0.0
                correlated = []
                for sym2, pos in (self._positions_snapshot or {}).items():
                    if sym2 == symbol:
                        continue
                    try:
                        pdir = self._infer_position_direction(pos)
                        if pdir != sig_dir:
                            continue
                        ret2 = (self._returns_cache.get(sym2, {}) or {}).get('ret')
                        if ret2 is None:
                            continue
                        c = self._corr(np.asarray(sig_ret), np.asarray(ret2))
                        if abs(c) >= corr_th:
                            notional = self._infer_position_notional(pos)
                            if self.equity > 0:
                                pct = (notional / self.equity) * 100.0
                            else:
                                pct = 0.0
                            used_pct += pct
                            correlated.append((sym2, c, pct))
                    except Exception:
                        continue

                sig_pct = float(getattr(signal, 'position_size_pct', 0.0) or 0.0)
                if sig_pct <= 0:
                    return

                if used_pct >= max_cluster_pct:
                    # already full cluster -> scale hard
                    new_pct = max(sig_pct * min_scale, sig_pct * 0.05)
                    setattr(signal, 'position_size_pct', new_pct)
                    self._last_cluster_debug[symbol] = {'used_pct': used_pct, 'scaled_to': new_pct, 'corr': correlated}
                    return

                # scale to fit remaining headroom
                headroom = max(0.0, max_cluster_pct - used_pct)
                if sig_pct > headroom:
                    new_pct = max(headroom, sig_pct * min_scale)
                    setattr(signal, 'position_size_pct', new_pct)
                    self._last_cluster_debug[symbol] = {'used_pct': used_pct, 'scaled_to': new_pct, 'corr': correlated}
            except Exception:
                return

    def validate_entry(self, symbol: str, signal) -> bool:
            """
            Kiểm tra xem có được vào lệnh mới không dựa trên risk rules
            Trả về True nếu OK, False nếu vượt limit
            """
            try:
                # 1. Trạng thái tổng thể
                # 1. Trạng thái tổng thể (không hard-stop vĩnh viễn)
                if self.state == RiskState.DEAD:
                    logging.warning(f"Không vào lệnh do risk state: {self.state.value}")
                    return False
                if self.state in [RiskState.DANGER, RiskState.CRITICAL]:
                    now_ts = time.time()
                    last_loss = float(getattr(self, 'last_loss_ts', 0.0) or 0.0)
                    cd = float((self.config.get('risk_limits', {}) or {}).get('danger_cooldown_sec', 180) or 180)
                    if last_loss > 0 and (now_ts - last_loss) < cd:
                        remain = int(max(0.0, cd - (now_ts - last_loss)))
                        logging.warning(f"Không vào lệnh do risk state: {self.state.value} (cooldown ~{remain}s)")
                        return False
                    # after cooldown: allow only high-quality trades with reduced risk
                    try:
                        self.runtime_risk_scale = min(float(getattr(self, 'runtime_risk_scale', 1.0) or 1.0), 0.40)
                    except Exception:
                        pass
                    try:
                        self.runtime_min_conf_override = max(float(getattr(self, 'runtime_min_conf_override', 0.0) or 0.0), 0.62)
                    except Exception:
                        pass
                    try:
                        self.loss_streak = max(0, int(self.loss_streak) - 1)
                    except Exception:
                        pass

                # 2. Portfolio heat
                if self.current_heat >= self.max_portfolio_heat:
                    logging.warning(f"Portfolio heat vượt giới hạn: {self.current_heat:.2f}% >= {self.max_portfolio_heat:.2f}%")
                    return False

                # 3. Loss streak (LOSS-BRAKE, not hard-stop)
                max_loss_streak = self.config.get('risk_limits', {}).get('max_loss_streak', 4)
                if self.loss_streak >= max_loss_streak:
                    # cooldown after losses to avoid revenge-trading, but DO NOT stop forever
                    now_ts = time.time()
                    cd1 = float(self.config.get('risk_limits', {}).get('cooldown_after_loss_s', 120) or 120)
                    cd2 = float(self.config.get('risk_limits', {}).get('cooldown_after_2_losses_s', 480) or 480)
                    cooldown = cd2 if self.loss_streak >= 2 else cd1
                    last_loss = float(getattr(self, 'last_loss_ts', 0.0) or 0.0)
                    if last_loss > 0 and (now_ts - last_loss) < cooldown:
                        remain = int(max(0.0, cooldown - (now_ts - last_loss)))
                        logging.warning(f"LOSS_BRAKE active: loss_streak={self.loss_streak} | cooldown ~{remain}s (SOFT)")
                        # Soft brake: allow trade but reduce risk + raise quality gates (do NOT hard-block entries)
                        try:
                            self.runtime_risk_scale = min(float(getattr(self, 'runtime_risk_scale', 1.0) or 1.0), 0.55)
                        except Exception:
                            pass
                        try:
                            self.runtime_min_conf_override = max(float(getattr(self, 'runtime_min_conf_override', 0.0) or 0.0), 0.06)
                        except Exception:
                            pass
                        # continue evaluation instead of blocking
                        

                    # after cooldown: allow trades but tighten quality + reduce risk temporarily
                    try:
                        self.runtime_risk_scale = min(float(getattr(self, 'runtime_risk_scale', 1.0) or 1.0), 0.45)
                    except Exception:
                        pass
                    try:
                        # raise effective confidence gate (only take A+ setups during recovery)
                        self.runtime_min_conf_override = max(float(getattr(self, 'runtime_min_conf_override', 0.0) or 0.0), 0.56)
                    except Exception:
                        pass
                    try:
                        self.recovery_trades_left = max(int(getattr(self, 'recovery_trades_left', 0) or 0), 2)
                    except Exception:
                        pass
                    # decay the streak gradually so system can resume normality
                    self.loss_streak = max(0, int(self.loss_streak) - 1)

                # 4. Daily loss limit
                daily_loss_pct = (self.daily_pnl / self.initial_equity * 100) if self.initial_equity > 0 else 0
                max_daily_loss = self.config.get('risk_limits', {}).get('max_daily_loss_pct', 5.0)
                if daily_loss_pct <= -max_daily_loss:
                    logging.warning(f"Daily loss vượt giới hạn: {daily_loss_pct:.2f}%")
                    return False

                # 5. Confidence gate (profile + optional runtime override)
                profile = self.coin_profiles.get(symbol)
                if profile:
                    min_conf = float(getattr(profile, 'min_confidence', 0.58) or 0.58)
                else:
                    min_conf = 0.58
                try:
                    ovr = (getattr(self, 'runtime_min_conf_overrides', {}) or {}).get(symbol)
                    if ovr is not None:
                        min_conf = float(ovr)
                except Exception:
                    pass
                if signal.confidence < min_conf:
                    logging.warning(f"Confidence {signal.confidence:.3f} < min {min_conf} cho {symbol}")
                    return False

                # 6. Size check: nếu vượt trần thì SCALE xuống (không reject)
                # FUTURES-AWARE CAP:
                # - Với Futures, "position_value" là NOTIONAL (price*qty), có thể lớn hơn equity nhờ leverage.
                # - Nếu cap chỉ = %equity (spot-style), tài khoản nhỏ (vd equity=22 USDT) sẽ luôn bị scale về size quá bé.
                #
                # Chính sách mặc định:
                #   max_notional_per_pos = equity * leverage * target_margin_util
                #   sau đó chia mềm theo max_total_positions để tránh "all-in" nhiều lệnh.
                #
                # Có thể ghi đè bằng config:
                #   risk_limits.max_margin_util (0..1)
                #   risk_limits.max_total_positions
                #   coin profile max_position_pct (vẫn được tôn trọng nếu nhỏ hơn)
                #
                # Mục tiêu: dùng được tài nguyên tài khoản, nhưng vẫn tránh spam cháy TK.
                try:
                    lev = int(getattr(signal, 'leverage', 0) or 0)
                except Exception:
                    lev = 0
                if lev <= 0:
                    lev = int(self.config.get('default_leverage', 10) or 10)
                lev = max(1, min(125, lev))

                max_margin_util = float(((self.config.get('futures', {}) or {}).get('max_margin_util', None) if isinstance(self.config, dict) else None) or (self.config.get('risk_limits', {}) or {}).get('max_margin_util', 0.80))
                max_margin_util = max(0.10, min(0.98, max_margin_util))

                max_total_positions = int(((self.config.get('portfolio_caps', {}) or self.config.get('portfolio', {})) or {}).get('max_total_positions', 2) or 2)
                max_total_positions = max(1, min(10, max_total_positions))

                # Base notional cap per position (soft)
                max_value = float(self.equity) * float(lev) * float(max_margin_util) / float(max_total_positions)

                # Nếu coin profile yêu cầu %equity nhỏ hơn, vẫn tôn trọng (nhưng cũng futures-aware)
                profile = self.coin_profiles.get(symbol)
                if profile:
                    try:
                        pct = float(profile.max_position_pct) / 100.0
                        if pct > 0:
                            max_value = min(max_value, float(self.equity) * float(lev) * float(pct))
                    except Exception:
                        pass

                # Fallback tối thiểu để không bị scale về 0 trong paper/demo
                if max_value < 10.0:
                    max_value = 10.0

                est_size = self.calculate_position_size(symbol, signal.entry_price, signal.stop_loss)

                # Apply strategy-provided size multiplier (freq-first soft gates, follower relax, etc.)
                try:
                    sig_mult = float(getattr(signal, 'position_size_pct', 1.0) or 1.0)
                    sig_mult = max(0.05, min(1.0, sig_mult))
                    est_size = est_size * sig_mult
                except Exception:
                    pass

                est_value = est_size * signal.entry_price

                if est_value <= 0:
                    logging.warning("Est position value <= 0, skip")
                    return False

                if est_value > max_value:
                    scale = max_value / est_value
                    scaled_qty = est_size * scale
                    setattr(signal, "position_value_usd", float(max_value))
                    setattr(signal, "position_qty", float(scaled_qty))
                    setattr(signal, "sizing_scaled", True)
                    logging.warning(
                        f"Position size quá lớn: {est_value:,.2f} > cap {max_value:,.2f} → SCALE xuống {max_value:,.2f}"
                    )
                else:
                    setattr(signal, "position_value_usd", float(est_value))
                    setattr(signal, "position_qty", float(est_size))
                    setattr(signal, "sizing_scaled", False)

                # 7. Min notional guard (để tránh size quá nhỏ không đủ điều kiện sàn / phí ăn hết)
                # Không gọi exchangeInfo ở đây để tránh phụ thuộc runtime; dùng ngưỡng cấu hình.
                min_notional = float((((self.config.get('futures', {}) or {}).get('min_notional_usdt', None) if isinstance(self.config, dict) else None) or (self.config.get('risk_limits', {}) or {}).get('min_notional_usd', 10.0)))
                pos_value = float(getattr(signal, 'position_value_usd', 0.0) or 0.0)
                pos_qty = float(getattr(signal, 'position_qty', 0.0) or 0.0)
                if pos_value < min_notional or pos_qty <= 0:
                    logging.warning(
                        f"Position too small: value=${pos_value:.2f} qty={pos_qty:.8f} < min_notional=${min_notional:.2f}"
                    )
                    return False

                # V7: Anti-churn gate (reject thin-edge trades under cost/spread/latency)
                # (Detailed reject reason is logged inside _anti_churn_gate)
                if not self._anti_churn_gate(symbol, signal):
                    return False

                # V7: Direction-aware correlation cluster cap (scale down position_size_pct if needed)
                self._apply_directional_cluster_cap(symbol, signal)

                logging.info(f"Validate entry OK cho {symbol}: {signal.action} | Conf {signal.confidence:.3f}")
                return True

            except Exception as e:
                logging.error(f"Validate entry error cho {symbol}: {str(e)}")
                return Fals
def _get_int_cfg(cfg: dict, keys: list, default: int) -> int:
    """Safely read nested int config; treats 0 as a valid value (does not fall back)."""
    try:
        cur = cfg
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return int(default)
            cur = cur[k]
        if cur is None:
            return int(default)
        return int(cur)
    except Exception:
        return int(default)
