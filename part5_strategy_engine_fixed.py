"""
Part 5: Strategy Engine - OPTIMIZED PROFIT & LIVE-LIKE VERSION 2026 (Tối ưu lợi nhuận cao cấp)
- RR >= 1:4 for scalping high winrate
- Trailing stop động sớm (1.2*ATR)
- Giảm threshold để 4-10 lệnh/ngày
- Add MACD crossover filter (MACD > signal for LONG)
- FIX tất cả lỗi trước
- NÂNG CẤP: Giảm min_conf=0, min_vol=0, loosen RSI >20 for LONG, add log if no signal
- PRO: Add grid strategy (15 levels ±3%, profit 0.5% per level), Martingale (double on loss, max 4)
- FIX BACKTEST: Pass df to extract_features/generate_signals for historical data
"""

import logging
import time
import numpy as np
import pandas as pd
from part1_security_marketdata_fixed import OrderBookSnapshot
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from error_handling import handle_errors, InvalidSignalError, DataError

class RegimeType(Enum):
    STRONG_UPTREND = "STRONG_UPTREND"
    UPTREND = "UPTREND"
    RANGING = "RANGING"
    DOWNTREND = "DOWNTREND"
    STRONG_DOWNTREND = "STRONG_DOWNTREND"
    VOLATILE = "VOLATILE"
    CHOPPY = "CHOPPY"

class SessionType(Enum):
    ASIAN = "ASIAN"
    EUROPEAN = "EUROPEAN"
    US = "US"
    OVERLAP_EU_US = "OVERLAP_EU_US"

@dataclass
class TradingSignal:
    symbol: str
    action: str
    timestamp: datetime
    entry_price: float
    stop_loss: float
    take_profit: float
    current_price: float
    position_size_pct: float = 1.0
    risk_reward_ratio: float = 0.0
    ml_probability: float = 0.5
    confidence: float = 0.5
    regime: str = "RANGING"
    btc_regime: str = "RANGING"
    volatility: float = 0.0
    rsi: float = 50.0
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    bb_position: float = 0.5
    trend_strength: float = 0.0
    order_flow_delta: float = 0.0
    aggressive_buying: bool = False
    aggressive_selling: bool = False
    bid_ask_imbalance: float = 0.0
    tf_5m_aligned: bool = False
    tf_15m_aligned: bool = False
    tf_1h_aligned: bool = False
    tf_4h_aligned: bool = False
    liquidity_score: float = 50.0
    spread_pct: float = 0.1
    slippage_estimate: float = 0.1
    session: str = "ASIAN"
    reason: str = ""
    features: Dict = field(default_factory=dict)
    trailing_active: bool = False
    trailing_offset: float = 0.0
    
    def __post_init__(self):
        if self.stop_loss > 0 and self.entry_price > 0:
            risk = abs(self.entry_price - self.stop_loss)
            reward = abs(self.take_profit - self.entry_price)
            if risk > 0:
                self.risk_reward_ratio = reward / risk
        atr = self.features.get('atr', 0.01)
        self.trailing_offset = 1.0 * atr
        self.trailing_active = False

class CompleteStrategyEngine:
    def __init__(self, market_data, brain, config: Dict):
        self.market_data = market_data
        self.brain = brain
        self.config = config
        
        # v10: runtime-tunable quality gates (used by the self-evolving tuner)
        # FREQ-FIRST defaults (target ~5-15+ trades/day across 4 symbols).
        # These are still guarded by risk controller + execution throttles.
        self.min_volatility = float(self.config.get('min_volatility', 0.00035) or 0.00035)
        self.max_volatility = 0.05
        self.min_momentum = 0.0
        self.min_confidence = float(self.config.get('min_confidence', 0.22) or 0.22)
        self.rr_min = float(self.config.get('rr_min', 1.08) or 1.08)
        # Adaptive parameters (tuned by AdaptiveEngine)
        self.compression_atr_ratio_max = float(self.config.get('compression_atr_ratio_max', 0.92) or 0.92)
        # Enable RANGE mode by default to increase frequency in chop.
        self.allow_range_mode = bool(self.config.get('allow_range_mode', True))
        self.range_rsi_oversold = 32
        self.range_rsi_overbought = 68
        self.range_dev_ma20_min = float(self.config.get('range_dev_ma20_min', 0.0012) or 0.0012)
        self.follower_signal_age_min = self.config.get('follower_signal_age_min') or {'SOLUSDT': 60, 'XRPUSDT': 75, 'ETHUSDT': 90}
        self.follower_requirements = {
            # Lower BTC strength requirements to unlock follower trades; still filtered by wick/ext.
            'SOLUSDT': {'min_btc_strength': 0.16, 'max_range_atr': 2.2, 'max_ext_ma50': 0.030},
            'XRPUSDT': {'min_btc_strength': 0.15, 'max_range_atr': 2.4, 'max_ext_ma50': 0.028},
            'ETHUSDT': {'min_btc_strength': 0.14, 'max_range_atr': 2.6, 'max_ext_ma50': 0.024},
        }
        self.allow_follower_when_btc_unclear = bool(self.config.get('allow_follower_when_btc_unclear', True))
        self.allow_contra_followers_in_range = bool(self.config.get('allow_contra_followers_in_range', True))
        # When BTC is unclear, require slightly higher confidence but do NOT fully block followers.
        self.follower_unclear_high_conf = float(self.config.get('follower_unclear_high_conf', 0.58) or 0.58)
        self.follower_unclear_soft_conf = float(self.config.get('follower_unclear_soft_conf', 0.45) or 0.45)
        self.follower_size_mult_unclear = float(self.config.get('follower_size_mult_unclear', 0.60) or 0.60)
        self.follower_size_mult_contra = float(self.config.get('follower_size_mult_contra', 0.60) or 0.60)

        self.fee_rate = float(self.config.get('fee_rate', 0.0004) or 0.0004)
        # Frequency-first: don't penalize ASIAN too hard.
        self.preferred_sessions = self.config.get('preferred_sessions') or [SessionType.ASIAN.value, SessionType.US.value, SessionType.OVERLAP_EU_US.value]

        # Cost gate tunables (frequency-first defaults). Can be tuned by AdaptiveEngine.
        # - reward_mult: lower -> more trades (still fee-aware)
        # - risk_mult  : lower -> allow tighter stops (but anti stop-hunt is still on)
        self.cost_gate_reward_mult = float(self.config.get('cost_gate_reward_mult', 1.01) or 1.05)
        self.cost_gate_risk_mult = float(self.config.get('cost_gate_risk_mult', 0.75) or 0.75)
        self.cost_gate_slip_cushion = float(self.config.get('cost_gate_slip_cushion', 0.00015) or 0.00015)
        
        self.backtest_mode = self.config.get('backtesting', {}).get('enabled', False)

        # BTC leader context (updated by TradingSystem each scan)
        self.btc_context = {
            'direction': None,   # 'LONG'/'SHORT'/None
            'strength': 0.0,     # 0..1
            'confidence': 0.0,   # 0..1
            'regime': 'RANGING',
            'ts': None,
            # For "BTC leader tuyệt đối": followers only trade when BTC signal is fresh
            'last_signal_ts': None,
            'last_signal_action': None,
        }
        # Track last rejection for stats/debug
        self.last_reject = {'symbol': None, 'code': None, 'reason': None}

    def set_runtime_params(self, **kwargs):
        """Allow TradingSystem (or tuner) to adjust gates without restarting."""
        for k, v in (kwargs or {}).items():
            if hasattr(self, k):
                try:
                    setattr(self, k, v)
                except Exception:
                    pass
    
    def update_btc_context(self, btc_features: Dict, btc_signal: Optional[TradingSignal] = None) -> Dict:
        """Update BTC leader context.

        We derive a robust direction even when no explicit BTC signal is produced,
        using trend + MA50 alignment. Followers can only trade when BTC direction is clear.
        """
        try:
            f = btc_features or {}
            momo = float(f.get('price_momentum', 0.0) or 0.0)
            close = float(f.get('close', 0.0) or 0.0)
            ma50 = float(f.get('ma50', close) or close)
            conf = float(getattr(btc_signal, 'confidence', 0.0) or 0.0)
            trend = abs(float(f.get('trend_strength', 0.0) or 0.0))

            # Direction: prefer explicit signal, else fall back to MA+momentum
            direction = None
            if btc_signal is not None and getattr(btc_signal, 'action', None) in ('LONG', 'SHORT'):
                direction = btc_signal.action
            else:
                # Primary: aligned momentum + MA50
                if momo > 0 and close >= ma50:
                    direction = 'LONG'
                elif momo < 0 and close <= ma50:
                    direction = 'SHORT'
                else:
                    # Fallback (freq-first): if momentum is meaningful, still provide a WEAK direction
                    # so followers don't starve under BTC_GATE when BTC is chopping.
                    momoref = float(getattr(self, 'btc_momo_dir_threshold', 0.0004) or 0.0004)
                    if abs(momo) >= momoref:
                        direction = 'LONG' if momo > 0 else 'SHORT'
                    else:
                        # Secondary: MACD crossover hint
                        mc = int(f.get('macd_crossover', 0) or 0)
                        ma20 = float(f.get('ma20', close) or close)
                        if mc == 1 and close >= ma20:
                            direction = 'LONG'
                        elif mc == -1 and close <= ma20:
                            direction = 'SHORT'

            # Fresh BTC signal timestamp:
            # - Followers require a "recent BTC direction". In practice, explicit BTC signals
            #   can be sparse; if we have a clear derived direction, we still refresh the
            #   freshness clock to prevent BTC_GATE_STALE starving follower trades.
            last_signal_ts = (self.btc_context or {}).get('last_signal_ts')
            last_signal_action = (self.btc_context or {}).get('last_signal_action')
            now_ts = datetime.now().isoformat()
            if btc_signal is not None and getattr(btc_signal, 'action', None) in ('LONG', 'SHORT'):
                last_signal_ts = now_ts
                last_signal_action = btc_signal.action
            else:
                # Derived direction: keep a freshness stamp as well (frequency-first)
                if direction in ('LONG', 'SHORT'):
                    last_signal_ts = now_ts
                    last_signal_action = direction

            # Strength: combine confidence + trend magnitude (normalized)
            # trend_strength here is momentum*20 in extract_features.
            trend_norm = min(1.0, trend / 0.06)  # 0.06 ~ strong impulse
            momo_norm = min(1.0, abs(momo) / float(getattr(self, 'btc_momo_norm_ref', 0.0030) or 0.0030))
            # Combine: model confidence + trend + momentum (so chop still yields low-but-nonzero strength)
            strength = min(1.0, 0.45 * conf + 0.35 * trend_norm + 0.20 * momo_norm)

            regime = 'RANGING'
            if direction == 'LONG':
                regime = 'UPTREND' if strength < 0.75 else 'STRONG_UPTREND'
            elif direction == 'SHORT':
                regime = 'DOWNTREND' if strength < 0.75 else 'STRONG_DOWNTREND'

            self.btc_context = {
                'direction': direction,
                'strength': float(strength),
                'confidence': float(conf),
                'regime': regime,
                'ts': datetime.now().isoformat(),
                'last_signal_ts': last_signal_ts,
                'last_signal_action': last_signal_action,
            }
        except Exception:
            # Keep last context if anything goes wrong
            pass
        return dict(self.btc_context or {})

    def get_min_volatility(self, symbol: str, features: Dict) -> float:
        """Adaptive volatility gate to avoid 'no trades' during quiet regimes.
        Returns a per-scan min volatility (ATR/price) threshold."""
        try:
            base = float(getattr(self, 'min_volatility', 0.0006) or 0.0006)
        except Exception:
            base = 0.0006
        # Per-symbol tweak: XRP often has lower 5m ATR/price
        try:
            if symbol.upper().startswith('XRP'):
                base = min(base, 0.00045)
        except Exception:
            pass
        # Relax in compression regimes
        try:
            atr_ratio = float(features.get('atr_ratio', 1.0) or 1.0)
            thr = float(getattr(self, 'compression_atr_ratio_max', 0.78) or 0.78)
            if atr_ratio < thr:
                base *= 0.65
        except Exception:
            pass
        # Starvation relax: if no trades for a while, lower the gate temporarily
        try:
            last_ts = float(getattr(self, '_last_trade_ts', 0.0) or 0.0)
            now_ts = time.time()
            relax_after = float(getattr(self, 'starvation_relax_after_s', 600.0) or 600.0)
            if last_ts <= 0.0 or (now_ts - last_ts) >= relax_after:
                base *= 0.55
        except Exception:
            pass
        # Hard floor to prevent extreme noise-trading
        return max(0.00025, float(base))

    def get_current_session(self) -> str:
        now = datetime.now().hour
        if 0 <= now < 7:
            return "ASIAN"
        elif 7 <= now < 13:
            return "EUROPEAN"
        elif 13 <= now < 17:
            return "OVERLAP_EU_US"
        else:
            return "US"
    
    
    def _get_htf_bias(self, symbol: str, now_ts: float) -> tuple:
        """Return (dir, strength) where dir in {-1,0,1}. Cached ~20s."""
        try:
            cache = self._htf_cache.get(symbol)
            if cache and (now_ts - cache.get('ts', 0.0) < 20.0):
                return int(cache.get('dir', 0)), float(cache.get('strength', 0.0))
            # Fetch HTF candles
            df15 = self.market_data.get_klines(symbol, '15m', limit=200)
            df1h = self.market_data.get_klines(symbol, '1h', limit=200)
            if df15 is None or df1h is None or len(df15) < 60 or len(df1h) < 60:
                self._htf_cache[symbol] = {'ts': now_ts, 'dir': 0, 'strength': 0.0}
                return 0, 0.0

            def _ma_dir(df):
                # robust close column
                ccol = 'close' if 'close' in df.columns else ('Close' if 'Close' in df.columns else None)
                if not ccol:
                    return 0, 0.0
                close = df[ccol].astype(float)
                ma20 = close.rolling(20).mean()
                ma50 = close.rolling(50).mean()
                if len(ma50) < 55:
                    return 0, 0.0
                d = 1 if (ma20.iloc[-1] > ma50.iloc[-1]) else (-1 if (ma20.iloc[-1] < ma50.iloc[-1]) else 0)
                # slope strength from ma20 change
                slope = float((ma20.iloc[-1] - ma20.iloc[-6]) / max(abs(ma20.iloc[-6]), 1e-12))
                return d, slope

            d15, s15 = _ma_dir(df15)
            d1h, s1h = _ma_dir(df1h)
            # Align bias only if both agree; else neutral
            if d15 != 0 and d15 == d1h:
                d = d15
                strength = float(abs(s15) * 80.0 + abs(s1h) * 120.0)
            else:
                d = 0
                strength = float(abs(s15) * 40.0 + abs(s1h) * 60.0)

            self._htf_cache[symbol] = {'ts': now_ts, 'dir': int(d), 'strength': float(strength)}
            return int(d), float(strength)
        except Exception:
            # Never block trading due to HTF issues
            return 0, 0.0

    def extract_features(self, symbol: str, df: Optional[pd.DataFrame] = None) -> Dict:  # Pass df for backtest
        try:
            if df is None:  # Live mode
                df = self.market_data.get_klines(symbol, '5m', limit=200)
            if df is None or len(df) < 50:
                logging.debug(f"No data for {symbol}")
                return {}
    
            # --- Robust OHLC extraction ---
            # Some data sources (or accidental upstream transforms) may change column casing.
            # When this happens, downstream code used to crash with UnboundLocalError.
            # We normalize here and fail gracefully if required columns are missing.
            def _col(name: str):
                if name in df.columns:
                    return name
                alt = name.capitalize()
                if alt in df.columns:
                    return alt
                alt2 = name.upper()
                if alt2 in df.columns:
                    return alt2
                return None
    
            ccol = _col('close')
            hcol = _col('high')
            lcol = _col('low')
            if not ccol or not hcol or not lcol:
                logging.error(f"[{symbol}] Missing OHLC columns: close={ccol}, high={hcol}, low={lcol} | cols={list(df.columns)}")
                return {}
    
            close = df[ccol].astype(float).values
            high = df[hcol].astype(float).values
            low = df[lcol].astype(float).values
            
            # RSI (Wilder) - stable for live gating
            delta = np.diff(close)
            gain = np.maximum(delta, 0.0)
            loss = np.maximum(-delta, 0.0)
            period = 14
            rsi = 50.0
            if len(delta) >= period + 1:
                avg_gain = float(np.mean(gain[:period]))
                avg_loss = float(np.mean(loss[:period]))
                for i in range(period, len(gain)):
                    avg_gain = (avg_gain * (period - 1) + float(gain[i])) / period
                    avg_loss = (avg_loss * (period - 1) + float(loss[i])) / period
                rs = avg_gain / max(avg_loss, 1e-10)
                rsi = 100.0 - (100.0 / (1.0 + rs))
            
            # True Range / ATR (fix bug: np.maximum chỉ nhận 2 tham số)
            # TR_t = max(high-low, abs(high-prev_close), abs(low-prev_close))
            if len(close) < 16:
                return {}
            prev_close = close[:-1]
            tr1 = high[1:] - low[1:]
            tr2 = np.abs(high[1:] - prev_close)
            tr3 = np.abs(low[1:] - prev_close)
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            atr_s = pd.Series(tr)
            atr = float(atr_s.rolling(window=14).mean().iloc[-1]) if len(tr) >= 14 else float(np.mean(tr))
    
            # ATR compression / expansion (helps avoid chop & catch bigger TP legs)
            atr_ma50 = float(atr_s.rolling(window=50).mean().iloc[-1]) if len(tr) >= 50 else float(atr_s.mean())
            atr_ratio = float(atr / max(atr_ma50, 1e-12))
            
            volatility = atr / close[-1]
            momentum = (close[-1] - close[-20]) / close[-20] if len(close) >= 20 else 0
            
            trend_strength = momentum * 20
            
            # MACD series (fix bug: trước đó macd là scalar -> signal luôn bằng chính nó)
            close_s = pd.Series(close)
            ema12_s = close_s.ewm(span=12, adjust=False).mean()
            ema26_s = close_s.ewm(span=26, adjust=False).mean()
            macd_s = ema12_s - ema26_s
            signal_s = macd_s.ewm(span=9, adjust=False).mean()
            macd = float(macd_s.iloc[-1])
            macd_signal = float(signal_s.iloc[-1])
            # crossover theo hướng + xác nhận (tránh flip liên tục)
            try:
                prev_macd = float(macd_s.iloc[-2])
                prev_sig = float(signal_s.iloc[-2])
            except Exception:
                prev_macd, prev_sig = macd, macd_signal
    
            macd_crossover = 0
            if macd > macd_signal and prev_macd <= prev_sig:
                macd_crossover = 1
            elif macd < macd_signal and prev_macd >= prev_sig:
                macd_crossover = -1
            
            # Add MA50/MA20 for trend & breakout context
            close_ser = pd.Series(close)
            ma50 = float(close_ser.rolling(window=50).mean().iloc[-1])
            ma20 = float(close_ser.rolling(window=20).mean().iloc[-1])
    
            # Breakout context (20-bar high/low excluding current candle)
            try:
                hh20 = float(np.max(high[-21:-1]))
                ll20 = float(np.min(low[-21:-1]))
            except Exception:
                hh20 = float(np.max(high[-20:]))
                ll20 = float(np.min(low[-20:]))
    
            # Liquidity sweep / false-break detection (helps catch local tops/bottoms)
            # Sweep HIGH: price wicks above previous HH but closes back below it (bull trap)
            # Sweep LOW : price wicks below previous LL but closes back above it (bear trap)
            try:
                prev_hh20 = float(hh20)
                prev_ll20 = float(ll20)
                cur_h = float(high[-1])
                cur_l = float(low[-1])
                cur_c = float(close[-1])
                sweep_high = (cur_h > (prev_hh20 + 0.10 * float(atr))) and (cur_c < prev_hh20)
                sweep_low  = (cur_l < (prev_ll20 - 0.10 * float(atr))) and (cur_c > prev_ll20)
            except Exception:
                sweep_high = False
                sweep_low = False
    
            # Wider swing context (50 bars) for structural TP/SL decisions
            try:
                hh50 = float(np.max(high[-51:-1]))
                ll50 = float(np.min(low[-51:-1]))
            except Exception:
                hh50 = float(hh20)
                ll50 = float(ll20)
    
            # VWAP (simple typical-price volume-weighted, 50 bars)
            try:
                if 'volume' in df.columns:
                    vol = df['volume'].values
                elif 'quote_volume' in df.columns:
                    vol = df['quote_volume'].values
                else:
                    vol = None
                if vol is not None and len(vol) >= 50:
                    tp = (high + low + close) / 3.0
                    vv = vol[-50:]
                    tt = tp[-50:]
                    denom = float(vv.sum())
                    vwap = float((tt * vv).sum() / denom) if denom > 0 else float(close[-1])
                else:
                    vwap = float(close[-1])
            except Exception:
                vwap = float(close[-1])
            
            htf_dir, htf_strength = self._get_htf_bias(symbol, time.time())
            features = {
                'volatility': float(volatility),
                'htf_dir': int(htf_dir),
                'htf_strength': float(htf_strength),
                'price_momentum': float(momentum),
                'rsi': float(rsi),
                'trend_strength': float(trend_strength),
                'atr': float(atr),
                'close': float(close[-1]),
                'macd_crossover': macd_crossover,
                'macd': macd,
                'macd_signal': macd_signal,
                'ma50': ma50,
                'ma20': ma20,
                'range_atr': float((high[-1] - low[-1]) / max(atr, 1e-12)),
                'ext_ma50_pct': float(abs(close[-1] - ma50) / max(close[-1], 1e-12)),
                'atr_ratio': atr_ratio,
                'hh20': hh20,
                'll20': ll20,
                'sweep_high': bool(sweep_high),
                'sweep_low': bool(sweep_low),
                'cur_high': float(cur_h) if 'cur_h' in locals() else float(high[-1]),
                'cur_low': float(cur_l) if 'cur_l' in locals() else float(low[-1]),
                'prev_hh20': float(prev_hh20) if 'prev_hh20' in locals() else float(hh20),
                'prev_ll20': float(prev_ll20) if 'prev_ll20' in locals() else float(ll20),
                'hh50': float(hh50),
                'll50': float(ll50),
                'vwap': float(vwap),
                'dist_vwap_pct': float((close[-1] - vwap) / max(close[-1], 1e-12)),
            }
            return features
        except Exception as e:
            logging.error(f"Error extract features {symbol}: {e}")
            return {}
    
    def generate_signals(self, symbol: str, features: Dict) -> Optional[TradingSignal]:
        if not features:
            logging.debug(f"No features for {symbol}")
            return None

        def _rej(code: str, reason: str):
            # Save last reject for TradingSystem stats/debug
            try:
                self.last_reject = {'symbol': symbol, 'code': code, 'reason': reason}
            except Exception:
                pass
            logging.debug(f"Reject {symbol}: {code} | {reason}")
            return None
            
        prob_long, prob_short = self.brain.predict_entry_probability(symbol, features)
        confidence_ml = max(prob_long, prob_short)
        # ML can be uninformative early on (near-0.5). We compute a rule-based confidence later and blend.

        score = 0.0
        reasons = []

        vol_ok = self.min_volatility < features['volatility'] < self.max_volatility
        if vol_ok: score += 1.0; reasons.append("Vol OK")
        else:
            logging.debug(f"Vol not ok for {symbol}: {features['volatility']}")

        try:
            min_momo = float(self.min_momentum)
            if bool(getattr(self, '_starve_relax', False)):
                min_momo *= float(getattr(self, 'starvation_momentum_mult', 0.60) or 0.60)
        except Exception:
            min_momo = float(getattr(self, 'min_momentum', 0.0012) or 0.0012)
        momo_ok = abs(features['price_momentum']) > min_momo
        if momo_ok: score += 1.0; reasons.append("Momo OK")
        else:
            logging.debug(f"Momo not ok for {symbol}: {features['price_momentum']}")

        rsi = features['rsi']
        # RSI gate: keep loose but directional (avoid overbought chasing)
        rsi_ok = (rsi > 35 and features['price_momentum'] > 0) or (rsi < 65 and features['price_momentum'] < 0)
        if rsi_ok: score += 1.0; reasons.append("RSI OK")
        else:
            logging.debug(f"RSI not ok for {symbol}: {rsi}")

        try:
            min_trend = 0.0015
            if bool(getattr(self, '_starve_relax', False)):
                min_trend *= float(getattr(self, 'starvation_trend_mult', 0.70) or 0.70)
        except Exception:
            min_trend = 0.0015
        trend_ok = abs(features['trend_strength']) > min_trend
        if trend_ok: score += 1.0; reasons.append("Trend OK")
        else:
            logging.debug(f"Trend not ok for {symbol}: {features['trend_strength']}")

        macd_ok = features.get('macd_crossover', 0)
        if macd_ok == 1 and features['price_momentum'] > 0:
            score += 1.5; reasons.append("MACD Bullish")
        elif macd_ok == -1 and features['price_momentum'] < 0:
            score += 1.5; reasons.append("MACD Bearish")
        else:
            logging.debug(f"MACD not ok for {symbol}: {macd_ok}")

        current_session = self.get_current_session()
        session_bonus = 1.2 if current_session in self.preferred_sessions else 0.8
        score += session_bonus

        # Base minimum score; relaxed dynamically if the system is starved of trades.
        min_score = 0.5
        try:
            last_ts = float(getattr(self, '_last_trade_ts', 0.0) or 0.0)
            now_ts = time.time()
            relax_after = float(getattr(self, 'starvation_relax_after_s', 600.0) or 600.0)
            if last_ts <= 0.0 or (now_ts - last_ts) >= relax_after:
                # Trade-starvation mode: loosen gates to collect data, but keep directional filters.
                min_score = float(getattr(self, 'starvation_min_score', 0.28) or 0.28)
                # Also relax momentum/trend thresholds a bit (local only).
                try:
                    self._starve_relax = True
                except Exception:
                    pass
        except Exception:
            pass
        # Provisional confidence (pre-HTF/ML blend) for gating decisions later
        try:
            ts = abs(float(features.get('trend_strength', 0.0) or 0.0))
            ts_n = max(0.0, min(1.0, (ts - 0.0012) / 0.0048))
            provisional_conf = 0.20 + 0.10 * min(6.0, float(score)) + 0.25 * ts_n
            provisional_conf = max(0.20, min(0.95, float(provisional_conf)))
        except Exception:
            provisional_conf = 0.35
        if score < min_score:
            logging.debug(f"Score low for {symbol}: {score} < {min_score}")
            return None

        # Compression -> breakout condition (helps "húp nhiều TP" and avoids chop)
        atr_ratio = float(features.get('atr_ratio', 1.0) or 1.0)
        try:
            thr = float(getattr(self, 'compression_atr_ratio_max', 0.78) or 0.78)
        except Exception:
            thr = 0.78
        in_compression = atr_ratio < thr
        close_px = float(features.get('close', 0.0) or 0.0)
        hh20 = float(features.get('hh20', close_px) or close_px)
        ll20 = float(features.get('ll20', close_px) or close_px)
        breakout_up = close_px > hh20
        breakout_dn = close_px < ll20


        # Liquidity sweep reversal (catch local tops/bottoms with wick reclaim)
        # NOTE: This is an override layer applied *after* we compute the base action.
        # It helps enter near wick-tops/wick-bottoms when a false-break happens.

        # ----------------------
        # Base action decision
        # ----------------------
        if features['price_momentum'] > 0 and rsi > 35 and trend_ok and (macd_ok == 1 or (in_compression and breakout_up)):
            action = "LONG"
        elif (
            # Trend pullback continuation: many trades, higher winrate
            float(features.get('close', 0.0) or 0.0) > float(features.get('ma50', features.get('close', 0.0)) or features.get('close', 0.0))
            and float(features.get('ma20', features.get('close', 0.0)) or features.get('close', 0.0)) >= float(features.get('ma50', features.get('close', 0.0)) or features.get('close', 0.0))
            and float(features.get('price_momentum', 0.0) or 0.0) > 0
            and 40 <= float(rsi) <= 62
            and float(features.get('macd', 0.0) or 0.0) >= float(features.get('macd_signal', 0.0) or 0.0)
            and abs((float(features.get('close', 0.0) or 0.0) - float(features.get('ma20', features.get('close', 0.0)) or features.get('close', 0.0))) / max(float(features.get('ma20', features.get('close', 0.0)) or features.get('close', 0.0)), 1e-12)) <= float(getattr(self, 'pullback_ma20_dev_max', 0.0032) or 0.0032)
            and float(features.get('dist_vwap_pct', 0.0) or 0.0) >= -0.0030
        ):
            action = 'LONG'
            reasons.append('TrendPullback LONG')
        elif (
            # Trend pullback continuation SHORT
            float(features.get('close', 0.0) or 0.0) < float(features.get('ma50', features.get('close', 0.0)) or features.get('close', 0.0))
            and float(features.get('ma20', features.get('close', 0.0)) or features.get('close', 0.0)) <= float(features.get('ma50', features.get('close', 0.0)) or features.get('close', 0.0))
            and float(features.get('price_momentum', 0.0) or 0.0) < 0
            and 38 <= float(rsi) <= 60
            and float(features.get('macd', 0.0) or 0.0) <= float(features.get('macd_signal', 0.0) or 0.0)
            and abs((float(features.get('close', 0.0) or 0.0) - float(features.get('ma20', features.get('close', 0.0)) or features.get('close', 0.0))) / max(float(features.get('ma20', features.get('close', 0.0)) or features.get('close', 0.0)), 1e-12)) <= float(getattr(self, 'pullback_ma20_dev_max', 0.0032) or 0.0032)
            and float(features.get('dist_vwap_pct', 0.0) or 0.0) <= 0.0030
        ):
            action = 'SHORT'
            reasons.append('TrendPullback SHORT')
        elif features['price_momentum'] < 0 and rsi < 65 and trend_ok and (macd_ok == -1 or (in_compression and breakout_dn)):
            action = "SHORT"
        else:
            # Fallback: only allow MA alignment when NOT in compression (otherwise wait for breakout)
            if (not in_compression) and features['close'] > features['ma50'] and features['price_momentum'] > 0:
                action = "LONG"
                reasons.append("Fallback MA Bullish")
            elif (not in_compression) and features['close'] < features['ma50'] and features['price_momentum'] < 0:
                action = "SHORT"
                reasons.append("Fallback MA Bearish")
            else:
                # Optional RANGE mode to increase trade frequency in chop,
                # still guarded by BTC leader gating later for followers.
                try:
                    if bool(getattr(self, 'allow_range_mode', False)):
                        # Range conditions: weak trend + compression + clear RSI extreme + deviation from MA20
                        tr = abs(float(features.get('trend_strength', 0.0) or 0.0))
                        ma20 = float(features.get('ma20', close_px) or close_px)
                        dev = abs(close_px - ma20) / max(ma20, 1e-12)
                        rsi_os = float(getattr(self, 'range_rsi_oversold', 32) or 32)
                        rsi_ob = float(getattr(self, 'range_rsi_overbought', 68) or 68)
                        dev_min = float(getattr(self, 'range_dev_ma20_min', 0.0016) or 0.0016)

                        if tr < 0.0015 and in_compression and dev >= dev_min:
                            if rsi <= rsi_os:
                                action = 'LONG'
                                reasons.append('RangeRevert RSI oversold')
                            elif rsi >= rsi_ob:
                                action = 'SHORT'
                                reasons.append('RangeRevert RSI overbought')
                            else:
                                logging.debug(f"No action or fallback for {symbol}")
                                return None
                        else:
                            logging.debug(f"No action or fallback for {symbol}")
                            return None
                    else:
                        logging.debug(f"No action or fallback for {symbol}")
                        return None
                except Exception:
                    logging.debug(f"No action or fallback for {symbol}")
                    return None

        # ----------------------
        # Sweep override (wick reclaim)
        # ----------------------
        try:
            sweep_high = bool(features.get('sweep_high', False))
            sweep_low = bool(features.get('sweep_low', False))
            # Prefer taking sweep reversals when trend is weak / compressed (mean-revert microstructure)
            if sweep_low and (not trend_ok or in_compression) and rsi <= 52:
                action = 'LONG'
                reasons.append('LiquiditySweep LOW reclaim')
            elif sweep_high and (not trend_ok or in_compression) and rsi >= 48:
                action = 'SHORT'
                reasons.append('LiquiditySweep HIGH reject')
        except Exception:
            pass


        # Entry style preflag (needed for HTF gate)
        entry_style = 'NORMAL'
        try:
            if bool(features.get('sweep_low', False)) and action == 'LONG':
                entry_style = 'SWEEP'
            elif bool(features.get('sweep_high', False)) and action == 'SHORT':
                entry_style = 'SWEEP'
        except Exception:
            entry_style = 'NORMAL'

        # ----------------------
        # HTF bias gate (15m+1h) to improve winrate / avoid countertrend churn
        # ----------------------
        try:
            htf_dir = int(features.get('htf_dir', 0) or 0)  # -1 short, +1 long, 0 neutral
            htf_strength = float(features.get('htf_strength', 0.0) or 0.0)
            # Only enforce when HTF is reasonably strong
            htf_enforce_thr = float(getattr(self, 'htf_enforce_strength', 0.28) or 0.28)
            if htf_dir != 0 and htf_strength >= htf_enforce_thr:
                if (htf_dir == 1 and action == 'SHORT') or (htf_dir == -1 and action == 'LONG'):
                    # Allow sweep reversals only when HTF is weak or neutral
                    if entry_style != 'SWEEP':
                        return _rej('HTF_GATE', f'HTF bias opposes action (dir={htf_dir}, strength={htf_strength:.2f})')
        except Exception:
            pass
        current_price = features['close']
        atr = features['atr']

        # Size multiplier used to downsize trades when gates are relaxed (freq-first CIO profile)
        size_mult = 1.0

        # --- BTC leader gating & smart entry filters for followers ---
        try:
            prof = (self.config.get('coin_profiles') or {}).get(symbol) or {}
            role = str(prof.get('role', 'FOLLOWER') or 'FOLLOWER').upper()
        except Exception:
            role = 'FOLLOWER'

        if symbol != 'BTCUSDT' and role == 'FOLLOWER':
            btc_dir = (self.btc_context or {}).get('direction')
            btc_strength = float((self.btc_context or {}).get('strength', 0.0) or 0.0)
            btc_sig_ts = (self.btc_context or {}).get('last_signal_ts')

            # FREQ-FIRST CIO MODE:
            # - When BTC is unclear/choppy, still allow RANGE-revert follower trades with reduced size.
            # - When follower signal is opposite BTC but BTC strength is weak, allow contrarian range trades (reduced size).
            relaxed_follow = False
            is_range_revert = any(('RangeRevert' in str(r)) for r in (reasons or [])) or (entry_style == 'SWEEP')
            allow_unclear = bool(getattr(self, 'allow_follower_when_btc_unclear', True))
            allow_contra = bool(getattr(self, 'allow_contra_followers_in_range', True))
            if btc_dir not in ('LONG', 'SHORT'):
                # BTC unclear: allow high-quality follower setups (range revert / breakout / high confidence)
                is_breakout = any(('Breakout' in str(r)) for r in (reasons or []))
                hi_conf = float(getattr(self, 'follower_unclear_high_conf', 0.62) or 0.62)
                if allow_unclear and (is_range_revert or is_breakout or provisional_conf >= hi_conf):
                    relaxed_follow = True
                    size_mult *= float(getattr(self, 'follower_size_mult_unclear', 0.50) or 0.50)
                    reasons.append('BTC unclear -> allow follower (reduced size)')
                else:
                    # Soft BTC gate: when BTC is unclear, still allow followers with decent confidence,
                    # but reduce size to avoid getting chopped.
                    soft_conf_base = float(getattr(self, 'follower_unclear_soft_conf', 0.45) or 0.45)
                    # IMPORTANT: keep soft_conf independent from runtime min_confidence (which can be raised by governor).
                    # Clamp into a practical range so followers do not starve when BTC is chopping.
                    soft_conf = min(0.55, max(0.38, soft_conf_base))
                    if provisional_conf >= soft_conf:
                        relaxed_follow = True
                        size_mult *= float(getattr(self, 'follower_size_mult_unclear', 0.60) or 0.60)
                        reasons.append(f'BTC unclear -> soft allow (conf {provisional_conf:.2f} >= {soft_conf:.2f})')
                    else:
                        # micro allow: for range-revert setups, allow tiny size even if confidence is not high
                        micro_allow = bool(getattr(self, 'allow_micro_follow_when_btc_unclear', True))
                        micro_mult = float(getattr(self, 'follower_size_mult_unclear_micro', 0.40) or 0.40)
                        if micro_allow and is_range_revert:
                            relaxed_follow = True
                            size_mult *= micro_mult
                            reasons.append('BTC unclear -> micro allow range-revert (tiny size)')
                        else:
                            return _rej('BTC_GATE', 'BTC has no clear direction')
            else:
                if action != btc_dir:
                    # Soft allow if follower has strong independent setup (breakout/trend) even when direction mismatches BTC.
                    try:
                        breakout_score = float(getattr(signal, 'breakout_score', 0.0) or 0.0)
                    except Exception:
                        breakout_score = 0.0
                    try:
                        ind_trend = float(getattr(signal, 'trend_strength', 0.0) or 0.0)
                    except Exception:
                        ind_trend = 0.0
                    strong_independent = (provisional_conf >= float(getattr(self, 'follower_contra_soft_conf', 0.62) or 0.62)) or (breakout_score >= 0.75) or (abs(ind_trend) >= 0.0022)
                    # Extra allow: if HTF is strongly aligned with follower direction, allow contrarian vs BTC unless BTC is extremely strong.
                    try:
                        htf_dir = int(features.get('htf_dir', 0) or 0)
                        htf_strength = float(features.get('htf_strength', 0.0) or 0.0)
                        htf_aligned = (htf_dir != 0 and ((action == 'LONG' and htf_dir > 0) or (action == 'SHORT' and htf_dir < 0)))
                        btc_too_strong = (btc_strength >= float(getattr(self, 'btc_contra_block_strength', 0.82) or 0.82))
                        if (not strong_independent) and htf_aligned and htf_strength >= float(getattr(self, 'follower_contra_htf_strength', 0.62) or 0.62) and (not btc_too_strong):
                            strong_independent = True
                            reasons.append('HTF aligned contra BTC -> allowed')
                    except Exception:
                        pass
                    if strong_independent:
                        relaxed_follow = True
                        size_mult *= float(getattr(self, 'follower_size_mult_contra', 0.55) or 0.55)
                        reasons.append('Independent setup contra BTC -> allowed (reduced size)')
                    else:
                        contra_max = float(getattr(self, 'contra_btc_strength_max', 0.55) or 0.55)
                    if allow_contra and is_range_revert and btc_strength <= contra_max:
                        relaxed_follow = True
                        size_mult *= float(getattr(self, 'follower_size_mult_contra', 0.55) or 0.55)
                        reasons.append('Range contra vs BTC -> allowed (reduced size)')
                    else:
                        return _rej('BTC_GATE', f'follower {action} != BTC {btc_dir}')

            # BTC signal freshness: only enforce when NOT in relaxed mode (otherwise followers will starve again)
            if not relaxed_follow:
                try:
                    age_map = getattr(self, 'follower_signal_age_min', None) or {'SOLUSDT': 20, 'XRPUSDT': 25, 'ETHUSDT': 30}
                    max_age_min = int((age_map or {}).get(symbol, 30) or 30)
                    if not btc_sig_ts:
                        return _rej('BTC_GATE_STALE', 'BTC signal not found (stale)')
                    ts = datetime.fromisoformat(str(btc_sig_ts))
                    age_min = (datetime.now() - ts).total_seconds() / 60.0
                    if age_min > float(max_age_min):
                        return _rej('BTC_GATE_STALE', f'BTC signal too old: {age_min:.1f}m > {max_age_min}m')
                except Exception:
                    return _rej('BTC_GATE_STALE', 'BTC signal timestamp parse failed')

            # Requirements by speed: SOL (fast) > XRP > ETH
            reqs = getattr(self, 'follower_requirements', None) or {}
            req = (reqs or {}).get(symbol, {'min_btc_strength': 0.50, 'max_range_atr': 2.2, 'max_ext_ma50': 0.018})

            # Strength gate: in relaxed follower mode, allow weaker BTC strength but with smaller size
            min_strength = float(req.get('min_btc_strength', 0.50) or 0.50)
            if btc_strength < min_strength:
                # If BTC is weak but the follower setup is very strong, allow a reduced-size entry.
                # This increases opportunities without turning off BTC leadership completely.
                weak_hi_conf = float(getattr(self, 'follower_weakbtc_high_conf', 0.66) or 0.66)
                if provisional_conf >= weak_hi_conf:
                    size_mult *= float(getattr(self, 'follower_size_mult_weakbtc_hi_conf', 0.60) or 0.60)
                    reasons.append(f'BTC weak ({btc_strength:.2f}<{min_strength:.2f}) but high conf {provisional_conf:.2f} -> allow (reduced size)')
                else:
                    min_relaxed = float(getattr(self, 'min_btc_strength_relaxed', 0.02) or 0.02)
                    if relaxed_follow and btc_strength >= min_relaxed:
                        size_mult *= float(getattr(self, 'follower_size_mult_weakbtc', 0.85) or 0.85)
                        reasons.append(f'BTC strength relaxed pass ({btc_strength:.2f})')
                    else:
                        return _rej('BTC_GATE', f'BTC strength {btc_strength:.2f} < {min_strength:.2f}')

            # Anti-stop-hunt filters (still enforced even in relaxed mode)
            range_atr = float(features.get('range_atr', 0.0) or 0.0)
            ext_ma50 = float(features.get('ext_ma50_pct', 0.0) or 0.0)
            if range_atr > float(req.get('max_range_atr', 2.2) or 2.2):
                return _rej('WICK_FILTER', f'range_atr {range_atr:.2f} > {float(req.get("max_range_atr",2.2)):.2f}')
            if ext_ma50 > float(req.get('max_ext_ma50', 0.018) or 0.018):
                return _rej('EXT_FILTER', f'ext_ma50 {ext_ma50:.3f} > {float(req.get("max_ext_ma50",0.018)):.3f}')

        # --- Smart SL/TP sizing by symbol speed (avoid SL hunts, aim for bigger TP) ---
        # Use ATR-based brackets; faster coins get wider SL to survive wicks.

        # SWEEP mode: wider stop beyond wick + target mean-revert to VWAP / mid-range
        sweep_sl_extra = 0.0
        sweep_tp_bias = 0.0
        try:
            if entry_style == 'SWEEP':
                # add extra buffer so SL sits beyond the sweep wick
                sweep_sl_extra = max(0.30 * float(atr), 0.0012 * float(current_price))
                # bias TP closer to structural mean (vwap/ma20) for high hit-rate
                sweep_tp_bias = 1.0
        except Exception:
            sweep_sl_extra = 0.0
            sweep_tp_bias = 0.0
        # --- Smart SL/TP sizing by symbol speed (avoid SL hunts, aim for bigger TP) ---
        # Use ATR-based brackets from config (more stable than fixed maps).
        # Key idea: widen the *price* stop on faster coins so RR logic doesn't trigger too early.
        sweep_sl_extra = 0.0
        sweep_tp_bias = 0.0
        try:
            if entry_style == 'SWEEP':
                sweep_sl_extra = max(0.30 * float(atr), 0.0012 * float(current_price))
                sweep_tp_bias = 1.0
        except Exception:
            sweep_sl_extra = 0.0
            sweep_tp_bias = 0.0

        # Base stop/target multipliers (config-driven)
        try:
            br_cfg = (self.config.get('brackets', {}) or {}) if isinstance(self.config, dict) else {}
        except Exception:
            br_cfg = {}
        base_stop_atr = float(br_cfg.get('stop_atr_mult', 1.6) or 1.6)
        base_tp_vs_sl = float(br_cfg.get('tp_vs_sl_min_mult', 2.6) or 2.6)

        # Speed factor per symbol (alts get wider stops to survive wicks on 5m)
        speed_factor = {
            'BTCUSDT': 1.00,
            'ETHUSDT': 1.10,
            'XRPUSDT': 1.35,
            'SOLUSDT': 1.45,
        }.get(symbol, 1.20)

        sl_pct = (base_stop_atr * speed_factor) * float(atr)
        sl_pct = float(sl_pct) + float(sweep_sl_extra)

        # Baseline TP: maintain thick RR; sweep entries may target mean first (higher hit-rate) then runner handles extension.
        tp_pct = max(float(sl_pct) * float(base_tp_vs_sl), float(atr) * 2.2)
        try:
            if sweep_tp_bias > 0:
                ma20 = float(features.get('ma20', current_price) or current_price)
                vwap = float(features.get('vwap', current_price) or current_price)
                if action == 'LONG':
                    tgt = max(ma20, vwap)
                    tp_pct = max(float(tp_pct) * 0.72, abs(tgt - current_price))
                else:
                    tgt = min(ma20, vwap)
                    tp_pct = max(float(tp_pct) * 0.72, abs(current_price - tgt))
        except Exception:
            pass
        
        try:
            if sweep_tp_bias > 0:
                # target toward vwap/ma20 but still keep RR floor handled later
                ma20 = float(features.get('ma20', current_price) or current_price)
                vwap = float(features.get('vwap', current_price) or current_price)
                if action == 'LONG':
                    tgt = max(ma20, vwap)
                    tp_pct = max(float(tp_pct)*0.75, abs(tgt - current_price))
                else:
                    tgt = min(ma20, vwap)
                    tp_pct = max(float(tp_pct)*0.75, abs(current_price - tgt))
        except Exception:
            pass

        # ----------------------
        # Cost-aware gating (fees + slippage)
        # Futures fee is proportional to notional, so the *required price move* to break even
        # is roughly: price * (fee_rate*2 + slippage*2 + spread_buffer).
        # We enforce:
        #   - TP distance >= break-even move * reward_mult
        #   - SL distance >= break-even move * risk_mult  (avoid "stop too tight -> fee death")
        # This prevents the common failure mode: trade is "right" but net PnL is negative due to fees.
        try:
            fee_rate = float(self.config.get('fee_rate', 0.0004) or 0.0004)
        except Exception:
            fee_rate = 0.0004
        try:
            slippage_pct = float(self.config.get('slippage_pct', 0.0006) or 0.0006)
        except Exception:
            slippage_pct = 0.0006
        try:
            spread_buf = float(self.config.get('microstructure', {}).get('spread_cost_buffer', 0.00015) or 0.00015)
        except Exception:
            spread_buf = 0.00015
        try:
            reward_mult = float(getattr(self, 'cost_gate_reward_mult', self.config.get('cost_gate_reward_mult', 1.35)) or 1.35)
        except Exception:
            reward_mult = 1.35
        try:
            risk_mult = float(getattr(self, 'cost_gate_risk_mult', self.config.get('cost_gate_risk_mult', 1.05)) or 1.05)
        except Exception:
            risk_mult = 1.05

        breakeven_move_abs = float(current_price) * max(0.0, (fee_rate * 2.0) + (slippage_pct * 2.0) + spread_buf)
        min_tp_abs = breakeven_move_abs * max(1.0, reward_mult)
        min_sl_abs = breakeven_move_abs * max(1.0, risk_mult)

        if tp_pct < min_tp_abs:
            tp_pct = min_tp_abs
        if sl_pct < min_sl_abs:
            sl_pct = min_sl_abs

        # Enforce minimum stop distance (pct of price) to avoid getting wicked out on tiny ATR
        try:
            min_stop_pct = float(((self.config.get('brackets', {}) or {}).get('min_stop_pct', 0.0)) or 0.0)
        except Exception:
            min_stop_pct = 0.0
        if min_stop_pct and min_stop_pct > 0:
            min_stop_abs = float(current_price) * float(min_stop_pct)
            if sl_pct < min_stop_abs:
                sl_pct = min_stop_abs

        # Enforce maximum stop distance (pct of price) to avoid huge losses on volatile spikes.
        # If computed stop exceeds cap materially, skip the trade (market too wild for this system).
        try:
            max_stop_pct = float(((self.config.get('brackets', {}) or {}).get('max_stop_pct', 0.0)) or 0.0)
            max_map = ((self.config.get('brackets', {}) or {}).get('max_stop_pct_map', {}) or {})
            max_stop_pct = float(max_map.get(symbol, max_stop_pct) or max_stop_pct)
        except Exception:
            max_stop_pct = 0.0
        if max_stop_pct and max_stop_pct > 0:
            max_stop_abs = float(current_price) * float(max_stop_pct)
            # If stop is way beyond cap, better to skip than to risk a fat tail loss.
            if sl_pct > (max_stop_abs * 1.05):
                logging.debug(f"Stop too wide for {symbol}: {sl_pct:.6f} > cap {max_stop_abs:.6f}")
                return None
            if sl_pct > max_stop_abs:
                sl_pct = max_stop_abs
        # Ensure TP is at least a multiple of SL (fee-positive RR baseline)
        try:
            tp_vs_sl = float(((self.config.get('brackets', {}) or {}).get('tp_vs_sl_min_mult', 0.0)) or 0.0)
        except Exception:
            tp_vs_sl = 0.0
        if tp_vs_sl and tp_vs_sl > 0 and tp_pct < (sl_pct * tp_vs_sl):
            tp_pct = sl_pct * tp_vs_sl
        rr = (tp_pct / sl_pct) if sl_pct > 0 else 0.0
        # v10: require minimum RR to survive fees/spread; tuned by tuner.
        if rr < float(getattr(self, 'rr_min', 1.0) or 1.0):
            logging.debug(f"RR too low for {symbol}: {rr:.2f} < {self.rr_min}")
            return None
        if action == "LONG":
            sl = current_price - sl_pct
            tp = current_price + tp_pct
        else:
            sl = current_price + sl_pct
            tp = current_price - tp_pct

        # Sanity guards (must preserve LONG/SHORT directionality)
        # - LONG : SL < entry < TP
        # - SHORT: TP < entry < SL
        # Also ensure prices stay positive and not absurdly close to zero.
        min_price = max(current_price * 0.01, 1e-12)
        sl = max(float(sl), min_price)
        tp = max(float(tp), min_price)

        if action == "LONG":
            # Fix ordering if something inverted
            if sl >= current_price:
                sl = max(min_price, current_price - abs(sl_pct))
            if tp <= current_price:
                tp = max(min_price, current_price + abs(tp_pct))
            if tp <= sl:
                # push TP away from SL to maintain positive RR
                tp = current_price + max(abs(tp_pct), abs(sl_pct) * 1.5)
        else:
            # SHORT
            if sl <= current_price:
                sl = max(min_price, current_price + abs(sl_pct))
            if tp >= current_price:
                tp = max(min_price, current_price - abs(tp_pct))
            if tp >= sl:
                # push TP below entry and below SL
                tp = max(min_price, current_price - max(abs(tp_pct), abs(sl_pct) * 1.5))


        # --- CIO-grade anti stop-hunt: structural stop around swing + buffer (avoid early stop-outs) ---
        try:
            # Buffer beyond swing to survive stop-hunts / 2nd sweep
            # Default: ~0.35*ATR (swing) + small price buffer (ticks)
            swing_buf_atr = float(getattr(self, 'swing_stop_buffer_atr', 0.35) or 0.35)
            swing_buf_px  = float(getattr(self, 'swing_stop_buffer_px_pct', 0.0008) or 0.0008) * float(current_price)
            pivot_buf = max(float(atr) * swing_buf_atr, swing_buf_px)

            max_risk_mult = float(getattr(self, 'swing_stop_max_risk_mult', 1.35) or 1.35)

            cur_high = float(features.get('cur_high', current_price) or current_price)
            cur_low  = float(features.get('cur_low', current_price) or current_price)

            if action == 'LONG':
                # Base structural: below recent swing low
                struct_sl = float(ll20) - pivot_buf
                # If this is a sweep reclaim, stop should sit beyond the sweep wick low
                if entry_style == 'SWEEP' and bool(features.get('sweep_low', False)):
                    struct_sl = min(struct_sl, cur_low - pivot_buf)
                sl_candidate = float(sl)
                try:
                    if entry_style == 'SWEEP':
                        # For sweep-reversal entries, stop must sit beyond the sweep wick + buffer.
                        sl_candidate = min(float(sl_candidate), float(struct_sl))
                    else:
                        # Anti stop-hunt: prefer STRUCTURAL stop only if it is WIDER (further) than current SL.
                        # (Old logic tightened stops -> churn + early cuts)
                        if float(struct_sl) < float(sl_candidate):
                            sl_candidate = float(struct_sl)
                except Exception:
                    pass

                # Clamp max risk expansion
                max_risk = float(abs(sl_pct)) * max_risk_mult
                if (current_price - sl_candidate) > max_risk:
                    sl_candidate = current_price - max_risk
                sl = max(min_price, sl_candidate)
            else:
                struct_sl = float(hh20) + pivot_buf
                if entry_style == 'SWEEP' and bool(features.get('sweep_high', False)):
                    struct_sl = max(struct_sl, cur_high + pivot_buf)
                sl_candidate = float(sl)
                try:
                    if entry_style == 'SWEEP':
                        # For sweep-reversal entries, stop must sit beyond the sweep wick + buffer.
                        sl_candidate = max(float(sl_candidate), float(struct_sl))
                    else:
                        # Anti stop-hunt: prefer STRUCTURAL stop only if it is WIDER (further) than current SL.
                        if float(struct_sl) > float(sl_candidate):
                            sl_candidate = float(struct_sl)
                except Exception:
                    pass

                max_risk = float(abs(sl_pct)) * max_risk_mult
                if (sl_candidate - current_price) > max_risk:
                    sl_candidate = current_price + max_risk
                sl = max(min_price, sl_candidate)
        except Exception:
            pass

        # Re-apply min/max stop distance AFTER structural adjustments (structural stop may tighten/widen).
        try:
            min_stop_pct2 = float(((self.config.get('brackets', {}) or {}).get('min_stop_pct', 0.0)) or 0.0)
        except Exception:
            min_stop_pct2 = 0.0
        try:
            max_stop_pct2 = float(((self.config.get('brackets', {}) or {}).get('max_stop_pct', 0.0)) or 0.0)
            max_map2 = ((self.config.get('brackets', {}) or {}).get('max_stop_pct_map', {}) or {})
            max_stop_pct2 = float(max_map2.get(symbol, max_stop_pct2) or max_stop_pct2)
        except Exception:
            max_stop_pct2 = 0.0

        try:
            stop_dist = abs(float(current_price) - float(sl))
            if min_stop_pct2 and min_stop_pct2 > 0:
                min_stop_abs2 = float(current_price) * float(min_stop_pct2)
                if stop_dist < min_stop_abs2:
                    if action == "LONG":
                        sl = float(current_price) - float(min_stop_abs2)
                    else:
                        sl = float(current_price) + float(min_stop_abs2)
            if max_stop_pct2 and max_stop_pct2 > 0:
                max_stop_abs2 = float(current_price) * float(max_stop_pct2)
                stop_dist2 = abs(float(current_price) - float(sl))
                if stop_dist2 > max_stop_abs2:
                    # clamp to cap rather than skip (risk governor still protects account risk)
                    if action == "LONG":
                        sl = float(current_price) - float(max_stop_abs2)
                    else:
                        sl = float(current_price) + float(max_stop_abs2)
        except Exception:
            pass


        # --- Structural TP: hug recent swing (hh20/ll20) for mean-revert, extend on breakout ---
        try:
            tp_pivot_buf = float(atr) * float(getattr(self, 'tp_pivot_atr_buf', 0.10) or 0.10)
            # When not breaking out, target near the nearest swing level (avoid missing fill by a hair).
            # When breaking out, allow extension to keep TP 'dày'.
            if action == 'LONG':
                if close_px <= hh20:
                    # Take profit just below resistance
                    struct_tp = max(min_price, float(hh20) - tp_pivot_buf)
                    # Don't reduce TP too aggressively; keep at least the structural target
                    # Aim TP close to swing (hug resistance), but never below minimum fee-positive target
                    fee_rt = 2.0 * float(self.fee_rate)
                    slip_cushion = float(getattr(self, 'cost_gate_slip_cushion', 0.00020) or 0.00020)
                    cost_pct = fee_rt + slip_cushion
                    rew_mult = float(getattr(self, 'cost_gate_reward_mult', 1.01) or 1.01)
                    min_tp = float(current_price) * (cost_pct * rew_mult)
                    tp_floor = float(current_price) + min_tp
                    tp = max(tp_floor, max(float(tp), struct_tp))
                else:
                    # Breakout long: extend target by a fraction of the recent range
                    rng = max(0.0, float(hh20) - float(ll20))
                    ext = max(float(tp_pct), rng * float(getattr(self, 'breakout_tp_range_mult', 0.55) or 0.55))
                    tp = max(float(tp), float(current_price) + ext)
            else:
                if close_px >= ll20:
                    # Take profit just above support
                    struct_tp = max(min_price, float(ll20) + tp_pivot_buf)
                    # Aim TP close to swing (hug support), but never below minimum fee-positive target
                    fee_rt = 2.0 * float(self.fee_rate)
                    slip_cushion = float(getattr(self, 'cost_gate_slip_cushion', 0.00020) or 0.00020)
                    cost_pct = fee_rt + slip_cushion
                    rew_mult = float(getattr(self, 'cost_gate_reward_mult', 1.01) or 1.01)
                    min_tp = float(current_price) * (cost_pct * rew_mult)
                    tp_floor = float(current_price) - min_tp
                    tp = min(tp_floor, max(float(tp), struct_tp))
                else:
                    # Breakout short: extend target by a fraction of the recent range
                    rng = max(0.0, float(hh20) - float(ll20))
                    ext = max(float(tp_pct), rng * float(getattr(self, 'breakout_tp_range_mult', 0.55) or 0.55))
                    tp = min(float(tp), float(current_price) - ext)
        except Exception:
            pass

        # Recompute RR after structural adjustment
        try:
            risk = abs(current_price - float(sl))
            reward = abs(float(tp) - current_price)
            rr = (reward / risk) if risk > 0 else 0.0
        except Exception:
            rr = rr

        # --- Cost/fee-aware gate (FREQ-FIRST CIO): prefer more trades, but downsize when edge is small ---
        # Approx round-trip cost: fees + small slippage cushion.
        try:
            fee_rt = 2.0 * float(self.fee_rate)  # round trip
            slip_cushion = float(getattr(self, 'cost_gate_slip_cushion', 0.00020) or 0.00020)
            cost_pct = fee_rt + slip_cushion
            reward_pct = abs(float(tp) - current_price) / max(current_price, 1e-12)
            risk_pct = abs(current_price - float(sl)) / max(current_price, 1e-12)

            # Primary gate (freq-first defaults)
            rew_mult = float(getattr(self, 'cost_gate_reward_mult', 1.01) or 1.05)
            # IMPORTANT: risk_mult too high can starve the system when ATR is small.
            # We keep it fee-aware but allow trades by widening SL + downsizing (below).
            risk_mult = float(getattr(self, 'cost_gate_risk_mult', 0.45) or 0.45)

            # Soft-pass zone: if just below the gate, allow but reduce size
            soft_rew_mult = float(getattr(self, 'cost_gate_soft_reward_mult', 0.98) or 0.98)
            soft_risk_mult = float(getattr(self, 'cost_gate_soft_risk_mult', 0.90) or 0.90)
            soft_size = float(getattr(self, 'cost_gate_soft_size_mult', 0.55) or 0.55)

            if reward_pct < (rew_mult * cost_pct):
                if reward_pct >= (soft_rew_mult * rew_mult * cost_pct):
                    size_mult *= soft_size
                    reasons.append('COST soft-pass (downsized)')
                else:
                    return _rej('COST_GATE', f'reward_pct {reward_pct:.5f} < {rew_mult:.2f}*cost {cost_pct:.5f}')

            # If SL is too tight relative to friction, do NOT reject outright.
            # Instead: widen SL to the minimum viable risk band and downsize position
            # so the *account currency risk* stays roughly the same.
            min_risk_pct = (risk_mult * cost_pct)
            if risk_pct < min_risk_pct:
                # soft zone: allow but downsize (and optionally widen stop)
                if risk_pct >= (soft_risk_mult * min_risk_pct):
                    size_mult *= soft_size
                    reasons.append('RISK soft-pass (downsized)')
                else:
                    try:
                        # widen stop to min_risk_pct and scale size to keep $risk ~ constant
                        old_risk_pct = max(risk_pct, 1e-12)
                        target_risk_pct = max(min_risk_pct, old_risk_pct)
                        if action == 'LONG':
                            sl = float(current_price) * (1.0 - target_risk_pct)
                        else:
                            sl = float(current_price) * (1.0 + target_risk_pct)
                        # keep same $ risk by reducing size proportionally
                        size_mult *= max(0.20, min(1.0, old_risk_pct / target_risk_pct))
                        reasons.append('RISK widened SL + downsized')
                    except Exception:
                        return _rej('COST_GATE', f'risk_pct {risk_pct:.5f} < {risk_mult:.2f}*cost {cost_pct:.5f}')
        except Exception:
            pass

        # ---- Final confidence blend & gate ----
        try:
            # Rule-based confidence from score + trend; bounded to [0.2, 0.95]
            ts = abs(float(features.get('trend_strength', 0.0) or 0.0))
            # normalize trend strength roughly (0.0015..0.006)
            ts_n = max(0.0, min(1.0, (ts - 0.0012) / 0.0048))
            score_n = max(0.0, min(1.0, float(score) / 10.0))
            conf_rule = 0.28 + 0.22 * ts_n + 0.20 * score_n
            conf_rule = max(0.20, min(0.95, float(conf_rule)))
        except Exception:
            conf_rule = 0.35

        try:
            # ML informative? (avoid near-0.5 noise)
            ml_inf = (abs(float(prob_long) - 0.5) >= 0.08) or (abs(float(prob_short) - 0.5) >= 0.08)
        except Exception:
            ml_inf = False

        # Blend rule confidence with ML probability when ML is informative.
        # Using max() here often over-inflates confidence (e.g., always 0.95) and hurts calibration.
        confidence = float(conf_rule)
        if ml_inf:
            try:
                ml_adj = (float(confidence_ml) - 0.5) * 0.60
            except Exception:
                ml_adj = 0.0
            confidence = float(conf_rule) + float(ml_adj)
        confidence = float(max(0.20, min(0.95, confidence)))

        
        # Brain edge adjustment (self-evolve): nudge confidence up/down based on recent real trades
        try:
            brain = getattr(self, 'brain', None)
            if brain is not None and hasattr(brain, 'get_edge_adjustment'):
                confidence = float(max(0.0, min(0.99, confidence + float(brain.get_edge_adjustment(symbol, features, action) or 0.0))))
        except Exception:
            pass

# Final gate: use blended confidence + per-coin min_conf if present
        try:
            prof_cfg = (self.config.get('coin_profiles') or {}).get(symbol) or {}
            coin_min_conf = float(prof_cfg.get('min_confidence', 0.0) or 0.0)
        except Exception:
            coin_min_conf = 0.0

        min_conf = float(max(self.min_confidence, coin_min_conf))
        # If HTF (15m+1h) aligns with the signal direction, allow slightly lower threshold
        try:
            htf_dir = int(features.get('htf_dir', 0) or 0)
            htf_strength = float(features.get('htf_strength', 0.0) or 0.0)
            if htf_dir != 0 and ((signal == 'LONG' and htf_dir > 0) or (signal == 'SHORT' and htf_dir < 0)):
                # stronger HTF => more permissive (more trades) while keeping directionality
                k = 0.88 if htf_strength > 0.6 else 0.92
                min_conf = float(min_conf * k)
        except Exception:
            pass

        # If HTF bias is strong AGAINST the signal, reject to reduce whipsaw/fee bleed
        try:
            htf_dir = int(features.get('htf_dir', 0) or 0)
            htf_strength = float(features.get('htf_strength', 0.0) or 0.0)
            if htf_dir != 0 and htf_strength >= 0.58:
                if (signal == 'LONG' and htf_dir < 0) or (signal == 'SHORT' and htf_dir > 0):
                    # allow only very high confidence to trade against strong HTF
                    if float(confidence) < float(max(min_conf + 0.12, 0.78)):
                        return _rej('HTF_CONTRA', f'htf_dir={htf_dir} strength={htf_strength:.2f} contra signal conf={confidence:.3f}')
        except Exception:
            pass


        
        # ---- Regime policy (self-evolving) ----
        try:
            brain = getattr(self, 'brain', None)
            if brain is not None and hasattr(brain, 'get_regime_policy'):
                pol = brain.get_regime_policy(symbol, features) or {}
                try:
                    ae = (self.config.get('adaptive_engine', {}) or {}) if isinstance(self.config, dict) else {}
                    damp = 0.55 if bool(ae.get('enabled', True)) else 1.0
                except Exception:
                    damp = 1.0

                min_conf = float(min_conf) + float(pol.get('min_conf_adj', 0.0) or 0.0) * damp
                min_conf = float(max(0.35, min(0.90, min_conf)))

                try:
                    self.rr_min = float(getattr(self, 'rr_min', 1.08) or 1.08) + float(pol.get('rr_min_adj', 0.0) or 0.0) * damp
                    self.rr_min = float(max(0.90, min(2.20, self.rr_min)))
                except Exception:
                    pass

                try:
                    self.swing_stop_buffer_atr = float(getattr(self, 'swing_stop_buffer_atr', 0.35) or 0.35) + float(pol.get('stop_buf_adj', 0.0) or 0.0) * damp
                    self.swing_stop_buffer_atr = float(max(0.20, min(0.95, self.swing_stop_buffer_atr)))
                except Exception:
                    pass

                try:
                    self.last_trail_mult_adj = float(pol.get('trail_mult_adj', 0.0) or 0.0) * damp
                except Exception:
                    self.last_trail_mult_adj = 0.0
        except Exception:
            pass

# ---- Anti-chop / low-edge filter ----
        # Nếu thị trường quá "phẳng" (trend_strength thấp, volatility thấp, giá bám MA50),
        # thì tín hiệu dễ nhiễu -> tỉ lệ lỗ cao. Ta bỏ qua để tăng winrate/expectancy.
        try:
            vol = float(features.get('volatility', 0.0) or 0.0)
            trend = float(features.get('trend_strength', 0.0) or 0.0)
            ext_ma50 = float(features.get('ext_ma50_pct', 0.0) or 0.0)
            atr_ratio = float(features.get('atr_ratio', 0.0) or 0.0)
            hs = float(features.get('htf_strength', 0.0) or 0.0)

            # Low-edge zone: volatility rất thấp + trend yếu + giá sát MA50
            # (atr_ratio nhỏ thường là nén, nhưng nén + trend yếu thường chop)
            if (vol < 0.00022 and trend < 0.0012 and ext_ma50 < 0.0018 and hs < 0.52 and atr_ratio < 0.0012):
                return _rej('CHOP', f'low-edge chop vol={vol:.5f} trend={trend:.5f} extMA50={ext_ma50:.5f}')
        except Exception:
            pass
        if confidence < min_conf:
            return _rej('CONF', f'confidence {confidence:.3f} < min_conf {min_conf:.3f}')

        # ---- Hard cap SL distance (prevents huge losses in spikes) ----
        try:
            br = (self.config.get('brackets') or {}) if isinstance(self.config, dict) else {}
            max_stop_pct = float(br.get('max_stop_pct', 0.012) or 0.012)
            m = br.get('max_stop_pct_map', {}) or {}
            cap = float(m.get(symbol, max_stop_pct) or max_stop_pct)
            cap = max(0.0025, min(0.04, cap))

            # HTF yếu: không siết cap quá chặt (tránh stop-out liên tục trên 5m).
            # Thay vào đó, hệ thống sẽ downsize thông qua confidence/portfolio caps và cost-gate.
            try:
                _ = float(features.get('htf_strength', 0.0) or 0.0)
            except Exception:
                pass
            if action == 'LONG':
                sl_cap = float(current_price) * (1.0 - cap)
                # keep SL below entry, but not too far
                sl = max(float(sl), sl_cap)
                # ensure strictly below entry
                sl = min(float(sl), float(current_price) * 0.9995)
            else:
                sl_cap = float(current_price) * (1.0 + cap)
                sl = min(float(sl), sl_cap)
                sl = max(float(sl), float(current_price) * 1.0005)
        except Exception:
            pass

        
        # --- Enforce minimum risk:reward (avoid win-mỏng lose-dày) ---
        try:
            br = (self.config.get('brackets', {}) or {}) if isinstance(self.config, dict) else {}
            rr_map = br.get('rr_min_map', {}) if isinstance(br, dict) else {}
            rr_min = float(rr_map.get(symbol, br.get('rr_min', 1.25) if isinstance(br, dict) else 1.25) or 1.25)
            rr_min = max(0.80, min(4.0, rr_min))
            risk = abs(float(current_price) - float(sl))
            if risk > 0:
                min_reward = risk * rr_min
                if action == 'LONG':
                    tp = max(float(tp), float(current_price) + min_reward)
                else:
                    tp = min(float(tp), float(current_price) - min_reward)
        except Exception:
            pass
        reason = f"Score {score:.1f} | RR {rr:.2f} | Session {current_session} | Conf {confidence:.3f} | " + " | ".join(reasons)

        return TradingSignal(
            symbol=symbol,
            action=action,
            timestamp=datetime.now(),
            entry_price=current_price,
            stop_loss=sl,
            take_profit=tp,
            current_price=current_price,
            position_size_pct=max(0.15, min(1.0, float(size_mult))),
            confidence=confidence,
            ml_probability=prob_long if action == "LONG" else prob_short,
            session=current_session,
            features=features,
            reason=reason
        )