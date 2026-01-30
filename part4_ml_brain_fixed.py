"""
═══════════════════════════════════════════════════════════════
🧠 PART 4: ML BRAIN - PERSISTENT & SELF-EVOLVING V2.1 - FIXED
═══════════════════════════════════════════════════════════════
- Persistent brain (survives restart)
- Auto-save every 30 minutes
- Auto-load on startup
- Continuous evolution tracking
- Coin-specific models
- Feature importance per coin
- FIX: Thêm tạo thư mục + kiểm tra quyền ghi
- FIX: predict_entry_probability xử lý scaler chưa fit
- FIX: get_confidence an toàn
- FIX MỚI: predict_entry_probability luôn trả về tuple (prob_long, prob_short)
- LIVE FIX: Không bao giờ trả về (0.5, 0.5) khi chưa có model
"""

import os
import json
import sqlite3
import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from scipy import stats
import threading
from functools import lru_cache

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

from error_handling import handle_errors, MLError


class PersistentTradingBrain:
    def __init__(self, db_path: str = 'data/trading_brain_v2.db', config: Dict = None):
        self.db_path = db_path
        self.config = config or {}

        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            try:
                os.makedirs(db_dir, exist_ok=True)
                logging.info(f"Đã tạo thư mục: {db_dir}")
            except Exception as e:
                logging.critical(f"Không thể tạo thư mục {db_dir}: {e}")
                raise RuntimeError(f"Không thể tạo thư mục database: {e}")

        test_file = os.path.join(db_dir or '.', 'write_test.tmp')
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
        except Exception as e:
            logging.critical(f"Thư mục không cho phép ghi: {e}")
            raise RuntimeError("Database directory read-only")

        self._init_database()

        self.models = {}

        # Self-evolve lightweight edges + regime policy (persisted)
        self.edge_weights = {}
        self.regime_policy = {}  # regime_policy[symbol][bucket] = stats/adjustments

        self.scalers = {}
        self.global_model = None
        self.global_scaler = RobustScaler()

        self.feature_importance = {}
        self.feature_importance_by_coin = {}

        self.learning_buffer = []
        self.retrain_threshold = self.config.get('ml_settings', {}).get('retrain_after_trades', 50)
        self.min_training_samples = self.config.get('ml_settings', {}).get('min_training_samples', 100)

        self.model_accuracy_history = []
        self.last_retrain_time = datetime.now()
        self.evolution_level = 1
        self.total_training_sessions = 0

        self.feature_names = [
            'rsi', 'macd', 'macd_histogram', 'bb_position', 'trend_strength',
            'order_flow_delta', 'cvd', 'aggressive_buying', 'aggressive_selling',
            'bid_ask_imbalance', 'volatility', 'price_momentum', 'funding_rate',
            'btc_correlation', 'session_encoded'
        ]

        self.brain_state_file = 'data/brain_state_v2.pkl'
        self.auto_save_interval = self.config.get('ml_settings', {}).get(
            'persistent_brain', {}).get('save_interval_minutes', 30)

        self.auto_save_enabled = True
        self.save_thread = None

        if self.config.get('ml_settings', {}).get('persistent_brain', {}).get('auto_load_on_startup', True):
            self._load_brain_state()

        self._start_auto_save_thread()
        logging.info("🧠 Persistent Trading Brain initialized")

    # ===================== DB =====================
    def _init_database(self):
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS trades
                        (id INTEGER PRIMARY KEY AUTOINCREMENT,
                         timestamp TEXT NOT NULL,
                         symbol TEXT NOT NULL,
                         side TEXT NOT NULL,
                         entry_price REAL NOT NULL,
                         exit_price REAL NOT NULL,
                         quantity REAL NOT NULL,
                         pnl REAL NOT NULL,
                         pnl_pct REAL NOT NULL,
                         duration_seconds INTEGER,
                         exit_reason TEXT,
                         regime TEXT,
                         confidence REAL,
                         features TEXT)''')
            conn.commit()
        finally:
            if conn:
                conn.close()

    
    def log_trade(self, trade_data: Dict):
        """Persist a closed trade to sqlite for later learning/analytics.
        trade_data must include: timestamp, symbol, side, entry_price, exit_price, quantity, pnl, pnl_pct, duration_seconds, exit_reason, regime, confidence, features.
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            feats = trade_data.get('features', {}) or {}
            try:
                feats_s = json.dumps(feats, ensure_ascii=False)
            except Exception:
                feats_s = str(feats)
            c.execute(
                """INSERT INTO trades (timestamp, symbol, side, entry_price, exit_price, quantity, pnl, pnl_pct, duration_seconds, exit_reason, regime, confidence, features)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(trade_data.get('timestamp') or datetime.now().isoformat()),
                    str(trade_data.get('symbol') or ''),
                    str(trade_data.get('side') or ''),
                    float(trade_data.get('entry_price') or 0.0),
                    float(trade_data.get('exit_price') or 0.0),
                    float(trade_data.get('quantity') or 0.0),
                    float(trade_data.get('pnl') or 0.0),
                    float(trade_data.get('pnl_pct') or 0.0),
                    int(trade_data.get('duration_seconds') or 0),
                    str(trade_data.get('exit_reason') or ''),
                    str(trade_data.get('regime') or ''),
                    float(trade_data.get('confidence') or 0.0),
                    feats_s,
                )
            )
            conn.commit()
        except Exception as e:
            logging.error(f"Failed to log trade: {e}")
        finally:
            if conn:
                conn.close()

    def _edge_key(self, features: Dict, action: str) -> str:
        """Create a compact context key for self-evolve (no heavy ML)."""
        try:
            a = str(action).upper()
            htf_dir = int(features.get('htf_dir', 0) or 0)
            htf_str = float(features.get('htf_strength', 0.0) or 0.0)
            trend = float(features.get('trend_strength', 0.0) or 0.0)
            atr_ratio = float(features.get('atr_ratio', 1.0) or 1.0)
            sweep = 1 if (bool(features.get('sweep_low', False)) or bool(features.get('sweep_high', False))) else 0
            # regime buckets
            htf_bucket = 'A' if (htf_dir != 0 and htf_str >= 0.25) else ('N' if htf_dir == 0 else 'W')
            tr_bucket = 'T' if abs(trend) >= 0.0022 else ('C' if atr_ratio < 0.82 else 'M')
            return f"{a}|{htf_bucket}|{tr_bucket}|S{sweep}"
        except Exception:
            return f"{str(action).upper()}|UNK"

    def update_edge(self, trade_data: Dict):
        """Update light-weight edge weights based on realized outcome."""
        try:
            sym = str(trade_data.get('symbol') or '')
            side = str(trade_data.get('side') or '')
            pnl = float(trade_data.get('pnl') or 0.0)
            feats = trade_data.get('features', {}) or {}
            key = self._edge_key(feats, side)
            if not hasattr(self, 'edge_weights'):
                self.edge_weights = {}
            if sym not in self.edge_weights:
                self.edge_weights[sym] = {}
            w = float((self.edge_weights[sym] or {}).get(key, 0.0) or 0.0)
            # EMA update: reward wins, penalize losses, cap magnitude
            lr = float(self.config.get('self_evolve', {}).get('edge_lr', 0.08) or 0.08) if isinstance(self.config, dict) else 0.08
            reward = 1.0 if pnl > 0 else (-1.0 if pnl < 0 else 0.0)
            w = (1.0 - lr) * w + lr * reward
            w = float(max(-0.95, min(0.95, w)))
            self.edge_weights[sym][key] = w
        except Exception:
            pass

    def get_edge_adjustment(self, symbol: str, features: Dict, action: str) -> float:
        """Return [-0.25..+0.25] confidence adjustment learned from recent trades."""
        try:
            sym = str(symbol)
            if not hasattr(self, 'edge_weights'):
                return 0.0
            key = self._edge_key(features or {}, action)
            w = float((self.edge_weights.get(sym) or {}).get(key, 0.0) or 0.0)
            # map weight to small confidence delta
            return float(max(-0.25, min(0.25, w * 0.18)))
        except Exception:
            return 0.0

# ===================== TRAINING =====================
    def add_trade_to_buffer(self, trade_data: Dict):
        # Update edge weights immediately on each closed trade
        try:
            self.update_edge(trade_data)
        except Exception:
            pass
        self.learning_buffer.append(trade_data)
        if len(self.learning_buffer) >= self.retrain_threshold:
            threading.Thread(target=self._retrain_models, daemon=True).start()

    def _retrain_models(self):
        """Light-weight retrain: updates edge weights from recent DB trades.
        Avoids heavy dependencies; works on Python 3.8 with current piplist.
        """
        try:
            logging.info("Bắt đầu retrain models (edge self-evolve)...")
            self.last_retrain_time = datetime.now()
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("SELECT symbol, side, pnl, features FROM trades ORDER BY id DESC LIMIT 2000")
            rows = c.fetchall()
            conn.close()
            if not rows:
                return
            for sym, side, pnl, feats_s in rows[: min(len(rows), 600)]:
                try:
                    feats = json.loads(feats_s) if isinstance(feats_s, str) else {}
                except Exception:
                    feats = {}
                self.update_edge({
                    'symbol': sym,
                    'side': side,
                    'pnl': float(pnl or 0.0),
                    'features': feats,
                })
            try:
                self.total_training_sessions = int(getattr(self, 'total_training_sessions', 0) or 0) + 1
                self.evolution_level = float(getattr(self, 'evolution_level', 1.0) or 1.0) + 0.01
            except Exception:
                pass
            logging.info(f"Self-evolve retrain ok | sessions={getattr(self,'total_training_sessions',0)}")
        except Exception as e:
            logging.error(f"Retrain failed: {e}")

    # ===================== AUTO SAVE =====================
    def _start_auto_save_thread(self):
        def saver():
            while self.auto_save_enabled:
                time.sleep(self.auto_save_interval * 60)
                try:
                    self._save_brain_state()
                except Exception as e:
                    logging.error(f"Auto-save failed: {e}")

        self.save_thread = threading.Thread(target=saver, daemon=True)
        self.save_thread.start()

    def _save_brain_state(self):
        joblib.dump({
            'models': self.models,
            'scalers': self.scalers,
            'global_model': self.global_model,
            'global_scaler': self.global_scaler,
            'feature_importance': self.feature_importance,
            'feature_importance_by_coin': self.feature_importance_by_coin,
            'evolution_level': self.evolution_level,
            'total_training_sessions': self.total_training_sessions,
            'model_accuracy_history': self.model_accuracy_history[-100:],
            'last_retrain': self.last_retrain_time.isoformat(),
            'edge_weights': getattr(self, 'edge_weights', {}),
            'regime_policy': getattr(self, 'regime_policy', {})
        }, self.brain_state_file)

    def _load_brain_state(self):
        if not os.path.exists(self.brain_state_file):
            return
        try:
            state = joblib.load(self.brain_state_file)
            self.models = state.get('models', {})
            self.scalers = state.get('scalers', {})
            self.global_model = state.get('global_model')
            self.global_scaler = state.get('global_scaler')
            self.feature_importance = state.get('feature_importance', {})
            self.feature_importance_by_coin = state.get('feature_importance_by_coin', {})
            self.evolution_level = state.get('evolution_level', 1)
            self.total_training_sessions = state.get('total_training_sessions', 0)
            self.model_accuracy_history = state.get('model_accuracy_history', [])
            self.last_retrain_time = datetime.fromisoformat(state.get('last_retrain', datetime.now().isoformat()))
            self.edge_weights = state.get('edge_weights', {}) or {}
            self.regime_policy = state.get('regime_policy', {}) or {}
        except Exception as e:
            logging.error(f"Failed to load brain state: {e}")

        # ===================== LIVE HEURISTIC =====================
    def _heuristic_predict(self, features: Dict) -> Tuple[float, float]:
        rsi = features.get("rsi", 50)
        momentum = features.get("price_momentum", 0)
        trend = features.get("trend_strength", 0)
        oflow = features.get("order_flow_delta", 0)
        vol = features.get("volatility", 1)

        score = (50 - rsi) * 0.02 + momentum * 2 + trend * 1.5 + oflow * 0.5 - vol * 0.3
        p_long = 1 / (1 + np.exp(-score))
        p_long = float(np.clip(p_long, 0.15, 0.85))
        return p_long, 1 - p_long

    # ===================== CORE PREDICTION =====================
    def predict_entry_probability(self, symbol: str, features: Dict) -> Tuple[float, float]:
        if symbol not in self.models or symbol not in self.scalers:
            return self._heuristic_predict(features)

        try:
            model = self.models[symbol]
            scaler = self.scalers[symbol]
            X = np.array([[features.get(f, 0.0) for f in self.feature_names]])
            Xs = scaler.transform(X) if hasattr(scaler, 'mean_') and scaler.mean_ is not None else X
            probas = model.predict_proba(Xs)[0]
            if len(probas) == 2:
                return float(probas[1]), float(probas[0])
            return self._heuristic_predict(features)
        except Exception:
            return self._heuristic_predict(features)

    def get_confidence(self, symbol: str, features: Dict) -> float:
        p1, p2 = self.predict_entry_probability(symbol, features)
        return float(max(p1, p2))



class AdvancedTradingBrain(PersistentTradingBrain):
    """Compatibility wrapper.
    Some older builds instantiated AdvancedTradingBrain directly and expected
    PersistentTradingBrain behaviors (auto-load, auto-save, DB init).
    """
    def __init__(self, db_path: str = 'data/trading_brain_v2.db', config: dict = None):
        super().__init__(db_path=db_path, config=config)

    # Backward-compatible aliases (in case other modules call these)
    def _load_brain_state(self):
        return super()._load_brain_state()

    def _save_brain_state(self):
        return super()._save_brain_state()