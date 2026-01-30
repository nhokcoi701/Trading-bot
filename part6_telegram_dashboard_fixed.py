"""
Part 6: Telegram Dashboard - Complete & Fixed Version
- Hoàn thiện phần bị truncated
- Thêm rate limit cho send_message
- Hoàn thiện visualization (chart) nếu có matplotlib
- Format đẹp, emoji phong phú hơn
- Cập nhật: Báo cáo tổng hợp 5 phút/lần, việt hóa, chi tiết mode & coin
- Note lệnh Telegram ở dưới
- FIX: dùng getattr cho metrics để tránh AttributeError
- FIX: Hiển thị giá hiện tại real-time (từ market_data) cho mọi coin, kể cả chưa mở lệnh
- FIX: Truyền market_data vào TelegramReporter để lấy giá 1:1 như live ở mọi mode
- FIX MỚI: Bỏ decorator @handle_errors cho send_message để tránh TypeError
- NÂNG CẤP: Lệnh /report báo chi tiết từng mode riêng (coin, lệnh, hiệu suất, pnl, số dư) - mỗi mode 1 khung riêng, fix placeholder
"""

import os
import requests
from requests.adapters import HTTPAdapter
try:
    # urllib3 is a dependency of requests; Retry is available without extra pip installs
    from urllib3.util.retry import Retry
except Exception:
    Retry = None
import io
import logging
# Silence urllib3 retry warnings (can leak full request URL)
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('urllib3.connectionpool').setLevel(logging.ERROR)
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_ENABLED = True
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
except ImportError:
    VISUALIZATION_ENABLED = False
    logging.warning("⚠️ Matplotlib/seaborn không có - visualization bị tắt")


class TelegramReporter:
    """Telegram reporter with rate limit, rich formatting & visualization support"""
    
    def __init__(self, bot_token: str, chat_id: str, market_data=None):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.market_data = market_data  # Truyền market_data để lấy giá real-time
        self.last_send_time = 0
        self._fail_count = 0
        self._disabled_until = 0.0
        self.rate_limit_seconds = 1  # Rate limit Telegram API
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.trades_today = 0
        self.wins_today = 0
        self.last_day = datetime.now().date()
        self.last_week = datetime.now().isocalendar()[1]

        # Network tuning (Vietnam/ISP environments can be unstable for api.telegram.org)
        self.tg_connect_timeout = float(os.getenv('TELEGRAM_CONNECT_TIMEOUT', '10') or 10)
        self.tg_read_timeout = float(os.getenv('TELEGRAM_READ_TIMEOUT', '40') or 40)
        # Optional proxy (SOCKS/HTTP). Example: TELEGRAM_PROXY=socks5h://127.0.0.1:1080
        self.tg_proxy = (os.getenv('TELEGRAM_PROXY') or '').strip()

        # Clamp timeouts to safe minimums (some envs set too low causing false timeouts)
        try:
            self.tg_connect_timeout = max(5.0, float(self.tg_connect_timeout))
        except Exception:
            self.tg_connect_timeout = 10.0
        try:
            self.tg_read_timeout = max(20.0, float(self.tg_read_timeout))
        except Exception:
            self.tg_read_timeout = 40.0


        self._last_error_log_ts = 0.0

        # Reusable session with retries
        self._session = requests.Session()
        try:
            if Retry is not None:
                retry = Retry(
                    total=3,
                    connect=3,
                    read=3,
                    backoff_factor=0.6,
                    status_forcelist=(429, 500, 502, 503, 504),
                    allowed_methods=frozenset(['GET', 'POST'])
                )
                adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
                self._session.mount('https://', adapter)
                self._session.mount('http://', adapter)
        except Exception:
            # Safe fallback: session without retries
            pass

        self.fee_rate = 0.0004  # 0.04% ước tính cho futures (taker)

        # Số dư và pnl cho từng mode
        self.mode_balances = {
            'demo': 1000.0,
            'shadow': 1000.0,
            'paper': 1000.0,
            'live': 1000.0
        }
        self.mode_pnl = {
            'demo': 0.0,
            'shadow': 0.0,
            'paper': 0.0,
            'live': 0.0
        }
    
    def send_message(self, text: str, parse_mode: str = 'HTML'):
        """Gửi message với rate limit, không decorator để tránh TypeError"""
        try:
            current_time = time.time()
            if float(getattr(self, '_disabled_until', 0.0) or 0.0) > current_time:
                return
            if current_time - self.last_send_time < self.rate_limit_seconds:
                time.sleep(self.rate_limit_seconds - (current_time - self.last_send_time))
            
            proxies = None
            if self.tg_proxy:
                proxies = {'http': self.tg_proxy, 'https': self.tg_proxy}
            response = self._session.post(
                f"https://api.telegram.org/bot{self.bot_token}/sendMessage",
                data={'chat_id': self.chat_id, 'text': text, 'parse_mode': parse_mode},
                timeout=(self.tg_connect_timeout, self.tg_read_timeout),
                proxies=proxies
            )
            self.last_send_time = time.time()
            
            try:
                payload = response.json()
            except Exception:
                payload = {'ok': False, 'raw': getattr(response, 'text', '')}

            if not payload.get('ok'):
                # Log đủ để debug các lỗi thường gặp: chat_id sai, bot bị block, token sai...
                # Avoid log spam when Telegram is blocked; log at most once every 60s
                now_ts = time.time()
                if now_ts - float(getattr(self, '_last_error_log_ts', 0.0) or 0.0) > 60.0:
                    self._last_error_log_ts = now_ts
                    logging.error(f"Telegram send failed (HTTP {getattr(response, 'status_code', '?')}): {payload}")
        except Exception as e:
            try:
                fc = int(getattr(self, '_fail_count', 0) or 0) + 1
                self._fail_count = fc
                if fc >= 3:
                    # disable Telegram for 3 minutes
                    self._disabled_until = time.time() + 180
            except Exception:
                pass
            now_ts = time.time()
            if now_ts - float(getattr(self, '_last_error_log_ts', 0.0) or 0.0) > 60.0:
                self._last_error_log_ts = now_ts
                logging.error(f"Error sending Telegram message: {e}")

    def ping(self) -> bool:
        """Kiểm tra nhanh Telegram bot có hoạt động không.

        - getMe: xác nhận token OK
        - sendMessage: xác nhận chat_id OK
        """
        try:
            proxies = None
            if self.tg_proxy:
                proxies = {'http': self.tg_proxy, 'https': self.tg_proxy}
            r = self._session.get(
                f"https://api.telegram.org/bot{self.bot_token}/getMe",
                timeout=(self.tg_connect_timeout, self.tg_read_timeout),
                proxies=proxies
            )
            data = r.json() if hasattr(r, 'json') else {}
            if not data.get('ok'):
                logging.error(f"Telegram getMe failed: {data}")
                return False
        except Exception as e:
            # Log at most once per minute (Telegram can be blocked at network level)
            now_ts = time.time()
            if now_ts - float(getattr(self, '_last_error_log_ts', 0.0) or 0.0) > 60.0:
                self._last_error_log_ts = now_ts
                logging.error(f"Telegram getMe error: {e}")
            return False

        try:
            self.send_message("✅ Telegram connected. Bot is alive.")
            return True
        except Exception:
            return False


    # ===================== DASHBOARD HELPERS =====================

    def _html_escape(self, s: str) -> str:
        return (s.replace('&', '&amp;')
                 .replace('<', '&lt;')
                 .replace('>', '&gt;'))

    def _fmt_money(self, x: float) -> str:
        try:
            return f"{x:,.2f}"
        except Exception:
            return str(x)

    def _fmt_px(self, v: float) -> str:
        try:
            if v is None or v == 0:
                return '-'
            return f"{v:.4f}" if v < 100 else f"{v:.2f}"
        except Exception:
            return '-'

    def _coin_box(self, symbol: str, price: float, pos, default_leverage: int) -> tuple:
        # pos có thể là Position hoặc None
        side = str(getattr(pos, 'side', '-') or '-').upper()
        entry = float(getattr(pos, 'entry_price', 0.0) or 0.0)
        sl = float(getattr(pos, 'stop_loss', 0.0) or 0.0)
        tp = float(getattr(pos, 'take_profit', 0.0) or 0.0)
        qty = float(getattr(pos, 'quantity', 0.0) or 0.0)
        lev = int(getattr(pos, 'leverage', default_leverage) or default_leverage)
        add_n = int(getattr(pos, 'pyramid_count', 0) or 0) if pos is not None else 0

        has_pos = pos is not None and qty > 0 and entry > 0

        # Volume USD: size notional ước tính = qty * (giá) * leverage
        ref_price = float(price or entry or 0.0)
        vol_usd = abs(qty * ref_price * lev) if has_pos else 0.0

        # Fee sàn ước tính (round-trip) theo taker fee
        fee_est = abs(qty * entry * lev) * float(self.fee_rate) * 2 if has_pos else 0.0

        # PnL tạm tính
        upnl = 0.0
        if has_pos and price and price > 0:
            if side == 'LONG':
                upnl = (price - entry) * qty * lev - fee_est
            elif side == 'SHORT':
                upnl = (entry - price) * qty * lev - fee_est

        lines = []
        lines.append(f"┌─ {symbol:<8} ─────────────────────────┐")
        lines.append(f"│ Px     : {self._fmt_px(price):>14}               │")
        lines.append(f"│ Entry  : {(self._fmt_px(entry) if has_pos else '-'):>14}               │")
        lines.append(f"│ Side   : {side:<14}               │")
        lines.append(f"│ Add    : {add_n:<14}               │")
        sltp = f"{self._fmt_px(sl) if has_pos else '-'} / {self._fmt_px(tp) if has_pos else '-'}"
        lines.append(f"│ SL/TP  : {sltp:<20}          │")
        lines.append(f"│ Vol$   : {self._fmt_money(vol_usd):>12}  Lev:{lev:>3}x      │")
        lines.append(f"│ Fee$   : {self._fmt_money(fee_est):>12}  uPnL:{self._fmt_money(upnl):>9} │")
        lines.append(f"└──────────────────────────────────────┘")
        return "\n".join(lines), upnl, fee_est

    def report_dashboard(self, risk_metrics, modes, positions_by_mode, force_send: bool = False):
        '''Dashboard: mỗi MODE 1 message, gồm 4 khung coin.

        - Dù chưa có lệnh vẫn phải có giá hiện tại.
        - Mỗi khung coin: tên coin, giá hiện tại, giá vào, SL/TP, volume USD, đòn bẩy, tổng phí sàn, pnl tạm tính.
        - Có tổng số dư của mode.
        '''
        symbols = []
        if self.market_data is not None and hasattr(self.market_data, 'symbols'):
            try:
                symbols = list(self.market_data.symbols)
            except Exception:
                symbols = []
        if not symbols:
            symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']

        default_leverage = 1
        try:
            # Ưu tiên leverage trong config nếu có, fallback 1
            default_leverage = int(getattr(risk_metrics, 'default_leverage', 1) or 1)
        except Exception:
            default_leverage = 1

        now_str = datetime.now().strftime('%H:%M:%S %d/%m/%Y')

        for mode in ['demo', 'shadow', 'paper', 'live']:
            enabled = bool(modes.get(mode, False))
            pos_map = (positions_by_mode or {}).get(mode, {}) or {}

            total_upnl = 0.0
            total_fee = 0.0
            boxes = []

            for sym in symbols[:4]:
                try:
                    price = float(self.market_data.get_current_price(sym)) if self.market_data else 0.0
                except Exception:
                    price = 0.0

                pos = pos_map.get(sym)
                box, upnl, fee = self._coin_box(sym, price, pos, default_leverage)
                total_upnl += upnl
                total_fee += fee
                boxes.append(box)

            base_balance = float(self.mode_balances.get(mode, 1000.0))
            balance = base_balance + total_upnl

            header = (
                f"📟 <b>MODE {mode.upper()}</b> | {'🟢BẬT' if enabled else '⚪TẮT'}\n"
                f"⏱ {now_str}\n"
                f"💼 Số dư: <b>${self._fmt_money(balance)}</b> | uPnL: <b>${self._fmt_money(total_upnl)}</b> | Phí ước tính: <b>${self._fmt_money(total_fee)}</b>\n"
                f"<i>(Giá real-time: market_data)</i>\n"
            )

            body = "\n\n".join(boxes)
            msg = header + "<pre>" + self._html_escape(body) + "</pre>"
            self.send_message(msg)

    # ===================== END DASHBOARD HELPERS =====================

    def report_status(self, risk_metrics, modes: Dict, positions: Dict):
        """Backward compatible: nếu caller vẫn truyền positions dạng dict symbol->pos,
        ta coi như cùng 1 snapshot cho tất cả mode.
        """
        positions_by_mode = {
            'demo': positions or {},
            'shadow': positions or {},
            'paper': positions or {},
            'live': positions or {},
        }
        self.report_dashboard(risk_metrics, modes, positions_by_mode)

    def report_detailed(self, modes: Dict, positions: Dict, risk_metrics, trades: List[Dict]):
        """
        /report: Báo cáo hiệu suất THỰC (đọc từ các file trades_*.jsonl + events_*.jsonl nếu có).

        Fixes:
        - Trước đây code dùng key 'pnl' (không tồn tại trong jsonl) => win luôn = 0.
        - Win nhỏ (do phí) bị làm tròn thành 0.00 => tưởng không cập nhật.
        - Số dư: ưu tiên lấy balance/equity từ events CLOSE payload (nếu có), không tự cộng giả.
        """

        # Helper: load jsonl safely (supports current folder or data/)
        def _load_jsonl(relpath: str):
            out = []
            try:
                import json
                from pathlib import Path
                p = Path(relpath)
                if not p.exists():
                    return out
                for line in p.read_text(encoding='utf-8', errors='ignore').splitlines():
                    line = (line or '').strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except Exception:
                        continue
            except Exception:
                return out
            return out

        def _fmt_money(x: float) -> str:
            try:
                x = float(x or 0.0)
            except Exception:
                x = 0.0
            # Với pnl rất nhỏ (thường do fee), vẫn hiển thị 4 decimals để không bị “mất”
            if abs(x) < 0.01:
                return f"{x:+.4f}"
            return f"{x:+.2f}"

        def _aggregate(mode: str):
            # Trades
            trades_rows = _load_jsonl(f"trades_{mode}.jsonl")
            if not trades_rows:
                trades_rows = _load_jsonl(f"data/trades_{mode}.jsonl")

            closes = [r for r in trades_rows if str(r.get('event', r.get('type', ''))).upper() == 'CLOSE']

            pnls = []
            for r in closes:
                v = r.get('pnl_net', None)
                if v is None:
                    v = r.get('pnl', 0.0)
                try:
                    pnls.append(float(v))
                except Exception:
                    pnls.append(0.0)

            win = [p for p in pnls if p > 0]
            loss = [p for p in pnls if p < 0]

            # Balance/equity from events
            ev_rows = _load_jsonl(f"events_{mode}.jsonl")
            if not ev_rows:
                ev_rows = _load_jsonl(f"data/events_{mode}.jsonl")

            last_bal = None
            last_eq = None
            last_ts = None
            for ev in reversed(ev_rows):
                if str(ev.get('type', '')).upper() == 'CLOSE':
                    payload = ev.get('payload') or {}
                    if isinstance(payload, dict):
                        last_bal = payload.get('balance', None)
                        last_eq = payload.get('equity', None)
                    last_ts = ev.get('ts')
                    break

            return {
                'n': len(pnls),
                'win_n': len(win),
                'loss_n': len(loss),
                'win_pnl': sum(win) if win else 0.0,
                'loss_pnl': sum(loss) if loss else 0.0,
                'net_pnl': sum(pnls) if pnls else 0.0,
                'balance': float(last_bal) if last_bal is not None else None,
                'equity': float(last_eq) if last_eq is not None else None,
                'last_ts': last_ts,
            }

        for mode, enabled in (modes or {}).items():
            if not enabled:
                continue

            st = _aggregate(str(mode).lower())
            n = st['n']
            win_n = st['win_n']
            loss_n = st['loss_n']
            winrate = (win_n / max(n, 1) * 100.0)
            msg = f"📊 <b>/REPORT – HIỆU SUẤT MODE {str(mode).upper()}</b>\n\n"
            msg += f"🧾 Trades(close): <b>{n}</b> | ✅ Win: <b>{win_n}</b> | ❌ Lose: <b>{loss_n}</b> | 🎯 Winrate: <b>{winrate:.1f}%</b>\n"
            msg += f"💰 PnL Win: <b>{_fmt_money(st['win_pnl'])}</b> | PnL Lose: <b>{_fmt_money(st['loss_pnl'])}</b> | Net: <b>{_fmt_money(st['net_pnl'])}</b>\n"

            if st['balance'] is not None or st['equity'] is not None:
                bal_txt = f"{st['balance']:.4f}" if st['balance'] is not None else "N/A"
                eq_txt = f"{st['equity']:.4f}" if st['equity'] is not None else "N/A"
                msg += f"🏦 Balance: <b>${bal_txt}</b> | Equity: <b>${eq_txt}</b>\n"
                if st['last_ts']:
                    msg += f"🕒 Last close: <code>{st['last_ts']}</code>\n"

            # Snapshot open positions (optional)
            try:
                if positions:
                    open_syms = list(positions.keys())[:8]
                    if open_syms:
                        msg += "\n📌 Vị thế đang mở (snapshot):\n"
                        for s in open_syms:
                            ppos = positions.get(s)
                            side = getattr(ppos, 'side', None) or (ppos.get('side') if isinstance(ppos, dict) else None)
                            entry = getattr(ppos, 'entry_price', None) or (ppos.get('entry') if isinstance(ppos, dict) else None)
                            msg += f"• {s}: {side or '?'} @ {entry if entry is not None else '?'}\n"
            except Exception:
                pass

            self.send_message(msg)

    def report_daily_summary(self):
        if self.trades_today == 0:
            return
        
        msg = f"📅 <b>BÁO CÁO HÀNG NGÀY</b> - Ngày {datetime.now().strftime('%d/%m/%Y')}\n\n"
        msg += f"💰 PnL ngày: <b>${self.daily_pnl:.2f}</b>\n"
        msg += f"📊 Trades: {self.trades_today}\n"
        msg += f"🏆 Wins: {self.wins_today}\n"
        winrate = (self.wins_today / self.trades_today * 100) if self.trades_today > 0 else 0
        msg += f"📈 Winrate: {winrate:.1f}%\n\n"
        msg += "<i>Tiếp tục giữ vững phong độ!</i>"
        self.send_message(msg)

    def report_weekly_summary(self, risk_metrics, trades_this_week: List[Dict]):
        if not trades_this_week:
            return
        
        wins = [t for t in trades_this_week if t.get('pnl', 0) > 0]
        losses = [t for t in trades_this_week if t.get('pnl', 0) < 0]
        
        msg = f"📆 <b>BÁO CÁO TUẦN</b> - Tuần kết thúc {datetime.now().strftime('%d/%m/%Y')}\n\n"
        msg += f"💰 PnL tuần này: <b>${risk_metrics.weekly_pnl:,.2f} ({risk_metrics.weekly_pnl_pct:+.2f}%)</b>\n\n"
        msg += "📊 Thống kê:\n"
        msg += f"├─ Tổng lệnh: {len(trades_this_week)}\n"
        msg += f"├─ Thắng / Thua: {len(wins)} / {len(losses)}\n"
        msg += f"├─ Winrate: {len(wins)/max(len(trades_this_week),1)*100:.1f}%\n"
        msg += "└─ Profit Factor: ∞ (nếu không có thua) hoặc tính toán tương ứng\n\n"
        msg += "<i>Tuần mới - cơ hội mới! 🔥</i>"
        self.send_message(msg)

    def send_error_alert(self, error_data: Dict):
        msg = f"🚨 <b>LỖI NGHIÊM TRỌNG</b>\n\n"
        msg += f"Loại: {error_data.get('type', 'Không rõ')}\n"
        msg += f"Ngữ cảnh: {error_data.get('context', 'N/A')}\n"
        msg += f"Chi tiết: {error_data.get('message', 'Không có')[:300]}\n\n"
        msg += f"Thời gian: {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}\n\n"
        msg += "<i>Vui lòng kiểm tra log ngay!</i>"
        self.send_message(msg)

    def send_chart(self, symbol: str):
        if not VISUALIZATION_ENABLED:
            self.send_message("Visualization không khả dụng (thiếu matplotlib)")
            return
        
        try:
            df = self.market_data.get_klines(symbol, '5m', limit=200)
            if df is None or df.empty:
                self.send_message(f"Không có dữ liệu chart cho {symbol}")
                return
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df.index, df['close'], label='Close Price', color='cyan', linewidth=2)
            ax.set_title(f"Chart {symbol} - 5m", fontsize=14)
            ax.set_xlabel("Time", fontsize=12)
            ax.set_ylabel("Price", fontsize=12)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            
            # Không phụ thuộc python-telegram-bot: dùng Bot API trực tiếp
            files = {'photo': (f"{symbol}.png", buf.getvalue())}
            requests.post(
                f"https://api.telegram.org/bot{self.bot_token}/sendPhoto",
                data={'chat_id': self.chat_id, 'caption': f"📈 {symbol} chart (5m)"},
                files=files,
                timeout=30
            )
            buf.close()
            plt.close(fig)
        except Exception as e:
            logging.error(f"Send chart failed: {e}")
            self.send_message(f"Lỗi gửi chart {symbol}: {str(e)}")

    def update_daily_stats(self, pnl: float, is_win: bool):
        self.daily_pnl += pnl
        self.trades_today += 1
        if is_win:
            self.wins_today += 1
        
        now = datetime.now()
        if now.date() != self.last_day:
            self.report_daily_summary()
            self.daily_pnl = 0.0
            self.trades_today = 0
            self.wins_today = 0
            self.last_day = now.date()
        
        if now.isocalendar()[1] != self.last_week:
            # Tránh crash do thay đổi chữ ký hàm report_weekly_summary
            try:
                self.report_weekly_summary(None, [])
            except TypeError:
                try:
                    self.report_weekly_summary()
                except Exception:
                    pass
            except Exception:
                pass
            self.weekly_pnl = 0.0
            self.last_week = now.isocalendar()[1]
