"""
═══════════════════════════════════════════════════════════════
🔐 IMPROVED SETUP SCRIPT V2.5 - FULL FIXED ENCRYPTION
═══════════════════════════════════════════════════════════════
- FIX: Không dùng SecurityManager.encrypt_api_keys (không tồn tại)
- Mã hóa hoàn toàn độc lập bằng hàm encrypt_keys
- Tương thích Python 3.8 + cryptography 41.0.7
- Thêm kiểm tra file ghi được trước khi mã hóa
- Cải thiện thông báo tiếng Việt
"""

import os
import sys
import json
import subprocess
import getpass
import re
import traceback
from datetime import datetime
from typing import Dict, List

# Import cần thiết cho mã hóa (giữ nguyên phiên bản cũ)
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64

# Import SecurityManager chỉ để validate (nếu cần)
try:
    from part1_security_marketdata_fixed import SecurityManager, EnhancedMarketDataEngine
except ImportError:
    print("⚠️ Không tìm thấy part1_security_marketdata_fixed.py")
    sys.exit(1)

from binance.client import Client
from binance.exceptions import BinanceAPIException

def print_banner():
    print("\n" + "="*80)
    print("🏆 GOD-TIER TRADING SYSTEM V2.0 - ENHANCED SETUP SCRIPT V2.5 (FIX ENCRYPT)")
    print("="*80)
    print("✨ Script sẽ giúp bạn:")
    print("  1. Kiểm tra môi trường Python & dependencies")
    print("  2. Validate API keys Binance (bỏ qua nếu lỗi)")
    print("  3. Mã hóa API keys → .env.encrypted")
    print("  4. Cấu hình Telegram")
    print("  5. Tạo thư mục & kiểm tra pre-flight")
    print("\n⚠️ LƯU Ý:")
    print("  - Passphrase phải ≥12 ký tự, có chữ hoa/thường/số/ký tự đặc biệt")
    print("  - GHI NHỚ PASSPHRASE - KHÔNG CÓ CÁCH KHÔI PHỤC NẾU QUÊN!")
    print("="*80 + "\n")

def check_python_version():
    print("🐍 Kiểm tra phiên bản Python...")
    v = sys.version_info
    print(f"   Phiên bản hiện tại: {v.major}.{v.minor}.{v.micro}")
    if v.major < 3 or (v.major == 3 and v.minor < 8):
        print("❌ Cần Python 3.8 trở lên!")
        return False
    print("✅ Python OK")
    return True

def check_pip():
    print("\n📦 Kiểm tra pip...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True, text=True, check=True
        )
        print(f"   {result.stdout.strip()}")
        return True
    except:
        print("❌ Không tìm thấy pip")
        return False

def upgrade_pip():
    print("\n📦 Nâng cấp pip (nếu cần)...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
            check=True, capture_output=True
        )
        print("✅ Pip upgraded")
    except:
        print("⚠️ Không nâng cấp được pip (tiếp tục)")

def verify_critical_imports() -> List[str]:
    """Verify critical imports.

    IMPORTANT:
    - Keep pip list unchanged. This function should NOT falsely report missing packages
      due to PyPI-name vs import-name mismatch.
    - Some environments intentionally do NOT install optional packages (e.g., ccxt).
      We therefore check modules by their real import names and treat some as optional.
    """

    # Core libs (must exist for the system to run)
    required_modules = [
        'numpy', 'pandas', 'cryptography',
    ]

    # Common ML stack (optional in some deployments; the bot still runs with heuristics)
    optional_modules = [
        'scipy', 'sklearn',
    ]

    # Exchange + Telegram (import names differ from pip names)
    # - python-binance -> import binance
    # - python-telegram-bot -> import telegram
    # - APScheduler -> import apscheduler
    exchange_modules = [
        'binance',
    ]
    telegram_modules = [
        'telegram',
    ]
    scheduler_modules = [
        'apscheduler',
    ]

    missing = []
    for mod in required_modules:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)

    # Optional checks: do not fail hard, but report if missing
    for mod in optional_modules + exchange_modules + telegram_modules + scheduler_modules:
        try:
            __import__(mod)
        except ImportError:
            # keep name for display
            missing.append(mod)

    # De-duplicate while keeping order
    seen = set()
    out = []
    for x in missing:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def install_missing_packages(missing: List[str]):
    if not missing:
        return
    print(f"\n📦 Cài đặt gói thiếu: {', '.join(missing)}")
    for pkg in missing:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            print(f"   ✓ {pkg}")
        except:
            print(f"   ✗ Không cài được {pkg}")

def validate_binance_api(api_key: str, api_secret: str) -> bool:
    if not api_key or not api_secret:
        print("API key/secret trống → bỏ qua validate")
        return False
    try:
        client = Client(api_key, api_secret)
        client.futures_account()  # Test quyền futures
        print("✅ API keys hợp lệ (Futures access OK)")
        return True
    except BinanceAPIException as e:
        print(f"⚠️ Validate thất bại: {e}")
        return False
    except Exception as e:
        print(f"⚠️ Lỗi bất ngờ khi validate: {e}")
        return False

def create_strong_passphrase() -> str:
    while True:
        print("\nTạo passphrase mạnh (≥12 ký tự, chữ hoa/thường/số/ký tự đặc biệt):")
        passphrase = getpass.getpass("Nhập passphrase: ").strip()
        confirm = getpass.getpass("Xác nhận lại passphrase: ").strip()

        if passphrase != confirm:
            print("❌ Passphrase không khớp!")
            continue

        if len(passphrase) < 12:
            print("❌ Passphrase quá ngắn (cần ≥12 ký tự)")
            continue

        has_upper = any(c.isupper() for c in passphrase)
        has_lower = any(c.islower() for c in passphrase)
        has_digit = any(c.isdigit() for c in passphrase)
        has_special = bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', passphrase))

        if not (has_upper and has_lower and has_digit and has_special):
            print("❌ Passphrase yếu! Cần đủ: chữ hoa, chữ thường, số, ký tự đặc biệt")
            continue

        print("✓ Passphrase mạnh và khớp!")
        return passphrase

def encrypt_keys(api_key: str, api_secret: str, passphrase: str, telegram_bot_token: str = '', telegram_chat_id: str = '') -> bool:
    """Hàm mã hóa độc lập - KHÔNG dùng SecurityManager"""
    try:
        # Kiểm tra có ghi file được không
        test_file = ".write_test.tmp"
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)

        # Tạo key từ passphrase
        salt = b'god_tier_salt_2026'  # Có thể random nhưng cần cố định để decrypt khớp
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key = base64.urlsafe_b64encode(kdf.derive(passphrase.encode()))
        fernet = Fernet(key)

        # Nội dung (giữ tương thích ngược với SecurityManager.decrypt_api_keys)
        # Có thể kèm thêm secrets khác (Telegram...) để không lộ trong config.json
        content = f"api_key={api_key}\napi_secret={api_secret}\n"
        if telegram_bot_token and telegram_chat_id:
            content += (
                f"telegram_bot_token={telegram_bot_token}\n"
                f"telegram_chat_id={telegram_chat_id}\n"
            )
        encrypted = fernet.encrypt(content.encode())

        # Ghi file
        with open('.env.encrypted', 'wb') as f:
            f.write(encrypted)

        print("✓ Mã hóa thành công → file .env.encrypted đã được tạo")
        return True

    except PermissionError:
        print("✗ Không có quyền ghi file vào thư mục hiện tại!")
        return False
    except Exception as e:
        print(f"✗ Lỗi mã hóa: {str(e)}")
        return False

def ask_telegram_config():
    print("\n📱 Cấu hình Telegram? (khuyến nghị)")
    setup_tg = input("Nhập 'y' để cấu hình, 'n' để bỏ qua: ").strip().lower()
    if setup_tg == 'y':
        bot_token = input("Bot Token (từ BotFather): ").strip()
        chat_id = input("Chat ID (từ @userinfobot): ").strip()
        if bot_token and chat_id:
            # NOTE: KHÔNG lưu token/chat_id vào config.json.
            # Token/chat_id sẽ được đưa vào .env.encrypted.
            return {
                'enabled': True,
                'bot_token': bot_token,
                'chat_id': chat_id
            }
    return None

def update_config(tg_config):
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        config = {}
    
    # Không lưu token/chat_id vào config. Chỉ lưu mapping tới ENV.
    config['telegram'] = {
        'enabled': bool(tg_config.get('enabled', False)),
        'env': {
            'bot_token': 'TELEGRAM_BOT_TOKEN',
            'chat_id': 'TELEGRAM_CHAT_ID'
        }
    }
    with open('config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print("✓ Đã cập nhật Telegram vào config.json")

def create_directories():
    print("\n📁 Tạo thư mục cần thiết...")
    dirs = ['logs', 'models', 'backups', 'data']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"   ✓ {d}/")

def run_preflight_checks() -> bool:
    print("\n🚀 Pre-flight checks...")
    checks = [
        (os.path.exists('.env.encrypted'), ".env.encrypted"),
        (os.path.exists('config.json'), "config.json")
    ]
    all_pass = True
    for ok, name in checks:
        print(f"   {'✅' if ok else '❌'} {name}")
        if not ok:
            all_pass = False
    return all_pass

def main():
    print_banner()
    
    if not check_python_version():
        sys.exit(1)
    
    if not check_pip():
        sys.exit(1)
    
    # Không tự ý upgrade pip để tránh lệch piplist/telegram
    if os.environ.get('ALLOW_PIP_UPGRADE', '').strip() == '1':
        upgrade_pip()
    else:
        print("\nℹ️ Bỏ qua upgrade pip (set ALLOW_PIP_UPGRADE=1 nếu bạn muốn)")
    
    missing = verify_critical_imports()
    if missing:
        print("\n⚠️ Phát hiện module thiếu: %s" % ', '.join(missing))
        print("   → Theo yêu cầu: KHÔNG tự cài/không đổi piplist.")
        print("   → Nếu bạn muốn tự cài thủ công, hãy chạy: pip install <ten_goi>")
        print("   (Bỏ qua và tiếp tục setup; một số tính năng có thể bị hạn chế)")

    api_key = input("\nNhập API Key Binance: ").strip()
    api_secret = input("Nhập API Secret Binance: ").strip()
    
    validate_binance_api(api_key, api_secret)
    
    passphrase = create_strong_passphrase()
    
    print("\nĐang mã hóa API keys...")
    if encrypt_keys(api_key, api_secret, passphrase):
        print("Mã hóa hoàn tất!")
    else:
        print("Mã hóa thất bại → dừng setup")
        sys.exit(1)
    
    # Luôn hỏi Telegram (nếu có, sẽ được mã hóa vào .env.encrypted và chỉ lưu ENV mapping trong config)
    tg_config = ask_telegram_config()
    if tg_config and tg_config.get('enabled'):
        print("\nĐang cập nhật .env.encrypted kèm Telegram secrets...")
        if not encrypt_keys(api_key, api_secret, passphrase, tg_config.get('bot_token',''), tg_config.get('chat_id','')):
            print("⚠️ Không thể ghi .env.encrypted kèm Telegram. Tiếp tục với API keys thôi.")
        update_config(tg_config)
    
    create_directories()
    
    if run_preflight_checks():
        print("\n✅ Pre-flight checks passed!")
    else:
        print("\n⚠️ Một số file chưa có, nhưng bạn vẫn có thể chạy tiếp")
    
    print("\n" + "="*80)
    print("🎉 SETUP HOÀN TẤT!")
    print("="*80)
    print(f"Passphrase của bạn: {passphrase}")
    print("GHI LẠI NGAY - KHÔNG CHIA SẺ!")
    print("\nTiếp theo:")
    print("  python trading_system_v2.py")
    print("  Nhập passphrase khi được hỏi")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Đã hủy setup")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Lỗi nghiêm trọng: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
