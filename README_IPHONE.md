# iPhone Web Dashboard (giữ nguyên toàn bộ tính năng của bản PC)

## Mục tiêu
- Không bỏ qua file/tính năng nào trong `claude3_fix/`
- Chạy bot như trên PC nhưng điều khiển bằng iPhone qua trình duyệt (PWA)
- Bot chạy 24/7 phải chạy trên server (VPS/Fly.io/host), iPhone chỉ là bảng điều khiển

## Chạy local
```bash
pip install -r requirements.txt
uvicorn iphone_webapp.main:app --host 0.0.0.0 --port 8000
```
Mở: http://<ip>:8000

## Chạy bot
- Nhấn Start để chạy `python -u claude3_fix/trading_system_v2.py` như PC
- Log lấy từ `claude3_fix/trading_v2.log`
- Passphrase `.env.encrypted` có thể set bằng biến môi trường `ENV_PASSPHRASE` để chạy non-interactive
