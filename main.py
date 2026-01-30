from __future__ import annotations
import os, subprocess, threading, time, pathlib, signal
from typing import Optional, Dict, Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse

APP_ROOT = pathlib.Path(__file__).resolve().parent.parent
BOT_DIR = APP_ROOT / "claude3_fix"
LOG_FILE = BOT_DIR / "trading_v2.log"

# Global process handle
_proc: Optional[subprocess.Popen] = None
_start_ts: Optional[float] = None
_lock = threading.Lock()

def _env() -> Dict[str, str]:
    env = os.environ.copy()
    # allow non-interactive passphrase for .env.encrypted
    if "ENV_PASSPHRASE" in env:
        env["ENV_PASSPHRASE"] = env["ENV_PASSPHRASE"]
    return env

def start_bot() -> Dict[str, Any]:
    global _proc, _start_ts
    with _lock:
        if _proc and _proc.poll() is None:
            return {"ok": True, "status": "already_running", "pid": _proc.pid}
        cmd = ["python", "-u", "trading_system_v2.py"]
        _proc = subprocess.Popen(
            cmd,
            cwd=str(BOT_DIR),
            env=_env(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _start_ts = time.time()
        return {"ok": True, "status": "started", "pid": _proc.pid}

def stop_bot() -> Dict[str, Any]:
    global _proc, _start_ts
    with _lock:
        if not _proc or _proc.poll() is not None:
            _proc = None
            _start_ts = None
            return {"ok": True, "status": "not_running"}
        try:
            _proc.send_signal(signal.SIGTERM)
            _proc.wait(timeout=10)
        except Exception:
            try:
                _proc.kill()
            except Exception:
                pass
        pid = _proc.pid
        _proc = None
        _start_ts = None
        return {"ok": True, "status": "stopped", "pid": pid}

def bot_status() -> Dict[str, Any]:
    with _lock:
        running = bool(_proc and _proc.poll() is None)
        pid = _proc.pid if running else None
        uptime = (time.time() - _start_ts) if (running and _start_ts) else 0.0
    return {"running": running, "pid": pid, "uptime_sec": round(uptime, 1)}

def tail_log(lines: int = 200) -> str:
    try:
        if not LOG_FILE.exists():
            return f"(log not found at {LOG_FILE})"
        data = LOG_FILE.read_text(errors="ignore").splitlines()
        return "\n".join(data[-lines:])
    except Exception as e:
        return f"(error reading log: {e})"

app = FastAPI(title="Trading System iPhone Control")

INDEX_HTML = """<!doctype html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Trading System Control</title>
  <style>
    body { font-family: -apple-system, system-ui, Arial; margin: 16px; }
    .row { display:flex; gap:12px; flex-wrap: wrap; }
    button { font-size: 18px; padding: 12px 16px; border-radius: 10px; border: 0; }
    pre { background:#f5f5f7; padding: 12px; border-radius: 10px; overflow:auto; max-height: 60vh; }
    .card { background:#fff; border:1px solid #eee; padding:12px; border-radius: 12px; margin: 12px 0; }
    .muted { color:#666; }
  </style>
</head>
<body>
  <h2>Trading System (PC-equivalent) — iPhone Dashboard</h2>
  <div class="card">
    <div id="status" class="muted">Loading...</div>
    <div class="row" style="margin-top:12px;">
      <button onclick="doPost('/start')">▶ Start</button>
      <button onclick="doPost('/stop')">⏹ Stop</button>
      <button onclick="refresh()">↻ Refresh</button>
    </div>
  </div>

  <div class="card">
    <div class="row" style="align-items:center; justify-content:space-between;">
      <b>Logs (tail)</b>
      <span class="muted">Auto-refresh 5s</span>
    </div>
    <pre id="log">(loading...)</pre>
  </div>

<script>
async function doPost(path){
  const res = await fetch(path, {method:'POST'});
  const j = await res.json();
  await refresh();
  alert(JSON.stringify(j));
}
async function refresh(){
  const s = await (await fetch('/status')).json();
  document.getElementById('status').innerText =
    (s.running ? 'RUNNING' : 'STOPPED') + ' | pid=' + (s.pid ?? '-') + ' | uptime=' + s.uptime_sec + 's';
  const log = await (await fetch('/logs?lines=250')).text();
  document.getElementById('log').innerText = log;
}
refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML

@app.get("/status")
def status():
    return JSONResponse(bot_status())

@app.post("/start")
def start():
    return JSONResponse(start_bot())

@app.post("/stop")
def stop():
    return JSONResponse(stop_bot())

@app.get("/logs", response_class=PlainTextResponse)
def logs(lines: int = 200):
    lines = max(50, min(2000, int(lines)))
    return tail_log(lines)
