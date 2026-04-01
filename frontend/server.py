import asyncio
import json
import os
import sqlite3
import threading
import time
from multiprocessing import Queue
from queue import Empty

from aiohttp import web

from core import config

_connected_ws: set = set()
_DB_CHANGE_KEYWORDS = (
    "Created new identity",
    "Saved transcript",
    "Registered voice",
    "Verified existing identity",
    "Registered face",
)

FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _read_db_snapshot(db_path: str) -> dict:
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        conn.execute("PRAGMA journal_mode = WAL;")
        cur = conn.cursor()

        cur.execute("SELECT id, name FROM users")
        users_raw = cur.fetchall()

        cur.execute("SELECT user_id FROM face_embeddings")
        face_user_ids = {r[0] for r in cur.fetchall()}

        cur.execute("SELECT user_id FROM voice_embeddings")
        voice_user_ids = {r[0] for r in cur.fetchall()}

        cur.execute("SELECT COUNT(*) FROM face_embeddings")
        face_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM voice_embeddings")
        voice_count = cur.fetchone()[0]

        cur.execute("""
            SELECT ch.id, ch.user_id, u.name, ch.transcript, ch.timestamp
            FROM chat_history ch
            JOIN users u ON ch.user_id = u.id
            ORDER BY ch.id DESC
            LIMIT 50
        """)
        chat_raw = cur.fetchall()
        conn.close()

        users = [
            {
                "id": uid,
                "name": name,
                "has_face": uid in face_user_ids,
                "has_voice": uid in voice_user_ids,
            }
            for uid, name in users_raw
        ]

        chat_history = [
            {"id": r[0], "user_id": r[1], "user_name": r[2], "transcript": r[3], "timestamp": str(r[4])}
            for r in reversed(chat_raw)
        ]

        return {
            "type": "db_update",
            "users": users,
            "chat_history": chat_history,
            "face_count": face_count,
            "voice_count": voice_count,
            "chat_count": len(chat_raw),
        }
    except Exception as e:
        return {"type": "db_update", "error": str(e), "users": [], "chat_history": [], "face_count": 0, "voice_count": 0, "chat_count": 0}


# ---------------------------------------------------------------------------
# Broadcast helpers
# ---------------------------------------------------------------------------

async def _broadcast(msg: dict):
    if not _connected_ws:
        return
    text = json.dumps(msg)
    dead = set()
    for ws in list(_connected_ws):
        try:
            await ws.send_str(text)
        except Exception:
            dead.add(ws)
    _connected_ws.difference_update(dead)


# ---------------------------------------------------------------------------
# Background async tasks
# ---------------------------------------------------------------------------

async def _log_drain_loop(log_queue: Queue, db_path: str):
    while True:
        drained = 0
        while drained < 50:  # batch up to 50 per tick
            try:
                msg = log_queue.get_nowait()
                await _broadcast({"type": "log", **msg})
                # trigger a DB snapshot if this log line signals a DB change
                if any(kw in msg.get("text", "") for kw in _DB_CHANGE_KEYWORDS):
                    snapshot = _read_db_snapshot(db_path)
                    await _broadcast(snapshot)
                drained += 1
            except Empty:
                break
            except Exception:
                break
        await asyncio.sleep(0.05)


async def _db_poll_loop(db_path: str):
    while True:
        await asyncio.sleep(5)
        snapshot = _read_db_snapshot(db_path)
        await _broadcast(snapshot)


# ---------------------------------------------------------------------------
# HTTP / WebSocket handlers
# ---------------------------------------------------------------------------

async def _index_handler(request):
    return web.FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


async def _db_api_handler(request):
    db_path = request.app["db_path"]
    return web.json_response(_read_db_snapshot(db_path))


async def _ws_handler(request):
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)
    _connected_ws.add(ws)

    # Send initial DB snapshot immediately so the panel isn't empty
    snapshot = _read_db_snapshot(request.app["db_path"])
    await ws.send_str(json.dumps(snapshot))

    try:
        async for _ in ws:
            pass  # we don't expect messages from the client
    finally:
        _connected_ws.discard(ws)

    return ws


# ---------------------------------------------------------------------------
# Server startup
# ---------------------------------------------------------------------------

def _run_server_loop(log_queue: Queue, db_path: str, host: str, port: int):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    app = web.Application()
    app["db_path"] = db_path

    app.router.add_get("/", _index_handler)
    app.router.add_get("/ws", _ws_handler)
    app.router.add_get("/api/db", _db_api_handler)

    async def _serve():
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        asyncio.ensure_future(_log_drain_loop(log_queue, db_path))
        asyncio.ensure_future(_db_poll_loop(db_path))
        # run forever
        while True:
            await asyncio.sleep(3600)

    loop.run_until_complete(_serve())


def start_monitoring_server(log_queue: Queue, db_path: str):
    host = config.MONITORING_HOST
    port = config.MONITORING_PORT
    t = threading.Thread(
        target=_run_server_loop,
        args=(log_queue, db_path, host, port),
        daemon=True,
        name="MonitoringServer",
    )
    t.start()
    print(f"[System] Monitoring dashboard at http://localhost:{port}")
