import os
import asyncio
import logging
import sqlite3
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import httpx
from telethon import TelegramClient, events
from telethon.sessions import StringSession
from telethon.errors import RPCError

# ---------- CONFIG ----------
API_ID = int(os.environ["TG_API_ID"])
API_HASH = os.environ["TG_API_HASH"]
TG_SESSION = os.environ["TG_SESSION"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")
ALLOWED_CHATS_RAW = os.getenv("KISSA_ALLOWED_CHATS", "").strip()
FOLLOWUP_ENABLED = os.getenv("FOLLOWUP_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
FIRST_MESSAGE_ENABLED = os.getenv("FIRST_MESSAGE_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
FOLLOWUP_HOURS = max(1, int(os.getenv("FOLLOWUP_HOURS", "4")))
FOLLOWUP_COOLDOWN_HOURS = max(1, int(os.getenv("FOLLOWUP_COOLDOWN_HOURS", "24")))
FIRST_MESSAGE_TEXT = os.getenv("FIRST_MESSAGE_TEXT", "").strip()
TIMEZONE = ZoneInfo("Europe/Moscow")
DB_PATH = os.getenv("KISSA_DB_PATH", "kissa_memory.sqlite3")
HISTORY_BATCH_SIZE = 40
MAX_CONTEXT_MESSAGES = 30
MAX_RELEVANT_OLD_MESSAGES = 10
MAX_MESSAGE_CHARS = 2500

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("kissa")

def parse_allowed_chats(raw):
    result = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            result.add(int(item))
        except ValueError:
            # usernames can be written as @username
            result.add(item.lstrip("@"))
    return result

ALLOWED_CHATS = parse_allowed_chats(ALLOWED_CHATS_RAW)

def now_local():
    return datetime.now(TIMEZONE)

def dt_to_iso(dt):
    return dt.astimezone(TIMEZONE).isoformat()

def iso_to_dt(value):
    try:
        dt = datetime.fromisoformat(value)
        return dt if dt.tzinfo else dt.replace(tzinfo=TIMEZONE)
    except (TypeError, ValueError):
        return None

def in_quiet_hours():
    hour = now_local().hour
    return hour >= 0 and hour < 7

def is_allowed(chat_id, username=None):
    if not ALLOWED_CHATS:
        return False
    return chat_id in ALLOWED_CHATS or (username and username.lower() in {str(x).lower() for x in ALLOWED_CHATS})

# ---------- SQLITE MEMORY ----------
db = sqlite3.connect(DB_PATH, check_same_thread=False)
db.row_factory = sqlite3.Row
db_lock = asyncio.Lock()

def init_db():
    db.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            chat_id INTEGER NOT NULL,
            message_id INTEGER NOT NULL,
            sent_at TEXT NOT NULL,
            sender TEXT NOT NULL,
            text TEXT NOT NULL,
            outgoing INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY(chat_id, message_id)
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS chat_state (
            chat_id INTEGER PRIMARY KEY,
            last_incoming_at TEXT,
            last_outgoing_at TEXT,
            last_followup_at TEXT,
            last_first_message_at TEXT,
            history_complete INTEGER NOT NULL DEFAULT 0
        )
    """)
    db.commit()

def save_message(chat_id, message_id, sent_at, sender, text, outgoing):
    if not text:
        return
    db.execute(
        """INSERT OR IGNORE INTO messages
           (chat_id, message_id, sent_at, sender, text, outgoing)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (int(chat_id), int(message_id), sent_at, sender, text[:MAX_MESSAGE_CHARS], int(outgoing)),
    )
    db.commit()

def update_state(chat_id, **fields):
    if not fields:
        return
    db.execute("INSERT OR IGNORE INTO chat_state(chat_id) VALUES (?)", (int(chat_id),))
    allowed = {"last_incoming_at", "last_outgoing_at", "last_followup_at", "last_first_message_at", "history_complete"}
    fields = {k: v for k, v in fields.items() if k in allowed}
    if fields:
        sql = ", ".join(f"{k}=?" for k in fields)
        db.execute(f"UPDATE chat_state SET {sql} WHERE chat_id=?", (*fields.values(), int(chat_id)))
        db.commit()

def get_state(chat_id):
    row = db.execute("SELECT * FROM chat_state WHERE chat_id=?", (int(chat_id),)).fetchone()
    return dict(row) if row else {}

def store_telegram_message(chat_id, message, outgoing=False):
    text = (getattr(message, "message", None) or "").strip()
    if not text:
        return
    date = getattr(message, "date", None) or datetime.now(TIMEZONE)
    if date.tzinfo is None:
        date = date.replace(tzinfo=ZoneInfo("UTC"))
    date = date.astimezone(TIMEZONE)
    sender = "assistant" if outgoing else "user"
    save_message(chat_id, message.id, dt_to_iso(date), sender, text, outgoing)

async def backfill_history(entity, chat_id):
    """Import accessible chat history, 40 Telegram messages per request."""
    state = get_state(chat_id)
    if state.get("history_complete"):
        return
    log.info("Начинаю импорт истории для чата %s", chat_id)
    offset_id = 0
    imported = 0
    try:
        while True:
            batch = await client.get_messages(entity, limit=HISTORY_BATCH_SIZE, offset_id=offset_id)
            if not batch:
                break
            for msg in batch:
                store_telegram_message(chat_id, msg, outgoing=bool(getattr(msg, "out", False)))
            imported += len(batch)
            oldest_id = min((m.id for m in batch if getattr(m, "id", None)), default=None)
            if not oldest_id or len(batch) < HISTORY_BATCH_SIZE:
                break
            offset_id = oldest_id
            await asyncio.sleep(0.15)
        update_state(chat_id, history_complete=1)
        log.info("Импорт истории завершён для %s; получено сообщений: %s", chat_id, imported)
    except Exception:
        log.exception("Не удалось полностью импортировать историю чата %s; повторим позже", chat_id)

def recent_context(chat_id):
    rows = db.execute(
        "SELECT sender, text, sent_at FROM messages WHERE chat_id=? ORDER BY message_id DESC LIMIT ?",
        (int(chat_id), MAX_CONTEXT_MESSAGES),
    ).fetchall()
    return list(reversed(rows))

def find_relevant_old_messages(chat_id, current_text):
    words = {w.lower().strip(".,!?;:()[]{}\"'«»") for w in current_text.split() if len(w) >= 4}
    if not words:
        return []
    rows = db.execute(
        """SELECT sender, text, sent_at FROM messages
           WHERE chat_id=? ORDER BY message_id DESC LIMIT 1500""",
        (int(chat_id),),
    ).fetchall()
    recent_ids = {row["text"] for row in recent_context(chat_id)}
    scored = []
    for row in rows:
        text = row["text"]
        if text in recent_ids:
            continue
        tokens = {w.lower().strip(".,!?;:()[]{}\"'«»") for w in text.split() if len(w) >= 4}
        score = len(words & tokens)
        if score:
            scored.append((score, row["sent_at"], row["sender"], text))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [{"sender": s, "text": t, "sent_at": d} for _, d, s, t in scored[:MAX_RELEVANT_OLD_MESSAGES]]

# ---------- AI ----------
SYSTEM_PROMPT = """Ты отвечаешь на сообщения в личном Telegram-аккаунте владельца.
Пиши естественно, кратко и спокойно, как обычный человек. Обычно 1 короткое предложение.
Не используй эмодзи, не упоминай ИИ, бота, модель или промпт. Не выдумывай факты о владельце.
Учитывай контекст и сохранённую историю. Не говори «не знаю, что ответить» — ответь просто и уместно.
Если сообщение — только оскорбление без вопроса или просьбы, ответь «ок».
Если в сообщении есть нормальный вопрос или просьба вместе с оскорблением, ответь по существу.
На романтическое признание или приглашение на свидание откажи вежливо и однозначно, без флирта и ложных обещаний.
Не отправляй длинные объяснения, если их не просили."""

async def generate_reply(chat_id, incoming_text):
    recent = recent_context(chat_id)
    old = find_relevant_old_messages(chat_id, incoming_text)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if old:
        old_text = "\n".join(f'{m["sender"]}: {m["text"]}' for m in old)
        messages.append({"role": "system", "content": "Подходящие фрагменты старой истории (используй только если уместно):\n" + old_text})
    for row in recent:
        role = "assistant" if row["sender"] == "assistant" else "user"
        messages.append({"role": role, "content": row["text"]})
    if not recent or recent[-1]["sender"] != "user" or recent[-1]["text"] != incoming_text:
        messages.append({"role": "user", "content": incoming_text})
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/KissaPlugins/kissa-autoreply",
        "X-Title": "Kissa AI AutoReply",
    }
    payload = {"model": MODEL, "messages": messages, "temperature": 0.7, "max_tokens": 120}
    async with httpx.AsyncClient(timeout=45) as http:
        response = await http.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
    answer = data["choices"][0]["message"]["content"].strip()
    return answer[:MAX_MESSAGE_CHARS] if answer else "понял"

# ---------- TELEGRAM ----------
client = TelegramClient(StringSession(TG_SESSION), API_ID, API_HASH)
chat_locks = {}

async def handle_incoming(event):
    chat_id = event.chat_id
    if not chat_id or not event.is_private:
        return
    sender = await event.get_sender()
    username = getattr(sender, "username", None)
    if not is_allowed(chat_id, username):
        return
    text = (event.raw_text or "").strip()
    if not text:
        return

    store_telegram_message(chat_id, event.message, outgoing=False)
    update_state(chat_id, last_incoming_at=dt_to_iso(now_local()), last_followup_at=None)
    # Import older messages in the background so the first reply is not delayed.
    if not get_state(chat_id).get("history_complete"):
        asyncio.create_task(backfill_history(event.input_chat, chat_id))

    lock = chat_locks.setdefault(chat_id, asyncio.Lock())
    async with lock:
        try:
            answer = await generate_reply(chat_id, text)
            sent = await event.respond(answer)
            # Save the exact outgoing response locally.
            if sent:
                store_telegram_message(chat_id, sent, outgoing=True)
            else:
                save_message(chat_id, int(datetime.now().timestamp() * 1000), dt_to_iso(now_local()), "assistant", answer, 1)
            update_state(chat_id, last_outgoing_at=dt_to_iso(now_local()))
            log.info("Ответ отправлен в чат %s", chat_id)
        except httpx.HTTPStatusError as exc:
            log.error("OpenRouter вернул HTTP %s: %s", exc.response.status_code, exc.response.text[:500])
        except Exception:
            log.exception("Ошибка при обработке сообщения в чате %s", chat_id)

@client.on(events.NewMessage(incoming=True))
async def on_new_message(event):
    await handle_incoming(event)

@client.on(events.NewMessage(outgoing=True))
async def on_outgoing_message(event):
    if not event.is_private or not event.chat_id:
        return
    text = (event.raw_text or "").strip()
    if text:
        store_telegram_message(event.chat_id, event.message, outgoing=True)
        update_state(event.chat_id, last_outgoing_at=dt_to_iso(now_local()))

async def followup_loop():
    while True:
        try:
            if FOLLOWUP_ENABLED and not in_quiet_hours():
                rows = db.execute("SELECT chat_id, last_incoming_at, last_outgoing_at, last_followup_at FROM chat_state").fetchall()
                for row in rows:
                    chat_id = row["chat_id"]
                    if not is_allowed(chat_id):
                        continue
                    incoming_dt = iso_to_dt(row["last_incoming_at"])
                    outgoing_dt = iso_to_dt(row["last_outgoing_at"])
                    followup_dt = iso_to_dt(row["last_followup_at"])
                    if not incoming_dt or not outgoing_dt or outgoing_dt < incoming_dt:
                        continue
                    now = now_local()
                    if now - outgoing_dt < timedelta(hours=FOLLOWUP_HOURS):
                        continue
                    if followup_dt and now - followup_dt < timedelta(hours=FOLLOWUP_COOLDOWN_HOURS):
                        continue
                    try:
                        entity = await client.get_input_entity(chat_id)
                        prompt = "Продолжи переписку одним коротким естественным сообщением, если это уместно. Не упоминай, что прошло время. Не повторяй прошлое сообщение. Если писать неуместно, верни ровно [SKIP]."
                        answer = await generate_reply(chat_id, prompt)
                        if answer.strip() == "[SKIP]" or in_quiet_hours():
                            continue
                        await client.send_message(entity, answer)
                        save_message(chat_id, int(datetime.now().timestamp() * 1000), dt_to_iso(now_local()), "assistant", answer, 1)
                        update_state(chat_id, last_outgoing_at=dt_to_iso(now_local()), last_followup_at=dt_to_iso(now_local()))
                        await asyncio.sleep(2)
                    except Exception:
                        log.exception("Не удалось отправить follow-up в чат %s", chat_id)
            await asyncio.sleep(60)
        except Exception:
            log.exception("Ошибка в цикле follow-up")
            await asyncio.sleep(60)

async def first_message_loop():
    """Optional scheduled first message. Configure FIRST_MESSAGE_TEXT; sends at most once per contact per 24h."""
    while True:
        try:
            if FIRST_MESSAGE_ENABLED and FIRST_MESSAGE_TEXT and not in_quiet_hours():
                for target in ALLOWED_CHATS:
                    if not isinstance(target, int):
                        try:
                            entity = await client.get_input_entity(target)
                            chat_id = getattr(entity, "user_id", None)
                            if not chat_id:
                                continue
                        except Exception:
                            log.exception("Не удалось найти контакт %s", target)
                            continue
                    else:
                        chat_id = target
                    state = get_state(chat_id)
                    last = iso_to_dt(state.get("last_first_message_at"))
                    if last and now_local() - last < timedelta(hours=24):
                        continue
                    try:
                        entity = await client.get_input_entity(target)
                        await client.send_message(entity, FIRST_MESSAGE_TEXT)
                        save_message(chat_id, int(datetime.now().timestamp() * 1000), dt_to_iso(now_local()), "assistant", FIRST_MESSAGE_TEXT, 1)
                        update_state(chat_id, last_first_message_at=dt_to_iso(now_local()), last_outgoing_at=dt_to_iso(now_local()))
                        await asyncio.sleep(2)
                    except Exception:
                        log.exception("Не удалось отправить первое сообщение контакту %s", target)
            await asyncio.sleep(300)
        except Exception:
            log.exception("Ошибка в цикле первого сообщения")
            await asyncio.sleep(60)

async def main():
    init_db()
    await client.start()
    me = await client.get_me()
    log.info("Kissa AutoReply запущен для аккаунта %s", getattr(me, "username", None) or me.id)
    tasks = [asyncio.create_task(followup_loop()), asyncio.create_task(first_message_loop())]
    try:
        await client.run_until_disconnected()
    finally:
        for task in tasks:
            task.cancel()
        db.close()

if __name__ == "__main__":
    asyncio.run(main())
