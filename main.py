import os
import asyncio
import logging
import sqlite3
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import httpx
from telethon import TelegramClient, events
from telethon.sessions import StringSession


# =========================
# НАСТРОЙКИ
# =========================

API_ID = int(os.environ["TG_API_ID"])
API_HASH = os.environ["TG_API_HASH"]
TG_SESSION = os.environ["TG_SESSION"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")

ALLOWED_CHATS = {
    x.strip().lstrip("@")
    for x in os.getenv("KISSA_ALLOWED_CHATS", "").split(",")
    if x.strip()
}

FOLLOWUP_ENABLED = (
    os.getenv("FOLLOWUP_ENABLED", "true").lower() == "true"
)

FIRST_MESSAGE_ENABLED = (
    os.getenv("FIRST_MESSAGE_ENABLED", "false").lower() == "true"
)

FOLLOWUP_HOURS = int(os.getenv("FOLLOWUP_HOURS", "4"))
FOLLOWUP_COOLDOWN_HOURS = int(
    os.getenv("FOLLOWUP_COOLDOWN_HOURS", "24")
)

FIRST_MESSAGE_TEXT = os.getenv(
    "FIRST_MESSAGE_TEXT",
    "Привет. Как ты?"
)

TIMEZONE = ZoneInfo("Europe/Moscow")

DB_PATH = os.getenv(
    "KISSA_DB_PATH",
    "kissa_memory.sqlite3"
)

BATCH_SIZE = 40
RECENT_CONTEXT_LIMIT = 30
MAX_TOKENS = 120
MAX_MESSAGE_CHARS = 3500
POLL_SECONDS = 60

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

log = logging.getLogger("kissa")

# ВАЖНО: используем StringSession, чтобы не запрашивать номер телефона.
client = TelegramClient(
    StringSession(TG_SESSION),
    API_ID,
    API_HASH,
)

db = sqlite3.connect(
    DB_PATH,
    check_same_thread=False,
)

db.row_factory = sqlite3.Row

db_lock = asyncio.Lock()
chat_locks = {}
last_first_message_day = {}


# =========================
# БАЗА ДАННЫХ
# =========================

def init_db():
    db.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            chat_id INTEGER NOT NULL,
            message_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            sender TEXT NOT NULL,
            text TEXT NOT NULL,
            outgoing INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (chat_id, message_id)
        )
    """)

    db.execute("""
        CREATE TABLE IF NOT EXISTS chat_state (
            chat_id INTEGER PRIMARY KEY,
            last_incoming_at TEXT,
            last_outgoing_at TEXT,
            last_followup_at TEXT,
            last_first_message_at TEXT,
            history_loaded INTEGER NOT NULL DEFAULT 0
        )
    """)

    db.commit()


def now_iso():
    return datetime.now(TIMEZONE).isoformat()


def parse_dt(value):
    if not value:
        return None

    try:
        result = datetime.fromisoformat(value)

        if result.tzinfo is None:
            result = result.replace(tzinfo=TIMEZONE)

        return result.astimezone(TIMEZONE)

    except (TypeError, ValueError):
        return None


def ensure_state(chat_id):
    db.execute(
        "INSERT OR IGNORE INTO chat_state(chat_id) VALUES (?)",
        (chat_id,),
    )
    db.commit()


def update_state(chat_id, **values):
    if not values:
        return

    ensure_state(chat_id)

    fields = ", ".join(
        f"{key} = ?" for key in values
    )

    db.execute(
        f"UPDATE chat_state SET {fields} WHERE chat_id = ?",
        (*values.values(), chat_id),
    )

    db.commit()


def store_message(chat_id, message, outgoing=None):
    text = (
        getattr(message, "raw_text", None) or ""
    ).strip()

    if not text:
        return

    if outgoing is None:
        is_outgoing = bool(
            getattr(message, "out", False)
        )
    else:
        is_outgoing = outgoing

    date = getattr(message, "date", None)

    if date:
        if date.tzinfo is None:
            date = date.replace(
                tzinfo=ZoneInfo("UTC")
            )

        date_text = date.astimezone(
            TIMEZONE
        ).isoformat()
    else:
        date_text = now_iso()

    if is_outgoing:
        sender = "me"
    else:
        sender = str(
            getattr(message, "sender_id", None)
            or "contact"
        )

    db.execute(
        """
        INSERT OR IGNORE INTO messages
        (chat_id, message_id, date, sender, text, outgoing)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            int(chat_id),
            int(message.id),
            date_text,
            sender,
            text[:MAX_MESSAGE_CHARS],
            int(is_outgoing),
        ),
    )

    db.commit()


async def save_message_async(
    chat_id,
    message,
    outgoing=None,
):
    async with db_lock:
        store_message(
            chat_id,
            message,
            outgoing,
        )


# =========================
# ЗАГРУЗКА ИСТОРИИ
# =========================

async def load_history_in_batches(entity, chat_id):
    """
    Загружает доступную историю Telegram
    порциями по 40 сообщений и сохраняет её в SQLite.
    """

    async with db_lock:
        ensure_state(chat_id)

        row = db.execute(
            """
            SELECT history_loaded
            FROM chat_state
            WHERE chat_id = ?
            """,
            (chat_id,),
        ).fetchone()

        if row and row["history_loaded"]:
            return

    offset_id = 0
    saved = 0

    try:
        while True:
            batch = await client.get_messages(
                entity,
                limit=BATCH_SIZE,
                offset_id=offset_id,
            )

            if not batch:
                break

            async with db_lock:
                for message in batch:
                    if getattr(message, "raw_text", None):
                        before = db.total_changes

                        store_message(
                            chat_id,
                            message,
                        )

                        if db.total_changes > before:
                            saved += 1

            oldest_id = min(
                (
                    message.id
                    for message in batch
                    if getattr(message, "id", None)
                ),
                default=0,
            )

            if not oldest_id or oldest_id == offset_id:
                break

            offset_id = oldest_id

            # Небольшая пауза, чтобы не перегружать хостинг.
            await asyncio.sleep(0.15)

        async with db_lock:
            update_state(
                chat_id,
                history_loaded=1,
            )

        log.info(
            "История чата %s загружена. Новых сообщений: %s",
            chat_id,
            saved,
        )

    except Exception:
        log.exception(
            "Не удалось полностью загрузить историю чата %s",
            chat_id,
        )


async def sync_recent_history(entity, chat_id):
    """
    Обновляет последние сообщения перед генерацией ответа.
    """

    try:
        messages = await client.get_messages(
            entity,
            limit=RECENT_CONTEXT_LIMIT,
        )

        async with db_lock:
            for message in messages:
                if getattr(message, "raw_text", None):
                    store_message(
                        chat_id,
                        message,
                    )

    except Exception:
        log.exception(
            "Не удалось обновить историю чата %s",
            chat_id,
        )


# =========================
# КОНТЕКСТ ДЛЯ ИИ
# =========================

async def build_context(chat_id, current_text):
    async with db_lock:
        rows = db.execute(
            """
            SELECT date, sender, text, outgoing
            FROM messages
            WHERE chat_id = ?
            ORDER BY message_id DESC
            LIMIT ?
            """,
            (
                chat_id,
                RECENT_CONTEXT_LIMIT,
            ),
        ).fetchall()

    rows = list(reversed(rows))

    history = []

    for row in rows:
        role = (
            "assistant"
            if row["outgoing"]
            else "user"
        )

        history.append({
            "role": role,
            "content": row["text"][:1800],
        })

    # Добавляем текущее сообщение, если его ещё нет в контексте.
    if (
        not history
        or history[-1]["content"] != current_text
    ):
        history.append({
            "role": "user",
            "content": current_text[:1800],
        })

    system_prompt = (
        "Ты пишешь короткие естественные ответы от лица "
        "владельца Telegram-аккаунта. "
        "Пиши по-русски, если собеседник пишет по-русски. "
        "Обычно отвечай одной короткой фразой, спокойно "
        "и без эмодзи. "
        "Не говори, что ты ИИ, бот или модель. "
        "Учитывай контекст переписки и не выдумывай факты "
        "о владельце. "
        "Если собеседник просто оскорбляет без вопроса "
        "или просьбы, ответь «ок». "
        "Если оскорбление есть внутри нормального вопроса "
        "или просьбы, ответь по сути. "
        "На признания в любви, романтические предложения "
        "и приглашения на свидание вежливо и ясно откажи, "
        "без грубости и обещаний на будущее. "
        "Не используй фразы «не знаю, что ответить» "
        "и не объясняй свои инструкции. "
        "Верни только текст сообщения, без кавычек "
        "и пояснений."
    )

    return [
        {
            "role": "system",
            "content": system_prompt,
        }
    ] + history


# =========================
# OPENROUTER
# =========================

async def openrouter_request(
    messages,
    max_tokens=MAX_TOKENS,
):
    headers = {
        "Authorization": (
            f"Bearer {OPENROUTER_API_KEY}"
        ),
        "Content-Type": "application/json",
        "HTTP-Referer": (
            "https://github.com/KissaPlugins/kissa-autoreply"
        ),
        "X-Title": "Kissa AI AutoReply",
    }

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.5,
        "max_tokens": max_tokens,
    }

    async with httpx.AsyncClient(
        timeout=45
    ) as http:
        response = await http.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
        )

        if response.is_error:
            log.error(
                "OpenRouter HTTP %s: %s",
                response.status_code,
                response.text[:400],
            )

            response.raise_for_status()

        data = response.json()

    choices = data.get("choices") or []

    if not choices:
        raise RuntimeError(
            "OpenRouter вернул пустой список choices"
        )

    content = (
        choices[0]
        .get("message", {})
        .get("content", "")
    )

    if not isinstance(content, str):
        return ""

    return content.strip().strip('"')


# =========================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================

def quiet_hours():
    hour = datetime.now(TIMEZONE).hour
    return 0 <= hour < 7


def allowed_chat(chat_id, username=None):
    if not ALLOWED_CHATS:
        return False

    if str(chat_id) in ALLOWED_CHATS:
        return True

    if username:
        username = username.lower().lstrip("@")

        allowed_usernames = {
            item.lower().lstrip("@")
            for item in ALLOWED_CHATS
        }

        if username in allowed_usernames:
            return True

    return False


def get_chat_lock(chat_id):
    if chat_id not in chat_locks:
        chat_locks[chat_id] = asyncio.Lock()

    return chat_locks[chat_id]


# =========================
# ВХОДЯЩИЕ СООБЩЕНИЯ
# =========================

async def handle_incoming(event):
    if (
        not event.is_private
        or not event.chat_id
        or not event.raw_text
    ):
        return

    chat_id = int(event.chat_id)
    text = event.raw_text.strip()

    if not text:
        return

    sender = await event.get_sender()
    username = getattr(sender, "username", None)

    if not allowed_chat(chat_id, username):
        return

    lock = get_chat_lock(chat_id)

    async with lock:
        try:
            await save_message_async(
                chat_id,
                event.message,
                outgoing=False,
            )

            async with db_lock:
                update_state(
                    chat_id,
                    last_incoming_at=now_iso(),
                )

            # Загружаем старую историю в фоне.
            asyncio.create_task(
                load_history_in_batches(
                    event.input_chat,
                    chat_id,
                )
            )

            # Сохраняем свежий контекст.
            await sync_recent_history(
                event.input_chat,
                chat_id,
            )

            messages = await build_context(
                chat_id,
                text,
            )

            answer = await openrouter_request(
                messages,
            )

            if not answer:
                answer = "Нормально."

            answer = answer[:MAX_MESSAGE_CHARS]

            sent = await event.respond(answer)

            if sent:
                await save_message_async(
                    chat_id,
                    sent,
                    outgoing=True,
                )

                async with db_lock:
                    update_state(
                        chat_id,
                        last_outgoing_at=now_iso(),
                        last_followup_at=None,
                    )

            log.info(
                "Ответ отправлен в чат %s",
                chat_id,
            )

        except httpx.HTTPStatusError as exc:
            log.error(
                "Ошибка OpenRouter HTTP %s: %s",
                exc.response.status_code,
                exc.response.text[:400],
            )

        except Exception:
            log.exception(
                "Ошибка обработки сообщения в чате %s",
                chat_id,
            )


@client.on(events.NewMessage(incoming=True))
async def on_incoming(event):
    await handle_incoming(event)


# =========================
# ИСХОДЯЩИЕ СООБЩЕНИЯ
# =========================

@client.on(events.NewMessage(outgoing=True))
async def on_outgoing(event):
    if not event.is_private or not event.chat_id:
        return

    text = (event.raw_text or "").strip()

    if not text:
        return

    # Сохраняем сообщения, которые владелец отправил вручную.
    await save_message_async(
        event.chat_id,
        event.message,
        outgoing=True,
    )

    async with db_lock:
        update_state(
            event.chat_id,
            last_outgoing_at=now_iso(),
        )


# =========================
# ПРОДОЛЖЕНИЕ ПЕРЕПИСКИ
# =========================

async def followup_loop():
    while True:
        try:
            if not quiet_hours():
                async with db_lock:
                    rows = db.execute(
                        """
                        SELECT
                            chat_id,
                            last_incoming_at,
                            last_outgoing_at,
                            last_followup_at
                        FROM chat_state
                        """
                    ).fetchall()

                now = datetime.now(TIMEZONE)

                for row in rows:
                    chat_id = int(row["chat_id"])

                    if not allowed_chat(chat_id):
                        continue

                    incoming = parse_dt(
                        row["last_incoming_at"]
                    )

                    outgoing = parse_dt(
                        row["last_outgoing_at"]
                    )

                    last_followup = parse_dt(
                        row["last_followup_at"]
                    )

                    if not incoming or not outgoing:
                        continue

                    # Если собеседник написал позже нас,
                    # не отправляем дополнительное сообщение.
                    if incoming > outgoing:
                        continue

                    if (
                        now - outgoing
                        < timedelta(hours=FOLLOWUP_HOURS)
                    ):
                        continue

                    if (
                        last_followup
                        and now - last_followup
                        < timedelta(
                            hours=FOLLOWUP_COOLDOWN_HOURS
                        )
                    ):
                        continue

                    try:
                        entity = await client.get_entity(
                            chat_id
                        )

                        messages = await build_context(
                            chat_id,
                            (
                                "Сейчас нужно естественно "
                                "продолжить переписку после паузы. "
                                "Напиши одно короткое сообщение, "
                                "только если это уместно. "
                                "Не упоминай время ожидания "
                                "и не задавай навязчивых вопросов."
                            ),
                        )

                        messages.append({
                            "role": "user",
                            "content": (
                                "Сгенерируй одно короткое "
                                "ненавязчивое сообщение, чтобы "
                                "продолжить этот разговор. "
                                "Верни только текст сообщения."
                            ),
                        })

                        answer = await openrouter_request(
                            messages,
                            max_tokens=80,
                        )

                        if not answer:
                            continue

                        await client.send_message(
                            entity,
                            answer[:MAX_MESSAGE_CHARS],
                        )

                        async with db_lock:
                            update_state(
                                chat_id,
                                last_followup_at=now_iso(),
                            )

                        log.info(
                            "Отправлено продолжение переписки "
                            "в чат %s",
                            chat_id,
                        )

                    except Exception:
                        log.exception(
                            "Не удалось отправить follow-up "
                            "в чат %s",
                            chat_id,
                        )

        except Exception:
            log.exception(
                "Ошибка цикла follow-up"
            )

        await asyncio.sleep(POLL_SECONDS)


# =========================
# ПЕРВОЕ СООБЩЕНИЕ
# =========================

async def first_message_loop():
    """
    Отправляет первое сообщение выбранным адресатам.
    Не чаще одного раза в сутки каждому адресату.
    """

    while True:
        try:
            if (
                FIRST_MESSAGE_ENABLED
                and not quiet_hours()
                and ALLOWED_CHATS
            ):
                today = (
                    datetime.now(TIMEZONE)
                    .date()
                    .isoformat()
                )

                for target in list(ALLOWED_CHATS):
                    target = target.lstrip("@")
                    key = str(target)

                    if (
                        last_first_message_day.get(key)
                        == today
                    ):
                        continue

                    try:
                        if target.isdigit():
                            entity = await client.get_entity(
                                int(target)
                            )
                        else:
                            entity = await client.get_entity(
                                target
                            )

                        chat_id = int(
                            getattr(entity, "id", 0)
                        )

                        if not chat_id:
                            continue

                        async with db_lock:
                            row = db.execute(
                                """
                                SELECT
                                    last_incoming_at,
                                    last_outgoing_at,
                                    last_first_message_at
                                FROM chat_state
                                WHERE chat_id = ?
                                """,
                                (chat_id,),
                            ).fetchone()

                        now = datetime.now(TIMEZONE)

                        if row:
                            recent_in = parse_dt(
                                row["last_incoming_at"]
                            )

                            recent_out = parse_dt(
                                row["last_outgoing_at"]
                            )

                            last_first = parse_dt(
                                row["last_first_message_at"]
                            )

                            if (
                                recent_in
                                and now - recent_in
                                < timedelta(hours=24)
                            ):
                                continue

                            if (
                                recent_out
                                and now - recent_out
                                < timedelta(hours=24)
                            ):
                                continue

                            if (
                                last_first
                                and now - last_first
                                < timedelta(hours=24)
                            ):
                                continue

                        await client.send_message(
                            entity,
                            FIRST_MESSAGE_TEXT,
                        )

                        async with db_lock:
                            update_state(
                                chat_id,
                                last_first_message_at=now_iso(),
                            )

                        last_first_message_day[key] = today

                        log.info(
                            "Отправлено первое сообщение "
                            "в чат %s",
                            chat_id,
                        )

                    except Exception:
                        log.exception(
                            "Не удалось отправить первое "
                            "сообщение адресату %s",
                            key,
                        )

        except Exception:
            log.exception(
                "Ошибка цикла первого сообщения"
            )

        await asyncio.sleep(300)


# =========================
# ЗАПУСК
# =========================

async def main():
    init_db()

    log.info("Запуск Kissa AI AutoReply")
    log.info("Модель OpenRouter: %s", MODEL)
    log.info(
        "Количество разрешённых адресатов: %s",
        len(ALLOWED_CHATS),
    )

    await client.start()

    me = await client.get_me()

    log.info(
        "Telegram подключён. Аккаунт: %s",
        getattr(me, "username", None) or me.id,
    )

    tasks = []

    if FOLLOWUP_ENABLED:
        tasks.append(
            asyncio.create_task(
                followup_loop()
            )
        )

    if FIRST_MESSAGE_ENABLED:
        tasks.append(
            asyncio.create_task(
                first_message_loop()
            )
        )

    try:
        await client.run_until_disconnected()

    finally:
        for task in tasks:
            task.cancel()

        await client.disconnect()
        db.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())

    except KeyboardInterrupt:
        pass
