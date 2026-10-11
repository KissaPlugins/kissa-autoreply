
import os
import asyncio
import logging
import sqlite3
import base64
import tempfile
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import cv2
import httpx
from telethon import TelegramClient, events
from telethon.sessions import StringSession
from telethon.tl.types import DocumentAttributeSticker, DocumentAttributeAnimated


# =========================
# НАСТРОЙКИ
# =========================

API_ID = int(os.environ["TG_API_ID"])
API_HASH = os.environ["TG_API_HASH"]
SESSION = os.environ["TG_SESSION"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "openrouter/free"
)

ALLOWED_CHATS = {
    item.strip().lower().lstrip("@")
    for item in os.getenv("KISSA_ALLOWED_CHATS", "").split(",")
    if item.strip()
}

FOLLOWUP_ENABLED = os.getenv("FOLLOWUP_ENABLED", "true").lower() == "true"
FIRST_MESSAGE_ENABLED = os.getenv("FIRST_MESSAGE_ENABLED", "false").lower() == "true"

FOLLOWUP_HOURS = 4
FOLLOWUP_COOLDOWN_HOURS = 24
FIRST_MESSAGE_TEXT = os.getenv("FIRST_MESSAGE_TEXT", "Привет. Как ты?")

TIMEZONE = ZoneInfo("Europe/Moscow")
DB_PATH = "kissa_memory.sqlite3"

RECENT_CONTEXT_LIMIT = 30
MAX_TOKENS = 300
MAX_MESSAGE_CHARS = 3500
MAX_MEDIA_BYTES = 15 * 1024 * 1024
POLL_SECONDS = 60
VIDEO_FRAMES = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("kissa")


# =========================
# БАЗА ДАННЫХ И ПАМЯТЬ
# =========================

def db_connect():
    connection = sqlite3.connect(DB_PATH)
    connection.execute("""
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
    connection.execute("""
        CREATE TABLE IF NOT EXISTS chat_state (
            chat_id INTEGER PRIMARY KEY,
            last_incoming TEXT,
            last_outgoing TEXT,
            last_followup TEXT,
            last_first_message TEXT,
            history_loaded INTEGER DEFAULT 0
        )
    """)
    connection.commit()
    return connection


def ensure_state(chat_id):
    with db_connect() as db:
        db.execute(
            "INSERT OR IGNORE INTO chat_state (chat_id) VALUES (?)",
            (chat_id,)
        )


def save_message(chat_id, message_id, date, sender, text, outgoing):
    if not text:
        return

    with db_connect() as db:
        db.execute("""
            INSERT OR IGNORE INTO messages
            (chat_id, message_id, date, sender, text, outgoing)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            chat_id,
            message_id,
            date.isoformat(),
            sender,
            text[:MAX_MESSAGE_CHARS],
            int(outgoing)
        ))

        if outgoing:
            db.execute("""
                UPDATE chat_state
                SET last_outgoing = ?
                WHERE chat_id = ?
            """, (date.isoformat(), chat_id))
        else:
            db.execute("""
                UPDATE chat_state
                SET last_incoming = ?
                WHERE chat_id = ?
            """, (date.isoformat(), chat_id))


def get_state(chat_id):
    ensure_state(chat_id)
    with db_connect() as db:
        row = db.execute("""
            SELECT last_incoming, last_outgoing, last_followup,
                   last_first_message, history_loaded
            FROM chat_state WHERE chat_id = ?
        """, (chat_id,)).fetchone()

    keys = (
        "last_incoming",
        "last_outgoing",
        "last_followup",
        "last_first_message",
        "history_loaded"
    )
    return dict(zip(keys, row))


async def save_message_async(*args):
    await asyncio.to_thread(save_message, *args)


async def load_history_in_batches(client, entity, chat_id):
    """Сохраняет доступную историю чата небольшими порциями."""
    state = get_state(chat_id)
    if state["history_loaded"]:
        return

    offset_id = 0
    try:
        while True:
            messages = await client.get_messages(
                entity,
                limit=40,
                offset_id=offset_id
            )
            if not messages:
                break

            for message in messages:
                if not message.id or not message.date:
                    continue

                text = (message.raw_text or "").strip()
                if not text:
                    if getattr(message, "photo", None):
                        text = "[Фото]"
                    elif getattr(message, "video", None):
                        text = "[Видео]"
                    else:
                        continue

                sender = "me" if message.out else "contact"
                await save_message_async(
                    chat_id,
                    message.id,
                    message.date,
                    sender,
                    text,
                    message.out
                )

            oldest_id = min(m.id for m in messages if m.id)
            if oldest_id == offset_id:
                break
            offset_id = oldest_id

        with db_connect() as db:
            db.execute(
                "UPDATE chat_state SET history_loaded = 1 WHERE chat_id = ?",
                (chat_id,)
            )
        log.info("История загружена для чата %s", chat_id)

    except Exception:
        log.exception("Не удалось загрузить историю чата %s", chat_id)


async def sync_recent_history(client, entity, chat_id):
    try:
        messages = await client.get_messages(
            entity,
            limit=RECENT_CONTEXT_LIMIT
        )
        for message in reversed(messages):
            if not message.id or not message.date:
                continue

            text = (message.raw_text or "").strip()
            if not text:
                if getattr(message, "photo", None):
                    text = "[Фото]"
                elif getattr(message, "video", None):
                    text = "[Видео]"
                else:
                    continue

            await save_message_async(
                chat_id,
                message.id,
                message.date,
                "me" if message.out else "contact",
                text,
                message.out
            )
    except Exception:
        log.exception("Ошибка синхронизации истории")


def build_context(chat_id):
    with db_connect() as db:
        rows = db.execute("""
            SELECT sender, text FROM (
                SELECT message_id, sender, text
                FROM messages
                WHERE chat_id = ?
                ORDER BY message_id DESC
                LIMIT ?
            )
            ORDER BY message_id ASC
        """, (chat_id, RECENT_CONTEXT_LIMIT)).fetchall()

    messages = [{
        "role": "system",
        "content": (
            "Ты пишешь коротко, естественно и по-человечески. "
            "Отвечай на языке собеседника. Учитывай историю диалога. "
            "Не выдумывай факты и не обещай того, чего не можешь сделать. "
            "Если сообщение состоит только из оскорбления и не содержит "
            "вопроса или просьбы, ответь кратко: «ок». "
            "Если есть нормальный вопрос или просьба, ответь по существу, "
            "даже если сообщение грубое. На романтические признания и "
            "приглашения отвечай вежливо, без взаимных обещаний. "
            "Не утверждай, что ты ИИ, если об этом не спрашивают."
        )
    }]

    for sender, text in rows:
        messages.append({
            "role": "assistant" if sender == "me" else "user",
            "content": text
        })

    return messages


# =========================
# OPENROUTER
# =========================

async def openrouter_request(messages):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/KissaPlugins/kissa-autoreply",
        "X-Title": "Kissa AI AutoReply"
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": 0.5,
        "max_tokens": MAX_TOKENS
    }

    async with httpx.AsyncClient(timeout=90) as http:
        response = await http.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        data = response.json()

    return data["choices"][0]["message"]["content"].strip()


# =========================
# МЕДИА
# =========================

def is_sticker_or_gif(message):
    """Стикеры и анимированные GIF не обрабатываем."""
    if getattr(message, "sticker", None):
        return True
    if getattr(message, "gif", None):
        return True

    document = getattr(message, "document", None)
    if document:
        for attribute in document.attributes:
            if isinstance(
                attribute,
                (DocumentAttributeSticker, DocumentAttributeAnimated)
            ):
                return True

    mime = getattr(getattr(message, "file", None), "mime_type", "") or ""
    name = getattr(getattr(message, "file", None), "name", "") or ""

    if mime == "image/gif" or name.lower().endswith(".gif"):
        return True

    return False


async def download_media_bytes(message):
    try:
        size = getattr(getattr(message, "file", None), "size", None)
        if size and size > MAX_MEDIA_BYTES:
            log.warning("Медиа слишком большое: %s байт", size)
            return None

        data = await message.download_media(file=bytes)
        if not isinstance(data, bytes):
            return None
        if len(data) > MAX_MEDIA_BYTES:
            log.warning("Загруженное медиа превышает лимит")
            return None
        return data

    except Exception:
        log.exception("Не удалось скачать медиа")
        return None


def image_content(data, mime="image/jpeg"):
    encoded = base64.b64encode(data).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime};base64,{encoded}"
        }
    }


def extract_video_frames(video_bytes, max_frames=VIDEO_FRAMES):
    """Извлекает несколько кадров, не отправляя модели весь видеоролик."""
    paths = []
    frames = []

    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4") as temp:
            temp.write(video_bytes)
            temp.flush()

            capture = cv2.VideoCapture(temp.name)
            if not capture.isOpened():
                capture.release()
                return []

            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                capture.release()
                return []

            count = min(max_frames, total)
            positions = sorted(set(
                int(i * (total - 1) / max(1, count - 1))
                for i in range(count)
            ))

            for position in positions:
                capture.set(cv2.CAP_PROP_POS_FRAMES, position)
                ok, frame = capture.read()
                if not ok:
                    continue

                height, width = frame.shape[:2]
                longest = max(height, width)
                if longest > 960:
                    scale = 960 / longest
                    frame = cv2.resize(
                        frame,
                        (int(width * scale), int(height * scale))
                    )

                ok, encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                )
                if ok:
                    frames.append(encoded.tobytes())

            capture.release()

    except Exception:
        log.exception("Не удалось извлечь кадры из видео")

    return frames


def is_video(message):
    if getattr(message, "video", None):
        return True

    file = getattr(message, "file", None)
    mime = getattr(file, "mime_type", "") if file else ""
    return bool(mime and mime.startswith("video/"))


def is_photo(message):
    return bool(getattr(message, "photo", None))


# =========================
# ДОПУСК К ЧАТУ И ВРЕМЯ
# =========================

def allowed_chat(sender):
    if not ALLOWED_CHATS:
        return False

    sender_id = str(getattr(sender, "id", "")).lower()
    username = (getattr(sender, "username", "") or "").lower().lstrip("@")

    return sender_id in ALLOWED_CHATS or (
        username and username in ALLOWED_CHATS
    )


def quiet_hours():
    hour = datetime.now(TIMEZONE).hour
    return hour >= 0 and hour < 7


def now_iso():
    return datetime.now(TIMEZONE).isoformat()


def parse_date(value):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=TIMEZONE)
        return parsed
    except ValueError:
        return None


# =========================
# TELEGRAM
# =========================

client = TelegramClient(
    StringSession(SESSION),
    API_ID,
    API_HASH
)


@client.on(events.NewMessage(incoming=True))
async def incoming_handler(event):
    if not event.is_private or not event.chat_id:
        return

    message = event.message
    if not message:
        return

    if is_sticker_or_gif(message):
        log.info("Стикер/GIF проигнорирован")
        return

    raw_text = (event.raw_text or "").strip()
    photo = is_photo(message)
    video = is_video(message)

    if not raw_text and not photo and not video:
        return

    if quiet_hours():
        log.info("Входящее сообщение пропущено из-за ночного режима")
        return

    try:
        sender = await event.get_sender()
        if not allowed_chat(sender):
            return

        chat_id = event.chat_id
        entity = await event.get_chat()

        if photo:
            stored_text = raw_text or "[Фото]"
        elif video:
            stored_text = raw_text or "[Видео]"
        else:
            stored_text = raw_text

        await save_message_async(
            chat_id,
            message.id,
            message.date or datetime.now(TIMEZONE),
            getattr(sender, "username", None) or str(sender.id),
            stored_text,
            False
        )

        asyncio.create_task(
            load_history_in_batches(client, entity, chat_id)
        )
        await sync_recent_history(client, entity, chat_id)

        context = build_context(chat_id)

        if photo or video:
            media = await download_media_bytes(message)
            if not media:
                return

            content = []
            if raw_text:
                content.append({
                    "type": "text",
                    "text": raw_text
                })

            if photo:
                content.append(image_content(media, "image/jpeg"))

            elif video:
                frames = await asyncio.to_thread(
                    extract_video_frames,
                    media
                )
                if not frames:
                    log.warning("В видео не удалось получить кадры")
                    return

                content.append({
                    "type": "text",
                    "text": (
                        "Это несколько кадров из одного видео, "
                        "показанных по порядку. Опиши только то, что "
                        "можно определить по кадрам. Не придумывай "
                        "звук или события между кадрами."
                    )
                })
                content.extend(image_content(frame) for frame in frames)

            context.append({
                "role": "user",
                "content": content
            })

        else:
            context.append({
                "role": "user",
                "content": raw_text
            })

        answer = await openrouter_request(context)
        if not answer:
            return

        answer = answer[:MAX_MESSAGE_CHARS]
        await event.respond(answer)

        await save_message_async(
            chat_id,
            message.id + 1_000_000_000,
            datetime.now(TIMEZONE),
            "me",
            answer,
            True
        )

    except httpx.HTTPStatusError as exc:
        log.error(
            "OpenRouter HTTP %s: %s",
            exc.response.status_code,
            exc.response.text[:1000]
        )
    except Exception:
        log.exception("Ошибка обработки входящего сообщения")


@client.on(events.NewMessage(outgoing=True))
async def outgoing_handler(event):
    if not event.is_private or not event.chat_id:
        return
    if not event.message or not event.message.id:
        return

    text = (event.raw_text or "").strip()
    if not text:
        return

    try:
        await save_message_async(
            event.chat_id,
            event.message.id,
            event.message.date or datetime.now(TIMEZONE),
            "me",
            text,
            True
        )
    except Exception:
        log.exception("Ошибка сохранения исходящего сообщения")


# =========================
# АВТОСООБЩЕНИЯ
# =========================

async def followup_loop():
    while True:
        try:
            if FOLLOWUP_ENABLED and not quiet_hours():
                with db_connect() as db:
                    rows = db.execute("""
                        SELECT chat_id, last_incoming, last_outgoing,
                               last_followup
                        FROM chat_state
                        WHERE last_incoming IS NOT NULL
                    """).fetchall()

                now = datetime.now(TIMEZONE)

                for chat_id, incoming, outgoing, last_followup in rows:
                    incoming_dt = parse_date(incoming)
                    outgoing_dt = parse_date(outgoing)
                    followup_dt = parse_date(last_followup)

                    if not incoming_dt:
                        continue
                    if outgoing_dt and outgoing_dt > incoming_dt:
                        continue
                    if now - incoming_dt < timedelta(hours=FOLLOWUP_HOURS):
                        continue
                    if followup_dt and now - followup_dt < timedelta(
                        hours=FOLLOWUP_COOLDOWN_HOURS
                    ):
                        continue

                    try:
                        entity = await client.get_entity(chat_id)
                        context = build_context(chat_id)
                        context.append({
                            "role": "user",
                            "content": (
                                "Составь одно короткое, ненавязчивое "
                                "сообщение, чтобы продолжить разговор. "
                                "Не упоминай, что прошло несколько часов. "
                                "Не повторяй предыдущие сообщения."
                            )
                        })

                        text = await openrouter_request(context)
                        if text:
                            await client.send_message(
                                entity,
                                text[:MAX_MESSAGE_CHARS]
                            )
                            with db_connect() as db:
                                db.execute("""
                                    UPDATE chat_state
                                    SET last_followup = ?, last_outgoing = ?
                                    WHERE chat_id = ?
                                """, (
                                    now.isoformat(),
                                    now.isoformat(),
                                    chat_id
                                ))
                    except Exception:
                        log.exception("Ошибка follow-up для %s", chat_id)

                    await asyncio.sleep(1)

        except Exception:
            log.exception("Ошибка цикла follow-up")

        await asyncio.sleep(POLL_SECONDS)


async def first_message_loop():
    while True:
        try:
            if FIRST_MESSAGE_ENABLED and not quiet_hours():
                now = datetime.now(TIMEZONE)

                for item in ALLOWED_CHATS:
                    try:
                        if item.isdigit():
                            entity = await client.get_entity(int(item))
                        else:
                            entity = await client.get_entity(item)

                        chat_id = entity.id
                        state = get_state(chat_id)
                        incoming = parse_date(state["last_incoming"])
                        outgoing = parse_date(state["last_outgoing"])
                        last_first = parse_date(state["last_first_message"])

                        if incoming and now - incoming < timedelta(hours=24):
                            continue
                        if outgoing and now - outgoing < timedelta(hours=24):
                            continue
                        if last_first and now - last_first < timedelta(hours=24):
                            continue

                        await client.send_message(entity, FIRST_MESSAGE_TEXT)

                        with db_connect() as db:
                            db.execute("""
                                UPDATE chat_state
                                SET last_first_message = ?, last_outgoing = ?
                                WHERE chat_id = ?
                            """, (
                                now.isoformat(),
                                now.isoformat(),
                                chat_id
                            ))

                    except Exception:
                        log.exception("Ошибка первого сообщения для %s", item)

                    await asyncio.sleep(1)

        except Exception:
            log.exception("Ошибка цикла первого сообщения")

        await asyncio.sleep(POLL_SECONDS)


# =========================
# ЗАПУСК
# =========================

async def main():
    db_connect().close()
    await client.start()
    log.info("Kissa AI AutoReply запущен")

    tasks = [
        asyncio.create_task(followup_loop()),
        asyncio.create_task(first_message_loop())
    ]

    try:
        await client.run_until_disconnected()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
