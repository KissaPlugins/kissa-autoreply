import os
import asyncio
import logging
import sqlite3
import base64
import tempfile

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import httpx
import cv2

from telethon import TelegramClient, events
from telethon.sessions import StringSession
from telethon.tl.types import (
    DocumentAttributeSticker,
    DocumentAttributeAnimated,
)


# =========================
# НАСТРОЙКИ
# =========================

API_ID = int(os.environ["TG_API_ID"])
API_HASH = os.environ["TG_API_HASH"]
TG_SESSION = os.environ["TG_SESSION"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")

ALLOWED_CHATS = {
    x.strip().lstrip("@").lower()
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

# Медиа
MAX_MEDIA_BYTES = 18 * 1024 * 1024
MAX_VIDEO_FRAMES = 6
MAX_IMAGE_SIDE = 2048

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

log = logging.getLogger("kissa")

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
            date TEXT NOT NULL DEFAULT '',
            sender TEXT NOT NULL DEFAULT 'contact',
            text TEXT NOT NULL DEFAULT '',
            outgoing INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (chat_id, message_id)
        )
    """)

    # Миграция старой базы: добавляем отсутствующие столбцы.
    columns = {
        row["name"]
        for row in db.execute(
            "PRAGMA table_info(messages)"
        ).fetchall()
    }

    required_columns = {
        "date": "TEXT NOT NULL DEFAULT ''",
        "sender": "TEXT NOT NULL DEFAULT 'contact'",
        "text": "TEXT NOT NULL DEFAULT ''",
        "outgoing": "INTEGER NOT NULL DEFAULT 0",
    }

    for column, definition in required_columns.items():
        if column not in columns:
            db.execute(
                f"ALTER TABLE messages ADD COLUMN "
                f"{column} {definition}"
            )

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

    # Миграция состояния, если база была создана старой версией.
    state_columns = {
        row["name"]
        for row in db.execute(
            "PRAGMA table_info(chat_state)"
        ).fetchall()
    }

    state_required = {
        "last_incoming_at": "TEXT",
        "last_outgoing_at": "TEXT",
        "last_followup_at": "TEXT",
        "last_first_message_at": "TEXT",
        "history_loaded": "INTEGER NOT NULL DEFAULT 0",
    }

    for column, definition in state_required.items():
        if column not in state_columns:
            db.execute(
                f"ALTER TABLE chat_state ADD COLUMN "
                f"{column} {definition}"
            )

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
    allowed_fields = {
        "last_incoming_at",
        "last_outgoing_at",
        "last_followup_at",
        "last_first_message_at",
        "history_loaded",
    }

    values = {
        key: value
        for key, value in values.items()
        if key in allowed_fields
    }

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


def store_message(chat_id, message, outgoing=None, text_override=None):
    text = (
        text_override
        if text_override is not None
        else getattr(message, "raw_text", None) or ""
    )

    text = text.strip()

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
    text_override=None,
):
    async with db_lock:
        store_message(
            chat_id,
            message,
            outgoing,
            text_override,
        )


# =========================
# ЗАГРУЗКА ИСТОРИИ
# =========================

async def load_history_in_batches(entity, chat_id):
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
                for message in reversed(batch):
                    if (
                        getattr(message, "raw_text", None)
                        or getattr(message, "media", None)
                    ):
                        before = db.total_changes

                        stored_text = (
                            message.raw_text
                            or (
                                "[Фото]"
                                if getattr(message, "photo", None)
                                else "[Медиа]"
                            )
                        )

                        store_message(
                            chat_id,
                            message,
                            text_override=stored_text,
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
    try:
        messages = await client.get_messages(
            entity,
            limit=RECENT_CONTEXT_LIMIT,
        )

        async with db_lock:
            for message in reversed(messages):
                if (
                    getattr(message, "raw_text", None)
                    or getattr(message, "media", None)
                ):
                    stored_text = (
                        message.raw_text
                        or (
                            "[Фото]"
                            if getattr(message, "photo", None)
                            else "[Медиа]"
                        )
                    )

                    store_message(
                        chat_id,
                        message,
                        text_override=stored_text,
                    )

    except Exception:
        log.exception(
            "Не удалось обновить историю чата %s",
            chat_id,
        )


# =========================
# ОБРАБОТКА ФОТО И ВИДЕО
# =========================

def is_ignored_media(message):
    """
    Не отвечаем на стикеры, GIF и анимированные документы.
    """

    document = getattr(message, "document", None)

    if not document:
        return False

    attributes = getattr(document, "attributes", []) or []

    if any(
        isinstance(
            attribute,
            (
                DocumentAttributeSticker,
                DocumentAttributeAnimated,
            ),
        )
        for attribute in attributes
    ):
        return True

    mime = (
        getattr(message.file, "mime_type", None) or ""
    ).lower()

    name = (
        getattr(message.file, "name", None) or ""
    ).lower()

    if mime == "image/gif" or name.endswith(".gif"):
        return True

    return False


async def download_media_bytes(message):
    try:
        data = await message.download_media(file=bytes)

        if not isinstance(data, bytes) or not data:
            return None

        if len(data) > MAX_MEDIA_BYTES:
            log.info(
                "Медиа пропущено: размер больше лимита."
            )
            return None

        return data

    except Exception:
        log.exception("Ошибка скачивания медиа")
        return None


def resize_and_encode_frame(frame):
    height, width = frame.shape[:2]
    longest = max(height, width)

    if longest > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / longest

        frame = cv2.resize(
            frame,
            (
                max(1, int(width * scale)),
                max(1, int(height * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )

    success, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), 82],
    )

    if not success:
        return None

    encoded_base64 = base64.b64encode(
        encoded.tobytes()
    ).decode("ascii")

    return "data:image/jpeg;base64," + encoded_base64



async def photo_to_content(message):
    data = await download_media_bytes(message)

    if not data:
        return None

    try:
        import numpy as np

        image_array = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR,
        )

        if frame is None:
            log.warning("Не удалось прочитать изображение")
            return None

        image_url = resize_and_encode_frame(frame)

        if not image_url:
            return None

        return {
            "type": "image_url",
            "image_url": {
                "url": image_url,
            },
        }

    except Exception:
        log.exception("Ошибка обработки фотографии")
        return None



async def video_to_content(message):
    data = await download_media_bytes(message)

    if not data:
        return []

    temp_path = None
    cap = None

    try:
        video_name = (
            getattr(message.file, "name", None) or ""
        ).lower()

        video_suffix = os.path.splitext(video_name)[1]

        if video_suffix not in (
            ".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"
        ):
            video_suffix = ".mp4"

        with tempfile.NamedTemporaryFile(
            suffix=video_suffix,
            delete=False,
        ) as temp_file:
            temp_file.write(data)
            temp_path = temp_file.name

        cap = cv2.VideoCapture(temp_path)

        if not cap.isOpened():
            log.warning("Не удалось открыть видео")
            return []

        frame_count = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        )

        if frame_count <= 0:
            positions = [0]
        else:
            count = min(MAX_VIDEO_FRAMES, frame_count)

            if count == 1:
                positions = [0]
            else:
                positions = [
                    int(
                        index * (frame_count - 1)
                        / (count - 1)
                    )
                    for index in range(count)
                ]

        content_parts = []

        for frame_position in positions:
            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_position,
            )

            success, frame = cap.read()

            if not success:
                continue

            image_url = resize_and_encode_frame(frame)

            if image_url:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    },
                })

        return content_parts

    except Exception:
        log.exception("Ошибка обработки видео")
        return []

    finally:
        if cap is not None:
            cap.release()

        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass


async def prepare_incoming_content(message):
    """
    Возвращает контент для OpenRouter и текст для SQLite.
    GIF, стикеры, анимации, голосовые и прочие файлы пропускаются.
    """

    if is_ignored_media(message):
        return None, None

    text = (message.raw_text or "").strip()

    # Обычное текстовое сообщение.
    if not message.media:
        if not text:
            return None, None

        return text[:1800], text

    # Фотография.
    if message.photo:
        image_part = await photo_to_content(message)

        if not image_part:
            return None, None

        prompt = text or (
            "Посмотри на фотографию и ответь естественно "
            "с учётом того, что на ней изображено."
        )

        return [
            {
                "type": "text",
                "text": prompt[:1800],
            },
            image_part,
        ], "[Фото]" + (": " + text if text else "")

    # Видео.
    mime = (
        getattr(message.file, "mime_type", None) or ""
    ).lower()

    name = (
        getattr(message.file, "name", None) or ""
    ).lower()

    is_video = (
        mime.startswith("video/")
        or name.endswith(
            (".mp4", ".mov", ".m4v", ".webm")
        )
    )

    if is_video:
        frames = await video_to_content(message)

        if not frames:
            return None, None

        prompt = text or (
            "Проанализируй кадры этого видео и ответь "
            "по существу. Не придумывай детали, которых "
            "не видно на кадрах."
        )

        return [
            {
                "type": "text",
                "text": prompt[:1800],
            },
            *frames,
        ], "[Видео]" + (": " + text if text else "")

    # Не отвечаем на голосовые, аудио, документы,
    # стикеры и неподдерживаемые типы медиа.
    return None, None


# =========================
# КОНТЕКСТ ДЛЯ ИИ
# =========================

SYSTEM_PROMPT = (
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



async def build_context(chat_id, current_text=None):
    async with db_lock:
        rows = db.execute(
            """
            SELECT date, sender, text, outgoing
            FROM messages
            WHERE chat_id = ?
            ORDER BY message_id DESC
            LIMIT ?
            """,
            (chat_id, RECENT_CONTEXT_LIMIT),
        ).fetchall()

    rows = list(reversed(rows))

    history = []

    for row in rows:
        role = "assistant" if row["outgoing"] else "user"
        text = (row["text"] or "").strip()

        if not text:
            continue

        # Объединяем подряд идущие сообщения одной роли.
        if history and history[-1]["role"] == role:
            history[-1]["content"] += "\n" + text[:1800]
        else:
            history.append({
                "role": role,
                "content": text[:1800],
            })

    if current_text is not None:
        current_text = str(current_text).strip()

        if current_text:
            if history and history[-1]["role"] == "user":
                if history[-1]["content"] != current_text:
                    history[-1]["content"] += "\n" + current_text[:1800]
            else:
                history.append({
                    "role": "user",
                    "content": current_text[:1800],
                })

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
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
        timeout=60
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


        log.info(
            "OpenRouter response: %s",
            str(data)[:2500],
        )
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

    # Некоторые мультимодальные модели возвращают
    # содержимое в виде списка блоков.
    if isinstance(content, list):
        text_parts = []

        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    text_parts.append(
                        part.get("text", "")
                    )

        content = "\n".join(text_parts)

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
        return username in ALLOWED_CHATS

    return False


def get_chat_lock(chat_id):
    if chat_id not in chat_locks:
        chat_locks[chat_id] = asyncio.Lock()

    return chat_locks[chat_id]


# =========================
# ВХОДЯЩИЕ СООБЩЕНИЯ
# =========================

async def handle_incoming(event):
    if not event.is_private or not event.chat_id:
        return

    chat_id = int(event.chat_id)
    message = event.message

    sender = await event.get_sender()
    username = getattr(sender, "username", None)

    if not allowed_chat(chat_id, username):
        return

    # Сохраняем прежнее поведение тихих часов.
    if quiet_hours():
        log.info(
            "Тихие часы: сообщение в чате %s пропущено",
            chat_id,
        )
        return


    content, stored_text = await prepare_incoming_content(
        message
    )

if content is None:
    log.info(
        "Пропущено неподдерживаемое медиа в чате %s",
        chat_id,
    )
    return

if isinstance(content, str) and not content.strip():
    return


    lock = get_chat_lock(chat_id)

    async with lock:
        try:
            await save_message_async(
                chat_id,
                message,
                outgoing=False,
                text_override=stored_text,
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

            # Синхронизируем последние сообщения.
            await sync_recent_history(
                event.input_chat,
                chat_id,
            )

            messages = await build_context(chat_id)

            # Заменяем текущее текстовое представление
            # мультимодальным содержимым, чтобы изображение
            # действительно попало в запрос к модели.
            last_user_index = None

            for index in range(
                len(messages) - 1,
                0,
                -1,
            ):
                if messages[index]["role"] == "user":
                    last_user_index = index
                    break

            if last_user_index is not None:
                messages[last_user_index]["content"] = content
            else:
                messages.append({
                    "role": "user",
                    "content": content,
                })

            answer = await openrouter_request(messages)

            if not answer:
                log.warning(
                    "OpenRouter вернул пустой ответ для чата %s",
                    chat_id,
                )
                return

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
            if FOLLOWUP_ENABLED and not quiet_hours():
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
                            chat_id
                        )

                        messages.append({
                            "role": "user",
                            "content": (
                                "Сгенерируй одно короткое "
                                "ненавязчивое сообщение, чтобы "
                                "естественно продолжить этот "
                                "разговор. Не упоминай время "
                                "ожидания. Верни только текст."
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
            log.exception("Ошибка цикла follow-up")

        await asyncio.sleep(POLL_SECONDS)


# =========================
# ПЕРВОЕ СООБЩЕНИЕ
# =========================

async def first_message_loop():
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

                    if (
                        last_first_message_day.get(target)
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

                        last_first_message_day[target] = today

                        log.info(
                            "Отправлено первое сообщение "
                            "в чат %s",
                            chat_id,
                        )

                    except Exception:
                        log.exception(
                            "Не удалось отправить первое "
                            "сообщение адресату %s",
                            target,
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
            asyncio.create_task(followup_loop())
        )

    if FIRST_MESSAGE_ENABLED:
        tasks.append(
            asyncio.create_task(first_message_loop())
        )

    try:
        await client.run_until_disconnected()

    finally:
        for task in tasks:
            task.cancel()

        await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )

        await client.disconnect()
        db.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())

    except KeyboardInterrupt:
        pass
