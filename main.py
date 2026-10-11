
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
import numpy as np

from telethon import TelegramClient, events
from telethon.sessions import StringSession
from telethon.tl.types import (
    DocumentAttributeSticker,
    DocumentAttributeAnimated,
    User,
)

# ==================================================
# НАСТРОЙКИ
# ==================================================

API_ID = int(os.environ["TG_API_ID"])
API_HASH = os.environ["TG_API_HASH"]
TG_SESSION = os.environ["TG_SESSION"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

# Укажи здесь модель, поддерживающую изображения,
# если хочешь анализировать фото и видео.
MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

ALLOWED_CHATS = {
    item.strip().lstrip("@").lower()
    for item in os.getenv("KISSA_ALLOWED_CHATS", "").split(",")
    if item.strip()
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
    "Привет! Как ты?"
)

TIMEZONE = ZoneInfo("Europe/Moscow")
DB_PATH = os.getenv("KISSA_DB_PATH", "kissa_memory.sqlite3")

BATCH_SIZE = 40
RECENT_CONTEXT_LIMIT = 30
MAX_TOKENS = 420
MAX_MESSAGE_CHARS = 3500
POLL_SECONDS = 60

MAX_MEDIA_BYTES = 18 * 1024 * 1024
MAX_VIDEO_FRAMES = 6
MAX_IMAGE_SIDE = 1600

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

db = sqlite3.connect(DB_PATH, check_same_thread=False)
db.row_factory = sqlite3.Row

db_lock = asyncio.Lock()
chat_locks = {}
last_first_message_day = {}


# ==================================================
# БАЗА ДАННЫХ И МИГРАЦИЯ СТАРОЙ БАЗЫ
# ==================================================

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

    columns = {
        row["name"]
        for row in db.execute("PRAGMA table_info(messages)").fetchall()
    }

    required = {
        "date": "TEXT NOT NULL DEFAULT ''",
        "sender": "TEXT NOT NULL DEFAULT 'contact'",
        "text": "TEXT NOT NULL DEFAULT ''",
        "outgoing": "INTEGER NOT NULL DEFAULT 0",
    }

    for column, definition in required.items():
        if column not in columns:
            db.execute(
                f"ALTER TABLE messages ADD COLUMN {column} {definition}"
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

    state_columns = {
        row["name"]
        for row in db.execute("PRAGMA table_info(chat_state)").fetchall()
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
                f"ALTER TABLE chat_state ADD COLUMN {column} {definition}"
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
        (int(chat_id),),
    )


def update_state(chat_id, **values):
    allowed = {
        "last_incoming_at",
        "last_outgoing_at",
        "last_followup_at",
        "last_first_message_at",
        "history_loaded",
    }
    values = {key: value for key, value in values.items() if key in allowed}
    if not values:
        return

    ensure_state(chat_id)
    fields = ", ".join(f"{key} = ?" for key in values)

    db.execute(
        f"UPDATE chat_state SET {fields} WHERE chat_id = ?",
        (*values.values(), int(chat_id)),
    )
    db.commit()


def store_message(chat_id, message, outgoing=None, text_override=None):
    text = (
        text_override
        if text_override is not None
        else getattr(message, "raw_text", None) or ""
    )
    text = str(text).strip()

    if not text:
        return

    is_outgoing = (
        bool(getattr(message, "out", False))
        if outgoing is None
        else bool(outgoing)
    )

    date = getattr(message, "date", None)
    if date:
        if date.tzinfo is None:
            date = date.replace(tzinfo=ZoneInfo("UTC"))
        date_text = date.astimezone(TIMEZONE).isoformat()
    else:
        date_text = now_iso()

    sender = (
        "me"
        if is_outgoing
        else str(getattr(message, "sender_id", None) or "contact")
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
    chat_id, message, outgoing=None, text_override=None
):
    async with db_lock:
        store_message(chat_id, message, outgoing, text_override)


# ==================================================
# ИСТОРИЯ ПЕРЕПИСКИ
# ==================================================

async def load_history_in_batches(entity, chat_id):
    async with db_lock:
        ensure_state(chat_id)
        row = db.execute(
            "SELECT history_loaded FROM chat_state WHERE chat_id = ?",
            (chat_id,),
        ).fetchone()
        db.commit()

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
                        text = message.raw_text or (
                            "[Фото]" if getattr(message, "photo", None)
                            else "[Медиа]"
                        )
                        before = db.total_changes
                        store_message(
                            chat_id, message, text_override=text
                        )
                        if db.total_changes > before:
                            saved += 1

            oldest_id = min(
                (m.id for m in batch if getattr(m, "id", None)),
                default=0,
            )
            if not oldest_id or oldest_id == offset_id:
                break

            offset_id = oldest_id
            await asyncio.sleep(0.15)

        async with db_lock:
            update_state(chat_id, history_loaded=1)

        log.info("История чата %s обработана; новых сообщений: %s",
                 chat_id, saved)

    except Exception:
        log.exception("Ошибка загрузки истории чата %s", chat_id)


async def sync_recent_history(entity, chat_id):
    try:
        messages = await client.get_messages(
            entity, limit=RECENT_CONTEXT_LIMIT
        )
        async with db_lock:
            for message in reversed(messages):
                if (
                    getattr(message, "raw_text", None)
                    or getattr(message, "media", None)
                ):
                    text = message.raw_text or (
                        "[Фото]" if getattr(message, "photo", None)
                        else "[Медиа]"
                    )
                    store_message(
                        chat_id, message, text_override=text
                    )
    except Exception:
        log.exception("Ошибка синхронизации истории чата %s", chat_id)


# ==================================================
# ФОТО, ВИДЕО И ПРОПУСК НЕНУЖНЫХ МЕДИА
# ==================================================

def is_ignored_media(message):
    document = getattr(message, "document", None)
    if not document:
        return False

    attributes = getattr(document, "attributes", []) or []
    if any(
        isinstance(attr, (DocumentAttributeSticker, DocumentAttributeAnimated))
        for attr in attributes
    ):
        return True

    mime = (getattr(message.file, "mime_type", None) or "").lower()
    name = (getattr(message.file, "name", None) or "").lower()

    return mime == "image/gif" or name.endswith(".gif")


async def download_media_bytes(message):
    try:
        data = await message.download_media(file=bytes)
        if not isinstance(data, bytes) or not data:
            return None
        if len(data) > MAX_MEDIA_BYTES:
            log.info("Медиа пропущено: файл превышает лимит размера")
            return None
        return data
    except Exception:
        log.exception("Ошибка скачивания медиа")
        return None


def encode_frame(frame):
    height, width = frame.shape[:2]
    longest = max(height, width)

    if longest > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / longest
        frame = cv2.resize(
            frame,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    success, encoded = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
    )
    if not success:
        return None

    data = base64.b64encode(encoded.tobytes()).decode("ascii")
    return "data:image/jpeg;base64," + data


async def photo_to_content(message):
    data = await download_media_bytes(message)
    if not data:
        return None

    try:
        array = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if frame is None:
            return None
        url = encode_frame(frame)
        if not url:
            return None
        return {"type": "image_url", "image_url": {"url": url}}
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
        name = (getattr(message.file, "name", None) or "").lower()
        suffix = os.path.splitext(name)[1]
        if suffix not in (".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"):
            suffix = ".mp4"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            temp.write(data)
            temp_path = temp.name

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            log.warning("Не удалось открыть видео")
            return []

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 1:
            positions = [0]
        else:
            count = min(MAX_VIDEO_FRAMES, frame_count)
            positions = [
                int(i * (frame_count - 1) / (count - 1))
                for i in range(count)
            ]

        parts = []
        for position in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, position)
            success, frame = cap.read()
            if not success:
                continue
            url = encode_frame(frame)
            if url:
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": url},
                })
        return parts

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
    if is_ignored_media(message):
        return None, None

    text = (message.raw_text or "").strip()

    if not message.media:
        return (text[:1800], text) if text else (None, None)

    if message.photo:
        image_part = await photo_to_content(message)
        if not image_part:
            return None, None

        prompt = text or (
            "Посмотри на фотографию и ответь естественно "
            "с учётом того, что на ней изображено."
        )
        return [
            {"type": "text", "text": prompt[:1800]},
            image_part,
        ], "[Фото]" + (": " + text if text else "")

    mime = (getattr(message.file, "mime_type", None) or "").lower()
    name = (getattr(message.file, "name", None) or "").lower()

    is_video = mime.startswith("video/") or name.endswith(
        (".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv")
    )

    if is_video:
        frames = await video_to_content(message)
        if not frames:
            return None, None

        prompt = text or (
            "Проанализируй кадры этого видео. Ответь по существу "
            "и не придумывай детали, которых не видно."
        )
        return [
            {"type": "text", "text": prompt[:1800]},
            *frames,
        ], "[Видео]" + (": " + text if text else "")

    # Голосовые, аудио, документы и неподдерживаемые файлы пропускаем.
    return None, None


# ==================================================
# ИНСТРУКЦИЯ ДЛЯ ИИ
# ==================================================


SYSTEM_PROMPT = (
    "Ты отвечаешь на личные сообщения Telegram естественно, внимательно и по существу. "
    "Главное правило: ответь именно на последнее сообщение с учётом ближайшего контекста. "
    "Не меняй тему, не выдумывай факты о владельце и не соглашайся автоматически. "
    "Не используй заготовленные универсальные реакции вроде «Хахаха, точно!», "
    "«ахах», «понятно» или случайные вопросы, если они не подходят по смыслу. "
    "Не смейся над собеседником, который злится, жалуется, обижен или говорит, "
    "что ты несёшь бред, не понимаешь, стал хуже, или просит вернуть прежний стиль. "
    "В таких случаях спокойно признай проблему, извинись коротко и исправь ответ "
    "по теме; не спорь и не оправдывайся. "
    "Если собеседник оскорбляет, но задаёт вопрос или что-то просит, ответь на суть "
    "спокойно. Если сообщение — только оскорбление без вопроса и просьбы, можно "
    "кратко обозначить границу или не раздувать конфликт. Не отвечай смехом. "
    "Учитывай опечатки, сленг, мат, сокращения, мемы и фонетическое написание; "
    "восстанавливай смысл по контексту, но не выдумывай уверенность, если смысл "
    "действительно неясен. Тогда задай один короткий уточняющий вопрос. "
    "Не добавляй вопрос в конце каждого ответа. Не начинай случайные темы, "
    "не повторяй недавние ответы и не используй мемы без явной связи с сообщением. "
    "Если предлагают встречу, прогулку, созвон, игру или другое совместное занятие, "
    "вежливо откажи и не обещай на будущее. "
    "Перед отправкой проверь: ответ связан с последней репликой, не звучит как "
    "шаблон, не смеётся над серьёзной эмоцией и не содержит выдуманных деталей. "
    "Отвечай обычно коротко, по-человечески и на языке собеседника."
)


def is_generic_bad_reply(answer):
    """Detect repetitive filler that should not be sent as a context-aware reply."""
    normalized = " ".join((answer or "").lower().split()).strip(" !?.…")
    bad_replies = {
        "хахаха, точно", "хахаха точно", "ахах, точно", "ахах точно",
        "хаха, точно", "хаха точно", "понятно", "ясно", "ок", "окей",
        "ахахах", "хахаха", "ахаха", "лол"
    }
    return normalized in bad_replies


async def build_context(chat_id):
    async with db_lock:
        rows = db.execute(
            """
            SELECT sender, text, outgoing
            FROM messages
            WHERE chat_id = ?
            ORDER BY message_id DESC
            LIMIT ?
            """,
            (chat_id, RECENT_CONTEXT_LIMIT),
        ).fetchall()

    history = []
    for row in reversed(rows):
        text = (row["text"] or "").strip()
        if not text:
            continue

        role = "assistant" if row["outgoing"] else "user"
        if history and history[-1]["role"] == role:
            history[-1]["content"] += "\n" + text[:1800]
        else:
            history.append({"role": role, "content": text[:1800]})

    return [{"role": "system", "content": SYSTEM_PROMPT}] + history


# ==================================================
# OPENROUTER
# ==================================================

async def openrouter_request(messages, max_tokens=MAX_TOKENS):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/KissaPlugins/kissa-autoreply",
        "X-Title": "Kissa AI AutoReply",
    }

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.55,
        "max_tokens": max_tokens,
    }

    async with httpx.AsyncClient(timeout=90) as http:
        response = await http.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
        )

        if response.is_error:
            log.error(
                "OpenRouter HTTP %s: %s",
                response.status_code,
                response.text[:1000],
            )
            response.raise_for_status()

        data = response.json()

    choices = data.get("choices") or []
    if not choices:
        log.error("OpenRouter не вернул choices: %s", str(data)[:1500])
        return ""

    message = choices[0].get("message") or {}
    content = message.get("content", "")

    if isinstance(content, list):
        content = "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )

    if not isinstance(content, str):
        content = ""

    content = content.strip().strip('"').strip()

    if not content:
        log.warning(
            "OpenRouter вернул пустой ответ. Модель: %s; finish_reason: %s",
            data.get("model"),
            choices[0].get("finish_reason"),
        )
        return ""

    return content


# ==================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==================================================

def quiet_hours():
    hour = datetime.now(TIMEZONE).hour
    return 0 <= hour < 7


def allowed_chat(chat_id, username=None):
    if not ALLOWED_CHATS:
        return False

    if str(chat_id) in ALLOWED_CHATS:
        return True

    if username:
        return username.lower().lstrip("@") in ALLOWED_CHATS

    return False


def get_chat_lock(chat_id):
    if chat_id not in chat_locks:
        chat_locks[chat_id] = asyncio.Lock()
    return chat_locks[chat_id]


# ==================================================
# ВХОДЯЩИЕ СООБЩЕНИЯ
# ==================================================

async def handle_incoming(event):
    # Только личные сообщения от реальных пользователей. Не отвечаем ботам,
    # группам, каналам, собственным сообщениям и системным сущностям.
    if not event.is_private or not event.chat_id:
        return

    sender = await event.get_sender()
    if not isinstance(sender, User) or bool(getattr(sender, "bot", False)):
        log.info("Входящее сообщение пропущено: отправитель не обычный пользователь")
        return

    me = await client.get_me()
    if sender.id == me.id:
        return

    chat_id = int(event.chat_id)
    message = event.message

    # Ограничение KISSA_ALLOWED_CHATS отключено:
    # отвечаем всем личным чатам.

    if quiet_hours():
        log.info("Тихие часы: чат %s пропущен", chat_id)
        return

    content, stored_text = await prepare_incoming_content(message)

    if content is None:
        log.info("Неподдерживаемое медиа в чате %s пропущено", chat_id)
        return

    async with get_chat_lock(chat_id):
        try:
            await save_message_async(
                chat_id, message, outgoing=False, text_override=stored_text
            )

            async with db_lock:
                update_state(chat_id, last_incoming_at=now_iso())

            asyncio.create_task(
                load_history_in_batches(event.input_chat, chat_id)
            )
            await sync_recent_history(event.input_chat, chat_id)

            messages = await build_context(chat_id)

            # Последнее сообщение заменяем текстом вместе с фото/кадрами,
            # чтобы модель получила именно текущее входящее сообщение.
            last_user_index = next(
                (
                    i for i in range(len(messages) - 1, 0, -1)
                    if messages[i]["role"] == "user"
                ),
                None,
            )

            if last_user_index is not None:
                messages[last_user_index]["content"] = content
            else:
                messages.append({"role": "user", "content": content})

            answer = await openrouter_request(messages)

            if not answer:
                log.warning("Пустой ответ для чата %s; пробуем повторно", chat_id)
                retry_messages = messages + [{
                    "role": "user",
                    "content": (
                        "Предыдущая попытка не дала текста. Составь один "
                        "короткий естественный ответ именно на последнее сообщение. "
                        "Не отвечай автоматически «ок», если это неуместно. "
                        "Верни только текст ответа."
                    ),
                }]
                answer = await openrouter_request(retry_messages)
                if not answer:
                    log.error("Повторная попытка тоже не дала ответа для чата %s", chat_id)
                    return

            # Защита от повторов: не отправляем точную копию недавнего ответа.
            normalized_answer = " ".join(answer.lower().split()).strip(" .!?…\"'«»")
            recent_outgoing = [
                " ".join(str(item.get("content", "")).lower().split()).strip(" .!?…\"'«»")
                for item in messages
                if item.get("role") == "assistant" and isinstance(item.get("content"), str)
            ][-8:]
            if normalized_answer and normalized_answer in recent_outgoing:
                log.info("Повтор ответа предотвращён в чате %s", chat_id)
                messages.append({
                    "role": "user",
                    "content": (
                        "Перечитай последние сообщения и ответь именно на последнюю реплику. "
                        "Не используй ту же мысль или шаблон, что в предыдущих ответах. "
                        "Не смейся автоматически и не добавляй вопрос ради продолжения разговора. "
                        "Не выдумывай тему или детали, которых нет в сообщении. "
                        "Строго не упоминай миньонов, мультфильмы, персонажей и мемы, "
                        "если последнее сообщение явно не о них. Старые упоминания запрещено "
                        "переносить в новую тему. Ответь только по смыслу последней реплики, "
                        "учти поправки собеседника и не повторяй прежние шутки или шаблоны. "
                        "Не выдумывай деталей. Сначала проанализируй контекст, затем ответь естественно. "
                        "Верни только текст ответа."
                    ),
                })
                answer = await openrouter_request(messages)
                if not answer:
                    return

            # Жёсткий предохранитель от случайных упоминаний миньонов.
            # Если собеседник сам не поднял эту тему в последнем сообщении,
            # не отправляем такой ответ и просим модель сформулировать заново.
            latest_text = str(content).lower()
            minion_terms = ("миньон", "миньоны", "миньонов", "minion", "minions")
            answer_has_minion = any(term in answer.lower() for term in minion_terms)
            user_asked_about_minions = any(term in latest_text for term in minion_terms)
            if answer_has_minion and not user_asked_about_minions:
                log.warning("Удаляем неуместное упоминание миньонов в чате %s", chat_id)
                retry_messages = messages + [{
                    "role": "user",
                    "content": (
                        "ТВОЙ ПРЕДЫДУЩИЙ ОТВЕТ НЕПОДХОДЯЩИЙ: в нём случайная тема. "
                        "Составь ответ заново только на последнее сообщение и ближайший контекст. "
                        "Не упоминай миньонов, мультфильмы, персонажей или мемы. "
                        "Если собеседник делится болью или просит поддержки — искренне поддержи его, "
                        "можно развёрнуто. Если смеётся — ответь естественно на смех, без случайной темы. "
                        "Не выдумывай обстоятельств и не повторяй прошлые фразы. Верни только ответ."
                    ),
                }]
                replacement = await openrouter_request(retry_messages)
                if replacement and not any(term in replacement.lower() for term in minion_terms):
                    answer = replacement
                else:
                    # Безопасный нейтральный ответ вместо заведомо неуместного текста.
                    if any(word in latest_text for word in ("плохо", "груст", "тяжело", "устал", "одинок", "сброситься", "поддерж")):
                        answer = "Мне жаль, что тебе сейчас так тяжело. Я рядом и готов тебя выслушать. Хочешь рассказать, что случилось?"
                    elif any(word in latest_text for word in ("ахах", "хаха", "лол", "😂")):
                        answer = "Ахах, понял тебя"
                    else:
                        answer = "Что ты имеешь в виду?"

            if is_generic_bad_reply(answer):
            answer = "Извини, отвечу по сути, а не шаблоном."
        sent = await event.respond(answer[:MAX_MESSAGE_CHARS])

            if sent:
                await save_message_async(chat_id, sent, outgoing=True)
                async with db_lock:
                    update_state(
                        chat_id,
                        last_outgoing_at=now_iso(),
                        last_followup_at=None,
                    )

            log.info("Ответ отправлен в чат %s", chat_id)

        except httpx.HTTPStatusError as exc:
            log.error(
                "OpenRouter HTTP %s: %s",
                exc.response.status_code,
                exc.response.text[:500],
            )
        except Exception:
            log.exception("Ошибка обработки чата %s", chat_id)


@client.on(events.NewMessage(incoming=True))
async def on_incoming(event):
    await handle_incoming(event)


# ==================================================
# СОХРАНЕНИЕ ИСХОДЯЩИХ СООБЩЕНИЙ
# ==================================================

@client.on(events.NewMessage(outgoing=True))
async def on_outgoing(event):
    if not event.is_private or not event.chat_id:
        return

    # Не выполнять команды и не записывать историю в чатах с ботами.
    try:
        chat = await event.get_chat()
    except Exception:
        log.exception("Не удалось проверить адресата исходящего сообщения")
        return
    if not isinstance(chat, User) or bool(getattr(chat, "bot", False)):
        return

    text = (event.raw_text or "").strip()
    if not text:
        return

    # Команды владельца: ответы видны только в личном чате, где введена команда.
    command = text.split()[0].lower()
    if command in ("/status", "/status@kissa_autoreply_bot"):
        async with db_lock:
            row = db.execute(
                "SELECT COUNT(*) AS n FROM messages WHERE chat_id = ?",
                (int(event.chat_id),),
            ).fetchone()
        status = (
            "Kissa AI AutoReply — статус\n"
            f"Модель: {MODEL}\n"
            f"Тихие часы сейчас: {'да' if quiet_hours() else 'нет'}\n"
            f"Продолжение переписки: {'включено' if FOLLOWUP_ENABLED else 'выключено'}\n"
            f"Сообщений в памяти этого чата: {row['n'] if row else 0}\n"
            "Ответы: все личные чаты (если медиа поддерживается)."
        )
        await event.respond(status)
        log.info("Команда /status выполнена в чате %s", event.chat_id)
        return

    await save_message_async(event.chat_id, event.message, outgoing=True)

    async with db_lock:
        update_state(event.chat_id, last_outgoing_at=now_iso())


# ==================================================
# ПРОДОЛЖЕНИЕ ПЕРЕПИСКИ
# ==================================================

async def followup_loop():
    while True:
        try:
            if FOLLOWUP_ENABLED and not quiet_hours():
                async with db_lock:
                    rows = db.execute(
                        """
                        SELECT chat_id, last_incoming_at, last_outgoing_at,
                               last_followup_at
                        FROM chat_state
                        """
                    ).fetchall()

                now = datetime.now(TIMEZONE)

                for row in rows:
                    chat_id = int(row["chat_id"])
                    incoming = parse_dt(row["last_incoming_at"])
                    outgoing = parse_dt(row["last_outgoing_at"])
                    last_followup = parse_dt(row["last_followup_at"])

                    if not incoming or not outgoing:
                        continue
                    if incoming > outgoing:
                        continue
                    if now - outgoing < timedelta(hours=FOLLOWUP_HOURS):
                        continue
                    if (
                        last_followup
                        and now - last_followup
                        < timedelta(hours=FOLLOWUP_COOLDOWN_HOURS)
                    ):
                        continue

                    try:
                        entity = await client.get_entity(chat_id)
                        if not isinstance(entity, User) or bool(getattr(entity, "bot", False)):
                            log.info("Follow-up пропущен: чат %s не является обычным пользователем", chat_id)
                            continue
                        me = await client.get_me()
                        if entity.id == me.id:
                            continue
                        messages = await build_context(chat_id)
                        messages.append({
                            "role": "user",
                            "content": (
                                "Напиши одно короткое, естественное и ненавязчивое "
                                "сообщение, чтобы продолжить разговор. Не упоминай "
                                "время ожидания. Не используй пустые фразы вроде "
                                "«ок» или «понял». Верни только текст."
                            ),
                        })

                        answer = await openrouter_request(messages, max_tokens=100)
                        if not answer:
                            continue

                        sent = await client.send_message(
                            entity, answer[:MAX_MESSAGE_CHARS]
                        )
                        await save_message_async(chat_id, sent, outgoing=True)

                        async with db_lock:
                            update_state(
                                chat_id,
                                last_outgoing_at=now_iso(),
                                last_followup_at=now_iso(),
                            )

                        log.info("Продолжение отправлено в чат %s", chat_id)

                    except Exception:
                        log.exception("Ошибка follow-up для чата %s", chat_id)

        except Exception:
            log.exception("Ошибка цикла follow-up")

        await asyncio.sleep(POLL_SECONDS)


# ==================================================
# ПЕРВОЕ СООБЩЕНИЕ
# ==================================================

async def first_message_loop():
    while True:
        try:
            if FIRST_MESSAGE_ENABLED and not quiet_hours() and ALLOWED_CHATS:
                today = datetime.now(TIMEZONE).date().isoformat()

                for target in list(ALLOWED_CHATS):
                    if last_first_message_day.get(target) == today:
                        continue

                    try:
                        if target.isdigit():
                            entity = await client.get_entity(int(target))
                        else:
                            entity = await client.get_entity(target)

                        # Первое сообщение разрешено только обычным пользователям.
                        if not isinstance(entity, User) or bool(getattr(entity, "bot", False)):
                            log.info("Первое сообщение пропущено: %s не обычный пользователь", target)
                            continue
                        me = await client.get_me()
                        if entity.id == me.id:
                            continue
                        chat_id = int(getattr(entity, "id", 0))
                        if not chat_id:
                            continue

                        async with db_lock:
                            row = db.execute(
                                """
                                SELECT last_incoming_at, last_outgoing_at,
                                       last_first_message_at
                                FROM chat_state WHERE chat_id = ?
                                """,
                                (chat_id,),
                            ).fetchone()

                        now = datetime.now(TIMEZONE)

                        if row:
                            recent_in = parse_dt(row["last_incoming_at"])
                            recent_out = parse_dt(row["last_outgoing_at"])
                            last_first = parse_dt(row["last_first_message_at"])

                            if recent_in and now - recent_in < timedelta(hours=24):
                                continue
                            if recent_out and now - recent_out < timedelta(hours=24):
                                continue
                            if last_first and now - last_first < timedelta(hours=24):
                                continue

                        sent = await client.send_message(entity, FIRST_MESSAGE_TEXT)
                        await save_message_async(chat_id, sent, outgoing=True)

                        async with db_lock:
                            update_state(
                                chat_id,
                                last_first_message_at=now_iso(),
                                last_outgoing_at=now_iso(),
                            )

                        last_first_message_day[target] = today
                        log.info("Первое сообщение отправлено в чат %s", chat_id)

                    except Exception:
                        log.exception("Ошибка первого сообщения адресату %s", target)

        except Exception:
            log.exception("Ошибка цикла первого сообщения")

        await asyncio.sleep(300)


# ==================================================
# ЗАПУСК
# ==================================================

async def main():
    init_db()

    log.info("Запуск Kissa AI AutoReply")
    log.info("Модель OpenRouter: %s", MODEL)
    log.info("Автоответы: только личные чаты с обычными пользователями; боты и группы игнорируются")

    await client.start()
    me = await client.get_me()

    log.info(
        "Telegram подключён. Аккаунт: %s",
        getattr(me, "username", None) or me.id,
    )

    tasks = []

    if FOLLOWUP_ENABLED:
        tasks.append(asyncio.create_task(followup_loop()))

    if FIRST_MESSAGE_ENABLED:
        tasks.append(asyncio.create_task(first_message_loop()))

    try:
        await client.run_until_disconnected()
    finally:
        for task in tasks:
            task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)
        await client.disconnect()
        db.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
