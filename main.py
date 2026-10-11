import os
import io
import asyncio
import logging
import base64
import mimetypes

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

MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_VIDEO_BYTES = 10 * 1024 * 1024

SYSTEM_PROMPT = """
Ты Kissa — спокойный собеседник в Telegram.
Пиши коротко, естественно и по-человечески.
Не используй эмодзи без необходимости.
Не говори, что ты искусственный интеллект.
Не придумывай детали, которых нет в сообщении.
Если прислали обычное приветствие — поздоровайся в ответ.
Если сообщение оскорбительное и не содержит вопроса или просьбы,
ответь коротко: «ок».
Если в сообщении есть нормальный вопрос или просьба,
ответь именно на него, даже если присутствуют грубые слова.
На романтические признания и приглашения отвечай вежливым отказом.
Если прислали фото, опиши его или ответь на вопрос о нём.
Если прислали видео, используй доступную подпись и контекст.
Отвечай на русском, если собеседник пишет по-русски.
"""

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

# =========================
# OPENROUTER
# =========================

async def ask_ai(text, image_data=None, image_mime="image/jpeg"):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/KissaPlugins/kissa-autoreply",
        "X-Title": "Kissa AI AutoReply",
    }

    if image_data:
        encoded = base64.b64encode(image_data).decode("utf-8")
        user_content = [
            {
                "type": "text",
                "text": text or "Посмотри на это изображение и кратко опиши его.",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{image_mime};base64,{encoded}"
                },
            },
        ]
    else:
        user_content = text

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.6,
        "max_tokens": 120,
    }

    async with httpx.AsyncClient(timeout=60) as http:
        response = await http.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    answer = data["choices"][0]["message"].get("content", "")
    if isinstance(answer, list):
        answer = " ".join(
            item.get("text", "")
            for item in answer
            if isinstance(item, dict)
        )

    return str(answer).strip()


# =========================
# ОБРАБОТКА СООБЩЕНИЙ
# =========================

def chat_is_allowed(event):
    if not ALLOWED_CHATS:
        return True

    chat_id = str(event.chat_id)
    username = ""

    if event.sender and getattr(event.sender, "username", None):
        username = event.sender.username.lower()

    return (
        chat_id in ALLOWED_CHATS
        or username in {x.lower() for x in ALLOWED_CHATS}
    )


@client.on(events.NewMessage(incoming=True))
async def on_message(event):
    if not event.is_private:
        return

    if not chat_is_allowed(event):
        return

    message = event.message

    # Стикеры и GIF игнорируем.
    if message.sticker or message.gif:
        log.info("Стикер/GIF пропущен")
        return

    # Не отвечаем на сообщения без текста, фото или видео.
    text = (event.raw_text or "").strip()
    has_photo = bool(message.photo)
    has_video = bool(
        message.video
        or (
            message.document
            and getattr(message.document, "mime_type", "").startswith("video/")
        )
    )

    if not text and not has_photo and not has_video:
        return

    try:
        # Фото: передаём изображение модели.
        if has_photo:
            media = await event.download_media(file=bytes)

            if not media:
                log.warning("Не удалось скачать фото")
                return

            if len(media) > MAX_IMAGE_BYTES:
                await event.reply("Фото слишком большое, отправь его поменьше.")
                return

            answer = await ask_ai(
                text or "Кратко опиши фото.",
                image_data=media,
                image_mime="image/jpeg",
            )

        # Видео: пока отвечаем с учётом подписи.
        # Само видео модель здесь не просматривает.
        elif has_video:
            if message.file and message.file.size:
                if message.file.size > MAX_VIDEO_BYTES:
                    await event.reply(
                        "Видео слишком большое для обработки. "
                        "Можешь отправить короткий фрагмент или кадр."
                    )
                    return

            prompt = (
                "Пользователь отправил видео. "
                "Само видео здесь не проанализировано. "
                "Ответь с учётом подписи, если она есть. "
                "Если подписи нет, кратко попроси описать, что на видео."
            )
            if text:
                prompt += "\nПодпись к видео: " + text

            answer = await ask_ai(prompt)

        # Обычный текст.
        else:
            answer = await ask_ai(text)

        if not answer:
            answer = "Нормально."

        await event.reply(answer)
        log.info("Ответ отправлен в чат %s", event.chat_id)

    except httpx.HTTPStatusError as exc:
        log.error(
            "OpenRouter HTTP %s: %s",
            exc.response.status_code,
            exc.response.text[:500],
        )
    except Exception:
        log.exception("Ошибка обработки сообщения")


# =========================
# ЗАПУСК
# =========================

async def main():
    log.info("Запуск Kissa AI AutoReply")
    log.info("Модель OpenRouter: %s", MODEL)

    await client.start()
    me = await client.get_me()

    log.info(
        "Telegram подключён. Аккаунт: %s",
        getattr(me, "username", None) or me.id,
    )

    await client.run_until_disconnected()


if __name__ == "__main__":
    asyncio.run(main())

