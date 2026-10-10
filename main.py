
import os
import asyncio
import logging
from telethon import TelegramClient, events

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

API_ID = int(os.environ["TG_API_ID"])
API_HASH = os.environ["TG_API_HASH"]
SESSION_STRING = os.environ.get("TG_SESSION", "")

client = TelegramClient(
    "kissa_autoreply",
    API_ID,
    API_HASH,
)

# Не отвечаем на сообщения, отправленные самим аккаунтом.
# Не отвечаем в группах и каналах.
# Здесь пока используется простой шаблон, без внешнего ИИ.

@client.on(events.NewMessage(incoming=True))
async def auto_reply(event):
    if not event.is_private:
        return

    sender = await event.get_sender()
    if getattr(sender, "bot", False):
        return

    message = (event.raw_text or "").strip()
    if not message:
        return

    logging.info("Новое личное сообщение получено")

    # Простой стартовый вариант: ответ только на первое
    # входящее сообщение в диалоге после запуска.
    # Вместо шаблона позднее подключим генерацию ИИ.
    reply = "увидел, отвечу чуть позже"

    try:
        await event.reply(reply)
    except Exception:
        logging.exception("Не удалось отправить ответ")


async def main():
    if SESSION_STRING:
        from telethon.sessions import StringSession
        global client
        client = TelegramClient(
            StringSession(SESSION_STRING),
            API_ID,
            API_HASH,
        )

    await client.start()
    logging.info("Kissa AutoReply запущен")
    await client.run_until_disconnected()


if __name__ == "__main__":
    asyncio.run(main())
