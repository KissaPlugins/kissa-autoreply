import os
import asyncio
import logging
from telethon import TelegramClient, events
from telethon.sessions import StringSession

logging.basicConfig(level=logging.INFO)

API_ID = int(os.environ["TG_API_ID"])
API_HASH = os.environ["TG_API_HASH"]
SESSION_STRING = os.environ["TG_SESSION"]

client = TelegramClient(
    StringSession(SESSION_STRING),
    API_ID,
    API_HASH,
)

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

    await event.reply("Увидел сообщение, отвечу чуть позже.")

async def main():
    await client.connect()

    if not await client.is_user_authorized():
        logging.error("Сессия недействительна. Нужно создать новую.")
        return

    logging.info("Kissa AutoReply запущен")
    await client.run_until_disconnected()

if __name__ == "__main__":
    asyncio.run(main())


