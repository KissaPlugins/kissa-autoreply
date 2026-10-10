
import os
import asyncio
import logging
import json
import urllib.request
import urllib.error

from telethon import TelegramClient, events
from telethon.sessions import StringSession

logging.basicConfig(level=logging.INFO)

API_ID = int(os.environ["TG_API_ID"])
API_HASH = os.environ["TG_API_HASH"]
SESSION_STRING = os.environ["TG_SESSION"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

client = TelegramClient(
    StringSession(SESSION_STRING),
    API_ID,
    API_HASH,
)

async def ask_ai(message):
    def request_ai():
        payload = {
            "model": "openrouter/free",
            "messages": [
                {
                    "role": "system",
                    "content": (
                 
{
    "role": "system",
    "content": (
        "Пиши как обычный человек в личном чате. "
        "Отвечай естественно, разговорно и по делу, обычно коротко. "
        "Не используй эмодзи, смайлы, списки и официальные фразы. "
        "Не повторяй шаблоны вроде 'Чем могу помочь?' и 'Да, я понял'. "
        "Учитывай контекст переписки и отвечай именно на последнее сообщение. "
        "Не выдумывай факты о владельце аккаунта. "
        "Если собеседник предлагает встречаться, признаётся в любви, "
        "спрашивает о романтических чувствах или флиртует, мягко откажи. "
        "Например: 'Думаю, лучше останемся просто общаться' или "
        "'Ты хороший человек, но я не хочу отношений'. "
        "Если собеседник настаивает, спокойно повтори отказ без споров. "
        "Не обещай отношений и не говори от имени владельца о его чувствах. "
        "Если не хватает контекста, задай короткий уточняющий вопрос."
    ),
},

 ),
                },
                {"role": "user", "content": message},
            ],
        }

        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=45) as response:
            result = json.loads(response.read().decode("utf-8"))

        return result["choices"][0]["message"]["content"].strip()

    try:
        return await asyncio.to_thread(request_ai)
    except Exception:
        logging.exception("Ошибка запроса к ИИ")
        return "Спасибо за сообщение! Я отвечу тебе немного позже."

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

    reply = await ask_ai(message)
    await event.reply(reply)

async def main():
    await client.start()
    logging.info("Kissa AI AutoReply запущен")
    await client.run_until_disconnected()

if __name__ == "__main__":
    asyncio.run(main())
