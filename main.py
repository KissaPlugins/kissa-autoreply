import os
import asyncio
import json
import logging
import urllib.request
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
    API_HASH
)

async def ask_ai(message, history):
    def request_ai():
        messages = [
            {
                "role": "system",
                "content": (
                    "Ты отвечаешь за владельца личного Telegram-аккаунта. "
                    "Пиши как обычный человек в непринуждённой переписке. "
                    "Отвечай на последнее сообщение с учётом контекста. "
                    "Если тебе пишут «пр», «привет» или «здарова», "
                    "ответь коротким приветствием, например «Привет». "
                    "Не задавай лишних вопросов и не пиши «Чем могу помочь?», "
                    "если собеседник просто поздоровался. "
                    "Не используй эмодзи, смайлы, списки и официальные фразы. "
                    "Не говори, что ты ИИ, бот или программа. "
                    "Не выдумывай личные факты, обещания и планы владельца. "
                    "Если предлагают встречаться, признаются в любви "
                    "или флиртуют, мягко и однозначно откажи, не груби. "
                    "Например: «Ты хороший человек, но я не хочу отношений». "
                    "Не соглашайся на отношения и встречи. "
                    "Если настаивают, спокойно повтори отказ. "
                    "Отвечай обычно, коротко, чаще одним предложением. "
                    "Не повторяй постоянно одни и те же фразы. "
                    "Если сообщение непонятно, попроси пояснить коротко."
                )
            }
        ]

        messages.extend(history[-10:])
        messages.append({"role": "user", "content": message})

        payload = {
            "model": "openrouter/free",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 120
        }

        request = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            method="POST"
        )

        with urllib.request.urlopen(request, timeout=45) as response:
            result = json.loads(response.read().decode("utf-8"))

        return result["choices"][0]["message"]["content"].strip()

    try:
        return await asyncio.to_thread(request_ai)
    except Exception:
        logging.exception("Ошибка запроса к ИИ")
        return "Не знаю, что ответить. Давай чуть позже."


history_by_user = {}

@client.on(events.NewMessage(incoming=True))
async def auto_reply(event):
    if not event.is_private or event.out:
        return

    sender = await event.get_sender()
    if getattr(sender, "bot", False):
        return

    message = (event.raw_text or "").strip()
    if not message:
        return

    user_id = event.sender_id
    history = history_by_user.setdefault(user_id, [])

    reply = await ask_ai(message, history)

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    history_by_user[user_id] = history[-20:]

    await event.reply(reply)


async def main():
    await client.start()
    logging.info("Kissa AI AutoReply запущен")
    await client.run_until_disconnected()


if __name__ == "__main__":
    asyncio.run(main())

