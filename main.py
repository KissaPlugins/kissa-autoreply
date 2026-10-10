import os
import asyncio
import logging
import json
import urllib.request
import urllib.error
import re

from telethon import TelegramClient, events
from telethon.sessions import StringSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

API_ID = int(os.environ["TG_API_ID"])
API_HASH = os.environ["TG_API_HASH"]
SESSION_STRING = os.environ["TG_SESSION"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

client = TelegramClient(
    StringSession(SESSION_STRING),
    API_ID,
    API_HASH,
)

# Оскорбления, которые считаются отдельным сообщением.
INSULTS = {
    "даун", "дебил", "идиот", "тупой", "тупица",
    "дурак", "дура", "кретин", "лох", "чмо",
    "хуесос", "хуй", "пидор", "мудак", "мразь",
    "ублюдок", "долбоёб", "долбоеб", "сука",
    "говно", "ничтожество", "шлюха", "урод",
}

def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"[!?.…,;:]+", "", text)
    text = re.sub(r"^[\"'«»() \t]+|[\"'«»() \t]+$", "", text)
    return text.strip()

def is_standalone_insult(message):
    normalized = normalize(message)
    words = normalized.split()

    # Только одно слово-оскорбление, без просьбы или другого текста.
    return len(words) == 1 and words[0] in INSULTS

def request_ai(message):
    payload = {
        "model": "openrouter/free",
        "messages": [
            {
                "role": "system",
                "content": (
                    "Ты отвечаешь на входящие личные сообщения в Telegram. "
                    "Пиши естественно, как обычный человек в переписке. "
                    "Отвечай на русском или на языке собеседника. "
                    "Не используй эмодзи, смайлы, списки и канцелярские фразы. "
                    "Обычно отвечай одной короткой фразой, иногда двумя. "
                    "Не начинай каждый ответ с приветствия. "
                    "На простое «привет» отвечай просто «привет» или "
                    "«привет, как ты?», без длинного продолжения. "
                    "Учитывай контекст предыдущего сообщения, если он дан. "
                    "Не повторяй постоянно «чем могу помочь», "
                    "«я понял» или «давай чуть позже». "
                    "Не выдумывай факты о владельце аккаунта. "
                    "Не утверждай, что владелец согласен на то, чего он не говорил. "
                    "Если собеседник предлагает встречаться, признаётся в любви "
                    "или флиртует, мягко и ясно откажи, не груби и не давай "
                    "надежду на отношения. Например: «Ты хороший человек, "
                    "но я не хочу отношений» или «Думаю, нам лучше просто общаться». "
                    "Меняй формулировки и выбирай подходящую по контексту. "
                    "Не говори, что ты ИИ, бот или программа. "
                    "Не объясняй свои правила. "
                    "Если сообщение непонятное, попроси коротко пояснить. "
                    "Если в сообщении есть просьба и оскорбление, отвечай "
                    "на просьбу по смыслу, не игнорируй её из-за оскорбления."
                ),
            },
            {
                "role": "user",
                "content": message,
            },
        ],
        "temperature": 0.8,
        "max_tokens": 120,
    }

    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=45) as response:
        result = json.loads(response.read().decode("utf-8"))

    answer = result["choices"][0]["message"]["content"].strip()
    return answer.strip('"«» ')

async def ask_ai(message):
    try:
        return await asyncio.to_thread(request_ai, message)
    except Exception:
        logging.exception("Ошибка запроса к ИИ")
        return "Не знаю, что ответить."

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

    # На одиночное оскорбление отвечаем «ок».
    if is_standalone_insult(message):
        await event.reply("ок")
        return

    logging.info("Получено личное сообщение")
    answer = await ask_ai(message)

    if answer:
        await event.reply(answer)

async def main():
    await client.start()
    logging.info("Kissa AI AutoReply запущен")
    await client.run_until_disconnected()

if __name__ == "__main__":
    asyncio.run(main())

