import os
import re
import json
import asyncio
import logging
import urllib.request
import urllib.error

from telethon import TelegramClient, events


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

API_ID = int(os.environ["TG_API_ID"])
API_HASH = os.environ["TG_API_HASH"]
TG_SESSION = os.environ["TG_SESSION"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

MODEL = "openrouter/free"

client = TelegramClient(
    "kissa_session",
    API_ID,
    API_HASH
)

SYSTEM_PROMPT = """
Ты отвечаешь на личные сообщения Telegram от имени владельца аккаунта.
Пиши как обычный человек в непринуждённой переписке.

ОСНОВНЫЕ ПРАВИЛА:
- Отвечай на русском, если собеседник пишет по-русски.
- Пиши естественно, коротко и по смыслу сообщения.
- Обычно достаточно одного короткого предложения.
- Учитывай всю предоставленную историю переписки.
- Понимай короткие сообщения по контексту предыдущих реплик.
- Если собеседник отвечает «хочешь?», «и что?», «почему?» или похожим образом, выясни смысл из предыдущих сообщений.
- Не повторяй постоянно одну и ту же фразу.
- Не используй эмодзи, смайлики и стикерные выражения.
- Не используй шаблонные фразы вроде «Понимаю тебя», «Интересно», «Хороший вопрос», если они неуместны.
- Никогда не говори «я не знаю, что ответить», «мне нечего сказать» или похожие отговорки.
- Если сообщение непонятное, задай короткий естественный уточняющий вопрос или отреагируй на понятную часть.
- Не придумывай факты о владельце аккаунта, его жизни, чувствах, планах или местонахождении.
- Не утверждай, что сделал что-либо в реальном мире, если этого не было в переписке.
- Не рассказывай, что ты ИИ, бот или языковая модель.
- Возвращай только текст сообщения, без кавычек, пояснений и вариантов ответа.

ОТНОШЕНИЯ И ВСТРЕЧИ:
- Если собеседник предлагает встречаться, встречу, свидание или признаётся в любви, отвечай мягким, вежливым, но уверенным отказом.
- Не соглашайся на отношения или встречу.
- Не флиртуй в ответ и не давай ложной надежды.
- Формулируй отказ по-разному, чтобы он звучал естественно, но сохраняй одну позицию.
- Если собеседник повторяет предложение, спокойно подтверди отказ, не меняя позицию.

ОСКОРБЛЕНИЯ:
- Если сообщение состоит только из оскорбления, обычно отвечай «ок».
- Если оскорбление встречается внутри вопроса или просьбы, отвечай по существу вопроса или просьбы.
- Не вступай в перепалку и не оскорбляй в ответ.

ПРИВЕТСТВИЯ:
- На «привет» отвечай простым приветствием.
- Не добавляй ненужные вопросы и длинные приветствия.
"""


def is_standalone_insult(text):
    normalized = re.sub(
        r"[^\w\s]",
        "",
        text.lower()
    ).strip()

    insults = {
        "даун",
        "дебил",
        "идиот",
        "дурак",
        "тупой",
        "тупица",
        "хуесос",
        "мудак",
        "долбоеб",
        "долбаеб",
        "уебок",
        "ублюдок",
        "кретин",
        "лох",
        "чмо",
    }

    return normalized in insults


def request_ai(history):
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            *history
        ],
        "temperature": 0.8,
        "max_tokens": 120
    }

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/KissaPlugins/kissa-autoreply",
            "X-Title": "Kissa AI AutoReply"
        },
        method="POST"
    )

    with urllib.request.urlopen(request, timeout=45) as response:
        result = json.loads(response.read().decode("utf-8"))

    return result["choices"][0]["message"]["content"].strip()


async def get_chat_history(event, limit=25):
    messages = await client.get_messages(
        event.chat_id,
        limit=limit
    )

    messages = list(reversed(messages))
    history = []

    for message in messages:
        if not message.message or not message.message.strip():
            continue

        history.append({
            "role": "assistant" if message.out else "user",
            "content": message.message.strip()
        })

    return history


async def ask_ai(history):
    try:
        answer = await asyncio.to_thread(
            request_ai,
            history
        )

        answer = answer.strip()

        bad_phrases = [
            "не знаю, что ответить",
            "я не знаю, что ответить",
            "не знаю что ответить",
            "мне нечего ответить",
            "не могу придумать ответ",
            "не знаю, что сказать",
            "я не знаю, что сказать",
        ]

        if not answer or any(
            phrase in answer.lower()
            for phrase in bad_phrases
        ):
            retry_history = history + [{
                "role": "user",
                "content": (
                    "Сгенерируй другой ответ с учётом контекста. "
                    "Ответь естественно и коротко, как в обычной переписке. "
                    "Не используй отговорки и шаблонные фразы. "
                    "Верни только текст ответа."
                )
            }]

            answer = await asyncio.to_thread(
                request_ai,
                retry_history
            )

        answer = answer.strip()

        if not answer:
            return "А ты как думаешь?"

        return answer

    except Exception:
        logging.exception("Ошибка запроса к ИИ")
        return None


@client.on(events.NewMessage(incoming=True))
async def handle_message(event):
    try:
        if not event.is_private:
            return

        text = (event.raw_text or "").strip()

        if not text:
            return

        if is_standalone_insult(text):
            await event.reply("ок")
            return

        history = await get_chat_history(event)

        if not history:
            history = [{
                "role": "user",
                "content": text
            }]

        answer = await ask_ai(history)

        if not answer:
            logging.warning(
                "Ответ не отправлен: ИИ временно недоступен."
            )
            return

        await event.reply(answer)

        logging.info(
            "Ответ отправлен в личный чат %s",
            event.chat_id
        )

    except Exception:
        logging.exception("Ошибка обработки сообщения")


async def main():
    await client.start()
    logging.info("Kissa AI AutoReply запущен.")
    await client.run_until_disconnected()


if __name__ == "__main__":
    asyncio.run(main())

