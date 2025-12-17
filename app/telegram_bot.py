from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    CommandHandler,
    ContextTypes,
    filters,
)

from app.handler import handle_message


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        """
        Привет! Я AI-ассистент для анализа данных и Ad-hoc задач.

    Я могу помочь с:
    • продуктовой аналитикой
    • автоматизацией аналитических процессов
    • визуализацией
    • репортингом

    Чем я могу быть полезен тебе сегодня? 🙂
        """
    )


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text

    answer = handle_message(user_text)

    if answer["type"] == "image":
        with open(answer["image_path"], "rb") as photo:
            await update.message.reply_photo(photo=photo)


def run_bot(token: str):
    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", start))

    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, on_message)
    )

    app.run_polling()
