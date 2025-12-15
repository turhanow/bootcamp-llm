from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    ContextTypes,
    filters,
)
from telegram import Update
from app.handler import handle_message


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text

    answer = handle_message(user_text)

    await update.message.reply_text(answer)


def run_bot(token: str):
    app = ApplicationBuilder().token(token).build()

    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, on_message)
    )

    app.run_polling()
