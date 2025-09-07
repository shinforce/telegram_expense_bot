import os
import gspread
import logging
import sys
from datetime import datetime
from telegram import Update, Bot
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
import httpx
from math import ceil
import asyncio

# --- CONFIGURATION (READS FROM ENVIRONMENT) ---
GOOGLE_SHEETS_CREDENTIALS = os.environ.get("GCP_CREDENTIALS_PATH", "family-expense-bot-471309-a2c7653d9602.json")
GOOGLE_SHEET_NAME = "Расходы"
GOOGLE_WORKSHEET_NAME = "expenses_log"
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
WEBHOOK_URL = os.environ.get("WEBHOOK_URL")
OER_APP_ID = os.environ.get("OER_APP_ID")
DELETE_MESSAGE_DELAY = 60 

CURRENCIES = {
    #EURO
    "EUR": "EUR", 
    "EURO": "EUR", 
    "ЕВРО": "EUR",
    "ЕВР": "EUR",

    #RUB
    "RUB": "RUB", 
    "RUBL": "RUB", 
    "РУБ": "RUB", 
    "РУБЛ": "RUB", 
    "РУБЛЕЙ": "RUB"
}

# --- DELAYED DELETION FUNCTION ---
async def delete_message_after_delay(bot: Bot, chat_id: int, message_id: int, delay: int):
    """Waits for a delay and then deletes a specific message."""
    await asyncio.sleep(delay)
    try:
        await bot.delete_message(chat_id=chat_id, message_id=message_id)
        logger.info(f"Deleted confirmation message {message_id} from chat {chat_id}.")
    except Exception as e:
        logger.warning(f"Could not delete message {message_id}: {e}")

async def reply_and_schedule_delete(update: Update, context, text: str):
    """Sends a reply and immediately schedules it for deletion."""
    sent_message = await update.message.reply_text(text)
    asyncio.create_task(
        delete_message_after_delay(
            bot=context.bot,
            chat_id=sent_message.chat_id,
            message_id=sent_message.message_id,
            delay=DELETE_MESSAGE_DELAY
        )
    )

# --- SETUP LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- SETUP BOT APPLICATION ---
# Create the Application instance once
application = Application.builder().token(TELEGRAM_TOKEN).build()

# --- CURRENCY CONVERSION FUNCTION ---
async def convert_currency(amount: float, from_currency: str, to_currency: str = "RSD") -> float | None:
    """Converts an amount from one currency to another using the Open Exchange Rates API."""
    if from_currency == to_currency:
        return amount

    api_url = f"https://openexchangerates.org/api/latest.json?app_id={OER_APP_ID}"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(api_url)
            response.raise_for_status()  # Raises an exception for bad responses (4xx or 5xx)
            data = response.json()
            rates = data["rates"]

        # Convert from the source currency to the base currency (USD)
        amount_in_usd = amount / rates[from_currency]
        
        # Convert from the base currency (USD) to the target currency
        converted_amount = amount_in_usd * rates[to_currency]
        
        return ceil(converted_amount)
    except Exception as e:
        logger.error(f"Currency conversion failed: {e}")
        return None

# --- GOOGLE SHEETS FUNCTION ---
def add_expense_to_sheet(description: str, amount: float, user_name: str):
    """Adds a new row to the Google Sheet."""
    try:
        gc = gspread.service_account(filename=GOOGLE_SHEETS_CREDENTIALS)
        sh = gc.open(GOOGLE_SHEET_NAME).worksheet('expenses_log')  
        now = datetime.now().strftime("%Y-%m-%d") # %H:%M:%S")
        row_to_add = [now, description, amount, user_name]
        sh.append_row(row_to_add)
        logger.info(f"Added to sheet: {row_to_add}")
        return True
    except Exception as e:
        logger.error(f"Error writing to Google Sheets: {e}")
        return False

# --- TELEGRAM BOT HANDLERS ---
async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """A simple command to check if the bot is alive."""
    await update.message.reply_text("Hi! I'm alive and ready to log expenses.")

async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Parses messages and calls the sheet function."""
    user = update.effective_user
    text = update.message.text
    
    logger.info(f"Received message from {user.first_name}: {text}")
    
    text = text.upper().split('\n')
    
    for text in text:
        currency = "RSD"
        conversion_message = ''
        try:
            parts = text.strip().split()
            if parts[0].isnumeric():
                amount = int(parts[0])
                if parts[1] in CURRENCIES.keys():
                    currency = CURRENCIES[parts[1]]
                    description = " ".join(parts[2:])
                else:
                    description = " ".join(parts[1:])
            else:                
                if parts[-1] in CURRENCIES.keys():
                    amount = int(parts[-2])
                    description = " ".join(parts[:-2])
                    currency = CURRENCIES[parts[-1]]
                else:
                    amount = int(parts[-1])
                    description = " ".join(parts[:-1])
            
            user_name = user.full_name
            
            if not description:
                await reply_and_schedule_delete(update, context, "Please provide a description before the amount.")
                return

            rsd_amount = await convert_currency(amount, currency, "RSD")
            
            if rsd_amount is None:
                await reply_and_schedule_delete(update, context, f"❌ Could not convert {original_currency} to RSD. Expense not logged.")
                continue

            if currency != 'RSD':
                conversion_message = f' (converted {amount} {currency})'

            if add_expense_to_sheet(description, rsd_amount, user_name):
                reply_text = f"✅ Logged: '{description}' for {rsd_amount} RSD{conversion_message} from {user.first_name}."
                sent_message = await update.message.reply_text(reply_text)
                await reply_and_schedule_delete(update, context, reply_text)
            else:
                await reply_and_schedule_delete(update, context, "❌ Failed to log expense. Check the server logs.")
                
        except (ValueError, IndexError):
            logger.warning(f"Could not parse message: {text}")
            await reply_and_schedule_delete(update, context, "Hmm, I didn't get that. Please use the format: `Description Amount` (e.g., `Coffee 500`) or `Amount Description` (e.g., `500 Coffee`)")

# --- LIFESPAN EVENT HANDLER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events."""
    # Startup logic
    # CORRECTED: Create handler objects and then add them
    start_command_handler = CommandHandler("start", start_handler)
    expense_message_handler = MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler)
    application.add_handler(start_command_handler)
    application.add_handler(expense_message_handler)

    webhook_path = f"/{TELEGRAM_TOKEN}"
    full_webhook_url = f"{WEBHOOK_URL}{webhook_path}"
    await application.bot.set_webhook(url=full_webhook_url, allowed_updates=Update.ALL_TYPES)
    logger.info("Application started and webhook is set.")

    yield  # The application runs

    # Shutdown logic
    await application.bot.delete_webhook()
    logger.info("Webhook deleted.")

# --- SETUP WEB SERVER using FastAPI ---
app = FastAPI(lifespan=lifespan)

@app.get("/")
def health_check():
    """This endpoint is for Render's health checks."""
    return {"status": "ok"}

@app.post("/{token}")
async def process_update(token: str, request: Request):
    """This endpoint receives updates from Telegram."""
    if token != TELEGRAM_TOKEN:
        return {"status": "invalid token"}

    async with application:
        update = Update.de_json(await request.json(), application.bot)
        await application.process_update(update)

    return {"status": "ok"}
