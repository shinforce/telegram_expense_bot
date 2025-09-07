import os
import gspread
import logging
import sys
from datetime import datetime
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager

# --- CONFIGURATION (READS FROM ENVIRONMENT) ---
GOOGLE_SHEETS_CREDENTIALS = os.environ.get("GCP_CREDENTIALS_PATH", "family-expense-bot-471309-a2c7653d9602.json")
GOOGLE_SHEET_NAME = "Расходы"
GOOGLE_WORKSHEET_NAME = "expenses_log"
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
WEBHOOK_URL = os.environ.get("WEBHOOK_URL")
PORT = int(os.environ.get('PORT', 8443))

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

# --- GOOGLE SHEETS FUNCTION ---
def add_expense_to_sheet(description: str, amount: float, user_name: str, currency: str = "RSD"):
    """Adds a new row to the Google Sheet."""
    try:
        gc = gspread.service_account(filename=GOOGLE_SHEETS_CREDENTIALS)
        sh = gc.open(GOOGLE_SHEET_NAME).worksheet('expenses_log')  # Assumes you're using the first sheet
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
                await update.message.reply_text("Please provide a description before the amount.")
                return

            if add_expense_to_sheet(description, amount, user_name, currency):
                await update.message.reply_text(f"✅ Logged: '{description}' for {amount} {currency} (from {user.first_name}).")
            else:
                await update.message.reply_text("❌ Failed to log expense. Check the server logs.")
                
        except (ValueError, IndexError):
            logger.warning(f"Could not parse message: {text}")
            await update.message.reply_text(
                "Hmm, I didn't get that. Please use the format: `Description Amount` (e.g., `Coffee 500`) or `Amount Description` (e.g., `500 Coffee`)"
            )

# --- LIFESPAN EVENT HANDLER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events."""
    # Startup logic
    application.add_handler(message_handler)
    webhook_path = f"/{TELEGRAM_TOKEN}"
    full_webhook_url = f"{WEBHOOK_URL}{webhook_path}"
    await application.bot.set_webhook(url=full_webhook_url, allowed_updates=Update.ALL_TYPES)
    logger.info("Application started and webhook is set.")
    
    yield  # The application runs while the lifespan context is active
    
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
