import os
import gspread
import logging
import sys
from datetime import datetime
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes
import asyncio

# --- CONFIGURATION (READS FROM ENVIRONMENT) ---
GOOGLE_SHEETS_CREDENTIALS = os.environ.get("GCP_CREDENTIALS_PATH", "family-expense-bot-471309-a2c7653d9602.json")
GOOGLE_SHEET_NAME = "Расходы"
GOOGLE_WORKSHEET_NAME = "expenses_log"
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
WEBHOOK_URL = os.environ.get("WEBHOOK_URL")
PORT = int(os.environ.get('PORT', 8443))

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

# --- GOOGLE SHEETS FUNCTION ---
def add_expense_to_sheet(description: str, amount: float, user_name: str):
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
    
    try:
        parts = text.strip().split()
        if parts[0].isnumeric():
            amount = int(parts[0])
            description = " ".join(parts[1:])
        else: 
            amount = int(parts[-1])
            description = " ".join(parts[:-1])
        
        user_name = user.full_name
        
        if not description:
            await update.message.reply_text("Please provide a description before the amount.")
            return

        if add_expense_to_sheet(description, amount, user_name):
            await update.message.reply_text(f"✅ Logged: '{description}' for {amount} RSD (from {user.first_name}).")
        else:
            await update.message.reply_text("❌ Failed to log expense. Check the server logs.")
            
    except (ValueError, IndexError):
        logger.warning(f"Could not parse message: {text}")
        await update.message.reply_text(
            "Hmm, I didn't get that. Please use the format: `Description Amount` (e.g., `Coffee 500`) or `Amount Description` (e.g., `500 Coffee`)"
        )

# --- MAIN APPLICATION LOGIC ---
async def main() -> None:
    """Set up and run the bot."""
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Add command and message handlers
    application.add_handler(CommandHandler("start", start_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))

    # We use the token as a secret path to make our webhook URL unique and hard to guess.
    url_path = f"/{TELEGRAM_TOKEN}"
    full_webhook_url = f"{WEBHOOK_URL}{url_path}"

    # 1. Initialize the Application
    await application.initialize()
    
    # 2. Start the webhook server and set the webhook
    #    This single call handles everything: setting the URL with Telegram
    #    and starting the local server to listen for updates.
    await application.updater.start_webhook(
        listen="0.0.0.0",
        port=PORT,
        url_path=url_path,
        webhook_url=full_webhook_url
    )

    # 3. Start the bot's update processing
    await application.start()
    
    logger.info("Bot is running and webhook is set.")
    
    # 4. Keep the script running
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(main())
