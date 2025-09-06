import os
from flask import Flask
import threading
import gspread
import logging
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes
from datetime import datetime

GOOGLE_SHEETS_CREDENTIALS = os.environ.get("GCP_CREDENTIALS_PATH", "credentials.json")
GOOGLE_SHEET_NAME = "Расходы"
GOOGLE_WORKSHEET_NAME = "expenses_log"
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")

gc = gspread.service_account(filename='family-expense-bot-471309-a2c7653d9602.json')

def add_expense_to_sheet(description: str, amount: float, user_name: str):
    """Adds a new row to the Google Sheet."""
    try:
        gc = gspread.service_account(filename=GOOGLE_SHEETS_CREDENTIALS)
        sh = gc.open(GOOGLE_SHEET_NAME).worksheet('expenses_log')  # Assumes you're using the first sheet
        
        # Get current date and time
        now = datetime.now().strftime("%Y-%m-%d") # %H:%M:%S")
        
        row_to_add = [now, description, amount, user_name]
        sh.append_row(row_to_add)
        logging.info(f"Added to sheet: {row_to_add}")
        return True
    except Exception as e:
        logging.error(f"Error writing to Google Sheets: {e}")
        return False

# --- TELEGRAM BOT HANDLERS ---
async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """A simple command to check if the bot is alive."""
    await update.message.reply_text("Hi! I'm alive and ready to log expenses.")

async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Parses messages and calls the sheet function."""
    user = update.effective_user
    text = update.message.text
    
    logging.info(f"Received message from {user.first_name}: {text}")
    
    # Simple parsing: splits the message into description and amount
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
        # This handles cases where the message isn't in the expected format
        logging.warning(f"Could not parse message: {text}")
        await update.message.reply_text(
            "Hmm, I didn't get that. Please use the format: `Description Amount` (e.g., `Coffee 500`) or `Amount Description` (e.g., `500 Coffee`)"
        )

app = Flask(__name__)

@app.route('/')
def hello():
    """A simple endpoint to check if the app is running."""
    return "Hello, the bot is alive!"
    
def run_bot():
    """The function that contains your bot's starting logic."""
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    application.add_handler(CommandHandler("start", start_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))
    
    logger.info("Bot is starting polling...")
    application.run_polling()

def main():
    """Starts the Telegram bot."""
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # This handler will respond to all text messages
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))
    
    logging.info("Bot is starting...")
    application.run_polling()

if __name__ == "__main__":
    #main()
    # Start the bot in a separate thread
    bot_thread = threading.Thread(target=run_bot)
    bot_thread.start()
    
    # Run the Flask app
    # Render will provide the PORT environment variable
    #import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
