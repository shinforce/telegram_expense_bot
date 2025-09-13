import os
import gspread
import logging
import sys
import re
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from telegram import Update, Bot
from telegram.request import HTTPXRequest
from telegram.ext import Application, MessageHandler, CommandHandler, filters
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
import httpx
from math import ceil
import asyncio

# --- CONFIGURATION ---
class Config:
    """Centralized configuration management"""
    
    GOOGLE_SHEETS_CREDENTIALS = os.environ.get("GCP_CREDENTIALS_PATH", "family-expense-bot-471309-a2c7653d9602.json")
    GOOGLE_SHEET_NAME = "Расходы"
    GOOGLE_WORKSHEET_NAME = "expenses_log"
    TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
    WEBHOOK_URL = os.environ.get("WEBHOOK_URL")
    OER_APP_ID = os.environ.get("OER_APP_ID")
    DELETE_MESSAGE_DELAY = 60
    
    # Validation limits
    MIN_AMOUNT = 0.01
    MAX_AMOUNT = 1_000_000
    MAX_DESCRIPTION_LENGTH = 200
    
    # Currency mappings
    CURRENCIES = {
        # EURO
        "EUR": "EUR", "EURO": "EUR", "ЕВРО": "EUR", "ЕВР": "EUR", "€": "EUR",
        # RUB
        "RUB": "RUB", "RUBL": "RUB", "РУБ": "RUB", "РУБЛ": "RUB", "РУБЛЕЙ": "RUB", "₽": "RUB",
        # USD
        "USD": "USD", "DOLLAR": "USD", "ДОЛЛАР": "USD", "$": "USD",
        # RSD (Serbian Dinar)
        "RSD": "RSD", "DIN": "RSD", "DINAR": "RSD", "ДИН": "RSD", "ДИНАР": "RSD"
    }
    
    DEFAULT_CURRENCY = "RSD"

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- DATA CLASSES ---
@dataclass
class Expense:
    """Represents a parsed expense"""
    description: str
    amount: float
    currency: str
    original_text: str
    user_name: str = ""
    user_id: int = 0
    
    def validate(self) -> Tuple[bool, str]:
        """Validate the expense data"""
        if not self.description or not self.description.strip():
            return False, "Description cannot be empty"
        
        if len(self.description) > Config.MAX_DESCRIPTION_LENGTH:
            return False, f"Description too long (max {Config.MAX_DESCRIPTION_LENGTH} characters)"
        
        if self.amount <= Config.MIN_AMOUNT:
            return False, f"Amount must be greater than {Config.MIN_AMOUNT}"
        
        if self.amount > Config.MAX_AMOUNT:
            return False, f"Amount seems too large (max {Config.MAX_AMOUNT}). Please check."
        
        if self.currency not in ["EUR", "RUB", "USD", "RSD"]:
            return False, f"Unknown currency: {self.currency}"
        
        return True, ""

@dataclass
class ParseResult:
    """Result of expense parsing"""
    success: bool
    expense: Optional[Expense] = None
    error_message: str = ""

# --- EXPENSE PARSER ---
class ExpenseParser:
    """Handles parsing of expense messages with multiple formats"""
    
    # Regex patterns for different expense formats
    PATTERNS = [
        # Format: "100.50 EUR Coffee with friends"
        (r'^(\d+(?:[.,]\d+)?)\s+([A-Z€$₽]{1,6})\s+(.+)$', ['amount', 'currency', 'description']),
        # Format: "100.50 Coffee with friends EUR"
        (r'^(\d+(?:[.,]\d+)?)\s+(.+?)\s+([A-Z€$₽]{1,6})$', ['amount', 'description', 'currency']),
        # Format: "Coffee with friends 100.50 EUR"
        (r'^(.+?)\s+(\d+(?:[.,]\d+)?)\s+([A-Z€$₽]{1,6})$', ['description', 'amount', 'currency']),
        # Format: "100.50 Coffee" (no currency)
        (r'^(\d+(?:[.,]\d+)?)\s+(.+)$', ['amount', 'description']),
        # Format: "Coffee 100.50" (no currency)
        (r'^(.+?)\s+(\d+(?:[.,]\d+)?)$', ['description', 'amount']),
    ]
    
    @classmethod
    def parse_line(cls, text: str) -> ParseResult:
        """Parse a single line of expense text"""
        if not text or not text.strip():
            return ParseResult(False, error_message="Empty text")
        
        # Clean and prepare text
        text = text.strip()
        original_text = text
        text_upper = text.upper()
        
        # Try each pattern
        for pattern, field_order in cls.PATTERNS:
            match = re.match(pattern, text_upper)
            if match:
                try:
                    # Extract values based on field order
                    values = {}
                    for i, field in enumerate(field_order):
                        values[field] = match.group(i + 1)
                    
                    # Parse amount (handle both . and , as decimal separator)
                    amount_str = values.get('amount', '0')
                    amount_str = amount_str.replace(',', '.')
                    amount = float(amount_str)
                    
                    # Get currency (default to RSD if not specified)
                    currency_str = values.get('currency', Config.DEFAULT_CURRENCY)
                    currency = Config.CURRENCIES.get(currency_str, currency_str)
                    
                    # Get description
                    description = values.get('description', '').strip()
                    
                    # Create expense object
                    expense = Expense(
                        description=description,
                        amount=amount,
                        currency=currency,
                        original_text=original_text
                    )
                    
                    # Validate
                    is_valid, error_msg = expense.validate()
                    if not is_valid:
                        return ParseResult(False, error_message=error_msg)
                    
                    return ParseResult(True, expense=expense)
                    
                except (ValueError, KeyError) as e:
                    logger.debug(f"Pattern matched but parsing failed: {e}")
                    continue
        
        return ParseResult(False, error_message="Could not parse expense format")
    
    @classmethod
    def parse_message(cls, text: str) -> List[ParseResult]:
        """Parse a message that may contain multiple expense lines"""
        lines = text.split('\n')
        results = []
        
        for line in lines:
            line = line.strip()
            if line:  # Skip empty lines
                result = cls.parse_line(line)
                results.append(result)
        
        return results

# --- CURRENCY CONVERTER ---
class CurrencyConverter:
    """Handles currency conversion with caching capabilities"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.cache = {}
        self.cache_timestamp = None
        self.cache_duration = 3600  # 1 hour in seconds
    
    async def get_rates(self) -> Optional[Dict[str, float]]:
        """Fetch exchange rates with caching"""
        current_time = datetime.now()
        
        # Check if cache is valid
        if self.cache and self.cache_timestamp:
            time_diff = (current_time - self.cache_timestamp).total_seconds()
            if time_diff < self.cache_duration:
                logger.info("Using cached exchange rates")
                return self.cache
        
        # Fetch new rates
        try:
            api_url = f"https://openexchangerates.org/api/latest.json?app_id={self.api_key}"
            async with httpx.AsyncClient() as client:
                response = await client.get(api_url, timeout=10.0)
                response.raise_for_status()
                data = response.json()
                self.cache = data["rates"]
                self.cache_timestamp = current_time
                logger.info("Fetched fresh exchange rates")
                return self.cache
        except Exception as e:
            logger.error(f"Failed to fetch exchange rates: {e}")
            # Return cached rates if available, even if expired
            if self.cache:
                logger.warning("Using expired cache due to API failure")
                return self.cache
            return None
    
    async def convert(self, amount: float, from_currency: str, to_currency: str = "RSD") -> Optional[float]:
        """Convert amount between currencies"""
        if from_currency == to_currency:
            return amount
        
        rates = await self.get_rates()
        if not rates:
            return None
        
        try:
            # Check if currencies exist in rates
            if from_currency not in rates or to_currency not in rates:
                logger.error(f"Currency not found: {from_currency} or {to_currency}")
                return None
            
            # Convert through USD (base currency)
            amount_in_usd = amount / rates[from_currency]
            converted_amount = amount_in_usd * rates[to_currency]
            
            return ceil(converted_amount)
        except Exception as e:
            logger.error(f"Conversion calculation failed: {e}")
            return None

# --- GOOGLE SHEETS MANAGER ---
class GoogleSheetsManager:
    """Manages Google Sheets operations"""
    
    def __init__(self, credentials_file: str, sheet_name: str, worksheet_name: str):
        self.credentials_file = credentials_file
        self.sheet_name = sheet_name
        self.worksheet_name = worksheet_name
        self._worksheet = None
    
    def _get_worksheet(self):
        """Get or create worksheet connection"""
        try:
            if self._worksheet is None:
                gc = gspread.service_account(filename=self.credentials_file)
                sh = gc.open(self.sheet_name)
                self._worksheet = sh.worksheet(self.worksheet_name)
            return self._worksheet
        except Exception as e:
            logger.error(f"Failed to connect to Google Sheets: {e}")
            self._worksheet = None  # Reset on error
            return None
    
    def add_expense(self, expense: Expense, amount_rsd: float) -> bool:
        """Add expense to Google Sheet"""
        try:
            worksheet = self._get_worksheet()
            if worksheet is None:
                return False
            
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            row = [
                now,
                expense.description,
                amount_rsd,
                expense.user_name,
                expense.currency,  # Original currency
                expense.amount     # Original amount
            ]
            
            worksheet.append_row(row)
            logger.info(f"Added to sheet: {row}")
            return True
            
        except Exception as e:
            logger.error(f"Error writing to Google Sheets: {e}")
            self._worksheet = None  # Reset connection on error
            return False

# --- MESSAGE MANAGER ---
class MessageManager:
    """Handles message sending and deletion"""
    
    def __init__(self, delete_delay: int = 60):
        self.delete_delay = delete_delay
        self.pending_deletions = []
    
    async def delete_message_after_delay(self, bot: Bot, chat_id: int, message_id: int):
        """Delete a message after specified delay"""
        await asyncio.sleep(self.delete_delay)
        try:
            await bot.delete_message(chat_id=chat_id, message_id=message_id)
            logger.info(f"Deleted message {message_id} from chat {chat_id}")
        except Exception as e:
            logger.warning(f"Could not delete message {message_id}: {e}")
    
    async def send_temp_message(self, update: Update, context, text: str, parse_mode: str = None):
        """Send a message and schedule it for deletion"""
        sent_message = await update.message.reply_text(text, parse_mode=parse_mode)
        
        # Schedule deletion
        asyncio.create_task(
            self.delete_message_after_delay(
                bot=context.bot,
                chat_id=sent_message.chat_id,
                message_id=sent_message.message_id
            )
        )
        
        return sent_message

# --- BOT HANDLER ---
class ExpenseBotHandler:
    """Main bot logic handler"""
    
    def __init__(self):
        self.parser = ExpenseParser()
        self.converter = CurrencyConverter(Config.OER_APP_ID)
        self.sheets_manager = GoogleSheetsManager(
            Config.GOOGLE_SHEETS_CREDENTIALS,
            Config.GOOGLE_SHEET_NAME,
            Config.GOOGLE_WORKSHEET_NAME
        )
        self.message_manager = MessageManager(Config.DELETE_MESSAGE_DELAY)
    
    async def handle_start_command(self, update: Update, context):
        """Handle /start command"""
        welcome_message = (
            "👋 Hi! I'm your expense tracking bot.\n\n"
            "Send me expenses in formats like:\n"
            "• `500 Coffee`\n"
            "• `Coffee 500`\n"
            "• `15.50 EUR Lunch`\n"
            "• `Taxi 1200 RSD`\n\n"
            "You can send multiple expenses at once (one per line)."
        )
        await update.message.reply_text(welcome_message, parse_mode='Markdown')
    
    async def handle_message(self, update: Update, context):
        """Process expense messages"""
        user = update.effective_user
        text = update.message.text
        
        logger.info(f"Received message from {user.first_name} ({user.id}): {text}")
        
        # Parse all expenses in the message
        results = self.parser.parse_message(text)
        
        if not results:
            await self.message_manager.send_temp_message(
                update, context,
                "❓ Please send an expense in format: `Amount Description` or `Description Amount`",
                parse_mode='Markdown'
            )
            return
        
        # Process each parsed expense
        success_count = 0
        error_messages = []
        
        for i, result in enumerate(results, 1):
            if not result.success:
                error_messages.append(f"Line {i}: {result.error_message}")
                continue
            
            expense = result.expense
            expense.user_name = user.full_name
            expense.user_id = user.id
            
            # Convert currency if needed
            conversion_note = ""
            if expense.currency != Config.DEFAULT_CURRENCY:
                rsd_amount = await self.converter.convert(
                    expense.amount, 
                    expense.currency, 
                    Config.DEFAULT_CURRENCY
                )
                
                if rsd_amount is None:
                    error_messages.append(
                        f"Line {i}: Failed to convert {expense.amount} {expense.currency}"
                    )
                    continue
                
                conversion_note = f" (converted from {expense.amount} {expense.currency})"
            else:
                rsd_amount = expense.amount
            
            # Add to Google Sheets
            if self.sheets_manager.add_expense(expense, rsd_amount):
                success_count += 1
                
                # Send success message
                success_msg = (
                    f"✅ Logged: *{expense.description}*\n"
                    f"Amount: {rsd_amount} RSD{conversion_note}\n"
                    f"By: {user.first_name}"
                )
                await self.message_manager.send_temp_message(
                    update, context, success_msg, parse_mode='Markdown'
                )
            else:
                error_messages.append(f"Line {i}: Failed to save to sheet")
        
        # Report any errors
        if error_messages:
            error_report = "❌ *Errors:*\n" + "\n".join(error_messages)
            await self.message_manager.send_temp_message(
                update, context, error_report, parse_mode='Markdown'
            )

# --- APPLICATION SETUP ---
# Initialize bot application
httpx_request = HTTPXRequest(
    connect_timeout=10.0,
    read_timeout=10.0,
    pool_timeout=10.0
)
application = Application.builder().token(Config.TELEGRAM_TOKEN).request(httpx_request).build()

# Initialize bot handler
bot_handler = ExpenseBotHandler()

# --- LIFESPAN MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    await application.initialize()
    
    # Register command handlers
    application.add_handler(
        CommandHandler("start", bot_handler.handle_start_command)
    )
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, bot_handler.handle_message)
    )
    
    # Setup webhook
    webhook_path = f"/{Config.TELEGRAM_TOKEN}"
    full_webhook_url = f"{Config.WEBHOOK_URL}{webhook_path}"
    await application.bot.set_webhook(url=full_webhook_url, allowed_updates=Update.ALL_TYPES)
    logger.info(f"Application started, webhook set to {full_webhook_url}")
    
    yield  # Application runs
    
    # Shutdown
    await application.shutdown()
    await application.bot.delete_webhook()
    logger.info("Webhook deleted, application shut down.")

# --- FASTAPI APPLICATION ---
app = FastAPI(lifespan=lifespan)

@app.api_route("/", methods=["GET", "HEAD"])
def health_check():
    """Health check endpoint for monitoring"""
    return {"status": "ok", "service": "expense-bot"}

@app.post("/{token}")
async def process_update(token: str, request: Request):
    """Process incoming Telegram updates"""
    if token != Config.TELEGRAM_TOKEN:
        logger.warning(f"Invalid token received: {token[:10]}...")
        return {"status": "invalid token"}
    
    try:
        update = Update.de_json(await request.json(), application.bot)
        await application.process_update(update)
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Error processing update: {e}")
        return {"status": "error", "message": str(e)}
