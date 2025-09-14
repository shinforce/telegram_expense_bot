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
from openai import OpenAI
from pydantic import BaseModel, Field
from enum import Enum

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
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    OPENAI_MODEL = "gpt-4o-mini"
    AI_TEMPERATURE = 0.1  # Low temperature for consistent categorization
    AI_CONFIDENCE_THRESHOLD = 0.6
    
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

@dataclass
class CategoryInfo:
    """Represents a category with its description"""
    name: str
    description: str
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "description": self.description
        }

@dataclass
class CategoryResult:
    """Result of AI categorization"""
    category: str
    confidence: float
    reasoning: str

# --- EXPENSE PARSER ---
class ExpenseParser:
    """Handles parsing of expense messages with multiple formats"""
    
    @classmethod
    def _extract_amount(cls, text: str) -> Tuple[Optional[float], str]:
        """Extract amount from text and return (amount, remaining_text)"""
        # Pattern to find amount at the beginning or end of text
        amount_pattern = r'(\d+(?:[.,]\d+)?)'
        
        # Check if amount is at the beginning
        start_match = re.match(r'^' + amount_pattern + r'\s+(.*)$', text)
        if start_match:
            amount_str = start_match.group(1).replace(',', '.')
            try:
                return float(amount_str), start_match.group(2)
            except ValueError:
                pass
        
        # Check if amount is at the end
        end_match = re.match(r'^(.*?)\s+' + amount_pattern + r'$', text)
        if end_match:
            amount_str = end_match.group(2).replace(',', '.')
            try:
                return float(amount_str), end_match.group(1)
            except ValueError:
                pass
        
        # Check if amount is in the middle with currency
        middle_match = re.match(r'^(.*?)\s+' + amount_pattern + r'\s+(.*)$', text)
        if middle_match:
            amount_str = middle_match.group(2).replace(',', '.')
            try:
                return float(amount_str), f"{middle_match.group(1)} {middle_match.group(3)}"
            except ValueError:
                pass
        
        return None, text
    
    @classmethod
    def _extract_currency(cls, text: str) -> Tuple[Optional[str], str]:
        """Extract currency from text and return (currency_code, remaining_text)"""
        words = text.split()
        
        # Check first word for currency
        if words and words[0].upper() in Config.CURRENCIES:
            currency = Config.CURRENCIES[words[0].upper()]
            remaining = ' '.join(words[1:]) if len(words) > 1 else ''
            return currency, remaining
        
        # Check last word for currency
        if words and words[-1].upper() in Config.CURRENCIES:
            currency = Config.CURRENCIES[words[-1].upper()]
            remaining = ' '.join(words[:-1]) if len(words) > 1 else ''
            return currency, remaining
        
        return None, text
    
    @classmethod
    def parse_line(cls, text: str) -> ParseResult:
        """Parse a single line of expense text"""
        if not text or not text.strip():
            return ParseResult(False, error_message="Empty text")
        
        original_text = text.strip()
        working_text = original_text
        
        # Step 1: Extract amount (required)
        amount, remaining_after_amount = cls._extract_amount(working_text)
        if amount is None:
            return ParseResult(False, error_message="No valid amount found")
        
        # Step 2: Try to extract currency from the remaining text
        currency, description = cls._extract_currency(remaining_after_amount)
        
        # If no currency found, use default and entire remaining text as description
        if currency is None:
            currency = Config.DEFAULT_CURRENCY
            description = remaining_after_amount
        
        # Clean up description
        description = description.strip()
        
        # If no description, return error
        if not description:
            return ParseResult(False, error_message="Description cannot be empty")
        
        # Create expense object
        expense = Expense(
            description=description,
            amount=int(amount), #don't need fractions, remove ceil for public version
            currency=currency,
            original_text=original_text
        )
        
        # Validate
        is_valid, error_msg = expense.validate()
        if not is_valid:
            return ParseResult(False, error_message=error_msg)
        
        logger.debug(f"Parsed: amount={amount}, currency={currency}, desc={description}")
        return ParseResult(True, expense=expense)
    
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
            
            return ceil(converted_amount) #don't need fractions, remove ceil for public version
        except Exception as e:
            logger.error(f"Conversion calculation failed: {e}")
            return None

# --- CATEGORY SHEET MANAGER ---
class CategorySheetManager:
    """Manages reading categories from Google Sheets"""
    
    def __init__(self, credentials_file: str, sheet_name: str):
        self.credentials_file = credentials_file
        self.sheet_name = sheet_name
    
    def fetch_categories(self) -> List[CategoryInfo]:
        """Fetch categories from Google Sheets 'categories' worksheet"""
        try:
            # Get the worksheet - always fetch fresh data
            gc = gspread.service_account(filename=self.credentials_file)
            sh = gc.open(self.sheet_name)
            categories_worksheet = sh.worksheet('categories')
            
            # Get all records as dictionaries
            records = categories_worksheet.get_all_records()
            
            # Convert to CategoryInfo objects
            categories = []
            for record in records:
                if record.get('category_name') and record.get('description_for_prompt'):
                    categories.append(
                        CategoryInfo(
                            name=record['category_name'],
                            description=record['description_for_prompt']
                        )
                    )
            
            logger.info(f"Fetched {len(categories)} categories from sheet")
            return categories
            
        except Exception as e:
            logger.error(f"Error fetching categories from sheet: {e}")
            return []

# --- AI CATEGORY DETERMINER ---
class AICategoryDeterminer:
    """Handles AI-based category determination using OpenAI"""
    
    def __init__(self, api_key: str, category_sheet_manager: CategorySheetManager):
        self.client = OpenAI(api_key=api_key) if api_key else None
        self.category_sheet_manager = category_sheet_manager
        self._category_enum = None
        self._category_model = None
        self.enabled = bool(api_key)
        
        if not self.enabled:
            logger.warning("OpenAI API key not found. AI categorization disabled.")
    
    def _create_category_enum(self, categories: List[CategoryInfo]):
        """Dynamically create an Enum from categories"""
        # Create enum members from category names
        enum_dict = {cat.name.upper().replace(' ', '_'): cat.name for cat in categories}
        return Enum('CategoryEnum', enum_dict)
    
    def _create_pydantic_model(self, categories: List[CategoryInfo]):
        """Create a Pydantic model with the category enum for structured output"""
        # Create the enum
        category_enum = self._create_category_enum(categories)
        
        # Create Pydantic model dynamically
        class ExpenseCategory(BaseModel):
            category: category_enum = Field(description="The category that best matches the expense")
            confidence: float = Field(description="Confidence level from 0 to 1", ge=0, le=1)
            reasoning: str = Field(description="Brief explanation for the category choice")
        
        return ExpenseCategory, category_enum
    
    def _build_system_prompt(self, categories: List[CategoryInfo]) -> str:
        """Build the system prompt with category descriptions"""
        categories_text = "\n".join([
            f"- {cat.name}: {cat.description}"
            for cat in categories
        ])
        
        return f"""You are an expense categorization assistant. Your task is to categorize expenses based on their description and the user who made them.

Available categories and their descriptions:
{categories_text}

Instructions:
1. Analyze the expense description carefully
2. Consider the user's name when relevant (some categories may be person-specific)
3. Choose the most appropriate category from the list above
4. Provide a confidence score (0-1) based on how well the expense matches the category
5. Give a brief reasoning for your choice

Be consistent and logical in your categorization."""
    
    async def determine_category(self, 
                                description: str, 
                                user_name: str,
                                amount: Optional[float] = None) -> CategoryResult:
        """
        Determine category for an expense using AI
        
        Returns:
            CategoryResult with category_name, confidence, and reasoning
        """
        if not self.enabled:
            return CategoryResult("UNCATEGORIZED", 0.0, "AI categorization not enabled")
        
        try:
            # Fetch current categories
            categories = self.category_sheet_manager.fetch_categories()
            if not categories:
                logger.error("No categories available")
                return CategoryResult("OTHER", 0.0, "No categories configured")
            
            # Create or update Pydantic model if needed
            if not self._category_model or len(categories) != len(self._category_enum.__members__):
                self._category_model, self._category_enum = self._create_pydantic_model(categories)
                logger.info(f"Created category model with {len(categories)} categories")
            
            # Build the user prompt
            user_prompt = f"""Categorize this expense:
Description: {description}
User: {user_name}
Amount: {amount if amount else 'Not specified'}"""
            
            # Make the API call with structured output
            completion = self.client.beta.chat.completions.parse(
                model=Config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": self._build_system_prompt(categories)},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=self._category_model,
                temperature=Config.AI_TEMPERATURE
            )
            
            # Parse the response
            result = completion.choices[0].message.parsed
            
            # Get the actual category name from the enum value
            category_name = result.category.value if hasattr(result.category, 'value') else str(result.category)
            
            logger.info(f"Categorized '{description}' as '{category_name}' with confidence {result.confidence}")
            
            return CategoryResult(category_name, result.confidence, result.reasoning)
            
        except Exception as e:
            logger.error(f"Error in AI categorization: {e}")
            return CategoryResult("ERROR", 0.0, f"AI categorization failed: {str(e)}")

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
    
    def add_expense(self, expense: Expense, amount_rsd: float, 
                   category: Optional[str] = None, confidence: Optional[float] = None) -> bool:
        """Add expense to Google Sheet with optional category information"""
        try:
            worksheet = self._get_worksheet()
            if worksheet is None:
                return False
            
            now = datetime.now().strftime("%Y-%m-%d") #%H:%M:%S")
            row = [
                now,
                expense.description,
                amount_rsd,
                expense.user_name
            ]
            
            # Add category info if AI categorization is enabled
            if category is not None:
                row.extend([
                    category#,
                    #confidence if confidence is not None else 0.0
                ])
            
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
        
        # Initialize AI categorization services if API key is available
        self.ai_enabled = bool(Config.OPENAI_API_KEY)
        if self.ai_enabled:
            self.category_sheet_manager = CategorySheetManager(
                Config.GOOGLE_SHEETS_CREDENTIALS,
                Config.GOOGLE_SHEET_NAME
            )
            self.ai_categorizer = AICategoryDeterminer(
                Config.OPENAI_API_KEY,
                self.category_sheet_manager
            )
        else:
            self.category_sheet_manager = None
            self.ai_categorizer = None
            logger.info("AI categorization disabled - no OpenAI API key")
    
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
        
        if self.ai_enabled:
            welcome_message += "\n\n🤖 AI categorization is enabled!"
        
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
            
            # Determine category using AI if enabled
            category_result = None
            if self.ai_enabled:
                category_result = await self.ai_categorizer.determine_category(
                    description=expense.description,
                    user_name=expense.user_name,
                    amount=expense.amount
                )
            
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
            
            # Add to Google Sheets with category if available
            if category_result:
                success = self.sheets_manager.add_expense(
                    expense, rsd_amount, 
                    category_result.category, 
                    category_result.confidence
                )
            else:
                success = self.sheets_manager.add_expense(expense, rsd_amount)
            
            if success:
                success_count += 1
                
                # Build success message
                success_msg = (
                    f"✅ Logged: *{expense.description}*\n"
                    f"Amount: {rsd_amount} RSD{conversion_note}\n"
                )
                
                # Add category info if available
                if category_result:
                    category_emoji = "🤖" if category_result.confidence > 0.7 else "🤔"
                    #success_msg += f"Category: {category_emoji} {category_result.category} ({category_result.confidence:.0%})\n"
                    success_msg += f"Category: {category_result.category}\n"
                    
                    # Add reasoning if confidence is low
                    if category_result.confidence < Config.AI_CONFIDENCE_THRESHOLD:
                        success_msg += f"💭 {category_result.reasoning}\n"
                
                success_msg += f"By: {user.first_name}"
                
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


