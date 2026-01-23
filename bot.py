import os
import logging
import asyncio
from datetime import datetime, timedelta
import time
import aiohttp
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from dotenv import load_dotenv


load_dotenv()

BOT_NAME = "Nova"
BOT_VERSION = "2.0.0"
DEVELOPER_NAME = "Karthi"  
BOT_DESCRIPTION = (
    "I'm Nova — a hybrid AI assistant running on AWS. "
    "I combine fast logic with advanced AI reasoning to help you."
)


RATE_LIMIT_SECONDS = 2


MAX_MEMORY_MESSAGES = 10  
MAX_USERS_IN_MEMORY = 1000  


AI_TEXT_MODEL = "openai/gpt-4o-mini-2024-07-18" 
AI_VISION_MODEL = "openai/gpt-4o-2024-11-20"
AI_TEMPERATURE = 0.7
AI_TIMEOUT = 30
MAX_IMAGE_SIZE_MB = 20

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")


if not OPENROUTER_API_KEY or not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("❌ Missing environment variables: OPENROUTER_API_KEY or TELEGRAM_BOT_TOKEN")

user_memory = {}
user_last_seen = {}

def get_system_prompt(language: str = "en", is_vision: bool = False) -> str:
    """Get system prompt with language and vision support."""
    base_prompt = (
        f"You are {BOT_NAME}, a highly capable AI assistant. "
        "Core behaviors:\n"
        "- Respond naturally and conversationally\n"
        "- Provide accurate, helpful information\n"
        "- Keep responses concise (2-3 sentences for simple queries, more for complex ones)\n"
        "- Never repeat user input verbatim\n"
        "- Ask clarifying questions only when truly necessary\n"
        "- Use emojis sparingly and professionally\n"
        "- For code or technical content, use proper formatting\n"
        "- Admit when you don't know something rather than guessing\n"
    )
    
    if is_vision:
        base_prompt += (
            "- You can see and analyze images\n"
            "- Describe images clearly and accurately\n"
            "- Extract text from images when present (OCR)\n"
            "- Identify objects, people, scenes, and activities\n"
            "- Provide detailed analysis when requested\n"
        )
    
    if language != "en":
        base_prompt += f"- IMPORTANT: Respond in the same language as the user ({language})\n"
    
    return base_prompt

def is_low_quality_input(text: str) -> bool:
    """Check if input is too trivial to send to AI."""
    text = text.strip().lower()
    trivial_set = {
        "hi", "hello", "hey", "ok", "okay", "yo", "?", ".", 
        "thanks", "thank you", "lol", "hmm", "k"
    }
    return len(text) <= 2 or text in trivial_set


def is_echo(user_text: str, ai_text: str) -> bool:
    """Detect if AI response is just echoing user input."""
    u = ''.join(c for c in user_text.lower() if c.isalnum())
    a = ''.join(c for c in ai_text.lower() if c.isalnum())
    return a == u or len(ai_text.strip()) <= 3


def get_ist_time() -> str:
    """Get current time in IST format."""
    now_ist = datetime.utcnow() + timedelta(hours=5, minutes=30)
    return now_ist.strftime('%H:%M:%S, %d %B %Y')


def escape_markdown(text: str) -> str:
    """Escape special markdown characters to prevent formatting issues."""
    special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    for char in special_chars:
        text = text.replace(char, f'\\{char}')
    return text


def cleanup_old_users():
    """Remove oldest users from memory if limit exceeded (prevent memory leak)."""
    if len(user_memory) > MAX_USERS_IN_MEMORY:
        users_to_remove = len(user_memory) - MAX_USERS_IN_MEMORY + 100
        oldest_users = sorted(user_last_seen.items(), key=lambda x: x[1])[:users_to_remove]
        for user_id, _ in oldest_users:
            user_memory.pop(user_id, None)
            user_last_seen.pop(user_id, None)
        logger.info(f"Cleaned up {users_to_remove} users from memory")


def detect_language(text: str) -> str:
    """Simple language detection based on character sets."""
    if any('\u0600' <= c <= '\u06FF' for c in text):
        return "ar"
    elif any('\u0400' <= c <= '\u04FF' for c in text):
        return "ru"
    elif any('\u4E00' <= c <= '\u9FFF' for c in text):
        return "zh"
    elif any('\u3040' <= c <= '\u309F' or '\u30A0' <= c <= '\u30FF' for c in text):
        return "ja"
    elif any('\uAC00' <= c <= '\uD7AF' for c in text):
        return "ko"
    elif any('\u0900' <= c <= '\u097F' for c in text):
        return "hi"
    elif any('\u0980' <= c <= '\u09FF' for c in text):
        return "bn"
    elif any('\u0C80' <= c <= '\u0CFF' for c in text):
        return "kn"
    elif any('\u0B80' <= c <= '\u0BFF' for c in text):
        return "ta"
    elif any('\u0C00' <= c <= '\u0C7F' for c in text):
        return "te"
    else:
        return "en"

async def download_image(photo, context) -> str:
    """Download image from Telegram and return local path."""
    import tempfile
    file = await photo.get_file()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_path = temp_file.name
    temp_file.close()
    await file.download_to_drive(temp_path)
    return temp_path


def encode_image_base64(image_path: str) -> str:
    """Encode image to base64 for API."""
    import base64
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def validate_image_size(file_size: int) -> bool:
    """Validate image size."""
    max_size_bytes = MAX_IMAGE_SIZE_MB * 1024 * 1024
    return file_size <= max_size_bytes


def compress_image(image_path: str) -> str:
    """Compress image to reduce size and save API costs."""
    from PIL import Image
    import os
    
    try:
        img = Image.open(image_path)
        
        if img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = background
        
        original_size = os.path.getsize(image_path) / (1024 * 1024)
        
        if original_size > 2:
            max_width = 1920
            if img.width > max_width:
                ratio = max_width / img.width
                new_size = (max_width, int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            compressed_path = image_path.replace('.jpg', '_compressed.jpg')
            img.save(compressed_path, quality=85, optimize=True)
            
            new_size = os.path.getsize(compressed_path) / (1024 * 1024)
            logger.info(f"Compressed image: {original_size:.2f}MB → {new_size:.2f}MB")
            
            return compressed_path
        
        return image_path
        
    except Exception as e:
        logger.warning(f"Image compression failed: {e}, using original")
        return image_path

async def ai_response(messages: list, use_vision: bool = False) -> str:
    """
    Call OpenRouter API to get AI response.
    Uses aiohttp for async HTTP requests with proper timeout handling.
    Automatically selects vision model when use_vision=True.
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/yourusername/telegram-bot",
        "X-Title": BOT_NAME
    }

    model = AI_VISION_MODEL if use_vision else AI_TEXT_MODEL

    data = {
        "model": model,
        "temperature": AI_TEMPERATURE,
        "messages": messages
    }

    try:
        timeout = aiohttp.ClientTimeout(total=AI_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=data, headers=headers) as response:
                response.raise_for_status()
                result = await response.json()
                return result["choices"][0]["message"]["content"]
    except aiohttp.ClientError as e:
        logger.error(f"API request failed: {e}")
        raise
    except asyncio.TimeoutError:
        logger.error("API request timed out")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in AI response: {e}")
        raise

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    user_name = update.message.from_user.first_name
    welcome_message = (
        f"Hi {user_name}! 👋\n\n"
        f"I'm *{BOT_NAME}*, your AI assistant.\n\n"
        f"{BOT_DESCRIPTION}\n\n"
        "💬 Just send me a message and I'll help you!\n"
        "📋 Use /help to see available commands."
    )
    await update.message.reply_text(welcome_message, parse_mode="Markdown")
    logger.info(f"User {update.message.from_user.id} started the bot")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command."""
    help_text = (
        "🤖 *Available Commands:*\n\n"
        "/start - Start the bot\n"
        "/help - Show this help message\n"
        "/time - Get current IST time\n"
        "/clear - Clear conversation history\n"
        "/about - Show bot information\n\n"
        "💡 *Tips:*\n"
        "• Just send me any message to chat\n"
        "• I remember our last few messages\n"
        "• Ask me anything - questions, calculations, advice, etc.\n"
        "• 🌍 I support multiple languages - just write in your language"
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")


async def time_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /time command."""
    current_time = get_ist_time()
    await update.message.reply_text(f"🕒 Current IST time:\n*{current_time}*", parse_mode="Markdown")


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /clear command to reset conversation history."""
    user_id = update.message.from_user.id
    if user_id in user_memory:
        user_memory[user_id] = []
        await update.message.reply_text("✅ Conversation history cleared!")
        logger.info(f"Cleared memory for user {user_id}")
    else:
        await update.message.reply_text("ℹ️ No conversation history to clear.")


async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /about command to show bot information."""
    about_text = (
        f"🤖 *{BOT_NAME} - Hybrid AI Assistant*\n\n"
        f"📌 *Version:* {BOT_VERSION}\n"
        f"🧠 *AI Models:*\n"
        f"  • Text: {AI_TEXT_MODEL}\n"
        f"  • Vision: {AI_VISION_MODEL}\n"
        f"👨‍💻 *Developer:* {DEVELOPER_NAME}\n\n"
        f"📝 *Description:*\n{BOT_DESCRIPTION}\n\n"
        f"✨ *Capabilities:*\n"
        f"• 💬 Natural conversation\n"
        f"• 🌍 Multi-language support\n\n"
        f"💡 Use /help to see available commands"
    )
    await update.message.reply_text(about_text, parse_mode="Markdown")
    logger.info(f"User {update.message.from_user.id} requested bot info")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming text messages."""
    user_id = update.message.from_user.id
    user_text = update.message.text.strip()
    user_msg = user_text.lower()

    now = time.time()
    if user_id in user_last_seen and now - user_last_seen[user_id] < RATE_LIMIT_SECONDS:
        await update.message.reply_text("⏳ Please wait a moment before sending another message.")
        return
    user_last_seen[user_id] = now

    identity_questions = [
        "what is your name", "who are you", "your name", 
        "tell me your name", "what's your name", "whats your name"
    ]
    if user_msg in identity_questions:
        await update.message.reply_text(
            f"🤖 My name is *{BOT_NAME}*.\n\n{BOT_DESCRIPTION}",
            parse_mode="Markdown"
        )
        return

    time_queries = ["time", "what is time", "current time", "what time is it", "whats the time"]
    if user_msg in time_queries:
        current_time = get_ist_time()
        await update.message.reply_text(f"🕒 Current IST time: *{current_time}*", parse_mode="Markdown")
        return

    if is_low_quality_input(user_text):
        await update.message.reply_text(
            "👋 Hi! Ask me a question or tell me what you need help with 🙂"
        )
        return

    if user_id not in user_memory:
        user_memory[user_id] = []

    user_memory[user_id].append({"role": "user", "content": user_text})
    user_memory[user_id] = user_memory[user_id][-MAX_MEMORY_MESSAGES:]
    
    if len(user_memory) % 100 == 0:
        cleanup_old_users()

    detected_lang = detect_language(user_text)
    system_prompt = get_system_prompt(detected_lang)
    
    messages = [
        {"role": "system", "content": system_prompt}
    ] + user_memory[user_id]

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        reply = await ai_response(messages)
    except asyncio.TimeoutError:
        await update.message.reply_text(
            "⏱️ Request timed out. The AI is taking too long to respond. Please try again with a simpler question."
        )
        logger.error(f"Timeout error for user {user_id}")
        return
    except aiohttp.ClientError:
        await update.message.reply_text(
            "❌ Sorry, I'm having trouble connecting to my AI service. Please try again in a moment."
        )
        logger.error(f"API error for user {user_id}")
        return
    except Exception as e:
        await update.message.reply_text(
            "❌ An unexpected error occurred. Please try again."
        )
        logger.error(f"Unexpected error for user {user_id}: {e}")
        return

    if is_echo(user_text, reply):
        reply = "🤖 I'm here to help — could you give me a bit more detail?"

    user_memory[user_id].append({"role": "assistant", "content": reply})

    try:
        await update.message.reply_text(reply)
        logger.info(f"Responded to user {user_id}")
    except Exception as e:
        logger.error(f"Failed to send message to user {user_id}: {e}")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle errors in the bot."""
    logger.error(f"Update {update} caused error: {context.error}")
    
    if update and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "❌ An error occurred while processing your request. Please try again."
            )
        except Exception:
            pass

def main():
    """Start the bot."""
    logger.info("🚀 Starting Telegram AI Bot...")
    
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("time", time_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(CommandHandler("about", about_command))
    
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming photos."""
        try:
            user_id = update.message.from_user.id
            photo = update.message.photo[-1]
            
            if not validate_image_size(photo.file_size):
                await update.message.reply_text(
                    f"❌ Image is too large. Maximum size: {MAX_IMAGE_SIZE_MB}MB"
                )
                return
            
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
            
            image_path = await download_image(photo, context)
            image_path = compress_image(image_path)
            image_base64 = encode_image_base64(image_path)
            
            user_text = update.message.caption or "Analyze this image"
            if user_id not in user_memory:
                user_memory[user_id] = []
            
            user_memory[user_id].append({"role": "user", "content": user_text})
            user_memory[user_id] = user_memory[user_id][-MAX_MEMORY_MESSAGES:]
            
            detected_lang = detect_language(user_text)
            system_prompt = get_system_prompt(detected_lang, is_vision=True)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64
                            }
                        }
                    ]
                }
            ]
            
            try:
                reply = await ai_response(messages, use_vision=True)
            except Exception as e:
                await update.message.reply_text(
                    "❌ Failed to analyze image. Please try again."
                )
                logger.error(f"Vision API error for user {user_id}: {e}")
                return
            
            user_memory[user_id].append({"role": "assistant", "content": reply})
            
            await update.message.reply_text(reply)
            logger.info(f"Analyzed image for user {user_id}")
            
            import os as os_module
            try:
                os_module.remove(image_path)
            except:
                pass
                
        except Exception as e:
            logger.error(f"Photo handler error: {e}")
            await update.message.reply_text(
                "❌ An error occurred while processing the image."
            )
    
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    
    app.add_error_handler(error_handler)
    
    logger.info(f"🤖 {BOT_NAME} is now running on AWS...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
