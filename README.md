# 🤖 Nova – Hybrid AI Telegram Bot

Nova is a **Hybrid AI-powered Telegram Bot** built with **Python**, designed to run on **AWS EC2** and integrate **LLM-based intelligence** with deterministic logic.
It combines fast rule-based responses with advanced AI reasoning to deliver a smart, scalable chatbot experience.

---

## 🚀 Features

* 💬 **Hybrid AI Responses**
  * Deterministic commands (time, identity, greetings)
  * AI-powered conversational replies using LLMs
* 🧠 **Short-term Memory**
  * Maintains recent conversation context per user
* 🛡️ **Anti-Echo & Spam Protection**
  * Prevents repetitive echo replies
  * Rate limiting to avoid bot overload
* 🌍 **Multi-Language Support**
  * Automatic language detection
  * Responds in user's language
* 🧾 **Environment-based Secrets**
  * Secure handling of API keys via `.env`
* ☁️ **Cloud-Ready**
  * Designed to run on AWS EC2 with Elastic IP
* 📈 **Scalable Architecture**
  * Modular structure for future features (metrics, voice, monitoring)

---

## 🧰 Tech Stack

* **Language:** Python 3.10+
* **Framework:** `python-telegram-bot` v20.7
* **AI Provider:** OpenRouter (LLM API)
* **Text Model:** GPT-4o-mini for conversations
* **HTTP Client:** aiohttp (async)

* **Cloud:** AWS EC2 + Elastic IP
* **OS:** Ubuntu / Windows
* **Version Control:** Git & GitHub

---

## 📁 Project Structure

```
Nova_bot/
├── bot.py              # Main bot with all handlers and logic
├── requirements.txt    # Python dependencies
├── .env                # Example environment variables
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

---

## 🔐 Environment Variables

Create a `.env` file in the project root:

```env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

> ⚠️ Never commit `.env` files or API keys to GitHub. Use `.env.example` as a template.

---

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/karthi206/Nova_bot.git
cd Nova_bot

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Running the Bot

```bash
python bot.py
```

You should see:

```
🚀 Starting Telegram AI Bot...
🤖 Nova is now running on AWS...
```

---

## 🧪 Available Commands

| Command | Description |
|---------|-------------|
| `/start` | Initialize the bot and see welcome message |
| `/help` | View all available commands and tips |
| `/time` | Get current server time in IST format |
| `/clear` | Clear your conversation history |
| `/about` | Learn about Nova's capabilities and models |

---

## 💬 How to Use

1. **Text Messages:** Simply send a message and Nova will respond using AI
2. **Natural Queries:** Ask anything - math, coding, advice, etc.
3. **Multi-language:** Type in any language, Nova responds in the same language

### Example Interactions

```
User: What is your name?
Nova: 🤖 My name is Nova. I'm Nova — a hybrid AI assistant running on AWS...

User: What time is it?
Nova: 🕒 Current IST time: 14:30:45, 23 January 2026

User: [Sends image]
Nova: I can see an image of... [detailed analysis]
```

---

## 🔧 Configuration

Edit these constants in `bot.py` to customize behavior:

```python
MAX_MEMORY_MESSAGES = 10          # Conversation context window
MAX_USERS_IN_MEMORY = 1000        # Max concurrent users
RATE_LIMIT_SECONDS = 2            # Minimum seconds between messages
AI_TEMPERATURE = 0.7              # AI creativity (0-1)
AI_TIMEOUT = 30                   # API timeout in seconds
MAX_IMAGE_SIZE_MB = 20            # Max image upload size
```

---

## 🧠 Architecture

### Message Flow

```
User Message
    ↓
Rate Limit Check
    ↓
Low Quality Input Check
    ↓
Language Detection
    ↓
System Prompt Creation
    ↓
API Call to OpenRouter
    ↓
Response Validation
    ↓
Echo Check
    ↓
Store in Memory
    ↓
Send to User
```

### Memory Management

- Keeps last 10 messages per user
- Removes oldest users when limit (1000) exceeded
- Automatic cleanup every 100 interactions
- No persistent storage (resets on bot restart)

---

## 🚀 Deployment on AWS EC2

```bash
# SSH into your EC2 instance
ssh -i your-key.pem ubuntu@your-elastic-ip

# Clone and setup
git clone https://github.com/karthi206/Nova_bot.git
cd Nova_bot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create .env with your credentials
nano .env

# Run bot (consider using screen or systemd)
python bot.py

# Or with screen for background execution:
screen -S nova
python bot.py
# Press Ctrl+A then D to detach
```

---

## 📊 Monitoring

Check logs in real-time:

```bash
# If using systemd:
journalctl -u nova-bot -f

# If running in screen:
screen -r nova
```

---

## 🔮 Future Enhancements

* 📊 Prometheus & Grafana metrics
* 🎙️ Voice input (Speech-to-Text)
* 🧠 Long-term memory (Redis / PostgreSQL)
* 🌐 Webhook-based deployment (Heroku/Railway)
* 🔒 AWS Secrets Manager integration
* 🧪 Load testing & performance monitoring
* 📱 Multi-platform support
* 🤖 Custom fine-tuned models

---

## 🐛 Troubleshooting

### Bot not responding
- Verify `TELEGRAM_BOT_TOKEN` is correct
- Check internet connection
- Review logs for API errors

### API errors
- Ensure `OPENROUTER_API_KEY` has credits
- Check rate limits
- Verify image size is under 20MB

### Timeout errors
- Increase `AI_TIMEOUT` value
- Check API service status
- Try with a simpler query

---

## 🧑‍💻 Author

**Karthikeyan A**
AI / Cloud Enthusiast
Built as a hands-on AIML + AWS learning project 🚀

**GitHub:** [@karthi206](https://github.com/karthi206)

---

## 📜 License

This project is licensed under the **MIT License** - see LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push and create a Pull Request

---

**Made with ❤️ by Karthikeyan A**
