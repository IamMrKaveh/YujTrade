# YujTradeBot – Telegram Crypto Signal Bot 🤖📈

A scalable and modular Telegram bot that fetches candle data for **60 cryptocurrencies** from the **CoinEx** exchange and analyzes them across multiple **timeframes** using technical indicators. It then suggests the **best low-risk, high-reward signal** to the user via Telegram.

---

## 🔧 Features
- ✅ Multi-timeframe support (5m, 15m, 1h, 4h, 1d)
- 📊 Analyzes 60 coins from CoinEx
- 🧠 Uses indicators like RSI, MA, MACD, Volume
- 📈 Suggests best signal with lowest risk and highest reward
- 💬 Telegram Bot interface with commands like `/start`, `/status`, `/signals`
- ⚡ Cached Kline data for faster analysis
- 🔁 Async & non-blocking structure
- 🧪 Includes unit tests for analysis components

---

## 🚀 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/yourname/YujTradeBot.git
   cd YujTradeBot

2. Install dependencies:

pip install -r requirements.txt

3. Add your secrets to config/secrets.py or use .env:

TELEGRAM_BOT_TOKEN = "your-telegram-bot-token"
COINEX_API_KEY = "your-coinex-api-key"
COINEX_API_SECRET = "your-coinex-api-secret"

4. Run the bot:

python main.py

=========================================================================================
=========================================================================================

YujTradeBot – ربات تلگرامی تحلیل بازار رمزارز 📈🤖
یک ربات مقیاس‌پذیر و ماژولار که داده کندل ۶۰ ارز دیجیتال را از صرافی CoinEx دریافت می‌کند و آن‌ها را در تایم‌فریم‌های مختلف تحلیل کرده و بهترین سیگنال با کم‌ترین ریسک و بیشترین سوددهی را به کاربران تلگرام پیشنهاد می‌دهد.

🔧 امکانات
✅ پشتیبانی از تایم‌فریم‌های مختلف (5m، 15m، 1h، 4h، 1d)

📊 تحلیل ۶۰ ارز از CoinEx

📉 استفاده از اندیکاتورهای RSI، MA، MACD، Volume و...

🧠 پیشنهاد سیگنال با امتیاز بالا و ریسک پایین

💬 ارتباط از طریق ربات تلگرام با دستورات /start, /signals, /status

⚡ کشینگ دیتا برای تحلیل سریع‌تر

🔁 ساختار غیربلوکینگ و async

🧪 دارای تست واحد برای بخش‌های کلیدی



🚀 روش اجرا

1. کلون پروژه :

git clone https://github.com/yourname/YujTradeBot.git
cd YujTradeBot

2. نصب وابستگی‌ها:

pip install -r requirements.txt

3. اطلاعات API را در config/secrets.py وارد کنید:

TELEGRAM_BOT_TOKEN = "توکن ربات تلگرام"
COINEX_API_KEY = "کلید API کوینکس"
COINEX_API_SECRET = "رمز API کوینکس"

4. اجرای ربات :

python main.py

