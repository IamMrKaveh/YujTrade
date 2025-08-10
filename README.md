# 📊 Advanced Crypto Analyzing Trading Bot  
ربات تحلیلگر پیشرفته بازار کریپتو 🚀

---

## 🇮🇷 فارسی

### 📝 معرفی
این پروژه یک **ربات تحلیلگر پیشرفته بازار ارز دیجیتال** است که قابلیت‌های گسترده‌ای دارد:  
🔹 تحلیل تکنیکال چندبازه‌ای  
🔹 پیش‌بینی قیمت با **مدل LSTM**  
🔹 استفاده از داده‌های **آن‌چین**  
🔹 تحلیل احساسات بازار بر اساس اخبار و شاخص ترس و طمع  

خروجی ربات شامل:
- 📈 سیگنال‌های خرید/فروش/نگه‌داری
- 🏷 امتیاز اطمینان سیگنال
- 🎯 سطوح ورود و خروج داینامیک
- 📋 دلایل دقیق تحلیلی

---

### ⚙️ قابلیت‌ها
- 📊 **اندیکاتورهای تکنیکال**: SMA، EMA، RSI، MACD، باند بولینگر، استوکاستیک، ATR، ایچیموکو، Williams %R، CCI، تحلیل حجم.  
- 🔍 **شناسایی الگوها**: پوشای صعودی/نزولی، سقف و کف دوقلو، سر و شانه، پرچم، مثلث، کنج.  
- 🤖 **پیش‌بینی قیمت با LSTM**: پیش‌بینی مبتنی بر یادگیری ماشین با TensorFlow/Keras.  
- ⏳ **تأیید چندبازه‌ای**: هم‌ترازی سیگنال‌ها در تایم‌فریم‌های مختلف.  
- 📉 **تحلیل بازار**: تشخیص روند، نوسان‌پذیری، سطوح حمایت/مقاومت.  
- 🔗 **داده‌های آن‌چین**: آدرس‌های فعال، حجم تراکنش‌ها، جریان صرافی‌ها.  
- 📰 **تحلیل احساسات**: شاخص ترس و طمع، امتیازدهی اخبار با CryptoPanic API.  
- 🛡 **محاسبه سطوح داینامیک**: استاپ‌لاس، تارگت، نقطه سر به سر، و تریلینگ‌استاپ.

---

### 📦 پیش‌نیازها
- Python 3.9 یا بالاتر  
- نصب کتابخانه‌های زیر:  
pandas, numpy, ccxt, tensorflow, scikit-learn, python-telegram-bot, web3, requests, aiosqlite

---

### 🔑 متغیرهای محیطی
برای اجرا باید مقادیر زیر را در `.env` یا محیط سیستم ست کنید:  
COINEX_API_KEY=کلید API کوینکس
COINEX_SECRET=کلید مخفی کوینکس
BOT_TOKEN=توکن ربات تلگرام
CRYPTOPANIC_KEY=کلید API سایت CryptoPanic
ALCHEMY_URL=آدرس HTTP سرویس Alchemy

---

### ▶️ اجرا
```bash
pip install -r requirements.txt
python main.py
🧪 بک‌تست و Paper Trading
📜 بک‌تست: با استفاده از داده‌های تاریخی و ماژول داخلی WalkForwardOptimizer می‌توانید عملکرد استراتژی‌ها را در بازه‌های زمانی گذشته بررسی کنید.

🧩 Paper Trading: امکان اجرای معاملات مجازی (بدون ریسک مالی) برای تست عملی استراتژی قبل از ورود به بازار واقعی.

نمونه اجرای بک‌تست:
from main import WalkForwardOptimizer, TradingService

optimizer = WalkForwardOptimizer(trading_service=TradingService())
results = optimizer.run(symbol="BTC/USDT", timeframe="1h", lookback_days=30, test_days=7)
print(results)

---

🇺🇸 English

---

📝 Overview
This is an Advanced Cryptocurrency Trading Bot designed to perform:

📊 Multi-timeframe technical analysis

🤖 Price prediction using LSTM

🔗 On-chain data integration

📰 Market sentiment analysis based on news & Fear & Greed Index

The bot outputs:

📈 Buy/Sell/Hold signals

🏷 Confidence scores

🎯 Dynamic entry/exit levels

📋 Detailed analytical reasons

---

⚙️ Features
Technical Indicators: SMA, EMA, RSI, MACD, Bollinger Bands, Stochastic, ATR, Ichimoku, Williams %R, CCI, Volume Analysis.

Pattern Recognition: Bullish/Bearish Engulfing, Double Top/Bottom, Head & Shoulders, Flags, Triangles, Wedges.

LSTM Price Prediction with TensorFlow/Keras.

Multi-Timeframe Confirmation for consistent signals.

Market Analysis: Trend detection, volatility, support/resistance.

On-Chain Data: Active addresses, transaction volumes, exchange flows.

Sentiment Analysis via Fear & Greed Index + CryptoPanic API news scoring.

Dynamic Level Calculation: Stop-loss, take-profit, breakeven, trailing stop.

---

📦 Requirements
Python 3.9+

Install the following libraries:
pandas, numpy, ccxt, tensorflow, scikit-learn, python-telegram-bot, web3, requests, aiosqlite

---

🔑 Environment Variables
Set the following in .env or your system environment:
COINEX_API_KEY=your_coinex_key
COINEX_SECRET=your_coinex_secret
BOT_TOKEN=your_telegram_bot_token
CRYPTOPANIC_KEY=your_cryptopanic_api_key
ALCHEMY_URL=your_alchemy_http_url

▶️ Run
pip install -r requirements.txt
python main.py

---

🧪 Backtesting & Paper Trading
📜 Backtesting: Use historical data and the built-in WalkForwardOptimizer to test strategies over past market conditions.

🧩 Paper Trading: Execute simulated trades (no real money) to validate your strategy before going live.

Example backtest:
from main import WalkForwardOptimizer, TradingService

optimizer = WalkForwardOptimizer(trading_service=TradingService())
results = optimizer.run(symbol="BTC/USDT", timeframe="1h", lookback_days=30, test_days=7)
print(results)

---

📜 License
Licensed under GPL v3.
You are free to use, modify, and distribute with proper credit.