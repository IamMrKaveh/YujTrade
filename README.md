# 🤖 ربات تحلیل و سیگنال‌دهی پیشرفته ارز دیجیتال

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Open Source](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://opensource.org/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![Telegram](https://img.shields.io/badge/Telegram-Bot-blue?logo=telegram)](https://telegram.org/)

---

## 🇮🇷 نسخه فارسی (Persian Version)

به ربات پیشرفته تحلیل تکنیکال و سیگنال‌دهی ارزهای دیجیتال خوش آمدید! 🚀 این پروژه یک ابزار قدرتمند و متن-باز است که با استفاده از ترکیبی از اندیکاتورهای تکنیکال کلاسیک، الگوهای کندل استیک، تحلیل حجمی، یادگیری عمیق (LSTM)، تحلیل احساسات بازار و داده‌های آن-چین، فرصت‌های معاملاتی را شناسایی کرده و از طریق یک ربات تلگرام کاربرپسند به شما اطلاع می‌دهد.

### ✨ ویژگی‌های کلیدی

- ✅ **تحلیل جامع تکنیکال**: استفاده از بیش از ۱۵ اندیکاتور محبوب مانند `RSI`, `MACD`, `Bollinger Bands`, `Ichimoku Cloud`, `SuperTrend` و `ADX`.
- 🧠 **مدل یادگیری عمیق (LSTM)**: پیش‌بینی قیمت آینده با استفاده از یک مدل شبکه عصبی بازگشتی برای افزایش دقت سیگنال‌ها.
- 🕒 **تحلیل چند-زمانی (Multi-Timeframe)**: تأیید سیگنال‌ها در تایم‌فریم‌های بالاتر برای کاهش نویز و افزایش اطمینان.
- 😊 **تحلیل احساسات بازار**: دریافت شاخص ترس و طمع (Fear & Greed) و تحلیل اخبار از طریق CryptoPanic API برای درک بهتر جو بازار.
- 🔗 **تحلیل آن-چین (On-Chain)**: بررسی معیارهایی مانند آدرس‌های فعال و حجم تراکنش‌ها برای تأیید قدرت روند (در صورت ارائه Alchemy URL).
- 📈 **شناسایی الگوهای کلاسیک**: تشخیص خودکار الگوهایی مانند `Engulfing`, `Double Top/Bottom` و `Head and Shoulders`.
-  dynamic **سطوح ورود و خروج داینامیک**: محاسبه هوشمند نقاط ورود، حد سود، حد ضرر و حتی حد ضرر متحرک (Trailing Stop) بر اساس نوسانات بازار (ATR) و سطوح فیبوناچی.
- 🤖 **رابط کاربری تلگرام**: تعامل آسان با ربات از طریق دستورات و دکمه‌های شیشه‌ای برای دریافت تحلیل‌های سریع یا جامع.
- ⚙️ **پیکربندی آسان**: مدیریت آسان لیست ارزها، تایم‌فریم‌ها و تنظیمات کلیدی از طریق فایل `config.json` و متغیرهای محیطی.
- 🗄️ **پایگاه داده داخلی**: ذخیره‌سازی و کش کردن داده‌های OHLCV در یک پایگاه داده SQLite برای دسترسی سریع‌تر و کاهش درخواست‌ها به صرافی.

### 🛠️ نصب و راه‌اندازی

برای اجرای این ربات، مراحل زیر را دنبال کنید:

#### ۱. پیش‌نیازها
- پایتون نسخه 3.9 یا بالاتر
- `pip` و `venv`
- دسترسی به یک صرافی پشتیبانی شده توسط `ccxt` (کد فعلی برای CoinEx تنظیم شده است)

#### ۲. کلون کردن مخزن
```bash
git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
cd your-repository-name
```

#### ۳. ایجاد و فعال‌سازی محیط مجازی
- **در ویندوز:**
  ```bash
  python -m venv venv
  .\venv\Scripts\activate
  ```
- **در macOS/Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

#### ۴. نصب نیازمندی‌ها
پیشنهاد می‌شود یک فایل `requirements.txt` ایجاد کرده و کتابخانه‌های زیر را در آن قرار دهید:
```
pandas
numpy
tensorflow
scikit-learn
ccxt
python-telegram-bot[ext]
aiosqlite
requests
web3
```
سپس با دستور زیر نصب کنید:
```bash
pip install -r requirements.txt
```

#### ۵. پیکربندی
مهم‌ترین بخش، تنظیم کلیدهای API و توکن‌ها است. یک فایل به نام `.env` در ریشه پروژه ایجاد کرده و مقادیر زیر را در آن قرار دهید:

```env
# کلیدهای API صرافی (مثال برای CoinEx)
COINEX_API_KEY="YOUR_COINEX_API_KEY"
COINEX_SECRET="YOUR_COINEX_SECRET_KEY"

# توکن ربات تلگرام از BotFather
BOT_TOKEN="YOUR_TELEGRAM_BOT_TOKEN"

# (اختیاری) کلید API برای تحلیل اخبار
CRYPTOPANIC_KEY="YOUR_CRYPTOPANIC_API_KEY"

# (اختیاری) کلید API برای تحلیل آن-چین
ALCHEMY_URL="YOUR_ALCHEMY_ETHEREUM_MAINNET_URL"
```
همچنین می‌توانید فایل `config.json` را برای تغییر لیست ارزها، تایم‌فریم‌ها و حداقل امتیاز اطمینان سیگنال ویرایش کنید.

### ▶️ نحوه استفاده
پس از اتمام مراحل نصب و پیکربندی، ربات را با دستور زیر اجرا کنید:
```bash
python main.py
```
سپس به ربات تلگرام خود رفته و دستور `/start` را ارسال کنید. از طریق دکمه‌های نمایش داده شده می‌توانید تحلیل سریع (`1m`) یا تحلیل جامع (برای تمام تایم‌فریم‌ها) را درخواست دهید.

### 🤝 مشارکت
مشارکت شما باعث پیشرفت جامعه متن-باز می‌شود. هرگونه مشارکت از طریق Pull Request یا ثبت Issue مورد استقبال قرار می‌گیرد.

۱. پروژه را Fork کنید.
۲. یک شاخه جدید برای ویژگی خود ایجاد کنید (`git checkout -b feature/AmazingFeature`).
۳. تغییرات خود را Commit کنید (`git commit -m 'Add some AmazingFeature'`).
۴. به شاخه خود Push کنید (`git push origin feature/AmazingFeature`).
۵. یک Pull Request باز کنید.

### 📜 لایسنس
این پروژه تحت لایسنس `GPL v3` منتشر شده است. برای اطلاعات بیشتر فایل `LICENSE` را مطالعه کنید.

### ⚠️ سلب مسئولیت
این پروژه یک ابزار **آموزشی و تحقیقاتی** است و به هیچ عنوان توصیه مالی یا سرمایه‌گذاری محسوب نمی‌شود. بازار ارزهای دیجیتال بسیار پرنوسان است و ریسک بالایی دارد. نویسندگان این پروژه هیچ مسئولیتی در قبال سود یا زیان احتمالی شما نخواهند داشت. **همیشه خودتان تحقیق کنید (DYOR)**.

---
---

## 🇬🇧 English Version

Welcome to the Advanced Crypto Technical Analysis & Signaling Bot! 🚀 This project is a powerful, open-source tool that identifies trading opportunities by combining classic technical indicators, candlestick patterns, volume analysis, Deep Learning (LSTM), market sentiment, and on-chain data. Signals are delivered through a user-friendly Telegram bot.

### ✨ Key Features

- ✅ **Comprehensive Technical Analysis**: Utilizes over 15 popular indicators, including `RSI`, `MACD`, `Bollinger Bands`, `Ichimoku Cloud`, `SuperTrend`, and `ADX`.
- 🧠 **Deep Learning Model (LSTM)**: Predicts future price movements using a Recurrent Neural Network to enhance signal accuracy.
- 🕒 **Multi-Timeframe Analysis**: Confirms signals on higher timeframes to reduce market noise and increase reliability.
- 😊 **Market Sentiment Analysis**: Fetches the Fear & Greed Index and analyzes news from the CryptoPanic API to gauge market sentiment.
- 🔗 **On-Chain Analysis**: Examines metrics like active addresses and transaction volume to confirm trend strength (requires an Alchemy URL).
- 📈 **Classic Pattern Recognition**: Automatically detects patterns such as `Engulfing`, `Double Top/Bottom`, and `Head and Shoulders`.
- 🎯 **Dynamic Entry & Exit Levels**: Intelligently calculates entry points, take-profit, stop-loss, and even a trailing stop based on market volatility (ATR) and Fibonacci levels.
- 🤖 **Telegram User Interface**: Interact with the bot easily through commands and inline buttons to get quick scans or comprehensive analyses.
- ⚙️ **Easy Configuration**: Manage the list of symbols, timeframes, and key settings through a `config.json` file and environment variables.
- 🗄️ **Internal Database**: Caches OHLCV data in a local SQLite database for faster access and to reduce API calls to the exchange.

### 🛠️ Getting Started

Follow these steps to get the bot up and running.

#### 1. Prerequisites
- Python 3.9+
- `pip` and `venv`
- Access to a crypto exchange supported by `ccxt` (the code is currently configured for CoinEx).

#### 2. Clone the Repository
```bash
git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
cd your-repository-name
```

#### 3. Create and Activate a Virtual Environment
- **On Windows:**
  ```bash
  python -m venv venv
  .\venv\Scripts\activate
  ```
- **On macOS/Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

#### 4. Install Dependencies
It's recommended to create a `requirements.txt` file and add the following libraries:
```
pandas
numpy
tensorflow
scikit-learn
ccxt
python-telegram-bot[ext]
aiosqlite
requests
web3
```
Then, install them using:
```bash
pip install -r requirements.txt
```

#### 5. Configuration
This is the most critical step. Create a file named `.env` in the project's root directory and add your credentials:

```env
# Exchange API Keys (Example for CoinEx)
COINEX_API_KEY="YOUR_COINEX_API_KEY"
COINEX_SECRET="YOUR_COINEX_SECRET_KEY"

# Telegram Bot Token from @BotFather
BOT_TOKEN="YOUR_TELEGRAM_BOT_TOKEN"

# (Optional) API Key for News Sentiment Analysis
CRYPTOPANIC_KEY="YOUR_CRYPTOPANIC_API_KEY"

# (Optional) API Key for On-Chain Analysis
ALCHEMY_URL="YOUR_ALCHEMY_ETHEREUM_MAINNET_URL"
```
You can also edit the `config.json` file to modify the list of symbols, timeframes, and the minimum signal confidence score.

### ▶️ Usage
After completing the installation and configuration, run the bot with the following command:
```bash
python main.py
```
Navigate to your Telegram bot and send the `/start` command. You can request a quick scan (`1m`) or a comprehensive analysis across all timeframes using the inline buttons.

### 🤝 Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the Branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

### 📜 License
This project is distributed under the `GPL v3` License. See the `LICENSE` file for more information.

### ⚠️ Disclaimer
This project is for **educational and research purposes only**. It is not financial advice. The cryptocurrency market is extremely volatile and carries a high degree of risk. The authors of this project are not responsible for any potential profits or losses. **Always Do Your Own Research (DYOR)**.