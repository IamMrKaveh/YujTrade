<div dir="rtl">

# 🤖 ربات تحلیلگر و معامله‌گر ارز دیجیتال

این پروژه یک ربات پیشرفته برای تحلیل بازار ارزهای دیجیتال و تولید سیگنال‌های معاملاتی است که با استفاده از پایتون و مجموعه‌ای از کتابخانه‌های قدرتمند توسعه داده شده است. ربات از طریق تلگرام با کاربر در ارتباط است و تحلیل‌های جامع و سریعی را ارائه می‌دهد.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-GPLv3-green.svg)](https://www.gnu.org/licenses/gpl-3.0)

---

## 🌟 ویژگی‌های اصلی

- **تحلیل چندبعدی**: ترکیب تحلیل تکنیکال، آن‌چین، سنتیمنتال و فاندامنتال برای تولید سیگنال‌های دقیق‌تر.
- **یادگیری ماشین (LSTM)**: استفاده از مدل‌های `LSTM` برای پیش‌بینی قیمت و افزایش اطمینان سیگنال‌ها.
- **ارتباط با تلگرام**: رابط کاربری ساده از طریق ربات تلگرام برای اجرای تحلیل‌ها و دریافت نتایج.
- **پردازش غیرهمزمان**: استفاده از `Celery` و `Redis` برای اجرای تحلیل‌های سنگین در پس‌زمینه بدون مسدود کردن ربات.
- **تحلیل چند تایم‌فریم (Multi-Timeframe)**: تأیید سیگنال‌ها با بررسی هم‌راستایی روند در تایم‌فریم‌های مختلف.
- **مدیریت پیکربندی پیشرفته**: قابلیت تنظیم پارامترهای ربات از طریق فایل‌های `.env` و `config.json`.
- **امنیت**: رمزنگاری کلیدهای API برای محافظت از اطلاعات حساس.
- **مانیتورینگ**: مانیتورینگ عملکرد ربات با استفاده از `Prometheus` و گزارش خطاها با `Sentry`.
- **بک‌تستینگ (Backtesting)**: قابلیت ارزیابی استراتژی‌ها با استفاده از داده‌های تاریخی (با `backtrader`).
- **بهینه‌سازی هایپرپارامترها**: استفاده از `Optuna` برای یافتن بهترین پارامترها برای مدل‌های یادگیری ماشین.

---

## 🏗️ معماری پروژه

پروژه به صورت ماژولار طراحی شده تا توسعه و نگهداری آن آسان باشد. در ادامه، بخش‌های اصلی شرح داده شده‌اند:

### ۱. هسته اصلی (`main.py`, `config.py`, `core.py`)
- **`main.py`**: نقطه ورود اصلی برنامه که تمام سرویس‌ها، ربات تلگرام، زمان‌بند (`APScheduler`) و سرور مانیتورینگ را راه‌اندازی می‌کند.
- **`config.py`**: مسئول مدیریت تنظیمات برنامه است. `Config` کلیدهای حساس را از متغیرهای محیطی می‌خواند و `ConfigManager` تنظیمات کاربر را از `config.json` مدیریت می‌کند.
- **`core.py`**: شامل ساختارهای داده اصلی (Data Classes) مانند `TradingSignal` و `MarketAnalysis` است.

### ۲. تحلیل تکنیکال (`indicators.py`, `analyzers.py`)
- **`indicators.py`**: مجموعه‌ای بزرگ از اندیکاتورهای تکنیکال (RSI, MACD, Bollinger Bands, ...) که با استفاده از کتابخانه‌های `TA-Lib` و `Pandas-TA` پیاده‌سازی شده‌اند.
- **`analyzers.py`**: کلاس‌هایی برای تحلیل الگوهای کندل استیک، دایورجنس‌ها، شرایط بازار (روند، قدرت، نوسان) و حجم معاملات.

### ۳. جمع‌آوری داده و سنتیمنت (`sentiment.py`)
این ماژول وظیفه جمع‌آوری داده از منابع خارجی را بر عهده دارد:
- **`ExchangeManager`**: ارتباط با صرافی (از طریق `CCXT`) برای دریافت داده‌های OHLCV، دفتر سفارشات (Order Book) و داده‌های مربوط به مشتقات (Derivatives). همچنین از `Redis` برای کش کردن داده‌ها استفاده می‌کند.
- **`NewsFetcher`**: دریافت شاخص ترس و طمع و آخرین اخبار از منابعی مانند CryptoPanic.
- **`MarketIndicesFetcher`**: دریافت داده‌های شاخص‌های بازار (مانند دامیننس بیت‌کوین) و داده‌های اقتصاد کلان.
- **`OnChainFetcher`**: دریافت داده‌های درون‌زنجیره‌ای از Glassnode.

### ۴. تولید سیگنال (`signals.py`)
قلب تپنده ربات که در آن سیگنال‌ها تولید می‌شوند:
- **`SignalGenerator`**: تمام تحلیل‌ها را ترکیب کرده و با ارزیابی شرایط، یک امتیاز اطمینان (Confidence Score) برای سیگنال‌های خرید یا فروش تولید می‌کند.
- **`MultiTimeframeAnalyzer`**: سیگنال‌ها را با بررسی تایم‌فریم‌های بالاتر اعتبارسنجی می‌کند.
- **`SignalRanking`**: سیگنال‌های تولید شده را بر اساس امتیاز و فاکتورهای دیگر رتبه‌بندی می‌کند.

### ۵. یادگیری ماشین (`lstm.py`, `train_models.py`)
- **`lstm.py`**: پیاده‌سازی مدل `LSTM` و یک `LSTMModelManager` برای مدیریت، آموزش و بارگذاری مدل‌های مختلف برای هر نماد و تایم‌فریم.
- **`train_models.py`**: اسکریپتی برای آموزش دسته‌ای تمام مدل‌های `LSTM` بر اساس داده‌های تاریخی.

### ۶. تعامل با کاربر (`telegram.py`, `tasks.py`)
- **`telegram.py`**: مدیریت کامل ربات تلگرام، شامل تعریف دستورات و دکمه‌ها و فرمت‌بندی پیام‌ها.
- **`tasks.py`**: تعریف وظایف `Celery` برای اجرای تحلیل‌های زمان‌بر در پس‌زمینه. این کار باعث می‌شود ربات همیشه پاسخگو بماند.

### ۷. ابزارهای کمکی (`utils.py`, `security.py`, `monitoring.py`)
- **`security.py`**: ابزارهایی برای رمزنگاری و محافظت از کلیدهای API.
- **`monitoring.py`**: تعریف متریک‌های `Prometheus` برای نظارت بر عملکرد برنامه.
- **`utils.py`**: شامل ابزارهای کاربردی مانند `RateLimiter` برای مدیریت درخواست‌ها به APIها.

---

## 🚀 راهنمای راه‌اندازی

۱. **پیش‌نیازها**:
   - پایتون نسخه 3.10 یا بالاتر
   - Redis
   - نصب TA-Lib (دستورالعمل نصب آن بسته به سیستم‌عامل متفاوت است)

۲. **نصب وابستگی‌ها**:
   ```bash
   pip install -r requirements.txt
   ```

۳. **پیکربندی**:
   - یک کپی از فایل `.env.example` (در صورت وجود) بسازید و نام آن را به `.env` تغییر دهید.
   - کلیدهای API خود را برای تلگرام، صرافی و سایر سرویس‌ها در فایل `.env` وارد کنید.
   - نمادهای مورد نظر خود را در فایل `symbols.txt` اضافه کنید.

۴. **رمزنگاری کلیدها (اختیاری ولی پیشنهادی)**:
   برای افزایش امنیت، اسکریپت زیر را اجرا کرده و یک رمز عبور برای رمزنگاری کلیدهای خود وارد کنید:
   ```bash
   python encrypt_keys.py
   ```

۵. **آموزش مدل‌ها (اختیاری)**:
   برای آموزش اولیه مدل‌های LSTM بر اساس داده‌های تاریخی، اسکریپت زیر را اجرا کنید:
   ```bash
   python train_models.py
   ```

۶. **اجرای برنامه**:
   - ابتدا سرور Celery را در یک ترمینال جداگانه اجرا کنید:
     ```bash
     celery -A module.celery_app worker --loglevel=info
     ```
   - سپس برنامه اصلی را اجرا کنید:
     ```bash
     python main.py
     ```

۷. **استفاده از ربات**:
   - ربات تلگرام خود را باز کرده و دستور `/start` را ارسال کنید.
   - از دکمه‌ها برای درخواست "تحلیل کامل" یا "اسکن سریع" استفاده کنید.

---
</div>

<br>
<hr>
<br>

<div dir="ltr">

# 🤖 Crypto Analyst & Trading Bot

This project is an advanced bot for analyzing the cryptocurrency market and generating trading signals, developed using Python and a suite of powerful libraries. The bot interacts with the user via Telegram, providing comprehensive and quick analyses.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-GPLv3-green.svg)](https://www.gnu.org/licenses/gpl-3.0)

---

## 🌟 Key Features

- **Multi-dimensional Analysis**: Combines technical, on-chain, sentimental, and fundamental analysis to generate more accurate signals.
- **Machine Learning (LSTM)**: Uses `LSTM` models to predict prices and increase signal confidence.
- **Telegram Integration**: Simple user interface via a Telegram bot to run analyses and receive results.
- **Asynchronous Processing**: Utilizes `Celery` and `Redis` to run heavy analyses in the background without blocking the bot.
- **Multi-Timeframe Analysis**: Confirms signals by checking for trend alignment across different timeframes.
- **Advanced Configuration Management**: Ability to set bot parameters through `.env` and `config.json` files.
- **Security**: Encrypts API keys to protect sensitive information.
- **Monitoring**: Monitors bot performance using `Prometheus` and reports errors with `Sentry`.
- **Backtesting**: Capability to evaluate strategies using historical data (with `backtrader`).
- **Hyperparameter Optimization**: Uses `Optuna` to find the best parameters for machine learning models.

---

## 🏗️ Project Architecture

The project is designed modularly to facilitate easy development and maintenance. The main components are described below:

### 1. Core (`main.py`, `config.py`, `core.py`)
- **`main.py`**: The main entry point of the application, which initializes all services, the Telegram bot, the scheduler (`APScheduler`), and the monitoring server.
- **`config.py`**: Responsible for managing application settings. `Config` reads sensitive keys from environment variables, and `ConfigManager` handles user settings from `config.json`.
- **`core.py`**: Contains the main data structures (Data Classes) like `TradingSignal` and `MarketAnalysis`.

### 2. Technical Analysis (`indicators.py`, `analyzers.py`)
- **`indicators.py`**: A large collection of technical indicators (RSI, MACD, Bollinger Bands, ...) implemented using the `TA-Lib` and `Pandas-TA` libraries.
- **`analyzers.py`**: Classes for analyzing candlestick patterns, divergences, market conditions (trend, strength, volatility), and trading volume.

### 3. Data Collection & Sentiment (`sentiment.py`)
This module is responsible for gathering data from external sources:
- **`ExchangeManager`**: Connects to the exchange (via `CCXT`) to fetch OHLCV, order book, and derivatives data. It also uses `Redis` for caching.
- **`NewsFetcher`**: Fetches the Fear & Greed Index and the latest news from sources like CryptoPanic.
- **`MarketIndicesFetcher`**: Fetches market index data (like Bitcoin dominance) and macroeconomic data.
- **`OnChainFetcher`**: Fetches on-chain data from Glassnode.

### 4. Signal Generation (`signals.py`)
The heart of the bot where signals are generated:
- **`SignalGenerator`**: Combines all analyses and generates a confidence score for buy or sell signals by evaluating conditions.
- **`MultiTimeframeAnalyzer`**: Validates signals by checking higher timeframes.
- **`SignalRanking`**: Ranks the generated signals based on their score and other factors.

### 5. Machine Learning (`lstm.py`, `train_models.py`)
- **`lstm.py`**: Implements the `LSTM` model and an `LSTMModelManager` to manage, train, and load different models for each symbol and timeframe.
- **`train_models.py`**: A script for batch-training all `LSTM` models based on historical data.

### 6. User Interaction (`telegram.py`, `tasks.py`)
- **`telegram.py`**: Manages the entire Telegram bot, including defining commands, buttons, and message formatting.
- **`tasks.py`**: Defines `Celery` tasks to run time-consuming analyses in the background, ensuring the bot remains responsive.

### 7. Utilities (`utils.py`, `security.py`, `monitoring.py`)
- **`security.py`**: Tools for encrypting and protecting API keys.
- **`monitoring.py`**: Defines `Prometheus` metrics for monitoring application performance.
- **`utils.py`**: Contains useful utilities like a `RateLimiter` for managing API requests.

---

## 🚀 Setup Guide

1. **Prerequisites**:
   - Python 3.10+
   - Redis
   - TA-Lib (installation instructions vary by OS)

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configuration**:
   - Create a copy of `.env.example` (if it exists) and rename it to `.env`.
   - Enter your API keys for Telegram, the exchange, and other services in the `.env` file.
   - Add your desired symbols to the `symbols.txt` file.

4. **Encrypt Keys (Optional but Recommended)**:
   For enhanced security, run the following script and enter a password to encrypt your keys:
   ```bash
   python encrypt_keys.py
   ```

5. **Train Models (Optional)**:
   To perform an initial training of the LSTM models on historical data, run the following script:
   ```bash
   python train_models.py
   ```

6. **Run the Application**:
   - First, run the Celery worker in a separate terminal:
     ```bash
     celery -A module.celery_app worker --loglevel=info
     ```
   - Then, run the main application:
     ```bash
     python main.py
     ```

7. **Using the Bot**:
   - Open your Telegram bot and send the `/start` command.
   - Use the buttons to request a "Full Analysis" or a "Quick Scan".

---
</div>