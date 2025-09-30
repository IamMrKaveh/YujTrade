# پروژه ربات تحلیل‌گر و سیگنال‌دهی ارزهای دیجیتال (Yuj Bot)

<div dir="rtl">

این پروژه یک ربات تلگرامی پیشرفته برای تحلیل بازار ارزهای دیجیتال و تولید سیگنال‌های معاملاتی (خرید/فروش) است. ربات با جمع‌آوری و تحلیل داده‌های گسترده از منابع مختلف، از جمله داده‌های تکنیکال، فاندامنتال، اقتصادی کلان، آن-چین و اخبار، سیگنال‌هایی با امتیاز اطمینان (Confidence Score) تولید می‌کند.

## 🌟 ویژگی‌های کلیدی

- **تحلیل چندجانبه**: ترکیب تحلیل تکنیکال، فاندامنتال، سنتیمنت بازار، داده‌های اقتصادی و مدل‌های یادگیری ماشین برای ارزیابی جامع.
- **منابع داده گسترده**:
  - **داده‌های بازار**: Binance, CoinDesk, CoinGecko
  - **شاخص‌های اقتصادی**: Alpha Vantage (CPI, نرخ بهره فدرال و...)
  - **شاخص‌های سنتی**: yfinance (DXY, S&P 500, VIX)
  - **اخبار و سنتیمنت**: CryptoPanic, CoinDesk, Alternative.me (شاخص ترس و طمع)
- **موتور تحلیل قدرتمند**:
  - **اندیکاتورهای تکنیکال**: استفاده از ده‌ها اندیکاتور محبوب از کتابخانه‌های `TA-Lib` و `Pandas-TA`.
  - **تحلیل شرایط بازار**: تشخیص روند (صعودی، نزولی، خنثی)، قدرت روند، نوسانات و شتاب روند.
  - **الگوهای شمعی**: شناسایی الگوهای کندل استیک ژاپنی با استفاده از `TA-Lib`.
  - **تحلیل چند تایم‌فریم (Multi-Timeframe)**: تأیید سیگنال‌ها در تایم‌فریم‌های بالاتر برای افزایش دقت.
  - **یادگیری ماشین (LSTM)**: استفاده از یک مدل شبکه عصبی LSTM برای پیش‌بینی قیمت و تأیید جهت سیگنال.
- **تحلیل داده‌های پیشرفته**:
  - **داده‌های مشتقات (Derivatives)**: تحلیل Funding Rate, Open Interest, و نسبت‌های Long/Short.
  - **تحلیل دفتر سفارشات (Order Book)**: بررسی عدم تعادل حجم خرید و فروش.
- **سیستم امتیازدهی و رتبه‌بندی**: هر سیگنال بر اساس مجموعه‌ای از معیارها و وزن‌های تعریف‌شده، یک **امتیاز اطمینان** دریافت کرده و سیگنال‌ها بر اساس آن رتبه‌بندی می‌شوند.
- **رابط کاربری تلگرام**:
  - ارائه سیگنال‌های برتر از طریق دستورات ساده.
  - قابلیت اجرای تحلیل‌های سریع (Quick Scan) یا جامع (Full Scan).
  - تحلیل‌های زمان‌بندی‌شده و ارسال خودکار نتایج.
- **امنیت**: رمزنگاری کلیدهای API برای جلوگیری از دسترسی غیرمجاز.
- **ماژولار و توسعه‌پذیر**: معماری ماژولار پروژه امکان افزودن منابع داده، اندیکاتورها و استراتژی‌های جدید را به‌سادگی فراهم می‌کند.

## 🏗️ معماری پروژه

پروژه از چندین ماژول اصلی تشکیل شده است:

- **`main.py`**: نقطه ورود اصلی برنامه که ربات تلگرام و تسک‌های زمان‌بندی‌شده را اجرا می‌کند.
- **`telegram.py`**: مسئول مدیریت تعاملات با کاربر از طریق تلگرام، فرمت‌بندی و ارسال پیام‌ها.
- **`tasks.py`**: مدیریت تسک‌های پس‌زمینه (Background Tasks) برای تحلیل‌های سنگین و جلوگیری از بلاک شدن ربات.
- **`signals.py`**: هسته اصلی تولید سیگنال. این ماژول ارکستراسیون تحلیل‌ها و محاسبه امتیاز اطمینان را بر عهده دارد.
- **`analyzers.py`**: شامل کلاس‌هایی برای تحلیل شرایط کلی بازار و شناسایی الگوهای کندل استیک.
- **`indicators.py`**: کتابخانه‌ای از کلاس‌های اندیکاتورهای تکنیکال.
- **`data_sources.py`**: مسئول دریافت داده از API‌های مختلف (Binance, CoinGecko و...).
- **`market.py`**: فراهم‌کننده داده‌های بازار با قابلیت کَش کردن (Caching) برای افزایش سرعت.
- **`lstm.py`**: مدیریت مدل‌های LSTM، شامل ساخت، آموزش، بارگذاری و پیش‌بینی.
- **`config.py`**: مدیریت تنظیمات پروژه از فایل‌های `.env` و `config.json`.
- **`security.py`**: ابزارهای رمزنگاری و امنیتی.

## 🚀 راهنمای راه‌اندازی

### پیش‌نیازها

- پایتون 3.10 یا بالاتر
- کتابخانه TA-Lib (برای نصب آن به [مستندات رسمی](https://github.com/mrjbq7/ta-lib) مراجعه کنید)
- Redis (برای کَش کردن داده‌ها)

### مراحل نصب

1.  **کلون کردن ریپازیتوری:**
    ```bash
    git clone https://github.com/IamMrKaveh/YujTrade.git
    cd YujTrade
    ```

2.  **ایجاد و فعال‌سازی محیط مجازی:**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **نصب وابستگی‌ها:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **پیکربندی متغیرهای محیطی:**
    - یک کپی از فایل `.env.example` (در صورت وجود) با نام `.env` بسازید یا فایل `.env` را مستقیماً ویرایش کنید.
    - کلیدهای API مورد نیاز را از سرویس‌های مربوطه دریافت کرده و در فایل `.env` قرار دهید:
      - `TELEGRAM_BOT_TOKEN`: توکن ربات تلگرام شما.
      - `TELEGRAM_CHAT_ID`: شناسه چت تلگرام برای دریافت گزارش‌ها.
      - `CRYPTOPANIC_KEY`, `ALPHA_VANTAGE_KEY`, `COINGECKO_KEY`, و ...

5.  **رمزنگاری کلیدهای API (اختیاری اما به‌شدت توصیه می‌شود):**
    برای افزایش امنیت، اسکریپت `encrypt_keys.py` را اجرا کنید. این اسکریپت از شما یک رمز عبور می‌خواهد و کلیدهای API موجود در فایل `.env` را رمزنگاری می‌کند.
    ```bash
    python encrypt_keys.py
    ```
    **مهم**: رمز عبور خود را به خاطر بسپارید و آن را در متغیر `ENCRYPTION_PASSWORD` در فایل `.env` قرار دهید.

6.  **پیکربندی نمادها و تایم‌فریم‌ها:**
    - لیست ارزهای مورد نظر خود را در فایل `symbols.txt` وارد کنید (هر نماد در یک خط، مثلاً `BTC/USDT`).
    - تنظیمات دیگر مانند تایم‌فریم‌ها، حداقل امتیاز اطمینان و... را در فایل `config.json` ویرایش کنید.

7.  **آموزش مدل‌های LSTM (اختیاری):**
    برای استفاده از قابلیت پیش‌بینی با مدل‌های یادگیری ماشین، می‌توانید مدل‌ها را با داده‌های تاریخی آموزش دهید.
    ```bash
    python train_models.py
    ```
    اگر این مرحله را انجام ندهید، ربات هنگام اولین تحلیل برای هر نماد، مدل مربوطه را به‌صورت خودکار آموزش خواهد داد.

8.  **اجرای ربات:**
    ```bash
    python main.py
    ```
    ربات اکنون فعال است و به دستورات در تلگرام پاسخ می‌دهد.

## 🤖 دستورات ربات در تلگرام

- **/start** یا **/help**: نمایش منوی اصلی و راهنمای دستورات.
- **⚡️ Quick Scan (1h)**: اجرای یک تحلیل سریع بر روی تایم‌فریم ۱ ساعته و نمایش بهترین سیگنال یافت‌شده.
- **🚀 Full Scan**: اجرای یک تحلیل جامع بر روی تمام تایم‌fریم‌های تعریف‌شده در `config.json` و نمایش بهترین سیگنال در میان تمام نتایج.
- **⚙️ Configuration**: مشاهده تنظیمات فعلی ربات.

## 🤝 مشارکت در پروژه

از مشارکت شما در این پروژه استقبال می‌کنیم. برای ارائه پیشنهاد، گزارش باگ یا افزودن ویژگی‌های جدید، لطفاً یک Issue جدید ثبت کنید یا یک Pull Request ارسال نمایید.

## 📜 مجوز (License)

این پروژه تحت مجوز **GNU General Public License v3.0** منتشر شده است. برای اطلاعات بیشتر فایل [LICENSE](LICENSE) را مطالعه کنید.

</div>

---

# Crypto Trading Signal Bot (Yuj Bot)

<div dir="ltr">

This project is an advanced Telegram bot for analyzing the cryptocurrency market and generating trading signals (Buy/Sell). The bot produces signals with a Confidence Score by collecting and analyzing extensive data from various sources, including technical, fundamental, macroeconomic, on-chain, and news data.

## 🌟 Key Features

- **Multi-faceted Analysis**: Combines technical, fundamental, market sentiment, economic data, and machine learning models for a comprehensive evaluation.
- **Extensive Data Sources**:
  - **Market Data**: Binance, CoinDesk, CoinGecko
  - **Economic Indicators**: Alpha Vantage (CPI, Fed Rate, etc.)
  - **Traditional Indices**: yfinance (DXY, S&P 500, VIX)
  - **News & Sentiment**: CryptoPanic, CoinDesk, Alternative.me (Fear & Greed Index)
- **Powerful Analysis Engine**:
  - **Technical Indicators**: Utilizes dozens of popular indicators from the `TA-Lib` and `Pandas-TA` libraries.
  - **Market Condition Analysis**: Determines trend (bullish, bearish, sideways), trend strength, volatility, and trend acceleration.
  - **Candlestick Patterns**: Identifies Japanese candlestick patterns using `TA-Lib`.
  - **Multi-Timeframe Analysis**: Confirms signals on higher timeframes to increase accuracy.
  - **Machine Learning (LSTM)**: Uses an LSTM neural network model to predict prices and confirm signal direction.
- **Advanced Data Analysis**:
  - **Derivatives Data**: Analyzes Funding Rate, Open Interest, and Long/Short ratios.
  - **Order Book Analysis**: Examines buy/sell volume imbalances.
- **Scoring and Ranking System**: Each signal receives a **Confidence Score** based on a set of predefined criteria and weights, and signals are ranked accordingly.
- **Telegram User Interface**:
  - Delivers top signals through simple commands.
  - Ability to run Quick Scans or Full Scans.
  - Scheduled analysis with automatic result delivery.
- **Security**: Encrypts API keys to prevent unauthorized access.
- **Modular and Extensible**: The project's modular architecture makes it easy to add new data sources, indicators, and strategies.

## 🏗️ Project Architecture

The project consists of several main modules:

- **`main.py`**: The main entry point that runs the Telegram bot and scheduled tasks.
- **`telegram.py`**: Manages user interactions via Telegram, formatting, and sending messages.
- **`tasks.py`**: Handles background tasks for heavy analyses to prevent the bot from blocking.
- **`signals.py`**: The core signal generation logic. This module orchestrates the analyses and calculates the confidence score.
- **`analyzers.py`**: Contains classes for analyzing overall market conditions and identifying candlestick patterns.
- **`indicators.py`**: A library of technical indicator classes.
- **`data_sources.py`**: Responsible for fetching data from various APIs (Binance, CoinGecko, etc.).
- **`market.py`**: A market data provider with caching capabilities to improve performance.
- **`lstm.py`**: Manages LSTM models, including creation, training, loading, and prediction.
- **`config.py`**: Manages project settings from `.env` and `config.json` files.
- **`security.py`**: Provides encryption and security utilities.

## 🚀 Setup Guide

### Prerequisites

- Python 3.10 or higher
- TA-Lib library (refer to the [official documentation](https://github.com/mrjbq7/ta-lib) for installation)
- Redis (for data caching)

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/IamMrKaveh/YujTrade.git
    cd YujTrade
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure environment variables:**
    - Create a copy of `.env.example` (if it exists) named `.env`, or edit the `.env` file directly.
    - Obtain the required API keys from the respective services and place them in the `.env` file:
      - `TELEGRAM_BOT_TOKEN`: Your Telegram bot token.
      - `TELEGRAM_CHAT_ID`: The Telegram chat ID for receiving reports.
      - `CRYPTOPANIC_KEY`, `ALPHA_VANTAGE_KEY`, `COINGECKO_KEY`, etc.

5.  **Encrypt API Keys (Optional but highly recommended):**
    For enhanced security, run the `encrypt_keys.py` script. It will ask for a password and encrypt the API keys in your `.env` file.
    ```bash
    python encrypt_keys.py
    ```
    **Important**: Remember your password and set it as the `ENCRYPTION_PASSWORD` variable in the `.env` file.

6.  **Configure symbols and timeframes:**
    - Add your desired cryptocurrency symbols to the `symbols.txt` file (one per line, e.g., `BTC/USDT`).
    - Edit other settings like timeframes, minimum confidence score, etc., in the `config.json` file.

7.  **Train LSTM Models (Optional):**
    To use the machine learning prediction feature, you can train the models with historical data.
    ```bash
    python train_models.py
    ```
    If you skip this step, the bot will automatically train the relevant model during its first analysis for each symbol.

8.  **Run the bot:**
    ```bash
    python main.py
    ```
    The bot is now active and will respond to commands in Telegram.

## 🤖 Bot Commands in Telegram

- **/start** or **/help**: Displays the main menu and command guide.
- **⚡️ Quick Scan (1h)**: Runs a quick analysis on the 1-hour timeframe and shows the best signal found.
- **🚀 Full Scan**: Performs a comprehensive analysis across all timeframes defined in `config.json` and displays the best signal among all results.
- **⚙️ Configuration**: View the bot's current settings.

## 🤝 Contributing

Contributions are welcome! To make a suggestion, report a bug, or add a new feature, please open a new Issue or submit a Pull Request.

## 📜 License

This project is licensed under the **GNU General Public License v3.0**. See the [LICENSE](LICENSE) file for more details.

</div>