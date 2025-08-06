# config/constants.py

import os
from enum import Enum
from typing import Dict, List, Tuple
from decimal import Decimal
from logger_config import logger

# =====================================================================

BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', 'token')

WAIT_MESSAGE =  "🔍 در حال تحلیل بازار برای یافتن بهترین فرصت معاملاتی\n" + \
                "⏳ این کار ممکن است چند دقیقه طول بکشد"
        
WAIT_TOO_LONG_MESSAGE = "⏳ تحلیل بیش از حد طول کشید. لطفاً دوباره تلاش کنید."

BEST_OPPORTUNITY_MESSAGE = "🎯 *بهترین فرصت معاملاتی یافت شده*\n" + f"{'='*10}\n\n"


TECHNICAL_ANALYZE = "📈 **تحلیل تکنیکال پیشرفته:**\n"

OVER_BUY = "🔴 (خرید بیش از حد)"

OVER_SELL = "🟢 (فروش بیش از حد)"

STRONG_OUTFLOW = "🟢 (جریان پول خروجی قوی)"
STRONG_INFLOW = "🔴 (جریان پول ورودی قوی)"

HIGH_VOLUME = "🟢 (حجم بالا)"
MEDIUM_VOLUME = "🟡 (حجم متوسط)"
LOW_VOLUME = "🔴 (حجم پایین)"

BALANCED = "🟡 (متعادل)"
NATURAL_ZONE = "🟡 (محدوده طبیعی)"

ASCENDING = "⬆️ (صعودی)"
DESCENDING = "⬇️ (نزولی)"
NO_TREND = "⚪️ (بدون جهت)"

NEAR_FIBONACCI_LEVELS = "\n🎯 **سطوح فیبوناچی نزدیک:**\n"

SIGNAL_POINTS = "\n🎯 **امتیاز سیگنال‌ها:**\n"

NO_SIGNAL_FOUND = "❌ متأسفانه در حال حاضر هیچ سیگنال معاملاتی با دقت بالا یافت نشد.\n\n" + \
                "🔍 **دلایل احتمالی:**\n" + \
                "• بازار در حالت تثبیت قرار دارد\n" + \
                "• شرایط تکنیکال مناسب معاملاتی وجود ندارد\n" + \
                "• همه سیگنال‌ها دارای ریسک بالا هستند\n\n" + \
                "💡 **پیشنهاد:**\n" + \
                "• 30-60 دقیقه دیگر مجدداً تلاش کنید\n" + \
                "• در انتظار شکل‌گیری الگوهای تکنیکال باشید\n" + \
                "• از معاملات پر ریسک خودداری کنید\n\n" + \
                "🔄 برای تحلیل مجدد /start را ارسال کنید."

ERROR_MESSAGE = "❌ خطایی در تحلیل بازار رخ داد. لطفا دوباره تلاش کنید.\n"

# =====================================================================

TIME_FRAMES = [
    "1m",
    "5m",
    "15m",
    "30m",
    "1h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
    "3d",
    "5d",
    "1w",
    "2w",
    "1M",
    "3M",
]

# انواع سیگنال
class SignalTypes(Enum):
    BUY = "buy"
    SELL = "sell"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"
    HOLD = "hold"

# سطوح اطمینان سیگنال
CONFIDENCE_LEVELS = {
    "LOW": (0, 50),
    "MEDIUM": (50, 75),
    "HIGH": (75, 90),
    "VERY_HIGH": (90, 100)
}

# رنگ‌های تلگرام
TELEGRAM_COLORS = {
    "BUY": "🟢",
    "SELL": "🔴",
    "STRONG_BUY": "💚",
    "STRONG_SELL": "❤️",
    "HOLD": "🟡",
    "WARNING": "⚠️",
    "INFO": "ℹ️",
    "SUCCESS": "✅",
    "ERROR": "❌"
}

def load_symbols():
    try:
        with open('symbols.txt', 'r', encoding='utf-8') as f:
            symbols = [line.strip().upper() for line in f.readlines() if line.strip()]
        logger.info(f"Loaded {len(symbols)} symbols from symbols.txt")
        return symbols
    except Exception as e:
        logger.error(f"Error loading symbols: {e}")

SYMBOLS = load_symbols()

# اندیکاتورهای فنی
class TechnicalIndicators:
    # اندیکاتورهای اسیلاتور
    RSI = "rsi"
    STOCHASTIC = "stochastic"
    CCI = "cci"
    WILLIAMS_R = "williams_r"
    
    # اندیکاتورهای روند
    MACD = "macd"
    EMA = "ema"
    SMA = "sma"
    BOLLINGER_BANDS = "bollinger_bands"
    
    # اندیکاتورهای حجم
    OBV = "obv"
    AD_LINE = "ad_line"
    CMF = "cmf"
    VWAP = "vwap"
    
    # اندیکاتورهای نوسان
    ATR = "atr"
    VOLATILITY = "volatility"

# وزن‌های اندیکاتورها در تولید سیگنال
INDICATOR_WEIGHTS = {
    TechnicalIndicators.RSI: 0.25,
    TechnicalIndicators.MACD: 0.30,
    TechnicalIndicators.EMA: 0.20,
    TechnicalIndicators.BOLLINGER_BANDS: 0.15,
    TechnicalIndicators.STOCHASTIC: 0.10,
}

# حدود نرمال اندیکاتورها
INDICATOR_THRESHOLDS = {
    TechnicalIndicators.RSI: {
        "oversold": 30,
        "overbought": 70,
        "strong_oversold": 20,
        "strong_overbought": 80
    },
    TechnicalIndicators.STOCHASTIC: {
        "oversold": 20,
        "overbought": 80
    },
    TechnicalIndicators.CCI: {
        "oversold": -100,
        "overbought": 100,
        "strong_oversold": -200,
        "strong_overbought": 200
    },
    TechnicalIndicators.WILLIAMS_R: {
        "oversold": -80,
        "overbought": -20
    }
}

# تنظیمات ریسک
RISK_MANAGEMENT = {
    "DEFAULT_STOP_LOSS_PERCENT": Decimal("2.0"),  # 2%
    "DEFAULT_TAKE_PROFIT_PERCENT": Decimal("4.0"),  # 4%
    "MAX_RISK_PER_TRADE": Decimal("2.0"),  # 2% از سرمایه
    "MAX_OPEN_POSITIONS": 5,
    "MIN_RISK_REWARD_RATIO": Decimal("2.0"),  # حداقل نسبت ریسک به ریوارد
}

# تنظیمات تلگرام
TELEGRAM_SETTINGS = {
    "MAX_MESSAGE_LENGTH": 4096,
    "MAX_CAPTION_LENGTH": 1024,
    "MAX_BUTTONS_PER_ROW": 3,
    "MAX_INLINE_BUTTONS": 100,
    "PARSE_MODE": "HTML",
    "DISABLE_WEB_PAGE_PREVIEW": True,
}

# قالب‌های پیام
MESSAGE_TEMPLATES = {
    "SIGNAL": """
🚀 <b>سیگنال {signal_type}</b> 
━━━━━━━━━━━━━━━━━━━━━━━━━━━
💰 <b>نماد:</b> {symbol} {emoji}
⏰ <b>تایم فریم:</b> {timeframe_persian}
📊 <b>قیمت فعلی:</b> ${current_price}
📈 <b>قیمت ورود:</b> ${entry_price}
🛑 <b>حد ضرر:</b> ${stop_loss}
🎯 <b>هدف:</b> ${take_profit}
💪 <b>اطمینان:</b> {confidence}%
📉 <b>نسبت ریسک/ریوارد:</b> 1:{risk_reward_ratio}
━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡️ <i>{timestamp}</i>
""",

    "MARKET_ANALYSIS": """
📊 <b>تحلیل بازار</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━
💰 <b>نماد:</b> {symbol} {emoji}
⏰ <b>تایم فریم:</b> {timeframe_persian}
📈 <b>روند:</b> {trend}
📊 <b>RSI:</b> {rsi}
📈 <b>MACD:</b> {macd_signal}
🎢 <b>Bollinger:</b> {bollinger_position}
━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡️ <i>{timestamp}</i>
""",

    "ERROR": """
❌ <b>خطا</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔴 {error_message}
⚡️ <i>{timestamp}</i>
""",

    "WELCOME": """
👋 <b>به ربات YujTrade خوش آمدید!</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚀 این ربات سیگنال‌های معاملاتی قدرتمند ارائه می‌دهد
📊 بر اساس تحلیل فنی پیشرفته
💎 کیفیت بالا و دقت قابل اطمینان
━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔹 /start - شروع مجدد
🔹 /help - راهنما
🔹 /settings - تنظیمات
🔹 /status - وضعیت
""",
}

# حداکثرها و حداقل‌ها
LIMITS = {
    "MAX_SYMBOLS_PER_USER": 50,
    "MAX_SIGNALS_PER_DAY": 100,
    "MAX_NOTIFICATIONS_PER_HOUR": 10,
    "MIN_CONFIDENCE_SCORE": 50.0,
    "MAX_CONFIDENCE_SCORE": 100.0,
    "MIN_TIMEFRAME_SECONDS": 60,  # 1 دقیقه
    "MAX_TIMEFRAME_SECONDS": 7776000,  # 3 ماه
}

# کدهای خطا
ERROR_CODES = {
    "API_ERROR": 1001,
    "NETWORK_ERROR": 1002,
    "VALIDATION_ERROR": 1003,
    "DATA_ERROR": 1004,
    "CALCULATION_ERROR": 1005,
    "TELEGRAM_ERROR": 1006,
    "DATABASE_ERROR": 1007,
    "CONFIGURATION_ERROR": 1008,
    "RATE_LIMIT_ERROR": 1009,
    "UNKNOWN_ERROR": 9999,
}

# وضعیت‌های سیستم
SYSTEM_STATUS = {
    "STARTING": "starting",
    "RUNNING": "running",
    "STOPPING": "stopping",
    "STOPPED": "stopped",
    "ERROR": "error",
    "MAINTENANCE": "maintenance",
}

# اولویت‌های سیگنال
SIGNAL_PRIORITIES = {
    "LOW": 1,
    "NORMAL": 2,
    "HIGH": 3,
    "CRITICAL": 4,
    "EMERGENCY": 5,
}

# فرمت‌های تاریخ و زمان
DATE_FORMATS = {
    "FULL": "%Y-%m-%d %H:%M:%S",
    "DATE": "%Y-%m-%d",
    "TIME": "%H:%M:%S",
    "PERSIAN_FULL": "%Y/%m/%d %H:%M:%S",
    "PERSIAN_DATE": "%Y/%m/%d",
}

# تنظیمات کش
CACHE_SETTINGS = {
    "CANDLES_TTL": 300,  # 5 دقیقه
    "INDICATORS_TTL": 180,  # 3 دقیقه
    "SIGNALS_TTL": 600,  # 10 دقیقه
    "USER_DATA_TTL": 3600,  # 1 ساعت
    "MARKET_DATA_TTL": 120,  # 2 دقیقه
}

# API endpoints کوینکس
COINEX_ENDPOINTS = {
    "KLINE": "/market/kline",
    "TICKER": "/market/ticker",
    "DEPTH": "/market/depth",
    "DEALS": "/market/deals",
    "CURRENCY_LIST": "/common/currency/list",
    "ASSET_CONFIG": "/common/asset/config",
}

# حالت‌های معاملاتی
TRADING_MODES = {
    "SIMULATION": "simulation",
    "PAPER": "paper", 
    "LIVE": "live",
    "BACKTEST": "backtest",
}
