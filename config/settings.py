# config/settings.py

from pydantic import BaseSettings, Field
from typing import Dict, List, Optional
import os
from pathlib import Path

class DatabaseSettings(BaseSettings):
    """تنظیمات پایگاه داده"""
    url: str = Field(default="sqlite:///./yujtrade.db", env="DATABASE_URL")
    echo: bool = Field(default=False, env="DATABASE_ECHO")
    pool_size: int = Field(default=10, env="DATABASE_POOL_SIZE")
    max_overflow: int = Field(default=20, env="DATABASE_MAX_OVERFLOW")
    
class TelegramSettings(BaseSettings):
    """تنظیمات تلگرام"""
    bot_token: str = Field(..., env="TELEGRAM_BOT_TOKEN")
    webhook_url: Optional[str] = Field(default=None, env="TELEGRAM_WEBHOOK_URL")
    use_webhook: bool = Field(default=False, env="TELEGRAM_USE_WEBHOOK")
    admin_chat_ids: List[int] = Field(default_factory=list, env="TELEGRAM_ADMIN_CHAT_IDS")
    max_retries: int = Field(default=3, env="TELEGRAM_MAX_RETRIES")
    timeout: int = Field(default=30, env="TELEGRAM_TIMEOUT")
    
class CoinexSettings(BaseSettings):
    """تنظیمات API کوینکس"""
    base_url: str = Field(default="https://api.coinex.com/v1", env="COINEX_BASE_URL")
    api_key: Optional[str] = Field(default=None, env="COINEX_API_KEY")
    secret_key: Optional[str] = Field(default=None, env="COINEX_SECRET_KEY")
    timeout: int = Field(default=30, env="COINEX_TIMEOUT")
    max_retries: int = Field(default=3, env="COINEX_MAX_RETRIES")
    rate_limit: int = Field(default=10, env="COINEX_RATE_LIMIT")  # درخواست در ثانیه
    
class TradingSettings(BaseSettings):
    """تنظیمات معاملاتی"""
    default_candle_limit: int = Field(default=60, env="TRADING_CANDLE_LIMIT")
    min_confidence_score: float = Field(default=70.0, env="TRADING_MIN_CONFIDENCE")
    max_signals_per_symbol: int = Field(default=3, env="TRADING_MAX_SIGNALS_PER_SYMBOL")
    risk_reward_ratio: float = Field(default=2.0, env="TRADING_RISK_REWARD_RATIO")
    enable_all_timeframes: bool = Field(default=True, env="TRADING_ENABLE_ALL_TIMEFRAMES")
    preferred_timeframes: List[str] = Field(
        default=["1h", "4h", "1d"], 
        env="TRADING_PREFERRED_TIMEFRAMES"
    )
    
class IndicatorSettings(BaseSettings):
    """تنظیمات اندیکاتورها"""
    rsi_period: int = Field(default=14, env="INDICATOR_RSI_PERIOD")
    rsi_overbought: float = Field(default=70.0, env="INDICATOR_RSI_OVERBOUGHT")
    rsi_oversold: float = Field(default=30.0, env="INDICATOR_RSI_OVERSOLD")
    
    macd_fast: int = Field(default=12, env="INDICATOR_MACD_FAST")
    macd_slow: int = Field(default=26, env="INDICATOR_MACD_SLOW")
    macd_signal: int = Field(default=9, env="INDICATOR_MACD_SIGNAL")
    
    ema_periods: List[int] = Field(default=[20, 50, 200], env="INDICATOR_EMA_PERIODS")
    sma_periods: List[int] = Field(default=[20, 50, 200], env="INDICATOR_SMA_PERIODS")
    
    bollinger_period: int = Field(default=20, env="INDICATOR_BOLLINGER_PERIOD")
    bollinger_std: float = Field(default=2.0, env="INDICATOR_BOLLINGER_STD")
    
    stoch_k_period: int = Field(default=14, env="INDICATOR_STOCH_K_PERIOD")
    stoch_d_period: int = Field(default=3, env="INDICATOR_STOCH_D_PERIOD")
    
    adx_period: int = Field(default=14, env="INDICATOR_ADX_PERIOD")
    atr_period: int = Field(default=14, env="INDICATOR_ATR_PERIOD")
    
class LoggingSettings(BaseSettings):
    """تنظیمات لاگ"""
    level: str = Field(default="INFO", env="LOG_LEVEL")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    file_path: str = Field(default="logs/yujtrade.log", env="LOG_FILE_PATH")
    max_file_size: int = Field(default=10485760, env="LOG_MAX_FILE_SIZE")  # 10MB
    backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")
    enable_console: bool = Field(default=True, env="LOG_ENABLE_CONSOLE")
    enable_file: bool = Field(default=True, env="LOG_ENABLE_FILE")
    
class RedisSettings(BaseSettings):
    """تنظیمات Redis"""
    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    db: int = Field(default=0, env="REDIS_DB")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    max_connections: int = Field(default=10, env="REDIS_MAX_CONNECTIONS")
    
class SecuritySettings(BaseSettings):
    """تنظیمات امنیتی"""
    secret_key: str = Field(..., env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="SECURITY_ALGORITHM")
    token_expire_minutes: int = Field(default=30, env="SECURITY_TOKEN_EXPIRE_MINUTES")
    
class Settings(BaseSettings):
    """تنظیمات اصلی برنامه"""
    
    # اطلاعات برنامه
    app_name: str = Field(default="YujTrade", env="APP_NAME")
    version: str = Field(default="1.0.0", env="APP_VERSION")
    description: str = Field(default="Advanced Crypto Trading Signal Bot", env="APP_DESCRIPTION")
    debug: bool = Field(default=False, env="DEBUG")
    
    # محیط اجرا
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # تنظیمات فرعی
    database: DatabaseSettings = DatabaseSettings()
    telegram: TelegramSettings = TelegramSettings()
    coinex: CoinexSettings = CoinexSettings()
    trading: TradingSettings = TradingSettings()
    indicators: IndicatorSettings = IndicatorSettings()
    logging: LoggingSettings = LoggingSettings()
    redis: RedisSettings = RedisSettings()
    security: SecuritySettings = SecuritySettings()
    
    # مسیرها
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data")
    logs_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "logs")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # ساخت پوشه‌های مورد نیاز
        self.data_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
    
    @property
    def is_production(self) -> bool:
        """آیا در محیط تولید هستیم؟"""
        return self.environment.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """آیا در محیط توسعه هستیم؟"""
        return self.environment.lower() == "development"
    
    def get_database_url(self) -> str:
        """دریافت URL پایگاه داده"""
        return self.database.url
    
    def get_redis_url(self) -> str:
        """دریافت URL Redis"""
        auth = f":{self.redis.password}@" if self.redis.password else ""
        return f"redis://{auth}{self.redis.host}:{self.redis.port}/{self.redis.db}"

# نمونه سینگلتون از تنظیمات
_settings: Optional[Settings] = None

def get_settings() -> Settings:
    """دریافت نمونه سینگلتون از تنظیمات"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

def reload_settings() -> Settings:
    """بارگذاری مجدد تنظیمات"""
    global _settings
    _settings = Settings()
    return _settings

# تنظیمات پیش‌فرض برای محیط‌های مختلف
ENVIRONMENT_SETTINGS = {
    "development": {
        "debug": True,
        "logging": {"level": "DEBUG"},
        "database": {"echo": True}
    },
    "production": {
        "debug": False,
        "logging": {"level": "WARNING"},
        "database": {"echo": False}
    },
    "testing": {
        "debug": True,
        "database": {"url": "sqlite:///:memory:"},
        "logging": {"level": "DEBUG"}
    }
}
