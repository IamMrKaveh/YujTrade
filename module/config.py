import getpass
import json
from pathlib import Path
from typing import Any, Dict, Optional

from decouple import config as decouple_config

from module.constants import SYMBOLS, TIME_FRAMES
from module.logger_config import logger
from module.security import KeyEncryptor


class Config:
    ENCRYPTION_PASSWORD = decouple_config("ENCRYPTION_PASSWORD", default=None)

    _encryptor = None
    if ENCRYPTION_PASSWORD:
        try:
            _encryptor = KeyEncryptor(ENCRYPTION_PASSWORD)
        except Exception:
            logger.critical("Failed to create encryptor. Check your ENCRYPTION_PASSWORD.")
            _encryptor = None

    @staticmethod
    def get_secret(key: str) -> Optional[str]:
        encrypted_key = f"ENCRYPTED_{key}"
        value = decouple_config(encrypted_key, default=None)
        if value and Config._encryptor:
            decrypted = Config._encryptor.decrypt(value)
            if not decrypted:
                logger.error(f"Failed to decrypt {key}. Please re-encrypt your keys.")
                return None
            return decrypted
        
        plain_value = decouple_config(key, default=None)
        return plain_value if isinstance(plain_value, str) else None


    TELEGRAM_BOT_TOKEN = get_secret("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = get_secret("TELEGRAM_CHAT_ID")
    CRYPTOPANIC_KEY = get_secret("CRYPTOPANIC_KEY")
    ALPHA_VANTAGE_KEY = get_secret("ALPHA_VANTAGE_KEY")
    COINGECKO_KEY = get_secret("COINGECKO_KEY")
    COINDESK_API_KEY = get_secret("COINDESK_API_KEY")
    MESSARI_API_KEY = get_secret("MESSARI_API_KEY")
    SENTRY_DSN = get_secret("SENTRY_DSN")

    TF_CPP_MIN_LOG_LEVEL = decouple_config("TF_CPP_MIN_LOG_LEVEL", default="3")
    TF_ENABLE_ONEDNN_OPTS = decouple_config("TF_ENABLE_ONEDNN_OPTS", default="0")
    
    REDIS_HOST = decouple_config("REDIS_HOST", default="localhost", cast=str)
    REDIS_PORT = decouple_config("REDIS_PORT", default=6379, cast=int)
    REDIS_TOKEN = decouple_config("REDIS_TOKEN", default="none", cast=str)

    PROMETHEUS_PORT = decouple_config("PROMETHEUS_PORT", default=9090, cast=int)


class ConfigManager:
    DEFAULT_CONFIG = {
        "symbols": SYMBOLS,
        "timeframes": TIME_FRAMES,
        "min_confidence_score": 0,
        "max_signals_per_timeframe": 1,
        "enable_scheduled_analysis": True,
        "schedule_hour": "*/1",
        "app_version": "3.0.0",
    }

    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return {**self.DEFAULT_CONFIG, **json.load(f)}
            except Exception as e:
                logger.warning(f"Error loading config: {e}. Using defaults.")
        return self.DEFAULT_CONFIG.copy()

    def save_config(self):
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving config: {e}")

    def get(self, key: str, default=None):
        return self.config.get(key, default)

    def set(self, key: str, value):
        self.config[key] = value
        self.save_config()