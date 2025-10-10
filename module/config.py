# config.py

import json
from pathlib import Path
from typing import Any, Dict, Optional

from decouple import config as decouple_config

from .constants import DEFAULT_INDICATOR_WEIGHTS, SYMBOLS, TIME_FRAMES, TIMEFRAME_BASED_INDICATOR_WEIGHTS
from .logger_config import logger
from .security import KeyEncryptor, get_password_from_key_manager


class Config:
    ENCRYPTION_PASSWORD = get_password_from_key_manager() or decouple_config("ENCRYPTION_PASSWORD", default=None)

    _encryptor = None
    if ENCRYPTION_PASSWORD:
        try:
            _encryptor = KeyEncryptor(ENCRYPTION_PASSWORD)
        except Exception:
            logger.critical("Failed to create encryptor. Check your ENCRYPTION_PASSWORD or key manager.")
            _encryptor = None

    @staticmethod
    def get_secret(key: str, cast: type = str) -> Optional[Any]:
        encrypted_key = f"ENCRYPTED_{key}"
        value = decouple_config(encrypted_key, default=None)
        if value and Config._encryptor:
            decrypted = Config._encryptor.decrypt(value)
            if not decrypted:
                logger.error(f"Failed to decrypt {key}. Please re-encrypt your keys.")
                return None
            try:
                return cast(decrypted)
            except (ValueError, TypeError):
                logger.error(f"Failed to cast decrypted key {key} to {cast}.")
                return None
        
        return decouple_config(key, default=None, cast=cast)


    TELEGRAM_BOT_TOKEN = get_secret("TELEGRAM_BOT_TOKEN")
    ADMIN_CHAT_ID = get_secret("ADMIN_CHAT_ID")
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


class ConfigManager:
    DEFAULT_CONFIG = {
        "symbols": SYMBOLS,
        "timeframes": TIME_FRAMES,
        "min_confidence_score": 0,
        "max_signals_per_timeframe": 1,
        "enable_scheduled_analysis": False,
        "schedule_hour": "*/1",
        "app_version": "4.1.0",
        "indicator_weights": DEFAULT_INDICATOR_WEIGHTS,
        "timeframe_based_weights": True,
        "soft_group_weights": {
            'momentum': 25, 'trend': 25, 'volatility': 25, 'volume': 25
        },
        "unanimity_threshold": 0.9,
        "current_timeframe": "1d"
    }

    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        loaded_config = {}
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    loaded_config = json.load(f)
            except Exception as e:
                logger.warning(f"Error loading config: {e}. Using defaults.")
        
        final_config = {**self.DEFAULT_CONFIG, **loaded_config}
        
        if 'indicator_weights' in loaded_config:
            final_config['indicator_weights'] = {
                **self.DEFAULT_CONFIG['indicator_weights'], 
                **loaded_config['indicator_weights']
            }
        
        return final_config

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
        
    def get_indicator_weights(self, timeframe: str) -> Dict[str, float]:
        if self.get("timeframe_based_weights", True):
            return TIMEFRAME_BASED_INDICATOR_WEIGHTS.get(timeframe, DEFAULT_INDICATOR_WEIGHTS)
        return self.get("indicator_weights", DEFAULT_INDICATOR_WEIGHTS)