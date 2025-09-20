import json
import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

from module.constants import SYMBOLS, TIME_FRAMES
from module.logger_config import logger

load_dotenv()

def _read_env_var(key: str, default=None):
    v = os.getenv(key, default)
    if v is None:
        return default
    v = str(v).strip()
    if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
        v = v[1:-1]
    return v if v != "" else default

class Config:
    COINEX_API_KEY = _read_env_var("COINEX_API_KEY")
    COINEX_SECRET = _read_env_var("COINEX_SECRET")
    CRYPTOPANIC_KEY = _read_env_var("CRYPTOPANIC_KEY")
    TELEGRAM_BOT_TOKEN = _read_env_var("TELEGRAM_BOT_TOKEN")
    TF_CPP_MIN_LOG_LEVEL = _read_env_var("TF_CPP_MIN_LOG_LEVEL")
    TF_ENABLE_ONEDNN_OPTS = _read_env_var("TF_ENABLE_ONEDNN_OPTS")
    ALCHEMY_URL = _read_env_var("ALCHEMY_URL")

class ConfigManager:
    DEFAULT_CONFIG = {
        'symbols': SYMBOLS,
        'timeframes': TIME_FRAMES,
        'min_confidence_score': 90,
        'max_signals_per_timeframe': 5,
        'risk_reward_threshold': 1.5
    }
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return {**self.DEFAULT_CONFIG, **json.load(f)}
            except Exception as e:
                logger.warning(f"Error loading config: {e}. Using defaults.")
        return self.DEFAULT_CONFIG.copy()
    
    def save_config(self):
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def get(self, key: str, default=None):
        return self.config.get(key, default)
    
    def set(self, key: str, value):
        self.config[key] = value
        self.save_config()