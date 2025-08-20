import json
import os
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv
from logger_config import logger

from constants import SYMBOLS, TIME_FRAMES
load_dotenv()

class Config:
    COINEX_API_KEY = os.getenv("COINEX_API_KEY")
    COINEX_SECRET = os.getenv("COINEX_SECRET")
    CRYPTOPANIC_KEY = os.getenv("CRYPTOPANIC_KEY")
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TF_CPP_MIN_LOG_LEVEL = os.getenv("TF_CPP_MIN_LOG_LEVEL")
    TF_ENABLE_ONEDNN_OPTS = os.getenv("TF_ENABLE_ONEDNN_OPTS")
    
#===================================================================================#

class ConfigManager:
    DEFAULT_CONFIG = {'symbols': SYMBOLS, 'timeframes': TIME_FRAMES, 'min_confidence_score': 90, 'max_signals_per_timeframe': 5, 'risk_reward_threshold': 1.5}
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    return {**self.DEFAULT_CONFIG, **json.load(f)}
            except Exception as e:
                logger.warning(f"Error loading config: {e}. Using defaults.")
        return self.DEFAULT_CONFIG.copy()
    
    def save_config(self):
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def get(self, key: str, default=None):
        return self.config.get(key, default)
    
    def set(self, key: str, value):
        self.config[key] = value
        self.save_config()
