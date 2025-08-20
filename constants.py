import os
from logger_config import logger

TIME_FRAMES = [
    "1m",
    "5m",
    "15m",
    "30m",
    "1h",
    "4h",
    "6h",
    "12h",
    "1d",
    "1w",
    "1M",
]

MULTI_TF_CONFIRMATION_MAP = {
    "1m": ["5m", "15m"],
    "5m": ["15m", "30m"],
    "15m": ["30m", "1h"],
    "30m": ["1h", "4h"],
    "1h": ["4h", "6h"],
    "4h": ["6h", "12h"],
    "6h": ["12h", "1d"],
    "12h": ["1d", "1w"],
    "1d": ["1w", "1M"],
    "1w": ["1M"],
    "1M": []
}

MULTI_TF_CONFIRMATION_WEIGHTS = {
    "1m": {"5m": 0.4, "15m": 0.6},
    "5m": {"15m": 0.4, "30m": 0.6},
    "15m": {"30m": 0.4, "1h": 0.6},
    "30m": {"1h": 0.4, "4h": 0.6},
    "1h": {"4h": 0.4, "6h": 0.6},
    "4h": {"6h": 0.4, "12h": 0.6},
    "6h": {"12h": 0.4, "1d": 0.6},
    "12h": {"1d": 0.4, "1w": 0.6},
    "1d": {"1w": 0.4, "1M": 0.6},
    "1w": {"1M": 1.0},
    "1M": {}
}

def load_symbols():
    try:
        if not os.path.exists('symbols.txt'):
            logger.error("symbols.txt file not found")
            return []
            
        with open('symbols.txt', 'r', encoding='utf-8') as f:
            symbols = [line.strip().upper() for line in f.readlines() if line.strip()]
        
        symbols = list(set(symbols))
        
        if not symbols:
            logger.warning("No symbols found in symbols.txt")
            return []
            
        logger.info(f"Loaded {len(symbols)} symbols from symbols.txt")
        return symbols
    except Exception as e:
        logger.error(f"Error loading symbols: {e}")
        return []

SYMBOLS = load_symbols()