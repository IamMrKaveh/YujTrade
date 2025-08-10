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

COINEX_API_KEY = 'B443BA2F2AD8473C92118E71A2A20486'

COINEX_SECRET = '538792AAF1AD5F5329106EF4EC469AA7A1BF387E10A6116C'
