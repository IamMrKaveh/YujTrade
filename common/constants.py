import os
from enum import Enum
from config.logger import logger

TIME_FRAMES = [
    "1h",
    "4h",
    "1d",
    "1w",
    "1M",
]

LONG_TERM_CONFIG = {
    'focus_timeframes': ['1d', '1w', '1M'],
    'min_confidence_threshold': {
        '1h': 82,
        '4h': 78,
        '1d': 72,
        '1w': 68,
        '1M': 65
    },
    'timeframe_priority_weights': {
        '1M': 2.0,
        '1w': 1.6,
        '1d': 1.2,
        '4h': 0.8,
        '1h': 0.5
    },
    'require_higher_tf_confirmation': True,
    'min_trend_strength': 'MODERATE',
    'min_risk_reward_ratio': 2.8,
    'min_volume_surge': 1.4,
    'max_signals_per_run': 3,
    'min_data_points': {
        '1h': 600,
        '4h': 500,
        '1d': 300,
        '1w': 200,
        '1M': 150
    }
}

INDICATOR_GROUPS = {
    'momentum': ['rsi', 'stoch', 'cci', 'macd', 'roc', 'mfi', 'stochrsi', 'williams_r', 'uo', 'mom', 'ppo'],
    'trend': ['sma', 'ema', 'dema', 'tema', 'hma', 'vwma', 'supertrend', 'adx', 'aroon', 'psar', 'ichimoku', 'kama', 'ma_ribbon'],
    'volatility': ['bb', 'atr', 'kc', 'dc', 'bbw', 'atr_bands', 'ulcer_index'],
    'volume': ['volume', 'obv', 'cmf', 'ad_line', 'pvt', 'vwap', 'force_index', 'eom', 'kvo', 'pvo']
}

MULTI_TF_CONFIRMATION_MAP = {
    "1h": ["4h", "1d"],
    "4h": ["1d", "1w"],
    "1d": ["1w", "1M"],
    "1w": ["1M"],
    "1M": [],
}

MULTI_TF_CONFIRMATION_WEIGHTS = {
    "1h": {"4h": 0.6, "1d": 0.4},
    "4h": {"1d": 0.7, "1w": 0.3},
    "1d": {"1w": 0.8, "1M": 0.2},
    "1w": {"1M": 1.0},
    "1M": {}
}

SIGNAL_EXPIRY_BY_TIMEFRAME = {
    '1h': 12, '4h': 24, '1d': 72, '1w': 168, '1M': 720
}

DECAY_HALF_LIFE_BY_TIMEFRAME = {
    '1h': 24, '4h': 48, '1d': 168, '1w': 720, '1M': 2160
}

THRESHOLD_BOUNDS = {
    'volatility_factor_min': 0.8,
    'volatility_factor_max': 1.5,
    'floor_threshold': 40
}


class AnalysisComponent(Enum):
    TECHNICAL_ANALYSIS = "technical_analysis"
    MARKET_CONTEXT = "market_context"
    EXTERNAL_DATA = "external_data"
    ML_MODELS = "ml_models"


def load_symbols():
    try:
        path = 'symbols.txt'
        if not os.path.exists(path):
            logger.error("symbols.txt file not found")
            return []
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [line.rstrip('\n\r') for line in f]
        seen = set()
        symbols = []
        for line in lines:
            s = line.strip()
            if not s:
                continue
            s = s.upper()
            if s not in seen:
                seen.add(s)
                symbols.append(s)
        if not symbols:
            logger.warning("No symbols found in symbols.txt")
            return []
        logger.info(f"Loaded {len(symbols)} symbols from symbols.txt")
        return symbols
    except Exception as e:
        logger.error(f"Error loading symbols: {e}")
        return []


SYMBOLS = load_symbols()