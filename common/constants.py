# constants.py

import os
from ..config.logger import logger

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

TIMEFRAME_BASED_INDICATOR_WEIGHTS = {
    "1h": {
        'rsi': 10, 'macd': 10, 'stoch': 8, 'mfi': 8, 'cci': 7, 'williams_r': 6,
        'bb': 8, 'supertrend': 9, 'psar': 6, 'ichimoku': 9, 'kama': 7, 'vwma': 7,
        'volume': 7, 'cmf': 7, 'obv': 6, 'pvt': 6, 'ad_line': 6, 'force_index': 6,
        'keltner': 7, 'donchian': 6, 'trix': 7, 'eom': 6, 'std_dev': 6, 'stochrsi': 9,
        'kst': 7, 'mass': 6, 'corr_coef': 6, 'elder_ray': 7, 'pivot': 6, 'momentum': 8,
        'dpo': 6, 'choppiness': 7, 'vortex': 8, 'awesome': 8, 'cmo': 7, 'rvi': 7,
        'pvr': 7, 'ado': 6, 'bop': 7, 'linreg': 6, 'linreg_slope': 6, 'median_price': 5,
        'typical_price': 5, 'weighted_close': 5, 'hma': 8, 'zlema': 8, 't3': 8,
        'dema': 8, 'tema': 8, 'fisher': 9, 'stc': 9,
        'trend': 12, 'strength': 9, 'divergence': 10, 'pattern': 9,
        'fear_greed': 6, 'btc_dominance': 4, 'dxy': 4,
        'lstm': 10, 'xgboost': 9,
        'multi_tf': 18,
        'funding_rate': 8, 'taker_ratio': 9, 'top_trader_sentiment': 10,
        'liquidation': 9, 'order_book': 8,
        'fundamental': 7, 'trending': 6, 'macro': 7,
        'aroon': 7, 'uo': 7, 'roc': 7, 'squeeze': 9, 'vwap': 8, 'qqe': 8,
        'connors_rsi': 7, 'smi': 7, 'tsi': 7, 'gann_hilo': 7, 'ma_ribbon': 9,
        'fractal': 6, 'chaikin_vol': 6, 'historical_vol': 6, 'ulcer_index': 6,
        'atr_bands': 7, 'bbw': 7, 'volume_osc': 6, 'kvo': 7, 'frama': 8,
        'vidya': 8, 'mama': 8, 'rmi': 7, 'rsi2': 7, 'ppo': 7, 'pvo': 6,
        'nvi': 6, 'pvi': 6, 'mfi_bw': 6, 'ht_dc': 7, 'ht_trend_mode': 8,
        'er': 7, 'coppock': 7, 'adx': 9
    },
    
    "4h": {
        'rsi': 12, 'macd': 14, 'stoch': 8, 'mfi': 10, 'cci': 8, 'williams_r': 7,
        'bb': 10, 'supertrend': 12, 'psar': 8, 'ichimoku': 13, 'kama': 10, 'vwma': 10,
        'volume': 8, 'cmf': 8, 'obv': 7, 'pvt': 7, 'ad_line': 7, 'force_index': 7,
        'keltner': 9, 'donchian': 9, 'trix': 10, 'eom': 7, 'std_dev': 7, 'stochrsi': 8,
        'kst': 10, 'mass': 6, 'corr_coef': 8, 'elder_ray': 8, 'pivot': 7, 'momentum': 8,
        'dpo': 7, 'choppiness': 8, 'vortex': 10, 'awesome': 10, 'cmo': 8, 'rvi': 8,
        'pvr': 8, 'ado': 7, 'bop': 8, 'linreg': 8, 'linreg_slope': 8, 'median_price': 6,
        'typical_price': 6, 'weighted_close': 6, 'hma': 10, 'zlema': 10, 't3': 10,
        'dema': 10, 'tema': 10, 'fisher': 9, 'stc': 10,
        'trend': 18, 'strength': 13, 'divergence': 14, 'pattern': 12,
        'fear_greed': 9, 'btc_dominance': 6, 'dxy': 6,
        'lstm': 14, 'xgboost': 12,
        'multi_tf': 22,
        'funding_rate': 9, 'taker_ratio': 10, 'top_trader_sentiment': 12,
        'liquidation': 10, 'order_book': 9,
        'fundamental': 11, 'trending': 8, 'macro': 10,
        'aroon': 9, 'uo': 8, 'roc': 8, 'squeeze': 9, 'vwap': 8, 'qqe': 9,
        'connors_rsi': 7, 'smi': 8, 'tsi': 9, 'gann_hilo': 9, 'ma_ribbon': 13,
        'fractal': 7, 'chaikin_vol': 7, 'historical_vol': 7, 'ulcer_index': 7,
        'atr_bands': 8, 'bbw': 8, 'volume_osc': 7, 'kvo': 8, 'frama': 10,
        'vidya': 10, 'mama': 10, 'rmi': 8, 'rsi2': 6, 'ppo': 9, 'pvo': 7,
        'nvi': 7, 'pvi': 7, 'mfi_bw': 7, 'ht_dc': 8, 'ht_trend_mode': 9,
        'er': 8, 'coppock': 10, 'adx': 12
    },
    
    "1d": {
        'rsi': 14, 'macd': 16, 'stoch': 7, 'mfi': 12, 'cci': 10, 'williams_r': 6,
        'bb': 12, 'supertrend': 15, 'psar': 10, 'ichimoku': 18, 'kama': 13, 'vwma': 13,
        'volume': 10, 'cmf': 10, 'obv': 9, 'pvt': 9, 'ad_line': 9, 'force_index': 9,
        'keltner': 11, 'donchian': 12, 'trix': 13, 'eom': 9, 'std_dev': 9, 'stochrsi': 6,
        'kst': 13, 'mass': 7, 'corr_coef': 11, 'elder_ray': 10, 'pivot': 8, 'momentum': 10,
        'dpo': 9, 'choppiness': 9, 'vortex': 12, 'awesome': 12, 'cmo': 9, 'rvi': 9,
        'pvr': 9, 'ado': 9, 'bop': 9, 'linreg': 11, 'linreg_slope': 11, 'median_price': 7,
        'typical_price': 7, 'weighted_close': 7, 'hma': 13, 'zlema': 13, 't3': 13,
        'dema': 13, 'tema': 13, 'fisher': 10, 'stc': 12,
        'trend': 25, 'strength': 17, 'divergence': 16, 'pattern': 14,
        'fear_greed': 12, 'btc_dominance': 8, 'dxy': 8,
        'lstm': 18, 'xgboost': 16,
        'multi_tf': 28,
        'funding_rate': 8, 'taker_ratio': 9, 'top_trader_sentiment': 11,
        'liquidation': 9, 'order_book': 8,
        'fundamental': 16, 'trending': 11, 'macro': 14,
        'aroon': 11, 'uo': 8, 'roc': 8, 'squeeze': 9, 'vwap': 7, 'qqe': 9,
        'connors_rsi': 6, 'smi': 8, 'tsi': 11, 'gann_hilo': 12, 'ma_ribbon': 17,
        'fractal': 10, 'chaikin_vol': 8, 'historical_vol': 9, 'ulcer_index': 9,
        'atr_bands': 10, 'bbw': 8, 'volume_osc': 7, 'kvo': 8, 'frama': 13,
        'vidya': 13, 'mama': 13, 'rmi': 8, 'rsi2': 4, 'ppo': 11, 'pvo': 7,
        'nvi': 9, 'pvi': 9, 'mfi_bw': 7, 'ht_dc': 10, 'ht_trend_mode': 13,
        'er': 10, 'coppock': 14, 'adx': 15
    },
    
    "1w": {
        'rsi': 15, 'macd': 18, 'stoch': 6, 'mfi': 13, 'cci': 11, 'williams_r': 5,
        'bb': 13, 'supertrend': 17, 'psar': 11, 'ichimoku': 22, 'kama': 15, 'vwma': 15,
        'volume': 11, 'cmf': 11, 'obv': 10, 'pvt': 10, 'ad_line': 10, 'force_index': 10,
        'keltner': 12, 'donchian': 14, 'trix': 15, 'eom': 10, 'std_dev': 10, 'stochrsi': 4,
        'kst': 15, 'mass': 7, 'corr_coef': 13, 'elder_ray': 11, 'pivot': 9, 'momentum': 11,
        'dpo': 10, 'choppiness': 9, 'vortex': 13, 'awesome': 13, 'cmo': 9, 'rvi': 9,
        'pvr': 9, 'ado': 10, 'bop': 9, 'linreg': 13, 'linreg_slope': 13, 'median_price': 8,
        'typical_price': 8, 'weighted_close': 8, 'hma': 15, 'zlema': 15, 't3': 15,
        'dema': 15, 'tema': 15, 'fisher': 10, 'stc': 13,
        'trend': 30, 'strength': 20, 'divergence': 18, 'pattern': 16,
        'fear_greed': 14, 'btc_dominance': 10, 'dxy': 10,
        'lstm': 20, 'xgboost': 18,
        'multi_tf': 32,
        'funding_rate': 6, 'taker_ratio': 7, 'top_trader_sentiment': 9,
        'liquidation': 7, 'order_book': 7,
        'fundamental': 20, 'trending': 13, 'macro': 18,
        'aroon': 12, 'uo': 8, 'roc': 8, 'squeeze': 8, 'vwap': 6, 'qqe': 9,
        'connors_rsi': 4, 'smi': 8, 'tsi': 12, 'gann_hilo': 14, 'ma_ribbon': 20,
        'fractal': 12, 'chaikin_vol': 9, 'historical_vol': 10, 'ulcer_index': 10,
        'atr_bands': 11, 'bbw': 8, 'volume_osc': 7, 'kvo': 8, 'frama': 15,
        'vidya': 15, 'mama': 15, 'rmi': 9, 'rsi2': 3, 'ppo': 12, 'pvo': 7,
        'nvi': 10, 'pvi': 10, 'mfi_bw': 7, 'ht_dc': 11, 'ht_trend_mode': 15,
        'er': 11, 'coppock': 16, 'adx': 17
    },
    
    "1M": {
        'rsi': 16, 'macd': 20, 'stoch': 5, 'mfi': 14, 'cci': 12, 'williams_r': 4,
        'bb': 14, 'supertrend': 19, 'psar': 12, 'ichimoku': 25, 'kama': 17, 'vwma': 17,
        'volume': 12, 'cmf': 12, 'obv': 11, 'pvt': 11, 'ad_line': 11, 'force_index': 11,
        'keltner': 13, 'donchian': 16, 'trix': 17, 'eom': 11, 'std_dev': 11, 'stochrsi': 3,
        'kst': 17, 'mass': 8, 'corr_coef': 15, 'elder_ray': 12, 'pivot': 10, 'momentum': 12,
        'dpo': 11, 'choppiness': 9, 'vortex': 14, 'awesome': 14, 'cmo': 9, 'rvi': 9,
        'pvr': 9, 'ado': 11, 'bop': 9, 'linreg': 15, 'linreg_slope': 15, 'median_price': 9,
        'typical_price': 9, 'weighted_close': 9, 'hma': 17, 'zlema': 17, 't3': 17,
        'dema': 17, 'tema': 17, 'fisher': 11, 'stc': 14,
        'trend': 35, 'strength': 23, 'divergence': 20, 'pattern': 18,
        'fear_greed': 16, 'btc_dominance': 12, 'dxy': 12,
        'lstm': 22, 'xgboost': 20,
        'multi_tf': 35,
        'funding_rate': 5, 'taker_ratio': 6, 'top_trader_sentiment': 8,
        'liquidation': 6, 'order_book': 6,
        'fundamental': 25, 'trending': 15, 'macro': 22,
        'aroon': 13, 'uo': 8, 'roc': 8, 'squeeze': 7, 'vwap': 5, 'qqe': 9,
        'connors_rsi': 3, 'smi': 8, 'tsi': 13, 'gann_hilo': 16, 'ma_ribbon': 23,
        'fractal': 14, 'chaikin_vol': 10, 'historical_vol': 11, 'ulcer_index': 11,
        'atr_bands': 12, 'bbw': 8, 'volume_osc': 7, 'kvo': 8, 'frama': 17,
        'vidya': 17, 'mama': 17, 'rmi': 9, 'rsi2': 2, 'ppo': 13, 'pvo': 7,
        'nvi': 11, 'pvi': 11, 'mfi_bw': 7, 'ht_dc': 12, 'ht_trend_mode': 17,
        'er': 12, 'coppock': 18, 'adx': 19
    }
}

DEFAULT_INDICATOR_WEIGHTS = TIMEFRAME_BASED_INDICATOR_WEIGHTS["1d"]

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