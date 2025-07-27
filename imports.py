import hashlib as hashlib
import asyncio as asyncio
import warnings as warnings
import sys as sys
import ccxt.async_support as ccxt
import os as os
import numpy as np
import pandas as pd
from numba import jit
from functools import lru_cache
from datetime import datetime
from telegram import Update
from telegram.ext import ContextTypes, ApplicationBuilder, CommandHandler

exchange = None

if not hasattr(np, 'NaN'):
    np.NaN = np.nan

# Now import pandas_ta after fixing numpy
try:
    import pandas_ta as ta
except ImportError as e:
    print(f"Error importing pandas_ta: {e}")
    print("Please install with: pip install pandas-ta")
    sys.exit(1)


    
from logger_config import logger

from exchange.exchange_config import close_exchange
from exchange.exchange_config import SYMBOLS
from telegrams.constants import BOT_TOKEN
from telegrams.handlers import start, status, help_command, show_symbols

from exchange.exchange_config import _validate_symbol_format, init_exchange
from exchange.timeframes import _get_optimal_limit, _validate_timeframe
from exchange.ohlcv import get_klines
from exchange.exchange_config import _validate_symbol_format, init_exchange

from market.main import analyze_market
from market.batch_processing import _process_all_symbols_in_batches
from market.signal_processing import _process_and_return_best_signal
from market.scoring import _calculate_combined_score
from market.dynamic_levels import _calculate_dynamic_levels_long, _calculate_dynamic_levels_short
from market.fibonacci import _get_nearby_fibonacci_levels
from market.indicators import _add_additional_indicators
from market.trend_volatility import _calculate_trend_direction, _calculate_volatility_metrics

from signals.core import check_signals
from signals.accuracy import calculate_signal_accuracy_score
from signals.indicators.indicator_management import calculate_indicators

from telegrams.constants import (
    OVER_SELL, OVER_BUY, BALANCED, STRONG_INFLOW, STRONG_OUTFLOW,
    NATURAL_ZONE, HIGH_VOLUME, MEDIUM_VOLUME, LOW_VOLUME,
    ASCENDING, DESCENDING, NO_TREND, WAIT_MESSAGE, WAIT_TOO_LONG_MESSAGE,
    NO_SIGNAL_FOUND, ERROR_MESSAGE,
    BEST_OPPORTUNITY_MESSAGE, TECHNICAL_ANALYZE, NEAR_FIBONACCI_LEVELS,
    SIGNAL_POINTS
)
from telegrams.message_builder import _build_status_message, _send_error_message, _send_status_message
from telegrams.system_info import _get_system_info, _test_exchange_connection
from telegrams.background import _background_analysis
from telegrams.indicators_status import (_get_cci_status, _get_mfi_status, _get_rsi_status,
                                _get_stoch_status, _get_trend_status, _get_volume_status,
                                _get_williams_status)