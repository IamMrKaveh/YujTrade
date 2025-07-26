import hashlib as hashlib
import asyncio as asyncio
import warnings as warnings
import sys as sys
import ccxt.async_support as ccxt
import os as os
import numpy as np
import pandas as pd
from numba import jit as jit
from functools import lru_cache as lru_cache
from logger_config import logger
from datetime import datetime as datetime
from telegram import Update as Update
from telegram.ext import ContextTypes as ContextTypes
import sys

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

NUMBA_AVAILABLE = True

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logger.warning("Numba not available. Using standard calculations.")
    
    # Define dummy jit decorator if numba is not available
    def jit():
        def decorator(func):
            return func
        return decorator


from telegram.ext import ApplicationBuilder, CommandHandler

from exchange.exchange_config import close_exchange
from exchange.exchange_config import SYMBOLS
from telegrams import start, status, show_symbols, help_command, BOT_TOKEN
from logger_config import logger