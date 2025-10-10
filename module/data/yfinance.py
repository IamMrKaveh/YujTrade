import json
import pandas as pd
import yfinance as yf

from module.logger_config import logger
from module.utils import async_retry
from .base import BaseFetcher


class YFinanceFetcher(BaseFetcher):
    @async_retry(attempts=3, delay=5)
    async def get_traditional_indices(self):
        cache_key = "cache:trad_indices_yf"
        cache_ttl = 600
        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)

        tickers = {
            "DXY": "DX-Y.NYB", "SPX": "^GSPC", "NDX": "^IXIC", "VIX": "^VIX",
            "GOLD": "GC=F", "OIL": "CL=F", "DJI": "^DJI", "RUT": "^RUT"
        }
        data = yf.download(tickers=list(tickers.values()), period="5d", interval="1d", progress=False, auto_adjust=True)
        if data is None or data.empty:
            return {}

        price_column = "Close"
        if price_column not in data.columns:
            logger.error("Could not find 'Close' in yfinance data.")
            return {}

        indices = {
            key: float(data[price_column][ticker].iloc[-1]) for key, ticker in tickers.items() 
            if ticker in data[price_column] and not pd.isna(data[price_column][ticker].iloc[-1])
        }

        if self.redis and indices:
            await self.redis.set(cache_key, json.dumps(indices), ex=cache_ttl)
        return indices
