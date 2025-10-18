import json
import pandas as pd
import yfinance as yf
from typing import Dict

from config.logger import logger
from common.utils import async_retry
from common.cache import CacheKeyBuilder, CacheCategory
from data.sources.base_fetcher import BaseFetcher


class YFinanceFetcher(BaseFetcher):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.source_name = "yfinance"

    @async_retry(attempts=3, delay=5)
    async def get_traditional_indices(self) -> Dict:
        cache_key = CacheKeyBuilder.indices_key(self.source_name, "traditional")
        cache_ttl = self.config_manager.get_cache_ttl("indices")

        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)

        tickers = {
            "DXY": "DX-Y.NYB",
            "SPX": "^GSPC",
            "NDX": "^IXIC",
            "VIX": "^VIX",
            "GOLD": "GC=F",
            "OIL": "CL=F",
            "DJI": "^DJI",
            "RUT": "^RUT",
        }
        data = yf.download(
            tickers=list(tickers.values()),
            period="5d",
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        if data is None or data.empty:
            return {}

        price_column = "Close"
        if price_column not in data.columns:
            logger.error("Could not find 'Close' in yfinance data.")
            return {}

        indices = {
            key: float(data[price_column][ticker].iloc[-1])
            for key, ticker in tickers.items()
            if ticker in data[price_column]
            and not pd.isna(data[price_column][ticker].iloc[-1])
        }

        if self.redis and indices:
            await self.redis.set(cache_key, json.dumps(indices), ex=cache_ttl)
        return indices
