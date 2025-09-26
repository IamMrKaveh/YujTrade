import asyncio
import json
from typing import Optional

import pandas as pd
import redis.asyncio as redis

from module.data_sources import BinanceFetcher, CoinDeskFetcher
from module.logger_config import logger
from module.utils import async_retry


class MarketDataProvider:
    def __init__(self, redis_client: Optional[redis.Redis] = None, 
                 coindesk_fetcher: Optional[CoinDeskFetcher] = None,
                 binance_fetcher: Optional[BinanceFetcher] = None):
        self._is_closed = False
        self.redis = redis_client
        self.coindesk_fetcher = coindesk_fetcher
        self.binance_fetcher = binance_fetcher

    async def close(self):
        if self._is_closed:
            return
        self._is_closed = True
        logger.info("MarketDataProvider closed.")

    @async_retry(attempts=3, delay=5)
    async def fetch_ohlcv_data(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        if self._is_closed:
            return pd.DataFrame()
        
        cache_key = f"cache:ohlcv:{symbol.replace('/', '_')}:{timeframe}:{limit}"
        if self.redis:
            try:
                cached = await self.redis.get(cache_key)
                if cached:
                    df = pd.read_json(cached, orient="split")
                    if not isinstance(df.index, pd.DatetimeIndex):
                         df.index = pd.to_datetime(df.index, unit='ms', utc=True)
                    else:
                         df.index = df.index.tz_localize('UTC') if df.index.tz is None else df.index.tz_convert('UTC')
                    return df
            except Exception as e:
                logger.warning(f"Redis GET failed for {cache_key}: {e}")

        # Always prefer Binance for OHLCV as it's more reliable for all timeframes
        if self.binance_fetcher:
            df = await self.binance_fetcher.get_historical_ohlc(symbol, timeframe, limit)
            if df is not None and not df.empty:
                if self.redis:
                    try:
                        await self.redis.set(cache_key, df.to_json(orient="split"), ex=60)
                    except Exception as e:
                        logger.warning(f"Redis SET failed for {cache_key}: {e}")
                return df

        # Fallback to CoinDesk only if Binance fails, and only for daily timeframes
        if self.coindesk_fetcher and 'd' in timeframe:
            logger.warning(f"Binance fetch failed for {symbol}. Falling back to CoinDesk for daily data.")
            df = await self.coindesk_fetcher.get_historical_ohlc(symbol, timeframe, limit)
            if df is not None and not df.empty:
                if self.redis:
                    try:
                        await self.redis.set(cache_key, df.to_json(orient="split"), ex=300)
                    except Exception as e:
                        logger.warning(f"Redis SET failed for {cache_key}: {e}")
                return df

        logger.error(f"Failed to fetch OHLCV from all available sources for {symbol} on {timeframe}.")
        return pd.DataFrame()