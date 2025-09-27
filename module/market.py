import asyncio
import json
from typing import Optional

import pandas as pd
import redis.asyncio as redis

from module.data_sources import BinanceFetcher, CoinDeskFetcher, MarketIndicesFetcher
from module.logger_config import logger
from module.utils import async_retry


class MarketDataProvider:
    def __init__(self, redis_client: Optional[redis.Redis] = None, 
                coindesk_fetcher: Optional[CoinDeskFetcher] = None,
                binance_fetcher: Optional[BinanceFetcher] = None,
                market_indices_fetcher: Optional[MarketIndicesFetcher] = None):
        self._is_closed = False
        self.redis = redis_client
        self.coindesk_fetcher = coindesk_fetcher
        self.binance_fetcher = binance_fetcher
        self.market_indices_fetcher = market_indices_fetcher


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
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                        df.set_index('timestamp', inplace=True)
                    return df
            except Exception as e:
                logger.warning(f"Redis GET failed for {cache_key}: {e}")

        df = pd.DataFrame()
        
        # 1. Try Binance
        if self.binance_fetcher:
            df = await self.binance_fetcher.get_historical_ohlc(symbol, timeframe, limit)

        # 2. Fallback to CoinDesk for major pairs if Binance fails or returns insufficient data
        if (df is None or len(df) < 100) and self.coindesk_fetcher:
            timeframe_lower = timeframe.lower()
            if 'd' in timeframe_lower or 'w' in timeframe_lower or 'm' in timeframe_lower:
                logger.warning(f"Binance fetch might be insufficient for {symbol}/{timeframe}. Trying CoinDesk.")
                try:
                    df_coindesk = await self.coindesk_fetcher.get_historical_ohlc(symbol, timeframe, limit)
                    if df_coindesk is not None and not df_coindesk.empty:
                        df = df_coindesk
                except Exception as e:
                    logger.warning(f"CoinDesk fetch failed for {symbol}/{timeframe}: {e}")

        # 3. Fallback to CoinGecko for other pairs
        if (df is None or len(df) < 100) and self.market_indices_fetcher:
            logger.warning(f"Primary sources failed for {symbol}/{timeframe}. Trying CoinGecko.")
            try:
                # Map timeframe to days for CoinGecko
                days = 365 if 'd' in timeframe.lower() else 90 if 'h' in timeframe.lower() else 30
                df_gecko = await self.market_indices_fetcher.get_historical_ohlc_coingecko(symbol, days=days)
                if df_gecko is not None and not df_gecko.empty:
                    df = df_gecko
            except Exception as e:
                logger.warning(f"CoinGecko fetch failed for {symbol}/{timeframe}: {e}")

        if df is not None and not df.empty:
            if self.redis:
                try:
                    df_to_cache = df.reset_index()
                    df_to_cache['timestamp'] = df_to_cache['timestamp'].astype(int) // 10**6
                    await self.redis.set(cache_key, df_to_cache.to_json(orient="split"), ex=120)
                except Exception as e:
                    logger.warning(f"Redis SET failed for {cache_key}: {e}")
            return df

        logger.error(f"Failed to fetch OHLCV from all available sources for {symbol} on {timeframe}.")
        return pd.DataFrame()