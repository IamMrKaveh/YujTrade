import asyncio
from typing import Optional, List, Union

import pandas as pd
import redis.asyncio as redis
import aiohttp

from module.data_sources import BinanceFetcher, CoinDeskFetcher, MarketIndicesFetcher
from module.exceptions import InsufficientDataError, NetworkError, DataError
from module.logger_config import logger
from module.utils import async_retry


class MarketDataProvider:
    def __init__(self, 
                 redis_client: Optional[redis.Redis] = None, 
                 coindesk_fetcher: Optional[CoinDeskFetcher] = None,
                 binance_fetcher: Optional[BinanceFetcher] = None,
                 market_indices_fetcher: Optional[MarketIndicesFetcher] = None,
                 session: Optional[aiohttp.ClientSession] = None):
        
        self._is_closed = False
        self.redis = redis_client
        
        self._owns_session = session is None
        self.session = session or aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        
        self.binance_fetcher = binance_fetcher or BinanceFetcher(
            redis_client=self.redis, 
            session=self.session
        )
        
        self.coindesk_fetcher = coindesk_fetcher
        
        self.market_indices_fetcher = market_indices_fetcher or MarketIndicesFetcher(
            redis_client=self.redis,
            session=self.session
        )
        
        self.fetchers: List[Union[BinanceFetcher, CoinDeskFetcher, MarketIndicesFetcher]] = [
            f for f in [self.binance_fetcher, self.coindesk_fetcher, self.market_indices_fetcher] 
            if f is not None
        ]

    async def close(self):
        if self._is_closed:
            return
        self._is_closed = True
        
        close_tasks = []
        for fetcher in self.fetchers:
            if hasattr(fetcher, "close") and callable(fetcher.close):
                try:
                    result = fetcher.close()
                    if asyncio.iscoroutine(result):
                        close_tasks.append(result)
                except Exception as e:
                    logger.warning(f"Failed closing fetcher {fetcher}: {e}")

        if self._owns_session and self.session and not self.session.closed:
            close_tasks.append(self.session.close())
            
        await asyncio.gather(*close_tasks, return_exceptions=True)
        logger.info("MarketDataProvider and all associated fetcher sessions closed.")

    @async_retry(attempts=3, delay=5, exceptions=(NetworkError, DataError))
    async def fetch_ohlcv_data(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        if self._is_closed:
            raise NetworkError("MarketDataProvider is closed")
        
        cache_ttl_map = {
            "1h": 120,
            "4h": 300,
            "1d": 600,
            "1w": 1800,
            "1M": 3600
        }
        cache_ttl = cache_ttl_map.get(timeframe, 120)
        
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
        
        fetch_sources = [
            (self.binance_fetcher.get_historical_ohlc, "binance") if self.binance_fetcher else None,
            (self.coindesk_fetcher.get_historical_ohlc, "coindesk") if self.coindesk_fetcher else None,
        ]
        
        fetch_sources = [s for s in fetch_sources if s is not None]

        for fetch_func, source_name in fetch_sources:
            try:
                df = await fetch_func(symbol, timeframe, limit)
                if df is not None and not df.empty and len(df) >= 100:
                    logger.debug(f"Fetched {len(df)} rows from {source_name} for {symbol}/{timeframe}")
                    break
            except (NetworkError, DataError) as e:
                logger.warning(f"Source {source_name} failed for {symbol}/{timeframe}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error from {source_name} for {symbol}/{timeframe}: {e}")

        if df is not None and not df.empty:
            if self.redis:
                try:
                    df_to_cache = df.reset_index()
                    df_to_cache['timestamp'] = df_to_cache['timestamp'].astype(int) // 10**6
                    await self.redis.set(cache_key, df_to_cache.to_json(orient="split"), ex=cache_ttl)
                except Exception as e:
                    logger.warning(f"Redis SET failed for {cache_key}: {e}")
            return df

        raise InsufficientDataError(
            f"Failed to fetch sufficient OHLCV for {symbol} on {timeframe} from all sources."
        )

