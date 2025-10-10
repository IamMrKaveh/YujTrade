# market.py

import asyncio
from typing import Optional, List, Union, Tuple
from io import StringIO
import pandas as pd
import redis.asyncio as redis
import aiohttp

from .data.binance import BinanceFetcher
from .data.coindesk import CoinDeskFetcher
from .data.marketindices import MarketIndicesFetcher
from .exceptions import InsufficientDataError, NetworkError, DataError
from .logger_config import logger
from .utils import async_retry
from .data_validator import DataQualityChecker


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
        
        self.data_quality_checker = DataQualityChecker()

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

    def _get_data_quality_score(self, df: pd.DataFrame, timeframe: str) -> float:
        if df is None or df.empty:
            return 0.0
        
        is_valid, _ = self.data_quality_checker.validate_data_quality(df, timeframe)
        if not is_valid:
            return 0.0

        score = 100.0
        
        try:
            no_gaps, _ = self.data_quality_checker.detect_data_gaps(df)
            if no_gaps is False:
                 score -= 30.0
        except ValueError:
            return 0.0

        if not self.data_quality_checker.check_sufficient_volume(df):
            score -= 20.0
            
        score -= df.isnull().sum().sum() * 0.1

        return score

    @async_retry(attempts=3, delay=5, exceptions=(NetworkError, DataError))
    async def fetch_ohlcv_data(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        if self._is_closed:
            raise NetworkError("MarketDataProvider is closed")
        
        cache_ttl_map = {
            "1h": 120, "4h": 300, "1d": 600, "1w": 1800, "1M": 3600
        }
        cache_ttl = cache_ttl_map.get(timeframe, 120)
        
        cache_key = f"cache:ohlcv:{symbol.replace('/', '_')}:{timeframe}:{limit}"
        if self.redis:
            try:
                cached = await self.redis.get(cache_key)
                if cached:
                    df = pd.read_json(StringIO(cached), orient="split")
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                        df.set_index('timestamp', inplace=True)
                    is_valid, _ = self.data_quality_checker.validate_data_quality(df, timeframe)
                    if not is_valid:
                        logger.debug(f"Cached data for {symbol}-{timeframe} is invalid, refetching")
                    else:
                        return df
            except Exception as e:
                logger.warning(f"Redis GET failed for {cache_key}: {e}")

        fetch_sources = [
            (self.binance_fetcher.get_historical_ohlc, "binance") if self.binance_fetcher else None,
            (self.coindesk_fetcher.get_historical_ohlc, "coindesk") if self.coindesk_fetcher else None,
        ]
        
        fetch_sources = [s for s in fetch_sources if s is not None]
        
        tasks = [fetch_func(symbol, timeframe, limit) for fetch_func, _ in fetch_sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        best_df = None
        best_score = -1.0
        
        for i, res_df in enumerate(results):
            source_name = fetch_sources[i][1]
            if isinstance(res_df, Exception) or res_df is None or res_df.empty:
                logger.warning(f"Source {source_name} failed for {symbol}/{timeframe}: {res_df}")
                continue

            if isinstance(res_df.index, pd.DatetimeIndex):
                res_df.index = res_df.index.tz_convert('UTC') if res_df.index.tz is not None else res_df.index.tz_localize('UTC')
            else:
                if 'timestamp' in res_df.columns:
                    res_df['timestamp'] = pd.to_datetime(res_df['timestamp'], unit='ms', utc=True)
                    res_df.set_index('timestamp', inplace=True)
                else:
                    logger.warning(f"No timestamp column in data from {source_name}, skipping")
                    continue

            score = self._get_data_quality_score(res_df, timeframe)
            logger.debug(f"Data quality score for {symbol}/{timeframe} from {source_name}: {score:.2f}")

            if score > best_score:
                best_score = score
                best_df = res_df
        
        if best_df is not None and not best_df.empty:
            if self.redis:
                try:
                    df_to_cache = best_df.reset_index()
                    df_to_cache['timestamp'] = df_to_cache['timestamp'].astype(int) // 10**6
                    await self.redis.set(cache_key, df_to_cache.to_json(orient="split"), ex=cache_ttl)
                except Exception as e:
                    logger.warning(f"Redis SET failed for {cache_key}: {e}")
            return best_df

        raise InsufficientDataError(
            f"Failed to fetch sufficient OHLCV for {symbol} on {timeframe} from all sources."
        )