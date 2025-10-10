import json
from typing import Any, Dict, List, Optional

import aiohttp
import pandas as pd
import redis
from module.logger_config import logger
from module.circuit_breaker import CircuitBreaker
from module.exceptions import APIRateLimitError, NetworkError
from module.utils import RateLimiter, async_retry


coindesk_rate_limiter = RateLimiter(max_requests=20, time_window=60)


class CoinDeskFetcher:
    def __init__(self, api_key: str, redis_client: Optional[redis.Redis] = None, session: Optional[aiohttp.ClientSession] = None):
        self.api_key = api_key
        self.base_url = "https://data-api.coindesk.com"
        self.redis = redis_client
        self.circuit_breaker = CircuitBreaker()
        self.session = session
        self._external_session = session is not None
        self._is_closed = False

    def _check_if_closed(self):
        if self._is_closed:
            raise RuntimeError("CoinDeskFetcher has been closed and cannot be used")

    async def _get_session(self) -> aiohttp.ClientSession:
        self._check_if_closed()
        if self.session and not self.session.closed:
            return self.session
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        return self.session

    async def close(self):
        if self._is_closed:
            return
        self._is_closed = True
        if self.session and not self._external_session and not self.session.closed:
            await self.session.close()
            self.session = None

    @async_retry(attempts=3, delay=5, exceptions=(NetworkError, APIRateLimitError))
    async def _fetch(self, endpoint: str, params: Optional[Dict] = None):
        self._check_if_closed()
        headers = {"x-api-key": self.api_key}
        url = f"{self.base_url}/{endpoint}"
        await coindesk_rate_limiter.wait_if_needed(endpoint)
        
        async def _do_fetch():
            session = await self._get_session()
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 429:
                    raise APIRateLimitError("CoinDesk rate limit exceeded")
                if response.status >= 400:
                    raise NetworkError(f"HTTP Error {response.status} for {url}")
                return await response.json()
        
        return await self.circuit_breaker.call(_do_fetch)

    async def get_historical_ohlc(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        self._check_if_closed()
        try:
            timeframe_lower = timeframe.lower()
            if 'd' in timeframe_lower or 'w' in timeframe_lower or 'm' in timeframe_lower:
                endpoint = f"spot/v1/historical/ohlcv/{symbol.upper().replace('/USDT', '-USD')}/d"
                params = {"limit": limit}
            else:
                endpoint = f"spot/v1/historical/ohlcv/{symbol.upper().replace('/USDT', '-USD')}/h"
                params = {"time_frame": timeframe, "limit": limit}

            data = await self._fetch(endpoint, params)
            
            if not data or 'data' not in data or not data['data']:
                return None

            ohlc_data = data['data']
            if not ohlc_data:
                return None

            df = pd.DataFrame(ohlc_data)
            df.rename(columns={
                'ts': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'
            }, inplace=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            df = df.astype(float).sort_index()
            
            return df.tail(limit)
        except Exception as e:
            logger.error(f"Error fetching historical OHLC from CoinDesk for {symbol}: {e}")
            return None

    async def get_news(self, symbols: List[str]) -> List[Dict]:
        self._check_if_closed()
        try:
            all_news = []
            news_url = "https://api.coindesk.com/v1/news"
            params = {'slug': ",".join(s.lower().replace('-usdt', '') for s in symbols)}
            headers = {"X-CoinDesk-API-Key": self.api_key}
            
            session = await self._get_session()
            async with session.get(news_url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and 'data' in data:
                        all_news.extend(data['data'])
            
            unique_news = list({item['id']: item for item in all_news}.values())
            return unique_news
        except Exception as e:
            logger.error(f"Error fetching news from CoinDesk: {e}")
            return []

    async def get_bitcoin_price_index(self) -> Optional[Dict[str, Any]]:
        self._check_if_closed()
        cache_key = "cache:coindesk:bpi"
        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)

        try:
            session = await self._get_session()
            async with session.get("https://api.coindesk.com/v1/bpi/currentprice.json") as response:
                if response.status == 200:
                    data = await response.json()
                    if self.redis:
                        await self.redis.set(cache_key, json.dumps(data), ex=300)
                    return data
            return None
        except Exception as e:
            logger.error(f"Error fetching Bitcoin Price Index from CoinDesk: {e}")
            return None

    async def get_ohlcv_chart(self, symbol: str, start: Optional[str] = None, end: Optional[str] = None) -> Optional[Dict]:
        self._check_if_closed()
        cache_key = f"cache:coindesk:ohlcv:{symbol}:{start}:{end}"
        try:
            if self.redis:
                cached = await self.redis.get(cache_key)
                if cached:
                    return json.loads(cached)
        except Exception as e:
            logger.warning(f"Redis GET failed for {cache_key}: {e}")
        
        try:
            params = {}
            if start:
                params['start'] = start
            if end:
                params['end'] = end
            endpoint = f"spot/v1/historical/ohlcv/{symbol.upper().replace('/USDT', '-USD')}"
            data = await self._fetch(endpoint, params)
            if data and self.redis:
                await self.redis.set(cache_key, json.dumps(data), ex=3600)
            return data
        except Exception as e:
            logger.error(f"Error fetching OHLCV chart for {symbol}: {e}")
            return None

    async def get_price_history(self, symbol: str, period: str = "1d") -> Optional[Dict]:
        self._check_if_closed()
        try:
            endpoint = f"spot/v1/historical/price/{symbol.upper().replace('/USDT', '-USD')}/{period}"
            data = await self._fetch(endpoint)
            return data
        except Exception as e:
            logger.error(f"Error fetching price history for {symbol}: {e}")
            return None

    async def get_latest_news(self, limit: int = 20) -> List[Dict]:
        self._check_if_closed()
        cache_key = f"cache:coindesk:latest_news:{limit}"
        try:
            if self.redis:
                cached = await self.redis.get(cache_key)
                if cached:
                    return json.loads(cached)
        except Exception as e:
            logger.warning(f"Redis GET failed for {cache_key}: {e}")
        
        try:
            news_url = "https://api.coindesk.com/v1/news/latest"
            params = {'limit': limit}
            headers = {"X-CoinDesk-API-Key": self.api_key}
            
            session = await self._get_session()
            async with session.get(news_url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    news_items = data.get('data', [])
                    if self.redis and news_items:
                        await self.redis.set(cache_key, json.dumps(news_items), ex=600)
                    return news_items
            return []
        except Exception as e:
            logger.error(f"Error fetching latest news: {e}")
            return []

    async def get_trending_news(self) -> List[Dict]:
        self._check_if_closed()
        try:
            news_url = "https://api.coindesk.com/v1/news/trending"
            headers = {"X-CoinDesk-API-Key": self.api_key}
            
            session = await self._get_session()
            async with session.get(news_url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('data', [])
            return []
        except Exception as e:
            logger.error(f"Error fetching trending news: {e}")
            return []

    async def get_market_summary(self, symbol: str) -> Optional[Dict]:
        self._check_if_closed()
        try:
            endpoint = f"spot/v1/market/summary/{symbol.upper().replace('/USDT', '-USD')}"
            data = await self._fetch(endpoint)
            return data
        except Exception as e:
            logger.error(f"Error fetching market summary for {symbol}: {e}")
            return None
