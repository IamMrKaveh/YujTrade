import asyncio
import json
from typing import Dict, List, Optional

import pandas as pd

from module.logger_config import logger
from module.core import FundamentalAnalysis
from module.exceptions import APIRateLimitError, NetworkError
from module.utils import RateLimiter, async_retry
from module.data.base import BaseFetcher


coingecko_rate_limiter = RateLimiter(max_requests=8, time_window=60)


class CoinGeckoFetcher(BaseFetcher):
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.base_url = "https://api.coingecko.com/api/v3"
        self.pro_base_url = "https://pro-api.coingecko.com/api/v3"
        self._coin_list: Optional[List[Dict]] = None
        self._coin_map: Optional[Dict[str, str]] = None

    async def _ensure_coin_list(self):
        if self._coin_list is None:
            coins = await self.get_coins_list()
            if coins:
                self._coin_list = coins
                self._coin_map = {c['symbol'].upper(): c['id'] for c in coins}

    async def _get_coin_id(self, symbol: str) -> Optional[str]:
        await self._ensure_coin_list()
        base_symbol = symbol.upper().split('/')[0]
        if self._coin_map:
            return self._coin_map.get(base_symbol, base_symbol.lower())
        return base_symbol.lower()
    
    def _get_cache_key(self, *args) -> str:
        key_parts = [self.__class__.__name__] + [str(arg) for arg in args]
        return "cache:" + ":".join(key_parts)

    @async_retry(attempts=3, delay=5, exceptions=(NetworkError, APIRateLimitError))
    async def _fetch_json(self, url, params=None, headers=None, limiter: Optional[RateLimiter] = None, endpoint_name: str = "default"):
        self._check_if_closed()
        if limiter:
            await limiter.wait_if_needed(endpoint_name)

        request_params = params.copy() if params else {}
        request_headers = headers.copy() if headers else {"accept": "application/json"}
        
        effective_url = url
        if self.api_key:
            effective_url = effective_url.replace(self.base_url, self.pro_base_url)
            if 'pro-api' in effective_url:
                request_headers['x-cg-pro-api-key'] = self.api_key
            else:
                request_params['x_cg_demo_api_key'] = self.api_key
        else:
            if 'pro-api' in url: # Should not happen if no key, but as a safeguard
                effective_url = url.replace(self.pro_base_url, self.base_url)


        async def _do_fetch():
            session = await self._get_session()
            async with session.get(effective_url, params=request_params, headers=request_headers) as response:
                if response.status == 429:
                    raise APIRateLimitError(f"Rate limit exceeded for {effective_url}")
                if response.status != 200:
                    logger.error(f"Error fetching from {effective_url}: {response.status} {await response.text()}")
                    raise NetworkError(f"HTTP Error {response.status} for {effective_url}")
                return await response.json()
        
        return await self.circuit_breaker.call(_do_fetch)

    async def get_historical_ohlc(self, symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
        coin_id = await self._get_coin_id(symbol)
        if not coin_id:
            return None
        url = f"{self.base_url}/coins/{coin_id}/ohlc"
        params = {'vs_currency': 'usd', 'days': str(days)}
        data = await self._fetch_json(url, params, limiter=coingecko_rate_limiter, endpoint_name=f"coins_ohlc_{coin_id}")
        
        if not data:
            return None
        
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        df['volume'] = 0.0
        return df

    async def get_fundamental_data(self, coin_id_or_symbol: str) -> Optional[FundamentalAnalysis]:
        coin_id = await self._get_coin_id(coin_id_or_symbol)
        if not coin_id:
            return None
        cache_key = self._get_cache_key("fundamental", coin_id)
        if self.redis:
            try:
                cached = await self.redis.get(cache_key)
                if cached:
                    data = json.loads(cached)
                    return FundamentalAnalysis(**data)
            except Exception as e:
                logger.warning(f"Redis GET failed for {cache_key}: {e}")

        try:
            url = f"{self.base_url}/coins/{coin_id}"
            params = {
                'localization': 'false', 
                'tickers': 'false', 
                'market_data': 'true', 
                'community_data': 'true', 
                'developer_data': 'true'
            }
            data = await self._fetch_json(url, params, limiter=coingecko_rate_limiter, endpoint_name=f"coins_{coin_id}_full")

            if not data:
                return None

            market_data = data.get('market_data', {})
            community_data = data.get('community_data', {})
            developer_data = data.get('developer_data', {})

            fundamental_data = FundamentalAnalysis(
                market_cap=market_data.get('market_cap', {}).get('usd', 0.0),
                total_volume=market_data.get('total_volume', {}).get('usd', 0.0),
                developer_score=developer_data.get('pull_requests_merged', 0) * 0.4 + developer_data.get('stars', 0) * 0.6,
                community_score=community_data.get('twitter_followers', 0)
            )

            if self.redis:
                await self.redis.set(cache_key, json.dumps(fundamental_data.__dict__), ex=43200)

            return fundamental_data
        except Exception as e:
            logger.warning(f"Could not fetch fundamental data for {coin_id}: {e}")
            return None
    
    async def get_coins_list(self) -> Optional[List[Dict]]:
        cache_key = self._get_cache_key("coins_list")
        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)

        url = f"{self.base_url}/coins/list"
        params = {'include_platform': 'false'}
        data = await self._fetch_json(url, params, limiter=coingecko_rate_limiter, endpoint_name="coins_list")
        
        if data and self.redis:
            await self.redis.set(cache_key, json.dumps(data), ex=86400)
        return data

    async def get_trending_searches(self) -> Optional[List[str]]:
        cache_key = self._get_cache_key("trending")
        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)
        
        url = f"{self.base_url}/search/trending"
        data = await self._fetch_json(url, limiter=coingecko_rate_limiter, endpoint_name="trending")
        
        if data and 'coins' in data:
            trending_symbols = [coin['item']['symbol'] for coin in data['coins']]
            if self.redis:
                await self.redis.set(cache_key, json.dumps(trending_symbols), ex=3600)
            return trending_symbols
        return None

    async def get_circulating_supply(self, coin_id_or_symbol: str) -> Optional[float]:
        self._check_if_closed()
        coin_id = await self._get_coin_id(coin_id_or_symbol)
        if not coin_id:
            return None
        try:
            url = f"{self.base_url}/coins/{coin_id}"
            params = {'market_data': 'true'}
            coin_data = await self._fetch_json(url, params, limiter=coingecko_rate_limiter, endpoint_name=f"coins_info_{coin_id}")

            if coin_data and 'market_data' in coin_data:
                circulating = coin_data['market_data'].get('circulating_supply')
                if circulating:
                    return float(circulating)
            
            return None
        except Exception as e:
            logger.error(f"Error fetching circulating supply for {coin_id}: {e}")
            return None