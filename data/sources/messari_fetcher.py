import asyncio
import json
from typing import Any, Dict, List, Optional

import aiohttp
import redis

from config.logger import logger
from utils.circuit_breaker import CircuitBreaker
from common.core import OnChainAnalysis
from common.exceptions import APIRateLimitError, DataError, NetworkError
from module.utils import RateLimiter, async_retry


messari_rate_limiter = RateLimiter(max_requests=20, time_window=60)


class MessariFetcher:
    def __init__(self, api_key: str, redis_client: Optional[redis.Redis] = None, session: Optional[aiohttp.ClientSession] = None):
        self.api_key = api_key
        self.base_url = "https://data.messari.io/api/v1"
        self.base_url_v2 = "https://data.messari.io/api/v2"
        self.redis = redis_client
        self.circuit_breaker = CircuitBreaker()
        self.session = session
        self._external_session = session is not None
        self._is_closed = False

    def _check_if_closed(self):
        if self._is_closed:
            raise RuntimeError("MessariFetcher has been closed and cannot be used")

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
        headers = {"x-messari-api-key": self.api_key}
        url = f"{self.base_url}/{endpoint}"
        await messari_rate_limiter.wait_if_needed(endpoint)
        
        async def _do_fetch():
            session = await self._get_session()
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 429:
                    raise APIRateLimitError("Messari rate limit exceeded")
                response.raise_for_status()
                return await response.json()
        
        try:
            return await self.circuit_breaker.call(_do_fetch)
        except Exception as e:
            if isinstance(e, (NetworkError, APIRateLimitError)):
                raise
            logger.error(f"Error in Messari _fetch for {url}: {e}", exc_info=True)
            raise DataError(f"Failed to fetch data from Messari API: {url}") from e

    async def get_asset_metrics(self, symbol: str) -> Optional[Dict[str, Any]]:
        self._check_if_closed()
        cache_key = f"cache:messari:asset_metrics:{symbol}"
        try:
            if self.redis:
                cached = await self.redis.get(cache_key)
                if cached:
                    return json.loads(cached)
        except Exception as e:
            logger.warning(f"Redis GET failed for {cache_key}: {e}")

        try:
            asset_id = symbol.split('/')[0].lower()
            data = await self._fetch(f"assets/{asset_id}/metrics")
            if data and 'data' in data:
                metrics = data['data']
                try:
                    if self.redis:
                        await self.redis.set(cache_key, json.dumps(metrics), ex=3600)
                except Exception as e:
                    logger.warning(f"Redis SET failed for {cache_key}: {e}")
                return metrics
            return None
        except Exception as e:
            logger.error(f"Error fetching asset metrics from Messari for {symbol}: {e}")
            return None

    async def get_on_chain_data(self, symbol: str) -> Optional[OnChainAnalysis]:
        self._check_if_closed()
        cache_key = f"cache:messari:onchain:{symbol}"
        try:
            if self.redis:
                cached = await self.redis.get(cache_key)
                if cached:
                    data = json.loads(cached)
                    return OnChainAnalysis(**data)
        except Exception as e:
            logger.warning(f"Redis GET failed for {cache_key}: {e}")
        
        try:
            metrics = await self.get_asset_metrics(symbol)
            if not metrics:
                return None

            analysis_data = {
                "mvrv": metrics.get("market_data", {}).get("mvrv_usd"),
                "sopr": metrics.get("on_chain_data", {}).get("sopr"),
                "active_addresses": metrics.get("on_chain_data", {}).get("active_addresses"),
                "realized_cap": metrics.get("marketcap", {}).get("realized_marketcap_usd"),
            }
            
            on_chain_analysis = OnChainAnalysis(**analysis_data)
            
            try:
                if self.redis:
                    await self.redis.set(cache_key, json.dumps(analysis_data), ex=3600)
            except Exception as e:
                logger.warning(f"Redis SET failed for {cache_key}: {e}")

            return on_chain_analysis
        except Exception as e:
            logger.error(f"Error processing on-chain data from Messari for {symbol}: {e}")
            return None

    async def get_asset_profile(self, symbol: str) -> Optional[Dict[str, Any]]:
        self._check_if_closed()
        cache_key = f"cache:messari:asset_profile:{symbol}"
        try:
            if self.redis:
                cached = await self.redis.get(cache_key)
                if cached:
                    return json.loads(cached)
        except Exception as e:
            logger.warning(f"Redis GET failed for {cache_key}: {e}")

        try:
            asset_id = symbol.split('/')[0].lower()
            data = await self._fetch(f"assets/{asset_id}/profile")
            if data and 'data' in data:
                profile = data['data']
                if self.redis:
                    await self.redis.set(cache_key, json.dumps(profile), ex=86400)
                return profile
            return None
        except Exception as e:
            logger.error(f"Error fetching asset profile from Messari for {symbol}: {e}")
            return None

    async def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        self._check_if_closed()
        cache_key = f"cache:messari:market_data:{symbol}"
        try:
            if self.redis:
                cached = await self.redis.get(cache_key)
                if cached:
                    return json.loads(cached)
        except Exception as e:
            logger.warning(f"Redis GET failed for {cache_key}: {e}")

        try:
            asset_id = symbol.split('/')[0].lower()
            data = await self._fetch(f"assets/{asset_id}/metrics/market-data")
            if data and 'data' in data:
                market = data['data']
                if self.redis:
                    await self.redis.set(cache_key, json.dumps(market), ex=600)
                return market
            return None
        except Exception as e:
            logger.error(f"Error fetching market data from Messari for {symbol}: {e}")
            return None

    async def get_news(self, symbol: str) -> List[Dict]:
        self._check_if_closed()
        try:
            asset_id = symbol.split('/')[0].lower()
            data = await self._fetch(f"news/{asset_id}")
            if data and 'data' in data:
                return data['data']
            return []
        except Exception as e:
            logger.error(f"Error fetching news from Messari for {symbol}: {e}")
            return []

    async def get_time_series_metrics(self, symbol: str, metric: str, start: Optional[str] = None, 
                                    end: Optional[str] = None) -> Optional[Dict]:
        self._check_if_closed()
        try:
            asset_id = symbol.split('/')[0].lower()
            params = {}
            if start:
                params['start'] = start
            if end:
                params['end'] = end
            data = await self._fetch(f"assets/{asset_id}/metrics/{metric}/time-series", params)
            return data.get('data') if data else None
        except Exception as e:
            logger.error(f"Error fetching time series metrics from Messari for {symbol}: {e}")
            return None

    async def get_asset_timeseries(self, symbol: str, metric_id: str, start: Optional[str] = None, end: Optional[str] = None, interval: str = "1d") -> Optional[Dict]:
        self._check_if_closed()
        cache_key = f"cache:messari:timeseries:{symbol}:{metric_id}:{interval}"
        try:
            if self.redis:
                cached = await self.redis.get(cache_key)
                if cached:
                    return json.loads(cached)
        except Exception as e:
            logger.warning(f"Redis GET failed for {cache_key}: {e}")
        
        try:
            asset_id = symbol.split('/')[0].lower()
            params = {"interval": interval}
            if start:
                params['start'] = start
            if end:
                params['end'] = end
            data = await self._fetch(f"assets/{asset_id}/metrics/{metric_id}/time-series", params)
            if data and self.redis:
                await self.redis.set(cache_key, json.dumps(data), ex=3600)
            return data
        except Exception as e:
            logger.error(f"Error fetching timeseries for {symbol}: {e}")
            return None

    async def get_all_assets(self) -> Optional[List[Dict]]:
        self._check_if_closed()
        cache_key = "cache:messari:all_assets"
        try:
            if self.redis:
                cached = await self.redis.get(cache_key)
                if cached:
                    return json.loads(cached)
        except Exception as e:
            logger.warning(f"Redis GET failed for {cache_key}: {e}")
        
        try:
            data = await self._fetch("assets")
            if data and 'data' in data and self.redis:
                await self.redis.set(cache_key, json.dumps(data['data']), ex=86400)
                return data['data']
            return None
        except Exception as e:
            logger.error(f"Error fetching all assets: {e}")
            return None

    async def get_market_metrics(self, symbol: str) -> Optional[Dict]:
        self._check_if_closed()
        cache_key = f"cache:messari:market_metrics:{symbol}"
        try:
            if self.redis:
                cached = await self.redis.get(cache_key)
                if cached:
                    return json.loads(cached)
        except Exception as e:
            logger.warning(f"Redis GET failed for {cache_key}: {e}")
        
        try:
            asset_id = symbol.split('/')[0].lower()
            data = await self._fetch(f"assets/{asset_id}/metrics/market-data")
            if data and self.redis:
                await self.redis.set(cache_key, json.dumps(data), ex=600)
            return data
        except Exception as e:
            logger.error(f"Error fetching market metrics for {symbol}: {e}")
            return None

    async def get_all_news(self, page: int = 1) -> Optional[List[Dict]]:
        self._check_if_closed()
        try:
            params = {"page": page}
            data = await self._fetch("news", params)
            if data and 'data' in data:
                return data['data']
            return None
        except Exception as e:
            logger.error(f"Error fetching all news: {e}")
            return None

    async def get_asset_research(self, symbol: str) -> Optional[Dict]:
        self._check_if_closed()
        cache_key = f"cache:messari:research:{symbol}"
        try:
            if self.redis:
                cached = await self.redis.get(cache_key)
                if cached:
                    return json.loads(cached)
        except Exception as e:
            logger.warning(f"Redis GET failed for {cache_key}: {e}")
        
        try:
            asset_id = symbol.split('/')[0].lower()
            data = await self._fetch(f"assets/{asset_id}/profile")
            if data and self.redis:
                await self.redis.set(cache_key, json.dumps(data), ex=86400)
            return data
        except Exception as e:
            logger.error(f"Error fetching research for {symbol}: {e}")
            return None

    async def get_supply_activity(self, symbol: str) -> Optional[Dict]:
        self._check_if_closed()
        try:
            asset_id = symbol.split('/')[0].lower()
            data = await self._fetch(f"assets/{asset_id}/metrics/supply")
            return data
        except Exception as e:
            logger.error(f"Error fetching supply activity for {symbol}: {e}")
            return None

    async def get_mining_stats(self, symbol: str) -> Optional[Dict]:
        self._check_if_closed()
        try:
            asset_id = symbol.split('/')[0].lower()
            data = await self._fetch(f"assets/{asset_id}/metrics/mining-stats")
            return data
        except Exception as e:
            logger.error(f"Error fetching mining stats for {symbol}: {e}")
            return None

    async def get_developer_activity(self, symbol: str) -> Optional[Dict]:
        self._check_if_closed()
        try:
            asset_id = symbol.split('/')[0].lower()
            data = await self._fetch(f"assets/{asset_id}/metrics/developer-activity")
            return data
        except Exception as e:
            logger.error(f"Error fetching developer activity for {symbol}: {e}")
            return None

    async def get_roi_data(self, symbol: str) -> Optional[Dict]:
        self._check_if_closed()
        try:
            asset_id = symbol.split('/')[0].lower()
            data = await self._fetch(f"assets/{asset_id}/metrics/roi-data")
            return data
        except Exception as e:
            logger.error(f"Error fetching ROI data for {symbol}: {e}")
            return None

    async def get_exchange_flows(self, symbol: str) -> Optional[Dict]:
        self._check_if_closed()
        try:
            asset_id = symbol.split('/')[0].lower()
            data = await self._fetch(f"assets/{asset_id}/metrics/exchange-flows")
            return data
        except Exception as e:
            logger.error(f"Error fetching exchange flows for {symbol}: {e}")
            return None

    async def get_comprehensive_metrics(self, symbol: str) -> Optional[Dict[str, Any]]:
        self._check_if_closed()
        cache_key = f"cache:messari:comprehensive:{symbol}"
        try:
            if self.redis:
                cached = await self.redis.get(cache_key)
                if cached:
                    return json.loads(cached)
        except Exception as e:
            logger.warning(f"Redis GET failed for {cache_key}: {e}")
        
        try:
            asset_id = symbol.split('/')[0].lower()
            
            tasks = {
                'metrics': self.get_asset_metrics(symbol),
                'profile': self.get_asset_profile(symbol),
                'market': self.get_market_data(symbol),
                'supply': self.get_supply_activity(symbol),
                'mining': self.get_mining_stats(symbol),
                'developer': self.get_developer_activity(symbol),
                'roi': self.get_roi_data(symbol),
                'flows': self.get_exchange_flows(symbol)
            }
            
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            comprehensive_data = {}
            
            for key, result in zip(tasks.keys(), results):
                if not isinstance(result, Exception) and result:
                    comprehensive_data[key] = result
            
            if self.redis and comprehensive_data:
                await self.redis.set(cache_key, json.dumps(comprehensive_data), ex=3600)
            
            return comprehensive_data
        except Exception as e:
            logger.error(f"Error fetching comprehensive metrics for {symbol}: {e}")
            return None

    async def get_circulating_supply(self, symbol: str) -> Optional[float]:
        self._check_if_closed()
        
        try:
            supply_data = await self.get_supply_activity(symbol)
            if supply_data and 'data' in supply_data:
                supply_metrics = supply_data['data']
                circulating = supply_metrics.get('supply_circulating')
                if circulating:
                    return float(circulating)
            
            metrics = await self.get_asset_metrics(symbol)
            if metrics and 'supply' in metrics:
                circulating = metrics['supply'].get('circulating')
                if circulating:
                    return float(circulating)
            
            return None
        except Exception as e:
            logger.error(f"Error fetching circulating supply for {symbol}: {e}")
            return None

    async def get_hash_rate_and_difficulty(self, symbol: str) -> Optional[Dict]:
        self._check_if_closed()
        
        try:
            mining_data = await self.get_mining_stats(symbol)
            if not mining_data or 'data' not in mining_data:
                return None
            
            mining_metrics = mining_data['data']
            return {
                'hash_rate': mining_metrics.get('hash_rate'),
                'difficulty': mining_metrics.get('difficulty'),
                'mining_revenue': mining_metrics.get('mining_revenue_native'),
                'mining_revenue_usd': mining_metrics.get('mining_revenue_usd')
            }
        except Exception as e:
            logger.error(f"Error fetching hash rate and difficulty for {symbol}: {e}")
            return None

    async def get_active_addresses_count(self, symbol: str) -> Optional[int]:
        self._check_if_closed()
        
        try:
            metrics = await self.get_asset_metrics(symbol)
            if metrics and 'on_chain_data' in metrics:
                active_addresses = metrics['on_chain_data'].get('active_addresses')
                if active_addresses:
                    return int(active_addresses)
            
            return None
        except Exception as e:
            logger.error(f"Error fetching active addresses for {symbol}: {e}")
            return None

    async def get_mvrv_ratio(self, symbol: str) -> Optional[float]:
        self._check_if_closed()
        
        try:
            metrics = await self.get_asset_metrics(symbol)
            if metrics and 'market_data' in metrics:
                mvrv = metrics['market_data'].get('mvrv_usd')
                if mvrv:
                    return float(mvrv)
            
            return None
        except Exception as e:
            logger.error(f"Error fetching MVRV ratio for {symbol}: {e}")
            return None

    async def get_sopr_metric(self, symbol: str) -> Optional[float]:
        self._check_if_closed()
        
        try:
            metrics = await self.get_asset_metrics(symbol)
            if metrics and 'on_chain_data' in metrics:
                sopr = metrics['on_chain_data'].get('sopr')
                if sopr:
                    return float(sopr)
            
            return None
        except Exception as e:
            logger.error(f"Error fetching SOPR for {symbol}: {e}")
            return None

    async def get_roi_metrics(self, symbol: str) -> Optional[Dict]:
        self._check_if_closed()
        
        try:
            roi_data = await self.get_roi_data(symbol)
            if not roi_data or 'data' not in roi_data:
                return None
            
            roi_metrics = roi_data['data']
            return {
                'roi_1y': roi_metrics.get('percent_change_last_1_year'),
                'roi_30d': roi_metrics.get('percent_change_last_30_days'),
                'roi_90d': roi_metrics.get('percent_change_last_90_days'),
                'roi_all_time': roi_metrics.get('percent_change_all_time')
            }
        except Exception as e:
            logger.error(f"Error fetching ROI metrics for {symbol}: {e}")
            return None

    async def get_developer_metrics(self, symbol: str) -> Optional[Dict]:
        self._check_if_closed()
        
        try:
            dev_data = await self.get_developer_activity(symbol)
            if not dev_data or 'data' not in dev_data:
                return None
            
            dev_metrics = dev_data['data']
            return {
                'commit_count_4_weeks': dev_metrics.get('commit_count_4_weeks'),
                'stars': dev_metrics.get('stars'),
                'contributors': dev_metrics.get('contributors'),
                'pull_requests_merged': dev_metrics.get('pull_requests_merged'),
                'pull_requests_opened': dev_metrics.get('pull_requests_opened')
            }
        except Exception as e:
            logger.error(f"Error fetching developer metrics for {symbol}: {e}")
            return None
