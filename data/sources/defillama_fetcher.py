import json
from typing import Dict, List, Optional

from config.logger import logger
from module.utils import RateLimiter
from .base_fetcher import BaseFetcher


defillama_rate_limiter = RateLimiter(max_requests=30, time_window=60)


class DeFiLlamaFetcher(BaseFetcher):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url = "https://api.llama.fi"

    async def get_total_tvl_for_chains(self):
        url = f"{self.base_url}/tvl/chains"
        return await self._fetch_json(url, limiter=defillama_rate_limiter)

    async def get_protocol_chart(self, protocol: str) -> Optional[Dict]:
        cache_key = f"cache:defillama:protocol_chart:{protocol}"
        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)

        url = f"{self.base_url}/protocol/{protocol}"
        data = await self._fetch_json(url, limiter=defillama_rate_limiter, endpoint_name=f"protocol_{protocol}")
        
        if data and self.redis:
            await self.redis.set(cache_key, json.dumps(data), ex=3600)
        return data

    async def get_defi_protocols(self) -> Optional[List[Dict]]:
        cache_key = "cache:defillama:protocols"
        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)

        url = f"{self.base_url}/protocols"
        data = await self._fetch_json(url, limiter=defillama_rate_limiter, endpoint_name="protocols")
        
        if data and self.redis:
            await self.redis.set(cache_key, json.dumps(data), ex=3600)
        return data

    async def get_chain_tvl(self, chain: str) -> Optional[Dict]:
        cache_key = f"cache:defillama:chain_tvl:{chain}"
        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)

        url = f"{self.base_url}/v2/historicalChainTvl/{chain}"
        data = await self._fetch_json(url, limiter=defillama_rate_limiter, endpoint_name=f"chain_tvl_{chain}")
        
        if data and self.redis:
            await self.redis.set(cache_key, json.dumps(data), ex=3600)
        return data

    async def get_stablecoins(self) -> Optional[Dict]:
        cache_key = "cache:defillama:stablecoins"
        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)

        url = f"{self.base_url}/stablecoins?includePrices=true"
        data = await self._fetch_json(url, limiter=defillama_rate_limiter, endpoint_name="stablecoins")
        
        if data and self.redis:
            await self.redis.set(cache_key, json.dumps(data), ex=3600)
        return data

    async def get_yields(self) -> Optional[List[Dict]]:
        cache_key = "cache:defillama:yields"
        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)

        url = f"{self.base_url}/pools"
        data = await self._fetch_json(url, limiter=defillama_rate_limiter, endpoint_name="yields")
        
        if data and 'data' in data and self.redis:
            await self.redis.set(cache_key, json.dumps(data['data']), ex=3600)
            return data['data']
        return data

    async def get_bridges_volume(self) -> Optional[Dict]:
        cache_key = "cache:defillama:bridges"
        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)

        url = f"{self.base_url}/bridges"
        data = await self._fetch_json(url, limiter=defillama_rate_limiter, endpoint_name="bridges")
        
        if data and self.redis:
            await self.redis.set(cache_key, json.dumps(data), ex=3600)
        return data

    async def get_coin_prices(self, coins: List[str]) -> Optional[Dict]:
        coins_string = ",".join(coins)
        cache_key = f"cache:defillama:prices:{coins_string}"
        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)

        url = f"{self.base_url}/prices/current/{coins_string}"
        data = await self._fetch_json(url, limiter=defillama_rate_limiter, endpoint_name="prices")
        
        if data and self.redis:
            await self.redis.set(cache_key, json.dumps(data), ex=300)
        return data

    async def get_protocol_tvl(self, protocol_slug: str) -> Optional[float]:
        self._check_if_closed()
        
        try:
            protocol_data = await self.get_protocol_chart(protocol_slug)
            if protocol_data and 'tvl' in protocol_data:
                tvl_list = protocol_data['tvl']
                if tvl_list and isinstance(tvl_list, list):
                    latest_tvl = tvl_list[-1]
                    if isinstance(latest_tvl, dict) and 'totalLiquidityUSD' in latest_tvl:
                        return float(latest_tvl['totalLiquidityUSD'])
            
            return None
        except Exception as e:
            logger.error(f"Error fetching protocol TVL for {protocol_slug}: {e}")
            return None

    async def get_chain_tvl_value(self, chain: str) -> Optional[float]:
        self._check_if_closed()
        
        try:
            chain_data = await self.get_chain_tvl(chain)
            if chain_data and isinstance(chain_data, list) and chain_data:
                latest_entry = chain_data[-1]
                if isinstance(latest_entry, dict) and 'tvl' in latest_entry:
                    return float(latest_entry['tvl'])
            
            return None
        except Exception as e:
            logger.error(f"Error fetching chain TVL for {chain}: {e}")
            return None

    async def get_stablecoin_supply(self) -> Optional[Dict]:
        self._check_if_closed()
        cache_key = "cache:defillama:stablecoin_supply"
        
        try:
            if self.redis:
                cached = await self.redis.get(cache_key)
                if cached:
                    return json.loads(cached)
        except Exception as e:
            logger.warning(f"Redis GET failed for {cache_key}: {e}")
        
        try:
            stablecoin_data = await self.get_stablecoins()
            if not stablecoin_data or 'peggedAssets' not in stablecoin_data:
                return None
            
            supply_data = {}
            for stablecoin in stablecoin_data['peggedAssets']:
                name = stablecoin.get('name')
                circulating = stablecoin.get('circulating', {}).get('peggedUSD')
                if name and circulating:
                    supply_data[name] = float(circulating)
            
            if self.redis and supply_data:
                await self.redis.set(cache_key, json.dumps(supply_data), ex=3600)
            
            return supply_data
        except Exception as e:
            logger.error(f"Error fetching stablecoin supply: {e}")
            return None

    async def get_yield_opportunities(self, chain: Optional[str] = None) -> Optional[List[Dict]]:
        self._check_if_closed()
        
        try:
            all_yields = await self.get_yields()
            if not all_yields:
                return None
            
            if chain:
                filtered_yields = [y for y in all_yields if y.get('chain', '').lower() == chain.lower()]
                return filtered_yields
            
            return all_yields
        except Exception as e:
            logger.error(f"Error fetching yield opportunities: {e}")
            return None
