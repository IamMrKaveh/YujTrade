import asyncio
import json
from typing import Dict, Optional

import aiohttp
import redis

from module.logger_config import logger
from .alphavantage import AlphaVantageFetcher
from .coingecko import CoinGeckoFetcher
from .defillama import DeFiLlamaFetcher
from .yfinance import YFinanceFetcher


class MarketIndicesFetcher:
    def __init__(
        self,
        coingecko_fetcher: Optional[CoinGeckoFetcher] = None,
        defillama_fetcher: Optional[DeFiLlamaFetcher] = None,
        yfinance_fetcher: Optional[YFinanceFetcher] = None,
        alphavantage_fetcher: Optional[AlphaVantageFetcher] = None,
        redis_client: Optional[redis.Redis] = None,
        session: Optional[aiohttp.ClientSession] = None
    ):
        self._is_closed = False
        self.redis = redis_client
        self.session = session or aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        self.cache_ttl = 600

        self.coingecko = coingecko_fetcher or CoinGeckoFetcher(redis_client=self.redis, session=self.session)
        self.defillama = defillama_fetcher or DeFiLlamaFetcher(redis_client=self.redis, session=self.session)
        self.yfinance = yfinance_fetcher or YFinanceFetcher(redis_client=self.redis, session=self.session)
        
        self.alphavantage = alphavantage_fetcher

    def _check_if_closed(self):
        if self._is_closed:
            raise RuntimeError("MarketIndicesFetcher has been closed and cannot be used")

    async def close(self):
        if self._is_closed:
            return
        self._is_closed = True
        
        tasks = [
            self.coingecko.close(),
            self.defillama.close(),
            self.yfinance.close()
        ]
        
        if self.alphavantage:
            tasks.append(self.alphavantage.close())
            
        if self.session and not self.session.closed:
            tasks.append(self.session.close())
            
        await asyncio.gather(*tasks, return_exceptions=True)

    async def get_crypto_indices(self) -> Dict[str, Optional[float]]:
        self._check_if_closed()
        cache_key = "cache:crypto_indices"
        if self.redis:
            try:
                cached_data = await self.redis.get(cache_key)
                if cached_data:
                    return json.loads(cached_data)
            except Exception as e:
                logger.warning(f"Redis get failed for {cache_key}: {e}")

        tasks = {
            "global": self.coingecko.get_global_market_data(),
            "defi_llama_tvl_total": self.defillama.get_total_tvl_for_chains()
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        global_data, defi_llama_tvl_data = results

        indices = {}
        if not isinstance(global_data, Exception) and global_data and "data" in global_data:
            market_data = global_data["data"]
            dominance = market_data.get("market_cap_percentage", {})
            indices["BTC.D"] = dominance.get("btc")
            indices["ETH.D"] = dominance.get("eth")
            indices["USDT.D"] = dominance.get("usdt")
            total_mcap = market_data.get("total_market_cap", {}).get("usd")
            indices["TOTAL"] = total_mcap
            if total_mcap and indices.get("BTC.D"):
                btc_mcap = total_mcap * (indices["BTC.D"] / 100)
                indices["TOTAL2"] = total_mcap - btc_mcap
                if indices.get("ETH.D"):
                    eth_mcap = total_mcap * (indices["ETH.D"] / 100)
                    indices["TOTAL3"] = indices["TOTAL2"] - eth_mcap
            others_dominance = 100 - sum(d for d in [indices.get("BTC.D"), indices.get("ETH.D"), indices.get("USDT.D")] if d is not None)
            indices["OTHERS.D"] = others_dominance

        if not isinstance(defi_llama_tvl_data, Exception) and isinstance(defi_llama_tvl_data, list) and defi_llama_tvl_data:
            total_tvl = sum(chain.get("tvl", 0) for chain in defi_llama_tvl_data)
            indices["DEFI_TVL"] = total_tvl

        if self.redis and indices:
            try:
                await self.redis.set(cache_key, json.dumps(indices), ex=self.cache_ttl)
            except Exception as e:
                logger.warning(f"Redis set failed for {cache_key}: {e}")
        return indices

    async def get_all_indices(self) -> Dict[str, Optional[float]]:
        self._check_if_closed()
        tasks = {
            "crypto": self.get_crypto_indices(),
            "traditional": self.yfinance.get_traditional_indices()
        }
        
        if self.alphavantage:
            tasks["macro"] = self.alphavantage.get_macro_economic_data()
        
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        all_indices = {}
        for key, result in zip(tasks.keys(), results):
            if not isinstance(result, Exception):
                all_indices.update(result)
        
        return all_indices
