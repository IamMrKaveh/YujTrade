import asyncio
import json
from typing import Dict, Optional

import aiohttp
import redis

from config.logger import logger
from common.cache import CacheKeyBuilder, CacheCategory
from config.settings import ConfigManager
from data.sources.alphavantage_fetcher import AlphaVantageFetcher
from data.sources.coingecko_fetcher import CoinGeckoFetcher
from data.sources.defillama_fetcher import DeFiLlamaFetcher
from data.sources.yfinance_fetcher import YFinanceFetcher
from data.sources.base_fetcher import BaseFetcher


class MarketIndicesFetcher(BaseFetcher):
    def __init__(
        self,
        coingecko_fetcher: Optional[CoinGeckoFetcher] = None,
        defillama_fetcher: Optional[DeFiLlamaFetcher] = None,
        yfinance_fetcher: Optional[YFinanceFetcher] = None,
        alphavantage_fetcher: Optional[AlphaVantageFetcher] = None,
        redis_client: Optional[redis.Redis] = None,
        session: Optional[aiohttp.ClientSession] = None,
        config_manager: Optional[ConfigManager] = None,
    ):
        super().__init__(
            redis_client=redis_client, session=session, config_manager=config_manager
        )
        self.source_name = "market_indices_aggregator"

        common_args = {
            "redis_client": redis_client,
            "session": session,
            "config_manager": config_manager,
        }

        self.coingecko = coingecko_fetcher or CoinGeckoFetcher(**common_args)
        self.defillama = defillama_fetcher or DeFiLlamaFetcher(**common_args)
        self.yfinance = yfinance_fetcher or YFinanceFetcher(**common_args)
        self.alphavantage = alphavantage_fetcher

    def _check_if_closed(self):
        if self._is_closed:
            raise RuntimeError(
                "MarketIndicesFetcher has been closed and cannot be used"
            )

    async def close(self):
        if self._is_closed:
            return
        self._is_closed = True
        # Child fetchers are managed by MarketDataProvider, so no need to close them here.
        # Session is also managed externally.
        logger.debug("MarketIndicesFetcher instance closed.")

    async def get_crypto_indices(self) -> Dict[str, Optional[float]]:
        self._check_if_closed()
        cache_key = CacheKeyBuilder.build(
            CacheCategory.INDICES, self.source_name, ["crypto"]
        )
        cache_ttl = self.config_manager.get_cache_ttl("indices")

        if self.redis:
            try:
                cached_data = await self.redis.get(cache_key)
                if cached_data:
                    return json.loads(cached_data)
            except Exception as e:
                logger.warning(f"Redis get failed for {cache_key}: {e}")

        # This method in coingecko_fetcher needs to be added or adapted
        async def get_global_market_data():
            try:
                url = f"{self.coingecko.base_url}/global"
                return await self.coingecko._fetch_json(url, endpoint_name="global")
            except Exception as e:
                logger.error(f"Failed to fetch global market data from coingecko: {e}")
                return None

        tasks = {
            "global": get_global_market_data(),
            "defi_llama_tvl_total": self.defillama.get_total_tvl_for_chains(),
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        global_data, defi_llama_tvl_data = results

        indices = {}
        if (
            not isinstance(global_data, Exception)
            and global_data
            and "data" in global_data
        ):
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
            others_dominance = 100 - sum(
                d
                for d in [
                    indices.get("BTC.D"),
                    indices.get("ETH.D"),
                    indices.get("USDT.D"),
                ]
                if d is not None
            )
            indices["OTHERS.D"] = others_dominance

        if (
            not isinstance(defi_llama_tvl_data, Exception)
            and isinstance(defi_llama_tvl_data, list)
            and defi_llama_tvl_data
        ):
            total_tvl = sum(chain.get("tvl", 0) for chain in defi_llama_tvl_data)
            indices["DEFI_TVL"] = total_tvl

        if self.redis and indices:
            try:
                await self.redis.set(cache_key, json.dumps(indices), ex=cache_ttl)
            except Exception as e:
                logger.warning(f"Redis set failed for {cache_key}: {e}")
        return indices

    async def get_all_indices(self) -> Dict[str, Optional[float]]:
        self._check_if_closed()
        tasks = {
            "crypto": self.get_crypto_indices(),
            "traditional": self.yfinance.get_traditional_indices(),
        }

        if self.alphavantage:
            tasks["macro"] = self.alphavantage.get_macro_economic_data()

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        all_indices = {}
        for key, result in zip(tasks.keys(), results):
            if not isinstance(result, Exception) and result:
                all_indices.update(result)

        return all_indices
