import asyncio
import json
from typing import Any, Dict

from .base import BaseFetcher
from module.utils import RateLimiter


alpha_vantage_rate_limiter = RateLimiter(max_requests=5, time_window=60)


class AlphaVantageFetcher(BaseFetcher):
    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"

    async def get_macro_economic_data(self) -> Dict[str, Any]:
        cache_key = "cache:macro_economic_data"
        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)

        if not self.api_key:
            return {}

        tasks = {
            "cpi": self._fetch_json(self.base_url, params={"function": "CPI", "interval": "monthly", "apikey": self.api_key}, limiter=alpha_vantage_rate_limiter, endpoint_name="cpi"),
            "fed_rate": self._fetch_json(self.base_url, params={"function": "FEDERAL_FUNDS_RATE", "interval": "monthly", "apikey": self.api_key}, limiter=alpha_vantage_rate_limiter, endpoint_name="fed_rate"),
            "gdp": self._fetch_json(self.base_url, params={"function": "REAL_GDP", "interval": "quarterly", "apikey": self.api_key}, limiter=alpha_vantage_rate_limiter, endpoint_name="gdp"),
            "unemployment": self._fetch_json(self.base_url, params={"function": "UNEMPLOYMENT", "apikey": self.api_key}, limiter=alpha_vantage_rate_limiter, endpoint_name="unemployment"),
            "treasury_yield": self._fetch_json(self.base_url, params={"function": "TREASURY_YIELD", "interval": "monthly", "maturity": "10year", "apikey": self.api_key}, limiter=alpha_vantage_rate_limiter, endpoint_name="treasury_yield"),
            "inflation": self._fetch_json(self.base_url, params={"function": "INFLATION", "apikey": self.api_key}, limiter=alpha_vantage_rate_limiter, endpoint_name="inflation"),
            "consumer_sentiment": self._fetch_json(self.base_url, params={"function": "CONSUMER_SENTIMENT", "apikey": self.api_key}, limiter=alpha_vantage_rate_limiter, endpoint_name="consumer_sentiment"),
            "retail_sales": self._fetch_json(self.base_url, params={"function": "RETAIL_SALES", "apikey": self.api_key}, limiter=alpha_vantage_rate_limiter, endpoint_name="retail_sales"),
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        macro_data = {}
        cpi_data, fed_rate_data, gdp_data, unemployment_data, yield_data, inflation_data, consumer_sentiment_data, retail_sales_data = results
        
        if not isinstance(cpi_data, Exception) and cpi_data and "data" in cpi_data and cpi_data["data"]:
            macro_data["CPI"] = float(cpi_data["data"][0]["value"])
        if not isinstance(fed_rate_data, Exception) and fed_rate_data and "data" in fed_rate_data and fed_rate_data["data"]:
            macro_data["FED_RATE"] = float(fed_rate_data["data"][0]["value"])
        if not isinstance(gdp_data, Exception) and gdp_data and "data" in gdp_data and gdp_data["data"]:
            macro_data["GDP"] = float(gdp_data["data"][0]["value"])
        if not isinstance(unemployment_data, Exception) and unemployment_data and "data" in unemployment_data and unemployment_data["data"]:
            macro_data["UNEMPLOYMENT"] = float(unemployment_data["data"][0]["value"])
        if not isinstance(yield_data, Exception) and yield_data and "data" in yield_data and yield_data["data"]:
            macro_data["TREASURY_YIELD_10Y"] = float(yield_data["data"][0]["value"])
        if not isinstance(inflation_data, Exception) and inflation_data and "data" in inflation_data and inflation_data["data"]:
            macro_data["INFLATION"] = float(inflation_data["data"][0]["value"])
        if not isinstance(consumer_sentiment_data, Exception) and consumer_sentiment_data and "data" in consumer_sentiment_data and consumer_sentiment_data["data"]:
            macro_data["CONSUMER_SENTIMENT"] = float(consumer_sentiment_data["data"][0]["value"])
        if not isinstance(retail_sales_data, Exception) and retail_sales_data and "data" in retail_sales_data and retail_sales_data["data"]:
            macro_data["RETAIL_SALES"] = float(retail_sales_data["data"][0]["value"])

        if self.redis and macro_data:
            await self.redis.set(cache_key, json.dumps(macro_data), ex=86400)
        
        return macro_data

    async def get_comprehensive_macro_data(self) -> Dict[str, Any]:
        self._check_if_closed()
        cache_key = "cache:macro_comprehensive"
        
        try:
            if self.redis:
                cached = await self.redis.get(cache_key)
                if cached:
                    return json.loads(cached)
        except Exception as e:
            logger.warning(f"Redis GET failed for {cache_key}: {e}")

        if not self.api_key:
            return {}

        tasks = {
            "cpi": self._fetch_json(self.base_url, params={"function": "CPI", "interval": "monthly", "apikey": self.api_key}, limiter=alpha_vantage_rate_limiter, endpoint_name="cpi"),
            "fed_rate": self._fetch_json(self.base_url, params={"function": "FEDERAL_FUNDS_RATE", "interval": "monthly", "apikey": self.api_key}, limiter=alpha_vantage_rate_limiter, endpoint_name="fed_rate"),
            "gdp": self._fetch_json(self.base_url, params={"function": "REAL_GDP", "interval": "quarterly", "apikey": self.api_key}, limiter=alpha_vantage_rate_limiter, endpoint_name="gdp"),
            "unemployment": self._fetch_json(self.base_url, params={"function": "UNEMPLOYMENT", "apikey": self.api_key}, limiter=alpha_vantage_rate_limiter, endpoint_name="unemployment"),
            "treasury_yield": self._fetch_json(self.base_url, params={"function": "TREASURY_YIELD", "interval": "monthly", "maturity": "10year", "apikey": self.api_key}, limiter=alpha_vantage_rate_limiter, endpoint_name="treasury_yield"),
            "inflation": self._fetch_json(self.base_url, params={"function": "INFLATION", "apikey": self.api_key}, limiter=alpha_vantage_rate_limiter, endpoint_name="inflation"),
            "consumer_sentiment": self._fetch_json(self.base_url, params={"function": "CONSUMER_SENTIMENT", "apikey": self.api_key}, limiter=alpha_vantage_rate_limiter, endpoint_name="consumer_sentiment"),
            "retail_sales": self._fetch_json(self.base_url, params={"function": "RETAIL_SALES", "apikey": self.api_key}, limiter=alpha_vantage_rate_limiter, endpoint_name="retail_sales")
        }
        
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        macro_data = {}
        keys_list = list(tasks.keys())
        
        for i, result in enumerate(results):
            key = keys_list[i]
            if not isinstance(result, Exception) and result and "data" in result and result["data"]:
                try:
                    macro_data[key.upper()] = float(result["data"][0]["value"])
                except (KeyError, ValueError, IndexError) as e:
                    logger.warning(f"Could not parse {key}: {e}")

        if self.redis and macro_data:
            await self.redis.set(cache_key, json.dumps(macro_data), ex=86400)
        
        return macro_data
