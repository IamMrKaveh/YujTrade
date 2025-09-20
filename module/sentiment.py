import asyncio
import json
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import aiohttp
import aiosqlite
import ccxt.async_support as ccxt
import numpy as np
import pandas as pd
import redis.asyncio as redis
import yfinance as yf

from module.config import Config
from module.core import DerivativesAnalysis, FundamentalAnalysis, OrderBookAnalysis
from module.logger_config import logger
from module.utils import RateLimiter, async_retry

rate_limiter = RateLimiter(max_requests=10, time_window=60)
coingecko_rate_limiter = RateLimiter(max_requests=30, time_window=60)


class MarketIndicesFetcher:
    def __init__(
        self,
        alpha_vantage_key: Optional[str] = None,
        redis_client: Optional[redis.Redis] = None,
    ):
        self.coingecko_url = "https://api.coingecko.com/api/v3"
        self.defillama_url = "https://api.llama.fi"
        self.alpha_vantage_url = "https://www.alphavantage.co/query"
        self.alpha_vantage_key = alpha_vantage_key
        self.redis = redis_client
        self.cache_ttl = 600

    async def _fetch_json(self, url, params=None):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=15) as response:
                    if response.status == 200:
                        return await response.json()
                    logger.warning(f"API request to {url} failed with status: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching from {url}: {e}")
            return None

    async def get_crypto_indices(self):
        cache_key = "cache:crypto_indices"
        if self.redis:
            try:
                cached_data = await self.redis.get(cache_key)
                if cached_data:
                    return json.loads(cached_data)
            except Exception as e:
                logger.warning(f"Redis get failed for {cache_key}: {e}")


        tasks = {
            "global": self._fetch_json(f"{self.coingecko_url}/global"),
            "defi": self._fetch_json(f"{self.defillama_url}/v2/historicalChainTvl"),
        }
        results = await asyncio.gather(*tasks.values())
        global_data, defi_data = results

        indices = {}
        if global_data and "data" in global_data:
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

        if defi_data and isinstance(defi_data, list) and defi_data:
            indices["DEFI"] = defi_data[-1].get("tvl")

        if self.redis and indices:
            try:
                await self.redis.set(cache_key, json.dumps(indices), ex=self.cache_ttl)
            except Exception as e:
                logger.warning(f"Redis set failed for {cache_key}: {e}")
        return indices

    async def get_traditional_indices_yf(self):
        cache_key = "cache:trad_indices_yf"
        if self.redis:
            try:
                cached = await self.redis.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception as e:
                logger.warning(f"Redis get failed for {cache_key}: {e}")


        tickers = {"DXY": "DX-Y.NYB", "SPX": "^GSPC", "NDX": "^IXIC", "VIX": "^VIX"}
        data = yf.download(tickers=list(tickers.values()), period="5d", interval="1d", progress=False)
        if data.empty:
            return {}

        indices = {
            key: data["Adj Close"][ticker].iloc[-1] for key, ticker in tickers.items() if ticker in data["Adj Close"] and not pd.isna(data["Adj Close"][ticker].iloc[-1])
        }

        if self.redis and indices:
            try:
                await self.redis.set(cache_key, json.dumps(indices), ex=self.cache_ttl)
            except Exception as e:
                logger.warning(f"Redis set failed for {cache_key}: {e}")
        return indices

    async def get_macro_economic_data(self) -> Dict[str, Any]:
        cache_key = "cache:macro_economic_data"
        if self.redis:
            try:
                cached = await self.redis.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception as e:
                logger.warning(f"Redis get failed for {cache_key}: {e}")

        if not self.alpha_vantage_key:
            return {}

        tasks = {
            "cpi": self._fetch_json(
                self.alpha_vantage_url,
                params={"function": "CPI", "interval": "monthly", "apikey": self.alpha_vantage_key}
            ),
            "fed_rate": self._fetch_json(
                self.alpha_vantage_url,
                params={"function": "FEDERAL_FUNDS_RATE", "interval": "monthly", "apikey": self.alpha_vantage_key}
            )
        }
        results = await asyncio.gather(*tasks.values())
        cpi_data, fed_rate_data = results
        
        macro_data = {}
        if cpi_data and "data" in cpi_data and cpi_data["data"]:
            macro_data["CPI"] = float(cpi_data["data"][0]["value"])
        if fed_rate_data and "data" in fed_rate_data and fed_rate_data["data"]:
            macro_data["FED_RATE"] = float(fed_rate_data["data"][0]["value"])

        if self.redis and macro_data:
            try:
                await self.redis.set(cache_key, json.dumps(macro_data), ex=86400)
            except Exception as e:
                logger.warning(f"Redis set failed for {cache_key}: {e}")
        
        return macro_data

    async def get_all_indices(self) -> Dict[str, Optional[float]]:
        crypto_task = self.get_crypto_indices()
        traditional_task = self.get_traditional_indices_yf()
        macro_task = self.get_macro_economic_data()

        crypto_indices, traditional_indices, macro_data = await asyncio.gather(crypto_task, traditional_task, macro_task)

        all_indices = {**(crypto_indices or {}), **(traditional_indices or {}), **(macro_data or {})}
        return all_indices


class NewsFetcher:
    def __init__(self, cryptopanic_key: str, redis_client: Optional[redis.Redis] = None):
        self.cryptopanic_key = cryptopanic_key
        self.redis = redis_client

    async def fetch_fear_greed(self, max_retries=3, retry_delay=5):
        cache_key = "cache:fear_greed"
        if self.redis:
            try:
                cached = await self.redis.get(cache_key)
                if cached:
                    return int(cached)
            except Exception as e:
                logger.warning(f"Redis get failed for {cache_key}: {e}")

        async with aiohttp.ClientSession() as session:
            for attempt in range(max_retries):
                try:
                    async with session.get("https://api.alternative.me/fng/", timeout=10) as response:
                        if response.status == 200:
                            j = await response.json()
                            data = j.get("data", [])
                            if data:
                                value = data[0].get("value")
                                if value is not None:
                                    value_int = int(value)
                                    if self.redis:
                                        try:
                                            await self.redis.set(cache_key, value_int, ex=3600)
                                        except Exception as e:
                                            logger.warning(f"Redis set failed for {cache_key}: {e}")
                                    return value_int

                    logger.warning(f"Attempt {attempt+1}: Invalid response from Fear & Greed API status {response.status}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                except aiohttp.ClientError as e:
                    logger.warning(f"Attempt {attempt+1}: Error connecting to Fear & Greed API: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                    continue

        logger.error("Failed to fetch Fear & Greed index after multiple attempts")
        return None

    async def fetch_news(self, currencies: List[str] = ["BTC", "ETH"]):
        key = "cache:news:" + ",".join(sorted(currencies))
        if self.redis:
            try:
                cached = await self.redis.get(key)
                if cached:
                    return json.loads(cached)
            except Exception as e:
                logger.warning(f"Redis get failed for {key}: {e}")

        try:
            url = "https://cryptopanic.com/api/v2/posts/"
            params = {"auth_token": self.cryptopanic_key, "currencies": ",".join(currencies), "public": "true"}

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = data.get("results", [])
                        if self.redis:
                            try:
                                await self.redis.set(key, json.dumps(results), ex=600)
                            except Exception as e:
                                logger.warning(f"Redis set failed for {key}: {e}")
                        return results
                    else:
                        logger.warning(f"CryptoPanic API returned status {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []

    def score_news(self, news_items: List[Dict[str, Any]]):
        score = 0
        for it in news_items:
            title = (it.get("title") or "").lower()
            if any(k in title for k in ["bull", "rally", "surge", "gain", "pump"]):
                score += 1
            if any(k in title for k in ["crash", "dump", "fall", "drop", "hack"]):
                score -= 1
        return score


class CoinGeckoFetcher:
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.redis = redis_client

    async def _get_coin_list(self):
        cache_key = "cache:coingecko:coin_list"
        if self.redis:
            try:
                cached = await self.redis.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception as e:
                logger.warning(f"Redis get failed for {cache_key}: {e}")

        async with aiohttp.ClientSession() as session:
            try:
                await coingecko_rate_limiter.wait_if_needed("coins_list")
                async with session.get(f"{self.base_url}/coins/list") as response:
                    if response.status == 200:
                        data = await response.json()
                        coin_list = {item["symbol"].upper(): item["id"] for item in data}
                        if self.redis:
                            try:
                                await self.redis.set(cache_key, json.dumps(coin_list), ex=86400)
                            except Exception as e:
                                logger.warning(f"Redis set failed for {cache_key}: {e}")
                        return coin_list
            except Exception as e:
                logger.error(f"Failed to fetch CoinGecko coin list: {e}")
        return {}

    async def get_fundamental_data(self, symbol: str) -> Optional[FundamentalAnalysis]:
        coin_list = await self._get_coin_list()
        coin_id = coin_list.get(symbol.upper())
        if not coin_id:
            logger.warning(f"Could not find CoinGecko ID for symbol: {symbol}")
            return None

        cache_key = f"cache:coingecko:fundamentals:{coin_id}"
        if self.redis:
            try:
                cached = await self.redis.get(cache_key)
                if cached:
                    data = json.loads(cached)
                    return FundamentalAnalysis(
                        market_cap=data.get("market_cap", 0),
                        circulating_supply=data.get("circulating_supply", 0),
                        developer_score=data.get("developer_score", 0),
                    )
            except Exception as e:
                logger.warning(f"Redis get failed for {cache_key}: {e}")

        async with aiohttp.ClientSession() as session:
            try:
                await coingecko_rate_limiter.wait_if_needed(f"coins_{coin_id}")
                async with session.get(f"{self.base_url}/coins/{coin_id}") as response:
                    if response.status == 200:
                        data = await response.json()
                        market_data = data.get("market_data", {})
                        fund_data = {
                            "market_cap": market_data.get("market_cap", {}).get("usd", 0),
                            "circulating_supply": market_data.get("circulating_supply", 0),
                            "developer_score": data.get("developer_score", 0),
                        }
                        if self.redis:
                            try:
                                await self.redis.set(cache_key, json.dumps(fund_data), ex=3600)
                            except Exception as e:
                                logger.warning(f"Redis set failed for {cache_key}: {e}")
                        return FundamentalAnalysis(**fund_data)
            except Exception as e:
                logger.error(f"Failed to fetch fundamental data for {symbol}: {e}")
        return None


class OnChainFetcher:
    def __init__(self, glassnode_api_key: str = "", coinmetrics_api_key: str = "", dune_api_key: str = ""):
        self.glassnode_api_key = glassnode_api_key
        self.coinmetrics_api_key = coinmetrics_api_key
        self.dune_api_key = dune_api_key

        self.glassnode_base_url = "https://api.glassnode.com/v1/metrics"
        self.coinmetrics_base_url = "https://api.coinmetrics.io/v4"
        self.dune_base_url = "https://api.dune.com/api/v1"

        self.last_glassnode_call = 0
        self.glassnode_rate_limit = 1

    async def _rate_limit_glassnode(self):
        current_time = time.time()
        time_since_last_call = current_time - self.last_glassnode_call
        if time_since_last_call < self.glassnode_rate_limit:
            await asyncio.sleep(self.glassnode_rate_limit - time_since_last_call)
        self.last_glassnode_call = time.time()

    async def _glassnode_request(self, endpoint: str, params: Dict = {}) -> Optional[Dict]:
        if not self.glassnode_api_key:
            return None

        await self._rate_limit_glassnode()

        url = f"{self.glassnode_base_url}/{endpoint}"

        default_params = {"a": "BTC", "api_key": self.glassnode_api_key}
        if params:
            default_params.update(params)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=default_params, timeout=30) as response:
                    response.raise_for_status()
                    return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"Glassnode API error: {e}")
            return None

    async def active_addresses(self) -> Optional[int]:
        if self.glassnode_api_key:
            try:
                data = await self._glassnode_request("addresses/active_count", {"i": "24h"})
                if data and len(data) > 0:
                    return int(data[-1]["v"])
            except Exception as e:
                logger.warning(f"Failed to get active addresses from Glassnode: {e}")
        return None

    async def transaction_volume(self) -> Optional[float]:
        if self.glassnode_api_key:
            try:
                data = await self._glassnode_request("transactions/transfers_volume_sum", {"i": "24h"})
                if data and len(data) > 0:
                    return float(data[-1]["v"])
            except Exception as e:
                logger.warning(f"Failed to get transaction volume from Glassnode: {e}")
        return None
    
    async def get_hash_rate(self) -> Optional[float]:
        if not self.glassnode_api_key:
            return None
        try:
            data = await self._glassnode_request("mining/hash_rate_mean", {"i": "24h"})
            if data and len(data) > 0:
                return float(data[-1]["v"])
        except Exception as e:
            logger.warning(f"Failed to get hash rate from Glassnode: {e}")
        return None
    
    async def get_eth_gas_fees(self) -> Optional[float]:
        if not self.glassnode_api_key:
            return None
        try:
            data = await self._glassnode_request("fees/gas_price_mean", {"a": "ETH", "i": "24h"})
            if data and len(data) > 0:
                return float(data[-1]["v"])
        except Exception as e:
            logger.warning(f"Failed to get gas fees from Glassnode: {e}")
        return None


class ExchangeManager:
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self._exchange = None
        self._exchange_lock = asyncio.Lock()
        self.db_path = "trading_bot.db"
        self._db_initialized = False
        self._is_closed = False
        self.redis = redis_client

    async def init_database(self):
        if self._db_initialized:
            return
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS ohlcv (
                        symbol TEXT, timeframe TEXT, timestamp INTEGER,
                        open REAL, high REAL, low REAL, close REAL, volume REAL,
                        PRIMARY KEY(symbol, timeframe, timestamp)
                    )
                """
                )
                await db.execute("CREATE INDEX IF NOT EXISTS idx_symbol_tf_ts ON ohlcv(symbol, timeframe, timestamp)")
                await db.commit()
            self._db_initialized = True
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            self._db_initialized = False


    @asynccontextmanager
    async def get_exchange(self, market_type="spot"):
        async with self._exchange_lock:
            if self._exchange is None or self._exchange.options.get("defaultType") != market_type:
                if self._exchange:
                    try:
                        await self._exchange.close()
                    except Exception:
                        pass
                self._exchange = ccxt.coinex(
                    {
                        "apiKey": Config.COINEX_API_KEY,
                        "secret": Config.COINEX_SECRET,
                        "enableRateLimit": True,
                        "options": {"defaultType": market_type},
                    }
                )
            yield self._exchange

    async def close(self):
        if self._is_closed:
            return
        self._is_closed = True
        async with self._exchange_lock:
            if self._exchange:
                try:
                    await self._exchange.close()
                except Exception as e:
                    logger.warning(f"Error closing exchange connection: {e}")
                self._exchange = None

    @async_retry(attempts=3, delay=5)
    async def fetch_ohlcv_data(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        if self._is_closed:
            return pd.DataFrame()

        cache_key = f"cache:ohlcv:{symbol.replace('/', '_')}:{timeframe}"
        if self.redis:
            try:
                cached = await self.redis.get(cache_key)
                if cached:
                    df = pd.read_json(cached, orient="split")
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                    return df
            except Exception as e:
                logger.warning(f"Redis get failed for {cache_key}: {e}")

        try:
            async with self.get_exchange() as exchange:
                ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv:
                return pd.DataFrame()

            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.astype(
                {"open": float, "high": float, "low": float, "close": float, "volume": float}
            )

            if self.redis:
                try:
                    await self.redis.set(cache_key, df.to_json(orient="split"), ex=300)
                except Exception as e:
                    logger.warning(f"Redis set failed for {cache_key}: {e}")
            return df
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}/{timeframe}: {e}")
            return pd.DataFrame()

    @async_retry(attempts=2, delay=3)
    async def fetch_derivatives_data(self, symbol: str) -> Optional[DerivativesAnalysis]:
        cache_key = f"cache:derivatives:{symbol}"
        if self.redis:
            try:
                cached = await self.redis.get(cache_key)
                if cached:
                    return DerivativesAnalysis(**json.loads(cached))
            except Exception as e:
                logger.warning(f"Redis get failed for {cache_key}: {e}")
        try:
            async with self.get_exchange(market_type="swap") as exchange:
                if not exchange.has.get("fetchOpenInterest"):
                    return None

                tasks = {
                    "oi": exchange.fetch_open_interest(symbol),
                    "funding": exchange.fetch_funding_rate(symbol),
                }
                results = await asyncio.gather(*tasks.values(), return_exceptions=True)
                oi_res, fr_res = results

                open_interest = 0.0
                if not isinstance(oi_res, Exception) and isinstance(oi_res, dict):
                    open_interest = float(oi_res.get("openInterestAmount", 0.0))

                funding_rate = 0.0
                if not isinstance(fr_res, Exception) and isinstance(fr_res, dict):
                    funding_rate = float(fr_res.get("fundingRate", 0.0))

                analysis = DerivativesAnalysis(open_interest=open_interest, funding_rate=funding_rate)
                
                if self.redis:
                    try:
                        await self.redis.set(cache_key, json.dumps(analysis.__dict__), ex=60)
                    except Exception as e:
                        logger.warning(f"Redis set failed for {cache_key}: {e}")
                return analysis
        except Exception as e:
            logger.warning(f"Could not fetch derivatives data for {symbol}: {e}")
        return None

    @async_retry(attempts=2, delay=3)
    async def fetch_order_book(self, symbol: str, limit: int = 25) -> Optional[Dict]:
        cache_key = f"cache:orderbook:{symbol}"
        if self.redis:
            try:
                cached = await self.redis.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception as e:
                logger.warning(f"Redis get failed for {cache_key}: {e}")
        try:
            async with self.get_exchange() as exchange:
                if not exchange.has.get("fetchL2OrderBook"):
                    return None
                orderbook = await exchange.fetch_l2_order_book(symbol, limit)
                if self.redis and orderbook:
                    try:
                        await self.redis.set(cache_key, json.dumps(orderbook), ex=10)
                    except Exception as e:
                        logger.warning(f"Redis set failed for {cache_key}: {e}")
                return orderbook
        except Exception as e:
            logger.warning(f"Could not fetch order book for {symbol}: {e}")
        return None