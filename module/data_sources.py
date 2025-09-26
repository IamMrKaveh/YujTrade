import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import aiohttp
import pandas as pd
import redis.asyncio as redis
import yfinance as yf

from module.config import Config
from module.logger_config import logger
from module.utils import RateLimiter, async_retry

coingecko_rate_limiter = RateLimiter(max_requests=45, time_window=60)
alpha_vantage_rate_limiter = RateLimiter(max_requests=5, time_window=60)
cryptopanic_rate_limiter = RateLimiter(max_requests=20, time_window=60)
alternative_me_rate_limiter = RateLimiter(max_requests=20, time_window=60)
coindesk_rate_limiter = RateLimiter(max_requests=20, time_window=60)
binance_rate_limiter = RateLimiter(max_requests=1200, time_window=60)


class BinanceFetcher:
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.base_url = "https://api.binance.com/api/v3"
        self.redis = redis_client

    @async_retry(attempts=3, delay=5)
    async def _fetch(self, endpoint: str, params: Optional[Dict] = None):
        url = f"{self.base_url}/{endpoint}"
        await binance_rate_limiter.wait_if_needed(endpoint)
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                return await response.json()

    async def get_historical_ohlc(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        cache_key = f"cache:binance:ohlc:{symbol}:{timeframe}:{limit}"
        if self.redis:
            try:
                cached = await self.redis.get(cache_key)
                if cached:
                    df = pd.read_json(cached, orient="split")
                    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                    df.set_index('timestamp', inplace=True)
                    return df
            except Exception as e:
                logger.warning(f"Redis get failed for {cache_key}: {e}")
        
        try:
            binance_symbol = symbol.replace('/', '').upper()
            params = {"symbol": binance_symbol, "interval": timeframe, "limit": limit}
            data = await self._fetch("klines", params)
            
            if not data:
                return None

            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            
            df = df[numeric_cols]

            if self.redis:
                await self.redis.set(cache_key, df.to_json(orient="split"), ex=300)
            
            return df
        except Exception as e:
            logger.error(f"Error fetching historical OHLC from Binance for {symbol}: {e}")
            return None


class CoinDeskFetcher:
    def __init__(self, api_key: str, redis_client: Optional[redis.Redis] = None):
        self.api_key = api_key
        self.base_url = "https://data-api.coindesk.com"
        self.redis = redis_client

    @async_retry(attempts=3, delay=5)
    async def _fetch(self, endpoint: str, params: Optional[Dict] = None):
        headers = {"x-api-key": self.api_key}
        url = f"{self.base_url}/{endpoint}"
        await coindesk_rate_limiter.wait_if_needed(endpoint)
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers) as response:
                response.raise_for_status()
                return await response.json()

    async def get_historical_ohlc(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        if 'd' not in timeframe and 'w' not in timeframe and 'M' not in timeframe:
            logger.debug("CoinDesk fetcher is intended for daily+ timeframes. Use Binance for shorter ones.")
            return None

        cache_key = f"cache:coindesk:ohlc:{symbol}:{timeframe}:{limit}"
        if self.redis:
            try:
                cached = await self.redis.get(cache_key)
                if cached:
                    df = pd.read_json(cached, orient="split")
                    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                    df.set_index('timestamp', inplace=True)
                    return df
            except Exception as e:
                logger.warning(f"Redis get failed for {cache_key}: {e}")
        
        try:
            params = {
                "instruments": symbol.upper().replace('/USDT', '-USD'),
                "time_frame": timeframe,
                "limit": limit,
            }
            endpoint = "spot/v1/historical/ohlcv/days" if timeframe == '1d' else "spot/v1/historical/ohlcv"
            data = await self._fetch(endpoint, params)
            
            if not data or 'data' not in data or 'DATA' not in data['data'] or not data['data']['DATA']:
                return None

            ohlc_data = data['data']['DATA']
            if not ohlc_data:
                return None

            df = pd.DataFrame(ohlc_data)
            df.rename(columns={
                'ts': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'
            }, inplace=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            df = df.astype(float)
            df = df.sort_index()

            if self.redis:
                await self.redis.set(cache_key, df.to_json(orient="split"), ex=300)
            
            return df.tail(limit)
        except Exception as e:
            logger.error(f"Error fetching historical OHLC from CoinDesk for {symbol}: {e}")
            return None

    async def get_news(self, symbols: List[str]) -> List[Dict]:
        try:
            all_news = []
            news_url = "https://api.coindesk.com/v1/news"
            params = {'slug': ",".join(s.lower().replace('-usdt', '') for s in symbols)}
            headers = {"X-CoinDesk-API-Key": self.api_key}
            
            async with aiohttp.ClientSession() as session:
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


class MarketIndicesFetcher:
    def __init__(self, alpha_vantage_key: Optional[str] = None, coingecko_key: Optional[str] = None, redis_client: Optional[redis.Redis] = None):
        self.coingecko_base_url = "https://api.coingecko.com/api/v3"
        if coingecko_key:
            self.coingecko_base_url = "https://pro-api.coingecko.com/api/v3"
        
        self.defillama_url = "https://api.llama.fi"
        self.alpha_vantage_url = "https://www.alphavantage.co/query"
        self.alpha_vantage_key = alpha_vantage_key
        self.coingecko_key = coingecko_key
        self.redis = redis_client
        self.cache_ttl = 600

    @async_retry(attempts=3, delay=5)
    async def _fetch_json(self, url, params=None, limiter: Optional[RateLimiter] = None, endpoint_name: str = "default"):
        if limiter:
            await limiter.wait_if_needed(endpoint_name)
        
        request_params = params.copy() if params else {}
        
        if "coingecko.com" in url and self.coingecko_key:
            request_params['x_cg_pro_api_key'] = self.coingecko_key

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=request_params, timeout=15) as response:
                    if response.status == 200:
                        return await response.json()
                    response.raise_for_status()
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
            "global": self._fetch_json(f"{self.coingecko_base_url}/global", limiter=coingecko_rate_limiter, endpoint_name="global"),
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

    @async_retry(attempts=3, delay=5)
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

        price_column = "Adj Close"
        if price_column not in data.columns:
            price_column = "Close"
        
        if price_column not in data.columns:
            logger.error("Could not find 'Adj Close' or 'Close' in yfinance data.")
            return {}

        indices = {
            key: data[price_column][ticker].iloc[-1] for key, ticker in tickers.items() if ticker in data[price_column] and not pd.isna(data[price_column][ticker].iloc[-1])
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
                params={"function": "CPI", "interval": "monthly", "apikey": self.alpha_vantage_key},
                limiter=alpha_vantage_rate_limiter, endpoint_name="cpi"
            ),
            "fed_rate": self._fetch_json(
                self.alpha_vantage_url,
                params={"function": "FEDERAL_FUNDS_RATE", "interval": "monthly", "apikey": self.alpha_vantage_key},
                limiter=alpha_vantage_rate_limiter, endpoint_name="fed_rate"
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
    def __init__(self, cryptopanic_key: str, coindesk_fetcher: Optional[CoinDeskFetcher] = None, redis_client: Optional[redis.Redis] = None):
        self.cryptopanic_key = cryptopanic_key
        self.coindesk_fetcher = coindesk_fetcher
        self.redis = redis_client

    @async_retry(attempts=3, delay=5)
    async def fetch_fear_greed(self):
        cache_key = "cache:fear_greed"
        if self.redis:
            try:
                cached = await self.redis.get(cache_key)
                if cached:
                    return int(cached)
            except Exception as e:
                logger.warning(f"Redis get failed for {cache_key}: {e}")

        await alternative_me_rate_limiter.wait_if_needed("fng")
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.alternative.me/fng/", timeout=10) as response:
                response.raise_for_status()
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
        return None

    @async_retry(attempts=3, delay=5)
    async def fetch_news(self, currencies: List[str] = ["BTC", "ETH"]):
        key = "cache:news:" + ",".join(sorted(currencies))
        if self.redis:
            try:
                cached = await self.redis.get(key)
                if cached:
                    return json.loads(cached)
            except Exception as e:
                logger.warning(f"Redis get failed for {key}: {e}")

        tasks = []
        url = "https://cryptopanic.com/api/v2/posts/"
        params = {"auth_token": self.cryptopanic_key, "currencies": ",".join(currencies), "public": "true"}
        tasks.append(self._fetch_cryptopanic(url, params))

        if self.coindesk_fetcher:
            tasks.append(self.coindesk_fetcher.get_news(currencies))

        results = await asyncio.gather(*tasks)
        all_news = []
        for res in results:
            if res:
                all_news.extend(res)
        
        unique_news = list({item.get('id') or item.get('title'): item for item in all_news}.values())

        if self.redis and unique_news:
            try:
                await self.redis.set(key, json.dumps(unique_news), ex=600)
            except Exception as e:
                logger.warning(f"Redis set failed for {key}: {e}")

        return unique_news

    async def _fetch_cryptopanic(self, url, params):
        await cryptopanic_rate_limiter.wait_if_needed("posts")
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=10) as response:
                response.raise_for_status()
                data = await response.json()
                return data.get("results", [])
        return []

    def score_news(self, news_items: List[Dict[str, Any]]):
        score = 0
        if not news_items:
            return score
        for it in news_items:
            title = (it.get("title") or "").lower()
            if any(k in title for k in ["bull", "rally", "surge", "gain", "pump"]):
                score += 1
            if any(k in title for k in ["crash", "dump", "fall", "drop", "hack"]):
                score -= 1
        return score