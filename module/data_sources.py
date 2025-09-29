import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import aiohttp
import pandas as pd
import redis.asyncio as redis
import yfinance as yf

from module.config import Config
from module.core import FundamentalAnalysis, OrderBook
from module.logger_config import logger
from module.utils import RateLimiter, async_retry

coingecko_rate_limiter = RateLimiter(max_requests=45, time_window=60)
alpha_vantage_rate_limiter = RateLimiter(max_requests=5, time_window=60)
cryptopanic_rate_limiter = RateLimiter(max_requests=20, time_window=60)
alternative_me_rate_limiter = RateLimiter(max_requests=20, time_window=60)
coindesk_rate_limiter = RateLimiter(max_requests=20, time_window=60)
binance_rate_limiter = RateLimiter(max_requests=1200, time_window=60)
defillama_rate_limiter = RateLimiter(max_requests=30, time_window=60)


class BinanceFetcher:
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.base_url = "https://api.binance.com/api/v3"
        self.fapi_url = "https://fapi.binance.com/fapi/v1"
        self.futures_data_url = "https://fapi.binance.com/futures/data"
        self.redis = redis_client

    @async_retry(attempts=3, delay=5)
    async def _fetch(self, url: str, endpoint: str, params: Optional[Dict] = None):
        full_url = f"{url}/{endpoint}"
        await binance_rate_limiter.wait_if_needed(endpoint)
        async with aiohttp.ClientSession() as session:
            async with session.get(full_url, params=params) as response:
                if response.status == 400:
                    error_data = await response.json()
                    if error_data.get("code") == -1121 or "Invalid symbol" in error_data.get("msg", ""):
                        logger.warning(f"Invalid symbol for Binance endpoint {endpoint}: {params.get('symbol')}")
                        return None
                response.raise_for_status()
                return await response.json()

    async def get_historical_ohlc(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        try:
            binance_symbol = symbol.replace('/', '').upper()
            params = {"symbol": binance_symbol, "interval": timeframe, "limit": limit}
            data = await self._fetch(self.base_url, "klines", params)
            
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
            
            df = df[numeric_cols].dropna()

            return df
        except Exception as e:
            logger.error(f"Error fetching historical OHLC from Binance for {symbol}: {e}")
            return None

    async def get_open_interest(self, symbol: str) -> Optional[float]:
        cache_key = f"cache:binance:open_interest:{symbol}"
        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached: return float(cached)
        
        try:
            binance_symbol = symbol.replace('/', '').upper()
            params = {"symbol": binance_symbol}
            data = await self._fetch(self.fapi_url, "openInterest", params)
            if data and 'openInterest' in data:
                open_interest = float(data['openInterest'])
                if self.redis:
                    await self.redis.set(cache_key, open_interest, ex=300)
                return open_interest
            return None
        except Exception as e:
            logger.error(f"Error fetching open interest from Binance for {symbol}: {e}")
            return None
            
    async def get_funding_rate(self, symbol: str) -> Optional[float]:
        cache_key = f"cache:binance:funding_rate:{symbol}"
        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached: return float(cached)

        try:
            binance_symbol = symbol.replace('/', '').upper()
            params = {"symbol": binance_symbol, "limit": 1}
            data = await self._fetch(self.fapi_url, "fundingRate", params)
            if data and isinstance(data, list) and data[0].get('fundingRate'):
                funding_rate = float(data[0]['fundingRate'])
                if self.redis:
                    await self.redis.set(cache_key, funding_rate, ex=300)
                return funding_rate
            return None
        except Exception as e:
            logger.error(f"Error fetching funding rate from Binance for {symbol}: {e}")
            return None

    async def get_taker_long_short_ratio(self, symbol: str) -> Optional[float]:
        cache_key = f"cache:binance:long_short_ratio:{symbol}"
        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached: return float(cached)

        try:
            binance_symbol = symbol.replace('/', '').upper()
            params = {"symbol": binance_symbol, "period": "5m", "limit": 1}
            data = await self._fetch(self.futures_data_url, "takerlongshortRatio", params)
            if data and isinstance(data, list) and data[0].get('buySellRatio'):
                ratio = float(data[0]['buySellRatio'])
                if self.redis:
                    await self.redis.set(cache_key, ratio, ex=300)
                return ratio
            return None
        except Exception as e:
            logger.error(f"Error fetching taker long/short ratio from Binance for {symbol}: {e}")
            return None

    async def get_top_trader_long_short_ratio_accounts(self, symbol: str) -> Optional[float]:
        try:
            binance_symbol = symbol.replace('/', '').upper()
            params = {"symbol": binance_symbol, "period": "5m", "limit": 1}
            data = await self._fetch(self.futures_data_url, "globalLongShortAccountRatio", params)
            if data and isinstance(data, list) and data[0].get('longShortRatio'):
                return float(data[0]['longShortRatio'])
            return None
        except Exception as e:
            logger.error(f"Error fetching top trader acc ratio from Binance for {symbol}: {e}")
            return None

    async def get_top_trader_long_short_ratio_positions(self, symbol: str) -> Optional[float]:
        try:
            binance_symbol = symbol.replace('/', '').upper()
            params = {"symbol": binance_symbol, "period": "5m", "limit": 1}
            data = await self._fetch(self.futures_data_url, "topLongShortPositionRatio", params)
            if data and isinstance(data, list) and data[0].get('longShortRatio'):
                return float(data[0]['longShortRatio'])
            return None
        except Exception as e:
            logger.error(f"Error fetching top trader pos ratio from Binance for {symbol}: {e}")
            return None

    async def get_liquidation_orders(self, symbol: str) -> Optional[List[Dict]]:
        try:
            binance_symbol = symbol.replace('/', '').upper()
            params = {"symbol": binance_symbol, "limit": 10}
            data = await self._fetch(self.fapi_url, "allForceOrders", params)
            return data if data else None
        except Exception as e:
            logger.error(f"Error fetching liquidation orders from Binance for {symbol}: {e}")
            return None

    async def get_order_book_depth(self, symbol: str) -> Optional[OrderBook]:
        try:
            binance_symbol = symbol.replace('/', '').upper()
            params = {"symbol": binance_symbol, "limit": 100}
            data = await self._fetch(self.base_url, "depth", params)
            if not data: return None

            bids = [(float(p), float(q)) for p, q in data.get('bids', [])]
            asks = [(float(p), float(q)) for p, q in data.get('asks', [])]
            
            return OrderBook(
                bids=bids,
                asks=asks,
                bid_ask_spread=asks[0][0] - bids[0][0] if asks and bids else 0,
                total_bid_volume=sum(q for _, q in bids),
                total_ask_volume=sum(q for _, q in asks)
            )
        except Exception as e:
            logger.error(f"Error fetching order book depth from Binance for {symbol}: {e}")
            return None

    async def get_mark_price(self, symbol: str) -> Optional[float]:
        try:
            binance_symbol = symbol.replace('/', '').upper()
            params = {"symbol": binance_symbol}
            data = await self._fetch(self.fapi_url, "premiumIndex", params)
            if data and 'markPrice' in data:
                return float(data['markPrice'])
            return None
        except Exception as e:
            logger.error(f"Error fetching mark price from Binance for {symbol}: {e}")
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
    def __init__(self, alpha_vantage_key: Optional[str] = None, coingecko_key: Optional[str] = None, 
                 redis_client: Optional[redis.Redis] = None):
        
        self.coingecko_key = coingecko_key
        self.defillama_url = "https://api.llama.fi"
        self.alpha_vantage_url = "https://www.alphavantage.co/query"
        self.alpha_vantage_key = alpha_vantage_key
        self.redis = redis_client
        self.cache_ttl = 600

    @async_retry(attempts=3, delay=5)
    async def _fetch_json(self, url, params=None, headers=None, limiter: Optional[RateLimiter] = None, endpoint_name: str = "default"):
        if limiter:
            await limiter.wait_if_needed(endpoint_name)
        
        request_params = params.copy() if params else {}
        request_headers = headers.copy() if headers else {"accept": "application/json"}
        
        effective_url = url
        if "coingecko.com" in url and self.coingecko_key:
            if self.coingecko_key.startswith('CG-'):
                effective_url = url.replace("https://api.coingecko.com", "https://pro-api.coingecko.com")
                request_headers['x-cg-pro-api-key'] = self.coingecko_key
            else:
                effective_url = url.replace("https://pro-api.coingecko.com", "https://api.coingecko.com")
                request_params['x_cg_demo_api_key'] = self.coingecko_key

        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(effective_url, params=request_params, headers=request_headers) as response:
                    if response.status == 200:
                        return await response.json()
                    logger.error(f"Error fetching from {effective_url} with params {request_params}: {response.status}, message='{await response.text()}'")
                    response.raise_for_status()
        except asyncio.TimeoutError:
            logger.error(f"Timeout during fetch from {effective_url}")
            return None
        except Exception as e:
            logger.error(f"Exception during fetch from {effective_url}: {e}")
            return None

    async def get_historical_ohlc_coingecko(self, symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
        coin_id = symbol.lower().split('/')[0]
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
        params = {'vs_currency': 'usd', 'days': str(days)}
        data = await self._fetch_json(url, params, limiter=coingecko_rate_limiter, endpoint_name=f"coins_ohlc_{coin_id}")
        
        if not data: return None
        
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        df['volume'] = 0.0 
        return df

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
            "global": self._fetch_json(f"https://api.coingecko.com/api/v3/global", limiter=coingecko_rate_limiter, endpoint_name="global"),
            "defi_llama_tvl_total": self._fetch_json(f"{self.defillama_url}/tvl/chains", limiter=defillama_rate_limiter),
        }
        results = await asyncio.gather(*tasks.values())
        global_data, defi_llama_tvl_data = results

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

        if defi_llama_tvl_data and isinstance(defi_llama_tvl_data, list) and defi_llama_tvl_data:
            total_tvl = sum(chain.get('tvl', 0) for chain in defi_llama_tvl_data)
            indices["DEFI_TVL"] = total_tvl

        if self.redis and indices:
            try:
                await self.redis.set(cache_key, json.dumps(indices), ex=self.cache_ttl)
            except Exception as e:
                logger.warning(f"Redis set failed for {cache_key}: {e}")
        return indices
        
    async def get_fundamental_data(self, coin_id: str) -> Optional[FundamentalAnalysis]:
        cache_key = f"cache:coingecko_fundamental:{coin_id}"
        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached:
                data = json.loads(cached)
                return FundamentalAnalysis(**data)

        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
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
    
    async def get_coingecko_derivatives(self) -> Optional[List[Dict]]:
        url = "https://api.coingecko.com/api/v3/derivatives"
        data = await self._fetch_json(url, limiter=coingecko_rate_limiter, endpoint_name="derivatives")
        return data

    async def get_trending_searches(self) -> Optional[List[str]]:
        url = "https://api.coingecko.com/api/v3/search/trending"
        data = await self._fetch_json(url, limiter=coingecko_rate_limiter, endpoint_name="trending")
        if data and 'coins' in data:
            return [item['item']['symbol'] for item in data['coins']]
        return None

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

        tickers = {
            "DXY": "DX-Y.NYB", "SPX": "^GSPC", "NDX": "^IXIC", "VIX": "^VIX",
            "GOLD": "GC=F", "OIL": "CL=F", "DJI": "^DJI", "RUT": "^RUT"
        }
        data = yf.download(tickers=list(tickers.values()), period="5d", interval="1d", progress=False, auto_adjust=True)
        if data is None or data.empty:
            return {}

        price_column = "Close"
        if price_column not in data.columns:
            logger.error("Could not find 'Close' in yfinance data.")
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
            "cpi": self._fetch_json(self.alpha_vantage_url, params={"function": "CPI", "interval": "monthly", "apikey": self.alpha_vantage_key}, limiter=alpha_vantage_rate_limiter),
            "fed_rate": self._fetch_json(self.alpha_vantage_url, params={"function": "FEDERAL_FUNDS_RATE", "interval": "monthly", "apikey": self.alpha_vantage_key}, limiter=alpha_vantage_rate_limiter),
            "gdp": self._fetch_json(self.alpha_vantage_url, params={"function": "REAL_GDP", "interval": "quarterly", "apikey": self.alpha_vantage_key}, limiter=alpha_vantage_rate_limiter),
            "unemployment": self._fetch_json(self.alpha_vantage_url, params={"function": "UNEMPLOYMENT", "apikey": self.alpha_vantage_key}, limiter=alpha_vantage_rate_limiter),
            "treasury_yield": self._fetch_json(self.alpha_vantage_url, params={"function": "TREASURY_YIELD", "interval": "monthly", "maturity": "10year", "apikey": self.alpha_vantage_key}, limiter=alpha_vantage_rate_limiter),
        }
        results = await asyncio.gather(*tasks.values())
        cpi_data, fed_rate_data, gdp_data, unemployment_data, yield_data = results
        
        macro_data = {}
        if cpi_data and "data" in cpi_data and cpi_data["data"]: macro_data["CPI"] = float(cpi_data["data"][0]["value"])
        if fed_rate_data and "data" in fed_rate_data and fed_rate_data["data"]: macro_data["FED_RATE"] = float(fed_rate_data["data"][0]["value"])
        if gdp_data and "data" in gdp_data and gdp_data["data"]: macro_data["GDP"] = float(gdp_data["data"][0]["value"])
        if unemployment_data and "data" in unemployment_data and unemployment_data["data"]: macro_data["UNEMPLOYMENT"] = float(unemployment_data["data"][0]["value"])
        if yield_data and "data" in yield_data and yield_data["data"]: macro_data["TREASURY_YIELD_10Y"] = float(yield_data["data"][0]["value"])

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
    async def fetch_fear_greed(self, limit: int = 1):
        cache_key = f"cache:fear_greed:{limit}"
        if self.redis:
            try:
                cached = await self.redis.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception as e:
                logger.warning(f"Redis get failed for {cache_key}: {e}")

        await alternative_me_rate_limiter.wait_if_needed("fng")
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"https://api.alternative.me/fng/?limit={limit}") as response:
                response.raise_for_status()
                j = await response.json()
                data = j.get("data", [])
                if data:
                    if self.redis:
                        await self.redis.set(cache_key, json.dumps(data), ex=3600)
                    return data
        return None

    @async_retry(attempts=3, delay=5)
    async def fetch_news(self, currencies: List[str] = ["BTC", "ETH"], kind: Optional[str] = None):
        key = "cache:news:" + ",".join(sorted(currencies)) + (f":{kind}" if kind else "")
        if self.redis:
            try:
                cached = await self.redis.get(key)
                if cached: return json.loads(cached)
            except Exception as e:
                logger.warning(f"Redis get failed for {key}: {e}")

        tasks = []
        url = "https://cryptopanic.com/api/v2/posts/"
        params = {"auth_token": self.cryptopanic_key, "currencies": ",".join(currencies), "public": "true"}
        if kind: params['kind'] = kind
        tasks.append(self._fetch_cryptopanic(url, params))

        if self.coindesk_fetcher and not kind: # CoinDesk doesn't support 'kind' filter
            tasks.append(self.coindesk_fetcher.get_news(currencies))

        results = await asyncio.gather(*tasks)
        all_news = []
        for res in results:
            if res: all_news.extend(res)
        
        unique_news = list({item.get('id') or item.get('title'): item for item in all_news}.values())

        if self.redis and unique_news:
            await self.redis.set(key, json.dumps(unique_news), ex=600)

        return unique_news

    async def _fetch_cryptopanic(self, url, params):
        await cryptopanic_rate_limiter.wait_if_needed("posts")
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                return data.get("results", [])
        return []

    def score_news(self, news_items: List[Dict[str, Any]]):
        score = 0
        if not news_items: return score
        for it in news_items:
            # CryptoPanic specific sentiment
            if 'votes' in it:
                score += int(it['votes'].get('positive', 0))
                score -= int(it['votes'].get('negative', 0))
            
            # General title sentiment
            title = (it.get("title") or "").lower()
            if any(k in title for k in ["bull", "rally", "surge", "gain", "pump", "partnership"]): score += 1
            if any(k in title for k in ["crash", "dump", "fall", "drop", "hack", "scam", "exploit"]): score -= 1
        return score