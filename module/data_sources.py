import asyncio
import json
from typing import Any, Dict, List, Optional

import aiohttp
import pandas as pd
import redis.asyncio as redis
import yfinance as yf

from module.circuit_breaker import CircuitBreaker
from module.core import FundamentalAnalysis, OrderBook, OnChainAnalysis
from module.exceptions import APIRateLimitError, InvalidSymbolError, NetworkError, DataError
from module.logger_config import logger
from module.utils import RateLimiter, async_retry

coingecko_rate_limiter = RateLimiter(max_requests=30, time_window=60)
alpha_vantage_rate_limiter = RateLimiter(max_requests=5, time_window=60)
cryptopanic_rate_limiter = RateLimiter(max_requests=20, time_window=60)
alternative_me_rate_limiter = RateLimiter(max_requests=20, time_window=60)
coindesk_rate_limiter = RateLimiter(max_requests=20, time_window=60)
binance_rate_limiter = RateLimiter(max_requests=1200, time_window=60)
defillama_rate_limiter = RateLimiter(max_requests=30, time_window=60)
messari_rate_limiter = RateLimiter(max_requests=20, time_window=60)


class BinanceFetcher:
    FUTURES_BLACKLIST = {'SHIBUSDT', 'PEPEUSDT', 'FLOKIUSDT', 'ASTRUSDT', 'ONDOUSDT'}

    def __init__(self, redis_client: Optional[redis.Redis] = None, session: Optional[aiohttp.ClientSession] = None):
        self.base_url = "https://api.binance.com/api/v3"
        self.fapi_url = "https://fapi.binance.com/fapi/v1"
        self.futures_data_url = "https://fapi.binance.com/futures/data"
        self.redis = redis_client
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=60)
        self.session = session
        self._external_session = session is not None
        self._is_closed = False

    def _check_if_closed(self):
        if self._is_closed:
            raise RuntimeError("BinanceFetcher has been closed and cannot be used")

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
    async def _fetch(self, url: str, endpoint: str, params: Optional[Dict] = None):
        self._check_if_closed()
        full_url = f"{url}/{endpoint}"
        await binance_rate_limiter.wait_if_needed(endpoint)
        
        async def _do_fetch():
            session = await self._get_session()
            async with session.get(full_url, params=params) as response:
                if response.status == 429:
                    raise APIRateLimitError(f"Rate limit exceeded for {endpoint}")
                if response.status == 418:
                    logger.warning(f"IP banned by Binance for {endpoint}. Sleeping for 120s.")
                    await asyncio.sleep(120)
                    raise APIRateLimitError(f"IP banned, retrying for {endpoint}")
                if response.status == 400:
                    try:
                        error_data = await response.json()
                        if error_data.get("code") == -1121 or "Invalid symbol" in error_data.get("msg", ""):
                            raise InvalidSymbolError(f"Invalid symbol for Binance: {params.get('symbol') if params else 'N/A'}")
                    except (aiohttp.ContentTypeError, json.JSONDecodeError):
                        raise NetworkError(f"Bad request to {full_url}")
                response.raise_for_status()
                return await response.json()
        
        try:
            return await self.circuit_breaker.call(_do_fetch)
        except Exception as e:
            if isinstance(e, (NetworkError, APIRateLimitError, InvalidSymbolError)):
                raise
            logger.error(f"Error in _fetch for {full_url}: {e}", exc_info=True)
            raise DataError(f"Failed to fetch data from {full_url}") from e

    async def get_historical_ohlc(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        self._check_if_closed()
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
        except InvalidSymbolError:
            logger.warning(f"Invalid symbol for Binance OHLC: {symbol}")
            return None
        except Exception as e:
            logger.error(f"Error fetching historical OHLC from Binance for {symbol}: {e}")
            raise DataError(f"Failed to fetch OHLC for {symbol}") from e

    async def get_open_interest(self, symbol: str) -> Optional[float]:
        self._check_if_closed()
        binance_symbol = symbol.replace('/', '').upper()
        if binance_symbol in self.FUTURES_BLACKLIST:
            return None
        
        cache_key = f"cache:binance:open_interest:{symbol}"
        try:
            if self.redis:
                cached = await self.redis.get(cache_key)
                if cached:
                    return float(cached)
        except Exception as e:
            logger.warning(f"Redis GET failed for {cache_key}: {e}")
        
        try:
            params = {"symbol": binance_symbol}
            data = await self._fetch(self.fapi_url, "openInterest", params)
            if data and 'openInterest' in data:
                open_interest = float(data['openInterest'])
                try:
                    if self.redis:
                        await self.redis.set(cache_key, open_interest, ex=300)
                except Exception as e:
                    logger.warning(f"Redis SET failed for {cache_key}: {e}")
                return open_interest
            return None
        except Exception as e:
            logger.error(f"Error fetching open interest from Binance for {symbol}: {e}")
            return None

    async def get_taker_long_short_ratio(self, symbol: str) -> Optional[float]:
        self._check_if_closed()
        binance_symbol = symbol.replace('/', '').upper()
        if binance_symbol in self.FUTURES_BLACKLIST:
            return None
        
        cache_key = f"cache:binance:long_short_ratio:{symbol}"
        try:
            if self.redis:
                cached = await self.redis.get(cache_key)
                if cached:
                    return float(cached)
        except Exception as e:
            logger.warning(f"Redis GET failed for {cache_key}: {e}")

        try:
            params = {"symbol": binance_symbol, "period": "5m", "limit": 1}
            data = await self._fetch(self.futures_data_url, "takerlongshortRatio", params)
            if data and isinstance(data, list) and data and data[0].get('buySellRatio'):
                ratio = float(data[0]['buySellRatio'])
                try:
                    if self.redis:
                        await self.redis.set(cache_key, ratio, ex=300)
                except Exception as e:
                    logger.warning(f"Redis SET failed for {cache_key}: {e}")
                return ratio
            return None
        except Exception as e:
            logger.error(f"Error fetching taker long/short ratio from Binance for {symbol}: {e}")
            return None

    async def get_top_trader_long_short_ratio_accounts(self, symbol: str) -> Optional[float]:
        self._check_if_closed()
        binance_symbol = symbol.replace('/', '').upper()
        if binance_symbol in self.FUTURES_BLACKLIST:
            return None
        try:
            params = {"symbol": binance_symbol, "period": "5m", "limit": 1}
            data = await self._fetch(self.futures_data_url, "globalLongShortAccountRatio", params)
            if data and isinstance(data, list) and data and data[0].get('longShortRatio'):
                return float(data[0]['longShortRatio'])
            return None
        except Exception as e:
            logger.error(f"Error fetching top trader acc ratio from Binance for {symbol}: {e}")
            return None

    async def get_top_trader_long_short_ratio_positions(self, symbol: str) -> Optional[float]:
        self._check_if_closed()
        binance_symbol = symbol.replace('/', '').upper()
        if binance_symbol in self.FUTURES_BLACKLIST:
            return None
        try:
            params = {"symbol": binance_symbol, "period": "5m", "limit": 1}
            data = await self._fetch(self.futures_data_url, "topLongShortPositionRatio", params)
            if data and isinstance(data, list) and data and data[0].get('longShortRatio'):
                return float(data[0]['longShortRatio'])
            return None
        except Exception as e:
            logger.error(f"Error fetching top trader pos ratio from Binance for {symbol}: {e}")
            return None

    async def get_liquidation_orders(self, symbol: str) -> Optional[List[Dict]]:
        self._check_if_closed()
        logger.debug(f"Skipping get_liquidation_orders for {symbol} as the endpoint is likely deprecated.")
        return None

    async def get_order_book_depth(self, symbol: str) -> Optional[OrderBook]:
        self._check_if_closed()
        try:
            binance_symbol = symbol.replace('/', '').upper()
            params = {"symbol": binance_symbol, "limit": 100}
            data = await self._fetch(self.base_url, "depth", params)
            if not data:
                return None

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
        self._check_if_closed()
        binance_symbol = symbol.replace('/', '').upper()
        if binance_symbol in self.FUTURES_BLACKLIST:
            return None
        try:
            params = {"symbol": binance_symbol}
            data = await self._fetch(self.fapi_url, "premiumIndex", params)
            if data and 'markPrice' in data:
                return float(data['markPrice'])
            return None
        except Exception as e:
            logger.error(f"Error fetching mark price from Binance for {symbol}: {e}")
            return None

    async def get_24h_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        self._check_if_closed()
        try:
            binance_symbol = symbol.replace('/', '').upper()
            params = {"symbol": binance_symbol}
            data = await self._fetch(self.base_url, "ticker/24hr", params)
            if data:
                return {
                    'price_change': float(data.get('priceChange', 0)),
                    'price_change_percent': float(data.get('priceChangePercent', 0)),
                    'weighted_avg_price': float(data.get('weightedAvgPrice', 0)),
                    'prev_close_price': float(data.get('prevClosePrice', 0)),
                    'last_price': float(data.get('lastPrice', 0)),
                    'bid_price': float(data.get('bidPrice', 0)),
                    'ask_price': float(data.get('askPrice', 0)),
                    'open_price': float(data.get('openPrice', 0)),
                    'high_price': float(data.get('highPrice', 0)),
                    'low_price': float(data.get('lowPrice', 0)),
                    'volume': float(data.get('volume', 0)),
                    'quote_volume': float(data.get('quoteVolume', 0)),
                    'open_time': data.get('openTime'),
                    'close_time': data.get('closeTime'),
                    'first_id': data.get('firstId'),
                    'last_id': data.get('lastId'),
                    'count': data.get('count')
                }
            return None
        except Exception as e:
            logger.error(f"Error fetching 24h ticker from Binance for {symbol}: {e}")
            return None

    async def get_long_short_ratio_history(self, symbol: str, period: str = "5m", limit: int = 30) -> Optional[List[Dict]]:
        self._check_if_closed()
        binance_symbol = symbol.replace('/', '').upper()
        if binance_symbol in self.FUTURES_BLACKLIST:
            return None
        
        try:
            params = {"symbol": binance_symbol, "period": period, "limit": limit}
            data = await self._fetch(self.futures_data_url, "takerlongshortRatio", params)
            return data if isinstance(data, list) else None
        except Exception as e:
            logger.error(f"Error fetching long/short ratio history from Binance for {symbol}: {e}")
            return None

    async def get_open_interest_history(self, symbol: str, period: str = "5m", limit: int = 30) -> Optional[List[Dict]]:
        self._check_if_closed()
        binance_symbol = symbol.replace('/', '').upper()
        if binance_symbol in self.FUTURES_BLACKLIST:
            return None
        
        try:
            params = {"symbol": binance_symbol, "period": period, "limit": limit}
            data = await self._fetch(self.fapi_url, "openInterestHist", params)
            return data if isinstance(data, list) else None
        except Exception as e:
            logger.error(f"Error fetching open interest history from Binance for {symbol}: {e}")
            return None

    async def get_funding_rate(self, symbol: str) -> Optional[float]:
        binance_symbol = symbol.replace('/', '').upper()
        if binance_symbol in self.FUTURES_BLACKLIST:
            return None
        
        cache_key = f"cache:binance:funding_rate:{symbol}"
        try:
            if self.redis:
                cached = await self.redis.get(cache_key)
                if cached:
                    return float(cached)
        except Exception as e:
            logger.warning(f"Redis GET failed for {cache_key}: {e}")
        
        try:
            params = {"symbol": binance_symbol}
            data = await self._fetch(self.fapi_url, "fundingRate", params)
            if data and isinstance(data, list) and len(data) > 0 and 'fundingRate' in data[0]:
                funding_rate = float(data[0]['fundingRate'])
                try:
                    if self.redis:
                        await self.redis.set(cache_key, funding_rate, ex=300)
                except Exception as e:
                    logger.warning(f"Redis SET failed for {cache_key}: {e}")
                return funding_rate
            return None
        except Exception as e:
            logger.error(f"Error fetching funding rate from Binance for {symbol}: {e}")
            return None

    async def get_kline_data(self, symbol: str, interval: str, start_time: Optional[int] = None, 
                            end_time: Optional[int] = None, limit: int = 500) -> Optional[List]:
        self._check_if_closed()
        try:
            binance_symbol = symbol.replace('/', '').upper()
            params = {"symbol": binance_symbol, "interval": interval, "limit": limit}
            if start_time:
                params["startTime"] = start_time
            if end_time:
                params["endTime"] = end_time
            data = await self._fetch(self.base_url, "klines", params)
            return data
        except Exception as e:
            logger.error(f"Error fetching kline data from Binance for {symbol}: {e}")
            return None

    async def get_ticker_price(self, symbol: str) -> Optional[float]:
        self._check_if_closed()
        try:
            binance_symbol = symbol.replace('/', '').upper()
            params = {"symbol": binance_symbol}
            data = await self._fetch(self.base_url, "ticker/price", params)
            if data and 'price' in data:
                return float(data['price'])
            return None
        except Exception as e:
            logger.error(f"Error fetching ticker price from Binance for {symbol}: {e}")
            return None

    async def get_exchange_info(self, symbol: Optional[str] = None) -> Optional[Dict]:
        self._check_if_closed()
        cache_key = f"cache:binance:exchange_info:{symbol}" if symbol else "cache:binance:exchange_info"
        try:
            if self.redis:
                cached = await self.redis.get(cache_key)
                if cached:
                    return json.loads(cached)
        except Exception as e:
            logger.warning(f"Redis GET failed for {cache_key}: {e}")
        
        try:
            params = {}
            if symbol:
                params["symbol"] = symbol.replace('/', '').upper()
            data = await self._fetch(self.base_url, "exchangeInfo", params)
            if data and self.redis:
                await self.redis.set(cache_key, json.dumps(data), ex=86400)
            return data
        except Exception as e:
            logger.error(f"Error fetching exchange info from Binance: {e}")
            return None

    async def get_agg_trades(self, symbol: str, limit: int = 500) -> Optional[List[Dict]]:
        self._check_if_closed()
        try:
            binance_symbol = symbol.replace('/', '').upper()
            params = {"symbol": binance_symbol, "limit": limit}
            data = await self._fetch(self.base_url, "aggTrades", params)
            return data
        except Exception as e:
            logger.error(f"Error fetching aggregate trades from Binance for {symbol}: {e}")
            return None

    async def get_all_futures_symbols(self) -> Optional[List[Dict]]:
        self._check_if_closed()
        cache_key = "cache:binance:futures_symbols"
        try:
            if self.redis:
                cached = await self.redis.get(cache_key)
                if cached:
                    return json.loads(cached)
        except Exception as e:
            logger.warning(f"Redis GET failed for {cache_key}: {e}")
        
        try:
            data = await self._fetch(self.fapi_url, "exchangeInfo")
            if data and 'symbols' in data:
                symbols = data['symbols']
                if self.redis:
                    await self.redis.set(cache_key, json.dumps(symbols), ex=86400)
                return symbols
            return None
        except Exception as e:
            logger.error(f"Error fetching futures symbols: {e}")
            return None

    async def get_composite_index(self, symbol: str) -> Optional[Dict]:
        self._check_if_closed()
        binance_symbol = symbol.replace('/', '').upper()
        try:
            params = {"symbol": binance_symbol}
            data = await self._fetch(self.fapi_url, "indexInfo", params)
            return data
        except Exception as e:
            logger.error(f"Error fetching composite index for {symbol}: {e}")
            return None

    async def get_historical_funding_rate(self, symbol: str, start_time: Optional[int] = None, limit: int = 1000) -> Optional[List[Dict]]:
        self._check_if_closed()
        binance_symbol = symbol.replace('/', '').upper()
        if binance_symbol in self.FUTURES_BLACKLIST:
            return None
        
        try:
            params = {"symbol": binance_symbol, "limit": limit}
            if start_time:
                params["startTime"] = start_time
            data = await self._fetch(self.fapi_url, "fundingRate", params)
            return data if isinstance(data, list) else None
        except Exception as e:
            logger.error(f"Error fetching historical funding rate for {symbol}: {e}")
            return None

    async def get_continuous_klines(self, symbol: str, contract_type: str, interval: str, limit: int = 500) -> Optional[List]:
        self._check_if_closed()
        binance_symbol = symbol.replace('/', '').upper()
        try:
            params = {
                "pair": binance_symbol,
                "contractType": contract_type,
                "interval": interval,
                "limit": limit
            }
            data = await self._fetch(self.fapi_url, "continuousKlines", params)
            return data
        except Exception as e:
            logger.error(f"Error fetching continuous klines for {symbol}: {e}")
            return None

    async def get_index_price_klines(self, symbol: str, interval: str, limit: int = 500) -> Optional[List]:
        self._check_if_closed()
        binance_symbol = symbol.replace('/', '').upper()
        try:
            params = {"pair": binance_symbol, "interval": interval, "limit": limit}
            data = await self._fetch(self.fapi_url, "indexPriceKlines", params)
            return data
        except Exception as e:
            logger.error(f"Error fetching index price klines for {symbol}: {e}")
            return None

    async def get_mark_price_klines(self, symbol: str, interval: str, limit: int = 500) -> Optional[List]:
        self._check_if_closed()
        binance_symbol = symbol.replace('/', '').upper()
        try:
            params = {"symbol": binance_symbol, "interval": interval, "limit": limit}
            data = await self._fetch(self.fapi_url, "markPriceKlines", params)
            return data
        except Exception as e:
            logger.error(f"Error fetching mark price klines for {symbol}: {e}")
            return None

    async def get_top_long_short_account_ratio(self, symbol: str, period: str = "5m", limit: int = 100) -> Optional[List[Dict]]:
        self._check_if_closed()
        binance_symbol = symbol.replace('/', '').upper()
        if binance_symbol in self.FUTURES_BLACKLIST:
            return None
        
        try:
            params = {"symbol": binance_symbol, "period": period, "limit": limit}
            data = await self._fetch(self.futures_data_url, "topLongShortAccountRatio", params)
            return data if isinstance(data, list) else None
        except Exception as e:
            logger.error(f"Error fetching top trader account ratio for {symbol}: {e}")
            return None

    async def get_top_long_short_position_ratio(self, symbol: str, period: str = "5m", limit: int = 100) -> Optional[List[Dict]]:
        self._check_if_closed()
        binance_symbol = symbol.replace('/', '').upper()
        if binance_symbol in self.FUTURES_BLACKLIST:
            return None
        
        try:
            params = {"symbol": binance_symbol, "period": period, "limit": limit}
            data = await self._fetch(self.futures_data_url, "topLongShortPositionRatio", params)
            return data if isinstance(data, list) else None
        except Exception as e:
            logger.error(f"Error fetching top trader position ratio for {symbol}: {e}")
            return None

    async def get_global_long_short_account_ratio(self, symbol: str, period: str = "5m", limit: int = 100) -> Optional[List[Dict]]:
        self._check_if_closed()
        binance_symbol = symbol.replace('/', '').upper()
        if binance_symbol in self.FUTURES_BLACKLIST:
            return None
        
        try:
            params = {"symbol": binance_symbol, "period": period, "limit": limit}
            data = await self._fetch(self.futures_data_url, "globalLongShortAccountRatio", params)
            return data if isinstance(data, list) else None
        except Exception as e:
            logger.error(f"Error fetching global long/short account ratio for {symbol}: {e}")
            return None

    async def get_basis_data(self, symbol: str, contract_type: str, period: str = "5m", limit: int = 100) -> Optional[List[Dict]]:
        self._check_if_closed()
        binance_symbol = symbol.replace('/', '').upper()
        try:
            params = {
                "pair": binance_symbol,
                "contractType": contract_type,
                "period": period,
                "limit": limit
            }
            data = await self._fetch(self.futures_data_url, "basis", params)
            return data if isinstance(data, list) else None
        except Exception as e:
            logger.error(f"Error fetching basis data for {symbol}: {e}")
            return None

    async def get_order_book_ticker(self, symbol: str) -> Optional[Dict]:
        self._check_if_closed()
        try:
            binance_symbol = symbol.replace('/', '').upper()
            params = {"symbol": binance_symbol}
            data = await self._fetch(self.base_url, "ticker/bookTicker", params)
            return data
        except Exception as e:
            logger.error(f"Error fetching order book ticker for {symbol}: {e}")
            return None

    async def get_recent_trades(self, symbol: str, limit: int = 500) -> Optional[List[Dict]]:
        self._check_if_closed()
        try:
            binance_symbol = symbol.replace('/', '').upper()
            params = {"symbol": binance_symbol, "limit": limit}
            data = await self._fetch(self.base_url, "trades", params)
            return data
        except Exception as e:
            logger.error(f"Error fetching recent trades for {symbol}: {e}")
            return None

    async def get_historical_trades(self, symbol: str, limit: int = 500, from_id: Optional[int] = None) -> Optional[List[Dict]]:
        self._check_if_closed()
        try:
            binance_symbol = symbol.replace('/', '').upper()
            params = {"symbol": binance_symbol, "limit": limit}
            if from_id:
                params["fromId"] = from_id
            data = await self._fetch(self.base_url, "historicalTrades", params)
            return data
        except Exception as e:
            logger.error(f"Error fetching historical trades for {symbol}: {e}")
            return None

    async def get_avg_price(self, symbol: str) -> Optional[Dict]:
        self._check_if_closed()
        try:
            binance_symbol = symbol.replace('/', '').upper()
            params = {"symbol": binance_symbol}
            data = await self._fetch(self.base_url, "avgPrice", params)
            return data
        except Exception as e:
            logger.error(f"Error fetching average price for {symbol}: {e}")
            return None

    async def get_all_tickers(self) -> Optional[List[Dict]]:
        self._check_if_closed()
        cache_key = "cache:binance:all_tickers"
        try:
            if self.redis:
                cached = await self.redis.get(cache_key)
                if cached:
                    return json.loads(cached)
        except Exception as e:
            logger.warning(f"Redis GET failed for {cache_key}: {e}")
        
        try:
            data = await self._fetch(self.base_url, "ticker/24hr")
            if data and self.redis:
                await self.redis.set(cache_key, json.dumps(data), ex=60)
            return data
        except Exception as e:
            logger.error(f"Error fetching all tickers: {e}")
            return None

    async def get_all_book_tickers(self) -> Optional[List[Dict]]:
        self._check_if_closed()
        try:
            data = await self._fetch(self.base_url, "ticker/bookTicker")
            return data
        except Exception as e:
            logger.error(f"Error fetching all book tickers: {e}")
            return None


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


class BaseFetcher:
    def __init__(self, redis_client: Optional[redis.Redis] = None, session: Optional[aiohttp.ClientSession] = None):
        self.redis = redis_client
        self.session = session
        self._external_session = session is not None
        self._is_closed = False
        self.circuit_breaker = CircuitBreaker()

    def _check_if_closed(self):
        if self._is_closed:
            raise RuntimeError(f"{self.__class__.__name__} has been closed and cannot be used")

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
    async def _fetch_json(self, url, params=None, headers=None, limiter: Optional[RateLimiter] = None, endpoint_name: str = "default"):
        self._check_if_closed()
        if limiter:
            await limiter.wait_if_needed(endpoint_name)

        async def _do_fetch():
            session = await self._get_session()
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 429:
                    raise APIRateLimitError(f"Rate limit exceeded for {url}")
                if response.status != 200:
                    logger.error(f"Error fetching from {url}: {response.status}")
                    raise NetworkError(f"HTTP Error {response.status} for {url}")
                return await response.json()
        
        return await self.circuit_breaker.call(_do_fetch)


class CoinGeckoFetcher(BaseFetcher):
    COIN_ID_MAP = {
        'BTC': 'bitcoin',
        'ETH': 'ethereum',
        'SOL': 'solana',
        'BNB': 'binancecoin',
        'XRP': 'ripple',
        'DOGE': 'dogecoin',
        'AVAX': 'avalanche-2',
        'ADA': 'cardano',
        'TRX': 'tron',
        'SHIB': 'shiba-inu',
        'LTC': 'litecoin',
        'LINK': 'chainlink',
        'DOT': 'polkadot',
        'SUI': 'sui',
        'ASTR': 'astar',
        'TON': 'the-open-network',
        'PEPE': 'pepe',
        'SEI': 'sei-network',
        'ARB': 'arbitrum',
        'FLOKI': 'floki',
        'GRT': 'the-graph',
        'GMX': 'gmx',
        'AAVE': 'aave',
        'XLM': 'stellar',
        'CRV': 'curve-dao-token',
        'INJ': 'injective-protocol',
        'UNI': 'uniswap',
        'ATOM': 'cosmos',
        'ONDO': 'ondo-finance',
        'ALGO': 'algorand'
    }

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.base_url = "https://api.coingecko.com/api/v3"
        self.pro_base_url = "https://pro-api.coingecko.com/api/v3"

    def _get_coin_id(self, symbol: str) -> str:
        base_symbol = symbol.upper().split('/')[0]
        return self.COIN_ID_MAP.get(base_symbol, base_symbol.lower())

    @async_retry(attempts=3, delay=5, exceptions=(NetworkError, APIRateLimitError))
    async def _fetch_json(self, url, params=None, headers=None, limiter: Optional[RateLimiter] = None, endpoint_name: str = "default"):
        self._check_if_closed()
        if limiter:
            await limiter.wait_if_needed(endpoint_name)

        request_params = params.copy() if params else {}
        request_headers = headers.copy() if headers else {"accept": "application/json"}
        
        effective_url = url
        if self.api_key:
            if self.api_key.startswith('CG-'):
                effective_url = url.replace(self.pro_base_url, self.base_url)
                request_params['x_cg_demo_api_key'] = self.api_key
            else:
                effective_url = url.replace(self.base_url, self.pro_base_url)
                request_headers['x-cg-pro-api-key'] = self.api_key

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
        coin_id = self._get_coin_id(symbol)
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

    async def get_global_market_data(self) -> Optional[Dict]:
        return await self._fetch_json(f"{self.base_url}/global", limiter=coingecko_rate_limiter, endpoint_name="global")

    async def get_fundamental_data(self, coin_id_or_symbol: str) -> Optional[FundamentalAnalysis]:
        coin_id = self._get_coin_id(coin_id_or_symbol)
        if not coin_id:
            return None
        cache_key = f"cache:coingecko_fundamental:{coin_id}"
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
    
    async def get_derivatives(self) -> Optional[list]:
        try:
            url = f"{self.base_url}/derivatives"
            return await self._fetch_json(url, limiter=coingecko_rate_limiter, endpoint_name="derivatives")
        except Exception as e:
            logger.warning(f"Could not fetch derivatives: {e}")
            return None

    async def get_trending_searches(self) -> Optional[list]:
        url = f"{self.base_url}/search/trending"
        data = await self._fetch_json(url, limiter=coingecko_rate_limiter, endpoint_name="trending")
        if data and 'coins' in data:
            return [item['item']['symbol'] for item in data['coins']]
        return None

    async def get_exchanges(self) -> Optional[list]:
        cache_key = "cache:coingecko:exchanges"
        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)

        url = f"{self.base_url}/exchanges"
        data = await self._fetch_json(url, limiter=coingecko_rate_limiter, endpoint_name="exchanges")
        if data and self.redis:
            await self.redis.set(cache_key, json.dumps(data), ex=3600)
        return data

    async def get_coin_market_chart(self, coin_id: str, days: int = 30) -> Optional[Dict]:
        cache_key = f"cache:coingecko:market_chart:{coin_id}:{days}"
        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)

        url = f"{self.base_url}/coins/{coin_id}/market_chart"
        params = {'vs_currency': 'usd', 'days': str(days)}
        data = await self._fetch_json(url, params, limiter=coingecko_rate_limiter, endpoint_name=f"market_chart_{coin_id}")
        
        if data and self.redis:
            await self.redis.set(cache_key, json.dumps(data), ex=1800)
        return data
    
    async def get_coins_list(self) -> Optional[List[Dict]]:
        cache_key = "cache:coingecko:coins_list"
        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)

        url = f"{self.base_url}/coins/list"
        params = {'include_platform': 'true'}
        data = await self._fetch_json(url, params, limiter=coingecko_rate_limiter, endpoint_name="coins_list")
        
        if data and self.redis:
            await self.redis.set(cache_key, json.dumps(data), ex=86400)
        return data

    async def get_coin_tickers(self, coin_id: str) -> Optional[Dict]:
        cache_key = f"cache:coingecko:tickers:{coin_id}"
        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)

        url = f"{self.base_url}/coins/{coin_id}/tickers"
        data = await self._fetch_json(url, limiter=coingecko_rate_limiter, endpoint_name=f"tickers_{coin_id}")
        
        if data and self.redis:
            await self.redis.set(cache_key, json.dumps(data), ex=600)
        return data

    async def get_global_defi_data(self) -> Optional[Dict]:
        cache_key = "cache:coingecko:global_defi"
        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)

        url = f"{self.base_url}/global/decentralized_finance_defi"
        data = await self._fetch_json(url, limiter=coingecko_rate_limiter, endpoint_name="global_defi")
        
        if data and self.redis:
            await self.redis.set(cache_key, json.dumps(data), ex=3600)
        return data

    async def get_categories(self) -> Optional[List[Dict]]:
        cache_key = "cache:coingecko:categories"
        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)

        url = f"{self.base_url}/coins/categories"
        data = await self._fetch_json(url, limiter=coingecko_rate_limiter, endpoint_name="categories")
        
        if data and self.redis:
            await self.redis.set(cache_key, json.dumps(data), ex=3600)
        return data

    async def get_nfts_list(self) -> Optional[List[Dict]]:
        cache_key = "cache:coingecko:nfts_list"
        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)

        url = f"{self.base_url}/nfts/list"
        data = await self._fetch_json(url, limiter=coingecko_rate_limiter, endpoint_name="nfts_list")
        
        if data and self.redis:
            await self.redis.set(cache_key, json.dumps(data), ex=86400)
        return data

    async def get_exchange_rates(self) -> Optional[Dict]:
        cache_key = "cache:coingecko:exchange_rates"
        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)

        url = f"{self.base_url}/exchange_rates"
        data = await self._fetch_json(url, limiter=coingecko_rate_limiter, endpoint_name="exchange_rates")
        
        if data and self.redis:
            await self.redis.set(cache_key, json.dumps(data), ex=3600)
        return data

    async def get_coin_contract_info(self, platform: str, contract_address: str) -> Optional[Dict]:
        cache_key = f"cache:coingecko:contract:{platform}:{contract_address}"
        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)

        url = f"{self.base_url}/coins/{platform}/contract/{contract_address}"
        data = await self._fetch_json(url, limiter=coingecko_rate_limiter, endpoint_name=f"contract_{platform}")
        
        if data and self.redis:
            await self.redis.set(cache_key, json.dumps(data), ex=3600)
        return data

    async def get_companies_holdings(self, coin_id: str = "bitcoin") -> Optional[Dict]:
        cache_key = f"cache:coingecko:companies:{coin_id}"
        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)

        url = f"{self.base_url}/companies/public_treasury/{coin_id}"
        data = await self._fetch_json(url, limiter=coingecko_rate_limiter, endpoint_name=f"companies_{coin_id}")
        
        if data and self.redis:
            await self.redis.set(cache_key, json.dumps(data), ex=86400)
        return data

    async def get_coin_info(self, coin_id: str) -> Optional[Dict]:
        cache_key = f"cache:coingecko:coin_info:{coin_id}"
        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)
        
        url = f"{self.base_url}/coins/{coin_id}"
        params = {'localization': 'false', 'tickers': 'true', 'market_data': 'true', 'community_data': 'true', 'developer_data': 'true', 'sparkline': 'true'}
        data = await self._fetch_json(url, params, limiter=coingecko_rate_limiter, endpoint_name=f"coins_{coin_id}")
        
        if data and self.redis:
            await self.redis.set(cache_key, json.dumps(data), ex=3600)
        return data

    async def get_markets_data(self, vs_currency: str = "usd", order: str = "market_cap_desc", per_page: int = 250, page: int = 1) -> Optional[List[Dict]]:
        cache_key = f"cache:coingecko:markets:{vs_currency}:{page}"
        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)
        
        url = f"{self.base_url}/coins/markets"
        params = {'vs_currency': vs_currency, 'order': order, 'per_page': per_page, 'page': page, 'sparkline': 'true', 'price_change_percentage': '1h,24h,7d,30d,1y'}
        data = await self._fetch_json(url, params, limiter=coingecko_rate_limiter, endpoint_name="markets")
        
        if data and self.redis:
            await self.redis.set(cache_key, json.dumps(data), ex=300)
        return data

    async def get_historical_market_data(self, coin_id: str, date: str) -> Optional[Dict]:
        cache_key = f"cache:coingecko:historical:{coin_id}:{date}"
        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)
        
        url = f"{self.base_url}/coins/{coin_id}/history"
        params = {'date': date, 'localization': 'false'}
        data = await self._fetch_json(url, params, limiter=coingecko_rate_limiter, endpoint_name=f"history_{coin_id}")
        
        if data and self.redis:
            await self.redis.set(cache_key, json.dumps(data), ex=86400)
        return data

    async def get_coin_market_chart_range(self, coin_id: str, from_timestamp: int, to_timestamp: int) -> Optional[Dict]:
        url = f"{self.base_url}/coins/{coin_id}/market_chart/range"
        params = {'vs_currency': 'usd', 'from': from_timestamp, 'to': to_timestamp}
        return await self._fetch_json(url, params, limiter=coingecko_rate_limiter, endpoint_name=f"chart_range_{coin_id}")

    async def get_exchange_volume_chart(self, exchange_id: str, days: int = 30) -> Optional[Dict]:
        url = f"{self.base_url}/exchanges/{exchange_id}/volume_chart"
        params = {'days': days}
        return await self._fetch_json(url, params, limiter=coingecko_rate_limiter, endpoint_name=f"exchange_volume_{exchange_id}")

    async def get_asset_platforms(self) -> Optional[List[Dict]]:
        cache_key = "cache:coingecko:asset_platforms"
        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)
        
        url = f"{self.base_url}/asset_platforms"
        data = await self._fetch_json(url, limiter=coingecko_rate_limiter, endpoint_name="asset_platforms")
        
        if data and self.redis:
            await self.redis.set(cache_key, json.dumps(data), ex=86400)
        return data

    async def get_token_price_by_addresses(self, platform: str, contract_addresses: List[str]) -> Optional[Dict]:
        addresses_string = ",".join(contract_addresses)
        url = f"{self.base_url}/simple/token_price/{platform}"
        params = {'contract_addresses': addresses_string, 'vs_currencies': 'usd', 'include_24hr_change': 'true', 'include_24hr_vol': 'true', 'include_market_cap': 'true'}
        return await self._fetch_json(url, params, limiter=coingecko_rate_limiter, endpoint_name=f"token_price_{platform}")

    async def get_trending_contracts(self) -> Optional[List[Dict]]:
        cache_key = "cache:coingecko:trending_contracts"
        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)
        
        url = f"{self.base_url}/search/trending"
        data = await self._fetch_json(url, limiter=coingecko_rate_limiter, endpoint_name="trending_contracts")
        
        if data and self.redis:
            await self.redis.set(cache_key, json.dumps(data), ex=1800)
        return data

    async def get_top_gainers_losers(self, vs_currency: str = "usd") -> Optional[Dict]:
        cache_key = f"cache:coingecko:top_gainers_losers:{vs_currency}"
        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)
        
        url = f"{self.base_url}/coins/top_gainers_losers"
        params = {'vs_currency': vs_currency}
        data = await self._fetch_json(url, params, limiter=coingecko_rate_limiter, endpoint_name="gainers_losers")
        
        if data and self.redis:
            await self.redis.set(cache_key, json.dumps(data), ex=600)
        return data

    async def get_recently_added_coins(self) -> Optional[List[Dict]]:
        cache_key = "cache:coingecko:recently_added"
        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)
        
        url = f"{self.base_url}/coins/list/new"
        data = await self._fetch_json(url, limiter=coingecko_rate_limiter, endpoint_name="recently_added")
        
        if data and self.redis:
            await self.redis.set(cache_key, json.dumps(data), ex=3600)
        return data

    async def get_search_results(self, query: str) -> Optional[Dict]:
        url = f"{self.base_url}/search"
        params = {'query': query}
        return await self._fetch_json(url, params, limiter=coingecko_rate_limiter, endpoint_name="search")


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
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        macro_data = {}
        cpi_data, fed_rate_data, gdp_data, unemployment_data, yield_data, inflation_data = results
        
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

        if self.redis and macro_data:
            await self.redis.set(cache_key, json.dumps(macro_data), ex=86400)
        
        return macro_data


class YFinanceFetcher(BaseFetcher):
    @async_retry(attempts=3, delay=5)
    async def get_traditional_indices(self):
        cache_key = "cache:trad_indices_yf"
        cache_ttl = 600
        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)

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
            await self.redis.set(cache_key, json.dumps(indices), ex=cache_ttl)
        return indices


class CryptoPanicFetcher(BaseFetcher):
    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.base_url = "https://cryptopanic.com/api/v1"

    async def fetch_posts(self, currencies: Optional[List[str]] = None, kind: Optional[str] = None):
        url = f"{self.base_url}/posts/"
        params = {"auth_token": self.api_key, "public": "true"}
        if currencies:
            params["currencies"] = ",".join(currencies)
        if kind:
            params['kind'] = kind
        
        data = await self._fetch_json(url, params, limiter=cryptopanic_rate_limiter, endpoint_name="posts")
        return data.get("results", [])

    async def fetch_media(self) -> List[Dict]:
        cache_key = "cache:cryptopanic:media"
        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)
        
        url = f"{self.base_url}/media/"
        params = {"auth_token": self.api_key}
        
        data = await self._fetch_json(url, params, limiter=cryptopanic_rate_limiter, endpoint_name="media")
        media_list = data.get("results", [])
        if self.redis and media_list:
            await self.redis.set(cache_key, json.dumps(media_list), ex=86400)
        return media_list


class AlternativeMeFetcher(BaseFetcher):
    async def fetch_fear_greed(self, limit: int = 1):
        cache_key = f"cache:fear_greed:{limit}"
        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached:
                return json.loads(cached)

        url = f"https://api.alternative.me/fng/?limit={limit}"
        data = await self._fetch_json(url, limiter=alternative_me_rate_limiter, endpoint_name="fng")
        
        results = data.get("data", [])
        if results and self.redis:
            await self.redis.set(cache_key, json.dumps(results), ex=3600)
        return results


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


class NewsFetcher:
    def __init__(self, 
                cryptopanic_fetcher: CryptoPanicFetcher,
                alternativeme_fetcher: AlternativeMeFetcher,
                coindesk_fetcher: Optional[CoinDeskFetcher] = None, 
                messari_fetcher: Optional[MessariFetcher] = None, 
                redis_client: Optional[redis.Redis] = None):
        
        self.cryptopanic = cryptopanic_fetcher
        self.alternativeme = alternativeme_fetcher
        self.coindesk = coindesk_fetcher
        self.messari = messari_fetcher
        self.redis = redis_client
        self._is_closed = False

    def _check_if_closed(self):
        if self._is_closed:
            raise RuntimeError("NewsFetcher has been closed and cannot be used")

    async def close(self):
        if self._is_closed:
            return
        self._is_closed = True
        
        tasks = [self.cryptopanic.close(), self.alternativeme.close()]
        if self.coindesk:
            tasks.append(self.coindesk.close())
        if self.messari:
            tasks.append(self.messari.close())
            
        await asyncio.gather(*tasks, return_exceptions=True)

    async def fetch_fear_greed(self, limit: int = 1):
        self._check_if_closed()
        return await self.alternativeme.fetch_fear_greed(limit)

    async def fetch_news(self, currencies: List[str] = ["BTC", "ETH"], kind: Optional[str] = None):
        self._check_if_closed()
        key = "cache:news:" + ",".join(sorted(currencies)) + (f":{kind}" if kind else "")
        if self.redis:
            try:
                cached = await self.redis.get(key)
                if cached:
                    return json.loads(cached)
            except Exception as e:
                logger.warning(f"Redis get failed for {key}: {e}")

        tasks = [self.cryptopanic.fetch_posts(currencies, kind)]

        if self.coindesk and not kind:
            tasks.append(self.coindesk.get_news(currencies))

        if self.messari and not kind:
            for currency in currencies:
                tasks.append(self.messari.get_news(f"{currency}/USDT"))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        all_news = []
        for res in results:
            if isinstance(res, list):
                all_news.extend(res)
            elif isinstance(res, Exception):
                logger.warning(f"News fetch failed: {res}")
        
        unique_news = list({item.get('id') or item.get('title'): item for item in all_news}.values())

        if self.redis and unique_news:
            await self.redis.set(key, json.dumps(unique_news), ex=600)

        return unique_news

    def score_news(self, news_items: List[Dict[str, Any]]):
        score = 0
        if not news_items:
            return score
        for it in news_items:
            if 'votes' in it:
                score += int(it['votes'].get('positive', 0))
                score -= int(it['votes'].get('negative', 0))
            
            title = (it.get("title") or "").lower()
            if any(k in title for k in ["bull", "rally", "surge", "gain", "pump", "partnership", "adoption", "breakthrough"]):
                score += 1
            if any(k in title for k in ["crash", "dump", "fall", "drop", "hack", "scam", "exploit", "regulatory", "ban"]):
                score -= 1
        return score

