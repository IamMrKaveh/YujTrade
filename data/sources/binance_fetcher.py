import asyncio
import json
from typing import Any, Dict, List, Optional

import aiohttp
import pandas as pd
import redis

from config.logger import logger
from utils.circuit_breaker import CircuitBreaker
from common.core import OrderBook
from common.exceptions import APIRateLimitError, DataError, InvalidSymbolError, NetworkError
from module.utils import RateLimiter, async_retry


binance_rate_limiter = RateLimiter(max_requests=1200, time_window=60)


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

    @async_retry(attempts=3, delay=5, exceptions=(NetworkError, APIRateLimitError), ignore_exceptions=(InvalidSymbolError,))
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
            raise
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
        except InvalidSymbolError:
            raise
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
        except InvalidSymbolError:
            raise
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
        except InvalidSymbolError:
            raise
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
        except InvalidSymbolError:
            raise
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
        except InvalidSymbolError:
            raise
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
        except InvalidSymbolError:
            raise
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
        except InvalidSymbolError:
            raise
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
        except InvalidSymbolError:
            raise
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
        except InvalidSymbolError:
            raise
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
        except InvalidSymbolError:
            raise
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
        except InvalidSymbolError:
            raise
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
        except InvalidSymbolError:
            raise
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
        except InvalidSymbolError:
            raise
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
        except InvalidSymbolError:
            raise
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
        except InvalidSymbolError:
            raise
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
        except InvalidSymbolError:
            raise
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
        except InvalidSymbolError:
            raise
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
        except InvalidSymbolError:
            raise
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
        except InvalidSymbolError:
            raise
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
        except InvalidSymbolError:
            raise
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
        except InvalidSymbolError:
            raise
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
        except InvalidSymbolError:
            raise
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
        except InvalidSymbolError:
            raise
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
        except InvalidSymbolError:
            raise
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
        except InvalidSymbolError:
            raise
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
        except InvalidSymbolError:
            raise
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
        except InvalidSymbolError:
            raise
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

    async def get_hash_rate(self, symbol: str = "BTC/USDT") -> Optional[Dict]:
        self._check_if_closed()
        binance_symbol = symbol.replace('/', '').upper()
        
        try:
            ticker_data = await self.get_24h_ticker(binance_symbol)
            if not ticker_data:
                return None
                
            return {
                'symbol': symbol,
                'volume_24h': ticker_data.get('volume', 0),
                'quote_volume_24h': ticker_data.get('quote_volume', 0),
                'trades_count': ticker_data.get('count', 0)
            }
        except InvalidSymbolError:
            raise
        except Exception as e:
            logger.error(f"Error calculating network activity metrics for {symbol}: {e}")
            return None

    async def get_circulating_supply_estimate(self, symbol: str) -> Optional[float]:
        self._check_if_closed()
        
        try:
            exchange_info = await self.get_exchange_info(symbol)
            if not exchange_info or 'symbols' not in exchange_info:
                return None
                
            for sym_info in exchange_info['symbols']:
                if sym_info['symbol'] == symbol.replace('/', '').upper():
                    base_asset = sym_info.get('baseAsset')
                    return None
                    
            return None
        except InvalidSymbolError:
            raise
        except Exception as e:
            logger.error(f"Error fetching circulating supply for {symbol}: {e}")
            return None
