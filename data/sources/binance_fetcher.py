import asyncio
import json
from typing import Any, Dict, List, Optional

import aiohttp
import pandas as pd
import redis.asyncio as redis

from config.logger import logger
from utils.circuit_breaker import CircuitBreaker
from common.core import OrderBook
from common.exceptions import (
    APIRateLimitError,
    DataError,
    InvalidSymbolError,
    NetworkError,
)
from common.utils import RateLimiter, async_retry
from data.sources.base_fetcher import BaseFetcher
from common.cache import CacheKeyBuilder, CacheCategory


binance_rate_limiter = RateLimiter(max_requests=1200, time_window=60)


class BinanceFetcher(BaseFetcher):
    FUTURES_BLACKLIST = {"SHIBUSDT", "PEPEUSDT", "FLOKIUSDT", "ASTRUSDT", "ONDOUSDT"}

    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        session: Optional[aiohttp.ClientSession] = None,
        **kwargs,
    ):
        super().__init__(redis_client=redis_client, session=session, **kwargs)
        self.base_url = "https://api.binance.com/api/v3"
        self.fapi_url = "https://fapi.binance.com/fapi/v1"
        self.futures_data_url = "https://fapi.binance.com/futures/data"
        self.source_name = "binance"

    @async_retry(
        attempts=3,
        delay=5,
        exceptions=(NetworkError, APIRateLimitError),
        ignore_exceptions=(InvalidSymbolError,),
    )
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
                    logger.warning(
                        f"IP banned by Binance for {endpoint}. Sleeping for 120s."
                    )
                    await asyncio.sleep(120)
                    raise APIRateLimitError(f"IP banned, retrying for {endpoint}")
                if response.status >= 400:
                    try:
                        error_data = await response.json()
                        if error_data.get(
                            "code"
                        ) == -1121 or "Invalid symbol" in error_data.get("msg", ""):
                            raise InvalidSymbolError(
                                f"Invalid symbol for Binance: {params.get('symbol') if params else 'N/A'}"
                            )
                    except (aiohttp.ContentTypeError, json.JSONDecodeError):
                        pass  # Raise the original error below

                    response.raise_for_status()

                return await response.json()

        try:
            return await self.circuit_breaker.call(_do_fetch)
        except aiohttp.ClientResponseError as e:
            if e.status == 400:
                raise InvalidSymbolError(
                    f"Invalid symbol for Binance: {params.get('symbol') if params else 'N/A'}"
                )
            raise NetworkError(f"HTTP Error {e.status} for {full_url}") from e
        except Exception as e:
            if isinstance(e, (NetworkError, APIRateLimitError, InvalidSymbolError)):
                raise
            logger.error(f"Error in _fetch for {full_url}: {e}", exc_info=True)
            raise DataError(f"Failed to fetch data from {full_url}") from e

    async def get_historical_ohlc(
        self, symbol: str, timeframe: str, limit: int
    ) -> Optional[pd.DataFrame]:
        self._check_if_closed()
        try:
            binance_symbol = symbol.replace("/", "").upper()
            params = {"symbol": binance_symbol, "interval": timeframe, "limit": limit}
            data = await self._fetch(self.base_url, "klines", params)

            if not data:
                return None

            df = pd.DataFrame(
                data,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_asset_volume",
                    "number_of_trades",
                    "taker_buy_base_asset_volume",
                    "taker_buy_quote_asset_volume",
                    "ignore",
                ],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)

            numeric_cols = ["open", "high", "low", "close", "volume"]
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

            df = df[numeric_cols].dropna()

            return df
        except InvalidSymbolError:
            logger.warning(f"Invalid symbol for Binance OHLC: {symbol}")
            return None
        except Exception as e:
            logger.error(
                f"Error fetching historical OHLC from Binance for {symbol}: {e}"
            )
            raise DataError(f"Failed to fetch OHLC for {symbol}") from e

    async def get_open_interest(self, symbol: str) -> Optional[float]:
        self._check_if_closed()
        binance_symbol = symbol.replace("/", "").upper()
        if binance_symbol in self.FUTURES_BLACKLIST:
            return None

        cache_key = CacheKeyBuilder.derivatives_key(
            self.source_name, "open_interest", symbol
        )
        cache_ttl = self.config_manager.get_cache_ttl("derivatives")

        try:
            if self.redis:
                cached = await self.redis.get(cache_key)
                if cached:
                    return float(cached)
        except Exception as e:
            logger.warning(f"Redis GET failed for {cache_key}: {e}", exc_info=True)

        try:
            params = {"symbol": binance_symbol}
            data = await self._fetch(self.fapi_url, "openInterest", params)
            if data and "openInterest" in data:
                open_interest = float(data["openInterest"])
                try:
                    if self.redis:
                        await self.redis.set(cache_key, open_interest, ex=cache_ttl)
                except Exception as e:
                    logger.warning(
                        f"Redis SET failed for {cache_key}: {e}", exc_info=True
                    )
                return open_interest
            return None
        except InvalidSymbolError:
            return None
        except Exception as e:
            logger.error(f"Error fetching open interest from Binance for {symbol}: {e}")
            return None

    async def get_taker_long_short_ratio(self, symbol: str) -> Optional[float]:
        self._check_if_closed()
        binance_symbol = symbol.replace("/", "").upper()
        if binance_symbol in self.FUTURES_BLACKLIST:
            return None

        cache_key = CacheKeyBuilder.derivatives_key(
            self.source_name, "long_short_ratio", symbol
        )
        cache_ttl = self.config_manager.get_cache_ttl("derivatives")

        try:
            if self.redis:
                cached = await self.redis.get(cache_key)
                if cached:
                    return float(cached)
        except Exception as e:
            logger.warning(f"Redis GET failed for {cache_key}: {e}", exc_info=True)

        try:
            params = {"symbol": binance_symbol, "period": "5m", "limit": 1}
            data = await self._fetch(
                self.futures_data_url, "takerlongshortRatio", params
            )
            if data and isinstance(data, list) and data and data[0].get("buySellRatio"):
                ratio = float(data[0]["buySellRatio"])
                try:
                    if self.redis:
                        await self.redis.set(cache_key, ratio, ex=cache_ttl)
                except Exception as e:
                    logger.warning(
                        f"Redis SET failed for {cache_key}: {e}", exc_info=True
                    )
                return ratio
            return None
        except InvalidSymbolError:
            return None
        except Exception as e:
            logger.error(
                f"Error fetching taker long/short ratio from Binance for {symbol}: {e}"
            )
            return None

    async def get_top_trader_long_short_ratio_accounts(
        self, symbol: str
    ) -> Optional[float]:
        self._check_if_closed()
        binance_symbol = symbol.replace("/", "").upper()
        if binance_symbol in self.FUTURES_BLACKLIST:
            return None
        try:
            params = {"symbol": binance_symbol, "period": "5m", "limit": 1}
            data = await self._fetch(
                self.futures_data_url, "globalLongShortAccountRatio", params
            )
            if (
                data
                and isinstance(data, list)
                and data
                and data[0].get("longShortRatio")
            ):
                return float(data[0]["longShortRatio"])
            return None
        except InvalidSymbolError:
            return None
        except Exception as e:
            logger.error(
                f"Error fetching top trader acc ratio from Binance for {symbol}: {e}"
            )
            return None

    async def get_top_trader_long_short_ratio_positions(
        self, symbol: str
    ) -> Optional[float]:
        self._check_if_closed()
        binance_symbol = symbol.replace("/", "").upper()
        if binance_symbol in self.FUTURES_BLACKLIST:
            return None
        try:
            params = {"symbol": binance_symbol, "period": "5m", "limit": 1}
            data = await self._fetch(
                self.futures_data_url, "topLongShortPositionRatio", params
            )
            if (
                data
                and isinstance(data, list)
                and data
                and data[0].get("longShortRatio")
            ):
                return float(data[0]["longShortRatio"])
            return None
        except InvalidSymbolError:
            return None
        except Exception as e:
            logger.error(
                f"Error fetching top trader pos ratio from Binance for {symbol}: {e}"
            )
            return None

    async def get_liquidation_orders(self, symbol: str) -> List[Dict]:
        self._check_if_closed()
        logger.debug(
            f"Skipping get_liquidation_orders for {symbol} as the endpoint is likely deprecated."
        )
        return []

    async def get_order_book_depth(self, symbol: str) -> Optional[OrderBook]:
        self._check_if_closed()
        try:
            binance_symbol = symbol.replace("/", "").upper()
            params = {"symbol": binance_symbol, "limit": 100}
            data = await self._fetch(self.base_url, "depth", params)
            if not data:
                return None

            bids = [(float(p), float(q)) for p, q in data.get("bids", [])]
            asks = [(float(p), float(q)) for p, q in data.get("asks", [])]

            return OrderBook(
                bids=bids,
                asks=asks,
                bid_ask_spread=asks[0][0] - bids[0][0] if asks and bids else 0,
                total_bid_volume=sum(q for _, q in bids),
                total_ask_volume=sum(q for _, q in asks),
            )
        except InvalidSymbolError:
            return None
        except Exception as e:
            logger.error(
                f"Error fetching order book depth from Binance for {symbol}: {e}"
            )
            return None

    async def get_mark_price(self, symbol: str) -> Optional[float]:
        self._check_if_closed()
        binance_symbol = symbol.replace("/", "").upper()
        if binance_symbol in self.FUTURES_BLACKLIST:
            return None
        try:
            params = {"symbol": binance_symbol}
            data = await self._fetch(self.fapi_url, "premiumIndex", params)
            if data and "markPrice" in data:
                return float(data["markPrice"])
            return None
        except InvalidSymbolError:
            return None
        except Exception as e:
            logger.error(f"Error fetching mark price from Binance for {symbol}: {e}")
            return None

    async def get_24h_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        self._check_if_closed()
        try:
            binance_symbol = symbol.replace("/", "").upper()
            params = {"symbol": binance_symbol}
            data = await self._fetch(self.base_url, "ticker/24hr", params)
            if data:
                return {
                    "price_change": float(data.get("priceChange", 0)),
                    "price_change_percent": float(data.get("priceChangePercent", 0)),
                    "weighted_avg_price": float(data.get("weightedAvgPrice", 0)),
                    "prev_close_price": float(data.get("prevClosePrice", 0)),
                    "last_price": float(data.get("lastPrice", 0)),
                    "bid_price": float(data.get("bidPrice", 0)),
                    "ask_price": float(data.get("askPrice", 0)),
                    "open_price": float(data.get("openPrice", 0)),
                    "high_price": float(data.get("highPrice", 0)),
                    "low_price": float(data.get("lowPrice", 0)),
                    "volume": float(data.get("volume", 0)),
                    "quote_volume": float(data.get("quoteVolume", 0)),
                    "open_time": data.get("openTime"),
                    "close_time": data.get("closeTime"),
                    "first_id": data.get("firstId"),
                    "last_id": data.get("lastId"),
                    "count": data.get("count"),
                }
            return None
        except InvalidSymbolError:
            return None
        except Exception as e:
            logger.error(f"Error fetching 24h ticker from Binance for {symbol}: {e}")
            return None

    async def get_long_short_ratio_history(
        self, symbol: str, period: str = "5m", limit: int = 30
    ) -> List[Dict]:
        self._check_if_closed()
        binance_symbol = symbol.replace("/", "").upper()
        if binance_symbol in self.FUTURES_BLACKLIST:
            return []

        try:
            params = {"symbol": binance_symbol, "period": period, "limit": limit}
            data = await self._fetch(
                self.futures_data_url, "takerlongshortRatio", params
            )
            return data if isinstance(data, list) else []
        except InvalidSymbolError:
            return []
        except Exception as e:
            logger.error(
                f"Error fetching long/short ratio history from Binance for {symbol}: {e}"
            )
            return []

    async def get_open_interest_history(
        self, symbol: str, period: str = "5m", limit: int = 30
    ) -> List[Dict]:
        self._check_if_closed()
        binance_symbol = symbol.replace("/", "").upper()
        if binance_symbol in self.FUTURES_BLACKLIST:
            return []

        try:
            params = {"symbol": binance_symbol, "period": period, "limit": limit}
            data = await self._fetch(self.fapi_url, "openInterestHist", params)
            return data if isinstance(data, list) else []
        except InvalidSymbolError:
            return []
        except Exception as e:
            logger.error(
                f"Error fetching open interest history from Binance for {symbol}: {e}"
            )
            return []

    async def get_funding_rate(self, symbol: str) -> Optional[float]:
        binance_symbol = symbol.replace("/", "").upper()
        if binance_symbol in self.FUTURES_BLACKLIST:
            return None

        cache_key = CacheKeyBuilder.derivatives_key(
            self.source_name, "funding_rate", symbol
        )
        cache_ttl = self.config_manager.get_cache_ttl("derivatives")

        try:
            if self.redis:
                cached = await self.redis.get(cache_key)
                if cached:
                    return float(cached)
        except Exception as e:
            logger.warning(f"Redis GET failed for {cache_key}: {e}", exc_info=True)

        try:
            params = {"symbol": binance_symbol, "limit": 1}
            data = await self._fetch(self.fapi_url, "fundingRate", params)
            if data and isinstance(data, list) and data[0].get("fundingRate"):
                funding_rate = float(data[0]["fundingRate"])
                try:
                    if self.redis:
                        await self.redis.set(cache_key, funding_rate, ex=cache_ttl)
                except Exception as e:
                    logger.warning(
                        f"Redis SET failed for {cache_key}: {e}", exc_info=True
                    )
                return funding_rate
            return None
        except InvalidSymbolError:
            return None
        except Exception:
            # Fallback to premiumIndex if fundingRate fails
            try:
                params = {"symbol": binance_symbol}
                data = await self._fetch(self.fapi_url, "premiumIndex", params)
                if data and "lastFundingRate" in data:
                    funding_rate = float(data["lastFundingRate"])
                    if self.redis:
                        await self.redis.set(cache_key, funding_rate, ex=cache_ttl)
                    return funding_rate
                return None
            except Exception as e:
                logger.error(
                    f"Error fetching funding rate (fallback) from Binance for {symbol}: {e}"
                )
                return None

    async def get_kline_data(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 500,
    ) -> List:
        self._check_if_closed()
        try:
            binance_symbol = symbol.replace("/", "").upper()
            params = {"symbol": binance_symbol, "interval": interval, "limit": limit}
            if start_time:
                params["startTime"] = start_time
            if end_time:
                params["endTime"] = end_time
            data = await self._fetch(self.base_url, "klines", params)
            return data if data else []
        except InvalidSymbolError:
            return []
        except Exception as e:
            logger.error(f"Error fetching kline data from Binance for {symbol}: {e}")
            return []

    async def get_ticker_price(self, symbol: str) -> Optional[float]:
        self._check_if_closed()
        try:
            binance_symbol = symbol.replace("/", "").upper()
            params = {"symbol": binance_symbol}
            data = await self._fetch(self.base_url, "ticker/price", params)
            if data and "price" in data:
                return float(data["price"])
            return None
        except InvalidSymbolError:
            return None
        except Exception as e:
            logger.error(f"Error fetching ticker price from Binance for {symbol}: {e}")
            return None

    async def get_exchange_info(self, symbol: Optional[str] = None) -> Optional[Dict]:
        self._check_if_closed()
        cache_key = CacheKeyBuilder.generic_key(
            self.source_name, "exchange_info", symbol
        )
        cache_ttl = self.config_manager.get_cache_ttl("generic")
        try:
            if self.redis:
                cached = await self.redis.get(cache_key)
                if cached:
                    return json.loads(cached)
        except Exception as e:
            logger.warning(f"Redis GET failed for {cache_key}: {e}", exc_info=True)

        try:
            params = {}
            if symbol:
                params["symbol"] = symbol.replace("/", "").upper()
            data = await self._fetch(self.base_url, "exchangeInfo", params)
            if data and self.redis:
                await self.redis.set(cache_key, json.dumps(data), ex=cache_ttl)
            return data
        except InvalidSymbolError:
            return None
        except Exception as e:
            logger.error(f"Error fetching exchange info from Binance: {e}")
            return None

    async def get_agg_trades(self, symbol: str, limit: int = 500) -> List[Dict]:
        self._check_if_closed()
        try:
            binance_symbol = symbol.replace("/", "").upper()
            params = {"symbol": binance_symbol, "limit": limit}
            data = await self._fetch(self.base_url, "aggTrades", params)
            return data if data else []
        except InvalidSymbolError:
            return []
        except Exception as e:
            logger.error(
                f"Error fetching aggregate trades from Binance for {symbol}: {e}"
            )
            return []

    async def get_all_futures_symbols(self) -> List[Dict]:
        self._check_if_closed()
        cache_key = CacheKeyBuilder.generic_key(self.source_name, "futures_symbols")
        cache_ttl = self.config_manager.get_cache_ttl("generic")
        try:
            if self.redis:
                cached = await self.redis.get(cache_key)
                if cached:
                    return json.loads(cached)
        except Exception as e:
            logger.warning(f"Redis GET failed for {cache_key}: {e}", exc_info=True)

        try:
            data = await self._fetch(self.fapi_url, "exchangeInfo")
            if data and "symbols" in data:
                symbols = data["symbols"]
                if self.redis:
                    await self.redis.set(cache_key, json.dumps(symbols), ex=cache_ttl)
                return symbols
            return []
        except Exception as e:
            logger.error(f"Error fetching futures symbols: {e}")
            return []
