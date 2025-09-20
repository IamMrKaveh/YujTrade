import asyncio
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import aiohttp
import aiosqlite
import ccxt.async_support as ccxt
import numpy as np
import pandas as pd
from web3 import Web3

from module.config import Config
from module.logger_config import logger
from module.utils import RateLimiter, async_retry

rate_limiter = RateLimiter(max_requests=10, time_window=60)

class SentimentFetcher:
    def __init__(self, cryptopanic_key: str):
        self.cryptopanic_key = cryptopanic_key
        self._fg_cache = None
        self._fg_ts = 0
        self._news_cache = {}
        
    async def fetch_fear_greed(self, max_retries=3, retry_delay=5):
        if time.time() - self._fg_ts < 3600 and self._fg_cache is not None:
            return self._fg_cache
        
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
                                    self._fg_cache = int(value)
                                    self._fg_ts = time.time()
                                    return self._fg_cache
                    
                    logger.warning(f"Attempt {attempt+1}: Invalid response from Fear & Greed API")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                except aiohttp.ClientError as e:
                    logger.warning(f"Attempt {attempt+1}: Error connecting to Fear & Greed API: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                    continue
        
        logger.error("Failed to fetch Fear & Greed index after multiple attempts")
        return None
    
    async def fetch_news(self, currencies: List[str] = ["BTC","ETH"]):
        key = tuple(currencies)
        
        if key in self._news_cache:
            return self._news_cache[key]
        
        try:                
            url = "https://cryptopanic.com/api/developer/v2/posts/"
            params = {"auth_token": self.cryptopanic_key, "currencies": ",".join(currencies)}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = data.get("results", [])
                        self._news_cache[key] = results
                        return results
                    else:
                        logger.warning(f"CryptoPanic API returned status {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []
    
    def score_news(self, news_items: List[Dict[str,Any]]):
        score = 0
        for it in news_items:
            title = (it.get("title") or "").lower()
            if any(k in title for k in ["bull", "rally", "surge", "gain", "pump"]):
                score += 1
            if any(k in title for k in ["crash", "dump", "fall", "drop", "hack"]):
                score -= 1
        return score

class OnChainFetcher:
    def __init__(self, alchemy_url: str = "", glassnode_api_key: str = "", 
                coinmetrics_api_key: str = "", dune_api_key: str = ""):
        self.alchemy_url = alchemy_url
        self.glassnode_api_key = glassnode_api_key
        self.coinmetrics_api_key = coinmetrics_api_key
        self.dune_api_key = dune_api_key
        
        self.web3 = None
        self.glassnode_base_url = "https://api.glassnode.com/v1/metrics"
        self.coinmetrics_base_url = "https://api.coinmetrics.io/v4"
        self.dune_base_url = "https://api.dune.com/api/v1"
        
        self.last_glassnode_call = 0
        self.glassnode_rate_limit = 1
        
        if alchemy_url and alchemy_url.strip():
            self._connect()

    def _connect(self):
        try:
            if not self.alchemy_url or not self.alchemy_url.strip():
                logger.warning("No Alchemy URL provided, skipping Web3 connection")
                return

            self.web3 = Web3(Web3.HTTPProvider(self.alchemy_url))
            if not self.web3.is_connected():
                raise ConnectionError("Failed to connect to Web3 provider")
            logger.info("Successfully connected to Web3 provider")
        except Exception as e:
            logger.error(f"Error connecting to Web3 provider: {e}")
            self.web3 = None

    def _ensure_connected(self):
        if not self.web3 or not self.web3.is_connected():
            self._connect()
        if not self.web3 or not self.web3.is_connected():
            raise ConnectionError("Web3 provider not available")

    async def _rate_limit_glassnode(self):
        current_time = time.time()
        time_since_last_call = current_time - self.last_glassnode_call
        if time_since_last_call < self.glassnode_rate_limit:
            await asyncio.sleep(self.glassnode_rate_limit - time_since_last_call)
        self.last_glassnode_call = time.time()

    async def _glassnode_request(self, endpoint: str, params: Dict = {}) -> Optional[Dict]:
        if not self.glassnode_api_key:
            logger.warning("Glassnode API key not provided")
            return None
            
        await self._rate_limit_glassnode()
        
        url = f"{self.glassnode_base_url}/{endpoint}"
        
        default_params = {'a': 'BTC', 'api_key': self.glassnode_api_key}
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

    async def _coinmetrics_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        if not self.coinmetrics_api_key:
            logger.warning("CoinMetrics API key not provided")
            return None
            
        url = f"{self.coinmetrics_base_url}/{endpoint}"
        headers = {'X-CM-API-Key': self.coinmetrics_api_key}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params, timeout=30) as response:
                    response.raise_for_status()
                    return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"CoinMetrics API error: {e}")
            return None

    async def _dune_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        if not self.dune_api_key:
            logger.warning("Dune API key not provided")
            return None
            
        url = f"{self.dune_base_url}/{endpoint}"
        headers = {'X-Dune-API-Key': self.dune_api_key}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params, timeout=30) as response:
                    response.raise_for_status()
                    return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"Dune API error: {e}")
            return None

    async def active_addresses(self, lookback_blocks: int = 100, use_glassnode: bool = True) -> Optional[int]:
        if use_glassnode and self.glassnode_api_key:
            try:
                yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                data = await self._glassnode_request('addresses/active_count', 
                                            {'s': yesterday, 'u': yesterday})
                if data and len(data) > 0:
                    return int(data[0]['v'])
            except Exception as e:
                logger.warning(f"Failed to get active addresses from Glassnode: {e}")
        
        try:
            if not self.web3:
                logger.warning("Web3 not initialized")
                return None

            self._ensure_connected()
            block = self.web3.eth.block_number
            start = max(0, block - lookback_blocks)
            addresses = set()

            for i in range(start, min(block, start + 50)):
                try:
                    blk = self.web3.eth.get_block(i, full_transactions=True)
                    for tx in blk.transactions:
                        if tx.get('from'):
                            addresses.add(tx['from'])
                        if tx.get('to'):
                            addresses.add(tx['to'])
                except Exception as e:
                    logger.warning(f"Error processing block {i}: {e}")
                    continue

            return len(addresses)
        except Exception as e:
            logger.error(f"Error fetching active addresses: {e}")
            return None

    async def transaction_volume(self, lookback_blocks: int = 100, use_glassnode: bool = True) -> Optional[float]:
        if use_glassnode and self.glassnode_api_key:
            try:
                yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                data = await self._glassnode_request('transactions/transfers_volume_sum', 
                                            {'s': yesterday, 'u': yesterday})
                if data and len(data) > 0:
                    return float(data[0]['v'])
            except Exception as e:
                logger.warning(f"Failed to get transaction volume from Glassnode: {e}")
        
        try:
            if not self.web3:
                return None
            self._ensure_connected()
            block = self.web3.eth.block_number
            start = max(0, block - lookback_blocks)
            total = 0
            for i in range(start, min(block, start + 10)):
                try:
                    blk = self.web3.eth.get_block(i, full_transactions=True)
                    for tx in blk.transactions:
                        total += tx.get('value', 0)
                except Exception:
                    continue
            return total / 1e18
        except Exception:
            return None

    def exchange_flow(self, exchange_addresses: List[str], lookback_blocks: int = 100) -> tuple:
        try:
            if not self.web3:
                return 0, 0
            self._ensure_connected()
            block = self.web3.eth.block_number
            start = max(0, block - lookback_blocks)
            inflow = 0
            outflow = 0
            for i in range(start, block):
                try:
                    blk = self.web3.eth.get_block(i, full_transactions=True)
                    for tx in blk.transactions:
                        if tx['to'] in exchange_addresses:
                            outflow += tx.get('value', 0)
                        if tx['from'] in exchange_addresses:
                            inflow += tx.get('value', 0)
                except Exception:
                    continue
            return inflow / 1e18, outflow / 1e18
        except Exception:
            return 0, 0

    async def get_network_value_to_transactions(self, days: int = 30) -> Optional[List[Dict]]:
        if not self.glassnode_api_key:
            return None
            
        since = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        return await self._glassnode_request('indicators/nvt', {'s': since})

    async def get_mvrv_ratio(self, days: int = 30) -> Optional[List[Dict]]:
        if not self.glassnode_api_key:
            return None
            
        since = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        return await self._glassnode_request('indicators/mvrv', {'s': since})

    async def get_long_term_holder_supply(self, days: int = 30) -> Optional[List[Dict]]:
        if not self.glassnode_api_key:
            return None
            
        since = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        return await self._glassnode_request('supply/long_term_holder', {'s': since})

    async def get_exchange_balances(self, days: int = 30) -> Optional[List[Dict]]:
        if not self.glassnode_api_key:
            return None
            
        since = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        return await self._glassnode_request('distribution/balance_exchanges', {'s': since})

    async def get_realized_price(self, days: int = 30) -> Optional[List[Dict]]:
        if not self.glassnode_api_key:
            return None
            
        since = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        return await self._glassnode_request('market/price_realized_usd', {'s': since})

    async def get_puell_multiple(self, days: int = 30) -> Optional[List[Dict]]:
        if not self.glassnode_api_key:
            return None
            
        since = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        return await self._glassnode_request('indicators/puell_multiple', {'s': since})

    async def get_fear_and_greed_index(self, days: int = 30) -> Optional[List[Dict]]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('https://api.alternative.me/fng/', 
                                        params={'limit': days, 'format': 'json'}) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return data.get('data', [])
        except Exception as e:
            logger.error(f"Error fetching Fear and Greed Index: {e}")
            return None

    async def get_whale_transactions(self, threshold_usd: float = 1000000, days: int = 7) -> Optional[List[Dict]]:
        if not self.glassnode_api_key:
            return None
            
        since = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        return await self._glassnode_request('transactions/transfers_volume_large', 
                                        {'s': since, 't': threshold_usd})

    async def get_comprehensive_metrics(self, days: int = 30) -> Dict[str, Any]:
        metrics = {}
        
        tasks = []
        
        if self.glassnode_api_key:
            tasks.extend([
                self.get_network_value_to_transactions(days),
                self.get_mvrv_ratio(days),
                self.get_long_term_holder_supply(days),
                self.get_exchange_balances(days),
                self.get_realized_price(days),
                self.get_puell_multiple(days),
                self.get_whale_transactions(days=days)
            ])
        
        tasks.extend([
            self.active_addresses(use_glassnode=True),
            self.transaction_volume(use_glassnode=True),
            self.get_fear_and_greed_index(days)
        ])
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        result_idx = 0
        if self.glassnode_api_key:
            metrics['nvt_ratio'] = results[result_idx] if not isinstance(results[result_idx], Exception) else None
            result_idx += 1
            metrics['mvrv_ratio'] = results[result_idx] if not isinstance(results[result_idx], Exception) else None
            result_idx += 1
            metrics['lth_supply'] = results[result_idx] if not isinstance(results[result_idx], Exception) else None
            result_idx += 1
            metrics['exchange_balances'] = results[result_idx] if not isinstance(results[result_idx], Exception) else None
            result_idx += 1
            metrics['realized_price'] = results[result_idx] if not isinstance(results[result_idx], Exception) else None
            result_idx += 1
            metrics['puell_multiple'] = results[result_idx] if not isinstance(results[result_idx], Exception) else None
            result_idx += 1
            metrics['whale_transactions'] = results[result_idx] if not isinstance(results[result_idx], Exception) else None
            result_idx += 1
        
        metrics['active_addresses'] = results[result_idx] if not isinstance(results[result_idx], Exception) else None
        result_idx += 1
        metrics['transaction_volume'] = results[result_idx] if not isinstance(results[result_idx], Exception) else None
        result_idx += 1
        metrics['fear_greed_index'] = results[result_idx] if not isinstance(results[result_idx], Exception) else None
        
        return metrics

    def export_to_dataframe(self, metrics: Dict[str, Any]) -> pd.DataFrame:
        try:
            data = []
            for metric_name, metric_data in metrics.items():
                if isinstance(metric_data, list) and len(metric_data) > 0:
                    for item in metric_data:
                        if isinstance(item, dict) and 't' in item and 'v' in item:
                            data.append({
                                'timestamp': item['t'],
                                'metric': metric_name,
                                'value': item['v']
                            })
            
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Error creating DataFrame: {e}")
            return pd.DataFrame()

class ExchangeManager:
    def __init__(self):
        self.exchange = None
        self._lock = asyncio.Lock()
        self._db_lock = asyncio.Lock()
        self._conn = None
        self.db_path = 'trading_bot.db'
        self.ohlcv_cache = {}
        self._db_initialized = False
        self._closed = False
        
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    @asynccontextmanager
    async def _get_db_connection(self):
        async with self._db_lock:
            try:
                if not self._conn or self._closed:
                    self._conn = await aiosqlite.connect(self.db_path)
                yield self._conn
                await self._conn.commit()
            except Exception as e:
                if self._conn:
                    try:
                        await self._conn.rollback()
                    except:
                        pass
                raise
                
    async def close(self):
        if self._closed:
            return
            
        self._closed = True
        
        async with self._db_lock:
            if self._conn:
                try:
                    await self._conn.close()
                except Exception:
                    pass
                finally:
                    self._conn = None
        
        await self.close_exchange()
        
        self.ohlcv_cache.clear()

    async def init_database(self):
        if self._db_initialized or self._closed:
            return
            
        try:
            async with self._db_lock:
                if self._db_initialized:
                    return
                    
                conn = None
                try:
                    conn = await aiosqlite.connect(self.db_path)
                    await conn.execute("""
                        CREATE TABLE IF NOT EXISTS ohlcv (
                            symbol TEXT, 
                            timeframe TEXT, 
                            timestamp INTEGER, 
                            open REAL, 
                            high REAL, 
                            low REAL, 
                            close REAL, 
                            volume REAL, 
                            PRIMARY KEY(symbol, timeframe, timestamp)
                        )
                    """)
                    await conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_tf_ts ON ohlcv(symbol, timeframe, timestamp)")
                    await conn.commit()
                    self._db_initialized = True
                    logger.info("Database initialized successfully")
                except Exception as e:
                    logger.error(f"Database initialization error: {e}")
                    raise
                finally:
                    if conn:
                        try:
                            await conn.close()
                        except:
                            pass
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    @asynccontextmanager
    async def get_exchange(self):
        if self._closed:
            raise RuntimeError("ExchangeManager is closed")
            
        async with self._lock:
            try:
                if self.exchange is None:
                    self.exchange = ccxt.coinex({
                        'apiKey': Config.COINEX_API_KEY,
                        'secret': Config.COINEX_SECRET,
                        'sandbox': False,
                        'enableRateLimit': True,
                        'timeout': 30000,
                        'options': {'defaultType': 'spot'}
                    })
                yield self.exchange
            except Exception as e:
                logger.error(f"Exchange error: {e}")
                if self.exchange:
                    try:
                        await self.exchange.close()
                    except:
                        pass
                    self.exchange = None
                raise

    async def close_exchange(self):
        async with self._lock:
            if self.exchange:
                try:
                    await self.exchange.close()
                    logger.info("Exchange connection closed successfully")
                except Exception as e:
                    logger.warning(f"Error closing exchange: {e}")
                finally:
                    self.exchange = None

    async def _save_ohlcv_to_db(self, df, symbol, timeframe):
        if self._closed or df.empty:
            return
            
        try:
            async with self._get_db_connection() as db:
                rows = []
                for idx, row in df.iterrows():
                    try:
                        if hasattr(row['timestamp'], 'timestamp'):
                            ts = int(row['timestamp'].timestamp() * 1000)
                        elif pd.api.types.is_datetime64_any_dtype(row['timestamp']):
                            ts = int(row['timestamp'].timestamp() * 1000)
                        else:
                            ts = int(row['timestamp'])
                        
                        row_data = (symbol, timeframe, ts,
                                    float(row['open']), float(row['high']),
                                    float(row['low']), float(row['close']),
                                    float(row['volume']))
                        
                        if all(not np.isnan(x) and not np.isinf(x) for x in row_data[3:]):
                            rows.append(row_data)
                    except (ValueError, TypeError, KeyError):
                        continue
                        
                if rows:
                    await db.executemany(
                        "INSERT OR REPLACE INTO ohlcv VALUES (?,?,?,?,?,?,?,?)", rows
                    )
        except Exception as e:
            logger.error(f"Database save error: {e}")
    
    @async_retry(attempts=3, delay=5, exceptions=(ccxt.NetworkError, ccxt.RequestTimeout, asyncio.TimeoutError))
    async def fetch_ohlcv_data(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        if self._closed: return pd.DataFrame()
        if not symbol or not timeframe or limit <= 0: return pd.DataFrame()

        since = None
        db_data = pd.DataFrame()

        try:
            async with self._get_db_connection() as db:
                cursor = await db.execute(
                    "SELECT * FROM ohlcv WHERE symbol = ? AND timeframe = ? ORDER BY timestamp DESC",
                    (symbol, timeframe)
                )
                rows = await cursor.fetchall()
                if rows:
                    db_data = pd.DataFrame(rows, columns=['symbol', 'timeframe', 'timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    db_data['timestamp'] = pd.to_datetime(db_data['timestamp'], unit='ms', utc=True)
                    latest_timestamp = db_data['timestamp'].max()
                    since = int(latest_timestamp.timestamp() * 1000)
                    logger.debug(f"Found {len(db_data)} records for {symbol}/{timeframe} in DB. Fetching new data since {latest_timestamp}.")
        except Exception as e:
            logger.error(f"DB read error for {symbol}/{timeframe}: {e}")

        logger.debug(f"Fetching OHLCV data for {symbol} on {timeframe} (limit: {limit})")
        
        try:
            await rate_limiter.wait_if_needed(f"ohlcv_{symbol}")
            
            ohlcv = None
            async with self.get_exchange() as exchange:
                try:
                    ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit, since=since)
                except ccxt.ExchangeError as ee:
                    logger.error(f"Exchange error for {symbol}: {ee}")
                    return db_data

            if not ohlcv:
                logger.debug(f"No new OHLCV data received for {symbol} on {timeframe}. Using DB data.")
                return db_data.sort_values('timestamp').reset_index(drop=True)

            df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
            
            if df.empty:
                return db_data.sort_values('timestamp').reset_index(drop=True)
                
            df = df.dropna()
            if df.empty:
                logger.warning(f"All new data invalid after cleaning for {symbol}")
                return db_data.sort_values('timestamp').reset_index(drop=True)

            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                
                for col in numeric_columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df = df.dropna(subset=numeric_columns)
                
                if df.empty:
                    logger.warning(f"No valid numeric new data for {symbol}")
                    return db_data.sort_values('timestamp').reset_index(drop=True)

                invalid_mask = (
                    (df[numeric_columns] <= 0).any(axis=1) |
                    np.isinf(df[numeric_columns]).any(axis=1) |
                    (df['high'] < df['low']) |
                    (df['high'] < df['open']) |
                    (df['high'] < df['close']) |
                    (df['low'] > df['open']) |
                    (df['low'] > df['close'])
                )
                
                df = df[~invalid_mask]
                
                if df.empty:
                    logger.warning(f"No valid new data after validation for {symbol}")
                    return db_data.sort_values('timestamp').reset_index(drop=True)

                for col in numeric_columns:
                    df[col] = df[col].astype('float32')

                if not self._closed:
                    await self._save_ohlcv_to_db(df, symbol, timeframe)
                
                combined_df = pd.concat([db_data, df]).drop_duplicates(subset=['timestamp'], keep='last')
                
                return combined_df.sort_values('timestamp').reset_index(drop=True)

            except Exception as e:
                logger.error(f"Data processing error for {symbol}: {e}")
                return db_data.sort_values('timestamp').reset_index(drop=True)

        except Exception as e:
            logger.error(f"Unexpected error fetching {symbol} on {timeframe}: {e}")
            return db_data.sort_values('timestamp').reset_index(drop=True)