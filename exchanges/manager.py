import pandas as pd
import ccxt.async_support as ccxt
from datetime import datetime
import os
import asyncio
from contextlib import asynccontextmanager
from exchanges.constants import COINEX_API_KEY, COINEX_SECRET
from logger_config import logger


class ExchangeManager:
    def __init__(self):
        self.exchange = None
        self._lock = asyncio.Lock()
    
    @asynccontextmanager
    async def get_exchange(self):
        async with self._lock:
            if self.exchange is None:
                self.exchange = ccxt.coinex({
                'apiKey': os.getenv('COINEX_API_KEY', COINEX_API_KEY),
                'secret': os.getenv('COINEX_SECRET', COINEX_SECRET),
                'sandbox': False,
                'enableRateLimit': True,
                'timeout': 30000,
                'options': {'defaultType': 'spot'}
            })
            
            try:
                yield self.exchange
            except Exception as e:
                logger.error(f"Error accessing exchange: {e}")
                await self.close_exchange()
                raise e
    
    async def close_exchange(self):
        async with self._lock:
            if self.exchange:
                await self.exchange.close()
                self.exchange = None
    
    async def fetch_ohlcv_data(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        logger.info(f"🔍 Fetching OHLCV data for {symbol} on {timeframe} (limit: {limit})")
        
        try:
            async with self.get_exchange() as exchange:
                start_time = asyncio.get_event_loop().time()
                ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                fetch_time = asyncio.get_event_loop().time() - start_time
                
                if not ohlcv:
                    logger.warning(f"⚠️ No OHLCV data received for {symbol} on {timeframe}")
                    return pd.DataFrame()
                
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                logger.info(f"✅ Successfully fetched {len(df)} candles for {symbol} on {timeframe} in {fetch_time:.2f}s")
                return df
                
        except ccxt.NetworkError as e:
            logger.error(f"🌐 Network error fetching {symbol} on {timeframe}: {e}")
            return pd.DataFrame()
        except ccxt.ExchangeError as e:
            logger.error(f"🏪 Exchange error fetching {symbol} on {timeframe}: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"❌ Unexpected error fetching {symbol} on {timeframe}: {e}")
            return pd.DataFrame()