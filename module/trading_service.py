import asyncio
from typing import List, Optional

from .analysis_engine import AnalysisEngine
from .config import ConfigManager
from .core import TradingSignal
from .market import MarketDataProvider
from .logger_config import logger
from .constants import LONG_TERM_CONFIG
from .signals import SignalGenerator, MultiTimeframeAnalyzer
from .data_sources import NewsFetcher, CryptoPanicFetcher, AlternativeMeFetcher
from .indicators import get_all_indicators
from .models import ModelManager


class TradingService:
    def __init__(self, market_data_provider: MarketDataProvider, config_manager: ConfigManager):
        self.market_data_provider = market_data_provider
        self.config_manager = config_manager
        
        indicators = get_all_indicators()
        
        redis_client = getattr(market_data_provider, 'redis', None)
        
        cryptopanic_fetcher = None
        alternativeme_fetcher = None
        
        try:
            from .config import Config
            if Config.CRYPTOPANIC_KEY:
                cryptopanic_fetcher = CryptoPanicFetcher(
                    api_key=Config.CRYPTOPANIC_KEY,
                    redis_client=redis_client
                )
            alternativeme_fetcher = AlternativeMeFetcher(redis_client=redis_client)
        except Exception as e:
            logger.warning(f"Failed to initialize news fetchers: {e}")
        
        news_fetcher = None
        if cryptopanic_fetcher and alternativeme_fetcher:
            news_fetcher = NewsFetcher(
                cryptopanic_fetcher=cryptopanic_fetcher,
                alternativeme_fetcher=alternativeme_fetcher,
                coindesk_fetcher=getattr(market_data_provider, 'coindesk_fetcher', None),
                messari_fetcher=None,
                redis_client=redis_client
            )
        
        multi_tf_analyzer = MultiTimeframeAnalyzer(
            market_data_provider=market_data_provider,
            indicators=indicators,
            redis_client=redis_client,
            cache_ttl=600
        )
        
        self.model_manager = ModelManager(redis_client=redis_client)

        self.signal_generator = SignalGenerator(
            market_data_provider=market_data_provider,
            news_fetcher=news_fetcher,
            market_indices_fetcher=getattr(market_data_provider, 'market_indices_fetcher', None),
            model_manager=self.model_manager,
            multi_tf_analyzer=multi_tf_analyzer,
            config=config_manager.config,
            binance_fetcher=getattr(market_data_provider, 'binance_fetcher', None),
            messari_fetcher=None
        )
        
        self.analysis_engine = AnalysisEngine(market_data_provider, config_manager)

    async def analyze_symbol(self, symbol: str, timeframe: str) -> Optional[TradingSignal]:
        logger.info(f"Analyzing {symbol} on {timeframe} timeframe...")
        try:
            limit_map = {
                "1h": 800,
                "4h": 600,
                "1d": 400,
                "1w": 250,
                "1M": 150
            }
            limit = limit_map.get(timeframe, 500)
            
            ohlcv_data = await self.market_data_provider.fetch_ohlcv_data(symbol, timeframe, limit=limit)
            if ohlcv_data is None or ohlcv_data.empty:
                logger.warning(f"No OHLCV data for {symbol} on {timeframe}.")
                return None

            signals = await self.signal_generator.generate_signals(ohlcv_data, symbol, timeframe)
            
            if signals and len(signals) > 0:
                signal = signals[0]
                logger.info(f"Generated signal for {symbol} on {timeframe}: {signal.signal_type.value} with confidence {signal.confidence_score:.2f}")
                return signal
            
            return None
        except Exception as e:
            logger.error(f"Error analyzing {symbol} on {timeframe}: {e}", exc_info=True)
            return None

    async def run_analysis_for_all_symbols(self) -> List[TradingSignal]:
        symbols = self.config_manager.get("symbols", [])
        focus_timeframes = LONG_TERM_CONFIG.get('focus_timeframes', ['1d', '1w', '1M'])
        all_timeframes = self.config_manager.get("timeframes", [])
        
        timeframes_to_use = [tf for tf in all_timeframes if tf in focus_timeframes] or all_timeframes

        tasks = [self.analyze_symbol(s, tf) for s in symbols for tf in timeframes_to_use]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        signals = []
        for res in results:
            if isinstance(res, TradingSignal):
                signals.append(res)
            elif isinstance(res, Exception):
                logger.error(f"Signal generation failed: {res}")
        
        def signal_priority_score(signal: TradingSignal) -> float:
            tf_weight = LONG_TERM_CONFIG['timeframe_priority_weights'].get(signal.timeframe, 0.5)
            rr_normalized = min(signal.risk_reward_ratio / 5.0, 1.0)
            confidence_normalized = signal.confidence_score / 100.0
            
            trend_alignment_bonus = 0.0
            if signal.market_context:
                mc = signal.market_context
                trend_val = mc.get('trend')
                is_aligned = (signal.signal_type == SignalType.BUY and trend_val == 'bullish') or \
                            (signal.signal_type == SignalType.SELL and trend_val == 'bearish')
                if is_aligned:
                    trend_alignment_bonus = 0.15
            
            priority_score = (
                confidence_normalized * 0.35 +
                rr_normalized * 0.30 +
                tf_weight * 0.25 +
                trend_alignment_bonus * 0.10
            )
            
            return priority_score * 1000
        
        signals.sort(key=signal_priority_score, reverse=True)
        
        max_signals = LONG_TERM_CONFIG.get('max_signals_per_run', 3)
        final_signals = signals[:max_signals]
        
        if signals:
            logger.info(
                f"Selected {len(final_signals)} top quality long-term signals from {len(signals)} candidates"
            )
            for i, sig in enumerate(final_signals, 1):
                logger.info(
                    f"  #{i}: {sig.symbol} {sig.timeframe} {sig.signal_type.value.upper()} "
                    f"(Confidence: {sig.confidence_score:.1f}%, R/R: {sig.risk_reward_ratio:.2f})"
                )
        else:
            logger.info("No signals generated in this analysis cycle")
        
        return final_signals

    async def cleanup(self):
        logger.info("Cleaning up TradingService resources.")
        
        cleanup_tasks = []
        if hasattr(self.signal_generator, 'news_fetcher') and self.signal_generator.news_fetcher:
            cleanup_tasks.append(self.signal_generator.news_fetcher.close())
        
        if hasattr(self, 'model_manager') and self.model_manager:
            cleanup_tasks.append(self.model_manager.shutdown())

        cleanup_tasks.append(self.market_data_provider.close())
        
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)

