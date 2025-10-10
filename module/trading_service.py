import asyncio
from typing import List, Optional, Set

from .analysis_engine import AnalysisEngine
from .config import ConfigManager
from .core import TradingSignal
from .market import MarketDataProvider
from .logger_config import logger
from .constants import LONG_TERM_CONFIG
from .signals import SignalGenerator, MultiTimeframeAnalyzer, SignalRanking
from .data.news import NewsFetcher
from .data.cryptopanic import CryptoPanicFetcher
from .data.alternativeme import AlternativeMeFetcher
from .indicators.indicators import get_all_indicators
from .models import ModelManager
from .exceptions import InvalidSymbolError


class TradingService:
    def __init__(self, market_data_provider: MarketDataProvider, config_manager: ConfigManager):
        self.market_data_provider = market_data_provider
        self.config_manager = config_manager
        self.invalid_symbols: Set[str] = set()

        redis_client = getattr(market_data_provider, 'redis', None)

        self.model_manager = ModelManager(redis_client=redis_client)
        
        self.analysis_engine = AnalysisEngine(
            market_data_provider=market_data_provider,
            config_manager=config_manager,
            model_manager=self.model_manager
        )

        indicators = get_all_indicators()
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
        )
        
        self.signal_generator = SignalGenerator(
            market_data_provider=market_data_provider,
            news_fetcher=news_fetcher,
            market_indices_fetcher=getattr(market_data_provider, 'market_indices_fetcher', None),
            model_manager=self.model_manager,
            multi_tf_analyzer=multi_tf_analyzer,
            config=config_manager.config,
            binance_fetcher=getattr(market_data_provider, 'binance_fetcher', None),
            messari_fetcher=None,
            analysis_engine=self.analysis_engine
        )


    async def analyze_symbol(self, symbol: str, timeframe: str) -> Optional[TradingSignal]:
        if symbol in self.invalid_symbols:
            logger.debug(f"Skipping analysis for invalid symbol: {symbol}")
            return None
            
        logger.info(f"Analyzing {symbol} on {timeframe} timeframe...")
        try:
            limit = LONG_TERM_CONFIG['min_data_points'].get(timeframe, 600)

            ohlcv_data = await self.market_data_provider.fetch_ohlcv_data(symbol, timeframe, limit=limit)
            
            signal = await self.analysis_engine.run_full_analysis(symbol, timeframe, ohlcv_data)
            if signal:
                logger.info(f"Generated signal for {symbol} on {timeframe}: {signal.signal_type.value} with confidence {signal.confidence_score:.2f}")
                return signal

            return None
        except InvalidSymbolError:
            logger.warning(f"Symbol {symbol} is invalid. Adding to ignore list.")
            self.invalid_symbols.add(symbol)
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
            elif isinstance(res, Exception) and not isinstance(res, InvalidSymbolError):
                logger.error(f"Signal generation failed: {res}")

        ranked_signals = SignalRanking.rank_signals(signals)

        max_signals = LONG_TERM_CONFIG.get('max_signals_per_run', 3)
        final_signals = []
        
        if not ranked_signals:
            logger.info("No signals generated in this analysis cycle")
            return []

        best_score = SignalRanking.calculate_signal_score(ranked_signals[0])
        quality_threshold = max(best_score * 0.8, self.config_manager.get("min_absolute_signal_score", 65.0))

        for signal in ranked_signals:
            if len(final_signals) >= max_signals:
                break
            
            current_score = SignalRanking.calculate_signal_score(signal)
            if current_score >= quality_threshold:
                final_signals.append(signal)
            else:
                logger.debug(f"Signal for {signal.symbol}-{signal.timeframe} with score {current_score:.2f} did not meet quality threshold of {quality_threshold:.2f}")

        if final_signals:
            logger.info(
                f"Selected {len(final_signals)} top quality long-term signals from {len(signals)} candidates."
            )
            for i, sig in enumerate(final_signals, 1):
                logger.info(
                    f"  #{i}: {sig.symbol} {sig.timeframe} {sig.signal_type.value.upper()} "
                    f"(Confidence: {sig.confidence_score:.1f}%, R/R: {sig.risk_reward_ratio:.2f}, "
                    f"Trend: {sig.market_context.get('trend', 'N/A')})"
                )
        else:
            logger.info("No signals met the final quality criteria.")

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