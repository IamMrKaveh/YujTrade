import asyncio
from typing import Dict, List, Any, Optional

import pandas as pd

from .core import (
    TradingSignal, SignalType, MarketAnalysis, DynamicLevels,
    FundamentalAnalysis, OnChainAnalysis, DerivativesAnalysis,
    OrderBook, MacroEconomicData, TrendingData, BinanceFuturesData,
    TrendStrength
)
from .market import MarketDataProvider
from .indicator_factory import IndicatorFactory, IndicatorResult
from .analyzers import MarketConditionAnalyzer, PatternAnalyzer
from .logger_config import logger
from .utils import calculate_risk_reward_ratio, calculate_dynamic_levels
from .constants import LONG_TERM_CONFIG
from .data_validator import DataQualityChecker


class AnalysisEngine:
    def __init__(self, market_data_provider: MarketDataProvider, config_manager):
        self.market_data_provider = market_data_provider
        self.config_manager = config_manager
        self.indicator_factory = IndicatorFactory()
        self.market_analyzer = MarketConditionAnalyzer()
        self.pattern_analyzer = PatternAnalyzer()
        self.data_quality_checker = DataQualityChecker()

    async def run_full_analysis(self, symbol: str, timeframe: str, data: pd.DataFrame) -> Optional[TradingSignal]:
        if data.empty or len(data) < 50:
            logger.warning(f"Insufficient data for {symbol} on {timeframe} to perform analysis.")
            return None

        is_valid, quality_msg = self.data_quality_checker.validate_data_quality(data, timeframe)
        if not is_valid:
            logger.warning(f"Data quality check failed for {symbol}-{timeframe}: {quality_msg}")
            return None

        if timeframe in LONG_TERM_CONFIG.get('focus_timeframes', ['1d', '1w', '1M']):
            min_persistence_bars = 30 if timeframe == '1M' else 40
            if not self.data_quality_checker.check_trend_persistence(data, min_bars=min_persistence_bars):
                logger.info(f"Trend not persistent enough for long-term signal on {symbol}-{timeframe}")
                return None

        if not self.data_quality_checker.check_sufficient_volume(data, min_ratio=0.8):
            logger.info(f"Insufficient volume for reliable signal on {symbol}-{timeframe}")
            return None

        has_gaps, gap_msg = self.data_quality_checker.detect_data_gaps(data, max_gap_tolerance=5)
        if not has_gaps:
            logger.warning(f"Data gaps detected for {symbol}-{timeframe}: {gap_msg}")

        self.indicator_weights = self.config_manager.get_indicator_weights(timeframe)
        
        tasks = {
            "technical": self.perform_technical_analysis(data, symbol),
            "market_condition": self.analyze_market_context(data),
            "external_data": self.gather_external_data(symbol)
        }

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        technical_results, market_context, external_data = results

        if isinstance(technical_results, Exception) or isinstance(market_context, Exception) or isinstance(external_data, Exception):
            logger.error(f"Error during analysis for {symbol}/{timeframe}")
            return None

        min_trend_strength_str = LONG_TERM_CONFIG.get('min_trend_strength', 'MODERATE')
        try:
            min_trend_strength = TrendStrength[min_trend_strength_str.upper()]
        except KeyError:
            min_trend_strength = TrendStrength.MODERATE

        if market_context.trend_strength == TrendStrength.WEAK:
            if timeframe in ['1d', '1w', '1M']:
                logger.info(f"Weak trend detected for long-term timeframe {symbol}-{timeframe}, skipping")
                return None

        combined_score, reasons = self.calculate_combined_score(
            technical_results, market_context, external_data
        )

        signal_type = self.determine_signal_type(combined_score)
        if signal_type == SignalType.HOLD:
            return None
            
        confidence_score = abs(combined_score)
        
        min_confidence = LONG_TERM_CONFIG['min_confidence_threshold'].get(
            timeframe,
            self.config_manager.get("min_confidence_score", 70)
        )
        
        if confidence_score < min_confidence:
            logger.info(
                f"Confidence {confidence_score:.2f} below threshold {min_confidence} "
                f"for {symbol}-{timeframe}"
            )
            return None

        signal = await self.create_trading_signal(
            symbol, timeframe, data, signal_type, confidence_score, reasons, 
            market_context, external_data
        )

        min_rr_ratio = LONG_TERM_CONFIG.get('min_risk_reward_ratio', 2.0)
        if signal.risk_reward_ratio < min_rr_ratio:
            logger.info(
                f"Risk/Reward {signal.risk_reward_ratio:.2f} below minimum {min_rr_ratio} "
                f"for {symbol}-{timeframe}"
            )
            return None

        return signal

    async def perform_technical_analysis(self, data: pd.DataFrame, symbol: str) -> Dict[str, IndicatorResult]:
        indicator_tasks = []
        all_indicators = self.indicator_factory.get_all_indicator_names()
        
        for name in all_indicators:
            indicator_instance = self.indicator_factory.create(name)
            if indicator_instance:
                indicator_tasks.append(asyncio.to_thread(indicator_instance.calculate, data))

        indicator_results: List[IndicatorResult] = await asyncio.gather(*indicator_tasks, return_exceptions=True)
        
        processed_results = {
            res.name: res for res in indicator_results if isinstance(res, IndicatorResult) and res.value is not pd.NA
        }

        patterns = self.pattern_analyzer.detect_patterns(data)
        if patterns:
            pattern_strength = 1.0 if any("bullish" in p for p in patterns) else -1.0 if any("bearish" in p for p in patterns) else 0.0
            processed_results["pattern"] = IndicatorResult("pattern", pattern_strength, 100.0, ", ".join(patterns))

        return processed_results

    async def analyze_market_context(self, data: pd.DataFrame) -> MarketAnalysis:
        return await asyncio.to_thread(self.market_analyzer.analyze_market_condition, data)

    async def gather_external_data(self, symbol: str) -> Dict[str, Any]:
        tasks = {
            "fundamental": self._get_fundamental_data(symbol),
            "on_chain": self._get_on_chain_data(symbol),
            "derivatives": self._get_derivatives_data(symbol),
            "order_book": self._get_order_book(symbol),
            "macro": self._get_macro_data(),
            "trending": self._get_trending_data(symbol),
            "news": self._get_news_sentiment(symbol)
        }
        
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        external_data = {}
        keys = list(tasks.keys())
        for i, res in enumerate(results):
            if not isinstance(res, Exception):
                external_data[keys[i]] = res
            else:
                logger.warning(f"Failed to fetch external data for {keys[i]} on {symbol}: {res}")
                external_data[keys[i]] = None
        
        return external_data

    async def _get_fundamental_data(self, symbol: str) -> Optional[FundamentalAnalysis]:
        if not hasattr(self.market_data_provider, 'market_indices_fetcher') or not self.market_data_provider.market_indices_fetcher:
            return None
        try:
            coin_id = symbol.split('/')[0].lower()
            coingecko_fetcher = self.market_data_provider.market_indices_fetcher.coingecko
            return await coingecko_fetcher.get_fundamental_data(coin_id)
        except Exception as e:
            logger.warning(f"Failed to get fundamental data for {symbol}: {e}")
            return None

    async def _get_on_chain_data(self, symbol: str) -> Optional[OnChainAnalysis]:
        return None

    async def _get_derivatives_data(self, symbol: str) -> Optional[DerivativesAnalysis]:
        if not hasattr(self.market_data_provider, 'binance_fetcher') or not self.market_data_provider.binance_fetcher:
            return None
        
        try:
            binance = self.market_data_provider.binance_fetcher
            tasks = {
                'oi': binance.get_open_interest(symbol),
                'fr': binance.get_funding_rate(symbol),
                'taker': binance.get_taker_long_short_ratio(symbol),
            }
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            oi, fr, taker = results
            
            return DerivativesAnalysis(
                open_interest=oi if not isinstance(oi, Exception) else None,
                funding_rate=fr if not isinstance(fr, Exception) else None,
                taker_long_short_ratio=taker if not isinstance(taker, Exception) else None,
                coingecko_derivatives=[],
                binance_futures_data=None
            )
        except Exception as e:
            logger.warning(f"Failed to get derivatives data for {symbol}: {e}")
            return None

    async def _get_order_book(self, symbol: str) -> Optional[OrderBook]:
        if not hasattr(self.market_data_provider, 'binance_fetcher') or not self.market_data_provider.binance_fetcher:
            return None
        try:
            return await self.market_data_provider.binance_fetcher.get_order_book_depth(symbol)
        except Exception as e:
            logger.warning(f"Failed to get order book for {symbol}: {e}")
            return None

    async def _get_macro_data(self) -> Optional[MacroEconomicData]:
        return None

    async def _get_trending_data(self, symbol: str) -> Optional[TrendingData]:
        return None

    async def _get_news_sentiment(self, symbol: str) -> Dict[str, float]:
        return {"score": 0}

    def calculate_combined_score(self, technical_results: Dict[str, IndicatorResult], market_context: MarketAnalysis, external_data: Dict[str, Any]) -> tuple[float, List[str]]:
        total_score = 0.0
        total_weight = 0.0
        reasons = []

        for name, result in technical_results.items():
            weight = self.indicator_weights.get(name.lower(), 0)
            if weight > 0:
                score_contribution = self.get_indicator_score(result) * weight
                total_score += score_contribution
                total_weight += weight
                reasons.append(f"{name}: {result.interpretation} (Score: {score_contribution:.2f})")

        market_score, market_reasons = self.score_market_context(market_context)
        market_weight = self.indicator_weights.get("market_context", 15)
        total_score += market_score * market_weight
        total_weight += market_weight
        reasons.extend(market_reasons)

        external_score, external_reasons = self.score_external_data(external_data)
        external_weight = self.indicator_weights.get("external_data", 20)
        total_score += external_score * external_weight
        total_weight += external_weight
        reasons.extend(external_reasons)
        
        return (total_score / total_weight) if total_weight > 0 else 0, reasons

    def get_indicator_score(self, result: IndicatorResult) -> float:
        bullish_keywords = ["bullish", "oversold", "above", "buy", "up", "positive", "accumulation", "buy", "uptrend"]
        bearish_keywords = ["bearish", "overbought", "below", "down", "negative", "distribution", "sell", "downtrend"]
        
        interpretation = result.interpretation.lower()
        
        if any(keyword in interpretation for keyword in bullish_keywords):
            return result.signal_strength / 100.0
        if any(keyword in interpretation for keyword in bearish_keywords):
            return -result.signal_strength / 100.0
            
        return 0.0

    def score_market_context(self, context: MarketAnalysis) -> tuple[float, List[str]]:
        score = 0.0
        reasons = []

        if context.trend.value == "bullish":
            score += 1.0
        elif context.trend.value == "bearish":
            score -= 1.0
        reasons.append(f"Trend: {context.trend.value}")

        if context.market_condition.value == "oversold":
            score += 0.5
        elif context.market_condition.value == "overbought":
            score -= 0.5
        reasons.append(f"Condition: {context.market_condition.value}")

        if context.volume_confirmation:
            score += 0.3 if context.trend.value == "bullish" else -0.3
        reasons.append(f"Volume Confirmed: {context.volume_confirmation}")

        return score / 1.8, reasons

    def score_external_data(self, data: Dict[str, Any]) -> tuple[float, List[str]]:
        score = 0.0
        reasons = []
        
        if data.get("news") and data["news"]["score"] != 0:
            news_score = data["news"]["score"]
            score += news_score / 100
            reasons.append(f"News Sentiment: {news_score:.2f}")

        if data.get("derivatives"):
            derivatives: DerivativesAnalysis = data["derivatives"]
            if derivatives.funding_rate is not None:
                if derivatives.funding_rate > 0.001: 
                    score -= 0.1 
                if derivatives.funding_rate < -0.001: 
                    score += 0.1
                reasons.append(f"Funding Rate: {derivatives.funding_rate:.4f}")
            if derivatives.taker_long_short_ratio is not None:
                if derivatives.taker_long_short_ratio > 1: 
                    score += 0.1
                if derivatives.taker_long_short_ratio < 1: 
                    score -= 0.1
                reasons.append(f"Taker L/S Ratio: {derivatives.taker_long_short_ratio:.2f}")

        return score, reasons

    def determine_signal_type(self, score: float) -> SignalType:
        if score > 0.3:
            return SignalType.BUY
        if score < -0.3:
            return SignalType.SELL
        return SignalType.HOLD

    async def create_trading_signal(
        self, symbol: str, timeframe: str, data: pd.DataFrame, signal_type: SignalType,
        confidence_score: float, reasons: List[str], market_context: MarketAnalysis, 
        external_data: Dict[str, Any]
    ) -> TradingSignal:
        current_price = data['close'].iloc[-1]
        
        levels = calculate_dynamic_levels(data, signal_type, market_context.volatility)
        
        risk_reward = calculate_risk_reward_ratio(
            entry=levels.primary_entry,
            stop_loss=levels.tight_stop,
            take_profit=levels.primary_exit,
            signal_type=signal_type
        )
        
        predicted_profit = abs(levels.primary_exit - levels.primary_entry)

        return TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            entry_price=levels.primary_entry,
            exit_price=levels.primary_exit,
            stop_loss=levels.tight_stop,
            timestamp=pd.to_datetime(data.index[-1]),
            timeframe=timeframe,
            confidence_score=confidence_score,
            reasons=reasons,
            risk_reward_ratio=risk_reward,
            predicted_profit=predicted_profit,
            volume_analysis={"volume_trend": market_context.volume_trend},
            market_context=market_context.__dict__,
            dynamic_levels=levels.__dict__,
            fundamental_analysis=external_data.get("fundamental"),
            on_chain_analysis=external_data.get("on_chain"),
            derivatives_analysis=external_data.get("derivatives"),
            order_book=external_data.get("order_book"),
            macro_data=external_data.get("macro"),
            trending_data=external_data.get("trending")
        )

