import asyncio
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np

from .core import (
    TradingSignal, SignalType, MarketAnalysis,
    FundamentalAnalysis, OnChainAnalysis, DerivativesAnalysis,
    OrderBook, MacroEconomicData, TrendingData, BinanceFuturesData,
    TrendStrength, TrendDirection
)
from .market import MarketDataProvider
from .indicators.indicator_factory import IndicatorFactory
from .indicators.base import IndicatorResult
from .analyzers import MarketConditionAnalyzer, PatternAnalyzer
from .logger_config import logger
from .utils import calculate_risk_reward_ratio, calculate_dynamic_levels
from .constants import LONG_TERM_CONFIG, INDICATOR_GROUPS
from .data_validator import DataQualityChecker
from .models import ModelManager


class AnalysisEngine:
    def __init__(self, market_data_provider: MarketDataProvider, config_manager, model_manager: ModelManager):
        self.market_data_provider = market_data_provider
        self.config_manager = config_manager
        self.model_manager = model_manager
        self.indicator_factory = IndicatorFactory()
        self.market_analyzer = MarketConditionAnalyzer()
        self.pattern_analyzer = PatternAnalyzer()
        self.data_quality_checker = DataQualityChecker()
        self.indicator_weights = {}

    async def run_full_analysis(self, symbol: str, timeframe: str, data: pd.DataFrame) -> Optional[TradingSignal]:
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
            "external_data": self.gather_external_data(symbol),
            "ml_predictions": self.get_ml_predictions(symbol, timeframe, data)
        }

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        technical_results, market_context, external_data, ml_predictions = results

        if isinstance(technical_results, Exception) or isinstance(market_context, Exception):
            logger.error(f"Error during core analysis for {symbol}/{timeframe}. Tech: {technical_results}, Market: {market_context}")
            return None
        if isinstance(external_data, Exception):
            logger.warning(f"Failed to gather all external data for {symbol}/{timeframe}: {external_data}")
            external_data = {}
        if isinstance(ml_predictions, Exception):
            logger.warning(f"Failed to get ML predictions for {symbol}/{timeframe}: {ml_predictions}")
            ml_predictions = {}

        min_trend_strength_str = LONG_TERM_CONFIG.get('min_trend_strength', 'MODERATE')
        min_trend_strength = TrendStrength[min_trend_strength_str.upper()]

        if market_context.trend_strength.value < min_trend_strength.value:
            if timeframe in ['1d', '1w', '1M']:
                logger.info(f"Weak trend detected for long-term timeframe {symbol}-{timeframe}, skipping")
                return None

        combined_score, reasons, confidence_penalty = self.calculate_combined_score(
            technical_results, market_context, external_data, ml_predictions
        )

        signal_type = self.determine_signal_type(combined_score, market_context)
        if signal_type == SignalType.HOLD:
            return None

        raw_confidence = abs(combined_score)
        final_confidence = raw_confidence * (1 - confidence_penalty)

        min_confidence = LONG_TERM_CONFIG['min_confidence_threshold'].get(
            timeframe,
            self.config_manager.get("min_confidence_score", 70)
        )

        if final_confidence < min_confidence:
            logger.info(
                f"Confidence {final_confidence:.2f} (raw: {raw_confidence:.2f}, penalty: {confidence_penalty:.2%}) "
                f"below threshold {min_confidence} for {symbol}-{timeframe}"
            )
            return None

        signal = await self.create_trading_signal(
            symbol, timeframe, data, signal_type, final_confidence, reasons,
            market_context, external_data
        )

        if signal.risk_reward_ratio is not None:
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
        all_indicator_names = self.indicator_factory.get_all_indicator_names()

        for name in all_indicator_names:
            indicator_instance = self.indicator_factory.create(name)
            if indicator_instance:
                indicator_tasks.append(asyncio.to_thread(indicator_instance.calculate, data))

        indicator_results: List[IndicatorResult] = await asyncio.gather(*indicator_tasks, return_exceptions=True)

        processed_results = {}
        for i, res in enumerate(indicator_results):
            name = all_indicator_names[i]
            if isinstance(res, IndicatorResult) and res.value is not None and not pd.isna(res.value):
                processed_results[res.name] = res
            elif isinstance(res, Exception):
                logger.warning(f"Indicator '{name}' failed for {symbol} with error: {res}")
            else:
                logger.debug(f"Indicator '{name}' for {symbol} returned no valid result.")


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

    async def get_ml_predictions(self, symbol: str, timeframe: str, data: pd.DataFrame) -> Dict[str, float]:
        if not self.model_manager:
            return {}
        
        tasks = {
            "lstm": self.model_manager.predict("lstm", symbol, timeframe, data),
            "xgboost": self.model_manager.predict("xgboost", symbol, timeframe, data)
        }
        
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        predictions = {}
        
        for model_name, pred_result in zip(tasks.keys(), results):
            if isinstance(pred_result, Exception):
                logger.warning(f"ML prediction for {model_name} failed on {symbol}-{timeframe}: {pred_result}")
                continue
            if pred_result is not None and len(pred_result) > 0:
                predictions[model_name] = float(pred_result[0])
                logger.info(f"ML prediction from {model_name} for {symbol}-{timeframe}: {predictions[model_name]}")
        
        return predictions

    async def _get_fundamental_data(self, symbol: str) -> Optional[FundamentalAnalysis]:
        coingecko_fetcher = self.market_data_provider.market_indices_fetcher.coingecko
        if not coingecko_fetcher:
            return None
        try:
            return await coingecko_fetcher.get_fundamental_data(symbol)
        except Exception as e:
            logger.warning(f"Failed to get fundamental data for {symbol}: {e}")
            return None

    async def _get_on_chain_data(self, symbol: str) -> Optional[OnChainAnalysis]:
        return None

    async def _get_derivatives_data(self, symbol: str) -> Optional[DerivativesAnalysis]:
        binance_fetcher = self.market_data_provider.binance_fetcher
        if not binance_fetcher:
            return None

        try:
            tasks = {
                'oi': binance_fetcher.get_open_interest(symbol),
                'fr': binance_fetcher.get_funding_rate(symbol),
                'taker': binance_fetcher.get_taker_long_short_ratio(symbol),
                'top_trader_acc': binance_fetcher.get_top_trader_long_short_ratio_accounts(symbol),
                'top_trader_pos': binance_fetcher.get_top_trader_long_short_ratio_positions(symbol),
            }
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            oi, fr, taker, tta, ttp = results

            binance_futures = BinanceFuturesData(
                top_trader_long_short_ratio_accounts=tta if not isinstance(tta, Exception) else None,
                top_trader_long_short_ratio_positions=ttp if not isinstance(ttp, Exception) else None,
                liquidation_orders=[],
                mark_price=None
            )

            return DerivativesAnalysis(
                open_interest=oi if not isinstance(oi, Exception) else None,
                funding_rate=fr if not isinstance(fr, Exception) else None,
                taker_long_short_ratio=taker if not isinstance(taker, Exception) else None,
                coingecko_derivatives=[],
                binance_futures_data=binance_futures
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
        market_indices = self.market_data_provider.market_indices_fetcher
        if not market_indices:
            return None

        try:
            indices = await market_indices.get_all_indices()
            return MacroEconomicData(
                cpi=indices.get('CPI'),
                fed_rate=indices.get('FED_RATE'),
                treasury_yield_10y=indices.get('TREASURY_YIELD_10Y'),
                gdp=indices.get('GDP'),
                unemployment=indices.get('UNEMPLOYMENT')
            )
        except Exception as e:
            logger.warning(f"Failed to get macro data: {e}")
            return None

    async def _get_trending_data(self, symbol: str) -> Optional[TrendingData]:
        coingecko_fetcher = self.market_data_provider.market_indices_fetcher.coingecko
        if not coingecko_fetcher:
            return None

        try:
            trending = await coingecko_fetcher.get_trending_searches()
            return TrendingData(coingecko_trending=trending if trending else [])
        except Exception as e:
            logger.warning(f"Failed to get trending data: {e}")
            return None

    async def _get_news_sentiment(self, symbol: str) -> Dict[str, float]:
        return {"score": 0}

    def calculate_combined_score(self, technical_results: Dict[str, IndicatorResult], market_context: MarketAnalysis, external_data: Dict[str, Any], ml_predictions: Dict[str, float]) -> tuple[float, List[str], float]:
        total_score = 0.0
        total_weight = 0.0
        reasons = []

        group_scores: Dict[str, List[float]] = {group: [] for group in INDICATOR_GROUPS.keys()}
        group_weights: Dict[str, float] = {group: 0.0 for group in INDICATOR_GROUPS.keys()}
        
        for name, result in technical_results.items():
            weight = self.indicator_weights.get(name.lower(), 0)
            if weight > 0:
                score_contribution = self.get_indicator_score(result)
                found_group = False
                for group, indicators in INDICATOR_GROUPS.items():
                    if name.lower().startswith(tuple(indicators)):
                        group_scores[group].append(score_contribution * weight)
                        group_weights[group] += weight
                        found_group = True
                        break
                if not found_group:
                    total_score += score_contribution * weight
                    total_weight += weight
                reasons.append(f"{name}: {result.interpretation} (Score: {score_contribution * weight:.2f})")

        for group, scores in group_scores.items():
            if scores:
                group_total_weight = group_weights[group]
                max_weight = self.config_manager.get("max_group_weights", {}).get(group, 25)
                capped_weight = min(group_total_weight, max_weight)
                
                if capped_weight > 0:
                    weighted_sum_of_scores = sum(scores)
                    avg_score = weighted_sum_of_scores / group_total_weight if group_total_weight > 0 else 0
                    total_score += avg_score * capped_weight
                    total_weight += capped_weight

        market_score, market_reasons = self.score_market_context(market_context)
        market_weight = self.indicator_weights.get("market_context", 15)
        if market_score != 0:
            total_score += market_score * market_weight
            total_weight += market_weight
            reasons.extend(market_reasons)

        external_score, external_reasons, missing_sources, total_sources = self.score_external_data(external_data, market_context)
        if external_score != 0:
            external_weight = self.indicator_weights.get("external_data", 20)
            total_score += external_score * external_weight
            total_weight += external_weight
            reasons.extend(external_reasons)

        ml_score, ml_reasons = self.score_ml_predictions(ml_predictions, market_context.trend)
        if ml_score != 0:
            lstm_weight = self.indicator_weights.get("lstm", 15)
            xgboost_weight = self.indicator_weights.get("xgboost", 10)
            ml_weight = lstm_weight + xgboost_weight
            total_score += ml_score * ml_weight
            total_weight += ml_weight
            reasons.extend(ml_reasons)
        
        confidence_penalty = (missing_sources / total_sources) * 0.5 if total_sources > 0 else 0.0

        final_score = (total_score / total_weight) * 100 if total_weight > 0 else 0
        return np.clip(final_score, -100, 100), reasons, confidence_penalty

    def score_ml_predictions(self, predictions: Dict[str, float], current_trend: TrendDirection) -> tuple[float, List[str]]:
        score = 0.0
        reasons = []
        if not predictions:
            return score, reasons

        for model_name, pred_price in predictions.items():
            if pred_price > 0:
                direction = 1 if pred_price > 0 else -1
                score += direction
                reasons.append(f"ML Model ({model_name}): Predicts price {'increase' if direction > 0 else 'decrease'}")

        return np.clip(score, -1.0, 1.0), reasons

    def get_indicator_score(self, result: IndicatorResult) -> float:
        strength = np.clip(result.signal_strength / 100.0, 0, 1) if result.signal_strength is not None else 0.5
        
        bullish_keywords = ["bullish", "oversold", "above", "buy", "up", "positive", "accumulation", "uptrend", "support"]
        bearish_keywords = ["bearish", "overbought", "below", "sell", "down", "negative", "distribution", "downtrend", "resistance"]

        interpretation = result.interpretation.lower()

        direction = 0
        is_bullish = any(keyword in interpretation for keyword in bullish_keywords)
        is_bearish = any(keyword in interpretation for keyword in bearish_keywords)

        if is_bullish and not is_bearish:
            direction = 1
        elif is_bearish and not is_bullish:
            direction = -1
        
        return direction * strength

    def score_market_context(self, context: MarketAnalysis) -> tuple[float, List[str]]:
        score = 0.0
        reasons = []

        trend_map = {TrendDirection.BULLISH: 1.0, TrendDirection.BEARISH: -1.0, TrendDirection.SIDEWAYS: 0.0}
        score += trend_map.get(context.trend, 0.0)
        reasons.append(f"Trend: {context.trend.value}")

        strength_map = {TrendStrength.STRONG: 1.0, TrendStrength.MODERATE: 0.5, TrendStrength.WEAK: 0.1}
        score *= strength_map.get(context.trend_strength, 0.5)

        if context.market_condition.value == "oversold":
            score += 0.5
        elif context.market_condition.value == "overbought":
            score -= 0.5
        reasons.append(f"Condition: {context.market_condition.value}")

        if context.volume_confirmation:
            score += 0.3 if context.trend == TrendDirection.BULLISH else -0.3
        reasons.append(f"Volume Confirmed: {context.volume_confirmation}")

        return np.clip(score / 1.8, -1.0, 1.0), reasons

    def score_external_data(self, data: Dict[str, Any], market_context: MarketAnalysis) -> tuple[float, List[str], int, int]:
        score = 0.0
        reasons = []
        missing_sources = 0
        total_sources = len(data)

        for source, value in data.items():
            if value is None:
                missing_sources += 1

        if data.get("news") and data["news"].get("score") is not None and data["news"]["score"] != 0:
            news_score = data["news"]["score"]
            score += np.clip(news_score / 50.0, -1.0, 1.0)
            reasons.append(f"News Sentiment: {news_score:.2f}")

        if data.get("derivatives"):
            derivatives: DerivativesAnalysis = data["derivatives"]
            if derivatives and derivatives.funding_rate is not None:
                is_trending = market_context.trend != TrendDirection.SIDEWAYS
                oi_factor = np.log1p(derivatives.open_interest or 0) / np.log1p(1e9) if derivatives.open_interest else 0.5
                
                fr_impact = 0.0
                if is_trending:
                    if derivatives.funding_rate > 0.0001: fr_impact = 0.1
                    if derivatives.funding_rate < 0: fr_impact = -0.1
                else:
                    if derivatives.funding_rate > 0.001: fr_impact = -0.15
                    if derivatives.funding_rate < -0.001: fr_impact = 0.15
                
                score += fr_impact * oi_factor
                reasons.append(f"Funding Rate: {derivatives.funding_rate:.4f} (Impact: {fr_impact * oi_factor:.2f})")

            if derivatives and derivatives.taker_long_short_ratio is not None:
                if derivatives.taker_long_short_ratio > 1: score += 0.1
                if derivatives.taker_long_short_ratio < 1: score -= 0.1
                reasons.append(f"Taker L/S Ratio: {derivatives.taker_long_short_ratio:.2f}")

        return np.clip(score, -1.0, 1.0), reasons, missing_sources, total_sources

    def determine_signal_type(self, score: float, market_context: MarketAnalysis) -> SignalType:
        volatility = market_context.volatility or 0.0
        hurst = market_context.hurst_exponent or 0.5

        if hurst > 0.55:
            base_threshold = 0.3
        else:
            base_threshold = 0.5

        volatility_factor = 1.0 + np.clip((volatility - 2.0) / 8.0, -0.5, 1.5)
        adaptive_threshold = base_threshold * volatility_factor * 100

        if score > adaptive_threshold:
            return SignalType.BUY
        if score < -adaptive_threshold:
            return SignalType.SELL
        return SignalType.HOLD

    async def create_trading_signal(
        self, symbol: str, timeframe: str, data: pd.DataFrame, signal_type: SignalType,
        confidence_score: float, reasons: List[str], market_context: MarketAnalysis,
        external_data: Dict[str, Any]
    ) -> TradingSignal:

        levels = calculate_dynamic_levels(data, signal_type, market_context.volatility, market_context)

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
            fundamental_analysis=external_data.get("fundamental") if external_data else None,
            on_chain_analysis=external_data.get("on_chain") if external_data else None,
            derivatives_analysis=external_data.get("derivatives") if external_data else None,
            order_book=external_data.get("order_book") if external_data else None,
            macro_data=external_data.get("macro") if external_data else None,
            trending_data=external_data.get("trending") if external_data else None
        )