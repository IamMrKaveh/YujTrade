# analysis_engine.py

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta

import pandas as pd
import numpy as np
import talib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .core import (
    DynamicLevels, TradingSignal, SignalType, MarketAnalysis,
    FundamentalAnalysis, OnChainAnalysis, DerivativesAnalysis,
    OrderBook, MacroEconomicData, TrendingData, BinanceFuturesData,
    TrendStrength, TrendDirection
)
from .market import MarketDataProvider
from .indicators.indicator_factory import IndicatorFactory
from .indicators.base import IndicatorResult
from .analyzers import MarketConditionAnalyzer, PatternAnalyzer
from .logger_config import logger
from .utils import calculate_risk_reward_ratio, calculate_dynamic_levels, detect_market_regime, IndicatorNormalizer, calculate_fibonacci_levels, calculate_pivot_points
from .constants import LONG_TERM_CONFIG, INDICATOR_GROUPS
from .data_validator import DataQualityChecker
from .models import ModelManager


class TimestampedIndicatorResult:
    def __init__(self, result: IndicatorResult, timestamp: datetime):
        self.result = result
        self.timestamp = timestamp
        self.name = result.name
        self.value = result.value
        self.signal_strength = result.signal_strength
        self.interpretation = result.interpretation


class AdaptiveThresholdManager:
    def __init__(self):
        self.performance_history: Dict[str, List[Dict]] = {}
        self.min_samples = 50
        
    def record_performance(self, volatility_regime: str, hurst_range: str, 
                          threshold: float, signal_success: bool):
        key = f"{volatility_regime}_{hurst_range}"
        if key not in self.performance_history:
            self.performance_history[key] = []
        
        self.performance_history[key].append({
            'threshold': threshold,
            'success': signal_success,
            'timestamp': datetime.now(timezone.utc)
        })
        
        if len(self.performance_history[key]) > 100:
            self.performance_history[key] = self.performance_history[key][-100:]
    
    def get_optimal_threshold(self, volatility_regime: str, hurst_range: str, 
                             default_threshold: float) -> float:
        key = f"{volatility_regime}_{hurst_range}"
        
        if key not in self.performance_history or len(self.performance_history[key]) < self.min_samples:
            return default_threshold
        
        history = self.performance_history[key]
        threshold_performance = {}
        
        for record in history:
            threshold = round(record['threshold'], 2)
            if threshold not in threshold_performance:
                threshold_performance[threshold] = {'success': 0, 'total': 0}
            
            threshold_performance[threshold]['total'] += 1
            if record['success']:
                threshold_performance[threshold]['success'] += 1
        
        best_threshold = default_threshold
        best_accuracy = 0
        
        for threshold, perf in threshold_performance.items():
            if perf['total'] >= 10:
                accuracy = perf['success'] / perf['total']
                confidence_interval = 1.96 * np.sqrt((accuracy * (1 - accuracy)) / perf['total'])
                if accuracy - confidence_interval > 0.55:
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_threshold = threshold
        
        return best_threshold


class MLConfidenceCalibrator:
    def __init__(self):
        self.calibration_data: Dict[str, List[Tuple[float, bool, float]]] = {}
        self.calibration_bins = 10
        
    def add_prediction(self, model_name: str, confidence: float, actual_result: bool, timestamp: datetime = None):
        if model_name not in self.calibration_data:
            self.calibration_data[model_name] = []
        
        time_decay_factor = 1.0
        if timestamp:
            days_old = (datetime.now(timezone.utc) - timestamp).days
            time_decay_factor = np.exp(-days_old / 90)
        
        self.calibration_data[model_name].append((confidence, actual_result, time_decay_factor))
        
        if len(self.calibration_data[model_name]) > 500:
            self.calibration_data[model_name] = self.calibration_data[model_name][-500:]


class IndicatorCorrelationManager:
    def __init__(self):
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.correlation_threshold = 0.7

    def compute_correlations(self, processed_results: Dict[str, Any], data: pd.DataFrame):
        indicator_values = {
            name: item['result'].value 
            for name, item in processed_results.items() 
            if item.get('result') and isinstance(item['result'].value, (int, float)) and not pd.isna(item['result'].value)
        }
        
        if not indicator_values or len(indicator_values) < 2:
            self.correlation_matrix = None
            return

        temp_data = {}
        for name, item in processed_results.items():
            if item.get('result') and isinstance(item['result'].value, (int, float)):
                temp_data[name] = np.random.rand(10) * item['result'].value

        if len(temp_data) < 2:
            self.correlation_matrix = None
            return

        df = pd.DataFrame(temp_data)
        self.correlation_matrix = df.corr()

    def get_decorrelation_weights(self, processed_results: Dict[str, Any]) -> Dict[str, float]:
        if self.correlation_matrix is None or self.correlation_matrix.empty:
            return {name: 1.0 for name in processed_results.keys()}

        weights = {}
        correlated_groups = []
        
        indicators = list(self.correlation_matrix.columns)
        
        for i, indicator1 in enumerate(indicators):
            is_grouped = any(indicator1 in group for group in correlated_groups)
            if is_grouped:
                continue
            
            new_group = {indicator1}
            for j in range(i + 1, len(indicators)):
                indicator2 = indicators[j]
                if self.correlation_matrix.loc[indicator1, indicator2] > self.correlation_threshold:
                    new_group.add(indicator2)
            
            if len(new_group) > 1:
                correlated_groups.append(new_group)

        for group in correlated_groups:
            group_size = len(group)
            for indicator in group:
                weights[indicator] = 1.0 / group_size
        
        for indicator in processed_results.keys():
            if indicator not in weights:
                weights[indicator] = 1.0
        
        return weights


class AnalysisEngine:
    CRITICAL_SOURCES = {'ml', 'derivatives', 'market_condition'}
    OPTIONAL_SOURCES = {'fundamental', 'trending', 'news', 'macro', 'order_book', 'on_chain'}
    
    MAX_DATA_AGE_MINUTES = {
        '1h': 45,
        '4h': 60,
        '1d': 240,
        '1w': 1440,
        '1M': 10080
    }
    
    def __init__(self, market_data_provider: MarketDataProvider, config_manager, model_manager: ModelManager):
        self.market_data_provider = market_data_provider
        self.config_manager = config_manager
        self.model_manager = model_manager
        self.indicator_factory = IndicatorFactory()
        self.market_analyzer = MarketConditionAnalyzer()
        self.pattern_analyzer = PatternAnalyzer()
        self.data_quality_checker = DataQualityChecker()
        self.indicator_weights = {}
        self.indicator_normalizer = IndicatorNormalizer()
        
        self.threshold_manager = AdaptiveThresholdManager()
        self.ml_calibrator = MLConfidenceCalibrator()
        self.correlation_manager = IndicatorCorrelationManager()

    async def run_full_analysis(self, symbol: str, timeframe: str, data: pd.DataFrame) -> Optional[TradingSignal]:
        analysis_timestamp = datetime.now(timezone.utc)
        data_collection_timestamp = data.index[-1].to_pydatetime() if isinstance(data.index, pd.DatetimeIndex) else analysis_timestamp
        if data_collection_timestamp.tzinfo is None:
            data_collection_timestamp = data_collection_timestamp.replace(tzinfo=timezone.utc)

        try:
            is_valid, quality_msg = self.data_quality_checker.validate_data_quality(data, timeframe)
            if not is_valid:
                logger.warning(f"Data quality check failed for {symbol}-{timeframe}: {quality_msg}")
                return None
        except ValueError as e:
            logger.error(f"Data validation failed for {symbol}-{timeframe}: {e}")
            return None

        overall_quality_score = self.data_quality_checker.calculate_overall_quality_score(data, timeframe)
        if overall_quality_score < 0.2:
            logger.warning(f"Overall data quality too low ({overall_quality_score:.2f}) for {symbol}-{timeframe}")
            return None

        if timeframe in LONG_TERM_CONFIG.get('focus_timeframes', ['1d', '1w', '1M']):
            min_persistence_bars = 45 if timeframe == '1M' else 60
            persistence_ok, _ = self.data_quality_checker.check_trend_persistence(data, min_bars=min_persistence_bars)
            if not persistence_ok:
                logger.info(f"Trend not persistent enough for long-term signal on {symbol}-{timeframe}. Skipping analysis.")
                return None

        if not self.data_quality_checker.check_sufficient_volume(data, min_ratio=0.3):
            logger.info(f"Insufficient volume for reliable signal on {symbol}-{timeframe}")
            return None

        has_gaps, gap_msg = self.data_quality_checker.detect_data_gaps(data, max_gap_tolerance=5)
        if has_gaps:
            logger.warning(f"Data gaps detected for {symbol}-{timeframe}: {gap_msg}")
            raise ValueError(f"Data gap too large for {symbol}-{timeframe} - aborting signal generation")

        self.indicator_weights = self.config_manager.get_indicator_weights(timeframe)

        market_regime = detect_market_regime(data)
        adjusted_weights = self.adjust_weights_by_regime(self.indicator_weights, market_regime)

        tasks = {
            "technical": self.perform_technical_analysis(data, symbol, data_collection_timestamp),
            "market_condition": self.analyze_market_context(data, analysis_timestamp),
            "external_data": self.gather_external_data(symbol, analysis_timestamp),
            "ml_predictions": self.get_ml_predictions(symbol, timeframe, data, analysis_timestamp)
        }

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        technical_results, market_context_data, external_data, ml_predictions = results

        if isinstance(technical_results, Exception) or isinstance(market_context_data, Exception):
            logger.error(f"Error during core analysis for {symbol}/{timeframe}. Tech: {technical_results}, Market: {market_context_data}", exc_info=technical_results if isinstance(technical_results, Exception) else market_context_data)
            return None
            
        market_context = market_context_data.get('analysis') if isinstance(market_context_data, dict) else None
        if market_context is None:
            logger.error(f"Market context analysis failed for {symbol}/{timeframe}, cannot proceed.")
            return None
        market_context_timestamp = market_context_data.get('timestamp', analysis_timestamp) if isinstance(market_context_data, dict) else analysis_timestamp
            
        if isinstance(external_data, Exception):
            logger.warning(f"Failed to gather all external data for {symbol}/{timeframe}: {external_data}")
            external_data = {}
        if isinstance(ml_predictions, Exception):
            logger.warning(f"Failed to get ML predictions for {symbol}/{timeframe}: {ml_predictions}")
            ml_predictions = {}

        validated_external_data, external_timestamps = self._validate_and_sync_timestamps(
            external_data, analysis_timestamp, timeframe
        )
        
        all_timestamps = {
            'analysis': analysis_timestamp,
            'data_collection': data_collection_timestamp,
            'market_context': market_context_timestamp,
            'technical': data_collection_timestamp,
            **external_timestamps
        }
        
        if ml_predictions:
            all_timestamps['ml'] = analysis_timestamp

        is_synchronized, sync_msg = self._check_timestamp_synchronization(all_timestamps, timeframe)
        if not is_synchronized:
            logger.debug(f"Timestamp synchronization issue for {symbol}-{timeframe}: {sync_msg}")
            raise ValueError(f"Timestamp desync too high for {symbol}-{timeframe}: {sync_msg}")

        min_trend_strength_str = LONG_TERM_CONFIG.get('min_trend_strength', 'MODERATE')
        min_trend_strength = TrendStrength[min_trend_strength_str.upper()]

        if market_context.trend_strength.value < min_trend_strength.value and market_regime['market_type'] != 'ranging':
            if timeframe in ['1d', '1w', '1M']:
                logger.info(f"Weak trend detected for long-term timeframe {symbol}-{timeframe}, skipping")
                return None

        combined_score, reasons, confidence_penalty, source_weights, critical_sources_available = self.calculate_combined_score(
            technical_results, market_context, validated_external_data, ml_predictions, data, symbol, adjusted_weights
        )
        
        if not critical_sources_available:
            if timeframe in ["1d", "1w", "1M"]:
                logger.warning(f"Insufficient critical sources for long-term signal {symbol}-{timeframe}")
                return None
            logger.warning(f"Insufficient critical sources for {symbol}-{timeframe}")
            return None

        signal_type, threshold_info = self.determine_signal_type(combined_score, market_context, symbol, timeframe)
        if signal_type == SignalType.HOLD:
            return None

        raw_confidence = abs(combined_score)
        performance_factor = self._get_performance_factor(symbol, timeframe)
        
        if overall_quality_score < 0.5:
            raw_confidence *= 0.5
        elif overall_quality_score < 0.7:
            raw_confidence *= 0.7
            
        quality_adjusted_confidence = raw_confidence * performance_factor
        final_confidence = quality_adjusted_confidence * (1 - confidence_penalty)

        if final_confidence < 0.3:
            logger.info(f"Final confidence {final_confidence:.2f} below absolute minimum 0.3")
            return None

        base_min_confidence = LONG_TERM_CONFIG['min_confidence_threshold'].get(
            timeframe,
            self.config_manager.get("min_confidence_score", 70)
        )
        volatility_penalty = market_context.volatility / 10 if hasattr(market_context, 'volatility') and market_context.volatility else 0
        win_rate_penalty = 1 - (self._get_recent_win_rate(symbol, timeframe) / 100) if self._get_recent_win_rate(symbol, timeframe) else 0
        adaptive_min_confidence = base_min_confidence * (1 + volatility_penalty) * (1 + win_rate_penalty * 0.5)

        if final_confidence < adaptive_min_confidence:
            logger.info(
                f"Confidence {final_confidence:.2f} (raw: {raw_confidence:.2f}, penalty: {confidence_penalty:.2%}, quality: {overall_quality_score:.2f}, perf: {performance_factor:.2f}, adaptive_threshold: {adaptive_min_confidence:.2f}) "
                f"below threshold for {symbol}-{timeframe}"
            )
            return None

        btc_correlation_valid = await self._check_btc_correlation(symbol, signal_type, data)
        if not btc_correlation_valid:
            final_confidence *= 0.8
            reasons.append("Warning: Signal against BTC trend (confidence reduced)")

        levels = await asyncio.to_thread(self._calculate_dynamic_levels, data, signal_type, market_context, validated_external_data.get('order_book', {}).get('data'))
        
        rr_ratio = self._calculate_risk_reward_ratio(levels.primary_entry, levels.tight_stop, levels.primary_exit, signal_type)
        min_rr_ratio = LONG_TERM_CONFIG.get('min_risk_reward_ratio', 2.0)
        if rr_ratio < min_rr_ratio:
            logger.info(
                f"Risk/Reward {rr_ratio:.2f} below minimum {min_rr_ratio} "
                f"for {symbol}-{timeframe}"
            )
            return None

        signal = await self.create_trading_signal(
            symbol, timeframe, data, signal_type, final_confidence, reasons,
            market_context, validated_external_data, analysis_timestamp, data_collection_timestamp,
            levels, rr_ratio
        )

        max_age_minutes = self.MAX_DATA_AGE_MINUTES.get(timeframe, 15)
        data_age_minutes = (analysis_timestamp - data_collection_timestamp).total_seconds() / 60
        
        if data_age_minutes > max_age_minutes:
            logger.warning(f"Signal data too old ({data_age_minutes:.1f} minutes) for {symbol}-{timeframe}")
            return None

        self._validate_signal_consistency(signal, market_context, ml_predictions, technical_results, external_data)

        return signal

    async def perform_technical_analysis(self, data: pd.DataFrame, symbol: str, timestamp: datetime) -> Dict[str, Any]:
        indicator_tasks = []
        all_indicator_names = self.indicator_factory.get_all_indicator_names()

        indicator_groups = {}
        for name in all_indicator_names:
            found_group = False
            for group, indicators in INDICATOR_GROUPS.items():
                if name.lower().startswith(tuple(indicators)):
                    indicator_groups[name] = group
                    found_group = True
                    break
            if not found_group:
                indicator_groups[name] = 'other'

        for name in all_indicator_names:
            indicator_instance = self.indicator_factory.create(name)
            if indicator_instance:
                task = asyncio.to_thread(lambda inst=indicator_instance: inst.calculate(data))
                indicator_tasks.append(task)

        indicator_results: List[IndicatorResult] = await asyncio.gather(*indicator_tasks, return_exceptions=True)

        processed_results = {}
        for i, res in enumerate(indicator_results):
            name = all_indicator_names[i]
            if isinstance(res, IndicatorResult) and res.value is not None and not pd.isna(res.value):
                timestamped_result = TimestampedIndicatorResult(res, timestamp)
                res_dict = {'result': timestamped_result, 'group': indicator_groups.get(name, 'other')}
                processed_results[res.name] = res_dict
            elif isinstance(res, Exception):
                logger.debug(f"Indicator '{name}' failed for {symbol} with error: {res}")
            else:
                logger.debug(f"Indicator '{name}' for {symbol} returned no valid result.")

        patterns = self.pattern_analyzer.detect_patterns(data)
        if patterns:
            pattern_strength = 1.0 if any("bullish" in p for p in patterns) else -1.0 if any("bearish" in p for p in patterns) else 0.0
            pattern_result = IndicatorResult("pattern", pattern_strength, 100.0, ", ".join(patterns))
            timestamped_pattern = TimestampedIndicatorResult(pattern_result, timestamp)
            processed_results["pattern"] = {'result': timestamped_pattern, 'group': 'pattern'}

        if len(data) >= 14:
            # RSI calculation for divergence detection
            rsi_values = pd.Series(talib.RSI(data['close'], timeperiod=14), index=data.index)
            if not rsi_values.empty and not rsi_values.isna().all():
                for name in ['rsi', 'macd', 'stoch']:
                    if name in processed_results:
                        divergence = self.pattern_analyzer.detect_divergence(data, rsi_values, window=14)
                        if divergence:
                            div_strength = 0.8 if 'bullish' in divergence[0] else -0.8
                            div_result = IndicatorResult(f"{name}_divergence", div_strength, 80.0, ", ".join(divergence))
                            timestamped_div = TimestampedIndicatorResult(div_result, timestamp)
                            processed_results[f"{name}_divergence"] = {'result': timestamped_div, 'group': 'divergence'}

        return processed_results

    def _check_indicator_consistency(self, processed_results: Dict[str, Any], data: pd.DataFrame):
        rsi_result = processed_results.get('rsi', {}).get('result')
        macd_result = processed_results.get('macd', {}).get('result')
        sma_200_result = processed_results.get('sma_200', {}).get('result')
        
        current_price = data['close'].iloc[-1]
        
        if rsi_result and rsi_result.value > 70 and sma_200_result and current_price < sma_200_result.value:
            logger.warning("Inconsistency: RSI > 70 but price below SMA200 - potential overbought warning")
        
        if macd_result and macd_result.value > 0 and rsi_result and rsi_result.value < 30:
            logger.warning("Inconsistency: MACD bullish but RSI oversold - potential divergence")

    async def analyze_market_context(self, data: pd.DataFrame, timestamp: datetime) -> Dict[str, Any]:
        analysis = await asyncio.to_thread(self.market_analyzer.analyze_market_condition, data)
        return {
            'analysis': analysis,
            'timestamp': timestamp
        }

    async def gather_external_data(self, symbol: str, timestamp: datetime) -> Dict[str, Any]:
        tasks = {
            "fundamental": self._get_fundamental_data(symbol, timestamp),
            "on_chain": self._get_on_chain_data(symbol, timestamp),
            "derivatives": self._get_derivatives_data(symbol, timestamp),
            "order_book": self._get_order_book(symbol, timestamp),
            "macro": self._get_macro_data(timestamp),
            "trending": self._get_trending_data(symbol, timestamp),
            "news": self._get_news_sentiment(symbol, timestamp)
        }

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        external_data = {}
        keys = list(tasks.keys())
        for i, res in enumerate(results):
            if not isinstance(res, Exception):
                external_data[keys[i]] = {'data': res, 'timestamp': timestamp}
            else:
                logger.warning(f"Failed to fetch external data for {keys[i]} on {symbol}: {res}")
                external_data[keys[i]] = {'data': None, 'timestamp': timestamp}

        return external_data

    async def get_ml_predictions(self, symbol: str, timeframe: str, data: pd.DataFrame, timestamp: datetime) -> Dict[str, Dict[str, float]]:
        if not self.model_manager:
            return {}
        
        model_staleness_days = await self._check_model_staleness(symbol, timeframe)
        staleness_penalty = min(model_staleness_days / 7, 1.0)
        
        tasks = {
            "lstm": self.model_manager.predict_with_confidence("lstm", symbol, timeframe, data),
            "xgboost": self.model_manager.predict_with_confidence("xgboost", symbol, timeframe, data)
        }
        
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        predictions = {}
        
        for model_name, pred_result in zip(tasks.keys(), results):
            if isinstance(pred_result, Exception):
                logger.warning(f"ML prediction for {model_name} failed on {symbol}-{timeframe}: {pred_result}")
                continue
            if pred_result is not None and 'prediction' in pred_result and 'confidence' in pred_result:
                raw_confidence = pred_result.get('raw_confidence', pred_result['confidence'])
                calibrated_confidence = pred_result.get('confidence', raw_confidence)
                uncertainty = pred_result.get('uncertainty', 0.0)
                
                if uncertainty > 0.7:
                    logger.info(f"Ignoring {model_name} prediction due to high uncertainty: {uncertainty}")
                    continue
                
                final_confidence = calibrated_confidence * (1 - staleness_penalty) * (1 - uncertainty ** 2)
                
                pred_result['raw_confidence'] = raw_confidence
                pred_result['confidence'] = final_confidence
                pred_result['timestamp'] = timestamp
                predictions[model_name] = pred_result
                logger.info(f"ML prediction from {model_name} for {symbol}-{timeframe}: {pred_result['prediction']} "
                          f"(raw conf: {raw_confidence:.2f}, calibrated: {calibrated_confidence:.2f}, final: {final_confidence:.2f})")
        
        return predictions

    async def _check_model_staleness(self, symbol: str, timeframe: str) -> float:
        key = f"{symbol}-{timeframe}"
        if hasattr(self.model_manager, 'get_model'):
            model = await self.model_manager.get_model("lstm", symbol, timeframe)
            if model and hasattr(model, 'last_training_date') and model.last_training_date:
                days_since_training = (datetime.now(timezone.utc) - model.last_training_date).days
                return days_since_training
        return 0.0

    async def _get_fundamental_data(self, symbol: str, timestamp: datetime) -> Optional[FundamentalAnalysis]:
        coingecko_fetcher = self.market_data_provider.market_indices_fetcher.coingecko
        if not coingecko_fetcher:
            return None
        try:
            return await coingecko_fetcher.get_fundamental_data(symbol)
        except Exception as e:
            logger.warning(f"Failed to get fundamental data for {symbol}: {e}")
            return None

    async def _get_on_chain_data(self, symbol: str, timestamp: datetime) -> Optional[OnChainAnalysis]:
        return None

    async def _get_derivatives_data(self, symbol: str, timestamp: datetime) -> Optional[DerivativesAnalysis]:
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
                'fr_history': binance_fetcher.get_historical_funding_rate(symbol, limit=24)
            }
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            oi, fr, taker, tta, ttp, fr_history = results

            binance_futures = BinanceFuturesData(
                top_trader_long_short_ratio_accounts=tta if not isinstance(tta, Exception) else None,
                top_trader_long_short_ratio_positions=ttp if not isinstance(ttp, Exception) else None,
                liquidation_orders=[],
                mark_price=None
            )

            cumulative_funding = None
            if not isinstance(fr_history, Exception) and fr_history:
                try:
                    cumulative_funding = sum([float(item.get('fundingRate', 0)) for item in fr_history])
                except:
                    pass

            return DerivativesAnalysis(
                open_interest=oi if not isinstance(oi, Exception) else None,
                funding_rate=fr if not isinstance(fr, Exception) else None,
                taker_long_short_ratio=taker if not isinstance(taker, Exception) else None,
                coingecko_derivatives=[],
                binance_futures_data=binance_futures,
                cumulative_funding_rate=cumulative_funding
            )
        except Exception as e:
            logger.warning(f"Failed to get derivatives data for {symbol}: {e}")
            return None

    async def _get_order_book(self, symbol: str, timestamp: datetime) -> Optional[OrderBook]:
        if not hasattr(self.market_data_provider, 'binance_fetcher') or not self.market_data_provider.binance_fetcher:
            return None
        try:
            return await self.market_data_provider.binance_fetcher.get_order_book_depth(symbol)
        except Exception as e:
            logger.warning(f"Failed to get order book for {symbol}: {e}")
            return None

    async def _get_macro_data(self, timestamp: datetime) -> Optional[MacroEconomicData]:
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

    async def _get_trending_data(self, symbol: str, timestamp: datetime) -> Optional[TrendingData]:
        coingecko_fetcher = self.market_data_provider.market_indices_fetcher.coingecko
        if not coingecko_fetcher:
            return None

        try:
            trending = await coingecko_fetcher.get_trending_searches()
            return TrendingData(coingecko_trending=trending if trending else [])
        except Exception as e:
            logger.warning(f"Failed to get trending data: {e}")
            return None

    async def _get_news_sentiment(self, symbol: str, timestamp: datetime) -> Dict[str, float]:
        news_fetcher = getattr(self.market_data_provider, 'news_fetcher', None)
        if not news_fetcher:
            return {"score": 0}
        
        try:
            base_symbol = symbol.split('/')[0]
            sentiment = await news_fetcher.fetch_sentiment_analysis(currencies=[base_symbol])
            if sentiment and 'overall_score' in sentiment:
                details = sentiment.get('details', {})
                news_items = details.get('news_items', [])
                
                decayed_score = 0
                total_weight = 0
                current_time = timestamp
                half_life_hours = {'1h': 24, '4h': 48, '1d': 168, '1w': 720, '1M': 2160}.get(self.config_manager.get('current_timeframe', '1d'), 168)
                
                for item in news_items:
                    item_timestamp = item.get('timestamp')
                    if item_timestamp:
                        try:
                            item_time = datetime.fromisoformat(str(item_timestamp))
                            age_hours = (current_time - item_time).total_seconds() / 3600
                            
                            decay_factor = np.exp(-age_hours / (half_life_hours / 2))
                            
                            item_score = item.get('score', 0)
                            decayed_score += item_score * decay_factor
                            total_weight += decay_factor
                        except:
                            continue
                
                if total_weight > 0:
                    final_score = decayed_score / total_weight
                else:
                    final_score = sentiment['overall_score']
                
                return {"score": final_score, "details": sentiment}
            return {"score": 0}
        except Exception as e:
            logger.warning(f"Failed to get news sentiment for {symbol}: {e}")
            return {"score": 0}

    def _validate_and_sync_timestamps(self, external_data: Dict[str, Any], analysis_timestamp: datetime, timeframe: str) -> Tuple[Dict[str, Any], Dict[str, datetime]]:
        validated_data = {}
        timestamps = {}
        max_age_minutes = self.MAX_DATA_AGE_MINUTES.get(timeframe, 15)
        
        freshness_weights = {
            'critical': 1.0,
            'optional': 0.7
        }
        
        for source, value_dict in external_data.items():
            if not isinstance(value_dict, dict):
                validated_data[source] = {'data': value_dict, 'timestamp': analysis_timestamp}
                timestamps[source] = analysis_timestamp
                continue
            
            data_timestamp = value_dict.get('timestamp', analysis_timestamp)
            data_age = (analysis_timestamp - data_timestamp).total_seconds() / 60
            
            source_type = 'critical' if source in self.CRITICAL_SOURCES else 'optional'
            weight = freshness_weights[source_type]
            
            if data_age <= max_age_minutes:
                validated_data[source] = value_dict
                timestamps[source] = data_timestamp
            else:
                age_penalty = min(data_age / (max_age_minutes * 2), 1.0)
                adjusted_weight = weight * (1 - age_penalty)
                validated_data[source] = {'data': value_dict.get('data'), 'timestamp': data_timestamp, 'freshness_weight': adjusted_weight}
                timestamps[source] = data_timestamp
        
        return validated_data, timestamps

    def _check_timestamp_synchronization(self, all_timestamps: Dict[str, datetime], timeframe: str) -> Tuple[bool, str]:
        if not all_timestamps:
            return False, "No timestamps available"
        
        max_allowed_diff = timedelta(minutes=self.MAX_DATA_AGE_MINUTES.get(timeframe, 15))
        
        data_ts = all_timestamps.get('data_collection')
        analysis_ts = all_timestamps.get('analysis')

        if data_ts and analysis_ts:
            if abs(data_ts - analysis_ts) > max_allowed_diff:
                return False, f"Time difference between data ({data_ts}) and analysis ({analysis_ts}) is too high."
            if data_ts >= analysis_ts:
                return False, "Data timestamp is not before analysis timestamp (potential data leakage)"

        timestamps_list = list(all_timestamps.values())
        min_ts = min(timestamps_list)
        max_ts = max(timestamps_list)
        time_diff = max_ts - min_ts
        
        if time_diff > max_allowed_diff:
            return False, f"Time difference {time_diff.total_seconds()/60:.1f} minutes exceeds maximum {max_allowed_diff.total_seconds()/60:.1f} minutes"
        
        return True, "Timestamps synchronized"

    def _get_performance_factor(self, symbol: str, timeframe: str) -> float:
        key = f"{symbol}-{timeframe}"
        if hasattr(self, '_historical_performance') and key in self._historical_performance:
            perf = self._historical_performance[key]
            if perf['total'] >= 5:
                recent_success = perf.get('recent_success', [])
                if recent_success:
                    recent_win_rate = sum(recent_success) / len(recent_success)
                    return recent_win_rate * 1.2
                return (perf['success'] / perf['total']) * 1.2
        return 1.0

    def _get_recent_win_rate(self, symbol: str, timeframe: str) -> float:
        key = f"{symbol}-{timeframe}"
        perf = getattr(self, '_historical_performance', {}).get(key)
        if perf and perf.get('recent_success'):
            return sum(perf['recent_success']) / len(perf['recent_success']) * 100
        return 50.0

    def calculate_combined_score(self, technical_results: Dict[str, Any], market_context: MarketAnalysis, 
                                external_data: Dict[str, Any], ml_predictions: Dict[str, Dict[str, float]], 
                                data: pd.DataFrame, symbol: str, adjusted_weights: Dict[str, float]) -> Tuple[float, List[str], float, Dict[str, float], bool]:
        total_score = 0.0
        total_weight = 0.0
        reasons = []
        source_weights = {
            'technical': 0.0,
            'market': 0.0,
            'external': 0.0,
            'ml': 0.0
        }

        available_sources = set()
        if technical_results:
            available_sources.add('technical')
        if market_context:
            available_sources.add('market_condition')
        if ml_predictions:
            available_sources.add('ml')
        if external_data.get('derivatives', {}).get('data'):
            available_sources.add('derivatives')

        critical_sources_by_asset = {
            'BTC': {'ml', 'derivatives', 'market_condition', 'on_chain'},
            'ETH': {'ml', 'derivatives', 'market_condition', 'on_chain'},
            'default': {'ml', 'derivatives', 'market_condition'}
        }
        
        asset_key = symbol.split('/')[0]
        critical_sources = critical_sources_by_asset.get(asset_key, critical_sources_by_asset['default'])
        
        critical_missing = critical_sources - available_sources
        critical_available_count = len(critical_sources & available_sources)
        
        if critical_available_count < len(critical_sources) * 0.5:
            logger.warning(f"Only {critical_available_count}/{len(critical_sources)} critical sources available: {available_sources}")
            return 0.0, ["Insufficient critical data sources"], 1.0, source_weights, False

        self.correlation_manager.compute_correlations(technical_results, data)
        decorrelation_weights = self.correlation_manager.get_decorrelation_weights(technical_results)

        indicator_scores = {}
        for name, item in technical_results.items():
            result = item['result']
            normalized_score = self.indicator_normalizer.normalize_indicator(result.result, name)
            score_contribution = self.get_indicator_score(normalized_score, market_context)
            indicator_scores[name] = np.clip(score_contribution, -1.0, 1.0)
            reasons.append(f"{name}: {result.interpretation} (Score: {score_contribution:.2f})")

        tech_score = sum(indicator_scores.get(name, 0) * adjusted_weights.get(name.lower(), 0) * decorrelation_weights.get(name, 1.0)
                         for name in technical_results.keys())
        tech_weight = sum(adjusted_weights.get(name.lower(), 0) * decorrelation_weights.get(name, 1.0)
                          for name in technical_results.keys())

        if tech_weight > 0:
            total_score += (tech_score / tech_weight) * 100
            total_weight += tech_weight
            source_weights['technical'] = tech_weight

        market_score, market_reasons = self.score_market_context(market_context)
        market_score = np.clip(market_score, -1.0, 1.0)
        market_weight = adjusted_weights.get("market_context", 15) / 100.0
        if market_score != 0:
            total_score += market_score * market_weight * 100
            total_weight += market_weight
            source_weights['market'] = market_weight
            reasons.extend(market_reasons)

        external_score, external_reasons, source_availability = self.score_external_data(external_data, market_context, symbol)
        external_score = np.clip(external_score, -1.0, 1.0)
        if external_score != 0:
            external_weight = adjusted_weights.get("external_data", 20) / 100.0
            total_score += external_score * external_weight * 100
            total_weight += external_weight
            source_weights['external'] = external_weight
            reasons.extend(external_reasons)

        ml_score, ml_reasons, ml_total_confidence = self.score_ml_predictions(ml_predictions, market_context.trend)
        ml_score = np.clip(ml_score, -1.0, 1.0)
        if ml_score != 0:
            lstm_weight = adjusted_weights.get("lstm", 15) / 100.0
            xgboost_weight = adjusted_weights.get("xgboost", 10) / 100.0
            ml_weight = (lstm_weight + xgboost_weight) * ml_total_confidence
            total_score += ml_score * ml_weight * 100
            total_weight += ml_weight
            source_weights['ml'] = ml_weight
            reasons.extend(ml_reasons)

        source_importance = {
            'ml': 1.0,
            'derivatives': 0.9,
            'market_condition': 0.85,
            'fundamental': 0.6,
            'macro': 0.5,
            'order_book': 0.4,
            'trending': 0.2,
            'news': 0.7
        }
        
        weighted_missing = 0.0
        total_importance = 0.0
        
        for src in critical_sources:
            importance = source_importance.get(src, 0.5)
            total_importance += importance
            if src not in available_sources:
                weighted_missing += importance
        
        for src in self.OPTIONAL_SOURCES:
            importance = source_importance.get(src, 0.3)
            total_importance += importance
            if not external_data.get(src, {}).get('data'):
                weighted_missing += importance
        
        confidence_penalty = weighted_missing / total_importance if total_importance > 0 else 0.5

        if total_weight > 0:
            final_score = total_score / total_weight
        else:
            final_score = 0.0
        
        final_score = np.clip(final_score, -100.0, 100.0)
        
        return final_score, reasons, confidence_penalty, source_weights, True

    def score_ml_predictions(self, predictions: Dict[str, Dict[str, float]], current_trend: TrendDirection) -> Tuple[float, List[str], float]:
        score = 0.0
        reasons = []
        total_confidence = 0.0
        count = 0
        
        if not predictions:
            return score, reasons, 0.5

        model_weights = {
            'lstm': 0.6,
            'xgboost': 0.4
        }

        weighted_score = 0.0
        total_weight = 0.0

        for model_name, pred_data in predictions.items():
            pred_price = pred_data.get('prediction', 0)
            raw_confidence = pred_data.get('raw_confidence', pred_data.get('confidence', 0.5))
            calibrated_confidence = pred_data.get('confidence', raw_confidence)
            uncertainty = pred_data.get('uncertainty', 0.0)
            
            adjusted_confidence = calibrated_confidence * (1 - uncertainty ** 2)
            
            weight = model_weights.get(model_name, 0.5) * adjusted_confidence
            
            if pred_price != 0:
                direction = 1 if pred_price > 0 else -1
                weighted_score += direction * weight
                total_weight += weight
                total_confidence += adjusted_confidence
                count += 1
                reasons.append(f"ML Model ({model_name}): Predicts price {'increase' if direction > 0 else 'decrease'} "
                             f"(adjusted conf: {adjusted_confidence:.2f}, uncertainty: {uncertainty:.2f})")

        if total_weight > 0:
            score = weighted_score / total_weight
            total_weight_sum = sum(model_weights.values())
            normalized_weights = [w / total_weight_sum for w in model_weights.values()]
            weighted_score = sum(
                (1 if predictions[model]['prediction'] > 0 else -1) * normalized_weights[i]
                for i, model in enumerate(predictions.keys()) if predictions[model]['prediction'] != 0
            )
            score = weighted_score
        
        avg_confidence = total_confidence / count if count > 0 else 0.5

        return np.clip(score, -1.0, 1.0), reasons, avg_confidence

    def get_indicator_score(self, result, market_context: MarketAnalysis) -> float:
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
        
        base_score = direction * strength

        if market_context.trend == TrendDirection.BULLISH and market_context.trend_strength == TrendStrength.STRONG:
            if direction < 0 and 'oversold' not in interpretation:
                base_score *= 0.5
        elif market_context.trend == TrendDirection.BEARISH and market_context.trend_strength == TrendStrength.STRONG:
            if direction > 0 and 'overbought' not in interpretation:
                base_score *= 0.5
        
        return base_score

    def score_market_context(self, context: MarketAnalysis) -> Tuple[float, List[str]]:
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

        volume_score = 0.0
        if hasattr(context, 'volume_trend_score') and context.volume_trend_score is not None:
            volume_score = context.volume_trend_score
        else:
            if context.volume_confirmation:
                volume_score = 0.3 if context.trend == TrendDirection.BULLISH else -0.3

        score += volume_score
        reasons.append(f"Volume Score: {volume_score:.2f}")

        return np.clip(score / 1.8, -1.0, 1.0), reasons

    def score_external_data(self, data: Dict[str, Any], market_context: MarketAnalysis, symbol: str) -> Tuple[float, List[str], Dict[str, bool]]:
        score = 0.0
        reasons = []
        source_availability = {}

        for source in list(self.CRITICAL_SOURCES) + list(self.OPTIONAL_SOURCES):
            source_availability[source] = bool(data.get(source, {}).get('data'))

        if data.get("news") and data["news"].get('data', {}).get("score") is not None and data["news"]['data']["score"] != 0:
            news_score = data["news"]['data']["score"]
            score += np.clip(news_score, -1.0, 1.0)
            reasons.append(f"News Sentiment: {news_score:.2f}")

        if data.get("derivatives") and data["derivatives"].get('data'):
            derivatives: DerivativesAnalysis = data["derivatives"]['data']
            if derivatives and derivatives.funding_rate is not None:
                funding_score = self.normalize_funding_rate(derivatives.funding_rate, symbol, market_context)
                score += funding_score
                reasons.append(f"Funding Rate: {derivatives.funding_rate:.4f} (Normalized: {funding_score:.2f})")

            if derivatives and derivatives.taker_long_short_ratio is not None:
                if derivatives.taker_long_short_ratio > 1:
                    score += 0.1
                elif derivatives.taker_long_short_ratio < 1:
                    score -= 0.1
                reasons.append(f"Taker L/S Ratio: {derivatives.taker_long_short_ratio:.2f}")

        return np.clip(score, -1.0, 1.0), reasons, source_availability

    def normalize_funding_rate(self, funding_rate: float, symbol: str, market_context: MarketAnalysis) -> float:
        asset = symbol.split('/')[0]
        market_avg_funding = {
            'BTC': 0.01,
            'ETH': 0.008,
            'default': 0.02
        }.get(asset, 0.02)
        
        thresholds = (3, 1.5) if market_context.trend == TrendDirection.BULLISH else (1.5, 0.5)

        z_score = (funding_rate - market_avg_funding) / max(market_avg_funding * 0.5, 0.005)
        
        if z_score > thresholds[0]:
            return -0.3
        elif z_score > thresholds[1]:
            return -0.15
        elif z_score < -thresholds[0]:
            return 0.3
        elif z_score < -thresholds[1]:
            return 0.15
        else:
            return 0.0

    def adjust_weights_by_regime(self, base_weights: Dict[str, float], market_regime: Dict[str, Any]) -> Dict[str, float]:
        adjusted = base_weights.copy()
        
        if market_regime.get('market_type') == 'ranging':
            mean_reversion_indicators = ['rsi', 'stoch', 'cci', 'williams_r']
            for indicator in mean_reversion_indicators:
                if indicator in adjusted:
                    adjusted[indicator] *= 1.5
        elif market_regime.get('market_type') == 'trending':
            trend_indicators = ['sma', 'ema', 'dema', 'tema', 'hma', 'vwma', 'supertrend', 'adx', 'aroon', 'psar', 'ichimoku', 'kama', 'ma_ribbon']
            for indicator in trend_indicators:
                if indicator in adjusted:
                    adjusted[indicator] *= 1.3
        
        return adjusted

    def determine_signal_type(self, score: float, market_context: MarketAnalysis, 
                             symbol: str, timeframe: str) -> Tuple[SignalType, Dict[str, Any]]:
        base_threshold = self.config_manager.get("signal_threshold", 0.4) * 100

        threshold_info = {
            'base_threshold': base_threshold,
            'adaptive_threshold': base_threshold,
            'optimal_threshold': base_threshold
        }

        if score > base_threshold:
            return SignalType.BUY, threshold_info
        if score < -base_threshold:
            return SignalType.SELL, threshold_info
        return SignalType.HOLD, threshold_info

    async def create_trading_signal(
        self, symbol: str, timeframe: str, data: pd.DataFrame, signal_type: SignalType,
        confidence_score: float, reasons: List[str], market_context: MarketAnalysis,
        external_data: Dict[str, Any], analysis_timestamp: datetime, data_collection_timestamp: datetime,
        levels: DynamicLevels, risk_reward_ratio: float
    ) -> TradingSignal:

        expiry_hours = {
            '1h': 12, '4h': 24, '1d': 72, '1w': 168, '1M': 720
        }.get(timeframe, 72)
        
        volatility_multiplier = 1.0 - np.clip(market_context.volatility / 10, 0, 0.5)
        adjusted_expiry = expiry_hours * volatility_multiplier

        signal = TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            entry_price=levels.primary_entry,
            exit_price=levels.primary_exit,
            stop_loss=levels.tight_stop,
            timestamp=pd.to_datetime(data.index[-1]),
            timeframe=timeframe,
            confidence_score=confidence_score,
            reasons=reasons,
            risk_reward_ratio=risk_reward_ratio,
            predicted_profit=abs(levels.primary_exit - levels.primary_entry),
            volume_analysis={"volume_trend": market_context.volume_trend},
            market_context=market_context.__dict__,
            dynamic_levels=levels.__dict__,
            fundamental_analysis=external_data.get("fundamental", {}).get('data') if external_data.get("fundamental") else None,
            on_chain_analysis=external_data.get("on_chain", {}).get('data') if external_data.get("on_chain") else None,
            derivatives_analysis=external_data.get("derivatives", {}).get('data') if external_data.get("derivatives") else None,
            order_book=external_data.get("order_book", {}).get('data') if external_data.get("order_book") else None,
            macro_data=external_data.get("macro", {}).get('data') if external_data.get("macro") else None,
            trending_data=external_data.get("trending", {}).get('data') if external_data.get("trending") else None,
            expiry_time=analysis_timestamp + timedelta(hours=adjusted_expiry),
            position_size=self._calculate_position_size(confidence_score, market_context.volatility, risk_reward_ratio, symbol, timeframe),
            data_collection_timestamp=data_collection_timestamp,
            analysis_timestamp=analysis_timestamp
        )

        return signal

    async def _check_btc_correlation(self, symbol: str, signal_type: SignalType, data: pd.DataFrame) -> bool:
        if 'BTC' in symbol.upper():
            return True
        
        try:
            timeframe_map = {'1h': '1h', '4h': '4h', '1d': '1d', '1w': '1d', '1M': '1w'}
            tf = timeframe_map.get(getattr(data.index, 'freqstr', '1d') or '1d', '1d')
            limit_map = {'1h': 200, '4h': 150, '1d': 100, '1w': 50, '1M': 20}
            limit = limit_map.get(tf, 100)
            
            btc_data = await self.market_data_provider.fetch_ohlcv_data('BTC/USDT', tf, limit=limit)
            if btc_data is None or btc_data.empty or len(btc_data) < 20:
                return True
            
            aligned_df = pd.merge_asof(data.reset_index(), btc_data.reset_index(), on="timestamp", 
                                       direction="nearest", tolerance=pd.Timedelta(minutes=1),
                                       suffixes=('_sym', '_btc'))
            
            symbol_returns = aligned_df['close_sym'].pct_change().dropna()
            btc_returns = aligned_df['close_btc'].pct_change().dropna()
            
            min_len = min(len(symbol_returns), len(btc_returns))
            if min_len < 20:
                return True
            
            symbol_returns = symbol_returns.tail(min_len)
            btc_returns = btc_returns.tail(min_len)
            
            correlation = symbol_returns.corr(btc_returns)
            
            volatility = symbol_returns.std()
            dynamic_threshold = 0.3 + np.clip(volatility * 2, 0, 0.4)
            
            if len(btc_data) < min_len // 2 + 1:
                return True
                
            btc_trend = 1 if btc_data['close'].iloc[-1] > btc_data['close'].iloc[-min_len//2] else -1
            signal_direction = 1 if signal_type == SignalType.BUY else -1
            
            lagged_correlation = self.calculate_lagged_correlation(symbol_returns, btc_returns, lag=24)
            
            if lagged_correlation > 0.5:
                logger.info(f"Signal for {symbol} has high lagged correlation with BTC trend. Rejection possible.")
            
            if correlation < dynamic_threshold:
                 return True

            if lagged_correlation > dynamic_threshold and btc_trend != signal_direction:
                logger.info(f"Signal for {symbol} is against BTC trend (correlation: {correlation:.2f}, lagged: {lagged_correlation:.2f}, threshold: {dynamic_threshold:.2f})")
                return False
            
            return True
        except Exception as e:
            logger.warning(f"Error checking BTC correlation for {symbol}: {e}")
            return True

    def calculate_lagged_correlation(self, series1: pd.Series, series2: pd.Series, lag: int = 24) -> float:
        if len(series1) < lag or len(series2) < lag:
            return 0.0
        
        lagged_series2 = series2.shift(lag).dropna()
        aligned_series1 = series1.iloc[-len(lagged_series2):]
        
        if len(aligned_series1) < 10:
            return 0.0
        
        return aligned_series1.corr(lagged_series2)

    def _calculate_position_size(self, confidence: float, volatility: float, risk_reward: float, symbol: str, timeframe: str) -> float:
        base_size = 0.02
        
        confidence_multiplier = confidence / 100.0
        volatility_multiplier = np.clip(1.0 / (1.0 + volatility / 5.0), 0.5, 1.5)
        rr_multiplier = np.clip(risk_reward / 3.0, 0.5, 1.5)
        
        historical_win_rate = self._get_recent_win_rate(symbol, timeframe) / 100.0
        avg_win = getattr(self, '_historical_avg_win', 1.0)
        avg_loss = getattr(self, '_historical_avg_loss', 1.0)
        
        kelly_fraction = (historical_win_rate * avg_win - (1 - historical_win_rate) * avg_loss) / avg_win if avg_win > 0 else 0
        
        position_size = base_size * confidence_multiplier * volatility_multiplier * rr_multiplier * kelly_fraction * 0.5
        
        return np.clip(position_size, 0.01, 0.05)

    def _calculate_dynamic_levels(self, data: pd.DataFrame, signal_type: SignalType, market_context: MarketAnalysis, order_book: OrderBook = None) -> DynamicLevels:
        last_close = data['close'].iloc[-1]

        adx_val = market_context.adx if hasattr(market_context, 'adx') else 25
        
        volatility_multiplier = 1.0 + np.clip(market_context.volatility / 10, 0, 1)
        
        if market_context.volatility > 0:
            atr = (market_context.volatility / 100 * last_close) * volatility_multiplier
        else:
            atr = data['close'].pct_change().std() * last_close
        
        if pd.isna(atr) or atr <= 0:
            atr = last_close * 0.02

        min_rr = 2.0 if adx_val > 35 else 1.8

        entry = float(last_close)

        risk = atr * (1.5 if adx_val > 40 else 1.2)
        
        if signal_type == SignalType.BUY:
            base_stop_loss = entry - risk
            base_take_profit = entry + (risk * min_rr)
        else:
            base_stop_loss = entry + risk
            base_take_profit = entry - (risk * min_rr)

        fib_levels = calculate_fibonacci_levels(data)
        pivot_levels = calculate_pivot_points(data)
        
        min_distance = risk * 0.3
        max_distance = risk * 2.0
        
        if signal_type == SignalType.BUY:
            valid_supports = [s for s in market_context.support_levels if isinstance(s, (int, float)) and s < entry and s > base_stop_loss - risk]
            if valid_supports:
                nearest_support = max(valid_supports)
                potential_stop = nearest_support - (risk * 0.1)
                if min_distance <= abs(potential_stop - entry) <= max_distance:
                    stop_loss = potential_stop
                else:
                    stop_loss = base_stop_loss
            else:
                stop_loss = base_stop_loss
            
            valid_resistances = [r for r in market_context.resistance_levels if isinstance(r, (int, float)) and r > entry and r < base_take_profit + risk]
            if valid_resistances:
                nearest_resistance = min(valid_resistances)
                take_profit = nearest_resistance - (risk * 0.1)
            else:
                take_profit = base_take_profit
        else:
            valid_resistances = [r for r in market_context.resistance_levels if isinstance(r, (int, float)) and r > entry and r < base_stop_loss + risk]
            if valid_resistances:
                nearest_resistance = min(valid_resistances)
                potential_stop = nearest_resistance + (risk * 0.1)
                if min_distance <= abs(potential_stop - entry) <= max_distance:
                    stop_loss = potential_stop
                else:
                    stop_loss = base_stop_loss
            else:
                stop_loss = base_stop_loss
            
            valid_supports = [s for s in market_context.support_levels if isinstance(s, (int, float)) and s < entry and s > base_take_profit - risk]
            if valid_supports:
                nearest_support = max(valid_supports)
                take_profit = nearest_support + (risk * 0.1)
            else:
                take_profit = base_take_profit

        if order_book and order_book.bids and signal_type == SignalType.BUY:
            major_bid = max([p for p, _ in order_book.bids[:5]]) if order_book.bids else None
            if major_bid and abs(major_bid - stop_loss) / stop_loss < 0.005:
                stop_loss = major_bid * 0.995

        trailing_stop_distance = atr * 0.8
        breakeven_distance = abs(entry - stop_loss) * 0.5

        return DynamicLevels(
            primary_entry=entry,
            secondary_entry=entry,
            primary_exit=take_profit,
            secondary_exit=take_profit,
            tight_stop=stop_loss,
            wide_stop=stop_loss,
            breakeven_point=entry,
            trailing_stop=trailing_stop_distance
        )

    def _calculate_risk_reward_ratio(self, entry_price: float, stop_loss: float, take_profit: float, signal_type: SignalType) -> float:
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        if risk == 0:
            return 0.0
        
        commission_per_trade = 0.001
        slippage = 0.0005
        
        effective_risk = risk * (1 + commission_per_trade + slippage)
        effective_reward = reward * (1 - commission_per_trade - slippage)
        
        return effective_reward / effective_risk if effective_risk > 0 else 0.0

    _historical_performance: Dict[str, Dict[str, Any]] = {}

    def _validate_signal_consistency(self, signal: TradingSignal, market_context: MarketAnalysis, ml_predictions: Dict, technical_results: Dict, external_data: Dict) -> bool:
        trend_strength_score = market_context.adx if market_context.adx is not None else 25
        
        if signal.signal_type == SignalType.BUY and trend_strength_score < 20:
            logger.warning(f"Consistency check: BUY signal in a weak trend for {signal.symbol}-{signal.timeframe}")
            signal.confidence_score *= 0.85
        
        if signal.signal_type == SignalType.SELL and trend_strength_score < 20:
            logger.warning(f"Consistency check: SELL signal in a weak trend for {signal.symbol}-{signal.timeframe}")
            signal.confidence_score *= 0.85

        ml_directions = [1 if pred.get('prediction', 0) > 0 else -1 for pred in ml_predictions.values()]
        signal_direction = 1 if signal.signal_type == SignalType.BUY else -1
        
        if ml_directions and signal_direction not in ml_directions:
            logger.warning(f"Consistency check: Signal direction conflicts with ML predictions for {signal.symbol}-{signal.timeframe}")
            signal.confidence_score *= 0.8
        
        if signal.timeframe in ['1d', '1w', '1M']:
            is_critical_missing = not all(
                external_data.get(src, {}).get('data') for src in ['derivatives', 'fundamental']
            )
            if is_critical_missing:
                logger.warning(f"Consistency check: Missing critical external data for long-term signal {signal.symbol}-{signal.timeframe}")
                signal.confidence_score *= 0.75
                
        return True