import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd

from common.core import (
    TradingSignal, SignalType, DynamicLevels, TrendStrength
)
from data.data_provider import MarketDataProvider
from data.data_validator import DataQualityChecker
from features.feature_engineering import FeatureEngineer
from features.indicators.factory import IndicatorFactory
from analysis.market_analyzer import MarketConditionAnalyzer, PatternAnalyzer
from analysis.correlation import IndicatorCorrelationManager
from analysis.scoring import AnalysisScorer
from modeling.model_manager import ModelManager
from config.settings import ConfigManager
from config.logger import logger
from common.utils import calculate_risk_reward_ratio, calculate_dynamic_levels, detect_market_regime
from common.constants import LONG_TERM_CONFIG


class SignalGenerator:
    """
    Orchestrates the entire analysis pipeline to generate a trading signal.
    This is the refactored version of the old AnalysisEngine.
    """
    def __init__(
        self,
        data_provider: MarketDataProvider,
        model_manager: ModelManager,
        config_manager: ConfigManager,
    ):
        self.data_provider = data_provider
        self.model_manager = model_manager
        self.config_manager = config_manager
        
        # Initialize components
        self.feature_engineer = FeatureEngineer()
        self.market_analyzer = MarketConditionAnalyzer()
        self.pattern_analyzer = PatternAnalyzer()
        self.correlation_manager = IndicatorCorrelationManager()
        self.scorer = AnalysisScorer()
        self.data_validator = DataQualityChecker()

    async def generate_signal(self, symbol: str, timeframe: str, data: pd.DataFrame) -> Optional[TradingSignal]:
        """
        The main method to run a full analysis for a given symbol and timeframe.
        """
        analysis_timestamp = datetime.now(timezone.utc)
        
        # 1. Validate Data Quality
        is_valid, quality_msg = self.data_validator.validate_data_quality(data, timeframe)
        if not is_valid:
            logger.warning(f"Data quality check failed for {symbol}-{timeframe}: {quality_msg}")
            return None

        # 2. Basic Market Analysis (Regime, Context)
        market_regime = detect_market_regime(data)
        market_context = self.market_analyzer.analyze_market_condition(data)
        
        # Adjust indicator weights based on market regime
        base_weights = self.config_manager.get_indicator_weights(timeframe)
        adjusted_weights = self.adjust_weights_by_regime(base_weights, market_regime)

        # 3. Feature Engineering & Technical Analysis
        features_df = self.feature_engineer.create_features(data)
        
        # 4. Score Technical Indicators
        tech_score = self.scorer.score_technical_results(self.feature_engineer.indicator_factory.last_results, market_context)
        
        # 5. Score Market Context
        market_score, market_reasons = self.scorer.score_market_context(market_context)

        # 6. Gather and Score External Data (Derivatives, News, etc.)
        # This part would involve calling fetchers, similar to the old AnalysisEngine.
        # For simplicity, we'll assume it returns a score and reasons.
        # external_data = await self._gather_external_data(symbol)
        # external_score, external_reasons = self.scorer.score_external_data(external_data, market_context, symbol)
        external_score, external_reasons = 0.0, [] # Placeholder

        # 7. Get and Score ML Predictions
        current_price = data['close'].iloc[-1]
        # ml_predictions = await self.model_manager.predict_all(symbol, timeframe, data)
        # ml_score, ml_reasons, ml_confidence = self.scorer.score_ml_predictions(ml_predictions, current_price)
        ml_score, ml_reasons, ml_confidence = 0.0, [], 0.0 # Placeholder
        
        # 8. Calculate Combined Score
        final_score, all_reasons = self.calculate_combined_score(
            tech_score, market_score, external_score, ml_score, adjusted_weights, market_reasons + external_reasons + ml_reasons
        )

        # 9. Determine Signal Type
        signal_type = self.determine_signal_type(final_score)
        if signal_type == SignalType.HOLD:
            logger.info(f"No signal for {symbol}-{timeframe}. Final Score: {final_score:.2f}")
            return None

        # 10. Calculate Levels and Create Signal
        levels = calculate_dynamic_levels(data, signal_type.value, market_context)
        rr_ratio = calculate_risk_reward_ratio(levels['primary_entry'], levels['tight_stop'], levels['primary_exit'], signal_type.value)
        
        min_rr_ratio = LONG_TERM_CONFIG.get('min_risk_reward_ratio', 2.0)
        if rr_ratio < min_rr_ratio:
            logger.info(f"Risk/Reward {rr_ratio:.2f} below minimum {min_rr_ratio} for {symbol}-{timeframe}")
            return None
        
        # Create the final signal object
        signal = TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            entry_price=levels['primary_entry'],
            exit_price=levels['primary_exit'],
            stop_loss=levels['tight_stop'],
            timestamp=data.index[-1].to_pydatetime(),
            timeframe=timeframe,
            confidence_score=abs(final_score),
            reasons=all_reasons,
            risk_reward_ratio=rr_ratio,
            predicted_profit=abs(levels['primary_exit'] - levels['primary_entry']),
            volume_analysis={"volume_trend": market_context.volume_trend},
            market_context=market_context.__dict__,
            dynamic_levels=levels,
            analysis_timestamp=analysis_timestamp,
            # ... other fields would be populated here
        )
        
        return signal

    def calculate_combined_score(self, tech_score, market_score, external_score, ml_score, weights, reasons) -> Tuple[float, List[str]]:
        """
        Combines scores from all analysis parts into a single final score.
        """
        tech_weight = weights.get('technical_analysis', 40)
        market_weight = weights.get('market_context', 25)
        external_weight = weights.get('external_data', 15)
        ml_weight = weights.get('ml_models', 20)
        
        total_weight = tech_weight + market_weight + external_weight + ml_weight
        
        final_score = (
            (tech_score * tech_weight) +
            (market_score * market_weight) +
            (external_score * external_weight) +
            (ml_score * ml_weight)
        ) / total_weight
        
        return final_score * 100, reasons

    def determine_signal_type(self, score: float) -> SignalType:
        """
        Determines the signal type based on the final score and thresholds.
        """
        threshold = self.config_manager.get('signal_threshold', 50)
        if score > threshold:
            return SignalType.BUY
        if score < -threshold:
            return SignalType.SELL
        return SignalType.HOLD

    def adjust_weights_by_regime(self, base_weights: Dict[str, float], market_regime: Dict[str, Any]) -> Dict[str, float]:
        """
        Adjusts indicator weights based on the detected market regime.
        """
        adjusted = base_weights.copy()
        
        if market_regime.get('market_type') == 'ranging':
            mean_reversion_indicators = ['rsi', 'stoch', 'cci', 'williams_r']
            for indicator in mean_reversion_indicators:
                if indicator in adjusted:
                    adjusted[indicator] *= 1.5
        elif market_regime.get('market_type') == 'trending':
            trend_indicators = ['sma', 'ema', 'adx', 'supertrend', 'psar', 'ichimoku']
            for indicator in trend_indicators:
                if indicator in adjusted:
                    adjusted[indicator] *= 1.3
        
        return adjusted