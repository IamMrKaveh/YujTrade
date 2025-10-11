from typing import Dict, Any, List, Tuple
import numpy as np

from common.core import MarketAnalysis, TrendDirection, DerivativesAnalysis
from config.logger import logger


class AnalysisScorer:
    """
    A class dedicated to scoring different aspects of the market analysis.
    """

    def score_technical_results(
        self, processed_results: Dict[str, Any], market_context: MarketAnalysis
    ) -> float:
        """
        Calculates a score based on technical indicators.
        """
        # This is a simplified scoring logic. In a real-world scenario, this would be more complex.
        # The logic is moved from the old AnalysisEngine.
        total_score = 0
        count = 0
        for name, item in processed_results.items():
            result = item.get("result")
            if not result or not hasattr(result, "value") or np.isnan(result.value):
                continue

            score = self.get_indicator_score(result, market_context)
            total_score += score
            count += 1

        return (total_score / count) if count > 0 else 0.0

    def get_indicator_score(self, result, market_context: MarketAnalysis) -> float:
        strength = (
            np.clip(result.signal_strength / 100.0, 0, 1)
            if result.signal_strength is not None
            else 0.5
        )

        bullish_keywords = [
            "bullish",
            "oversold",
            "above",
            "buy",
            "up",
            "positive",
            "accumulation",
            "uptrend",
            "support",
        ]
        bearish_keywords = [
            "bearish",
            "overbought",
            "below",
            "sell",
            "down",
            "negative",
            "distribution",
            "downtrend",
            "resistance",
        ]

        interpretation = result.interpretation.lower()

        direction = 0
        is_bullish = any(keyword in interpretation for keyword in bullish_keywords)
        is_bearish = any(keyword in interpretation for keyword in bearish_keywords)

        if is_bullish and not is_bearish:
            direction = 1
        elif is_bearish and not is_bullish:
            direction = -1

        base_score = direction * strength

        if (
            market_context.trend == TrendDirection.BULLISH
            and "strong" in market_context.trend_strength.value
        ):
            if direction < 0 and "oversold" not in interpretation:
                base_score *= 0.5  # Penalize bearish signals in a strong uptrend
        elif (
            market_context.trend == TrendDirection.BEARISH
            and "strong" in market_context.trend_strength.value
        ):
            if direction > 0 and "overbought" not in interpretation:
                base_score *= 0.5  # Penalize bullish signals in a strong downtrend

        return base_score

    def score_market_context(self, context: MarketAnalysis) -> Tuple[float, List[str]]:
        """
        Scores the overall market context.
        """
        score = 0.0
        reasons = []

        trend_map = {
            TrendDirection.BULLISH: 1.0,
            TrendDirection.BEARISH: -1.0,
            TrendDirection.SIDEWAYS: 0.0,
        }
        score += trend_map.get(context.trend, 0.0)
        reasons.append(f"Trend: {context.trend.value}")

        strength_map = {"strong": 1.0, "moderate": 0.5, "weak": 0.1}
        score *= strength_map.get(context.trend_strength.value, 0.5)

        if context.market_condition.value == "oversold":
            score += 0.5
        elif context.market_condition.value == "overbought":
            score -= 0.5
        reasons.append(f"Condition: {context.market_condition.value}")

        volume_score = (
            context.volume_trend_score
            if hasattr(context, "volume_trend_score")
            and context.volume_trend_score is not None
            else 0.0
        )
        score += volume_score
        reasons.append(f"Volume Score: {volume_score:.2f}")

        return np.clip(score / 1.8, -1.0, 1.0), reasons

    def score_external_data(
        self, data: Dict[str, Any], market_context: MarketAnalysis, symbol: str
    ) -> Tuple[float, List[str]]:
        """
        Scores external data sources like news, derivatives, etc.
        """
        score = 0.0
        reasons = []

        # Score News
        if data.get("news") and data["news"].get("data", {}).get("score") is not None:
            news_score = data["news"]["data"]["score"]
            score += np.clip(news_score / 10.0, -0.5, 0.5)  # Normalize news score
            reasons.append(f"News Sentiment: {news_score:.2f}")

        # Score Derivatives
        if data.get("derivatives") and data["derivatives"].get("data"):
            derivatives: DerivativesAnalysis = data["derivatives"]["data"]
            if derivatives.funding_rate is not None:
                funding_score = self.normalize_funding_rate(
                    derivatives.funding_rate, symbol, market_context
                )
                score += funding_score
                reasons.append(
                    f"Funding Rate: {derivatives.funding_rate:.4f} (Score: {funding_score:.2f})"
                )

            if derivatives.taker_long_short_ratio is not None:
                ratio = derivatives.taker_long_short_ratio
                ratio_score = 0.2 if ratio > 1.1 else -0.2 if ratio < 0.9 else 0.0
                score += ratio_score
                reasons.append(
                    f"Taker L/S Ratio: {ratio:.2f} (Score: {ratio_score:.2f})"
                )

        return np.clip(score, -1.0, 1.0), reasons

    def normalize_funding_rate(
        self, funding_rate: float, symbol: str, market_context: MarketAnalysis
    ) -> float:
        """
        Normalizes funding rate into a score from -0.3 to 0.3.
        Negative funding is generally bullish, positive is bearish.
        """
        asset = symbol.split("/")[0]
        # These are just example averages, could be dynamically calculated
        market_avg_funding = {"BTC": 0.01, "ETH": 0.008, "default": 0.02}.get(
            asset, 0.02
        )

        # High funding rate is bearish, so score is negative
        if funding_rate > market_avg_funding * 2:
            return -0.3
        if funding_rate > market_avg_funding:
            return -0.15
        # Negative funding rate is bullish, so score is positive
        if funding_rate < -market_avg_funding:
            return 0.3
        if funding_rate < 0:
            return 0.15

        return 0.0

    def score_ml_predictions(
        self, predictions: Dict[str, Dict[str, float]], current_price: float
    ) -> Tuple[float, List[str], float]:
        """
        Scores predictions from machine learning models.
        """
        if not predictions:
            return 0.0, [], 0.0

        weighted_score = 0.0
        total_confidence = 0.0
        reasons = []

        model_weights = {"lstm": 0.6, "xgboost": 0.4}

        for model_name, pred_data in predictions.items():
            pred_price = pred_data.get("prediction", 0)
            confidence = pred_data.get("confidence", 0)

            if pred_price == 0:
                continue

            direction = 1 if pred_price > current_price else -1

            # Weight the score by the model's confidence and its predefined weight
            model_weight = model_weights.get(model_name, 0.5)
            weighted_score += direction * (confidence / 100.0) * model_weight
            total_confidence += confidence * model_weight

            reasons.append(
                f"ML ({model_name}): Predicts {'increase' if direction > 0 else 'decrease'} (Conf: {confidence:.1f}%)"
            )

        # Normalize the score by the total confidence to get a value between -1 and 1
        final_score = (
            (weighted_score / total_confidence * 100) if total_confidence > 0 else 0.0
        )
        avg_confidence = (
            total_confidence / sum(model_weights.values()) if predictions else 0.0
        )

        return np.clip(final_score, -1.0, 1.0), reasons, avg_confidence / 100.0
