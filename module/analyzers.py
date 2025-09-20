from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import talib
from scipy.signal import find_peaks

from module.core import (
    DerivativesAnalysis,
    DynamicLevels,
    MarketAnalysis,
    MarketCondition,
    OrderBookAnalysis,
    SignalType,
    TrendDirection,
    TrendStrength,
)
from module.indicators import RSIIndicator


class DerivativesAnalyzer:
    @staticmethod
    def analyze(derivatives_data: DerivativesAnalysis) -> Dict:
        analysis = {}
        if derivatives_data.funding_rate > 0.001:
            analysis["funding_sentiment"] = "bullish"
        elif derivatives_data.funding_rate < -0.001:
            analysis["funding_sentiment"] = "bearish"
        else:
            analysis["funding_sentiment"] = "neutral"
        analysis["oi_interpretation"] = "neutral"
        return analysis


class OrderBookAnalyzer:
    @staticmethod
    def analyze(order_book: Dict, wall_threshold: float = 10.0) -> OrderBookAnalysis:
        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])
        if not bids or not asks:
            return OrderBookAnalysis()
        total_bid_volume = sum(amount for _, amount in bids)
        total_ask_volume = sum(amount for _, amount in asks)
        market_depth_ratio = total_bid_volume / total_ask_volume if total_ask_volume > 0 else float("inf")
        avg_bid_size = total_bid_volume / len(bids) if bids else 0
        avg_ask_size = total_ask_volume / len(asks) if asks else 0
        buy_wall = next(((p, a) for p, a in bids if avg_bid_size > 0 and a > avg_bid_size * wall_threshold), None)
        sell_wall = next(((p, a) for p, a in asks if avg_ask_size > 0 and a > avg_ask_size * wall_threshold), None)
        return OrderBookAnalysis(buy_wall=buy_wall, sell_wall=sell_wall, market_depth_ratio=market_depth_ratio)


class PatternAnalyzer:
    CANDLE_PATTERNS = {
        'CDL2CROWS': 'Two Crows',
        'CDL3BLACKCROWS': 'Three Black Crows',
        'CDL3INSIDE': 'Three Inside Up/Down',
        'CDL3LINESTRIKE': 'Three-Line Strike',
        'CDL3OUTSIDE': 'Three Outside Up/Down',
        'CDL3STARSINSOUTH': 'Three Stars in the South',
        'CDL3WHITESOLDIERS': 'Three White Soldiers',
        'CDLABANDONEDBABY': 'Abandoned Baby',
        'CDLADVANCEBLOCK': 'Advance Block',
        'CDLBELTHOLD': 'Belt-hold',
        'CDLBREAKAWAY': 'Breakaway',
        'CDLCLOSINGMARUBOZU': 'Closing Marubozu',
        'CDLCONCEALBABYSWALL': 'Concealing Baby Swallow',
        'CDLCOUNTERATTACK': 'Counterattack',
        'CDLDARKCLOUDCOVER': 'Dark Cloud Cover',
        'CDLDOJI': 'Doji',
        'CDLDOJISTAR': 'Doji Star',
        'CDLDRAGONFLYDOJI': 'Dragonfly Doji',
        'CDLENGULFING': 'Engulfing Pattern',
        'CDLEVENINGDOJISTAR': 'Evening Doji Star',
        'CDLEVENINGSTAR': 'Evening Star',
        'CDLGAPSIDESIDEWHITE': 'Up/Down-gap side-by-side white lines',
        'CDLGRAVESTONEDOJI': 'Gravestone Doji',
        'CDLHAMMER': 'Hammer',
        'CDLHANGINGMAN': 'Hanging Man',
        'CDLHARAMI': 'Harami Pattern',
        'CDLHARAMICROSS': 'Harami Cross Pattern',
        'CDLHIGHWAVE': 'High-Wave Candle',
        'CDLHIKKAKE': 'Hikkake Pattern',
        'CDLHIKKAKEMOD': 'Modified Hikkake Pattern',
        'CDLHOMINGPIGEON': 'Homing Pigeon',
        'CDLIDENTICAL3CROWS': 'Identical Three Crows',
        'CDLINNECK': 'In-Neck Pattern',
        'CDLINVERTEDHAMMER': 'Inverted Hammer',
        'CDLKICKING': 'Kicking',
        'CDLKICKINGBYLENGTH': 'Kicking - bull/bear determined by the longer marubozu',
        'CDLLADDERBOTTOM': 'Ladder Bottom',
        'CDLLONGLEGGEDDOJI': 'Long-Legged Doji',
        'CDLLONGLINE': 'Long Line Candle',
        'CDLMARUBOZU': 'Marubozu',
        'CDLMATCHINGLOW': 'Matching Low',
        'CDLMATHOLD': 'Mat Hold',
        'CDLMORNINGDOJISTAR': 'Morning Doji Star',
        'CDLMORNINGSTAR': 'Morning Star',
        'CDLONNECK': 'On-Neck Pattern',
        'CDLPIERCING': 'Piercing Pattern',
        'CDLRICKSHAWMAN': 'Rickshaw Man',
        'CDLRISEFALL3METHODS': 'Rising/Falling Three Methods',
        'CDLSEPARATINGLINES': 'Separating Lines',
        'CDLSHOOTINGSTAR': 'Shooting Star',
        'CDLSHORTLINE': 'Short Line Candle',
        'CDLSPINNINGTOP': 'Spinning Top',
        'CDLSTALLEDPATTERN': 'Stalled Pattern',
        'CDLSTICKSANDWICH': 'Stick Sandwich',
        'CDLTAKURI': 'Takuri (Dragonfly Doji with very long lower shadow)',
        'CDLTASUKIGAP': 'Tasuki Gap',
        'CDLTHRUSTING': 'Thrusting Pattern',
        'CDLTRISTAR': 'Tristar Pattern',
        'CDLUNIQUE3RIVER': 'Unique 3 River',
        'CDLUPSIDEGAP2CROWS': 'Upside Gap Two Crows',
        'CDLXSIDEGAP3METHODS': 'Upside/Downside Gap Three Methods'
    }

    @staticmethod
    def detect_patterns(data: pd.DataFrame) -> List[str]:
        if len(data) < 3:
            return []
        
        op = data["open"].to_numpy(zero_copy_only=False)
        hi = data["high"].to_numpy(zero_copy_only=False)
        lo = data["low"].to_numpy(zero_copy_only=False)
        cl = data["close"].to_numpy(zero_copy_only=False)

        detected_patterns = []
        for pattern_func_name, pattern_desc in PatternAnalyzer.CANDLE_PATTERNS.items():
            try:
                pattern_func = getattr(talib, pattern_func_name)
                result = pattern_func(op, hi, lo, cl)
                if result[-1] != 0:
                    direction = "bullish" if result[-1] > 0 else "bearish"
                    detected_patterns.append(f"{pattern_desc} ({direction})")
            except Exception:
                continue
        
        return detected_patterns

    @staticmethod
    def detect_divergence(data: pd.DataFrame, indicator: pd.Series, window=14) -> List[str]:
        patterns = []
        if len(data) < window or len(indicator) < window:
            return patterns

        lows = data['low'].to_numpy()
        highs = data['high'].to_numpy()
        indicator_vals = indicator.to_numpy()

        price_low_peaks, _ = find_peaks(-lows, distance=window//2)
        if len(price_low_peaks) >= 2:
            last_peak_idx = price_low_peaks[-1]
            prev_peak_idx = price_low_peaks[-2]
            if lows[last_peak_idx] > lows[prev_peak_idx] and indicator_vals[last_peak_idx] < indicator_vals[prev_peak_idx]:
                patterns.append("bullish_divergence")

        price_high_peaks, _ = find_peaks(highs, distance=window//2)
        if len(price_high_peaks) >= 2:
            last_peak_idx = price_high_peaks[-1]
            prev_peak_idx = price_high_peaks[-2]
            if highs[last_peak_idx] < highs[prev_peak_idx] and indicator_vals[last_peak_idx] > indicator_vals[prev_peak_idx]:
                patterns.append("bearish_divergence")
                
        return patterns

class MarketConditionAnalyzer:
    def __init__(self):
        self.volume_analyzer = VolumeAnalyzer()

    def analyze_market_condition(self, data: pd.DataFrame, **kwargs) -> MarketAnalysis:
        trend = self._determine_trend(data)
        trend_strength = self._calculate_trend_strength(data)
        volatility = self._calculate_volatility(data)
        momentum = self._calculate_momentum(data)
        hurst = self._calculate_hurst_exponent(data["close"])

        return MarketAnalysis(
            trend=trend,
            trend_strength=trend_strength,
            volatility=volatility,
            momentum_score=momentum,
            hurst_exponent=hurst,
            **kwargs,
        )

    def _determine_trend(self, data: pd.DataFrame) -> TrendDirection:
        if len(data) < 50:
            return TrendDirection.SIDEWAYS
        sma_20 = data["close"].rolling(20).mean().iloc[-1]
        sma_50 = data["close"].rolling(50).mean().iloc[-1]
        if sma_20 > sma_50 * 1.01:
            return TrendDirection.BULLISH
        if sma_50 > sma_20 * 1.01:
            return TrendDirection.BEARISH
        return TrendDirection.SIDEWAYS

    def _calculate_trend_strength(self, data: pd.DataFrame) -> TrendStrength:
        if len(data) < 28:
            return TrendStrength.WEAK
        adx = talib.ADX(data["high"], data["low"], data["close"], timeperiod=14)
        adx_val = adx[-1] if not np.isnan(adx[-1]) else 0
        if adx_val > 25:
            return TrendStrength.STRONG
        if adx_val > 20:
            return TrendStrength.MODERATE
        return TrendStrength.WEAK

    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        if len(data) < 14:
            return 0.0
        atr = talib.ATR(data["high"], data["low"], data["close"], timeperiod=14)
        atr_val = atr[-1] if not np.isnan(atr[-1]) else 0
        price = data["close"].iloc[-1]
        return (atr_val / price) * 100 if price > 0 else 0.0

    def _calculate_momentum(self, data: pd.DataFrame) -> float:
        if len(data) < 10:
            return 0.0
        mom = talib.MOM(data["close"], timeperiod=10)
        return mom[-1] if not np.isnan(mom[-1]) else 0.0

    def _calculate_hurst_exponent(self, series: pd.Series, max_lag=100) -> Optional[float]:
        if len(series) < max_lag:
            return None
        series_np = series.to_numpy()
        lags = range(2, max_lag)
        tau = [
            np.sqrt(np.std(series_np[lag:] - series_np[:-lag]))
            for lag in lags
        ]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]


class VolumeAnalyzer:
    def analyze_volume_pattern(self, data: pd.DataFrame) -> Dict[str, float]:
        if len(data) < 20:
            return {}
        volume_ma_20 = data["volume"].rolling(20).mean().iloc[-1]
        current_volume = data["volume"].iloc[-1]
        ratio = current_volume / volume_ma_20 if volume_ma_20 > 0 else 1
        return {"volume_ratio": ratio}