# analyzers.py

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import talib
from scipy.signal import find_peaks

from module.core import (
    MarketAnalysis,
    MarketCondition,
    TrendDirection,
    TrendStrength,
)


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
        
        op = data["open"].to_numpy(dtype=np.float64)
        hi = data["high"].to_numpy(dtype=np.float64)
        lo = data["low"].to_numpy(dtype=np.float64)
        cl = data["close"].to_numpy(dtype=np.float64)

        detected_patterns = []
        pattern_scores = {}
        
        for pattern_func_name, pattern_desc in PatternAnalyzer.CANDLE_PATTERNS.items():
            try:
                pattern_func = getattr(talib, pattern_func_name)
                result = pattern_func(op, hi, lo, cl)
                if result[-1] != 0:
                    direction = "bullish" if result[-1] > 0 else "bearish"
                    pattern_name = f"{pattern_desc} ({direction})"
                    
                    context_score = PatternAnalyzer._analyze_pattern_context(data, direction, pattern_desc)
                    
                    pattern_scores[pattern_name] = context_score
                    
            except Exception:
                continue
        
        sorted_patterns = sorted(pattern_scores.items(), key=lambda x: x[1], reverse=True)
        detected_patterns = [pattern for pattern, score in sorted_patterns if score > 0.5]
        
        return detected_patterns

    @staticmethod
    def _analyze_pattern_context(data: pd.DataFrame, direction: str, pattern_name: str) -> float:
        score = 1.0
        
        if len(data) < 20:
            return score
        
        sma_20 = data['close'].rolling(20).mean()
        current_trend = "bullish" if data['close'].iloc[-1] > sma_20.iloc[-1] else "bearish"
        
        reversal_patterns = ['Hammer', 'Hanging Man', 'Inverted Hammer', 'Shooting Star', 
                            'Morning Star', 'Evening Star', 'Engulfing Pattern']
        is_reversal = any(rev in pattern_name for rev in reversal_patterns)
        
        if is_reversal:
            if (direction == "bullish" and current_trend == "bearish") or \
               (direction == "bearish" and current_trend == "bullish"):
                score *= 1.5
            else:
                score *= 0.7
        else:
            if direction == current_trend:
                score *= 1.3
            else:
                score *= 0.8
        
        if 'volume' in data.columns and len(data) >= 20:
            avg_volume = data['volume'].iloc[-20:-1].mean()
            current_volume = data['volume'].iloc[-1]
            if current_volume > avg_volume * 1.5:
                score *= 1.2
            elif current_volume < avg_volume * 0.7:
                score *= 0.8
        
        recent_high = data['high'].iloc[-20:].max()
        recent_low = data['low'].iloc[-20:].min()
        current_price = data['close'].iloc[-1]
        
        price_range = recent_high - recent_low
        if price_range > 0:
            distance_from_high = (recent_high - current_price) / price_range
            distance_from_low = (current_price - recent_low) / price_range
            
            if direction == "bullish" and distance_from_low < 0.2:
                score *= 1.3
            elif direction == "bearish" and distance_from_high < 0.2:
                score *= 1.3
        
        return score

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
                strength = abs(lows[last_peak_idx] - lows[prev_peak_idx]) / lows[prev_peak_idx]
                if strength > 0.02:
                    patterns.append("bullish_divergence_A")
                else:
                    patterns.append("bullish_divergence_B")

        price_high_peaks, _ = find_peaks(highs, distance=window//2)
        if len(price_high_peaks) >= 2:
            last_peak_idx = price_high_peaks[-1]
            prev_peak_idx = price_high_peaks[-2]
            
            if highs[last_peak_idx] < highs[prev_peak_idx] and indicator_vals[last_peak_idx] > indicator_vals[prev_peak_idx]:
                strength = abs(highs[last_peak_idx] - highs[prev_peak_idx]) / highs[prev_peak_idx]
                if strength > 0.02:
                    patterns.append("bearish_divergence_A")
                else:
                    patterns.append("bearish_divergence_B")
        
        if len(price_low_peaks) >= 2:
            last_peak_idx = price_low_peaks[-1]
            prev_peak_idx = price_low_peaks[-2]
            
            if lows[last_peak_idx] < lows[prev_peak_idx] and indicator_vals[last_peak_idx] < indicator_vals[prev_peak_idx]:
                patterns.append("hidden_bullish_divergence")

        if len(price_high_peaks) >= 2:
            last_peak_idx = price_high_peaks[-1]
            prev_peak_idx = price_high_peaks[-2]
            
            if highs[last_peak_idx] > highs[prev_peak_idx] and indicator_vals[last_peak_idx] > indicator_vals[prev_peak_idx]:
                patterns.append("hidden_bearish_divergence")
                
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

        volume_analysis = self.volume_analyzer.analyze_volume_pattern(data)
        volume_ratio = volume_analysis.get("volume_ratio", 1.0)
        volume_trend = "increasing" if volume_ratio > 1.1 else "decreasing"
        volume_confirmation = (trend == TrendDirection.BULLISH and volume_trend == "increasing") or \
                              (trend == TrendDirection.BEARISH and volume_trend == "increasing")

        volume_trend_score = None
        if volume_ratio > 1.5:
            volume_trend_score = 0.5 if trend == TrendDirection.BULLISH else -0.3
        elif volume_ratio > 1.2:
            volume_trend_score = 0.3 if trend == TrendDirection.BULLISH else -0.2
        elif volume_ratio < 0.8:
            volume_trend_score = -0.2 if trend == TrendDirection.BULLISH else 0.2
        else:
            volume_trend_score = 0.0

        rsi_values = talib.RSI(data["close"].to_numpy(), timeperiod=14)
        market_condition = MarketCondition.NEUTRAL
        if rsi_values.size > 0 and not pd.isna(rsi_values[-1]):
            last_rsi = rsi_values[-1]
            if last_rsi > 70:
                market_condition = MarketCondition.OVERBOUGHT
            elif last_rsi < 30:
                market_condition = MarketCondition.OVERSOLD

        support, resistance = self._calculate_support_resistance(data)
        trend_acceleration = self._calculate_trend_acceleration(data)

        return MarketAnalysis(
            trend=trend,
            trend_strength=trend_strength,
            volatility=volatility,
            momentum_score=momentum,
            hurst_exponent=hurst,
            volume_trend=volume_trend,
            support_levels=[support] if support else [],
            resistance_levels=[resistance] if resistance else [],
            market_condition=market_condition,
            trend_acceleration=trend_acceleration,
            volume_confirmation=volume_confirmation,
            volume_trend_score=volume_trend_score,
            adx=self._calculate_adx(data),
            **kwargs,
        )

    def _determine_trend(self, data: pd.DataFrame) -> TrendDirection:
        if len(data) < 50:
            return TrendDirection.SIDEWAYS
        sma_20 = data["close"].rolling(20).mean().iloc[-1]
        sma_50 = data["close"].rolling(50).mean().iloc[-1]
        if pd.isna(sma_20) or pd.isna(sma_50):
            return TrendDirection.SIDEWAYS
        if sma_20 > sma_50 * 1.01:
            return TrendDirection.BULLISH
        if sma_50 > sma_20 * 1.01:
            return TrendDirection.BEARISH
        return TrendDirection.SIDEWAYS

    def _calculate_trend_strength(self, data: pd.DataFrame) -> TrendStrength:
        if len(data) < 28:
            return TrendStrength.WEAK
        adx = talib.ADX(data["high"].to_numpy(), data["low"].to_numpy(), data["close"].to_numpy(), timeperiod=14)
        adx_val = adx[-1] if adx.size > 0 and not pd.isna(adx[-1]) else 0
        if adx_val > 25:
            return TrendStrength.STRONG
        if adx_val > 20:
            return TrendStrength.MODERATE
        return TrendStrength.WEAK

    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        if len(data) < 14:
            return 0.0
        atr = talib.ATR(data["high"].to_numpy(), data["low"].to_numpy(), data["close"].to_numpy(), timeperiod=14)
        atr_val = atr[-1] if atr.size > 0 and not pd.isna(atr[-1]) else 0.0
        price = data["close"].iloc[-1]
        return (atr_val / price) * 100 if price > 0 else 0.0

    def _calculate_momentum(self, data: pd.DataFrame) -> float:
        if len(data) < 10:
            return 0.0
        mom = talib.MOM(data["close"].to_numpy(), timeperiod=10)
        return mom[-1] if mom.size > 0 and not pd.isna(mom[-1]) else 0.0

    def _calculate_hurst_exponent(self, series: pd.Series, max_lag=100) -> Optional[float]:
        if len(series) < max_lag:
            return None
        series_np = series.to_numpy()
        lags = range(2, max_lag)
        try:
            tau = [
                np.sqrt(np.std(np.subtract(series_np[lag:], series_np[:-lag])))
                for lag in lags
            ]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0]
        except (ValueError, np.linalg.LinAlgError):
            return None

    def _calculate_support_resistance(self, data: pd.DataFrame, window: int = 50, prominence_factor: float = 0.02) -> Tuple[Optional[float], Optional[float]]:
        if len(data) < window:
            return (None, None)
        
        recent_data = data.tail(window)
        required_prominence = (recent_data['high'].max() - recent_data['low'].min()) * prominence_factor

        low_peaks, _ = find_peaks(-recent_data['low'].to_numpy(), prominence=required_prominence)
        high_peaks, _ = find_peaks(recent_data['high'].to_numpy(), prominence=required_prominence)

        support = None
        resistance = None

        if len(low_peaks) > 0:
            support_prices = recent_data['low'].iloc[low_peaks]
            recent_close = data['close'].iloc[-1]
            support_candidates = support_prices[support_prices < recent_close]
            if not support_candidates.empty:
                support = support_candidates.max()
            else:
                support = support_prices.mean()

        if len(high_peaks) > 0:
            resistance_prices = recent_data['high'].iloc[high_peaks]
            recent_close = data['close'].iloc[-1]
            resistance_candidates = resistance_prices[resistance_prices > recent_close]
            if not resistance_candidates.empty:
                resistance = resistance_candidates.min()
            else:
                resistance = resistance_prices.mean()
        
        return support, resistance

    def _calculate_trend_acceleration(self, data: pd.DataFrame, period: int = 10) -> float:
        if len(data) < period + 1:
            return 0.0
        mom = talib.MOM(data['close'].to_numpy(), timeperiod=period)
        if mom.size < 2 or pd.isna(mom[-1]) or pd.isna(mom[-2]):
            return 0.0
        accel = mom[-1] - mom[-2]
        return accel if not pd.isna(accel) else 0.0

    def _calculate_adx(self, data: pd.DataFrame) -> float:
        if len(data) < 28:
            return 25.0
        adx = talib.ADX(data["high"].to_numpy(), data["low"].to_numpy(), data["close"].to_numpy(), timeperiod=14)
        adx_val = adx[-1] if adx.size > 0 and not pd.isna(adx[-1]) else 25.0
        return adx_val


class VolumeAnalyzer:
    def analyze_volume_pattern(self, data: pd.DataFrame) -> Dict[str, float]:
        if len(data) < 20 or 'volume' not in data.columns or data['volume'].isnull().all():
            return {}
        volume_ma_20 = data["volume"].rolling(20).mean().iloc[-1]
        if pd.isna(volume_ma_20) or volume_ma_20 == 0:
            return {}
        current_volume = data["volume"].iloc[-1]
        ratio = current_volume / volume_ma_20
        return {"volume_ratio": ratio}