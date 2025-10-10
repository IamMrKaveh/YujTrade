# data_validator.py

from typing import Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from .logger_config import logger
from .constants import LONG_TERM_CONFIG


class DataQualityChecker:

    def __init__(self):
        self.min_data_points_map = LONG_TERM_CONFIG.get('min_data_points', {
            '1h': 500,
            '4h': 400,
            '1d': 200,
            '1w': 150,
            '1M': 100
        })

    def validate_data_quality(self, data: pd.DataFrame, timeframe: str) -> Tuple[bool, str]:
        if data is None or data.empty:
            return False, "Data is None or empty."

        min_required = self.min_data_points_map.get(timeframe, 200)
        if len(data) < min_required:
            return False, f"Insufficient data: {len(data)} < {min_required}"

        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"

        if data[required_cols].isnull().values.any():
            null_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            if null_ratio > 0.1:
                return False, f"Too many null values: {null_ratio:.2%}"

        if (data['high'] < data['low']).any():
            return False, "Invalid data: high is less than low in some rows."

        if (data['close'] <= 0).any() or (data['open'] <= 0).any():
            return False, "Invalid data: zero or negative prices found."

        price_changes = data['close'].pct_change().abs()
        if (price_changes > 0.75).any():
            logger.warning(f"Excessive volatility detected: price changed more than 75% in one candle.")

        if 'volume' in data.columns and data['volume'].mean() < 1:
            logger.warning(f"Average volume is extremely low: {data['volume'].mean():.2f}")

        if not self._check_data_freshness(data, timeframe):
            return False, f"Data is stale for timeframe {timeframe}"

        if self._detect_duplicate_timestamps(data):
            logger.warning("Duplicate timestamps detected in data")

        return True, "Data quality OK"

    def _check_data_freshness(self, data: pd.DataFrame, timeframe: str) -> bool:
        if not isinstance(data.index, pd.DatetimeIndex) or data.empty:
            return False
        
        last_timestamp = data.index[-1]
        current_time = datetime.now(last_timestamp.tz)
        
        freshness_thresholds = {
            '1h': timedelta(hours=4),
            '4h': timedelta(hours=16),
            '1d': timedelta(days=4),
            '1w': timedelta(days=28),
            '1M': timedelta(days=120)
        }
        
        threshold = freshness_thresholds.get(timeframe, timedelta(days=1))
        return current_time - last_timestamp <= threshold

    def _detect_duplicate_timestamps(self, data: pd.DataFrame) -> bool:
        if not isinstance(data.index, pd.DatetimeIndex):
            return False
        return data.index.duplicated().any()

    def calculate_overall_quality_score(self, data: pd.DataFrame, timeframe: str, volatility: float = None) -> float:
        score = 1.0
        
        is_valid, msg = self.validate_data_quality(data, timeframe)
        if not is_valid:
            score *= 0.8
        
        weighted_null_ratio = self._calculate_weighted_null_ratio(data)
        score *= (1 - weighted_null_ratio)
        
        has_gaps, gap_penalty = self.detect_data_gaps(data)
        if has_gaps:
             score *= (1 - gap_penalty * 0.5)
        
        if not self.check_sufficient_volume(data):
            score *= 0.9
        
        persistence_ok, persistence_penalty = self.check_trend_persistence(data, volatility=volatility)
        if not persistence_ok:
            score *= (1 - persistence_penalty * 0.5)
        
        price_volatility = data['close'].pct_change().std()
        if price_volatility > 0.1:
            score *= 0.9
        
        return np.clip(score, 0.0, 1.0)

    def _calculate_weighted_null_ratio(self, data: pd.DataFrame) -> float:
        if data.empty:
            return 1.0
        
        total_cells = len(data) * len(data.columns)
        if total_cells == 0:
            return 0.0
        
        null_counts = data.isnull().sum()
        weighted_nulls = 0
        
        for col in data.columns:
            weight = 2.0 if col in ['open', 'high', 'low', 'close'] else 0.5 if col == 'volume' else 1.0
            weighted_nulls += null_counts[col] * weight
        
        total_weight = sum(2.0 if col in ['open', 'high', 'low', 'close'] else 0.5 if col == 'volume' else 1.0 for col in data.columns)
        
        return weighted_nulls / (total_cells * total_weight / len(data.columns)) if total_weight > 0 else 0.0

    def check_trend_persistence(self, data: pd.DataFrame, min_bars: int = 30, volatility: float = None) -> Tuple[bool, float]:
        if len(data) < min_bars: 
            return False, 0.3

        closes = data['close'].tail(min_bars)
        sma_short = closes.rolling(window=min(10, min_bars // 3)).mean().dropna()
        if len(sma_short) < 2: 
            return False, 0.3

        direction_changes = np.sign(sma_short.diff().dropna()).diff().ne(0).sum()
        
        max_allowed_changes = self._get_volatility_adjusted_max_changes(min_bars, volatility)

        is_persistent = direction_changes <= max_allowed_changes
        penalty = min(direction_changes / (max_allowed_changes + 1), 0.3) if not is_persistent else 0.0
        
        if not is_persistent:
            logger.debug(f"Trend changed {int(direction_changes)} times in {min_bars} bars (max allowed: {max_allowed_changes})")

        return is_persistent, penalty

    def _get_volatility_adjusted_max_changes(self, min_bars: int, volatility: float) -> int:
        base_divisor = 8
        if volatility is not None:
            if volatility > 0.05: # High volatility
                base_divisor = 5
            elif volatility > 0.03: # Medium volatility
                base_divisor = 7
            else: # Low volatility
                base_divisor = 9
        
        return max(4, min_bars // base_divisor)

    def check_sufficient_volume(self, data: pd.DataFrame, min_ratio: float = 0.5) -> bool:
        if 'volume' not in data.columns or len(data) < 40: 
            return True

        recent_volume = data['volume'].tail(20).mean()
        historical_volume = data['volume'].iloc[:-20].mean()

        if historical_volume == 0: 
            return recent_volume > 0

        volume_ratio = recent_volume / historical_volume
        return volume_ratio >= min_ratio

    def _map_freq_str(self, freq_str: str) -> str:
        if freq_str is None:
            return None
        freq_lower = freq_str.lower()
        if 'w' in freq_lower:
            return '1W'
        if 'm' in freq_lower:
            return '1M'
        if 'd' in freq_lower:
            return '1D'
        return freq_str

    def detect_data_gaps(self, data: pd.DataFrame, max_gap_tolerance: int = 3) -> Tuple[bool, float]:
        if not isinstance(data.index, pd.DatetimeIndex) or len(data) < 2:
            return False, 0.0
        
        time_diffs = data.index.to_series().diff().dropna()
        if time_diffs.empty: 
            return False, 0.0
        
        inferred_freq = self._map_freq_str(pd.infer_freq(data.index))
        freq_str = self._map_freq_str(getattr(data.index, 'freqstr', None)) or inferred_freq

        try:
            expected_interval = pd.to_timedelta(freq_str)
        except (ValueError, TypeError):
            expected_interval = time_diffs.median()
        
        if expected_interval <= pd.Timedelta(0): 
            return False, 0.0

        large_gaps = time_diffs[time_diffs > expected_interval * (max_gap_tolerance + 1)]

        if not large_gaps.empty:
            gap_count = len(large_gaps)
            total_time = (data.index[-1] - data.index[0]).total_seconds()
            total_gap_time = large_gaps.sum().total_seconds()

            if total_time > 0:
                gap_ratio = total_gap_time / total_time
                if gap_ratio > 0.1:
                    raise ValueError("Data gap too large — aborting signal generation")

            penalty = min(0.05 * gap_count, 0.5)
            
            return True, penalty

        return False, 0.0