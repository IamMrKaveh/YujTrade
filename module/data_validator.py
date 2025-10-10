from typing import Tuple
import pandas as pd
import numpy as np

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
            if null_ratio > 0.05:
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

        return True, "Data quality OK"

    def check_trend_persistence(self, data: pd.DataFrame, min_bars: int = 30) -> bool:
        if len(data) < min_bars: return False

        closes = data['close'].tail(min_bars)
        sma_short = closes.rolling(window=min(10, min_bars // 3)).mean().dropna()
        if len(sma_short) < 2: return False

        direction_changes = np.sign(sma_short.diff().dropna()).diff().ne(0).sum()
        max_allowed_changes = max(2, min_bars // 10)

        is_persistent = direction_changes <= max_allowed_changes
        if not is_persistent:
            logger.debug(f"Trend changed {int(direction_changes)} times in {min_bars} bars (max allowed: {max_allowed_changes})")

        return is_persistent

    def check_sufficient_volume(self, data: pd.DataFrame, min_ratio: float = 0.8) -> bool:
        if 'volume' not in data.columns or len(data) < 40: return True

        recent_volume = data['volume'].tail(20).mean()
        historical_volume = data['volume'].iloc[:-20].mean()

        if historical_volume == 0: return recent_volume > 0

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

    def detect_data_gaps(self, data: pd.DataFrame, max_gap_tolerance: int = 3) -> Tuple[bool, str]:
        if not isinstance(data.index, pd.DatetimeIndex) or len(data) < 2:
            return True, "Cannot detect gaps without a DatetimeIndex or sufficient data."

        time_diffs = data.index.to_series().diff().dropna()
        if time_diffs.empty: return True, "No gaps detected."
        
        inferred_freq = self._map_freq_str(pd.infer_freq(data.index))
        freq_str = self._map_freq_str(data.index.freqstr) or inferred_freq

        try:
            expected_interval = pd.to_timedelta(freq_str)
        except (ValueError, TypeError):
            expected_interval = time_diffs.median()
        
        if pd.isna(expected_interval):
            expected_interval = time_diffs.median()
        
        if expected_interval <= pd.Timedelta(0): return True, "Gaps cannot be checked, expected interval is zero or negative."

        large_gaps = time_diffs[time_diffs > expected_interval * max_gap_tolerance]

        if not large_gaps.empty:
            return False, f"Found {len(large_gaps)} large gaps, max gap: {large_gaps.max()}"

        return True, "No significant data gaps detected"