from typing import Tuple
import pandas as pd
import numpy as np

from module.logger_config import logger
from module.constants import LONG_TERM_CONFIG


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
        min_required = self.min_data_points_map.get(timeframe, 200)
        if len(data) < min_required:
            return False, f"Insufficient data: {len(data)} < {min_required}"
        
        if 'volume' in data.columns:
            avg_volume = data['volume'].mean()
            if avg_volume < 1:
                return False, f"Volume too low: {avg_volume:.2f}"
        
        if 'close' in data.columns:
            price_changes = data['close'].pct_change().dropna()
            if len(price_changes) > 0:
                price_volatility = price_changes.std()
                if price_volatility > 0.5:
                    return False, f"Excessive volatility: {price_volatility:.2%}"
        
        null_count = data.isnull().sum().sum()
        total_cells = len(data) * len(data.columns)
        if total_cells > 0:
            null_ratio = null_count / total_cells
            if null_ratio > 0.05:
                return False, f"Too many null values: {null_ratio:.2%}"
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
        
        if 'high' in data.columns and 'low' in data.columns:
            invalid_hl = (data['high'] < data['low']).sum()
            if invalid_hl > 0:
                return False, f"Found {invalid_hl} bars with high < low"
        
        if 'close' in data.columns:
            zero_prices = (data['close'] <= 0).sum()
            if zero_prices > 0:
                return False, f"Found {zero_prices} bars with zero or negative prices"
        
        return True, "Data quality OK"
    
    def check_trend_persistence(self, data: pd.DataFrame, min_bars: int = 30) -> bool:
        
        if len(data) < min_bars:
            return False
        
        if 'close' not in data.columns:
            return False
        
        closes = data['close'].tail(min_bars)
        if len(closes) < min_bars:
            return False
        
        sma_short = closes.rolling(min(10, min_bars // 3)).mean()
        
        if sma_short.isnull().all():
            return False
        
        direction_changes = (sma_short.diff().fillna(0) > 0).astype(int).diff().abs()
        total_changes = direction_changes.sum()
        
        max_allowed_changes = 3
        is_persistent = total_changes <= max_allowed_changes
        
        if not is_persistent:
            logger.debug(f"Trend changed {int(total_changes)} times in {min_bars} bars (max allowed: {max_allowed_changes})")
        
        return is_persistent
    
    def check_sufficient_volume(self, data: pd.DataFrame, min_ratio: float = 1.0) -> bool:
        
        if 'volume' not in data.columns or len(data) < 20:
            return True
        
        recent_volume = data['volume'].tail(20).mean()
        historical_volume = data['volume'].head(len(data) - 20).mean()
        
        if historical_volume == 0:
            return recent_volume > 0
        
        volume_ratio = recent_volume / historical_volume
        
        return volume_ratio >= min_ratio
    
    def detect_data_gaps(self, data: pd.DataFrame, max_gap_tolerance: int = 5) -> Tuple[bool, str]:
        
        if not isinstance(data.index, pd.DatetimeIndex):
            return True, "Index is not DatetimeIndex, skipping gap detection"
        
        if len(data) < 2:
            return True, "Not enough data to detect gaps"
        
        time_diffs = data.index.to_series().diff().dropna()
        
        if len(time_diffs) == 0:
            return True, "No time differences to analyze"
        
        median_diff = time_diffs.median()
        
        large_gaps = time_diffs[time_diffs > median_diff * max_gap_tolerance]
        
        if len(large_gaps) > 0:
            gap_count = len(large_gaps)
            max_gap = large_gaps.max()
            return False, f"Found {gap_count} large gaps, max gap: {max_gap}"
        
        return True, "No significant data gaps detected"
    
