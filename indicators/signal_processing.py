import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from .volume_indicators import _check_volume_filter
from .cache_utils import cached_calculation, NUMBA_AVAILABLE, jit
from logger_config import logger


@cached_calculation("ensemble_signal_scoring")
def _ensemble_signal_scoring(df, signals_dict, weights=None):
	"""ترکیب چندین سیگنال با وزن‌دهی"""
	try:
		if not signals_dict:
			return 0
			
		if weights is None:
			weights = dict.fromkeys(signals_dict.keys(), 1)
		
		total_score = 0
		total_weight = 0
		
		for signal_name, signal_value in signals_dict.items():
			if signal_name in weights and signal_value is not None:
				weight = weights[signal_name]
				total_score += signal_value * weight
				total_weight += weight
		
		return total_score / total_weight if total_weight > 0 else 0
	except Exception as e:
		logger.error(f"Error in ensemble signal scoring: {e}")
		return 0
	
@cached_calculation("adaptive_threshold_calculator")
def _adaptive_threshold_calculator(df, indicator_values, percentile_low=20, percentile_high=80):
	"""محاسبه آستانه‌های تطبیقی با ابزارهای پیشرفته"""
	try:
		if df is None or indicator_values is None:
			return {'low': 30, 'high': 70}
			
		# استفاده از scipy برای محاسبات آماری پیشرفته
		clean_values = indicator_values.dropna()
		if len(clean_values) == 0:
			return {'low': 30, 'high': 70}
		
		# محاسبه آستانه‌ها با روش‌های مختلف
		low_threshold = np.percentile(clean_values, percentile_low)
		high_threshold = np.percentile(clean_values, percentile_high)
		
		# استفاده از Z-score برای تشخیص نقاط غیرعادی
		z_scores = np.abs(stats.zscore(clean_values))
		outlier_threshold = 2.0
		
		# تنظیم آستانه‌ها بر اساس نقاط غیرعادی
		if np.any(z_scores > outlier_threshold):
			median_val = np.median(clean_values)
			mad = stats.median_abs_deviation(clean_values)
			low_threshold = max(low_threshold, median_val - 2 * mad)
			high_threshold = min(high_threshold, median_val + 2 * mad)
		
		return {
			'low': low_threshold,
			'high': high_threshold,
			'median': np.median(clean_values),
			'std': np.std(clean_values)
		}
	except Exception:
		return {'low': 30, 'high': 70}

def _extract_signal_type(signal_data):
	"""Extract signal type from signal data"""
	if isinstance(signal_data, dict):
		return signal_data.get('type', 'neutral')
	elif isinstance(signal_data, str):
		return signal_data.lower()
	else:
		return 'neutral'

@cached_calculation("check_trend_filter")
def _check_trend_filter(df, signal_data, min_trend_strength):
	"""Check if trend strength supports the signal"""
	if len(df) < 10:
		return True
	
	recent_closes = df['close'].tail(10).values
	if len(recent_closes) == 0 or recent_closes[0] == 0:
		return True
	
	trend_strength = (recent_closes[-1] - recent_closes[0]) / recent_closes[0]
	signal_type = _extract_signal_type(signal_data)
	
	if (signal_type == 'buy' and trend_strength < -min_trend_strength) or (signal_type == 'sell' and trend_strength > min_trend_strength):
		return False
	
	return True

@cached_calculation("filter_false_signals")
def _filter_false_signals(df, signal_data, min_volume_ratio=1.2, min_trend_strength=0.1):
	"""فیلتر سیگنال‌های کاذب"""
	try:
		if df is None or not signal_data:
			return False
		
		if not _check_volume_filter(df, min_volume_ratio):
			return False
		
		if not _check_trend_filter(df, signal_data, min_trend_strength):
			return False
		
		return True
	except Exception:
		return True


