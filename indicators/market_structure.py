from logger_config import logger
import numpy as np
import pandas as pd
from .support_resistance import find_swing_points, get_recent_high, get_recent_low
from .cache_utils import cached_calculation


def _detect_breaks(structure_breaks, current_price, swing_highs, swing_lows):
	"""Detect bullish and bearish breaks"""
	# Check if current price breaks recent swing high (bullish break)
	recent_high = get_recent_high(swing_highs)
	if recent_high and current_price > recent_high:
		structure_breaks.loc[structure_breaks.index[-1], 'bullish_break'] = True
	
	# Check if current price breaks recent swing low (bearish break)
	recent_low = get_recent_low(swing_lows)
	if recent_low and current_price < recent_low:
		structure_breaks.loc[structure_breaks.index[-1], 'bearish_break'] = True


@cached_calculation('market_structure_breaks')
def _detect_market_structure_breaks(df, swing_strength=5):
	"""تشخیص Market Structure Breaks"""
	try:
		if df is None or len(df) < swing_strength * 2:
			return None
		
		high = df['high']
		low = df['low']
		
		structure_breaks = pd.DataFrame(index=df.index)
		structure_breaks['bullish_break'] = False
		structure_breaks['bearish_break'] = False
		
		# Find swing highs and lows
		swing_highs, swing_lows = find_swing_points(high, low, swing_strength)
		
		# Detect breaks
		current_price = df['close'].iloc[-1]
		_detect_breaks(structure_breaks, current_price, swing_highs, swing_lows)
		
		return structure_breaks
	except Exception as e:
		logger.error(f"Error detecting Market Structure Breaks: {e}")
		return None


@cached_calculation('market_structure_score')
def _calculate_market_structure_score(df, lookback=20):
	"""Calculate market structure quality score"""
	try:
		if df is None or len(df) < lookback:
			return 0
		
		recent_data = df.tail(lookback)
		highs = recent_data['high'].values
		lows = recent_data['low'].values
		closes = recent_data['close'].values
		
		# Higher highs and higher lows for uptrend
		higher_highs = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
		higher_lows = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i-1])
		
		# Lower highs and lower lows for downtrend
		lower_highs = sum(1 for i in range(1, len(highs)) if highs[i] < highs[i-1])
		lower_lows = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i-1])
		
		# Price momentum consistency
		up_moves = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i-1])
		down_moves = sum(1 for i in range(1, len(closes)) if closes[i] < closes[i-1])
		
		# Volume trend consistency
		if 'volume' in recent_data.columns:
			volumes = recent_data['volume'].values
			volume_trend = sum(1 for i in range(1, len(volumes)) if volumes[i] > volumes[i-1])
			volume_consistency = volume_trend / (len(volumes) - 1) if len(volumes) > 1 else 0.5
		else:
			volume_consistency = 0.5
		
		# Calculate structure strength
		uptrend_strength = (higher_highs + higher_lows) / (2 * (lookback - 1))
		downtrend_strength = (lower_highs + lower_lows) / (2 * (lookback - 1))
		
		# Momentum consistency
		momentum_consistency = max(up_moves, down_moves) / (len(closes) - 1) if len(closes) > 1 else 0.5
		
		# Final structure score
		if uptrend_strength > downtrend_strength:
			structure_score = (uptrend_strength * 0.4 + momentum_consistency * 0.4 + volume_consistency * 0.2) * 100
		else:
			structure_score = (downtrend_strength * 0.4 + momentum_consistency * 0.4 + volume_consistency * 0.2) * 100
		
		return min(structure_score, 100)
		
	except Exception:
		return 0


@cached_calculation('market_microstructure_internal')
def _calculate_market_microstructure_internal(df, period):
	"""Internal market microstructure calculation function"""
	try:
		if df is None or len(df) < period:
			return None
		
		high = df['high']
		low = df['low']
		close = df['close']
		volume = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)
		
		# Bid-Ask Spread Proxy
		spread_proxy = (high - low) / close
		avg_spread = spread_proxy.rolling(window=period).mean()
		
		# Market Depth Indicator
		price_impact = (high - low) / volume
		market_depth = price_impact.rolling(window=period).mean()
		
		# Order Flow Imbalance
		price_change = close.pct_change()
		volume_weighted_price_change = price_change * volume
		order_flow = volume_weighted_price_change.rolling(window=period).sum()
		
		# Liquidity Score
		liquidity_score = volume / (high - low)
		liquidity_score = liquidity_score.replace([np.inf, -np.inf], 0).fillna(0)
		avg_liquidity = liquidity_score.rolling(window=period).mean()
		
		return {
			'spread_proxy': avg_spread,
			'market_depth': market_depth,
			'order_flow': order_flow,
			'liquidity_score': avg_liquidity
		}
	except Exception as e:
		logger.error(f"Error calculating market microstructure: {e}")
		return None


def _calculate_market_microstructure(df, period=20):
	"""محاسبه ساختار میکرو بازار with caching"""
	return _calculate_market_microstructure_internal(df, period)


@cached_calculation('market_regime')
def _detect_market_regime(df, lookback=50):
	"""تشخیص رژیم بازار"""
	try:
		if df is None or len(df) < lookback:
			return None
			
		close = df['close']
		
		# محاسبه نوسانات
		returns = close.pct_change()
		volatility = returns.rolling(lookback).std() * np.sqrt(252)  # سالانه
		
		# محاسبه ترند
		sma_short = close.rolling(10).mean()
		sma_long = close.rolling(50).mean()
		trend = sma_short - sma_long
		
		# تعیین رژیم بازار
		regime = pd.Series(index=df.index, dtype=str)
		
		for i in range(lookback, len(df)):
			vol = volatility.iloc[i]
			tr = trend.iloc[i]
			
			if vol > volatility.rolling(lookback).quantile(0.75).iloc[i]:
				if tr > 0:
					regime.iloc[i] = 'Bull_Volatile'
				else:
					regime.iloc[i] = 'Bear_Volatile'
			else:
				if tr > 0:
					regime.iloc[i] = 'Bull_Stable'
				else:
					regime.iloc[i] = 'Bear_Stable'
		
		return regime
	except Exception:
		return None


@cached_calculation('market_structure_analysis')
def analyze_market_structure(df, swing_strength=5, lookback=20, microstructure_period=20, regime_lookback=50):
	"""
	تحلیل جامع ساختار بازار شامل تمام اجزای مهم
	
	Args:
		df (pd.DataFrame): داده‌های قیمت شامل ستون‌های high, low, close و volume (اختیاری)
		swing_strength (int): قدرت شناسایی نقاط چرخش
		lookback (int): دوره بررسی برای محاسبه امتیاز ساختاری
		microstructure_period (int): دوره محاسبه میکروساختار
		regime_lookback (int): دوره بررسی برای تشخیص رژیم بازار
	
	Returns:
		dict: نتایج تحلیل شامل تمام بخش‌های مختلف
	"""
	try:
		if df is None or len(df) < max(swing_strength * 2, lookback, microstructure_period, regime_lookback):
			logger.error("Insufficient data for market structure analysis")
			return None
		
		# تشخیص شکست‌های ساختاری
		structure_breaks = _detect_market_structure_breaks(df, swing_strength)
		
		# محاسبه امتیاز کیفیت ساختار
		structure_score = _calculate_market_structure_score(df, lookback)
		
		# محاسبه میکروساختار بازار
		microstructure = _calculate_market_microstructure(df, microstructure_period)
		
		# تشخیص رژیم بازار
		market_regime = _detect_market_regime(df, regime_lookback)
		
		# جمع‌آوری نتایج
		analysis_results = {
			'structure_breaks': structure_breaks,
			'structure_score': structure_score,
			'microstructure': microstructure,
			'market_regime': market_regime,
			'current_regime': market_regime.iloc[-1] if market_regime is not None and len(market_regime) > 0 else None,
			'last_bullish_break': structure_breaks['bullish_break'].iloc[-1] if structure_breaks is not None else False,
			'last_bearish_break': structure_breaks['bearish_break'].iloc[-1] if structure_breaks is not None else False,
			'analysis_timestamp': pd.Timestamp.now()
		}
		
		# اضافه کردن خلاصه وضعیت بازار
		market_summary = _generate_market_summary(analysis_results)
		analysis_results['market_summary'] = market_summary
		
		logger.info(f"Market structure analysis completed successfully. Score: {structure_score:.2f}")
		return analysis_results
		
	except Exception as e:
		logger.error(f"Error in market structure analysis: {e}")
		return None


def _generate_market_summary(analysis_results):
	"""تولید خلاصه وضعیت بازار بر اساس نتایج تحلیل"""
	try:
		summary = {
			'structure_quality': 'Unknown',
			'trend_direction': 'Unknown',
			'volatility_level': 'Unknown',
			'break_signal': 'None',
			'overall_sentiment': 'Neutral'
		}
		
		# ارزیابی کیفیت ساختار
		score = analysis_results.get('structure_score', 0)
		if score >= 70:
			summary['structure_quality'] = 'Strong'
		elif score >= 50:
			summary['structure_quality'] = 'Moderate'
		else:
			summary['structure_quality'] = 'Weak'
		
		# تعیین جهت ترند و سطح نوسانات
		current_regime = analysis_results.get('current_regime')
		if current_regime:
			if 'Bull' in current_regime:
				summary['trend_direction'] = 'Bullish'
				summary['overall_sentiment'] = 'Positive'
			elif 'Bear' in current_regime:
				summary['trend_direction'] = 'Bearish'
				summary['overall_sentiment'] = 'Negative'
			
			if 'Volatile' in current_regime:
				summary['volatility_level'] = 'High'
			else:
				summary['volatility_level'] = 'Low'
		
		# بررسی سیگنال‌های شکست
		if analysis_results.get('last_bullish_break'):
			summary['break_signal'] = 'Bullish Break'
			summary['overall_sentiment'] = 'Positive'
		elif analysis_results.get('last_bearish_break'):
			summary['break_signal'] = 'Bearish Break'
			summary['overall_sentiment'] = 'Negative'
		
		return summary
		
	except Exception as e:
		logger.error(f"Error generating market summary: {e}")
		return {
			'structure_quality': 'Unknown',
			'trend_direction': 'Unknown',
			'volatility_level': 'Unknown',
			'break_signal': 'None',
			'overall_sentiment': 'Neutral'
		}


@cached_calculation('market_structure_signals')
def get_market_structure_signals(df, sensitivity='medium'):
	"""
	دریافت سیگنال‌های ساده و کاربردی برای معاملات
	
	Args:
		df (pd.DataFrame): داده‌های قیمت
		sensitivity (str): حساسیت تحلیل ('low', 'medium', 'high')
	
	Returns:
		dict: سیگنال‌های ساده شامل buy/sell/hold
	"""
	try:
		# تنظیم پارامترها بر اساس حساسیت
		params = {
			'low': {'swing_strength': 8, 'lookback': 30, 'score_threshold': 60},
			'medium': {'swing_strength': 5, 'lookback': 20, 'score_threshold': 50},
			'high': {'swing_strength': 3, 'lookback': 15, 'score_threshold': 40}
		}
		
		config = params.get(sensitivity, params['medium'])
		
		# انجام تحلیل
		analysis = analyze_market_structure(
			df, 
			swing_strength=config['swing_strength'],
			lookback=config['lookback']
		)
		
		if not analysis:
			return {'signal': 'HOLD', 'confidence': 0, 'reason': 'Insufficient data'}
		
		# تولید سیگنال
		signal = 'HOLD'
		confidence = 0
		reason = 'No clear signal'
		
		score = analysis['structure_score']
		summary = analysis['market_summary']
		
		# سیگنال خرید
		if (analysis['last_bullish_break'] and 
			summary['trend_direction'] == 'Bullish' and 
			score >= config['score_threshold']):
			signal = 'BUY'
			confidence = min(score, 90)
			reason = 'Bullish break with strong structure'
		
		# سیگنال فروش
		elif (analysis['last_bearish_break'] and 
			  summary['trend_direction'] == 'Bearish' and 
			  score >= config['score_threshold']):
			signal = 'SELL'
			confidence = min(score, 90)
			reason = 'Bearish break with strong structure'
		
		# سیگنال نگهداری
		elif score < config['score_threshold']:
			signal = 'HOLD'
			confidence = 100 - score
			reason = 'Weak market structure'
		
		return {
			'signal': signal,
			'confidence': confidence,
			'reason': reason,
			'structure_score': score,
			'market_regime': analysis['current_regime'],
			'timestamp': pd.Timestamp.now()
		}
		
	except Exception as e:
		logger.error(f"Error generating market structure signals: {e}")
		return {'signal': 'HOLD', 'confidence': 0, 'reason': 'Analysis error'}

