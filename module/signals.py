import asyncio
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import redis.asyncio as redis

from module.analyzers import MarketConditionAnalyzer, PatternAnalyzer
from module.constants import MULTI_TF_CONFIRMATION_MAP, MULTI_TF_CONFIRMATION_WEIGHTS, DEFAULT_INDICATOR_WEIGHTS
from module.core import (
    BinanceFuturesData, DerivativesAnalysis, DynamicLevels, FundamentalAnalysis,
    IndicatorResult, MacroEconomicData, MarketAnalysis, OnChainAnalysis, OrderBook,
    SignalType, TradingSignal, TrendDirection, TrendStrength, TrendingData
)
from module.data_sources import BinanceFetcher, MarketIndicesFetcher, MessariFetcher, NewsFetcher
from module.exceptions import DataError, InsufficientDataError
from module.indicators import get_all_indicators
from module.logger_config import logger
from module.models import ModelManager
from module.market import MarketDataProvider
from module.resource_manager import managed_tf_session


class MultiTimeframeAnalyzer:
    def __init__(self, market_data_provider: MarketDataProvider, indicators, redis_client: Optional[redis.Redis] = None, cache_ttl=600):
        self.market_data_provider = market_data_provider
        self.indicators = indicators
        self.redis = redis_client
        self.cache_ttl = cache_ttl
        self._lock = asyncio.Lock()

    async def _get_cache(self, key: str) -> Optional[Dict]:
        if not self.redis:
            return None
        try:
            cached_data = await self.redis.get(key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Redis GET failed for {key}: {e}")
        return None

    async def _set_cache(self, key: str, value: Dict):
        if not self.redis:
            return
        try:
            await self.redis.set(key, json.dumps(value), ex=self.cache_ttl)
        except Exception as e:
            logger.warning(f"Redis SET failed for {key}: {e}")

    async def fetch_and_analyze_batch(self, symbol: str, timeframes: List[str]) -> Dict[str, Dict[str, str]]:
        tasks = {tf: self.market_data_provider.fetch_ohlcv_data(symbol, tf) for tf in timeframes}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        analyses = {}
        for tf, result in zip(tasks.keys(), results):
            cache_key = f"mtf_analysis:{symbol}:{tf}"
            if isinstance(result, pd.DataFrame) and not result.empty:
                analysis = self._analyze_indicators(result)
                analyses[tf] = analysis
                await self._set_cache(cache_key, analysis)
            elif isinstance(result, Exception):
                logger.warning(f"Failed to fetch data for {symbol}-{tf} in batch: {result}")
        return analyses

    async def is_direction_aligned(self, symbol: str, exec_tf: str, threshold: float = 0.7) -> bool:
        try:
            confirm_tfs = MULTI_TF_CONFIRMATION_MAP.get(exec_tf, [])
            if not confirm_tfs:
                logger.info(f"No confirmation timeframes for {exec_tf}, alignment approved by default")
                return True

            all_needed_tfs = [exec_tf] + confirm_tfs
            
            cached_analyses = {}
            tfs_to_fetch = []

            for tf in all_needed_tfs:
                cache_key = f"mtf_analysis:{symbol}:{tf}"
                cached = await self._get_cache(cache_key)
                if cached:
                    cached_analyses[tf] = cached
                else:
                    tfs_to_fetch.append(tf)
            
            if tfs_to_fetch:
                new_analyses = await self.fetch_and_analyze_batch(symbol, tfs_to_fetch)
                cached_analyses.update(new_analyses)

            base_signals = cached_analyses.get(exec_tf)
            if not base_signals:
                logger.warning(f"No base signals for {exec_tf}, alignment failed")
                return False

            total_score, total_weight = 0.0, 0.0
            for tf in confirm_tfs:
                confirm_signals = cached_analyses.get(tf)
                if confirm_signals:
                    weight = MULTI_TF_CONFIRMATION_WEIGHTS.get(exec_tf, {}).get(tf, 1.0)
                    matches = sum(1 for k in base_signals
                                if k in confirm_signals and self._signals_match(base_signals[k], confirm_signals[k]))
                    score = matches / len(base_signals) if base_signals else 0
                    total_score += score * weight
                    total_weight += weight
                    logger.debug(f"MTF {tf} alignment score: {score:.2f}, weight: {weight}")
            
            if total_weight == 0:
                logger.warning(f"Total weight is zero for {symbol}-{exec_tf}")
                return False
            
            avg_score = total_score / total_weight
            is_aligned = avg_score >= threshold
            logger.info(f"MTF alignment for {symbol}-{exec_tf}: {avg_score:.2f} (threshold: {threshold}), aligned: {is_aligned}")
            return is_aligned
        except Exception as e:
            logger.error(f"Error in multi-timeframe analysis for {symbol}: {e}", exc_info=True)
            return False

    def _signals_match(self, signal1: str, signal2: str) -> bool:
        bullish_signals = ['bullish', 'oversold', 'above', 'buy', 'up', 'accumulation', 'bull_power', 'upward']
        bearish_signals = ['bearish', 'overbought', 'below', 'sell', 'down', 'distribution', 'bear_power', 'downward']

        s1_lower = signal1.lower()
        s2_lower = signal2.lower()

        signal1_is_bullish = any(p in s1_lower for p in bullish_signals)
        signal2_is_bullish = any(p in s2_lower for p in bullish_signals)
        signal1_is_bearish = any(p in s1_lower for p in bearish_signals)
        signal2_is_bearish = any(p in s2_lower for p in bearish_signals)

        return (signal1_is_bullish and signal2_is_bullish) or (signal1_is_bearish and signal2_is_bearish)

    def _analyze_indicators(self, df: pd.DataFrame) -> Dict[str, str]:
        signals = {}
        if df.empty or not self.indicators:
            return signals

        for name, ind in self.indicators.items():
            try:
                result = ind.calculate(df)
                if result and hasattr(result, 'interpretation') and result.interpretation:
                    signals[name] = result.interpretation
            except Exception as e:
                logger.warning(f"Error calculating {name}: {e}")
        return signals


class SignalGenerator:
    def __init__(self, market_data_provider: MarketDataProvider, 
                news_fetcher: Optional[NewsFetcher] = None,
                market_indices_fetcher: Optional[MarketIndicesFetcher] = None,
                model_manager: Optional[ModelManager] = None,
                multi_tf_analyzer: Optional[MultiTimeframeAnalyzer] = None,
                config: Optional[Dict] = None,
                binance_fetcher: Optional[BinanceFetcher] = None,
                messari_fetcher: Optional[MessariFetcher] = None):
        self.market_data_provider = market_data_provider
        self.news_fetcher = news_fetcher
        self.market_indices_fetcher = market_indices_fetcher
        self.model_manager = model_manager
        self.multi_tf_analyzer = multi_tf_analyzer
        self.config = config or {'min_confidence_score': 75}
        self.binance_fetcher = binance_fetcher
        self.messari_fetcher = messari_fetcher
        self.indicators = get_all_indicators()
        self.market_analyzer = MarketConditionAnalyzer()
        self.indicator_weights = self.config.get('indicator_weights', DEFAULT_INDICATOR_WEIGHTS)

    def _safe_dataframe(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        if df is None:
            return pd.DataFrame()
        
        if isinstance(df, pd.Series):
            logger.error("Received Series instead of DataFrame, converting to DataFrame")
            df = df.to_frame()
        
        if not isinstance(df, pd.DataFrame):
            logger.error(f"Received unexpected type: {type(df)}, expected DataFrame")
            return pd.DataFrame()
        
        if df.empty:
            return pd.DataFrame()
        
        try:
            df_copy = df.copy()
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            
            for col in numeric_cols:
                if col in df_copy.columns:
                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
            
            df_copy = df_copy.replace([np.inf, -np.inf], np.nan)
            
            df_copy = df_copy.dropna(subset=[c for c in numeric_cols if c in df_copy.columns], how='any')
            
            for col in ['open', 'high', 'low', 'close']:
                if col in df_copy.columns:
                    df_copy = df_copy[df_copy[col] > 0]
            
            if 'volume' in df_copy.columns:
                df_copy = df_copy[df_copy['volume'] >= 0]
            
            if 'high' in df_copy.columns and 'low' in df_copy.columns:
                df_copy = df_copy[df_copy['high'] >= df_copy['low']]
            
            if not isinstance(df_copy.index, pd.DatetimeIndex):
                df_copy = df_copy.reset_index(drop=True)
                
            return df_copy
        except Exception as e:
            logger.error(f"Error in _safe_dataframe: {e}")
            return pd.DataFrame()

    def calculate_dynamic_levels(self, data: pd.DataFrame, market_analysis: MarketAnalysis, signal_type: SignalType) -> DynamicLevels:
        last_close = data['close'].iloc[-1]
        
        atr = market_analysis.volatility / 100 * last_close if market_analysis.volatility > 0 else data['close'].pct_change().std() * last_close
        
        if pd.isna(atr) or atr == 0 or (isinstance(atr, (int, float)) and atr == 0):
            atr = last_close * 0.01

        if signal_type == SignalType.BUY:
            primary_entry = float(last_close)
            secondary_entry = float(last_close - 0.5 * atr)
            primary_exit = float(last_close + 2 * atr)
            secondary_exit = float(last_close + 3.5 * atr)
            tight_stop = float(primary_entry - 1.2 * atr)
            wide_stop = float(primary_entry - 2.0 * atr)
            breakeven_point = float(primary_entry + 0.2 * atr)
        else:
            primary_entry = float(last_close)
            secondary_entry = float(last_close + 0.5 * atr)
            primary_exit = float(last_close - 2 * atr)
            secondary_exit = float(last_close - 3.5 * atr)
            tight_stop = float(primary_entry + 1.2 * atr)
            wide_stop = float(primary_entry + 2.0 * atr)
            breakeven_point = float(primary_entry - 0.2 * atr)

        return DynamicLevels(
            primary_entry=primary_entry,
            secondary_entry=secondary_entry,
            primary_exit=primary_exit,
            secondary_exit=secondary_exit,
            tight_stop=tight_stop,
            wide_stop=wide_stop,
            breakeven_point=breakeven_point,
            trailing_stop=float(atr * 0.7)
        )

    def _calculate_confidence_score(self, symbol: str, signal_type: SignalType, indicators: Dict[str, IndicatorResult], 
                                  market_analysis: MarketAnalysis, patterns: List, context_data: Dict) -> Tuple[float, List[str]]:
        score, total_weight = 0.0, 0.0
        reasons = []
        is_buy = signal_type == SignalType.BUY

        def add_reason(source: str, interpretation: str, points: float, strength: float = -1.0):
            strength_str = f", strength: {strength:.0f}%" if strength >= 0 else ""
            reasons.append(f"{source}: {interpretation}{strength_str} -> {points:+.1f} pts")

        bullish_keywords = ['bullish', 'oversold', 'above', 'buy', 'up', 'accumulation', 'bull_power', 
                           'upward', 'easy_upward_move', 'strong_uptrend', 'positive_momentum', 
                           'buyers_control', 'strong_bulls', 'breakout', 'trending_market', 'breakout_above',
                           'price_above', 'at_lower_channel', 'support_level', 'smart_money_buying', 'crowd_buying']
        
        bearish_keywords = ['bearish', 'overbought', 'below', 'sell', 'down', 'distribution', 'bear_power',
                           'downward', 'easy_downward_move', 'strong_downtrend', 'negative_momentum',
                           'sellers_control', 'strong_bears', 'breakdown', 'reversal_warning', 'breakdown_below',
                           'price_below', 'at_upper_channel', 'resistance_level', 'smart_money_selling', 'crowd_selling']

        for name, res in indicators.items():
            if not res or not hasattr(res, 'interpretation') or not res.interpretation:
                continue
            
            weight_key = name.split('_')[0].lower()
            weight = self.indicator_weights.get(weight_key, 0)

            if weight == 0:
                continue
            
            total_weight += weight
            points = 0
            
            interp_lower = res.interpretation.lower()
            is_bullish = any(s in interp_lower for s in bullish_keywords)
            is_bearish = any(s in interp_lower for s in bearish_keywords)
            
            signal_strength = res.signal_strength if hasattr(res, 'signal_strength') and res.signal_strength is not None else 50.0
            
            if (is_buy and is_bullish) or (not is_buy and is_bearish):
                points = (signal_strength / 100) * weight
            elif (is_buy and is_bearish) or (not is_buy and is_bullish):
                points = - (signal_strength / 100) * weight * 1.2
            
            if points != 0:
                score += points
                add_reason(res.name, res.interpretation, points, signal_strength)

        trend_weight = self.indicator_weights.get('trend', 15)
        total_weight += trend_weight
        if (is_buy and market_analysis.trend == TrendDirection.BULLISH) or (not is_buy and market_analysis.trend == TrendDirection.BEARISH):
            adx_res = indicators.get('adx')
            adx_strength = adx_res.signal_strength / 100 if adx_res and hasattr(adx_res, 'signal_strength') else 0.5
            points = trend_weight * (0.5 + 0.5 * adx_strength)
            score += points
            add_reason("Trend Align", f"{market_analysis.trend.value} ({market_analysis.trend_strength.value})", points)
        elif market_analysis.trend != TrendDirection.SIDEWAYS:
            points = -trend_weight * 0.75
            score += points
            add_reason("Trend Misalign", f"{market_analysis.trend.value}", points)

        pattern_weight = self.indicator_weights.get('pattern', 10)
        for p in patterns:
            total_weight += pattern_weight
            if (is_buy and 'bullish' in p) or (not is_buy and 'bearish' in p):
                score += pattern_weight
                add_reason("Pattern", p, pattern_weight)
        
        multi_tf_weight = self.indicator_weights.get('multi_tf', 10)
        if 'multi_tf_confirmation' in context_data:
            total_weight += multi_tf_weight
            if context_data['multi_tf_confirmation']:
                score += multi_tf_weight
                add_reason("Multi-TF", "Confirmation aligned", multi_tf_weight)
            else:
                score -= multi_tf_weight
                add_reason("Multi-TF", "Confirmation misaligned", -multi_tf_weight)

        derivatives = context_data.get('derivatives')
        if derivatives:
            if derivatives.funding_rate:
                weight = self.indicator_weights.get('funding_rate', 9)
                total_weight += weight
                fr = derivatives.funding_rate
                if fr > 0.0005:
                    points = weight * min(fr / 0.001, 1.0)
                    score += -points if is_buy else points
                    add_reason("Funding", f"High Positive ({fr:.4f})", -points if is_buy else points)
                elif fr < -0.0005:
                    points = weight * min(abs(fr) / 0.001, 1.0)
                    score += points if is_buy else -points
                    add_reason("Funding", f"High Negative ({fr:.4f})", points if is_buy else -points)
            
            if derivatives.taker_long_short_ratio:
                weight = self.indicator_weights.get('taker_ratio', 10)
                total_weight += weight
                ratio = derivatives.taker_long_short_ratio
                if ratio > 1.1:
                    points = weight * min((ratio - 1.0) * 2, 1.0)
                    score += points if is_buy else -points*0.5
                    add_reason("Taker L/S", f"Longs Dominate ({ratio:.2f})", points if is_buy else -points*0.5)
                elif ratio < 0.9:
                    points = weight * min((1.0 - ratio) * 2, 1.0)
                    score += -points*0.5 if is_buy else points
                    add_reason("Taker L/S", f"Shorts Dominate ({ratio:.2f})", -points*0.5 if is_buy else points)

            if derivatives.binance_futures_data and derivatives.binance_futures_data.top_trader_long_short_ratio_accounts:
                bfd = derivatives.binance_futures_data
                weight = self.indicator_weights.get('top_trader_sentiment', 12)
                total_weight += weight
                
                ratios = [r for r in [bfd.top_trader_long_short_ratio_accounts, bfd.top_trader_long_short_ratio_positions] if r is not None]
                if ratios:
                    avg_ratio = sum(ratios) / len(ratios)
                    if avg_ratio > 1.2:
                        points = weight * min((avg_ratio - 1.0) * 1.5, 1.0)
                        score += -points if is_buy else points
                        add_reason("TopTraders", f"Contrarian Bullish ({avg_ratio:.2f})", -points if is_buy else points)
                    elif avg_ratio < 0.8:
                        points = weight * min((1.0 - avg_ratio) * 1.5, 1.0)
                        score += points if is_buy else -points
                        add_reason("TopTraders", f"Contrarian Bearish ({avg_ratio:.2f})", points if is_buy else -points)

        news_score = context_data.get('news_score', 0)
        news_weight = self.indicator_weights.get('trending', 7)
        if news_score != 0:
            total_weight += news_weight
            points = (news_score / 10) * news_weight
            score += points
            add_reason("News Sentiment", f"Score: {news_score}", points)

        if total_weight > 0:
            final_score = (score / total_weight) * 100
        else:
            final_score = 0
            
        final_score = max(0, min(100, final_score))
        return final_score, sorted(reasons, key=lambda x: abs(float(x.split('->')[-1].replace('pts', '').strip())), reverse=True)


    def _determine_signal_type(self, indicators: Dict[str, IndicatorResult], market_analysis: MarketAnalysis) -> SignalType:
        bullish_score = 0
        bearish_score = 0

        bullish_keywords = ['bullish', 'oversold', 'above', 'buy', 'up', 'accumulation', 'bull_power', 'upward', 'breakout']
        bearish_keywords = ['bearish', 'overbought', 'below', 'sell', 'down', 'distribution', 'bear_power', 'downward', 'breakdown']

        for name, res in indicators.items():
            if not res: continue
            interp = res.interpretation.lower()
            if any(k in interp for k in bullish_keywords):
                bullish_score += 1
            elif any(k in interp for k in bearish_keywords):
                bearish_score += 1
        
        if market_analysis.trend == TrendDirection.BULLISH:
            bullish_score += 2
        elif market_analysis.trend == TrendDirection.BEARISH:
            bearish_score += 2

        if bullish_score > bearish_score + 1:
            return SignalType.BUY
        elif bearish_score > bullish_score + 1:
            return SignalType.SELL
        else:
            return SignalType.HOLD

    async def _gather_context_data(self, symbol: str, timeframe: str) -> Dict:
        context_data = {}
        coin_id = symbol.split('/')[0].lower()

        tasks = {
            'news': self.news_fetcher.fetch_news([symbol.split('/')[0]]) if self.news_fetcher else asyncio.sleep(0),
            'fear_greed': self.news_fetcher.fetch_fear_greed() if self.news_fetcher else asyncio.sleep(0),
            'derivatives': self._get_derivatives_data(symbol),
            'multi_tf': self.multi_tf_analyzer.is_direction_aligned(symbol, timeframe) if self.multi_tf_analyzer else asyncio.sleep(0),
            'fundamental': self.market_indices_fetcher.coingecko.get_fundamental_data(coin_id) if self.market_indices_fetcher else asyncio.sleep(0),
            'on_chain': self.messari_fetcher.get_on_chain_data(symbol) if self.messari_fetcher else asyncio.sleep(0),
            'order_book': self.binance_fetcher.get_order_book_depth(symbol) if self.binance_fetcher else asyncio.sleep(0),
            'macro_data': self.market_indices_fetcher.yfinance.get_traditional_indices() if self.market_indices_fetcher else asyncio.sleep(0),
            'trending_data': self.market_indices_fetcher.coingecko.get_trending_searches() if self.market_indices_fetcher else asyncio.sleep(0)
        }
        
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        news, fear_greed, derivatives, mtf, fundamental, on_chain, order_book, macro, trending = results

        if not isinstance(news, Exception) and news:
            context_data['news_score'] = self.news_fetcher.score_news(news)
        if not isinstance(fear_greed, Exception) and fear_greed:
            context_data['fear_greed'] = fear_greed[0]
        if not isinstance(derivatives, Exception):
            context_data['derivatives'] = derivatives
        if not isinstance(mtf, Exception):
            context_data['multi_tf_confirmation'] = mtf
        if not isinstance(fundamental, Exception):
            context_data['fundamental'] = fundamental
        if not isinstance(on_chain, Exception):
            context_data['on_chain'] = on_chain
        if not isinstance(order_book, Exception):
            context_data['order_book'] = order_book
        if not isinstance(macro, Exception) and macro:
            context_data['macro_data'] = MacroEconomicData(
                cpi=macro.get('CPI'),
                fed_rate=macro.get('FED_RATE'),
                treasury_yield_10y=macro.get('TREASURY_YIELD_10Y'),
                gdp=macro.get('GDP'),
                unemployment=macro.get('UNEMPLOYMENT')
            )
        if not isinstance(trending, Exception) and trending:
            context_data['trending_data'] = TrendingData(coingecko_trending=trending)
            
        return context_data

    async def _get_derivatives_data(self, symbol: str) -> Optional[DerivativesAnalysis]:
        if not self.binance_fetcher: 
            return None
        
        tasks = {
            'open_interest': self.binance_fetcher.get_open_interest(symbol),
            'funding_rate': self.binance_fetcher.get_funding_rate(symbol),
            'taker_ratio': self.binance_fetcher.get_taker_long_short_ratio(symbol),
            'top_trader_acc': self.binance_fetcher.get_top_trader_long_short_ratio_accounts(symbol),
            'top_trader_pos': self.binance_fetcher.get_top_trader_long_short_ratio_positions(symbol),
            'liquidation_orders': self.binance_fetcher.get_liquidation_orders(symbol),
            'mark_price': self.binance_fetcher.get_mark_price(symbol),
            'coingecko_derivs': self.market_indices_fetcher.coingecko.get_derivatives() if self.market_indices_fetcher else asyncio.sleep(0)
        }
        
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        oi, fr, tr, tta, ttp, liq, mp, cgd = results

        binance_futures = BinanceFuturesData(
            top_trader_long_short_ratio_accounts=tta if not isinstance(tta, Exception) else None,
            top_trader_long_short_ratio_positions=ttp if not isinstance(ttp, Exception) else None,
            liquidation_orders=liq if not isinstance(liq, Exception) and liq else [],
            mark_price=mp if not isinstance(mp, Exception) else None
        )

        return DerivativesAnalysis(
            open_interest=oi if not isinstance(oi, Exception) else None,
            funding_rate=fr if not isinstance(fr, Exception) else None,
            taker_long_short_ratio=tr if not isinstance(tr, Exception) else None,
            coingecko_derivatives=cgd if not isinstance(cgd, Exception) and cgd else [],
            binance_futures_data=binance_futures
        )

    async def generate_signals(self, data: pd.DataFrame, symbol: str, timeframe: str) -> List[TradingSignal]:
        safe_data = self._safe_dataframe(data)
        if len(safe_data) < 50:
            logger.warning(f"Not enough data for {symbol} on {timeframe} after cleaning ({len(safe_data)} rows)")
            return []

        start_time = time.time()
        
        indicator_results = {}
        for name, ind in self.indicators.items():
            try:
                result = ind.calculate(safe_data)
                indicator_results[name] = result
            except Exception as e:
                logger.warning(f"Error calculating {name} for {symbol}: {e}")
                indicator_results[name] = None
        
        market_analysis = self.market_analyzer.analyze_market_condition(safe_data)
        patterns = PatternAnalyzer.detect_patterns(safe_data)
        
        signal_type = self._determine_signal_type(indicator_results, market_analysis)
        if signal_type == SignalType.HOLD:
            return []

        context_data = await self._gather_context_data(symbol, timeframe)
        
        confidence, reasons = self._calculate_confidence_score(symbol, signal_type, indicator_results, market_analysis, patterns, context_data)
        
        if self.model_manager:
            async with managed_tf_session():
                lstm_pred_task = self.model_manager.predict('lstm', symbol, timeframe, safe_data)
                xgb_pred_task = self.model_manager.predict('xgboost', symbol, timeframe, safe_data)
                lstm_pred, xgb_pred = await asyncio.gather(lstm_pred_task, xgb_pred_task)
            
            last_close = safe_data['close'].iloc[-1]
            if lstm_pred is not None and lstm_pred.size > 0:
                pred_change = (lstm_pred[0] - last_close) / last_close
                weight = self.indicator_weights.get('lstm', 15)
                if (signal_type == SignalType.BUY and pred_change > 0.001) or (signal_type == SignalType.SELL and pred_change < -0.001):
                    confidence = (confidence + 100) / 2
                    reasons.insert(0, f"LSTM Prediction Confirms: Change {pred_change:.2%}")
                else:
                    confidence *= 0.9
                    reasons.append(f"LSTM Prediction Contradicts: Change {pred_change:.2%}")

        if confidence < self.config.get('min_confidence_score', 0):
            return []

        dynamic_levels = self.calculate_dynamic_levels(safe_data, market_analysis, signal_type)
        risk_reward = abs(dynamic_levels.primary_exit - dynamic_levels.primary_entry) / abs(dynamic_levels.primary_entry - dynamic_levels.tight_stop) if (dynamic_levels.primary_entry - dynamic_levels.tight_stop) != 0 else 0
        
        volume_analysis = self.market_analyzer.volume_analyzer.analyze_volume_pattern(safe_data)

        signal = TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            entry_price=dynamic_levels.primary_entry,
            exit_price=dynamic_levels.primary_exit,
            stop_loss=dynamic_levels.tight_stop,
            timestamp=datetime.now(),
            timeframe=timeframe,
            confidence_score=confidence,
            reasons=reasons,
            risk_reward_ratio=risk_reward,
            predicted_profit=(dynamic_levels.primary_exit - dynamic_levels.primary_entry) if signal_type == SignalType.BUY else (dynamic_levels.primary_entry - dynamic_levels.primary_exit),
            volume_analysis=volume_analysis,
            market_context={
                'trend': market_analysis.trend.value,
                'trend_strength': market_analysis.trend_strength.value,
                'volatility': market_analysis.volatility,
                'market_condition': market_analysis.market_condition.value,
                'support': market_analysis.support_levels[0],
                'resistance': market_analysis.resistance_levels[0]
            },
            dynamic_levels=dynamic_levels.__dict__,
            fundamental_analysis=context_data.get('fundamental'),
            on_chain_analysis=context_data.get('on_chain'),
            derivatives_analysis=context_data.get('derivatives'),
            order_book=context_data.get('order_book'),
            macro_data=context_data.get('macro_data'),
            trending_data=context_data.get('trending_data')
        )

        logger.info(f"Generated {signal.signal_type.value} signal for {symbol} on {timeframe} with confidence {signal.confidence_score:.2f} in {time.time() - start_time:.2f}s")
        return [signal]


class SignalRanking:
    @staticmethod
    def rank_signals(signals: List[TradingSignal]) -> List[TradingSignal]:
        def signal_score(signal: TradingSignal) -> float:
            base_score = signal.confidence_score
            
            rr = signal.risk_reward_ratio or 0
            rr_bonus = 0
            if rr >= 3.0:
                rr_bonus = 15
            elif rr >= 2.0:
                rr_bonus = 10
            elif rr >= 1.5:
                rr_bonus = 5
            
            trend_bonus = 0
            if signal.market_context:
                mc = signal.market_context
                trend_val_str = mc.get('trend')
                trend_val = TrendDirection(trend_val_str) if isinstance(trend_val_str, str) else trend_val_str
                
                is_aligned = (signal.signal_type == SignalType.BUY and trend_val == TrendDirection.BULLISH) or \
                                (signal.signal_type == SignalType.SELL and trend_val == TrendDirection.BEARISH)
                
                if is_aligned:
                    trend_bonus = 10
                    strength_val_str = mc.get('trend_strength')
                    strength_val = TrendStrength(strength_val_str) if isinstance(strength_val_str, str) else strength_val_str
                    if strength_val == TrendStrength.STRONG:
                        trend_bonus += 5
            
            final_score = base_score + rr_bonus + trend_bonus
            return final_score

        return sorted(signals, key=signal_score, reverse=True)
    
