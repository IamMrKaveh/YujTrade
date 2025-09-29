import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta

from module.analyzers import (MarketConditionAnalyzer,
                              PatternAnalyzer)
from module.constants import (MULTI_TF_CONFIRMATION_MAP,
                              MULTI_TF_CONFIRMATION_WEIGHTS)
from module.core import (DerivativesAnalysis, DynamicLevels, FundamentalAnalysis,
                         IndicatorResult, MarketAnalysis, OnChainAnalysis,
                         SignalType, TradingSignal, TrendDirection,
                         TrendStrength, BinanceFuturesData, MacroEconomicData, TrendingData, OrderBook)
from module.data_sources import (MarketIndicesFetcher, NewsFetcher)
from module.indicators import (ADXIndicator, ATRIndicator,
                               BollingerBandsIndicator, CCIIndicator,
                               ChaikinMoneyFlowIndicator, IchimokuIndicator,
                               MACDIndicator, MovingAverageIndicator,
                               OBVIndicator, ParabolicSARIndicator,
                               RSIIndicator, SqueezeMomentumIndicator,
                               StochasticIndicator, SuperTrendIndicator,
                               VolumeIndicator, WilliamsRIndicator,
                               VWAPIndicator, MoneyFlowIndexIndicator, AroonIndicator,
                               UltimateOscillatorIndicator, ROCIndicator, ADLineIndicator,
                               ForceIndexIndicator, VWMAIndicator, KeltnerChannelsIndicator,
                               DonchianChannelsIndicator, TRIXIndicator, EaseOfMovementIndicator,
                               StandardDeviationIndicator)
from module.logger_config import logger
from module.lstm import LSTMModelManager
from module.market import MarketDataProvider


class MultiTimeframeAnalyzer:
    def __init__(self, market_data_provider: MarketDataProvider, indicators, cache_ttl=300):
        self.market_data_provider = market_data_provider
        self.indicators = indicators
        self._cache = {}
        self._cache_expiry = {}
        self.cache_ttl = cache_ttl
        self._lock = asyncio.Lock()

    async def _get_cache(self, key):
        async with self._lock:
            if key in self._cache and time.time() < self._cache_expiry.get(key, 0):
                return self._cache[key]
        return None

    async def _set_cache(self, key, value):
        async with self._lock:
            self._cache[key] = value
            self._cache_expiry[key] = time.time() + self.cache_ttl

    async def is_direction_aligned(self, symbol: str, exec_tf: str, threshold: float = 0.6) -> bool:
        try:
            confirm_tfs = MULTI_TF_CONFIRMATION_MAP.get(exec_tf, [])
            if not confirm_tfs:
                return True
            
            base_key = (symbol, exec_tf)
            base_signals = await self._get_cache(base_key)
            if not base_signals:
                base_df = await self.market_data_provider.fetch_ohlcv_data(symbol, exec_tf)
                if base_df.empty or len(base_df) < 50:
                    return False
                base_signals = self._analyze_indicators(base_df)
                await self._set_cache(base_key, base_signals)

            if not base_signals:
                return False

            tasks = []
            for tf in confirm_tfs:
                tasks.append(self._get_confirmation_score(symbol, tf, base_signals, exec_tf))
            
            results = await asyncio.gather(*tasks)
            
            total_score = sum(r[0] for r in results)
            total_weight = sum(r[1] for r in results)

            if total_weight == 0:
                return False
            avg_score = total_score / total_weight
            return avg_score >= threshold
        except Exception as e:
            logger.error(f"Error in multi-timeframe analysis for {symbol}: {e}")
            return False

    async def _get_confirmation_score(self, symbol, tf, base_signals, exec_tf):
        try:
            confirm_key = (symbol, tf)
            confirm_signals = await self._get_cache(confirm_key)
            if not confirm_signals:
                confirm_df = await self.market_data_provider.fetch_ohlcv_data(symbol, tf)
                if confirm_df.empty or len(confirm_df) < 50:
                    return 0, 0
                confirm_signals = self._analyze_indicators(confirm_df)
                await self._set_cache(confirm_key, confirm_signals)

            if not confirm_signals:
                return 0, 0

            weight = MULTI_TF_CONFIRMATION_WEIGHTS.get(exec_tf, {}).get(tf, 1.0)
            matches = sum(1 for k in base_signals
                        if k in confirm_signals and self._signals_match(base_signals[k], confirm_signals[k]))
            score = matches / len(base_signals) if base_signals else 0
            return score * weight, weight
        except Exception as e:
            logger.warning(f"Error analyzing timeframe {tf} for {symbol}: {e}")
            return 0, 0


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
                if hasattr(result, 'interpretation') and result.interpretation:
                    signals[name] = result.interpretation
            except Exception as e:
                logger.warning(f"Error calculating {name}: {e}")
        return signals

class SignalGenerator:

    INDICATOR_WEIGHTS = {
        'rsi': 12, 'macd': 10, 'stoch': 8, 'mfi': 10, 'cci': 7, 'williams_r': 7,
        'bb': 7, 'supertrend': 9, 'psar': 7, 'ichimoku': 9,
        'volume': 6, 'cmf': 6, 'obv': 4,
        'trend': 15, 'strength': 10, 'divergence': 12, 'pattern': 9,
        'fear_greed': 8, 'btc_dominance': 5, 'dxy': 5,
        'lstm': 15, 'multi_tf': 10,
        'funding_rate': 9, 'taker_ratio': 10, 'top_trader_sentiment': 12,
        'liquidation': 10, 'order_book': 8,
        'fundamental': 10, 'trending': 7, 'macro': 10
    }
    
    def __init__(self, market_data_provider: MarketDataProvider, 
                 news_fetcher: Optional[NewsFetcher] = None,
                 market_indices_fetcher: Optional[MarketIndicesFetcher] = None,
                 lstm_model_manager: Optional[LSTMModelManager] = None,
                 multi_tf_analyzer: Optional[MultiTimeframeAnalyzer] = None,
                 config: Optional[Dict] = None):
        self.market_data_provider = market_data_provider
        self.indicators = {
            'sma_20': MovingAverageIndicator(20, "sma"), 'sma_50': MovingAverageIndicator(50, "sma"),
            'ema_12': MovingAverageIndicator(12, "ema"), 'ema_26': MovingAverageIndicator(26, "ema"),
            'rsi': RSIIndicator(), 'macd': MACDIndicator(), 'bb': BollingerBandsIndicator(),
            'stoch': StochasticIndicator(), 'volume': VolumeIndicator(), 'atr': ATRIndicator(),
            'ichimoku': IchimokuIndicator(), 'williams_r': WilliamsRIndicator(), 'cci': CCIIndicator(),
            'supertrend': SuperTrendIndicator(), 'adx': ADXIndicator(), 'cmf': ChaikinMoneyFlowIndicator(),
            'obv': OBVIndicator(), 'squeeze': SqueezeMomentumIndicator(), 'psar': ParabolicSARIndicator(),
            'vwap': VWAPIndicator(), 'mfi': MoneyFlowIndexIndicator(), 'aroon': AroonIndicator(),
            'uo': UltimateOscillatorIndicator(), 'roc': ROCIndicator(), 'ad_line': ADLineIndicator(),
            'force_index': ForceIndexIndicator(), 'vwma': VWMAIndicator(), 'keltner': KeltnerChannelsIndicator(),
            'donchian': DonchianChannelsIndicator(), 'trix': TRIXIndicator(), 'eom': EaseOfMovementIndicator(),
            'std_dev': StandardDeviationIndicator()
        }
        self.market_analyzer = MarketConditionAnalyzer()
        self.news_fetcher = news_fetcher
        self.market_indices_fetcher = market_indices_fetcher
        self.lstm_model_manager = lstm_model_manager
        self.multi_tf_analyzer = multi_tf_analyzer
        self.config = config or {'min_confidence_score': 75}

    def _safe_dataframe(self, df):
        if df is None or df.empty: return pd.DataFrame()
        try:
            df_copy = df.copy()
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df_copy.columns: df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
            df_copy = df_copy.replace([np.inf, -np.inf], np.nan).dropna(subset=numeric_cols, how='any')
            for col in ['open', 'high', 'low', 'close']:
                if col in df_copy.columns: df_copy = df_copy[df_copy[col] > 0]
            if 'volume' in df_copy.columns: df_copy = df_copy[df_copy['volume'] >= 0]
            if 'high' in df_copy.columns and 'low' in df_copy.columns and (df_copy['high'] < df_copy['low']).any():
                df_copy = df_copy[~(df_copy['high'] < df_copy['low'])]
            if not isinstance(df_copy.index, pd.DatetimeIndex): df_copy = df_copy.reset_index(drop=True)
            return df_copy
        except Exception as e:
            logger.error(f"Error in _safe_dataframe: {e}")
            return pd.DataFrame()

    def calculate_dynamic_levels(self, data: pd.DataFrame, market_analysis: MarketAnalysis, signal_type: SignalType) -> DynamicLevels:
        last_close = data['close'].iloc[-1]
        atr = market_analysis.volatility / 100 * last_close if market_analysis.volatility > 0 else data['close'].pct_change().std() * last_close
        if pd.isna(atr) or atr == 0: atr = last_close * 0.01

        if signal_type == SignalType.BUY:
            primary_entry, secondary_entry = last_close, last_close - 0.5 * atr
            primary_exit, secondary_exit = last_close + 2 * atr, last_close + 3.5 * atr
            tight_stop, wide_stop = primary_entry - 1.2 * atr, primary_entry - 2.0 * atr
        else:
            primary_entry, secondary_entry = last_close, last_close + 0.5 * atr
            primary_exit, secondary_exit = last_close - 2 * atr, last_close - 3.5 * atr
            tight_stop, wide_stop = primary_entry + 1.2 * atr, primary_entry + 2.0 * atr

        return DynamicLevels(
            primary_entry=primary_entry, secondary_entry=secondary_entry,
            primary_exit=primary_exit, secondary_exit=secondary_exit,
            tight_stop=tight_stop, wide_stop=wide_stop,
            breakeven_point=primary_entry + 0.2 * atr if signal_type == SignalType.BUY else primary_entry - 0.2 * atr,
            trailing_stop=atr * 0.7
        )

    def _calculate_confidence_score(self, symbol: str, signal_type: SignalType, indicators: Dict[str, IndicatorResult], 
                                  market_analysis: MarketAnalysis, patterns: List, context_data: Dict) -> Tuple[float, List[str]]:
        score, total_weight = 0.0, 0.0
        reasons = []
        is_buy = signal_type == SignalType.BUY

        def add_reason(source: str, interpretation: str, points: float, strength: float = -1):
            strength_str = f", strength: {strength:.0f}%" if strength >= 0 else ""
            reasons.append(f"{source} ({interpretation}{strength_str}) -> {points:+.1f} pts")

        for name, res in indicators.items():
            weight = self.INDICATOR_WEIGHTS.get(name.split('_')[0].lower(), 0)
            if weight == 0: continue
            total_weight += weight
            points = 0
            is_bullish = any(s in res.interpretation for s in ['bullish', 'oversold', 'above', 'buy', 'up', 'accumulation', 'bull_power', 'upward'])
            is_bearish = any(s in res.interpretation for s in ['bearish', 'overbought', 'below', 'sell', 'down', 'distribution', 'bear_power', 'downward'])
            if (is_buy and is_bullish) or (not is_buy and is_bearish):
                points = (res.signal_strength / 100) * weight
            elif (is_buy and is_bearish) or (not is_buy and is_bullish):
                points = - (res.signal_strength / 100) * weight * 1.2
            if points != 0:
                score += points
                add_reason(res.name, res.interpretation, points, res.signal_strength)

        trend_weight = self.INDICATOR_WEIGHTS.get('trend', 15)
        total_weight += trend_weight
        if (is_buy and market_analysis.trend == TrendDirection.BULLISH) or (not is_buy and market_analysis.trend == TrendDirection.BEARISH):
            adx_strength = indicators.get('adx', IndicatorResult("","",0,"")).signal_strength / 100
            points = trend_weight * (0.5 + 0.5 * adx_strength)
            score += points
            add_reason("Trend Align", f"{market_analysis.trend.value} ({market_analysis.trend_strength.value})", points)
        elif market_analysis.trend != TrendDirection.SIDEWAYS:
            points = -trend_weight * 0.75
            score += points
            add_reason("Trend Misalign", f"{market_analysis.trend.value}", points)

        pattern_weight = self.INDICATOR_WEIGHTS.get('pattern', 10)
        for p in patterns:
            total_weight += pattern_weight
            if (is_buy and 'bullish' in p) or (not is_buy and 'bearish' in p):
                score += pattern_weight
                add_reason("Pattern", p, pattern_weight)

        derivatives = context_data.get('derivatives')
        if derivatives:
            if derivatives.funding_rate:
                weight = self.INDICATOR_WEIGHTS.get('funding_rate', 9)
                total_weight += weight; fr = derivatives.funding_rate
                if fr > 0.0005: points = weight * min(fr / 0.001, 1.0); score += -points if is_buy else points; add_reason("Funding", "High Positive", score-sum(p[2] for p in reasons))
                elif fr < -0.0005: points = weight * min(abs(fr) / 0.001, 1.0); score += points if is_buy else -points; add_reason("Funding", "High Negative", score-sum(p[2] for p in reasons))
            
            if derivatives.taker_long_short_ratio:
                weight = self.INDICATOR_WEIGHTS.get('taker_ratio', 10)
                total_weight += weight; ratio = derivatives.taker_long_short_ratio
                if ratio > 1.1: points = weight * min((ratio - 1.0) * 2, 1.0); score += points if is_buy else -points*0.5; add_reason("Taker L/S", "Longs Dominate", score-sum(p[2] for p in reasons))
                elif ratio < 0.9: points = weight * min((1.0 - ratio) * 2, 1.0); score += -points*0.5 if is_buy else points; add_reason("Taker L/S", "Shorts Dominate", score-sum(p[2] for p in reasons))

            if derivatives.binance_futures_data:
                bfd = derivatives.binance_futures_data
                weight = self.INDICATOR_WEIGHTS.get('top_trader_sentiment', 12)
                total_weight += weight
                avg_ratio = (bfd.top_trader_long_short_ratio_accounts + bfd.top_trader_long_short_ratio_positions) / 2 if bfd.top_trader_long_short_ratio_accounts and bfd.top_trader_long_short_ratio_positions else bfd.top_trader_long_short_ratio_accounts or bfd.top_trader_long_short_ratio_positions
                if avg_ratio:
                    if avg_ratio > 1.2: points = weight * min((avg_ratio - 1.0) * 1.5, 1.0); score += points if is_buy else -points; add_reason("TopTraders", "Bullish", score-sum(p[2] for p in reasons))
                    elif avg_ratio < 0.8: points = weight * min((1.0 - avg_ratio) * 1.5, 1.0); score += -points if is_buy else points; add_reason("TopTraders", "Bearish", score-sum(p[2] for p in reasons))
                
                if bfd.liquidation_orders:
                    weight = self.INDICATOR_WEIGHTS.get('liquidation', 10)
                    total_weight += weight
                    sells = sum(float(o['origQty']) for o in bfd.liquidation_orders if o['side'] == 'SELL')
                    buys = sum(float(o['origQty']) for o in bfd.liquidation_orders if o['side'] == 'BUY')
                    if sells > buys * 2: points = weight * min(sells/(buys+1), 1.0); score += points if is_buy else -points; add_reason("Liqs", "Longs Liquidated", score-sum(p[2] for p in reasons))
                    if buys > sells * 2: points = weight * min(buys/(sells+1), 1.0); score += -points if is_buy else points; add_reason("Liqs", "Shorts Liquidated", score-sum(p[2] for p in reasons))

        order_book = context_data.get('order_book')
        if order_book and order_book.total_bid_volume and order_book.total_ask_volume > 0:
            weight = self.INDICATOR_WEIGHTS.get('order_book', 8)
            total_weight += weight
            imbalance = order_book.total_bid_volume / order_book.total_ask_volume
            if imbalance > 1.5: points = weight * min((imbalance - 1.0), 1.0); score += points if is_buy else -points; add_reason("OrderBook", "Bid Wall", score-sum(p[2] for p in reasons))
            elif imbalance < 0.66: points = weight * min((1.0/imbalance - 1.0), 1.0); score += -points if is_buy else points; add_reason("OrderBook", "Ask Wall", score-sum(p[2] for p in reasons))

        fundamental = market_analysis.fundamental_analysis
        if fundamental and fundamental.developer_score > 0:
            weight = self.INDICATOR_WEIGHTS.get('fundamental', 10)
            total_weight += weight
            points = weight * min(fundamental.developer_score / 10000, 1.0)
            if is_buy: score += points; add_reason("Dev Score", "Active", score-sum(p[2] for p in reasons))

        trending = context_data.get('trending_data')
        if trending and trending.coingecko_trending and symbol.split('/')[0] in trending.coingecko_trending[:5]:
            weight = self.INDICATOR_WEIGHTS.get('trending', 7)
            total_weight += weight
            score += weight if is_buy else -weight*0.5
            add_reason("Trending", "Top 5 on CG", weight if is_buy else -weight*0.5)

        if total_weight == 0: return 0.0, reasons
        final_score = (score / total_weight) * 100
        return min(max(final_score, 0), 100), reasons

    def _create_trading_signal(self, symbol, timeframe, signal_type, score, reasons, dynamic_levels, data, market_analysis, context_data):
        entry = dynamic_levels.primary_entry
        exit_price = dynamic_levels.primary_exit
        stop_loss = dynamic_levels.tight_stop

        rr = self._calculate_risk_reward(entry, exit_price, stop_loss)
        profit = ((exit_price - entry) / entry) * 100 if signal_type == SignalType.BUY else ((entry - exit_price) / entry) * 100

        return TradingSignal(
            symbol=symbol, signal_type=signal_type, entry_price=entry, exit_price=exit_price, stop_loss=stop_loss,
            timestamp=datetime.now(), timeframe=timeframe, confidence_score=score, reasons=reasons,
            risk_reward_ratio=rr, predicted_profit=profit,
            volume_analysis=self.market_analyzer.volume_analyzer.analyze_volume_pattern(data),
            market_context=vars(market_analysis), dynamic_levels=vars(dynamic_levels),
            fundamental_analysis=market_analysis.fundamental_analysis,
            on_chain_analysis=context_data.get('on_chain'),
            derivatives_analysis=context_data.get('derivatives'),
            order_book=context_data.get('order_book'),
            macro_data=context_data.get('macro_data'),
            trending_data=context_data.get('trending_data')
        )

    def _calculate_risk_reward(self, entry: float, exit_price: float, stop_loss: float) -> float:
        if pd.isna(entry) or pd.isna(exit_price) or pd.isna(stop_loss) or entry == stop_loss: return 0
        potential_profit = abs(exit_price - entry)
        potential_loss = abs(entry - stop_loss)
        return potential_profit / potential_loss if potential_loss > 0 else float('inf')

    async def generate_signals(self, data: pd.DataFrame, symbol: str, timeframe: str) -> List[TradingSignal]:
        data = self._safe_dataframe(data)
        if data.empty or len(data) < 100:
            logger.warning(f"Insufficient data for {symbol} on {timeframe}")
            return []

        indicator_results = {name: indicator.calculate(data) for name, indicator in self.indicators.items()}
        context_data = await self._gather_context_data(symbol)
        
        market_analysis = self.market_analyzer.analyze_market_condition(data, fundamental_analysis=context_data.get('fundamental'))
        patterns = PatternAnalyzer.detect_patterns(data)
        
        signals = []
        for signal_type in [SignalType.BUY, SignalType.SELL]:
            score, reasons = self._calculate_confidence_score(symbol, signal_type, indicator_results, market_analysis, patterns, context_data)
            
            if score >= self.config.get('min_confidence_score', 0):
                dynamic_levels = self.calculate_dynamic_levels(data, market_analysis, signal_type)
                trading_signal = self._create_trading_signal(symbol, timeframe, signal_type, score, reasons, dynamic_levels, data, market_analysis, context_data)
                
                if self.multi_tf_analyzer:
                    is_aligned = await self.multi_tf_analyzer.is_direction_aligned(symbol, timeframe)
                    if is_aligned:
                        trading_signal.confidence_score = min(100, score + self.INDICATOR_WEIGHTS['multi_tf'])
                        trading_signal.reasons.append("Confirm: Multi-Timeframe Alignment")
                    else:
                        trading_signal.confidence_score *= 0.8
                        trading_signal.reasons.append("Warning: Multi-Timeframe Misalignment")

                if self.lstm_model_manager:
                    await self.lstm_model_manager.train_model_if_needed(symbol, timeframe, data)
                    prediction = await self.lstm_model_manager.predict_async(symbol, timeframe, data)
                    if prediction is not None and len(prediction) > 0:
                        pred_price, last_close = prediction[0], data['close'].iloc[-1]
                        if (signal_type == SignalType.BUY and pred_price > last_close) or \
                           (signal_type == SignalType.SELL and pred_price < last_close):
                            trading_signal.confidence_score = min(100, trading_signal.confidence_score + self.INDICATOR_WEIGHTS['lstm'])
                            trading_signal.reasons.append(f"Confirm: LSTM Prediction ({pred_price:.4f})")
                
                signals.append(trading_signal)
        return signals

    async def _gather_context_data(self, symbol: str) -> Dict[str, Any]:
        asset_name = symbol.split('/')[0].lower()
        context_data = {}
        
        tasks = {}
        
        if self.market_indices_fetcher:
            coin_id = asset_name.replace('/usdt', '').lower()
            tasks['fundamental'] = self.market_indices_fetcher.get_fundamental_data(coin_id)
            tasks['cg_derivatives'] = self.market_indices_fetcher.get_coingecko_derivatives()
            tasks['trending'] = self.market_indices_fetcher.get_trending_searches()
            tasks['macro'] = self.market_indices_fetcher.get_macro_economic_data()

        if self.market_data_provider.binance_fetcher:
            bf = self.market_data_provider.binance_fetcher
            tasks['binance_oi'] = bf.get_open_interest(symbol)
            tasks['binance_fr'] = bf.get_funding_rate(symbol)
            tasks['binance_ls_ratio'] = bf.get_taker_long_short_ratio(symbol)
            tasks['binance_top_acc'] = bf.get_top_trader_long_short_ratio_accounts(symbol)
            tasks['binance_top_pos'] = bf.get_top_trader_long_short_ratio_positions(symbol)
            tasks['binance_liq'] = bf.get_liquidation_orders(symbol)
            tasks['order_book'] = bf.get_order_book_depth(symbol)
            tasks['mark_price'] = bf.get_mark_price(symbol)

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        res_map = dict(zip(tasks.keys(), results))
        
        # Helper to safely get result
        def get_res(key):
            res = res_map.get(key)
            if isinstance(res, Exception):
                logger.warning(f"Failed to gather context for '{key}': {res}")
                return None
            return res

        context_data['fundamental'] = get_res('fundamental')
        context_data['order_book'] = get_res('order_book')
        context_data['trending_data'] = TrendingData(coingecko_trending=get_res('trending') or [])
        
        macro_data_raw = get_res('macro') or {}
        context_data['macro_data'] = MacroEconomicData(
            cpi=macro_data_raw.get('CPI'),
            fed_rate=macro_data_raw.get('FED_RATE'),
            treasury_yield_10y=macro_data_raw.get('TREASURY_YIELD_10Y'),
            gdp=macro_data_raw.get('GDP'),
            unemployment=macro_data_raw.get('UNEMPLOYMENT')
        )

        binance_futures_data = BinanceFuturesData(
            top_trader_long_short_ratio_accounts=get_res('binance_top_acc'),
            top_trader_long_short_ratio_positions=get_res('binance_top_pos'),
            liquidation_orders=get_res('binance_liq') or [],
            mark_price=get_res('mark_price')
        )
        
        context_data['derivatives'] = DerivativesAnalysis(
            open_interest=get_res('binance_oi'),
            funding_rate=get_res('binance_fr'),
            taker_long_short_ratio=get_res('binance_ls_ratio'),
            coingecko_derivatives=get_res('cg_derivatives') or [],
            binance_futures_data=binance_futures_data
        )

        return context_data


class SignalRanking:
    @staticmethod
    def rank_signals(signals: List[TradingSignal]) -> List[TradingSignal]:
        def signal_score(signal: TradingSignal) -> float:
            base_score = signal.confidence_score
            
            rr = signal.risk_reward_ratio or 0
            rr_bonus = 0
            if rr >= 3.0: rr_bonus = 15
            elif rr >= 2.0: rr_bonus = 10
            elif rr >= 1.5: rr_bonus = 5
            
            trend_bonus = 0
            if signal.market_context:
                mc = signal.market_context
                is_aligned = (signal.signal_type == SignalType.BUY and mc.get('trend') == TrendDirection.BULLISH) or \
                             (signal.signal_type == SignalType.SELL and mc.get('trend') == TrendDirection.BEARISH)
                if is_aligned:
                    trend_bonus = 10
                    if mc.get('trend_strength') == TrendStrength.STRONG:
                        trend_bonus += 5
            
            final_score = base_score + rr_bonus + trend_bonus
            return final_score

        return sorted(signals, key=signal_score, reverse=True)