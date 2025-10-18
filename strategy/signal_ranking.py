from typing import List

from common.constants import LONG_TERM_CONFIG
from common.core import SignalType, TradingSignal


class SignalRanking:
    @staticmethod
    def calculate_signal_score(signal: TradingSignal) -> float:
        base_score = signal.confidence_score
        rr_multiplier = min(signal.risk_reward_ratio, 5.0) / 2.0 if signal.risk_reward_ratio > 0 else 0.5
        
        timeframe_priority = LONG_TERM_CONFIG['timeframe_priority_weights'].get(signal.timeframe, 1.0)
        base_score *= timeframe_priority
        
        base_score *= rr_multiplier
        
        if signal.macro_data and signal.macro_data.fed_rate is not None:
            if signal.signal_type == SignalType.BUY and signal.macro_data.fed_rate > 4.0:
                base_score *= 0.8
            elif signal.signal_type == SignalType.SELL and signal.macro_data.fed_rate < 2.0:
                base_score *= 0.8
        
        return base_score

    @staticmethod
    def rank_signals(signals: List[TradingSignal]) -> List[TradingSignal]:
        return sorted(signals, key=SignalRanking.calculate_signal_score, reverse=True)

