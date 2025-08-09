from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List


class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

@dataclass
class TradingSignal:
    symbol: str
    signal_type: SignalType
    entry_price: float
    exit_price: float
    stop_loss: float
    timestamp: datetime
    timeframe: str
    confidence_score: float
    reasons: List[str]
    risk_reward_ratio: float
    predicted_profit: float
    volume_analysis: Dict[str, float]
    market_context: Dict[str, Any]
    dynamic_levels: Dict[str, float]