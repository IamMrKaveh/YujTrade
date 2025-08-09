from dataclasses import dataclass
from enum import Enum
from typing import List


class MarketCondition(Enum):
    OVERSOLD = "oversold"
    OVERBOUGHT = "overbought"
    NEUTRAL = "neutral"
    
class TrendDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"

class TrendStrength(Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"

@dataclass
class MarketAnalysis:
    trend: TrendDirection
    trend_strength: TrendStrength
    volatility: float
    volume_trend: str
    support_levels: List[float]
    resistance_levels: List[float]
    momentum_score: float
    market_condition: MarketCondition
    trend_acceleration: float
    volume_confirmation: bool
    
@dataclass
class DynamicLevels:
    primary_entry: float
    secondary_entry: float
    primary_exit: float
    secondary_exit: float
    tight_stop: float
    wide_stop: float
    breakeven_point: float
    trailing_stop: float