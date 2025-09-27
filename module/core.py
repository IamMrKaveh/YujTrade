from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class TrendDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"


class MarketCondition(Enum):
    OVERSOLD = "oversold"
    OVERBOUGHT = "overbought"
    NEUTRAL = "neutral"


class TrendStrength(Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


@dataclass
class IndicatorResult:
    name: str
    value: float
    signal_strength: float
    interpretation: str


@dataclass
class FundamentalAnalysis:
    market_cap: float = 0.0
    circulating_supply: float = 0.0
    developer_score: float = 0.0


@dataclass
class OnChainAnalysis:
    mvrv: Optional[float] = None
    sopr: Optional[float] = None
    active_addresses: Optional[int] = None
    realized_cap: Optional[float] = None


@dataclass
class DerivativesAnalysis:
    open_interest: Optional[float] = None
    funding_rate: Optional[float] = None


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
    fundamental_analysis: Optional[FundamentalAnalysis] = None
    hurst_exponent: Optional[float] = None


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
    fundamental_analysis: Optional[FundamentalAnalysis] = None
    on_chain_analysis: Optional[OnChainAnalysis] = None
    derivatives_analysis: Optional[DerivativesAnalysis] = None