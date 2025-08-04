# models/indicators.py

from dataclasses import dataclass
from typing import Dict, Any, Optional, Union
from decimal import Decimal
from enum import Enum
from .base import BaseModel

class IndicatorType(Enum):
    """انواع اندیکاتور"""
    RSI = "rsi"
    MACD = "macd"
    BOLLINGER_BANDS = "bollinger_bands"
    SMA = "sma"
    EMA = "ema"
    STOCHASTIC = "stochastic"
    ADX = "adx"
    VOLUME_PROFILE = "volume_profile"

@dataclass
class IndicatorResult(BaseModel):
    """نتیجه محاسبه اندیکاتور"""
    type: IndicatorType
    symbol: str
    timeframe: str
    value: Union[Decimal, Dict[str, Decimal]]  # مقدار یا دیکشنری مقادیر
    signal: Optional[str] = None  # "buy", "sell", "neutral"
    strength: float = 0.0  # قدرت سیگنال 0-1
    confidence: float = 0.0  # اطمینان از سیگنال 0-1
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IndicatorResult':
        value = data['value']
        if isinstance(value, dict):
            value = {k: Decimal(str(v)) for k, v in value.items()}
        else:
            value = Decimal(str(value))
            
        return cls(
            type=IndicatorType(data['type']),
            symbol=data['symbol'],
            timeframe=data['timeframe'],
            value=value,
            signal=data.get('signal'),
            strength=data.get('strength', 0.0),
            confidence=data.get('confidence', 0.0)
        )