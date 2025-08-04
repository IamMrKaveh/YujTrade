# models/candle.py

# models/candle.py
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from decimal import Decimal
from datetime import datetime, timedelta
from enum import Enum
from .base import BaseModel, ModelValidator

class CandlePatternType(Enum):
    """انواع الگوی کندل"""
    DOJI = "doji"
    HAMMER = "hammer"
    HANGING_MAN = "hanging_man"
    SHOOTING_STAR = "shooting_star"
    ENGULFING_BULLISH = "engulfing_bullish"
    ENGULFING_BEARISH = "engulfing_bearish"
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"

@dataclass
class CandlePattern(BaseModel):
    """الگوی کندل شناسایی شده"""
    type: CandlePatternType
    candle_indices: List[int]  # اندیس کندل‌های دخیل در الگو
    confidence: float  # اطمینان از الگو 0-1
    bullish: bool  # آیا الگو صعودی است؟
    description: str
    
    def validate(self):
        super().validate()
        if not 0 <= self.confidence <= 1:
            raise ValidationError("اطمینان باید بین 0 و 1 باشد")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CandlePattern':
        return cls(
            type=CandlePatternType(data['type']),
            candle_indices=data['candle_indices'],
            confidence=data['confidence'],
            bullish=data['bullish'],
            description=data['description']
        )

@dataclass
class Candle(BaseModel):
    """مدل کندل OHLCV با قابلیت‌های پیشرفته"""
    symbol: str
    timeframe: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    quote_volume: Optional[Decimal] = None  # حجم quote
    trade_count: Optional[int] = None  # تعداد معامله
    taker_buy_volume: Optional[Decimal] = None  # حجم خرید taker
    
    def validate(self):
        super().validate()
        ModelValidator.validate_string(self.symbol, min_length=3, max_length=20)
        ModelValidator.validate_string(self.timeframe, min_length=2, max_length=5)
        
        # اعتبارسنجی قیمت‌ها
        for price_field in ['open', 'high', 'low', 'close']:
            price = getattr(self, price_field)
            ModelValidator.validate_decimal(price, min_value=Decimal('0'))
        
        # بررسی منطقی قیمت‌ها
        if not (self.low <= self.open <= self.high and self.low <= self.close <= self.high):
            raise ValidationError("قیمت‌های OHLC منطقی نیستند")
        
        ModelValidator.validate_decimal(self.volume, min_value=Decimal('0'))
    
    @property
    def is_green(self) -> bool:
        return self.close > self.open
    
    @property
    def is_red(self) -> bool:
        return self.close < self.open
    
    @property
    def is_doji(self) -> bool:
        """آیا کندل دوجی است؟"""
        body_size = abs(self.close - self.open)
        total_range = self.high - self.low
        return body_size / total_range < 0.1 if total_range > 0 else False
    
    @property
    def body_size(self) -> Decimal:
        return abs(self.close - self.open)
    
    @property
    def upper_shadow(self) -> Decimal:
        return self.high - max(self.open, self.close)
    
    @property
    def lower_shadow(self) -> Decimal:
        return min(self.open, self.close) - self.low
    
    @property
    def total_range(self) -> Decimal:
        return self.high - self.low
    
    @property
    def price_change(self) -> Decimal:
        return self.close - self.open
    
    @property
    def price_change_percent(self) -> float:
        if self.open == 0:
            return 0
        return float((self.close - self.open) / self.open * 100)
    
    @property
    def typical_price(self) -> Decimal:
        """قیمت معمول (HLC/3)"""
        return (self.high + self.low + self.close) / 3
    
    @property
    def weighted_price(self) -> Decimal:
        """قیمت وزنی (OHLC/4)"""
        return (self.open + self.high + self.low + self.close) / 4
    
    def get_volatility(self) -> float:
        """محاسبه نوسان کندل"""
        if self.typical_price == 0:
            return 0
        return float(self.total_range / self.typical_price * 100)
    
    def is_gap_up(self, previous_candle: 'Candle') -> bool:
        """آیا gap صعودی دارد؟"""
        return self.low > previous_candle.high
    
    def is_gap_down(self, previous_candle: 'Candle') -> bool:
        """آیا gap نزولی دارد؟"""
        return self.high < previous_candle.low
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Candle':
        return cls(
            symbol=data['symbol'],
            timeframe=data['timeframe'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            open=Decimal(str(data['open'])),
            high=Decimal(str(data['high'])),
            low=Decimal(str(data['low'])),
            close=Decimal(str(data['close'])),
            volume=Decimal(str(data['volume'])),
            quote_volume=Decimal(str(data['quote_volume'])) if data.get('quote_volume') else None,
            trade_count=data.get('trade_count'),
            taker_buy_volume=Decimal(str(data['taker_buy_volume'])) if data.get('taker_buy_volume') else None
        )

@dataclass
class CandleCollection(BaseModel):
    """مجموعه کندل‌ها با قابلیت‌های تحلیلی پیشرفته"""
    symbol: str
    timeframe: str
    candles: List[Candle]
    last_update: datetime
    is_complete: bool = True  # آیا داده‌ها کامل هستند؟
    patterns: List[CandlePattern] = field(default_factory=list)
    
    def validate(self):
        super().validate()
        ModelValidator.validate_string(self.symbol, min_length=3, max_length=20)
        ModelValidator.validate_string(self.timeframe, min_length=2, max_length=5)
        
        if not self.candles:
            raise ValidationError("حداقل یک کندل باید وجود داشته باشد")
        
        # بررسی ترتیب زمانی
        for i in range(1, len(self.candles)):
            if self.candles[i].timestamp <= self.candles[i-1].timestamp:
                raise ValidationError("کندل‌ها باید به ترتیب زمانی باشند")
    
    @property
    def latest_candle(self) -> Optional[Candle]:
        return self.candles[-1] if self.candles else None
    
    @property
    def oldest_candle(self) -> Optional[Candle]:
        return self.candles[0] if self.candles else None
    
    @property
    def count(self) -> int:
        return len(self.candles)
    
    @property
    def time_span(self) -> Optional[timedelta]:
        """بازه زمانی کل"""
        if len(self.candles) < 2:
            return None
        return self.latest_candle.timestamp - self.oldest_candle.timestamp
    
    def get_closes(self) -> List[Decimal]:
        return [candle.close for candle in self.candles]
    
    def get_highs(self) -> List[Decimal]:
        return [candle.high for candle in self.candles]
    
    def get_lows(self) -> List[Decimal]:
        return [candle.low for candle in self.candles]
    
    def get_opens(self) -> List[Decimal]:
        return [candle.open for candle in self.candles]
    
    def get_volumes(self) -> List[Decimal]:
        return [candle.volume for candle in self.candles]
    
    def get_typical_prices(self) -> List[Decimal]:
        return [candle.typical_price for candle in self.candles]
    
    def get_price_range(self) -> Tuple[Decimal, Decimal]:
        """محدوده قیمت (کمینه، بیشینه)"""
        if not self.candles:
            return Decimal('0'), Decimal('0')
        
        all_highs = self.get_highs()
        all_lows = self.get_lows()
        return min(all_lows), max(all_highs)
    
    def get_volume_stats(self) -> Dict[str, Decimal]:
        """آمار حجم معاملات"""
        volumes = self.get_volumes()
        if not volumes:
            return {}
        
        return {
            'total': sum(volumes),
            'average': sum(volumes) / len(volumes),
            'max': max(volumes),
            'min': min(volumes)
        }
    
    def get_price_change_percent(self) -> float:
        """درصد تغییر قیمت کل دوره"""
        if len(self.candles) < 2:
            return 0
        
        first_price = self.candles[0].open
        last_price = self.candles[-1].close
        
        if first_price == 0:
            return 0
        
        return float((last_price - first_price) / first_price * 100)
    
    def get_volatility_average(self) -> float:
        """میانگین نوسان"""
        if not self.candles:
            return 0
        
        volatilities = [candle.get_volatility() for candle in self.candles]
        return sum(volatilities) / len(volatilities)
    
    def get_green_red_ratio(self) -> Tuple[int, int]:
        """نسبت کندل‌های سبز به قرمز"""
        green_count = sum(1 for candle in self.candles if candle.is_green)
        red_count = len(self.candles) - green_count
        return green_count, red_count
    
    def slice_by_time(self, start_time: datetime, end_time: datetime) -> 'CandleCollection':
        """برش بر اساس زمان"""
        filtered_candles = [
            candle for candle in self.candles
            if start_time <= candle.timestamp <= end_time
        ]
        
        return CandleCollection(
            symbol=self.symbol,
            timeframe=self.timeframe,
            candles=filtered_candles,
            last_update=self.last_update,
            is_complete=False
        )
    
    def slice_by_count(self, count: int, from_end: bool = True) -> 'CandleCollection':
        """برش بر اساس تعداد"""
        if from_end:
            selected_candles = self.candles[-count:] if count < len(self.candles) else self.candles
        else:
            selected_candles = self.candles[:count]
        
        return CandleCollection(
            symbol=self.symbol,
            timeframe=self.timeframe,
            candles=selected_candles,
            last_update=self.last_update,
            is_complete=False
        )
    
    def detect_patterns(self) -> List[CandlePattern]:
        """شناسایی الگوهای کندل"""
        patterns = []
        
        for i in range(len(self.candles)):
            candle = self.candles[i]
            
            # شناسایی Doji
            if candle.is_doji:
                patterns.append(CandlePattern(
                    type=CandlePatternType.DOJI,
                    candle_indices=[i],
                    confidence=0.8,
                    bullish=False,  # خنثی
                    description="الگوی Doji - تردید در بازار"
                ))
            
            # شناسایی Hammer (نیاز به کندل قبلی برای تأیید)
            if i > 0:
                prev_candle = self.candles[i-1]
                if (candle.lower_shadow > candle.body_size * 2 and
                    candle.upper_shadow < candle.body_size * 0.1 and
                    prev_candle.is_red):
                    patterns.append(CandlePattern(
                        type=CandlePatternType.HAMMER,
                        candle_indices=[i-1, i],
                        confidence=0.7,
                        bullish=True,
                        description="الگوی Hammer - احتمال برگشت صعودی"
                    ))
        
        self.patterns = patterns
        return patterns
    
    def add_candle(self, candle: Candle):
        """افزودن کندل جدید"""
        if candle.symbol != self.symbol or candle.timeframe != self.timeframe:
            raise ValidationError("ارز یا تایم فریم کندل مطابقت ندارد")
        
        # حفظ ترتیب زمانی
        inserted = False
        for i, existing_candle in enumerate(self.candles):
            if candle.timestamp < existing_candle.timestamp:
                self.candles.insert(i, candle)
                inserted = True
                break
            elif candle.timestamp == existing_candle.timestamp:
                # جایگزینی کندل موجود
                self.candles[i] = candle
                inserted = True
                break
        
        if not inserted:
            self.candles.append(candle)
        
        self.last_update = datetime.now()
        self.update_timestamp()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CandleCollection':
        candles = [Candle.from_dict(c) for c in data['candles']]
        patterns = [CandlePattern.from_dict(p) for p in data.get('patterns', [])]
        
        return cls(
            symbol=data['symbol'],
            timeframe=data['timeframe'],
            candles=candles,
            last_update=datetime.fromisoformat(data['last_update']),
            is_complete=data.get('is_complete', True),
            patterns=patterns
        )