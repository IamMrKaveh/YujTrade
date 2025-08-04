# models/timeframe.py

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from .base import BaseModel

class TimeFrameType(Enum):
    """انواع تایم فریم"""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H2 = "2h"
    H4 = "4h"
    H6 = "6h"
    H8 = "8h"
    H12 = "12h"
    D1 = "1d"
    D2 = "2d"
    D3 = "3d"
    D5 = "5d"
    W1 = "1w"
    W2 = "2w"
    MN1 = "1M"
    MN3 = "3M"

class TimeFrameCategory(Enum):
    """دسته‌بندی تایم فریم‌ها"""
    SCALPING = "scalping"          # M1, M5
    INTRADAY = "intraday"          # M15, M30, H1, H2
    SHORT_TERM = "short_term"      # H4, H6, H8, H12
    SWING = "swing"                # D1, D2, D3
    POSITION = "position"          # W1, W2, MN1, MN3

class TimeFrameGroup(Enum):
    """گروه‌بندی تایم فریم‌ها بر اساس استفاده"""
    MINUTE = "minute"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"

@dataclass
class TimeFrameConfig:
    """تنظیمات اضافی برای هر تایم فریم"""
    max_candles: int = 1000        # حداکثر تعداد کندل قابل دریافت
    min_candles: int = 50          # حداقل تعداد کندل مورد نیاز
    chart_color: str = "#1f77b4"  # رنگ پیش‌فرض در چارت
    is_recommended: bool = False   # آیا این تایم فریم توصیه شده است
    risk_level: str = "medium"     # سطح ریسک: low, medium, high
    volatility_factor: float = 1.0 # ضریب نوسان
    analysis_weight: float = 1.0   # وزن در تحلیل چندتایم فریمه

@dataclass
class TimeFrame(BaseModel):
    """مدل تایم فریم با جزئیات کامل"""
    type: TimeFrameType
    interval_minutes: int
    display_name: str
    display_name_en: str = ""
    priority: int = 0
    is_active: bool = True
    category: TimeFrameCategory = TimeFrameCategory.INTRADAY
    group: TimeFrameGroup = TimeFrameGroup.HOURLY
    config: TimeFrameConfig = field(default_factory=TimeFrameConfig)
    description: str = ""
    typical_holding_period: str = ""  # مدت زمان معمول نگهداری پوزیشن
    suitable_for: List[str] = field(default_factory=list)  # مناسب برای چه نوع معامله‌گری
    related_timeframes: List[str] = field(default_factory=list)  # تایم فریم‌های مرتبط
    technical_indicators: List[str] = field(default_factory=list)  # اندیکاتورهای مناسب
    
    def __post_init__(self):
        """تنظیمات اولیه پس از ایجاد آبجکت"""
        if not self.display_name_en:
            self.display_name_en = self.type.value
        
        # تنظیم خودکار دسته‌بندی بر اساس interval_minutes
        self._auto_set_category()
        self._auto_set_group()
        self._set_default_config()
    
    def _auto_set_category(self):
        """تنظیم خودکار دسته‌بندی"""
        if self.interval_minutes <= 5:
            self.category = TimeFrameCategory.SCALPING
        elif self.interval_minutes <= 120:
            self.category = TimeFrameCategory.INTRADAY
        elif self.interval_minutes <= 720:
            self.category = TimeFrameCategory.SHORT_TERM
        elif self.interval_minutes <= 4320:  # 3 روز
            self.category = TimeFrameCategory.SWING
        else:
            self.category = TimeFrameCategory.POSITION
    
    def _auto_set_group(self):
        """تنظیم خودکار گروه"""
        if self.interval_minutes < 60:
            self.group = TimeFrameGroup.MINUTE
        elif self.interval_minutes < 1440:
            self.group = TimeFrameGroup.HOURLY
        elif self.interval_minutes < 10080:  # یک هفته
            self.group = TimeFrameGroup.DAILY
        elif self.interval_minutes < 43200:  # یک ماه
            self.group = TimeFrameGroup.WEEKLY
        else:
            self.group = TimeFrameGroup.MONTHLY
    
    def _set_default_config(self):
        """تنظیم پیکربندی پیش‌فرض بر اساس نوع تایم فریم"""
        category_configs = {
            TimeFrameCategory.SCALPING: TimeFrameConfig(
                max_candles=2000,
                min_candles=100,
                chart_color="#ff6b6b",
                risk_level="high",
                volatility_factor=2.0,
                analysis_weight=0.5
            ),
            TimeFrameCategory.INTRADAY: TimeFrameConfig(
                max_candles=1500,
                min_candles=75,
                chart_color="#4ecdc4",
                risk_level="medium",
                volatility_factor=1.5,
                analysis_weight=1.0
            ),
            TimeFrameCategory.SHORT_TERM: TimeFrameConfig(
                max_candles=1000,
                min_candles=50,
                chart_color="#45b7d1",
                is_recommended=True,
                risk_level="medium",
                volatility_factor=1.0,
                analysis_weight=1.5
            ),
            TimeFrameCategory.SWING: TimeFrameConfig(
                max_candles=800,
                min_candles=30,
                chart_color="#96ceb4",
                is_recommended=True,
                risk_level="low",
                volatility_factor=0.8,
                analysis_weight=2.0
            ),
            TimeFrameCategory.POSITION: TimeFrameConfig(
                max_candles=500,
                min_candles=20,
                chart_color="#feca57",
                risk_level="low",
                volatility_factor=0.5,
                analysis_weight=2.5
            )
        }
        
        if hasattr(self, 'config') and self.config and any(vars(self.config).values()):
            # اگر config از قبل تنظیم شده، آن را نگه می‌داریم
            pass
        else:
            self.config = category_configs.get(self.category, TimeFrameConfig())
    
    @property
    def value(self) -> str:
        """مقدار رشته‌ای تایم فریم"""
        return self.type.value
    
    @property
    def interval_seconds(self) -> int:
        """بازه زمانی به ثانیه"""
        return self.interval_minutes * 60
    
    @property
    def interval_hours(self) -> float:
        """بازه زمانی به ساعت"""
        return self.interval_minutes / 60
    
    @property
    def interval_days(self) -> float:
        """بازه زمانی به روز"""
        return self.interval_minutes / 1440
    
    @property
    def is_intraday(self) -> bool:
        """آیا تایم فریم روزانه است"""
        return self.interval_minutes < 1440
    
    @property
    def is_short_term(self) -> bool:
        """آیا تایم فریم کوتاه مدت است"""
        return self.category in [TimeFrameCategory.SCALPING, TimeFrameCategory.INTRADAY]
    
    @property
    def is_long_term(self) -> bool:
        """آیا تایم فریم بلند مدت است"""
        return self.category in [TimeFrameCategory.SWING, TimeFrameCategory.POSITION]
    
    def get_candles_for_period(self, days: int) -> int:
        """محاسبه تعداد کندل‌های مورد نیاز برای یک دوره زمانی"""
        total_minutes = days * 24 * 60
        return int(total_minutes / self.interval_minutes)
    
    def get_period_for_candles(self, candles: int) -> float:
        """محاسبه دوره زمانی بر اساس تعداد کندل (به روز)"""
        total_minutes = candles * self.interval_minutes
        return total_minutes / (24 * 60)
    
    def calculate_next_candle_time(self, current_time: datetime) -> datetime:
        """محاسبه زمان کندل بعدی"""
        interval_td = timedelta(minutes=self.interval_minutes)
        
        # گرد کردن زمان به نزدیک‌ترین بازه
        if self.interval_minutes < 60:
            # برای تایم فریم‌های دقیقه‌ای
            minutes = (current_time.minute // self.interval_minutes + 1) * self.interval_minutes
            next_time = current_time.replace(minute=0, second=0, microsecond=0)
            next_time += timedelta(minutes=minutes)
        else:
            # برای تایم فریم‌های ساعتی و بالاتر
            next_time = current_time + interval_td
        
        return next_time
    
    def is_compatible_with(self, other: 'TimeFrame') -> bool:
        """بررسی سازگاری با تایم فریم دیگر برای تحلیل چندتایم فریمه"""
        # تایم فریم‌های سازگار باید نسبت صحیحی داشته باشند
        ratio1 = other.interval_minutes / self.interval_minutes
        ratio2 = self.interval_minutes / other.interval_minutes
        
        # یکی از نسبت‌ها باید عدد صحیح باشد
        return ratio1.is_integer() or ratio2.is_integer()
    
    def get_higher_timeframes(self) -> List['TimeFrame']:
        """دریافت تایم فریم‌های بالاتر سازگار"""
        all_timeframes = self.get_all_active()
        higher_timeframes = []
        
        for tf in all_timeframes:
            if (tf.interval_minutes > self.interval_minutes and 
                self.is_compatible_with(tf)):
                higher_timeframes.append(tf)
        
        return sorted(higher_timeframes, key=lambda x: x.interval_minutes)
    
    def get_lower_timeframes(self) -> List['TimeFrame']:
        """دریافت تایم فریم‌های پایین‌تر سازگار"""
        all_timeframes = self.get_all_active()
        lower_timeframes = []
        
        for tf in all_timeframes:
            if (tf.interval_minutes < self.interval_minutes and 
                self.is_compatible_with(tf)):
                lower_timeframes.append(tf)
        
        return sorted(lower_timeframes, key=lambda x: x.interval_minutes, reverse=True)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TimeFrame':
        """ایجاد از دیکشنری"""
        config_data = data.get('config', {})
        config = TimeFrameConfig(**config_data) if config_data else TimeFrameConfig()
        
        return cls(
            type=TimeFrameType(data['type']),
            interval_minutes=data['interval_minutes'],
            display_name=data['display_name'],
            display_name_en=data.get('display_name_en', ''),
            priority=data.get('priority', 0),
            is_active=data.get('is_active', True),
            category=TimeFrameCategory(data.get('category', 'intraday')),
            group=TimeFrameGroup(data.get('group', 'hourly')),
            config=config,
            description=data.get('description', ''),
            typical_holding_period=data.get('typical_holding_period', ''),
            suitable_for=data.get('suitable_for', []),
            related_timeframes=data.get('related_timeframes', []),
            technical_indicators=data.get('technical_indicators', [])
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """تبدیل به دیکشنری"""
        return {
            'type': self.type.value,
            'interval_minutes': self.interval_minutes,
            'display_name': self.display_name,
            'display_name_en': self.display_name_en,
            'priority': self.priority,
            'is_active': self.is_active,
            'category': self.category.value,
            'group': self.group.value,
            'config': vars(self.config),
            'description': self.description,
            'typical_holding_period': self.typical_holding_period,
            'suitable_for': self.suitable_for,
            'related_timeframes': self.related_timeframes,
            'technical_indicators': self.technical_indicators
        }
    
    @classmethod
    def get_all_active(cls) -> List['TimeFrame']:
        """دریافت همه تایم فریم‌های فعال با جزئیات کامل"""
        timeframes = [
            cls(
                type=TimeFrameType.M1,
                interval_minutes=1,
                display_name="1 دقیقه",
                display_name_en="1 Minute",
                priority=1,
                description="مناسب برای اسکالپینگ و معاملات فوری",
                typical_holding_period="1-5 دقیقه",
                suitable_for=["اسکالپینگ", "معاملات فوری", "ورود دقیق"],
                related_timeframes=["5m", "15m"],
                technical_indicators=["RSI", "Stochastic", "MACD", "Bollinger Bands"]
            ),
            cls(
                type=TimeFrameType.M5,
                interval_minutes=5,
                display_name="5 دقیقه",
                display_name_en="5 Minutes",
                priority=2,
                description="مناسب برای اسکالپینگ و معاملات کوتاه مدت",
                typical_holding_period="5-30 دقیقه",
                suitable_for=["اسکالپینگ", "تأیید سیگنال", "معاملات سریع"],
                related_timeframes=["1m", "15m", "30m"],
                technical_indicators=["EMA", "RSI", "MACD", "Support/Resistance"]
            ),
            cls(
                type=TimeFrameType.M15,
                interval_minutes=15,
                display_name="15 دقیقه",
                display_name_en="15 Minutes",
                priority=3,
                description="مناسب برای معاملات روزانه و تحلیل کوتاه مدت",
                typical_holding_period="30 دقیقه - 2 ساعت",
                suitable_for=["معاملات روزانه", "تحلیل تکنیکال", "تأیید ترند"],
                related_timeframes=["5m", "30m", "1h"],
                technical_indicators=["Moving Averages", "RSI", "MACD", "Fibonacci"]
            ),
            cls(
                type=TimeFrameType.M30,
                interval_minutes=30,
                display_name="30 دقیقه",
                display_name_en="30 Minutes",
                priority=4,
                description="ترکیب خوب از سرعت و دقت برای معاملات روزانه",
                typical_holding_period="1-4 ساعت",
                suitable_for=["معاملات روزانه", "تحلیل متوسط مدت", "ترند فالوئینگ"],
                related_timeframes=["15m", "1h", "2h"],
                technical_indicators=["Trend Lines", "Moving Averages", "RSI", "Volume"]
            ),
            cls(
                type=TimeFrameType.H1,
                interval_minutes=60,
                display_name="1 ساعت",
                display_name_en="1 Hour",
                priority=5,
                description="استاندارد طلایی برای اکثر معامله‌گران",
                typical_holding_period="2-8 ساعت",
                suitable_for=["معاملات روزانه", "تحلیل جامع", "ترند شناسی"],
                related_timeframes=["30m", "2h", "4h"],
                technical_indicators=["All Major Indicators", "Chart Patterns", "Support/Resistance"]
            ),
            cls(
                type=TimeFrameType.H2,
                interval_minutes=120,
                display_name="2 ساعت",
                display_name_en="2 Hours",
                priority=6,
                description="مناسب برای معاملات متوسط مدت",
                typical_holding_period="4-12 ساعت",
                suitable_for=["معاملات متوسط", "تحلیل ترند", "کاهش نویز"],
                related_timeframes=["1h", "4h", "6h"],
                technical_indicators=["Trend Analysis", "Moving Averages", "Momentum Indicators"]
            ),
            cls(
                type=TimeFrameType.H4,
                interval_minutes=240,
                display_name="4 ساعت",
                display_name_en="4 Hours",
                priority=7,
                description="بهترین تایم فریم برای تحلیل ترند و سوئینگ",
                typical_holding_period="8 ساعت - 2 روز",
                suitable_for=["سوئینگ ترید", "تحلیل ترند", "نقاط ورود استراتژیک"],
                related_timeframes=["2h", "6h", "1d"],
                technical_indicators=["Trend Analysis", "Chart Patterns", "Fibonacci", "Pivot Points"]
            ),
            cls(
                type=TimeFrameType.H6,
                interval_minutes=360,
                display_name="6 ساعت",
                display_name_en="6 Hours",
                priority=8,
                description="مناسب برای تحلیل‌های بلند مدت‌تر",
                typical_holding_period="12 ساعت - 3 روز",
                suitable_for=["سوئینگ ترید", "تحلیل ساختاری", "ترندهای قوی"],
                related_timeframes=["4h", "8h", "12h"],
                technical_indicators=["Long-term Trends", "Major Support/Resistance", "Volume Analysis"]
            ),
            cls(
                type=TimeFrameType.H8,
                interval_minutes=480,
                display_name="8 ساعت",
                display_name_en="8 Hours",
                priority=9,
                description="تحلیل ساختار بازار و ترندهای قوی",
                typical_holding_period="1-4 روز",
                suitable_for=["سوئینگ ترید", "تحلیل ساختاری", "ترند‌های اصلی"],
                related_timeframes=["6h", "12h", "1d"],
                technical_indicators=["Structural Analysis", "Major Trends", "Key Levels"]
            ),
            cls(
                type=TimeFrameType.H12,
                interval_minutes=720,
                display_name="12 ساعت",
                display_name_en="12 Hours",
                priority=10,
                description="تحلیل نیمه روزانه برای ترندهای قوی",
                typical_holding_period="1-7 روز",
                suitable_for=["سوئینگ ترید", "پوزیشن گیری", "ترندهای اصلی"],
                related_timeframes=["8h", "1d", "2d"],
                technical_indicators=["Major Trends", "Key Support/Resistance", "Long-term Patterns"]
            ),
            cls(
                type=TimeFrameType.D1,
                interval_minutes=1440,
                display_name="روزانه",
                display_name_en="Daily",
                priority=11,
                description="مهم‌ترین تایم فریم برای تحلیل بلند مدت",
                typical_holding_period="3-30 روز",
                suitable_for=["سوئینگ ترید", "پوزیشن ترید", "تحلیل بنیادی"],
                related_timeframes=["12h", "2d", "1w"],
                technical_indicators=["All Major Indicators", "Long-term Trends", "Fundamental Analysis"]
            ),
            cls(
                type=TimeFrameType.D2,
                interval_minutes=2880,
                display_name="2 روزه",
                display_name_en="2 Days",
                priority=12,
                description="کاهش نویز روزانه و تمرکز بر ترندهای قوی",
                typical_holding_period="1-6 هفته",
                suitable_for=["پوزیشن ترید", "سرمایه‌گذاری", "ترندهای قوی"],
                related_timeframes=["1d", "3d", "1w"],
                technical_indicators=["Long-term Trends", "Major Patterns", "Fundamental Levels"]
            ),
            cls(
                type=TimeFrameType.D3,
                interval_minutes=4320,
                display_name="3 روزه",
                display_name_en="3 Days",
                priority=13,
                description="تحلیل ترندهای متوسط تا بلند مدت",
                typical_holding_period="2-8 هفته",
                suitable_for=["پوزیشن ترید", "سرمایه‌گذاری", "ترند‌های اصلی"],
                related_timeframes=["2d", "5d", "1w"],
                technical_indicators=["Trend Analysis", "Major Support/Resistance", "Cycle Analysis"]
            ),
            cls(
                type=TimeFrameType.D5,
                interval_minutes=7200,
                display_name="5 روزه",
                display_name_en="5 Days",
                priority=14,
                description="تحلیل هفتگی برای ترندهای بلند مدت",
                typical_holding_period="1-3 ماه",
                suitable_for=["پوزیشن ترید", "سرمایه‌گذاری بلند مدت"],
                related_timeframes=["3d", "1w", "2w"],
                technical_indicators=["Weekly Trends", "Long-term Patterns", "Cycle Analysis"]
            ),
            cls(
                type=TimeFrameType.W1,
                interval_minutes=10080,
                display_name="هفتگی",
                display_name_en="Weekly",
                priority=15,
                description="تحلیل ساختار کلی بازار و ترندهای اصلی",
                typical_holding_period="1-6 ماه",
                suitable_for=["پوزیشن ترید", "سرمایه‌گذاری", "تحلیل کلان"],
                related_timeframes=["5d", "2w", "1M"],
                technical_indicators=["Major Trends", "Long-term Cycles", "Fundamental Analysis"]
            ),
            cls(
                type=TimeFrameType.W2,
                interval_minutes=20160,
                display_name="2 هفته‌ای",
                display_name_en="Bi-weekly",
                priority=16,
                description="تحلیل ترندهای بسیار بلند مدت",
                typical_holding_period="2-12 ماه",
                suitable_for=["سرمایه‌گذاری بلند مدت", "تحلیل کلان"],
                related_timeframes=["1w", "1M", "3M"],
                technical_indicators=["Super Trends", "Long-term Cycles", "Macro Analysis"]
            ),
            cls(
                type=TimeFrameType.MN1,
                interval_minutes=43200,
                display_name="ماهانه",
                display_name_en="Monthly",
                priority=17,
                description="تحلیل بلند مدت و سرمایه‌گذاری",
                typical_holding_period="6 ماه - 2 سال",
                suitable_for=["سرمایه‌گذاری بلند مدت", "تحلیل کلان اقتصادی"],
                related_timeframes=["2w", "3M"],
                technical_indicators=["Super Long-term Trends", "Economic Cycles", "Fundamental Analysis"]
            ),
            cls(
                type=TimeFrameType.MN3,
                interval_minutes=129600,
                display_name="3 ماهه",
                display_name_en="Quarterly",
                priority=18,
                description="تحلیل فصلی و بررسی چرخه‌های اقتصادی",
                typical_holding_period="1-5 سال",
                suitable_for=["سرمایه‌گذاری طولانی مدت", "تحلیل چرخه‌ای"],
                related_timeframes=["1M"],
                technical_indicators=["Economic Cycles", "Fundamental Analysis", "Macro Trends"]
            )
        ]
        
        return [tf for tf in timeframes if tf.is_active]
    
    @classmethod
    def get_by_type(cls, timeframe_type: TimeFrameType) -> Optional['TimeFrame']:
        """دریافت تایم فریم بر اساس نوع"""
        all_timeframes = cls.get_all_active()
        for tf in all_timeframes:
            if tf.type == timeframe_type:
                return tf
        return None
    
    @classmethod
    def get_by_category(cls, category: TimeFrameCategory) -> List['TimeFrame']:
        """دریافت تایم فریم‌ها بر اساس دسته‌بندی"""
        all_timeframes = cls.get_all_active()
        return [tf for tf in all_timeframes if tf.category == category]
    
    @classmethod
    def get_recommended(cls) -> List['TimeFrame']:
        """دریافت تایم فریم‌های توصیه شده"""
        all_timeframes = cls.get_all_active()
        return [tf for tf in all_timeframes if tf.config.is_recommended]
    
    @classmethod
    def get_for_trading_style(cls, style: str) -> List['TimeFrame']:
        """دریافت تایم فریم‌های مناسب برای سبک معاملاتی خاص"""
        all_timeframes = cls.get_all_active()
        return [tf for tf in all_timeframes if style in tf.suitable_for]
    
    def __str__(self) -> str:
        return f"{self.display_name} ({self.type.value})"
    
    def __repr__(self) -> str:
        return f"TimeFrame(type={self.type.value}, interval={self.interval_minutes}min, category={self.category.value})"