# models/strategy.py
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from datetime import datetime, timedelta
from decimal import Decimal
from .base import BaseModel
from .signal import Signal

class StrategyType(Enum):
    """نوع استراتژی"""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    SCALPING = "scalping"
    SWING = "swing"
    ARBITRAGE = "arbitrage"
    MOMENTUM = "momentum"
    CONTRARIAN = "contrarian"
    GRID = "grid"
    MARTINGALE = "martingale"

class MarketCondition(Enum):
    """شرایط بازار"""
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    LOW_VOLATILITY = "low_volatility"
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"

class RiskLevel(Enum):
    """سطح ریسک"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class TradingHours:
    """ساعات معاملاتی"""
    start_time: str  # Format: "HH:MM"
    end_time: str   # Format: "HH:MM"
    timezone: str = "UTC"
    days_of_week: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Monday-Friday
    
    def is_active_now(self) -> bool:
        """آیا در حال حاضر در ساعات معاملاتی هستیم؟"""
        now = datetime.now()
        return (
            now.weekday() in self.days_of_week and
            self.start_time <= now.strftime("%H:%M") <= self.end_time
        )

@dataclass
class StrategyParameters:
    """پارامترهای استراتژی"""
    entry_conditions: Dict[str, Any] = field(default_factory=dict)
    exit_conditions: Dict[str, Any] = field(default_factory=dict)
    stop_loss_pips: Optional[float] = None
    take_profit_pips: Optional[float] = None
    trailing_stop: bool = False
    trailing_stop_distance: Optional[float] = None
    max_spread: Optional[float] = None
    min_volume: Optional[int] = None
    
    def validate(self) -> List[str]:
        """اعتبارسنجی پارامترها"""
        errors = []
        if self.stop_loss_pips and self.stop_loss_pips <= 0:
            errors.append("Stop loss must be positive")
        if self.take_profit_pips and self.take_profit_pips <= 0:
            errors.append("Take profit must be positive")
        if self.trailing_stop and not self.trailing_stop_distance:
            errors.append("Trailing stop distance required when trailing stop is enabled")
        return errors

@dataclass
class BacktestMetrics:
    """معیارهای بک‌تست"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_profit: Decimal = Decimal('0')
    total_loss: Decimal = Decimal('0')
    max_drawdown: Decimal = Decimal('0')
    sharpe_ratio: Optional[float] = None
    profit_factor: Optional[float] = None
    win_rate: float = 0.0
    average_win: Decimal = Decimal('0')
    average_loss: Decimal = Decimal('0')
    largest_win: Decimal = Decimal('0')
    largest_loss: Decimal = Decimal('0')
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    @property
    def net_profit(self) -> Decimal:
        """سود خالص"""
        return self.total_profit - abs(self.total_loss)
    
    @property
    def expectancy(self) -> Decimal:
        """انتظار ریاضی"""
        if self.total_trades == 0:
            return Decimal('0')
        return (self.win_rate * self.average_win) - ((1 - self.win_rate) * abs(self.average_loss))

@dataclass
class MarketFilter:
    """فیلتر شرایط بازار"""
    required_conditions: List[MarketCondition] = field(default_factory=list)
    forbidden_conditions: List[MarketCondition] = field(default_factory=list)
    min_volatility: Optional[float] = None
    max_volatility: Optional[float] = None
    min_volume: Optional[int] = None
    news_filter: bool = False
    economic_calendar_filter: bool = False
    
    def check_conditions(self, current_condition: MarketCondition, 
                        volatility: float, volume: int) -> bool:
        """بررسی شرایط فیلتر"""
        if current_condition in self.forbidden_conditions:
            return False
        if self.required_conditions and current_condition not in self.required_conditions:
            return False
        if self.min_volatility and volatility < self.min_volatility:
            return False
        if self.max_volatility and volatility > self.max_volatility:
            return False
        if self.min_volume and volume < self.min_volume:
            return False
        return True

@dataclass
class Strategy(BaseModel):
    """مدل استراتژی معاملاتی پیشرفته"""
    name: str
    type: StrategyType
    description: str
    version: str = "1.0.0"
    author: str = ""
    
    # تنظیمات اصلی
    timeframes: List[str] = field(default_factory=list)
    symbols: List[str] = field(default_factory=list)  # نمادهای قابل معامله
    indicators_required: List[str] = field(default_factory=list)
    
    # تنظیمات ریسک
    risk_level: RiskLevel = RiskLevel.MEDIUM
    min_confidence: float = 70.0
    risk_reward_min: float = 2.0
    max_risk_per_trade: float = 2.0  # درصد سرمایه
    max_daily_loss: float = 5.0      # درصد سرمایه
    max_open_positions: int = 3
    
    # پارامترهای استراتژی
    parameters: StrategyParameters = field(default_factory=StrategyParameters)
    
    # فیلترها
    market_filter: MarketFilter = field(default_factory=MarketFilter)
    trading_hours: Optional[TradingHours] = None
    
    # تنظیمات فعالیت
    is_active: bool = True
    is_backtesting: bool = False
    is_paper_trading: bool = False
    is_live_trading: bool = False
    
    # آمار عملکرد
    backtest_metrics: Optional[BacktestMetrics] = None
    last_updated: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)
    
    # تنظیمات اضافی
    config: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    category: str = ""
    
    def __post_init__(self):
        """اعمال تنظیمات پس از ایجاد"""
        if not self.parameters:
            self.parameters = StrategyParameters()
        if not self.market_filter:
            self.market_filter = MarketFilter()
    
    def validate(self) -> List[str]:
        """اعتبارسنجی استراتژی"""
        errors = []
        
        if not self.name:
            errors.append("Strategy name is required")
        if not self.timeframes:
            errors.append("At least one timeframe is required")
        if self.min_confidence < 0 or self.min_confidence > 100:
            errors.append("Confidence must be between 0 and 100")
        if self.risk_reward_min <= 0:
            errors.append("Risk/reward ratio must be positive")
        if self.max_risk_per_trade <= 0 or self.max_risk_per_trade > 100:
            errors.append("Risk per trade must be between 0 and 100")
        
        # اعتبارسنجی پارامترها
        errors.extend(self.parameters.validate())
        
        return errors
    
    def is_valid_for_symbol(self, symbol: str) -> bool:
        """آیا استراتژی برای این نماد قابل اجرا است؟"""
        if not self.symbols:
            return True  # اگر محدودیتی نباشد، همه نمادها مجاز هستند
        return symbol in self.symbols
    
    def is_valid_for_timeframe(self, timeframe: str) -> bool:
        """آیا استراتژی برای این تایم‌فریم قابل اجرا است؟"""
        return timeframe in self.timeframes
    
    def can_trade_now(self) -> bool:
        """آیا در حال حاضر می‌توان معامله کرد؟"""
        if not self.is_active:
            return False
        if self.trading_hours and not self.trading_hours.is_active_now():
            return False
        return True
    
    def update_performance(self, metrics: BacktestMetrics):
        """به‌روزرسانی معیارهای عملکرد"""
        self.backtest_metrics = metrics
        self.last_updated = datetime.now()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Strategy':
        """ایجاد از دیکشنری"""
        # پردازش تاریخ‌ها
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'last_updated' in data and isinstance(data['last_updated'], str):
            data['last_updated'] = datetime.fromisoformat(data['last_updated'])
        
        # پردازش پارامترها
        if 'parameters' in data:
            data['parameters'] = StrategyParameters(**data['parameters'])
        
        # پردازش فیلتر بازار
        if 'market_filter' in data:
            filter_data = data['market_filter']
            if 'required_conditions' in filter_data:
                filter_data['required_conditions'] = [
                    MarketCondition(c) for c in filter_data['required_conditions']
                ]
            if 'forbidden_conditions' in filter_data:
                filter_data['forbidden_conditions'] = [
                    MarketCondition(c) for c in filter_data['forbidden_conditions']
                ]
            data['market_filter'] = MarketFilter(**filter_data)
        
        # پردازش ساعات معاملاتی
        if 'trading_hours' in data and data['trading_hours']:
            data['trading_hours'] = TradingHours(**data['trading_hours'])
        
        # پردازش معیارهای بک‌تست
        if 'backtest_metrics' in data and data['backtest_metrics']:
            metrics_data = data['backtest_metrics']
            # تبدیل Decimal
            for key in ['total_profit', 'total_loss', 'max_drawdown', 'average_win', 
                       'average_loss', 'largest_win', 'largest_loss']:
                if key in metrics_data and metrics_data[key] is not None:
                    metrics_data[key] = Decimal(str(metrics_data[key]))
            # تبدیل تاریخ‌ها
            for key in ['start_date', 'end_date']:
                if key in metrics_data and isinstance(metrics_data[key], str):
                    metrics_data[key] = datetime.fromisoformat(metrics_data[key])
            data['backtest_metrics'] = BacktestMetrics(**metrics_data)
        
        return cls(
            name=data['name'],
            type=StrategyType(data['type']),
            description=data['description'],
            version=data.get('version', '1.0.0'),
            author=data.get('author', ''),
            timeframes=data.get('timeframes', []),
            symbols=data.get('symbols', []),
            indicators_required=data.get('indicators_required', []),
            risk_level=RiskLevel(data.get('risk_level', 'medium')),
            min_confidence=data.get('min_confidence', 70.0),
            risk_reward_min=data.get('risk_reward_min', 2.0),
            max_risk_per_trade=data.get('max_risk_per_trade', 2.0),
            max_daily_loss=data.get('max_daily_loss', 5.0),
            max_open_positions=data.get('max_open_positions', 3),
            parameters=data.get('parameters', StrategyParameters()),
            market_filter=data.get('market_filter', MarketFilter()),
            trading_hours=data.get('trading_hours'),
            is_active=data.get('is_active', True),
            is_backtesting=data.get('is_backtesting', False),
            is_paper_trading=data.get('is_paper_trading', False),
            is_live_trading=data.get('is_live_trading', False),
            backtest_metrics=data.get('backtest_metrics'),
            last_updated=data.get('last_updated', datetime.now()),
            created_at=data.get('created_at', datetime.now()),
            config=data.get('config', {}),
            tags=data.get('tags', []),
            category=data.get('category', '')
        )

@dataclass
class StrategyResult(BaseModel):
    """نتیجه اجرای استراتژی پیشرفته"""
    strategy: Strategy
    symbol: str
    timeframe: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # نتیجه سیگنال
    signal: Optional[Signal] = None
    confidence: float = 0.0
    score: float = 0.0
    
    # جزئیات فیلترها
    passed_filters: List[str] = field(default_factory=list)
    failed_filters: List[str] = field(default_factory=list)
    filter_scores: Dict[str, float] = field(default_factory=dict)
    
    # شرایط بازار
    market_condition: Optional[MarketCondition] = None
    volatility: Optional[float] = None
    volume: Optional[int] = None
    spread: Optional[float] = None
    
    # معیارهای ریسک
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    expected_risk_reward: Optional[float] = None
    position_size_suggested: Optional[float] = None
    
    # اطلاعات اضافی
    indicators_values: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: Optional[float] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def is_successful(self) -> bool:
        """آیا استراتژی موفق بوده؟"""
        return (
            self.signal is not None and 
            self.signal.is_valid and 
            len(self.failed_filters) == 0 and
            self.confidence >= self.strategy.min_confidence
        )
    
    @property
    def filter_pass_rate(self) -> float:
        """درصد موفقیت فیلترها"""
        total_filters = len(self.passed_filters) + len(self.failed_filters)
        if total_filters == 0:
            return 100.0
        return (len(self.passed_filters) / total_filters) * 100
    
    @property
    def quality_score(self) -> float:
        """امتیاز کیفیت کلی"""
        base_score = self.score
        confidence_bonus = (self.confidence - self.strategy.min_confidence) / 100
        filter_bonus = self.filter_pass_rate / 100
        
        return min(100.0, base_score + (confidence_bonus * 20) + (filter_bonus * 10))
    
    def add_error(self, error: str):
        """افزودن خطا"""
        if error not in self.errors:
            self.errors.append(error)
    
    def add_warning(self, warning: str):
        """افزودن هشدار"""
        if warning not in self.warnings:
            self.warnings.append(warning)
    
    def has_errors(self) -> bool:
        """آیا خطایی وجود دارد؟"""
        return len(self.errors) > 0
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyResult':
        """ایجاد از دیکشنری"""
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        
        if 'market_condition' in data and data['market_condition']:
            data['market_condition'] = MarketCondition(data['market_condition'])
        
        return cls(
            strategy=Strategy.from_dict(data['strategy']),
            symbol=data['symbol'],
            timeframe=data['timeframe'],
            timestamp=data.get('timestamp', datetime.now()),
            signal=Signal.from_dict(data['signal']) if data.get('signal') else None,
            confidence=data.get('confidence', 0.0),
            score=data.get('score', 0.0),
            passed_filters=data.get('passed_filters', []),
            failed_filters=data.get('failed_filters', []),
            filter_scores=data.get('filter_scores', {}),
            market_condition=data.get('market_condition'),
            volatility=data.get('volatility'),
            volume=data.get('volume'),
            spread=data.get('spread'),
            risk_assessment=data.get('risk_assessment', {}),
            expected_risk_reward=data.get('expected_risk_reward'),
            position_size_suggested=data.get('position_size_suggested'),
            indicators_values=data.get('indicators_values', {}),
            execution_time_ms=data.get('execution_time_ms'),
            errors=data.get('errors', []),
            warnings=data.get('warnings', [])
        )

@dataclass
class StrategyCollection:
    """مجموعه‌ای از استراتژی‌ها"""
    name: str
    strategies: List[Strategy] = field(default_factory=list)
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    
    def add_strategy(self, strategy: Strategy):
        """افزودن استراتژی"""
        if strategy not in self.strategies:
            self.strategies.append(strategy)
    
    def remove_strategy(self, strategy_name: str):
        """حذف استراتژی"""
        self.strategies = [s for s in self.strategies if s.name != strategy_name]
    
    def get_active_strategies(self) -> List[Strategy]:
        """دریافت استراتژی‌های فعال"""
        return [s for s in self.strategies if s.is_active]
    
    def get_strategies_by_type(self, strategy_type: StrategyType) -> List[Strategy]:
        """دریافت استراتژی‌ها بر اساس نوع"""
        return [s for s in self.strategies if s.type == strategy_type]
    
    def get_strategies_for_symbol(self, symbol: str) -> List[Strategy]:
        """دریافت استراتژی‌های قابل اجرا برای نماد"""
        return [s for s in self.strategies if s.is_valid_for_symbol(symbol)]