# models/signal.py

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from decimal import Decimal
from enum import Enum
from datetime import datetime, timedelta
from .base import BaseModel
from .indicators import IndicatorResult

class SignalType(Enum):
    """نوع سیگنال"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"
    SCALP_BUY = "scalp_buy"
    SCALP_SELL = "scalp_sell"
    SWING_BUY = "swing_buy"
    SWING_SELL = "swing_sell"

class SignalStrength(Enum):
    """قدرت سیگنال"""
    VERY_WEAK = "very_weak"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"
    EXTREMELY_STRONG = "extremely_strong"

class SignalStatus(Enum):
    """وضعیت سیگنال"""
    ACTIVE = "active"
    TRIGGERED = "triggered"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    PARTIAL = "partial"

class SignalSource(Enum):
    """منبع سیگنال"""
    TECHNICAL_ANALYSIS = "technical_analysis"
    FUNDAMENTAL_ANALYSIS = "fundamental_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    AI_MODEL = "ai_model"
    HYBRID = "hybrid"
    MANUAL = "manual"
    COPY_TRADING = "copy_trading"

class TradingStyle(Enum):
    """سبک معاملاتی"""
    SCALPING = "scalping"
    DAY_TRADING = "day_trading"
    SWING_TRADING = "swing_trading"
    POSITION_TRADING = "position_trading"
    ARBITRAGE = "arbitrage"

class MarketCondition(Enum):
    """وضعیت بازار"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"

class RiskLevel(Enum):
    """سطح ریسک"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class TakeProfitLevel:
    """سطح take profit"""
    level: int
    price: Decimal
    percentage: float
    description: Optional[str] = None

@dataclass
class SignalMetrics:
    """متریک‌های سیگنال"""
    confidence_score: float = 0.0  # 0-100
    risk_reward_ratio: Optional[float] = None
    win_probability: Optional[float] = None
    expected_return: Optional[float] = None
    maximum_loss: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    volatility_score: Optional[float] = None

@dataclass
class SignalValidation:
    """اعتبارسنجی سیگنال"""
    is_validated: bool = False
    validation_score: float = 0.0
    conflicting_signals: int = 0
    supporting_signals: int = 0
    market_structure_aligned: bool = False
    volume_confirmation: bool = False

@dataclass
class SignalExecution:
    """اجرای سیگنال"""
    recommended_position_size: Optional[float] = None
    max_position_size: Optional[float] = None
    entry_method: Optional[str] = None  # market, limit, stop
    partial_entry_levels: List[Decimal] = field(default_factory=list)
    scaling_strategy: Optional[str] = None
    execution_timeframe: Optional[str] = None

@dataclass
class SignalContext:
    """زمینه سیگنال"""
    market_session: Optional[str] = None  # asian, european, american
    news_events: List[str] = field(default_factory=list)
    economic_calendar: List[str] = field(default_factory=list)
    sector_performance: Optional[str] = None
    correlation_assets: List[str] = field(default_factory=list)

@dataclass
class SignalBacktest:
    """بک‌تست سیگنال"""
    historical_accuracy: Optional[float] = None
    average_return: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    avg_winning_trade: Optional[float] = None
    avg_losing_trade: Optional[float] = None
    total_trades: Optional[int] = None
    profit_factor: Optional[float] = None

@dataclass
class Signal(BaseModel):
    """مدل سیگنال معاملاتی پیشرفته"""
    # اطلاعات اصلی
    id: Optional[str] = None
    symbol: str = ""
    timeframe: str = ""
    type: SignalType = SignalType.HOLD
    strength: SignalStrength = SignalStrength.MODERATE
    status: SignalStatus = SignalStatus.ACTIVE
    source: SignalSource = SignalSource.TECHNICAL_ANALYSIS
    trading_style: TradingStyle = TradingStyle.SWING_TRADING
    
    # قیمت‌ها
    entry_price: Optional[Decimal] = None
    current_price: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    take_profit_levels: List[TakeProfitLevel] = field(default_factory=list)
    
    # متریک‌ها
    metrics: SignalMetrics = field(default_factory=SignalMetrics)
    validation: SignalValidation = field(default_factory=SignalValidation)
    execution: SignalExecution = field(default_factory=SignalExecution)
    context: SignalContext = field(default_factory=SignalContext)
    backtest: SignalBacktest = field(default_factory=SignalBacktest)
    
    # اندیکاتورها و تحلیل
    indicators_used: List[IndicatorResult] = field(default_factory=list)
    technical_patterns: List[str] = field(default_factory=list)
    support_resistance_levels: List[Decimal] = field(default_factory=list)
    
    # شرایط بازار
    market_condition: MarketCondition = MarketCondition.SIDEWAYS
    risk_level: RiskLevel = RiskLevel.MEDIUM
    market_sentiment: Optional[str] = None
    volume_profile: Optional[str] = None
    
    # زمان‌بندی
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    triggered_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    estimated_duration: Optional[timedelta] = None
    
    # اطلاعات اضافی
    notes: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)
    related_signals: List[str] = field(default_factory=list)
    
    # پیگیری عملکرد
    actual_entry_price: Optional[Decimal] = None
    actual_exit_price: Optional[Decimal] = None
    actual_return: Optional[float] = None
    trade_outcome: Optional[str] = None  # win, loss, breakeven
    
    @property
    def is_valid(self) -> bool:
        """آیا سیگنال معتبر است؟"""
        return (
            self.metrics.confidence_score >= 60 and
            self.entry_price and self.entry_price > 0 and
            self.status == SignalStatus.ACTIVE and
            (self.expires_at is None or self.expires_at > datetime.now())
        )
    
    @property
    def is_expired(self) -> bool:
        """آیا سیگنال منقضی شده؟"""
        return self.expires_at is not None and self.expires_at <= datetime.now()
    
    @property
    def risk_percentage(self) -> Optional[float]:
        """درصد ریسک در صورت وجود stop loss"""
        if not self.stop_loss or not self.entry_price or self.entry_price == 0:
            return None
        return float(abs(self.entry_price - self.stop_loss) / self.entry_price * 100)
    
    @property
    def potential_rewards(self) -> List[float]:
        """درصد سودهای بالقوه"""
        if not self.entry_price or self.entry_price == 0:
            return []
        
        rewards = []
        for tp in self.take_profit_levels:
            reward = float(abs(tp.price - self.entry_price) / self.entry_price * 100)
            rewards.append(reward)
        return rewards
    
    @property
    def time_to_expiry(self) -> Optional[timedelta]:
        """زمان باقی‌مانده تا انقضا"""
        if not self.expires_at:
            return None
        return self.expires_at - datetime.now()
    
    @property
    def signal_age(self) -> timedelta:
        """سن سیگنال"""
        return datetime.now() - self.created_at
    
    def calculate_position_size(self, account_balance: Decimal, risk_per_trade: float = 2.0) -> Optional[Decimal]:
        """محاسبه اندازه پوزیشن"""
        if not self.entry_price or not self.stop_loss:
            return None
        
        risk_amount = account_balance * (risk_per_trade / 100)
        price_diff = abs(self.entry_price - self.stop_loss)
        
        if price_diff == 0:
            return None
        
        position_size = risk_amount / price_diff
        return position_size
    
    def get_risk_reward_ratios(self) -> List[float]:
        """محاسبه نسبت ریسک به ریوارد برای هر سطح"""
        if not self.risk_percentage:
            return []
        
        ratios = []
        for reward in self.potential_rewards:
            if self.risk_percentage > 0:
                ratios.append(reward / self.risk_percentage)
        return ratios
    
    def update_status(self, new_status: SignalStatus, notes: Optional[str] = None):
        """بروزرسانی وضعیت سیگنال"""
        self.status = new_status
        self.updated_at = datetime.now()
        
        if new_status == SignalStatus.TRIGGERED:
            self.triggered_at = datetime.now()
        
        if notes:
            if self.notes:
                self.notes += f"\n{datetime.now().strftime('%Y-%m-%d %H:%M')}: {notes}"
            else:
                self.notes = f"{datetime.now().strftime('%Y-%m-%d %H:%M')}: {notes}"
    
    def add_alert(self, alert_message: str):
        """اضافه کردن هشدار"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        self.alerts.append(f"{timestamp}: {alert_message}")
    
    def get_signal_quality_score(self) -> float:
        """محاسبه امتیاز کیفیت سیگنال"""
        score = 0.0
        
        # امتیاز اعتماد (40% وزن)
        score += (self.metrics.confidence_score / 100) * 40
        
        # اعتبارسنجی (20% وزن)
        if self.validation.is_validated:
            score += self.validation.validation_score * 0.2
        
        # نسبت ریسک به ریوارد (20% وزن)
        if self.metrics.risk_reward_ratio and self.metrics.risk_reward_ratio > 1:
            rr_score = min(self.metrics.risk_reward_ratio / 3, 1) * 20
            score += rr_score
        
        # قدرت سیگنال (10% وزن)
        strength_scores = {
            SignalStrength.VERY_WEAK: 0.2,
            SignalStrength.WEAK: 0.4,
            SignalStrength.MODERATE: 0.6,
            SignalStrength.STRONG: 0.8,
            SignalStrength.VERY_STRONG: 1.0,
            SignalStrength.EXTREMELY_STRONG: 1.0
        }
        score += strength_scores.get(self.strength, 0.6) * 10
        
        # بک‌تست (10% وزن)
        if self.backtest.win_rate:
            score += (self.backtest.win_rate / 100) * 10
        
        return min(score, 100)
    
    def is_compatible_with_market(self) -> bool:
        """بررسی سازگاری با شرایط بازار"""
        # بررسی‌های ساده سازگاری
        if self.type in [SignalType.BUY, SignalType.STRONG_BUY] and self.market_condition == MarketCondition.TRENDING_DOWN:
            return False
        
        if self.type in [SignalType.SELL, SignalType.STRONG_SELL] and self.market_condition == MarketCondition.TRENDING_UP:
            return False
        
        return True
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Signal':
        """ایجاد نمونه از دیکشنری"""
        # پردازش take profit levels
        tp_levels = []
        if 'take_profit_levels' in data:
            for tp_data in data['take_profit_levels']:
                tp_levels.append(TakeProfitLevel(
                    level=tp_data['level'],
                    price=Decimal(str(tp_data['price'])),
                    percentage=tp_data['percentage'],
                    description=tp_data.get('description')
                ))
        
        # پردازش indicators
        indicators = []
        if 'indicators_used' in data and data['indicators_used']:
            indicators = [IndicatorResult.from_dict(ind) for ind in data['indicators_used']]
        
        return cls(
            # اطلاعات اصلی
            id=data.get('id'),
            symbol=data.get('symbol', ''),
            timeframe=data.get('timeframe', ''),
            type=SignalType(data.get('type', 'hold')),
            strength=SignalStrength(data.get('strength', 'moderate')),
            status=SignalStatus(data.get('status', 'active')),
            source=SignalSource(data.get('source', 'technical_analysis')),
            trading_style=TradingStyle(data.get('trading_style', 'swing_trading')),
            
            # قیمت‌ها
            entry_price=Decimal(str(data['entry_price'])) if data.get('entry_price') else None,
            current_price=Decimal(str(data['current_price'])) if data.get('current_price') else None,
            stop_loss=Decimal(str(data['stop_loss'])) if data.get('stop_loss') else None,
            take_profit_levels=tp_levels,
            
            # متریک‌ها
            metrics=SignalMetrics(
                confidence_score=data.get('confidence_score', 0.0),
                risk_reward_ratio=data.get('risk_reward_ratio'),
                win_probability=data.get('win_probability'),
                expected_return=data.get('expected_return'),
                maximum_loss=data.get('maximum_loss'),
                sharpe_ratio=data.get('sharpe_ratio'),
                volatility_score=data.get('volatility_score')
            ),
            
            # اعتبارسنجی
            validation=SignalValidation(
                is_validated=data.get('is_validated', False),
                validation_score=data.get('validation_score', 0.0),
                conflicting_signals=data.get('conflicting_signals', 0),
                supporting_signals=data.get('supporting_signals', 0),
                market_structure_aligned=data.get('market_structure_aligned', False),
                volume_confirmation=data.get('volume_confirmation', False)
            ),
            
            # اجرا
            execution=SignalExecution(
                recommended_position_size=data.get('recommended_position_size'),
                max_position_size=data.get('max_position_size'),
                entry_method=data.get('entry_method'),
                partial_entry_levels=[Decimal(str(level)) for level in data.get('partial_entry_levels', [])],
                scaling_strategy=data.get('scaling_strategy'),
                execution_timeframe=data.get('execution_timeframe')
            ),
            
            # زمینه
            context=SignalContext(
                market_session=data.get('market_session'),
                news_events=data.get('news_events', []),
                economic_calendar=data.get('economic_calendar', []),
                sector_performance=data.get('sector_performance'),
                correlation_assets=data.get('correlation_assets', [])
            ),
            
            # بک‌تست
            backtest=SignalBacktest(
                historical_accuracy=data.get('historical_accuracy'),
                average_return=data.get('average_return'),
                max_drawdown=data.get('max_drawdown'),
                win_rate=data.get('win_rate'),
                avg_winning_trade=data.get('avg_winning_trade'),
                avg_losing_trade=data.get('avg_losing_trade'),
                total_trades=data.get('total_trades'),
                profit_factor=data.get('profit_factor')
            ),
            
            # اندیکاتورها
            indicators_used=indicators,
            technical_patterns=data.get('technical_patterns', []),
            support_resistance_levels=[Decimal(str(level)) for level in data.get('support_resistance_levels', [])],
            
            # شرایط بازار
            market_condition=MarketCondition(data.get('market_condition', 'sideways')),
            risk_level=RiskLevel(data.get('risk_level', 'medium')),
            market_sentiment=data.get('market_sentiment'),
            volume_profile=data.get('volume_profile'),
            
            # زمان‌بندی
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.now(),
            updated_at=datetime.fromisoformat(data['updated_at']) if data.get('updated_at') else None,
            triggered_at=datetime.fromisoformat(data['triggered_at']) if data.get('triggered_at') else None,
            expires_at=datetime.fromisoformat(data['expires_at']) if data.get('expires_at') else None,
            estimated_duration=timedelta(seconds=data['estimated_duration']) if data.get('estimated_duration') else None,
            
            # اطلاعات اضافی
            notes=data.get('notes'),
            tags=data.get('tags', []),
            alerts=data.get('alerts', []),
            related_signals=data.get('related_signals', []),
            
            # پیگیری عملکرد
            actual_entry_price=Decimal(str(data['actual_entry_price'])) if data.get('actual_entry_price') else None,
            actual_exit_price=Decimal(str(data['actual_exit_price'])) if data.get('actual_exit_price') else None,
            actual_return=data.get('actual_return'),
            trade_outcome=data.get('trade_outcome')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """تبدیل به دیکشنری"""
        return {
            # اطلاعات اصلی
            'id': self.id,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'type': self.type.value,
            'strength': self.strength.value,
            'status': self.status.value,
            'source': self.source.value,
            'trading_style': self.trading_style.value,
            
            # قیمت‌ها
            'entry_price': str(self.entry_price) if self.entry_price else None,
            'current_price': str(self.current_price) if self.current_price else None,
            'stop_loss': str(self.stop_loss) if self.stop_loss else None,
            'take_profit_levels': [
                {
                    'level': tp.level,
                    'price': str(tp.price),
                    'percentage': tp.percentage,
                    'description': tp.description
                } for tp in self.take_profit_levels
            ],
            
            # متریک‌ها
            'confidence_score': self.metrics.confidence_score,
            'risk_reward_ratio': self.metrics.risk_reward_ratio,
            'win_probability': self.metrics.win_probability,
            'expected_return': self.metrics.expected_return,
            'maximum_loss': self.metrics.maximum_loss,
            'sharpe_ratio': self.metrics.sharpe_ratio,
            'volatility_score': self.metrics.volatility_score,
            
            # اعتبارسنجی
            'is_validated': self.validation.is_validated,
            'validation_score': self.validation.validation_score,
            'conflicting_signals': self.validation.conflicting_signals,
            'supporting_signals': self.validation.supporting_signals,
            'market_structure_aligned': self.validation.market_structure_aligned,
            'volume_confirmation': self.validation.volume_confirmation,
            
            # اجرا
            'recommended_position_size': self.execution.recommended_position_size,
            'max_position_size': self.execution.max_position_size,
            'entry_method': self.execution.entry_method,
            'partial_entry_levels': [str(level) for level in self.execution.partial_entry_levels],
            'scaling_strategy': self.execution.scaling_strategy,
            'execution_timeframe': self.execution.execution_timeframe,
            
            # زمینه
            'market_session': self.context.market_session,
            'news_events': self.context.news_events,
            'economic_calendar': self.context.economic_calendar,
            'sector_performance': self.context.sector_performance,
            'correlation_assets': self.context.correlation_assets,
            
            # بک‌تست
            'historical_accuracy': self.backtest.historical_accuracy,
            'average_return': self.backtest.average_return,
            'max_drawdown': self.backtest.max_drawdown,
            'win_rate': self.backtest.win_rate,
            'avg_winning_trade': self.backtest.avg_winning_trade,
            'avg_losing_trade': self.backtest.avg_losing_trade,
            'total_trades': self.backtest.total_trades,
            'profit_factor': self.backtest.profit_factor,
            
            # اندیکاتورها
            'indicators_used': [ind.to_dict() for ind in self.indicators_used],
            'technical_patterns': self.technical_patterns,
            'support_resistance_levels': [str(level) for level in self.support_resistance_levels],
            
            # شرایط بازار
            'market_condition': self.market_condition.value,
            'risk_level': self.risk_level.value,
            'market_sentiment': self.market_sentiment,
            'volume_profile': self.volume_profile,
            
            # زمان‌بندی
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'triggered_at': self.triggered_at.isoformat() if self.triggered_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'estimated_duration': self.estimated_duration.total_seconds() if self.estimated_duration else None,
            
            # اطلاعات اضافی
            'notes': self.notes,
            'tags': self.tags,
            'alerts': self.alerts,
            'related_signals': self.related_signals,
            
            # پیگیری عملکرد
            'actual_entry_price': str(self.actual_entry_price) if self.actual_entry_price else None,
            'actual_exit_price': str(self.actual_exit_price) if self.actual_exit_price else None,
            'actual_return': self.actual_return,
            'trade_outcome': self.trade_outcome
        }
    
    def get_summary(self) -> str:
        """خلاصه سیگنال"""
        summary = f"{self.type.value.upper()} {self.symbol} ({self.timeframe})\n"
        summary += f"Strength: {self.strength.value.title()}\n"
        summary += f"Confidence: {self.metrics.confidence_score:.1f}%\n"
        
        if self.entry_price:
            summary += f"Entry: ${self.entry_price}\n"
        
        if self.stop_loss:
            summary += f"Stop Loss: ${self.stop_loss}\n"
        
        if self.take_profit_levels:
            summary += f"Take Profits: {len(self.take_profit_levels)} levels\n"
        
        if self.metrics.risk_reward_ratio:
            summary += f"R:R Ratio: 1:{self.metrics.risk_reward_ratio:.2f}\n"
        
        return summary