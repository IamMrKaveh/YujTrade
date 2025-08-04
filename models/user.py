# models/user.py

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional, Set, Union
from enum import Enum
from decimal import Decimal
import uuid
from .base import BaseModel
from .timeframe import TimeFrameType

class UserRole(Enum):
    """نقش کاربر"""
    FREE = "free"
    PREMIUM = "premium"
    VIP = "vip"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"

class UserStatus(Enum):
    """وضعیت کاربر"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    BANNED = "banned"
    PENDING_VERIFICATION = "pending_verification"

class NotificationType(Enum):
    """انواع اعلان"""
    SIGNAL = "signal"
    NEWS = "news"
    MARKET_UPDATE = "market_update"
    PROMOTION = "promotion"
    SYSTEM = "system"

class RiskLevel(Enum):
    """سطح ریسک"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    EXPERT = "expert"

class TradingExperience(Enum):
    """تجربه معاملاتی"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    PROFESSIONAL = "professional"

@dataclass
class UserLimits(BaseModel):
    """محدودیت‌های کاربر بر اساس نقش"""
    max_signals_per_day: int = 10
    max_signals_per_hour: int = 2
    max_watchlist_symbols: int = 20
    max_custom_alerts: int = 5
    access_to_premium_features: bool = False
    access_to_historical_data: bool = False
    priority_support: bool = False
    
    @classmethod
    def get_limits_by_role(cls, role: UserRole) -> 'UserLimits':
        """دریافت محدودیت‌ها بر اساس نقش کاربر"""
        limits_map = {
            UserRole.FREE: cls(
                max_signals_per_day=10,
                max_signals_per_hour=2,
                max_watchlist_symbols=10,
                max_custom_alerts=3,
                access_to_premium_features=False,
                access_to_historical_data=False,
                priority_support=False
            ),
            UserRole.PREMIUM: cls(
                max_signals_per_day=50,
                max_signals_per_hour=10,
                max_watchlist_symbols=50,
                max_custom_alerts=15,
                access_to_premium_features=True,
                access_to_historical_data=True,
                priority_support=False
            ),
            UserRole.VIP: cls(
                max_signals_per_day=200,
                max_signals_per_hour=25,
                max_watchlist_symbols=100,
                max_custom_alerts=50,
                access_to_premium_features=True,
                access_to_historical_data=True,
                priority_support=True
            ),
            UserRole.ADMIN: cls(
                max_signals_per_day=1000,
                max_signals_per_hour=100,
                max_watchlist_symbols=500,
                max_custom_alerts=100,
                access_to_premium_features=True,
                access_to_historical_data=True,
                priority_support=True
            ),
            UserRole.SUPER_ADMIN: cls(
                max_signals_per_day=-1,  # Unlimited
                max_signals_per_hour=-1,  # Unlimited
                max_watchlist_symbols=-1,  # Unlimited
                max_custom_alerts=-1,  # Unlimited
                access_to_premium_features=True,
                access_to_historical_data=True,
                priority_support=True
            )
        }
        return limits_map.get(role, cls())

@dataclass
class NotificationSettings(BaseModel):
    """تنظیمات اعلان‌ها"""
    enabled_types: Set[NotificationType] = field(default_factory=lambda: {NotificationType.SIGNAL})
    quiet_hours_start: Optional[str] = None  # Format: "HH:MM"
    quiet_hours_end: Optional[str] = None    # Format: "HH:MM"
    timezone: str = "UTC"
    sound_enabled: bool = True
    vibration_enabled: bool = True
    email_notifications: bool = False
    sms_notifications: bool = False
    max_notifications_per_hour: int = 10
    
    def is_quiet_time(self, current_time: datetime) -> bool:
        """بررسی اینکه آیا در زمان سکوت هستیم یا نه"""
        if not self.quiet_hours_start or not self.quiet_hours_end:
            return False
        
        start_hour, start_min = map(int, self.quiet_hours_start.split(':'))
        end_hour, end_min = map(int, self.quiet_hours_end.split(':'))
        
        current_hour = current_time.hour
        current_min = current_time.minute
        
        start_minutes = start_hour * 60 + start_min
        end_minutes = end_hour * 60 + end_min
        current_minutes = current_hour * 60 + current_min
        
        if start_minutes <= end_minutes:
            return start_minutes <= current_minutes <= end_minutes
        else:  # Crosses midnight
            return current_minutes >= start_minutes or current_minutes <= end_minutes

@dataclass
class TradingProfile(BaseModel):
    """پروفایل معاملاتی کاربر"""
    experience_level: TradingExperience = TradingExperience.BEGINNER
    risk_tolerance: RiskLevel = RiskLevel.MODERATE
    preferred_markets: Set[str] = field(default_factory=set)  # forex, crypto, stocks, commodities
    trading_capital: Optional[Decimal] = None
    max_risk_per_trade: float = 2.0  # Percentage
    preferred_trading_sessions: Set[str] = field(default_factory=set)  # london, new_york, tokyo, sydney
    investment_goals: List[str] = field(default_factory=list)  # income, growth, hedging
    
    def get_recommended_position_size(self, account_balance: Decimal, stop_loss_pips: int) -> Decimal:
        """محاسبه اندازه پوزیشن پیشنهادی"""
        if not self.trading_capital or stop_loss_pips <= 0:
            return Decimal('0')
        
        risk_amount = self.trading_capital * (Decimal(str(self.max_risk_per_trade)) / 100)
        # Simple position sizing calculation
        position_size = risk_amount / Decimal(str(stop_loss_pips))
        return position_size

@dataclass
class UserPreferences(BaseModel):
    """تنظیمات کاربر"""
    preferred_timeframes: Set[TimeFrameType] = field(default_factory=set)
    preferred_symbols: Set[str] = field(default_factory=set)
    watchlist: Set[str] = field(default_factory=set)
    blocked_symbols: Set[str] = field(default_factory=set)
    min_confidence_score: float = 70.0
    max_signals_per_timeframe: int = 5
    language: str = "fa"  # fa, en, ar
    theme: str = "dark"  # dark, light
    chart_style: str = "candlestick"  # candlestick, line, bar
    auto_follow_signals: bool = False
    show_educational_content: bool = True
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserPreferences':
        return cls(
            preferred_timeframes=set(TimeFrameType(tf) for tf in data.get('preferred_timeframes', [])),
            preferred_symbols=set(data.get('preferred_symbols', [])),
            watchlist=set(data.get('watchlist', [])),
            blocked_symbols=set(data.get('blocked_symbols', [])),
            min_confidence_score=data.get('min_confidence_score', 70.0),
            max_signals_per_timeframe=data.get('max_signals_per_timeframe', 5),
            language=data.get('language', 'fa'),
            theme=data.get('theme', 'dark'),
            chart_style=data.get('chart_style', 'candlestick'),
            auto_follow_signals=data.get('auto_follow_signals', False),
            show_educational_content=data.get('show_educational_content', True)
        )

@dataclass
class UserStatistics(BaseModel):
    """آمار کاربر"""
    total_signals_received: int = 0
    signals_followed: int = 0
    successful_trades: int = 0
    failed_trades: int = 0
    total_pnl: Decimal = field(default_factory=lambda: Decimal('0'))
    win_rate: float = 0.0
    average_holding_time: Optional[timedelta] = None
    favorite_symbols: List[str] = field(default_factory=list)
    most_active_timeframe: Optional[TimeFrameType] = None
    registration_date: Optional[datetime] = None
    last_profitable_trade: Optional[datetime] = None
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    
    def calculate_win_rate(self) -> float:
        """محاسبه نرخ برد"""
        total_trades = self.successful_trades + self.failed_trades
        if total_trades == 0:
            return 0.0
        return (self.successful_trades / total_trades) * 100
    
    def update_statistics(self, trade_result: bool, pnl: Decimal):
        """به‌روزرسانی آمار پس از معامله"""
        if trade_result:
            self.successful_trades += 1
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            self.max_consecutive_wins = max(self.max_consecutive_wins, self.consecutive_wins)
            self.last_profitable_trade = datetime.now()
        else:
            self.failed_trades += 1
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)
        
        self.total_pnl += pnl
        self.win_rate = self.calculate_win_rate()

@dataclass
class UserSubscription(BaseModel):
    """اشتراک کاربر"""
    plan_id: str
    start_date: datetime
    end_date: datetime
    is_active: bool = True
    auto_renew: bool = False
    payment_method: Optional[str] = None
    last_payment_date: Optional[datetime] = None
    next_billing_date: Optional[datetime] = None
    
    def is_expired(self) -> bool:
        """بررسی انقضای اشتراک"""
        return datetime.now() > self.end_date
    
    def days_remaining(self) -> int:
        """روزهای باقی‌مانده از اشتراک"""
        if self.is_expired():
            return 0
        return (self.end_date - datetime.now()).days

@dataclass
class User(BaseModel):
    """مدل کاربر"""
    # Basic Information
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    telegram_id: str
    username: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    
    # Status and Role
    role: UserRole = UserRole.FREE
    status: UserStatus = UserStatus.ACTIVE
    is_verified: bool = False
    
    # Preferences and Settings
    preferences: UserPreferences = field(default_factory=UserPreferences)
    notification_settings: NotificationSettings = field(default_factory=NotificationSettings)
    trading_profile: TradingProfile = field(default_factory=TradingProfile)
    
    # Statistics and Tracking
    statistics: UserStatistics = field(default_factory=UserStatistics)
    signals_received_today: int = 0
    signals_received_this_hour: int = 0
    last_activity: Optional[datetime] = None
    last_signal_time: Optional[datetime] = None
    
    # Subscription and Limits
    subscription: Optional[UserSubscription] = None
    limits: UserLimits = field(default_factory=UserLimits)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    
    # Additional Fields
    referral_code: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    referred_by: Optional[str] = None
    notes: Optional[str] = None  # Admin notes
    
    def __post_init__(self):
        """تنظیمات پس از ایجاد کاربر"""
        self.limits = UserLimits.get_limits_by_role(self.role)
        if not self.statistics.registration_date:
            self.statistics.registration_date = self.created_at
    
    @property
    def display_name(self) -> str:
        """نام نمایشی کاربر"""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        elif self.first_name:
            return self.first_name
        elif self.username:
            return f"@{self.username}"
        else:
            return f"User {self.telegram_id}"
    
    @property
    def full_name(self) -> str:
        """نام کامل کاربر"""
        parts = [self.first_name, self.last_name]
        return " ".join(filter(None, parts)) or "Unknown"
    
    def can_receive_signal(self) -> bool:
        """بررسی اینکه آیا کاربر می‌تواند سیگنال دریافت کند"""
        if self.status != UserStatus.ACTIVE:
            return False
        
        # Check daily limit
        if self.limits.max_signals_per_day != -1 and self.signals_received_today >= self.limits.max_signals_per_day:
            return False
        
        # Check hourly limit
        if self.limits.max_signals_per_hour != -1 and self.signals_received_this_hour >= self.limits.max_signals_per_hour:
            return False
        
        # Check subscription
        if self.subscription and self.subscription.is_expired():
            return False
        
        # Check quiet hours
        if self.notification_settings.is_quiet_time(datetime.now()):
            return False
        
        return True
    
    def increment_signal_count(self):
        """افزایش شمارنده سیگنال‌های دریافت شده"""
        self.signals_received_today += 1
        self.signals_received_this_hour += 1
        self.statistics.total_signals_received += 1
        self.last_signal_time = datetime.now()
        self.last_activity = datetime.now()
    
    def reset_daily_counters(self):
        """ریست کردن شمارنده‌های روزانه"""
        self.signals_received_today = 0
    
    def reset_hourly_counters(self):
        """ریست کردن شمارنده‌های ساعتی"""
        self.signals_received_this_hour = 0
    
    def upgrade_role(self, new_role: UserRole):
        """ارتقاء نقش کاربر"""
        self.role = new_role
        self.limits = UserLimits.get_limits_by_role(new_role)
        self.updated_at = datetime.now()
    
    def is_premium_user(self) -> bool:
        """بررسی اینکه آیا کاربر پریمیوم است"""
        return self.role in [UserRole.PREMIUM, UserRole.VIP, UserRole.ADMIN, UserRole.SUPER_ADMIN]
    
    def has_access_to_feature(self, feature: str) -> bool:
        """بررسی دسترسی به ویژگی خاص"""
        feature_access = {
            'premium_signals': self.limits.access_to_premium_features,
            'historical_data': self.limits.access_to_historical_data,
            'priority_support': self.limits.priority_support,
            'custom_alerts': True,  # All users can create custom alerts
            'export_data': self.is_premium_user(),
            'api_access': self.role in [UserRole.VIP, UserRole.ADMIN, UserRole.SUPER_ADMIN],
            'bulk_operations': self.role in [UserRole.ADMIN, UserRole.SUPER_ADMIN]
        }
        return feature_access.get(feature, False)
    
    def get_subscription_status(self) -> str:
        """دریافت وضعیت اشتراک"""
        if not self.subscription:
            return "no_subscription"
        elif self.subscription.is_expired():
            return "expired"
        elif self.subscription.days_remaining() <= 7:
            return "expiring_soon"
        else:
            return "active"
    
    def to_dict(self) -> Dict[str, Any]:
        """تبدیل به دیکشنری برای ذخیره در دیتابیس"""
        return {
            'id': self.id,
            'telegram_id': self.telegram_id,
            'username': self.username,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'email': self.email,
            'phone': self.phone,
            'role': self.role.value,
            'status': self.status.value,
            'is_verified': self.is_verified,
            'preferences': self.preferences.to_dict(),
            'notification_settings': self.notification_settings.to_dict(),
            'trading_profile': self.trading_profile.to_dict(),
            'statistics': self.statistics.to_dict(),
            'signals_received_today': self.signals_received_today,
            'signals_received_this_hour': self.signals_received_this_hour,
            'last_activity': self.last_activity.isoformat() if self.last_activity else None,
            'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time else None,
            'subscription': self.subscription.to_dict() if self.subscription else None,
            'limits': self.limits.to_dict(),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'referral_code': self.referral_code,
            'referred_by': self.referred_by,
            'notes': self.notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """ایجاد از دیکشنری"""
        preferences = UserPreferences.from_dict(data.get('preferences', {}))
        notification_settings = NotificationSettings.from_dict(data.get('notification_settings', {}))
        trading_profile = TradingProfile.from_dict(data.get('trading_profile', {}))
        statistics = UserStatistics.from_dict(data.get('statistics', {}))
        
        subscription = None
        if data.get('subscription'):
            subscription = UserSubscription.from_dict(data['subscription'])
        
        limits = UserLimits.from_dict(data.get('limits', {}))
        
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            telegram_id=data['telegram_id'],
            username=data.get('username'),
            first_name=data.get('first_name'),
            last_name=data.get('last_name'),
            email=data.get('email'),
            phone=data.get('phone'),
            role=UserRole(data.get('role', 'free')),
            status=UserStatus(data.get('status', 'active')),
            is_verified=data.get('is_verified', False),
            preferences=preferences,
            notification_settings=notification_settings,
            trading_profile=trading_profile,
            statistics=statistics,
            signals_received_today=data.get('signals_received_today', 0),
            signals_received_this_hour=data.get('signals_received_this_hour', 0),
            last_activity=datetime.fromisoformat(data['last_activity']) if data.get('last_activity') else None,
            last_signal_time=datetime.fromisoformat(data['last_signal_time']) if data.get('last_signal_time') else None,
            subscription=subscription,
            limits=limits,
            created_at=datetime.fromisoformat(data.get('created_at', datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(data.get('updated_at', datetime.now().isoformat())),
            last_login=datetime.fromisoformat(data['last_login']) if data.get('last_login') else None,
            referral_code=data.get('referral_code', str(uuid.uuid4())[:8]),
            referred_by=data.get('referred_by'),
            notes=data.get('notes')
        )