# models/notification.py

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
from datetime import datetime, timedelta
from .base import BaseModel
from .signal import Signal

class NotificationType(Enum):
    """نوع اعلان"""
    SIGNAL = "signal"
    MARKET_UPDATE = "market_update"
    SYSTEM = "system"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    PRICE_ALERT = "price_alert"
    PORTFOLIO_UPDATE = "portfolio_update"

class NotificationPriority(Enum):
    """اولویت اعلان"""
    CRITICAL = 5  # بحرانی
    HIGH = 4      # بالا
    MEDIUM = 3    # متوسط
    LOW = 2       # کم
    INFO = 1      # اطلاعاتی

class NotificationStatus(Enum):
    """وضعیت اعلان"""
    PENDING = "pending"      # در انتظار ارسال
    SENT = "sent"           # ارسال شده
    DELIVERED = "delivered"  # تحویل داده شده
    READ = "read"           # خوانده شده
    FAILED = "failed"       # ناموفق
    EXPIRED = "expired"     # منقضی شده

class DeliveryChannel(Enum):
    """کانال تحویل"""
    TELEGRAM = "telegram"
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"

@dataclass
class NotificationSettings:
    """تنظیمات اعلان"""
    channels: List[DeliveryChannel] = field(default_factory=lambda: [DeliveryChannel.TELEGRAM])
    retry_count: int = 3
    retry_delay: int = 300  # ثانیه
    expires_at: Optional[datetime] = None
    schedule_at: Optional[datetime] = None
    silent: bool = False
    disable_web_page_preview: bool = False

@dataclass
class NotificationMetadata:
    """متادیتای اعلان"""
    source: Optional[str] = None
    category: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    custom_data: Dict[str, Any] = field(default_factory=dict)
    tracking_id: Optional[str] = None

@dataclass
class DeliveryAttempt:
    """تلاش تحویل"""
    channel: DeliveryChannel
    attempted_at: datetime
    status: NotificationStatus
    error_message: Optional[str] = None
    response_data: Optional[Dict[str, Any]] = None

@dataclass
class Notification(BaseModel):
    """مدل پیشرفته اعلان"""
    user_telegram_id: str
    type: NotificationType
    title: str
    message: str
    
    # اختیاری
    signal: Optional[Signal] = None
    priority: NotificationPriority = NotificationPriority.MEDIUM
    status: NotificationStatus = NotificationStatus.PENDING
    settings: NotificationSettings = field(default_factory=NotificationSettings)
    metadata: NotificationMetadata = field(default_factory=NotificationMetadata)
    
    # زمان‌بندی
    scheduled_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    read_at: Optional[datetime] = None
    
    # تلاش‌های تحویل
    delivery_attempts: List[DeliveryAttempt] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    
    # پیام‌های چندزبانه
    localized_messages: Dict[str, Dict[str, str]] = field(default_factory=dict)
    
    # گروه‌بندی
    group_id: Optional[str] = None
    parent_notification_id: Optional[str] = None
    
    def __post_init__(self):
        """پس از ایجاد شی"""
        if self.expires_at is None and self.priority == NotificationPriority.CRITICAL:
            # اعلانات بحرانی 24 ساعت اعتبار دارند
            self.expires_at = datetime.now() + timedelta(hours=24)
        elif self.expires_at is None:
            # سایر اعلانات 7 روز اعتبار دارند
            self.expires_at = datetime.now() + timedelta(days=7)
    
    @property
    def is_expired(self) -> bool:
        """بررسی انقضاء"""
        return self.expires_at is not None and datetime.now() > self.expires_at
    
    @property
    def is_scheduled(self) -> bool:
        """بررسی زمان‌بندی"""
        return self.scheduled_at is not None and datetime.now() < self.scheduled_at
    
    @property
    def can_retry(self) -> bool:
        """امکان تلاش مجدد"""
        return (self.retry_count < self.max_retries and 
                not self.is_expired and 
                self.status in [NotificationStatus.PENDING, NotificationStatus.FAILED])
    
    @property
    def last_delivery_attempt(self) -> Optional[DeliveryAttempt]:
        """آخرین تلاش تحویل"""
        return self.delivery_attempts[-1] if self.delivery_attempts else None
    
    @property
    def failed_attempts(self) -> List[DeliveryAttempt]:
        """تلاش‌های ناموفق"""
        return [attempt for attempt in self.delivery_attempts 
                if attempt.status == NotificationStatus.FAILED]
    
    @property
    def successful_attempts(self) -> List[DeliveryAttempt]:
        """تلاش‌های موفق"""
        return [attempt for attempt in self.delivery_attempts 
                if attempt.status in [NotificationStatus.SENT, NotificationStatus.DELIVERED]]
    
    def get_localized_message(self, language: str = 'fa') -> Dict[str, str]:
        """دریافت پیام محلی‌سازی شده"""
        if language in self.localized_messages:
            return self.localized_messages[language]
        return {'title': self.title, 'message': self.message}
    
    def add_delivery_attempt(self, channel: DeliveryChannel, status: NotificationStatus,
                            error_message: Optional[str] = None,
                            response_data: Optional[Dict[str, Any]] = None):
        """افزودن تلاش تحویل"""
        attempt = DeliveryAttempt(
            channel=channel,
            attempted_at=datetime.now(),
            status=status,
            error_message=error_message,
            response_data=response_data
        )
        self.delivery_attempts.append(attempt)
        
        if status == NotificationStatus.FAILED:
            self.retry_count += 1
        elif status == NotificationStatus.SENT:
            self.sent_at = datetime.now()
            self.status = NotificationStatus.SENT
        elif status == NotificationStatus.DELIVERED:
            self.delivered_at = datetime.now()
            self.status = NotificationStatus.DELIVERED
    
    def mark_as_read(self):
        """علامت‌گذاری به عنوان خوانده شده"""
        self.read_at = datetime.now()
        self.status = NotificationStatus.READ
    
    def mark_as_expired(self):
        """علامت‌گذاری به عنوان منقضی"""
        self.status = NotificationStatus.EXPIRED
    
    def add_tag(self, tag: str):
        """افزودن برچسب"""
        if tag not in self.metadata.tags:
            self.metadata.tags.append(tag)
    
    def remove_tag(self, tag: str):
        """حذف برچسب"""
        if tag in self.metadata.tags:
            self.metadata.tags.remove(tag)
    
    def has_tag(self, tag: str) -> bool:
        """بررسی وجود برچسب"""
        return tag in self.metadata.tags
    
    def set_custom_data(self, key: str, value: Any):
        """تنظیم داده سفارشی"""
        self.metadata.custom_data[key] = value
    
    def get_custom_data(self, key: str, default: Any = None) -> Any:
        """دریافت داده سفارشی"""
        return self.metadata.custom_data.get(key, default)
    
    def get_priority_text(self) -> str:
        """دریافت متن اولویت"""
        priority_texts = {
            NotificationPriority.CRITICAL: "بحرانی 🔴",
            NotificationPriority.HIGH: "بالا 🟠",
            NotificationPriority.MEDIUM: "متوسط 🟡",
            NotificationPriority.LOW: "کم 🟢",
            NotificationPriority.INFO: "اطلاعاتی 🔵"
        }
        return priority_texts.get(self.priority, "نامشخص")
    
    def get_status_text(self) -> str:
        """دریافت متن وضعیت"""
        status_texts = {
            NotificationStatus.PENDING: "در انتظار",
            NotificationStatus.SENT: "ارسال شده",
            NotificationStatus.DELIVERED: "تحویل داده شده",
            NotificationStatus.READ: "خوانده شده",
            NotificationStatus.FAILED: "ناموفق",
            NotificationStatus.EXPIRED: "منقضی شده"
        }
        return status_texts.get(self.status, "نامشخص")
    
    def format_for_telegram(self, language: str = 'fa') -> str:
        """فرمت برای تلگرام"""
        localized = self.get_localized_message(language)
        priority_emoji = {
            NotificationPriority.CRITICAL: "🚨",
            NotificationPriority.HIGH: "⚠️",
            NotificationPriority.MEDIUM: "ℹ️",
            NotificationPriority.LOW: "💡",
            NotificationPriority.INFO: "📢"
        }.get(self.priority, "📝")
        
        formatted_message = f"{priority_emoji} *{localized['title']}*\n\n"
        formatted_message += f"{localized['message']}\n"
        
        if self.signal:
            formatted_message += f"\n📈 سیگنال: {self.signal.symbol}\n"
            formatted_message += f"📊 نوع: {self.signal.action}\n"
            formatted_message += f"💰 قیمت: {self.signal.entry_price}\n"
        
        if self.metadata.tags:
            formatted_message += f"\n🏷️ برچسب‌ها: {', '.join(self.metadata.tags)}\n"
        
        formatted_message += f"\n⏰ زمان: {self.created_at.strftime('%Y-%m-%d %H:%M')}"
        
        return formatted_message
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Notification':
        """ایجاد از دیکشنری"""
        # تبدیل تاریخ‌ها
        for date_field in ['scheduled_at', 'expires_at', 'sent_at', 'delivered_at', 'read_at']:
            if data.get(date_field):
                data[date_field] = datetime.fromisoformat(data[date_field])
        
        # تبدیل تنظیمات
        if 'settings' in data:
            settings_data = data['settings']
            if 'channels' in settings_data:
                settings_data['channels'] = [DeliveryChannel(ch) for ch in settings_data['channels']]
            data['settings'] = NotificationSettings(**settings_data)
        
        # تبدیل متادیتا
        if 'metadata' in data:
            data['metadata'] = NotificationMetadata(**data['metadata'])
        
        # تبدیل تلاش‌های تحویل
        if 'delivery_attempts' in data:
            attempts = []
            for attempt_data in data['delivery_attempts']:
                attempt_data['channel'] = DeliveryChannel(attempt_data['channel'])
                attempt_data['status'] = NotificationStatus(attempt_data['status'])
                attempt_data['attempted_at'] = datetime.fromisoformat(attempt_data['attempted_at'])
                attempts.append(DeliveryAttempt(**attempt_data))
            data['delivery_attempts'] = attempts
        
        return cls(
            user_telegram_id=data['user_telegram_id'],
            type=NotificationType(data['type']),
            title=data['title'],
            message=data['message'],
            signal=Signal.from_dict(data['signal']) if data.get('signal') else None,
            priority=NotificationPriority(data.get('priority', NotificationPriority.MEDIUM.value)),
            status=NotificationStatus(data.get('status', NotificationStatus.PENDING.value)),
            settings=data.get('settings', NotificationSettings()),
            metadata=data.get('metadata', NotificationMetadata()),
            scheduled_at=data.get('scheduled_at'),
            expires_at=data.get('expires_at'),
            sent_at=data.get('sent_at'),
            delivered_at=data.get('delivered_at'),
            read_at=data.get('read_at'),
            delivery_attempts=data.get('delivery_attempts', []),
            retry_count=data.get('retry_count', 0),
            max_retries=data.get('max_retries', 3),
            localized_messages=data.get('localized_messages', {}),
            group_id=data.get('group_id'),
            parent_notification_id=data.get('parent_notification_id')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """تبدیل به دیکشنری"""
        result = super().to_dict()
        
        # تبدیل تاریخ‌ها به string
        for date_field in ['scheduled_at', 'expires_at', 'sent_at', 'delivered_at', 'read_at']:
            if getattr(self, date_field):
                result[date_field] = getattr(self, date_field).isoformat()
        
        # تبدیل enum ها
        result['type'] = self.type.value
        result['priority'] = self.priority.value
        result['status'] = self.status.value
        
        # تبدیل تنظیمات
        if self.settings:
            settings_dict = {
                'channels': [ch.value for ch in self.settings.channels],
                'retry_count': self.settings.retry_count,
                'retry_delay': self.settings.retry_delay,
                'silent': self.settings.silent,
                'disable_web_page_preview': self.settings.disable_web_page_preview
            }
            if self.settings.expires_at:
                settings_dict['expires_at'] = self.settings.expires_at.isoformat()
            if self.settings.schedule_at:
                settings_dict['schedule_at'] = self.settings.schedule_at.isoformat()
            result['settings'] = settings_dict
        
        # تبدیل متادیتا
        if self.metadata:
            result['metadata'] = {
                'source': self.metadata.source,
                'category': self.metadata.category,
                'tags': self.metadata.tags,
                'custom_data': self.metadata.custom_data,
                'tracking_id': self.metadata.tracking_id
            }
        
        # تبدیل تلاش‌های تحویل
        if self.delivery_attempts:
            attempts = []
            for attempt in self.delivery_attempts:
                attempt_dict = {
                    'channel': attempt.channel.value,
                    'attempted_at': attempt.attempted_at.isoformat(),
                    'status': attempt.status.value,
                    'error_message': attempt.error_message,
                    'response_data': attempt.response_data
                }
                attempts.append(attempt_dict)
            result['delivery_attempts'] = attempts
        
        return result