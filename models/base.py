# models/base.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, TypeVar, Generic, Type, Union
from datetime import datetime, timezone
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
import json
import uuid
import hashlib
from enum import Enum

T = TypeVar('T', bound='BaseModel')

class ValidationError(Exception):
    """خطای اعتبارسنجی مدل"""
    pass

class ModelStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    ARCHIVED = "archived"

@dataclass
class BaseModel(ABC):
    """کلاس پایه برای همه مدل‌ها با قابلیت‌های پیشرفته"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: ModelStatus = ModelStatus.ACTIVE
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """اعتبارسنجی پس از ایجاد"""
        self.validate()
    
    @abstractmethod
    def validate(self) -> None:
        """اعتبارسنجی مدل"""
        if not self.id:
            raise ValidationError("ID نمی‌تواند خالی باشد")
    
    def to_dict(self, exclude_fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """تبدیل مدل به dictionary با قابلیت حذف فیلدها"""
        exclude_fields = exclude_fields or []
        result = {}
        
        for key, value in self.__dict__.items():
            if key in exclude_fields:
                continue
                
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, Decimal):
                result[key] = str(value)
            elif isinstance(value, BaseModel):
                result[key] = value.to_dict()
            elif isinstance(value, list) and value and isinstance(value[0], BaseModel):
                result[key] = [item.to_dict() for item in value]
            elif isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, set):
                result[key] = list(value)
            else:
                result[key] = value
        return result
    
    def to_json(self, **kwargs) -> str:
        """تبدیل مدل به JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, **kwargs)
    
    @classmethod
    @abstractmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """ساخت مدل از dictionary"""
        pass
    
    def update_timestamp(self):
        """به‌روزرسانی زمان تغییر و نسخه"""
        self.updated_at = datetime.now(timezone.utc)
        self.version += 1
    
    def clone(self: T) -> T:
        """کلون کردن مدل با ID جدید"""
        data = self.to_dict(exclude_fields=['id', 'created_at', 'updated_at'])
        return self.__class__.from_dict(data)
    
    def get_hash(self) -> str:
        """محاسبه hash مدل"""
        data = self.to_dict(exclude_fields=['id', 'created_at', 'updated_at', 'version'])
        return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
    
    def add_metadata(self, key: str, value: Any):
        """افزودن metadata"""
        self.metadata[key] = value
        self.update_timestamp()
    
    def get_metadata(self, key: str, default=None):
        """دریافت metadata"""
        return self.metadata.get(key, default)

@dataclass
class ModelValidator:
    """اعتبارسنج عمومی برای مدل‌ها"""
    
    @staticmethod
    def validate_decimal(value: Any, min_value: Optional[Decimal] = None, 
                        max_value: Optional[Decimal] = None) -> Decimal:
        """اعتبارسنجی مقدار decimal"""
        if value is None:
            raise ValidationError("مقدار نمی‌تواند None باشد")
        
        try:
            decimal_value = Decimal(str(value))
        except:
            raise ValidationError(f"مقدار {value} معتبر نیست")
        
        if min_value is not None and decimal_value < min_value:
            raise ValidationError(f"مقدار نباید کمتر از {min_value} باشد")
        
        if max_value is not None and decimal_value > max_value:
            raise ValidationError(f"مقدار نباید بیشتر از {max_value} باشد")
        
        return decimal_value
    
    @staticmethod
    def validate_string(value: str, min_length: int = 1, max_length: int = 255) -> str:
        """اعتبارسنجی رشته"""
        if not isinstance(value, str):
            raise ValidationError("مقدار باید رشته باشد")
        
        if len(value) < min_length:
            raise ValidationError(f"طول رشته نباید کمتر از {min_length} باشد")
        
        if len(value) > max_length:
            raise ValidationError(f"طول رشته نباید بیشتر از {max_length} باشد")
        
        return value.strip()
    
    @staticmethod
    def validate_percentage(value: float) -> float:
        """اعتبارسنجی درصد (0-100)"""
        if not 0 <= value <= 100:
            raise ValidationError("درصد باید بین 0 و 100 باشد")
        return value

class ModelRepository(Generic[T]):
    """Repository pattern برای مدیریت مدل‌ها"""
    
    def __init__(self, model_class: Type[T]):
        self.model_class = model_class
        self._storage: Dict[str, T] = {}
    
    def save(self, model: T) -> T:
        """ذخیره مدل"""
        model.update_timestamp()
        self._storage[model.id] = model
        return model
    
    def find_by_id(self, model_id: str) -> Optional[T]:
        """جستجو بر اساس ID"""
        return self._storage.get(model_id)
    
    def find_all(self, status: Optional[ModelStatus] = None) -> List[T]:
        """دریافت همه مدل‌ها"""
        if status is None:
            return list(self._storage.values())
        return [m for m in self._storage.values() if m.status == status]
    
    def delete(self, model_id: str) -> bool:
        """حذف مدل"""
        if model_id in self._storage:
            del self._storage[model_id]
            return True
        return False
    
    def count(self) -> int:
        """تعداد مدل‌ها"""
        return len(self._storage)