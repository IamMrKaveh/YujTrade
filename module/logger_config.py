import asyncio
import logging
import os
import platform
import sys
import threading
import traceback
from datetime import datetime
from functools import wraps
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

import psutil


class DetailedFormatter(logging.Formatter):
    """فرمتر سفارشی برای نمایش جزئیات بیشتر"""
    
    def format(self, record):
        # اضافه کردن اطلاعات سیستم
        record.pid = os.getpid()
        record.thread_id = threading.get_ident()
        record.thread_name = threading.current_thread().name
        
        # اضافه کردن اطلاعات حافظه با try-except محکم‌تر
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            record.memory_mb = round(memory_info.rss / 1024 / 1024, 2) if hasattr(memory_info, 'rss') else 0
            
            # CPU percent گاهی None برمی‌گرداند
            cpu_percent = process.cpu_percent()
            record.cpu_percent = round(cpu_percent, 2) if cpu_percent is not None else 0
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            record.memory_mb = 0
            record.cpu_percent = 0
        
        # اضافه کردن stack trace فقط برای خطاها و فقط اگر از قبل موجود نباشد
        if record.levelno >= logging.ERROR and record.exc_info is None and not hasattr(record, 'stack_info'):
            try:
                record.stack_info = ''.join(traceback.format_stack()[:-1])
            except Exception:
                record.stack_info = "Stack trace unavailable"
        
        return super().format(record)

class YujTradeLogger:
    """کلاس مدیریت لاگ با قابلیت‌های پیشرفته"""
    
    _instances = {}  # برای جلوگیری از ایجاد چندین instance
    _lock = threading.Lock()
    
    def __new__(cls, name="YujTrade", log_dir="logs"):
        # Singleton pattern برای هر name
        with cls._lock:
            key = f"{name}_{log_dir}"
            if key not in cls._instances:
                cls._instances[key] = super().__new__(cls)
            return cls._instances[key]
    
    def __init__(self, name="YujTrade", log_dir="logs"):
        # جلوگیری از تکرار initialization
        if hasattr(self, '_initialized'):
            return
        
        self.log_dir = log_dir
        self.logger_name = name
        self.logger = None
        self.perf_logger = None
        self._initialized = False
        
        try:
            self._setup_directories()
            self._setup_logger()
            self._initialized = True
        except Exception as e:
            print(f"خطا در راه‌اندازی logger: {e}")
            # ایجاد یک basic logger در صورت خطا
            self._setup_basic_logger()
    
    def _setup_directories(self):
        """ایجاد دایرکتوری‌های مورد نیاز با error handling"""
        directories = [
            self.log_dir,
            os.path.join(self.log_dir, "errors"),
            os.path.join(self.log_dir, "debug"),
            os.path.join(self.log_dir, "performance")
        ]
        
        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
            except PermissionError:
                raise PermissionError(f"دسترسی کافی برای ایجاد دایرکتوری {directory} وجود ندارد")
            except Exception as e:
                raise Exception(f"خطا در ایجاد دایرکتوری {directory}: {e}")
    
    def _setup_basic_logger(self):
        """راه‌اندازی logger ساده در صورت خطا"""
        self.logger = logging.getLogger(self.logger_name)
        self.logger.setLevel(logging.DEBUG)
        
        if not self.logger.handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        self.perf_logger = self.logger
    
    def _setup_logger(self):
        """تنظیم logger اصلی"""
        self.logger = logging.getLogger(self.logger_name)
        
        # جلوگیری از تکرار handler ها
        if self.logger.handlers:
            return
        
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False  # جلوگیری از ارسال به parent loggers
        
        # فرمتر پیشرفته برای فایل‌ها
        detailed_formatter = DetailedFormatter(
            fmt='%(asctime)s | PID:%(pid)s | Thread:%(thread_name)s(%(thread_id)s) | '
                'MEM:%(memory_mb)sMB | CPU:%(cpu_percent)s%% | '
                '%(name)s | %(levelname)-8s | %(filename)s:%(lineno)d:%(funcName)s() | '
                '%(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # فرمتر ساده‌تر برای کنسول
        console_formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # تابع کمکی برای ایجاد file handler
        def create_file_handler(filepath, max_bytes, backup_count, level, formatter):
            try:
                handler = RotatingFileHandler(
                    filepath,
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding='utf-8'
                )
                handler.setLevel(level)
                handler.setFormatter(formatter)
                return handler
            except Exception as e:
                print(f"خطا در ایجاد file handler برای {filepath}: {e}")
                return None
        
        # Handler اصلی با rotating
        main_handler = create_file_handler(
            os.path.join(self.log_dir, "app.log"),
            10*1024*1024,  # 10MB
            5,
            logging.INFO,
            detailed_formatter
        )
        
        # Handler برای خطاها
        error_handler = create_file_handler(
            os.path.join(self.log_dir, "errors", "errors.log"),
            5*1024*1024,  # 5MB
            10,
            logging.ERROR,
            detailed_formatter
        )
        
        # Handler برای debug (روزانه)
        debug_handler = None
        try:
            debug_handler = TimedRotatingFileHandler(
                os.path.join(self.log_dir, "debug", "debug.log"),
                when='midnight',
                interval=1,
                backupCount=7,
                encoding='utf-8'
            )
            debug_handler.setLevel(logging.DEBUG)
            debug_handler.setFormatter(detailed_formatter)
        except Exception as e:
            print(f"خطا در ایجاد debug handler: {e}")
        
        # Handler برای performance
        performance_handler = create_file_handler(
            os.path.join(self.log_dir, "performance", "performance.log"),
            5*1024*1024,
            3,
            logging.INFO,
            detailed_formatter
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)  # کم کردن سطح console برای کاهش noise
        console_handler.setFormatter(console_formatter)
        
        # اضافه کردن handler های موجود
        handlers_to_add = [console_handler]
        if main_handler:
            handlers_to_add.append(main_handler)
        if error_handler:
            handlers_to_add.append(error_handler)
        if debug_handler:
            handlers_to_add.append(debug_handler)
        
        for handler in handlers_to_add:
            self.logger.addHandler(handler)
        
        # Performance logger جداگانه
        self.perf_logger = logging.getLogger(f"{self.logger_name}_Performance")
        self.perf_logger.setLevel(logging.INFO)
        self.perf_logger.propagate = False
        
        if performance_handler:
            self.perf_logger.addHandler(performance_handler)
        else:
            # fallback به console
            self.perf_logger.addHandler(console_handler)
        
        # لاگ اولیه سیستم
        self._log_system_info()
    
    def _log_system_info(self):
        """لاگ اطلاعات سیستم در شروع"""
        try:
            self.logger.info("="*80)
            self.logger.info("YujTrade Application Started")
            self.logger.info(f"Python Version: {sys.version}")
            self.logger.info(f"Platform: {platform.platform()}")
            
            # CPU count با error handling
            try:
                cpu_count = psutil.cpu_count()
                self.logger.info(f"CPU Count: {cpu_count}")
            except Exception:
                self.logger.info("CPU Count: Unable to determine")
            
            # Memory info با error handling
            try:
                total_memory = psutil.virtual_memory().total / 1024**3
                self.logger.info(f"Total Memory: {total_memory:.2f} GB")
            except Exception:
                self.logger.info("Total Memory: Unable to determine")
            
            self.logger.info(f"Process ID: {os.getpid()}")
            self.logger.info("="*80)
        except Exception as e:
            print(f"خطا در لاگ اطلاعات سیستم: {e}")
    
    def get_logger(self):
        """برگرداندن logger اصلی"""
        return self.logger
    
    def get_performance_logger(self):
        """برگرداندن performance logger"""
        return self.perf_logger
    
    def log_function_call(self, func):
        @wraps(func)
        def _sync_wrapper(*args, **kwargs):
            start_time = datetime.now()
            func_name = f"{func.__module__}.{func.__qualname__}"
            try:
                safe_args = str(args)
                safe_kwargs = str(kwargs)
            except Exception:
                safe_args = "args_unserializable"
                safe_kwargs = "kwargs_unserializable"
            try:
                result = func(*args, **kwargs)
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                try:
                    if self.logger:
                        self.logger.debug(f"CALL START: {func_name} with args={safe_args}, kwargs={safe_kwargs}")
                        self.logger.debug(f"CALL SUCCESS: {func_name} completed in {duration:.4f}s")
                except Exception:
                    pass
                try:
                    if duration > 0.1 and self.perf_logger:
                        self.perf_logger.info(f"PERFORMANCE: {func_name} executed in {duration:.4f}s")
                except Exception:
                    pass
                return result
            except Exception as e:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                try:
                    if self.logger:
                        self.logger.error(f"CALL ERROR: {func_name} failed after {duration:.4f}s - {str(e)}", exc_info=True)
                except Exception:
                    pass
                raise
        @wraps(func)
        async def _async_wrapper(*args, **kwargs):
            start_time = datetime.now()
            func_name = f"{func.__module__}.{func.__qualname__}"
            try:
                safe_args = str(args)
                safe_kwargs = str(kwargs)
            except Exception:
                safe_args = "args_unserializable"
                safe_kwargs = "kwargs_unserializable"
            try:
                result = await func(*args, **kwargs)
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                try:
                    if self.logger:
                        self.logger.debug(f"CALL START: {func_name} with args={safe_args}, kwargs={safe_kwargs}")
                        self.logger.debug(f"CALL SUCCESS: {func_name} completed in {duration:.4f}s")
                except Exception:
                    pass
                try:
                    if duration > 0.1 and self.perf_logger:
                        self.perf_logger.info(f"PERFORMANCE: {func_name} executed in {duration:.4f}s")
                except Exception:
                    pass
                return result
            except Exception as e:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                try:
                    if self.logger:
                        self.logger.error(f"CALL ERROR: {func_name} failed after {duration:.4f}s - {str(e)}", exc_info=True)
                except Exception:
                    pass
                raise
        if asyncio.iscoroutinefunction(func):
            return _async_wrapper
        return _sync_wrapper
    
    def log_with_context(self, level, message, **context):
        """لاگ با context اضافی"""
        if context:
            # فیلتر کردن مقادیر None و تبدیل به string
            filtered_context = {k: v for k, v in context.items() if v is not None}
            if filtered_context:
                context_str = " | ".join([f"{k}={v}" for k, v in filtered_context.items()])
                message = f"{message} | CONTEXT: {context_str}"
        
        self.logger.log(level, message)
    
    def log_api_call(self, endpoint, method, status_code, response_time, **extra):
        """لاگ مخصوص API calls"""
        # اعتبارسنجی ورودی‌ها
        if not isinstance(response_time, (int, float)):
            response_time = 0.0
        
        self.log_with_context(
            logging.INFO,
            f"API_CALL: {method} {endpoint} -> {status_code} ({response_time:.3f}s)",
            **extra
        )
    
    def log_trade_action(self, action, symbol, quantity, price, **extra):
        """لاگ مخصوص اقدامات معاملاتی"""
        # اعتبارسنجی ورودی‌ها
        try:
            quantity = float(quantity) if quantity is not None else 0.0
            price = float(price) if price is not None else 0.0
        except (ValueError, TypeError):
            quantity = 0.0
            price = 0.0
        
        self.log_with_context(
            logging.INFO,
            f"TRADE: {action} {quantity} {symbol} @ {price}",
            **extra
        )
    
    def log_error_with_details(self, error, context=None):
        """لاگ خطا با جزئیات کامل"""
        error_details = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.now().isoformat(),
        }
        
        if context:
            error_details.update(context)
        
        self.logger.error(
            f"DETAILED_ERROR: {error_details}",
            exc_info=True
        )
    
    def close(self):
        """بستن تمام handlers"""
        if self.logger:
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)
        
        if self.perf_logger and self.perf_logger != self.logger:
            for handler in self.perf_logger.handlers[:]:
                handler.close()
                self.perf_logger.removeHandler(handler)

# ایجاد instance سراسری با error handling
try:
    yuj_logger = YujTradeLogger()
    logger = yuj_logger.get_logger()
    perf_logger = yuj_logger.get_performance_logger()

except Exception as e:
    print(f"خطا در ایجاد logger سراسری: {e}")
    # ایجاد logger ساده در صورت خطا
    logger = logging.getLogger("YujTrade_Fallback")
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(console_handler)
    perf_logger = logger

# مثال‌های استفاده:
if __name__ == "__main__":
    try:
        # لاگ‌های معمولی
        logger.debug("این یک پیام debug است")
        logger.info("شروع پردازش داده‌ها")
        logger.warning("هشدار: حافظه در حال پر شدن")
        
        # لاگ با context
        yuj_logger.log_with_context(
            logging.INFO,
            "کاربر وارد شد",
            user_id=123,
            ip_address="192.168.1.1",
            session_id="abc123"
        )
        
        # لاگ API call
        yuj_logger.log_api_call(
            endpoint="/api/v1/orders",
            method="POST",
            status_code=200,
            response_time=0.245,
            user_id=123
        )
        
        # لاگ trade action
        yuj_logger.log_trade_action(
            action="BUY",
            symbol="BTCUSDT",
            quantity=0.001,
            price=45000,
            order_id="ORD123",
            strategy="DCA"
        )
        
        # استفاده از دکوریتر
        @yuj_logger.log_function_call
        def example_function(x, y):
            """تابع نمونه برای تست دکوریتر"""
            import time
            time.sleep(0.1)  # شبیه‌سازی پردازش
            return x + y
        
        result = example_function(5, 3)
        logger.info(f"نتیجه: {result}")
        
        # لاگ خطا با جزئیات
        try:
            raise ValueError("خطای نمونه")
        except Exception as e:
            yuj_logger.log_error_with_details(
                e,
                context={"operation": "test", "data": {"x": 1, "y": 2}}
            )
        
        print("تست با موفقیت انجام شد!")
        
    except Exception as e:
        print(f"خطا در اجرای تست: {e}")
    
    finally:
        # بستن logger
        try:
            yuj_logger.close()
        except:
            pass