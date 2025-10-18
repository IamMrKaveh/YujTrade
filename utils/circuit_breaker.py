import asyncio
import time
from enum import Enum
from typing import Callable, Any, Set, Type

from common.exceptions import APIRateLimitError, NetworkError, TradingBotException


class CircuitBreakerError(TradingBotException):
    """Exception raised when the circuit breaker is open."""

    pass


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        half_open_attempts: int = 3,
        failure_exceptions: Set[Type[Exception]] = None,
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.half_open_attempts = half_open_attempts
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self.half_open_success_count = 0
        self._lock = asyncio.Lock()
        self.failure_exceptions = failure_exceptions or {
            APIRateLimitError,
            NetworkError,
        }

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if (time.time() - self.last_failure_time) > self.timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_success_count = 0
                else:
                    raise CircuitBreakerError(
                        f"Circuit breaker is OPEN. Last failure at {self.last_failure_time}"
                    )

        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure(e)
            raise e

    async def _on_success(self):
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_success_count += 1
                if self.half_open_success_count >= self.half_open_attempts:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.last_failure_time = None
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0
                self.last_failure_time = None

    async def _on_failure(self, exc: Exception):
        if not any(isinstance(exc, exc_type) for exc_type in self.failure_exceptions):
            return

        async with self._lock:
            self.last_failure_time = time.time()
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self.failure_count = self.failure_threshold
            elif self.state == CircuitState.CLOSED:
                self.failure_count += 1
                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitState.OPEN
