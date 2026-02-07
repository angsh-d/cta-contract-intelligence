"""Per-provider circuit breaker for LLM API resilience."""

import asyncio
import time
from enum import StrEnum


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    Per-provider circuit breaker (async-safe with asyncio.Lock).

    Thresholds:
    - failure_threshold: 5 failures within window -> OPEN
    - failure_window: 300 seconds (5 minutes)
    - recovery_timeout: 30 seconds before HALF_OPEN
    """

    def __init__(
        self,
        provider_name: str,
        failure_threshold: int = 5,
        failure_window: int = 300,
        recovery_timeout: int = 30,
    ):
        self.provider_name = provider_name
        self.failure_threshold = failure_threshold
        self.failure_window = failure_window
        self.recovery_timeout = recovery_timeout

        self._state = CircuitState.CLOSED
        self._failures: list[float] = []
        self._opened_at: float = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._opened_at > self.recovery_timeout:
                return CircuitState.HALF_OPEN
        return self._state

    async def record_success(self) -> None:
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                self._failures.clear()

    async def record_failure(self) -> None:
        async with self._lock:
            now = time.monotonic()
            self._failures.append(now)
            self._failures = [t for t in self._failures if now - t < self.failure_window]

            if len(self._failures) >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = now

    async def can_execute(self) -> bool:
        async with self._lock:
            if self._state == CircuitState.OPEN:
                if time.monotonic() - self._opened_at > self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
            if self._state in (CircuitState.CLOSED, CircuitState.HALF_OPEN):
                return True
            return False
