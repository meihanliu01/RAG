import time
from collections import OrderedDict
from typing import Any, Callable, Optional


class TTLCache:
    """Bounded LRU cache with per-entry expiry. Used from a single event loop, so no locking."""

    def __init__(self, max_size: int, ttl_s: float, clock: Callable[[], float] = time.monotonic):
        self._max_size = max_size
        self._ttl_s = ttl_s
        self._clock = clock
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        item = self._data.get(key)
        if item is None or item[0] < self._clock():
            self._data.pop(key, None)
            self.misses += 1
            return None
        self._data.move_to_end(key)
        self.hits += 1
        return item[1]

    def set(self, key: str, value: Any) -> None:
        self._data[key] = (self._clock() + self._ttl_s, value)
        self._data.move_to_end(key)
        while len(self._data) > self._max_size:
            self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)
