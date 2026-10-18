"""Process-local anonymous bearer sessions; frames never have public disk paths."""
from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from threading import Lock


@dataclass(frozen=True)
class Observation:
    id: str
    frame: bytes
    scene: list[dict]
    navigation: dict
    description: str
    captured_at: float
    source: str = "camera"
    depth_available: bool = True

    @property
    def age(self) -> float:
        return max(0, time.monotonic() - self.captured_at)


@dataclass
class Session:
    touched_at: float = field(default_factory=time.monotonic)
    observation: Observation | None = None
    revision: int = 0
    cloud_busy: bool = False
    cloud_last_started: float = -1e10


class SessionStore:
    def __init__(self, ttl: float = 1800, maximum: int = 200):
        self.ttl, self.maximum = ttl, maximum
        self._items: dict[str, Session] = {}
        self._lock = Lock()

    def cleanup(self) -> None:
        with self._lock:
            now = time.monotonic()
            for key in list(self._items):
                if now - self._items[key].touched_at > self.ttl:
                    del self._items[key]

    def create(self) -> str:
        self.cleanup()
        with self._lock:
            if len(self._items) >= self.maximum:
                raise ValueError("Session capacity reached. Try again later.")
            token = secrets.token_urlsafe(32)
            self._items[token] = Session()
            return token

    def get(self, token: str | None) -> Session | None:
        with self._lock:
            item = self._items.get(token or "")
            if item is None:
                return None
            if time.monotonic() - item.touched_at > self.ttl:
                del self._items[token]
                return None
            item.touched_at = time.monotonic()
            return item

    def reserve(self, token: str) -> int:
        with self._lock:
            item = self._items[token]
            item.revision += 1
            return item.revision

    def publish(self, token: str, revision: int, observation: Observation) -> bool:
        with self._lock:
            item = self._items.get(token)
            if item is None or revision != item.revision:
                return False
            item.observation = observation
            return True

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
