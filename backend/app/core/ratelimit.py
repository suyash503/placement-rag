"""Per-IP sliding window limiter.

The public demo answers with a paid LLM behind it, so an unthrottled endpoint is
someone else's free API. In-memory state is fine here because the deployment is a
single container; a multi-instance deploy would need Redis.
"""

import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

from backend.app.core.logging import get_logger

log = get_logger("ratelimit")


class SlidingWindowLimiter:
    def __init__(self, limit: int, window_seconds: int) -> None:
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> tuple[bool, int]:
        now = time.time()
        hits = self._hits[key]

        while hits and now - hits[0] > self.window:
            hits.popleft()

        if len(hits) >= self.limit:
            return False, int(self.window - (now - hits[0])) + 1

        hits.append(now)

        if len(self._hits) > 5000:
            self._evict(now)

        return True, 0

    def _evict(self, now: float) -> None:
        stale = [k for k, v in self._hits.items() if not v or now - v[-1] > self.window]
        for k in stale:
            del self._hits[k]


_chat_limiter = SlidingWindowLimiter(limit=12, window_seconds=300)


def client_ip(request: Request) -> str:
    # Behind a proxy the socket address is the proxy's, so trust the first hop.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def limit_chat(request: Request) -> None:
    ip = client_ip(request)
    allowed, retry_after = _chat_limiter.check(ip)
    if not allowed:
        log.info("rate limited %s", ip)
        raise HTTPException(
            status_code=429,
            detail=f"Too many questions from this address. Try again in {retry_after}s.",
            headers={"Retry-After": str(retry_after)},
        )
