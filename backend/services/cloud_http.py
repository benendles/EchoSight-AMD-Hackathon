"""Small shared async transport. Errors and metrics never include provider bodies."""
from __future__ import annotations

import asyncio
import logging
import json
import math
import os
import time
from typing import Any
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger('echosight.cloud')


def seconds(name: str, default: float, maximum: float = 30) -> float:
    try:
        value = float(os.getenv(name, str(default)))
        return min(maximum, max(0.05, value)) if math.isfinite(value) else default
    except ValueError:
        return default


class CloudError(RuntimeError):
    """Contains only an application-owned, credential-free error code."""


class CloudHTTP:
    def __init__(self, *, base_url: str, headers: dict[str, str], timeout: float,
                 transport: httpx.AsyncBaseTransport | None = None):
        parsed = urlsplit(base_url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise CloudError("invalid_configuration")
        self.timeout = timeout
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/", headers=headers,
            timeout=httpx.Timeout(timeout, connect=min(5, timeout)),
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=4),
            follow_redirects=False, transport=transport,
        )

    async def request(self, method: str, path: str, **kwargs: Any) -> tuple[Any, httpx.Headers]:
        started = time.monotonic()
        outcome = "success"
        try:
            async with asyncio.timeout(self.timeout):
                response = await self.client.request(method, path, **kwargs)
                if not response.is_success:
                    raise CloudError(f"http_{response.status_code}")
                try:
                    return response.json(), response.headers
                except ValueError:
                    raise CloudError("invalid_json") from None
        except (TimeoutError, httpx.TimeoutException):
            outcome = "timeout"
            raise CloudError("timeout") from None
        except httpx.RequestError:
            outcome = "network_error"
            raise CloudError("network_error") from None
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except CloudError as exc:
            outcome = str(exc)
            raise
        finally:
            logger.info("cloud_request service=%s outcome=%s elapsed_ms=%.1f",
                        type(self).__name__, outcome, (time.monotonic() - started) * 1000)

    async def close(self) -> None:
        await self.client.aclose()


def completion_text(data: Any) -> str:
    try:
        choice = data["choices"][0]
        text = choice["message"]["content"]
        if choice.get("finish_reason") not in (None, "stop"):
            raise CloudError("incomplete_response")
        if not isinstance(text, str) or not text.strip() or len(text) > 16000:
            raise CloudError("invalid_response")
        return text.strip()
    except (KeyError, IndexError, TypeError):
        raise CloudError("invalid_response") from None


def completion_json(data: Any) -> Any:
    text = completion_text(data)
    # Some agents format requested JSON as a single fenced JSON block.
    # Accept only that exact presentation wrapper, never extract arbitrary prose.
    if text.startswith('```json\n') and text.endswith('\n```'):
        text = text[len('```json\n'):-len('\n```')]
    try:
        return json.loads(text)
    except ValueError:
        raise CloudError('invalid_response_json') from None
