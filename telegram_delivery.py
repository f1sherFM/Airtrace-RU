"""
Telegram delivery channel with retry and dead-letter handling (Issue 5.2).
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from http_transport import create_external_async_client

logger = logging.getLogger(__name__)


class InMemoryDeliveryStatusStore:
    def __init__(self):
        self._events: List[Dict[str, Any]] = []

    def add(self, event: Dict[str, Any]) -> None:
        self._events.append(event)

    def list_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._events[-limit:]


class JsonlDeadLetterSink:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    async def write(self, event: Dict[str, Any]) -> None:
        line = json.dumps(event, ensure_ascii=False)
        await asyncio.to_thread(self._append_line, line)

    def _append_line(self, line: str) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


class TelegramDeliveryService:
    def __init__(
        self,
        bot_token: str,
        *,
        max_retries: int = 3,
        retry_delay_seconds: float = 0.7,
        trust_env: Optional[bool] = None,
        dead_letter_sink: Optional[JsonlDeadLetterSink] = None,
        status_store: Optional[InMemoryDeliveryStatusStore] = None,
    ):
        self.bot_token = bot_token
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.trust_env = (
            trust_env
            if trust_env is not None
            else os.getenv("TELEGRAM_TRUST_ENV", "true").lower() == "true"
        )
        self.dead_letter_sink = dead_letter_sink or JsonlDeadLetterSink("logs/telegram_dead_letter.jsonl")
        self.status_store = status_store or InMemoryDeliveryStatusStore()

    @property
    def enabled(self) -> bool:
        return bool(self.bot_token)

    @staticmethod
    def _format_error(exc: Exception) -> str:
        message = str(exc).strip()
        if message:
            return message
        return exc.__class__.__name__

    async def send_message(self, chat_id: str, text: str, event_id: Optional[str] = None) -> Dict[str, Any]:
        attempts = 0
        last_error: Optional[str] = None
        if not self.enabled:
            result = {
                "channel": "telegram",
                "status": "failed",
                "attempts": 1,
                "event_id": event_id,
                "error": "telegram_not_configured",
            }
            self.status_store.add(result)
            return result

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        async with create_external_async_client(
            timeout_seconds=15.0,
            trust_env=self.trust_env,
            max_connections=10,
            max_keepalive_connections=5,
        ) as client:
            for i in range(self.max_retries + 1):
                attempts += 1
                try:
                    resp = await client.post(url, json={"chat_id": chat_id, "text": text})
                    resp.raise_for_status()
                    data = resp.json()
                    if not data.get("ok", False):
                        raise RuntimeError(f"telegram_api_error: {data}")
                    result = {"channel": "telegram", "status": "sent", "attempts": attempts, "event_id": event_id}
                    self.status_store.add(result)
                    return result
                except Exception as exc:
                    last_error = self._format_error(exc)
                    logger.warning(
                        "Telegram delivery attempt failed: attempt=%s/%s chat_id=%s event_id=%s error=%s",
                        attempts,
                        self.max_retries + 1,
                        chat_id,
                        event_id,
                        last_error,
                    )
                    if i < self.max_retries:
                        await asyncio.sleep(self.retry_delay_seconds)
                        continue

        failed = {
            "channel": "telegram",
            "status": "failed",
            "attempts": attempts,
            "event_id": event_id,
            "error": last_error or "unknown_delivery_error",
        }
        self.status_store.add(failed)
        await self.dead_letter_sink.write(
            {
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "channel": "telegram",
                "event_id": event_id,
                "chat_id": chat_id,
                "error": failed["error"],
                "attempts": attempts,
            }
        )
        return failed
