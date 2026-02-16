"""
Unit tests for Telegram delivery service (Issue 5.2).
"""

from unittest.mock import AsyncMock, patch

import pytest

from telegram_delivery import TelegramDeliveryService


class _MemoryDeadLetter:
    def __init__(self):
        self.items = []

    async def write(self, event):
        self.items.append(event)


@pytest.mark.asyncio
async def test_telegram_delivery_retries_then_success():
    sink = _MemoryDeadLetter()
    service = TelegramDeliveryService(bot_token="token", max_retries=2, retry_delay_seconds=0.0, dead_letter_sink=sink)

    class _RespOk:
        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True}

    class _RespFail:
        def raise_for_status(self):
            raise RuntimeError("temp error")

    with patch("httpx.AsyncClient.post", side_effect=[RuntimeError("net"), _RespFail(), _RespOk()]):
        result = await service.send_message(chat_id="123", text="hello", event_id="evt-1")

    assert result["status"] == "sent"
    assert result["attempts"] == 3
    assert sink.items == []


@pytest.mark.asyncio
async def test_telegram_delivery_dead_letter_on_exhausted_retries():
    sink = _MemoryDeadLetter()
    service = TelegramDeliveryService(bot_token="token", max_retries=1, retry_delay_seconds=0.0, dead_letter_sink=sink)

    with patch("httpx.AsyncClient.post", side_effect=RuntimeError("permanent net")):
        result = await service.send_message(chat_id="123", text="hello", event_id="evt-2")

    assert result["status"] == "failed"
    assert result["attempts"] == 2
    assert len(sink.items) == 1
    assert sink.items[0]["event_id"] == "evt-2"
