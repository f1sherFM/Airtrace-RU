"""
Validation tests for Python SDK starter assets (Issue #32).
"""

from pathlib import Path


def test_python_sdk_has_pip_ready_skeleton_and_version():
    pyproject = Path("sdk/python/pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "airtrace-ru-sdk"' in pyproject
    assert 'version = "0.3.1"' in pyproject
    assert "httpx" in pyproject

    init_file = Path("sdk/python/src/airtrace_sdk/__init__.py").read_text(encoding="utf-8")
    assert "AirTraceClient" in init_file
    assert "AirTraceError" in init_file


def test_python_sdk_client_has_retry_and_error_handling_and_example():
    client = Path("sdk/python/src/airtrace_sdk/client.py").read_text(encoding="utf-8")
    example = Path("examples/python_sdk_example.py").read_text(encoding="utf-8")

    assert "class AirTraceError" in client
    assert "for attempt in range(self.retries + 1)" in client
    assert "httpx.TimeoutException" in client
    assert "httpx.NetworkError" in client
    assert "raise AirTraceError" in client
    assert "get_current" in client
    assert "get_forecast" in client
    assert "get_history_by_city" in client
    assert "with AirTraceClient" in example
