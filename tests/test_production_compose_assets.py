"""
Validation tests for production compose profile assets (Issue #26).
"""

from pathlib import Path

def test_production_compose_profile_has_required_services_and_policies():
    content = Path("docker-compose.prod.yml").read_text(encoding="utf-8")

    assert "api:" in content
    assert "web:" in content
    assert "redis:" in content
    assert "db:" in content
    assert content.count("restart: unless-stopped") >= 4
    assert content.count("healthcheck:") >= 4
    assert "API_BASE_URL: http://api:8000" in content
    assert "profiles:" in content and "- with-db" in content
    assert "SENTRY_DSN: ${SENTRY_DSN:-}" in content
    assert "SENTRY_ENVIRONMENT: ${SENTRY_ENVIRONMENT:-production}" in content
    assert "SENTRY_RELEASE:" in content


def test_production_compose_doc_contains_startup_commands():
    path = Path("docs/production_compose_profile.md")
    content = path.read_text(encoding="utf-8")

    assert "docker-compose.prod.yml" in content
    assert "--profile with-db" in content
    assert "curl -fsS http://localhost:8000/health" in content
    assert "SENTRY_DSN" in content


def test_production_env_example_contains_sentry_and_alerts_vars():
    content = Path(".env.production.example").read_text(encoding="utf-8")

    assert "SENTRY_DSN=" in content
    assert "SENTRY_ENVIRONMENT=production" in content
    assert "SENTRY_RELEASE=airtrace-v2" in content
    assert "ALERTS_API_KEY=" in content
    assert "TELEGRAM_BOT_TOKEN=" in content


def test_web_app_uses_environment_api_base_url():
    content = Path("application/web/service.py").read_text(encoding="utf-8")
    assert 'os.getenv("API_BASE_URL")' in content
    assert '"http://127.0.0.1:8000"' in content
