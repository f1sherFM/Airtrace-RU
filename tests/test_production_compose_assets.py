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
    assert "docs/vps_deployment_runbook.md" in content


def test_production_env_example_contains_sentry_and_alerts_vars():
    content = Path(".env.production.example").read_text(encoding="utf-8")

    assert "SENTRY_DSN=" in content
    assert "SENTRY_ENVIRONMENT=production" in content
    assert "SENTRY_RELEASE=airtrace-v2" in content
    assert "ALERTS_API_KEY=" in content
    assert "TELEGRAM_BOT_TOKEN=" in content


def test_dockerfile_api_exists_with_runtime_prerequisites():
    content = Path("Dockerfile.api").read_text(encoding="utf-8")

    assert "FROM python:3.13-slim" in content
    assert "apt-get install -y --no-install-recommends curl" in content
    assert "COPY requirements.txt /app/requirements.txt" in content
    assert "pip install -r /app/requirements.txt" in content
    assert "COPY . /app" in content
    assert 'CMD ["uvicorn", "main:app"' in content


def test_dockerignore_excludes_local_runtime_state():
    content = Path(".dockerignore").read_text(encoding="utf-8")

    assert ".venv" in content
    assert ".env" in content
    assert "airtrace_local.db" in content
    assert "logs" in content


def test_vps_runbook_contains_clone_env_and_compose_steps():
    content = Path("docs/vps_deployment_runbook.md").read_text(encoding="utf-8")

    assert "git clone" in content
    assert "cp .env.production.example .env.production" in content
    assert "docker compose --env-file .env.production -f docker-compose.prod.yml --profile with-db up -d --build" in content
    assert "curl -fsS http://localhost:8000/health" in content


def test_web_app_uses_environment_api_base_url():
    service_content = Path("application/web/service.py").read_text(encoding="utf-8")
    app_js_content = Path("web/app.js").read_text(encoding="utf-8")
    server_content = Path("web/server.py").read_text(encoding="utf-8")

    assert 'os.getenv("API_BASE_URL", "").strip()' in service_content
    assert '"http://127.0.0.1:8000"' not in service_content
    assert "window.AIRTRACE_API_BASE_URL" in app_js_content
    assert "http://127.0.0.1:8000" not in app_js_content
    assert '_resolve_api_base_url()' in server_content
