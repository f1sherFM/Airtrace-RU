from core.app_factory import create_api_app


def test_stage1_app_factory_builds_app_and_registers_protected_routes():
    app = create_api_app()
    paths = {route.path for route in app.routes}

    assert "/weather/current" in paths
    assert "/weather/forecast" in paths
    assert "/history" in paths
    assert "/health" in paths
    assert "/v2/current" in paths
    assert "/v2/forecast" in paths
    assert "/v2/history" in paths
    assert "/v2/health" in paths
