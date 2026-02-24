from fastapi import FastAPI

from rate_limit_middleware import RateLimitMiddleware, get_rate_limit_manager, setup_rate_limiting


def test_manager_binds_and_controls_live_middleware_instance():
    app = FastAPI()
    manager = get_rate_limit_manager()
    manager.middleware = None
    manager._enabled = False

    setup_rate_limiting(app=app, enabled=True, skip_paths=[])

    assert manager.middleware is None

    stack = app.build_middleware_stack()

    assert stack is not None
    assert isinstance(manager.middleware, RateLimitMiddleware)
    assert manager.is_enabled() is True
    assert manager.middleware.enabled is True

    manager.disable()
    assert manager.is_enabled() is False
    assert manager.middleware.enabled is False

    manager.enable()
    assert manager.is_enabled() is True
    assert manager.middleware.enabled is True
