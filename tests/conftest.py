from native_cache import close_native_fixture_cache


def pytest_sessionfinish() -> None:
    close_native_fixture_cache()
