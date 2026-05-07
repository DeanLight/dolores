"""Minimal collected test so pytest exits successfully when no ``def test_*`` exist elsewhere."""


def test_smoke() -> None:
    assert True
