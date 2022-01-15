import pytest


@pytest.fixture(autouse=True)
def _calculator_tests_always_use_testdir(testdir):
    pass
