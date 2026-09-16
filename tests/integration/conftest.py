"""Everything under tests/integration/ renders Streamlit scripts via AppTest.
Apply the `integration` marker by location so individual files don't have to."""
import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        if "/tests/integration/" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
