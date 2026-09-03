import sys
import os
import pytest

# Clean conflicting system PROJ environment variables if present
os.environ.pop("PROJ_LIB", None)
os.environ.pop("GDAL_DATA", None)

from fastapi.testclient import TestClient

# Add project root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.db.database import init_db


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    """Ensures database tables are initialized before tests run"""
    init_db()


@pytest.fixture
def client():
    """FastAPI TestClient fixture"""
    with TestClient(app) as test_client:
        yield test_client
