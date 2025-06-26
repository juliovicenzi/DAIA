import pytest
from sqlalchemy import text

from daia.daia_db.write_db import get_engine


def test_connection():
    with get_engine().connect() as conn:
        conn.execute(text("SELECT 1"))
