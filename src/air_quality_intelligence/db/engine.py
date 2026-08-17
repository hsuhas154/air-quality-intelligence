from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from air_quality_intelligence.config.settings import settings


def get_engine() -> Engine:
    """Create and return the SQLAlchemy database engine."""
    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
    )
