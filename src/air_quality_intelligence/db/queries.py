from sqlalchemy import text
from sqlalchemy.engine import Engine


def get_table_names(engine: Engine) -> list[str]:
    """Return application table names from the public schema."""
    query = text(
        """
        SELECT tablename
        FROM pg_tables
        WHERE schemaname = 'public'
          AND tablename IN (
              'stations',
              'measurements',
              'weather',
              'daily_aqi',
              'forecasts'
          )
        ORDER BY tablename
        """
    )

    with engine.connect() as connection:
        return list(connection.execute(query).scalars())
