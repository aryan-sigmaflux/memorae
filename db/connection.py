from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from config import get_settings

settings = get_settings()

# NullPool is preferred for serverless / short-lived containers.
# Swap to AsyncAdaptedQueuePool + pool_size for long-running servers.
engine = create_async_engine(
    settings.database_url,
    echo=not settings.is_production,
    poolclass=NullPool,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


@asynccontextmanager
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency / context manager that yields a DB session and handles commit/rollback."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Run schema.sql against the database (idempotent)."""
    import pathlib
    import re

    schema_path = pathlib.Path(__file__).parent / "schema.sql"
    sql = schema_path.read_text()

    # Split by semicolon, but be careful with the DO block and function
    # A simpler way is to use the raw asyncpg connection which supports multi-statement execution
    async with engine.begin() as conn:
        # Get the underlying asyncpg connection
        raw_conn = await conn.get_raw_connection()
        # Use the underlying driver connection's execute method which supports multiples
        await raw_conn.driver_connection.execute(sql)


async def close_db() -> None:
    await engine.dispose()