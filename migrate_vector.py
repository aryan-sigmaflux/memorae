import asyncio
from db.connection import AsyncSessionLocal
from sqlalchemy import text
import db.connection

async def migrate():
    async with AsyncSessionLocal() as session:
        print("Adding pgvector extension and column...")
        await session.execute(text('CREATE EXTENSION IF NOT EXISTS "vector"'))
        await session.execute(text('ALTER TABLE kb_entries DROP COLUMN IF EXISTS embedding'))
        await session.execute(text('ALTER TABLE kb_entries ADD COLUMN embedding vector(768)'))
        await session.commit()
        print("✅ Database successfully migrated to pgvector!")

if __name__ == "__main__":
    asyncio.run(migrate())
