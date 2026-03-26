import asyncio
from db.connection import get_db
from sqlalchemy import text

async def migrate():
    async with get_db() as db:
        await db.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS google_access_token TEXT;"))
        await db.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS google_refresh_token TEXT;"))
        await db.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS google_token_expiry TIMESTAMPTZ;"))
        await db.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS google_email TEXT;"))
        await db.commit()
    print("Migration complete.")

if __name__ == "__main__":
    asyncio.run(migrate())
