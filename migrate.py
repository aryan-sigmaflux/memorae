import asyncio
from db.connection import get_db, init_db
from sqlalchemy import text

async def migrate():
    print("🚀 Starting migration...")
    
    # 1. Initialize core tables (Users, Conversations, etc.)
    print("📦 Initializing schema from schema.sql...")
    await init_db()
    
    # 2. Apply incremental column updates
    async with get_db() as db:
        print("🔧 Applying incremental column updates...")
        await db.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS google_access_token TEXT;"))
        await db.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS google_refresh_token TEXT;"))
        await db.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS google_token_expiry TIMESTAMPTZ;"))
        await db.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS google_email TEXT;"))
        await db.commit()
    
    print("✅ Migration complete!")

if __name__ == "__main__":
    asyncio.run(migrate())
