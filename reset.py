import asyncio
import asyncpg
import os
import shutil
from dotenv import load_dotenv

load_dotenv()

async def reset_db():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("❌ DATABASE_URL not found in .env")
        return
        
    db_url = db_url.replace("+asyncpg", "")
    
    print("⏳ Connecting to the database...")
    try:
        conn = await asyncpg.connect(db_url)
        print("🔗 Connected.")
        
        # 1. Find the User
        user_row = await conn.fetchrow("""
            SELECT id, display_name 
            FROM users 
            WHERE display_name ILIKE '%Namze%' 
               OR display_name = 'Namze'
            LIMIT 1
        """)
        
        if not user_row:
            print("ℹ️ User 'Namze' not found. Nothing to reset.")
            await conn.close()
            return
            
        user_id = user_row['id']
        display_name = user_row['display_name']
        print(f"👤 Found User: {display_name} (ID: {user_id}). Wiping their data...")
        
        # 2. Collect Media URLs to delete physical files
        media_rows = await conn.fetch("""
            SELECT DISTINCT(media_url) FROM (
                SELECT media_url FROM messages WHERE user_id = $1 AND media_url IS NOT NULL
                UNION
                SELECT media_url FROM kb_entries WHERE user_id = $1 AND media_url IS NOT NULL
            ) AS media
        """, user_id)
        media_urls = [row['media_url'] for row in media_rows]

        # 3. Delete from tables in order
        tables_to_delete = ["kb_entries", "messages", "reminders", "patches", "conversations"]
        for table in tables_to_delete:
            await conn.execute(f"DELETE FROM {table} WHERE user_id = $1", user_id)
            print(f"   - Deleted {table}")
            
        await conn.execute("DELETE FROM users WHERE id = $1", user_id)
        print(f"   - Deleted user Profile from database")

        await conn.close()
        
        # 4. Wipe Local Physical Files (media_bucket)
        if media_urls:
            print(f"📁 Deleting {len(media_urls)} physical media files...")
            count = 0
            for url in media_urls:
                # url is usually /media/filename.ext or similar
                # we need to map it to media_bucket/filename.ext
                local_path = url.lstrip("/").replace("media/", "media_bucket/")
                if os.path.exists(local_path):
                    try:
                        os.remove(local_path)
                        count += 1
                    except Exception as e:
                        print(f"⚠️ Could not remove {local_path}: {e}")
            print(f"✅ Deleted {count} physical files.")
        else:
            print("ℹ️ No physical media found to delete.")
            
        print(f"\n✨ USER '{display_name}' RESET COMPLETE ✨")
        
    except Exception as e:
        print(f"❌ Failed to reset user data: {e}")

if __name__ == "__main__":
    asyncio.run(reset_db())
