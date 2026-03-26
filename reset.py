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
        
    # asyncpg expects standard postgresql:// scheme, not postgresql+asyncpg://
    db_url = db_url.replace("+asyncpg", "")
    
    print("⏳ Connecting to the database...")
    try:
        conn = await asyncpg.connect(db_url)
        print("🔗 Connected. Wiping all data...")
        
        # 1. Wipe the Database
        # TRUNCATE CASCADE on users to wipe all dependent tables:
        # conversations, messages, kb_entries, reminders, patches.
        # We list them explicitly for clarity and comprehensive reset.
        tables = ["users", "conversations", "messages", "kb_entries", "reminders", "patches"]
        
        # Check which tables exist first to avoid errors if some weren't created yet
        conn_check = await conn.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        existing_tables = {row['table_name'] for row in conn_check}
        
        to_truncate = [t for t in tables if t in existing_tables]
        
        if to_truncate:
            # We use TRUNCATE CASCADE on the root tables to ensure all relations are wiped.
            truncate_query = f"TRUNCATE {', '.join(to_truncate)} CASCADE;"
            await conn.execute(truncate_query)
            print("✅ Postgres Database successfully wiped!")
            print("   - All Users and Profiles deleted")
            print("   - All Reminders (scheduled & sent) removed")
            print("   - All Chat History and Messages cleared")
            print("   - All Knowledge Base entries & Vector embeddings wiped")
            print("   - All AI suggested Patches deleted")
        else:
            print("ℹ️ No relevant tables found to wipe.")

        await conn.close()
        
        # 2. Wipe Local Physical Files (media_bucket)
        bucket = "media_bucket"
        if os.path.exists(bucket):
            print(f"📁 Emptying local storage: {bucket}/...")
            count = 0
            for item in os.listdir(bucket):
                path = os.path.join(bucket, item)
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                        count += 1
                    elif os.path.isdir(path):
                        shutil.rmtree(path)
                        count += 1
                except Exception as e:
                    print(f"⚠️ Could not remove {path}: {e}")
            print(f"✅ Emptied {bucket}/! Deleted {count} physical files/folders.")
        else:
            print(f"ℹ️ {bucket}/ directory not found, skipping storage wipe.")
            
        print("\n✨ SYSTEM RESET COMPLETE ✨")
        print("Your FastAPI server can stay running, it doesn't need to be restarted!\n")
        
    except Exception as e:
        print(f"❌ Failed to reset database: {e}")

if __name__ == "__main__":
    asyncio.run(reset_db())
