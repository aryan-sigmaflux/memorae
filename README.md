python .\main.py and ngrok http 8000

./venv/scripts/activate

## Project Structure

```text
memorae/
├── alembic/           # Database migrations
│   ├── env.py
│   ├── versions/
│   └── ...
├── db/                # Database connection and queries
│   ├── connection.py
│   ├── queries.py
│   └── schema.sql
├── jobs/              # Background scheduled tasks
│   └── reminders.py
├── models/            # SQLAlchemy/Pydantic models
│   ├── kb.py
│   └── patch.py
├── routers/           # API Endpoints (FastAPI)
│   ├── auth.py        # Google OAuth flow
│   └── webhook.py     # Telegram Webhook / Logic
├── services/          # Core Business Logic
│   ├── ai.py          # LLM, Intent, Datetime parsing
│   ├── google_auth.py # Google credentials management
│   ├── google_cal.py  # Google Calendar interactions
│   ├── kb.py          # Knowledge base management
│   ├── media.py       # Audio/Image processing
│   ├── telegram.py    # Telegram API wrapper
│   └── toon.py        # AI Persona & Intent Definitions
├── alembic.ini        # Migration config
├── config.py          # Settings & Env vars
├── main.py            # App Entrypoint
├── migrate.py         # Google Auth migration script
├── migrate_vector.py  # PGVector migration script
├── requirements.txt   # Dependencies
├── reset.py           # Database reset tool
└── .env               # Secrets & Config
```