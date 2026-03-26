from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # App
    app_env: str = "development"
    secret_key: str = "changeme"
    api_base_url: str = "http://localhost:8000"

    # Database
    database_url: str = "postgresql+asyncpg://user:password@localhost:5432/memorae"

    # Telegram
    telegram_bot_token: str = ""
    telegram_webhook_url: str = ""
    allowed_chat_ids: list[int] = []

    # AI (OpenRouter)
    openrouter_api_key: str = ""
    ai_model: str = "anthropic/claude-opus-4"
    ocr_model: str = "openai/gpt-4o-mini"
    ai_max_tokens: int = 1024
    openrouter_site_url: str = ""
    openrouter_site_name: str = "Memorae"

    @property
    def openrouter_base_url(self) -> str:
        return "https://openrouter.ai/api/v1"

    # Google
    google_client_id: str = ""
    google_client_secret: str = ""
    google_client_secrets_file: str = "client_secret.json"
    google_redirect_uri: str = "http://localhost:8000/auth/google/callback"

    # Toon / persona
    toon_name: str = "Memo"
    toon_persona: str = "friendly_assistant"

    # Reminders
    reminder_check_interval_minutes: int = 1
    default_timezone: str = "Asia/Kolkata"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()