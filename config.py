from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # App
    app_env: str = "development"
    secret_key: str = "4RDhsP6qpINMZdgxgeXx16E0"
    port: int = 8003
    api_base_url: str = "https://memo.sigmaflux.in/api"

    # Database
    database_url: str = "postgresql+asyncpg://user:password@localhost:5432/memorae"

    # Telegram
    telegram_bot_token: str = ""
    telegram_webhook_url: str = ""
    allowed_chat_ids: list[int] = []

    # AI (OpenRouter)
    openrouter_api_key: str = ""
    ai_model: str = "openai/gpt-4o-mini"
    # Cheap/fast model for internal steps (title generation, query rewriting, reranking).
    ai_fast_model: str = "openai/gpt-4o-mini"
    ocr_model: str = "mistralai/mistral-small-3.2-24b-instruct"

    # Embeddings (local Ollama). In Docker this points at the ollama service.
    ollama_base_url: str = "http://localhost:11434/v1"
    embedding_model: str = "nomic-embed-text"

    # Minimum cosine similarity a note must clear to be considered a search hit.
    # Below this, search_notes returns nothing so the model says "not found"
    # instead of answering from an irrelevant note.
    search_min_similarity: float = 0.25

    # Agent loop guards (see Memorae v2 §2.2)
    agent_max_iterations: int = 5
    agent_max_tool_retries: int = 2
    # Per-turn token ceiling. The loop aborts and forces a final reply once the
    # cumulative tokens across iterations exceed this, capping runaway cost.
    agent_max_tokens_per_turn: int = 20000
    # Optional price (USD per 1K tokens) used only to log an estimated turn cost.
    ai_price_per_1k_tokens: float = 0.0

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"
    ai_max_tokens: int = 1024
    openrouter_site_url: str = ""
    openrouter_site_name: str = "Memorae"

    @property
    def openrouter_base_url(self) -> str:
        return "https://openrouter.ai/api/v1"

    # MinIO / S3-compatible object storage (media bucket)
    minio_endpoint: str = ""               # host:port, no scheme, e.g. "localhost:9000"
    minio_access_key: str = ""
    minio_secret_key: str = ""
    minio_bucket: str = "memorae-media"
    minio_secure: bool = False             # True when MinIO is served over HTTPS
    minio_region: str = ""

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