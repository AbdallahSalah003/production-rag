"""
Centralized Configuration
Uses Pydantic-settings for validated env variables
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):

    sovereigneg_api_key: str  
    primary_model: str = "gpt-oss-120b"
    fallback_model: str = "gpt-4.1-nano" 

    langsmith_tracing: bool = True
    langsmith_api_key: str = ""
    langsmith_project: str = "production-rag"

    app_env: str = "development"
    log_level: str = "INFO"
    rate_limit: str = "20/minute"
    cache_ttl_seconds: int = 300
    max_retries: int = 3

    model_config  = {"env_file": ".env", "extra": "ignore"}

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance, load once, and reused everywhere."""
    return Settings()



