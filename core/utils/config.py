import os
from dataclasses import dataclass, field


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


@dataclass
class Settings:
    """
    Central configuration, sourced from environment variables with defaults.

    Construct once via get_settings(). Keeps env-var parsing in one place so
    the rest of the codebase never reads os.environ directly.
    """

    # LLM
    anthropic_api_key: str | None = field(default=None)
    llm_model: str = "claude-opus-4-8"
    llm_max_tokens: int = 4096

    # Retrieval
    default_top_k: int = 5
    max_context_chars: int = 8000

    # Storage
    db_path: str = "data/embeddings/vectors.npy"
    index_path: str = "data/indexes/memory_ivf"

    # Service
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY"),
            llm_model         = os.environ.get("NEURALVAULT_LLM_MODEL", "claude-opus-4-8"),
            llm_max_tokens    = _env_int("NEURALVAULT_LLM_MAX_TOKENS", 4096),
            default_top_k     = _env_int("NEURALVAULT_DEFAULT_TOP_K", 5),
            max_context_chars = _env_int("NEURALVAULT_MAX_CONTEXT_CHARS", 8000),
            db_path           = os.environ.get("NEURALVAULT_DB_PATH", "data/embeddings/vectors.npy"),
            index_path        = os.environ.get("NEURALVAULT_INDEX_PATH", "data/indexes/memory_ivf"),
            log_level         = os.environ.get("NEURALVAULT_LOG_LEVEL", "INFO"),
        )

    @property
    def llm_available(self) -> bool:
        return bool(self.anthropic_api_key)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the process-wide Settings singleton (lazy, env-sourced)."""
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings
