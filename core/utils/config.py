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
    llm_provider: str = "anthropic"           # "anthropic" | "groq" (OpenAI-compatible)
    anthropic_api_key: str | None = field(default=None)
    groq_api_key: str | None = field(default=None)
    llm_model: str = "claude-opus-4-8"
    llm_max_tokens: int = 4096
    llm_base_url: str | None = None           # OpenAI-compatible base URL (groq, etc.)

    # Retrieval
    default_top_k: int = 5
    max_context_chars: int = 8000

    # Storage
    db_path: str = "data/embeddings/vectors.npy"
    index_path: str = "data/indexes/memory_ivf"
    metadata_path: str = "data/processed/metadata.json"

    # Service
    log_level: str = "INFO"

    # Per-provider defaults applied when the corresponding env var is unset.
    _PROVIDER_DEFAULT_MODEL = {
        "anthropic": "claude-opus-4-8",
        "groq":      "llama-3.3-70b-versatile",
    }
    _PROVIDER_DEFAULT_BASE_URL = {
        "groq": "https://api.groq.com/openai/v1",
    }

    @classmethod
    def from_env(cls) -> "Settings":
        provider = os.environ.get("NEURALVAULT_LLM_PROVIDER", "anthropic").lower()
        return cls(
            llm_provider      = provider,
            anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY"),
            groq_api_key      = os.environ.get("GROQ_API_KEY"),
            llm_model         = os.environ.get(
                "NEURALVAULT_LLM_MODEL",
                cls._PROVIDER_DEFAULT_MODEL.get(provider, "claude-opus-4-8"),
            ),
            llm_base_url      = os.environ.get(
                "NEURALVAULT_LLM_BASE_URL",
                cls._PROVIDER_DEFAULT_BASE_URL.get(provider),
            ),
            llm_max_tokens    = _env_int("NEURALVAULT_LLM_MAX_TOKENS", 4096),
            default_top_k     = _env_int("NEURALVAULT_DEFAULT_TOP_K", 5),
            max_context_chars = _env_int("NEURALVAULT_MAX_CONTEXT_CHARS", 8000),
            db_path           = os.environ.get("NEURALVAULT_DB_PATH", "data/embeddings/vectors.npy"),
            index_path        = os.environ.get("NEURALVAULT_INDEX_PATH", "data/indexes/memory_ivf"),
            metadata_path     = os.environ.get("NEURALVAULT_METADATA_PATH", "data/processed/metadata.json"),
            log_level         = os.environ.get("NEURALVAULT_LOG_LEVEL", "INFO"),
        )

    @property
    def active_api_key(self) -> str | None:
        """API key for the currently selected provider."""
        if self.llm_provider == "groq":
            return self.groq_api_key
        return self.anthropic_api_key

    @property
    def llm_available(self) -> bool:
        return bool(self.active_api_key)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the process-wide Settings singleton (lazy, env-sourced)."""
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings
