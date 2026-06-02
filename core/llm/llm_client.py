from typing import Any

from core.utils.config import get_settings


class LLMUnavailableError(RuntimeError):
    """Raised when the LLM cannot be used (e.g. no API key configured)."""


class LLMClient:
    """
    Thin wrapper over the Anthropic Messages API for grounded answering.

    Defaults to Claude Opus 4.8 with adaptive thinking and streaming
    (so large answers don't hit the SDK's non-streaming timeout guard).
    The system prompt is passed as content blocks so the frozen-instruction
    prefix is cached across requests.

    The underlying Anthropic client is injectable (`client=`) so tests run
    fully offline without the SDK or an API key. The real client is imported
    lazily only when no client is injected.
    """

    def __init__(
        self,
        model: str | None = None,
        client: Any | None = None,
        api_key: str | None = None,
        max_tokens: int | None = None,
    ):
        settings        = get_settings()
        self.model      = model or settings.llm_model
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self._client    = client
        self._api_key   = api_key or settings.anthropic_api_key

    def _get_client(self) -> Any:
        if self._client is None:
            if not self._api_key:
                raise LLMUnavailableError(
                    "No ANTHROPIC_API_KEY configured — chat is unavailable. "
                    "Set the environment variable to enable LLM features."
                )
            try:
                import anthropic  # lazy: module imports fine without the SDK installed
            except ImportError as e:
                raise LLMUnavailableError(
                    "The 'anthropic' package is not installed — run: pip install anthropic"
                ) from e
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def complete(
        self,
        system_blocks: list[dict],
        messages: list[dict],
    ) -> dict:
        """
        Send one request and return the assistant's text plus token usage.

        Streams the response and assembles the final message via
        get_final_message(), per the SDK's recommended pattern.
        """
        client = self._get_client()

        with client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            thinking={"type": "adaptive"},
            system=system_blocks,
            messages=messages,
        ) as stream:
            final = stream.get_final_message()

        text = "".join(
            block.text for block in final.content if getattr(block, "type", None) == "text"
        )

        usage = getattr(final, "usage", None)
        usage_dict = {}
        if usage is not None:
            usage_dict = {
                "input_tokens":             getattr(usage, "input_tokens", None),
                "output_tokens":            getattr(usage, "output_tokens", None),
                "cache_read_input_tokens":  getattr(usage, "cache_read_input_tokens", None),
                "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", None),
            }

        return {"text": text, "usage": usage_dict}
