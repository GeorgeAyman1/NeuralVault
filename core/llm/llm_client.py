from typing import Any

from core.utils.config import get_settings


class LLMUnavailableError(RuntimeError):
    """Raised when the LLM cannot be used (e.g. no API key configured)."""


class LLMClient:
    """
    Wrapper over a chat LLM for grounded answering, supporting two providers:

      * "anthropic" (default) — Anthropic Messages API, Claude Opus 4.8 with
        adaptive thinking and streaming, system passed as cacheable content
        blocks so the frozen-instruction prefix is cached across requests.
      * "groq" — any OpenAI-compatible Chat Completions endpoint (Groq's free
        tier by default). The content-block system prompt is flattened into a
        single system message; prompt caching does not apply.

    The provider is selected via `provider=` or NEURALVAULT_LLM_PROVIDER.

    The underlying client is injectable (`client=`) so tests run fully offline
    without any SDK or API key. For "anthropic" the injected object mimics the
    SDK (`client.messages.stream(...)`); for "groq" it is a callable taking the
    request payload dict and returning the parsed JSON response dict.
    """

    def __init__(
        self,
        model: str | None = None,
        client: Any | None = None,
        api_key: str | None = None,
        max_tokens: int | None = None,
        provider: str | None = None,
        base_url: str | None = None,
    ):
        settings        = get_settings()
        self.provider   = (provider or settings.llm_provider).lower()
        self.model      = model or settings.llm_model
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self.base_url   = base_url or settings.llm_base_url
        self._client    = client
        if self.provider == "groq":
            self._api_key = api_key or settings.groq_api_key
        else:
            self._api_key = api_key or settings.anthropic_api_key

    def complete(self, system_blocks: list[dict], messages: list[dict]) -> dict:
        """Dispatch to the configured provider; returns {"text", "usage"}."""
        if self.provider == "groq":
            return self._complete_openai_compatible(system_blocks, messages)
        return self._complete_anthropic(system_blocks, messages)

    # ------------------------------------------------------------------ #
    # Anthropic                                                           #
    # ------------------------------------------------------------------ #

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

    def _complete_anthropic(
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

    # ------------------------------------------------------------------ #
    # OpenAI-compatible (Groq)                                            #
    # ------------------------------------------------------------------ #

    def _complete_openai_compatible(
        self,
        system_blocks: list[dict],
        messages: list[dict],
    ) -> dict:
        """
        Call an OpenAI-compatible Chat Completions endpoint.

        The Anthropic-style content-block system prompt is flattened into a
        single system message (cache_control hints are dropped — caching is an
        Anthropic feature). Conversation messages already use the shared
        {"role", "content"} shape, so they pass through unchanged.
        """
        system_text = "\n\n".join(
            b["text"] for b in system_blocks if b.get("type") == "text"
        )
        oai_messages = [{"role": "system", "content": system_text}]
        oai_messages += [
            {"role": m["role"], "content": m["content"]} for m in messages
        ]

        payload = {
            "model":      self.model,
            "max_tokens": self.max_tokens,
            "messages":   oai_messages,
        }
        data = self._post_chat(payload)

        choices = data.get("choices") or []
        text = choices[0]["message"]["content"] if choices else ""

        usage = data.get("usage") or {}
        usage_dict = {
            "input_tokens":                usage.get("prompt_tokens"),
            "output_tokens":               usage.get("completion_tokens"),
            "cache_read_input_tokens":     None,
            "cache_creation_input_tokens": None,
        }
        return {"text": text, "usage": usage_dict}

    def _post_chat(self, payload: dict) -> dict:
        """POST to /chat/completions; returns the parsed JSON response.

        An injected `client` (tests) is treated as a callable taking the
        payload and returning the response dict, keeping this path offline.
        """
        if self._client is not None:
            return self._client(payload)

        if not self._api_key:
            raise LLMUnavailableError(
                "No GROQ_API_KEY configured — chat is unavailable. "
                "Set the environment variable to enable LLM features."
            )
        try:
            import httpx  # lazy: module imports fine without httpx present
        except ImportError as e:
            raise LLMUnavailableError(
                "The 'httpx' package is required for the Groq provider — "
                "run: pip install httpx"
            ) from e

        base = (self.base_url or "https://api.groq.com/openai/v1").rstrip("/")
        resp = httpx.post(
            f"{base}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type":  "application/json",
            },
            json=payload,
            timeout=60.0,
        )
        resp.raise_for_status()
        return resp.json()
