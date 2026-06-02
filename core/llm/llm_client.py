from typing import Any


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
        model: str = "claude-opus-4-8",
        client: Any | None = None,
        api_key: str | None = None,
        max_tokens: int = 4096,
    ):
        self.model      = model
        self.max_tokens = max_tokens
        self._client    = client
        self._api_key   = api_key

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic  # lazy: module imports fine without the SDK installed
            self._client = (
                anthropic.Anthropic(api_key=self._api_key)
                if self._api_key
                else anthropic.Anthropic()
            )
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
