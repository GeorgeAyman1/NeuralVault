from typing import Any


# Frozen instruction block — kept byte-stable so it sits in the cached prefix.
# Volatile memory context is appended AFTER this, never interpolated into it.
SYSTEM_INSTRUCTIONS = (
    "You are NeuralVault, a long-term memory assistant. Answer the user's "
    "questions using the retrieved memories provided below. Ground every "
    "claim in those memories and cite them by their [index] when you use "
    "them. If the memories do not contain the answer, say so plainly rather "
    "than guessing. Be concise and direct."
)


class ContextBuilder:
    """
    Formats retrieved memories into an LLM context block.

    Produces a two-part system prompt:
      1. SYSTEM_INSTRUCTIONS — frozen, cacheable prefix
      2. memory context     — volatile, appended after the cache breakpoint

    A character budget bounds how many memories are included so the context
    never blows past the model's window.
    """

    def __init__(self, max_context_chars: int = 8000):
        self.max_context_chars = max_context_chars

    def build_context(self, memories: list[dict[str, Any]]) -> str:
        """Render retrieved memories into a numbered context block."""
        if not memories:
            return "No relevant memories were found."

        lines: list[str] = []
        used = 0
        for i, mem in enumerate(memories):
            text   = mem.get("text", "").strip()
            source = mem.get("metadata", {}).get("source_file") or \
                     mem.get("metadata", {}).get("source_type", "memory")
            score  = mem.get("score")

            header = f"[{i}] (source: {source}"
            if score is not None:
                header += f", relevance: {float(score):.3f}"
            header += ")"

            entry = f"{header}\n{text}"

            # Stop before exceeding the budget; always include at least one.
            if used + len(entry) > self.max_context_chars and lines:
                break

            lines.append(entry)
            used += len(entry)

        return "\n\n".join(lines)

    def build_system_blocks(self, memories: list[dict[str, Any]]) -> list[dict]:
        """
        Build the system prompt as content blocks for the Messages API.

        Block 0 (frozen instructions) carries cache_control so the prefix is
        cached across requests; block 1 (memory context) is volatile and sits
        after the breakpoint.
        """
        context = self.build_context(memories)
        return [
            {
                "type": "text",
                "text": SYSTEM_INSTRUCTIONS,
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": f"Retrieved memories:\n\n{context}",
            },
        ]
