from typing import Any


class ConversationMemory:
    """
    Holds a multi-turn conversation as Messages API message dicts.

    Keeps the last `max_turns` exchanges (a turn = one user + one assistant
    message) so the history sent to the model stays bounded. The first
    message must always be a user message, so trimming preserves that.
    """

    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self.messages: list[dict[str, Any]] = []

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})
        self._trim()

    def add_assistant(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})
        self._trim()

    def get_messages(self) -> list[dict[str, Any]]:
        return list(self.messages)

    def clear(self) -> None:
        self.messages = []

    def turn_count(self) -> int:
        return sum(1 for m in self.messages if m["role"] == "user")

    def _trim(self) -> None:
        # Keep at most max_turns * 2 messages; drop oldest pairs first and
        # ensure the surviving history still starts with a user message.
        max_messages = self.max_turns * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]
        while self.messages and self.messages[0]["role"] != "user":
            self.messages.pop(0)
