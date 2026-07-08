from dataclasses import dataclass
from typing import NewType

_CONTEXT_TAG = "preceding_context"
_MESSAGE_TAG = "message_to_evaluate"

@dataclass(frozen=True)
class ContextMessage:
    message_id: str
    content: str
    reasoning: bool
    context: str | None = None

    def formatted(self) -> str:
        parts = []
        if self.context is not None:
            parts.append(f"<{_CONTEXT_TAG}>\n{self.context}\n</{_CONTEXT_TAG}>")
        parts.append(f"<{_MESSAGE_TAG}>\n{self.content}\n</{_MESSAGE_TAG}>")
        return "\n\n".join(parts)


OutputSchema = NewType("OutputSchema", dict)


@dataclass(frozen=True)
class ModelConfig:
    model: str
    effort: str
    max_output_tokens: int
    schema: OutputSchema
    system_instructions: str
