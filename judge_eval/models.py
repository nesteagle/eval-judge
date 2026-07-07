from dataclasses import dataclass
from typing import NewType


@dataclass(frozen=True)
class ContextMessage:
    message_id: str
    content: str
    reasoning: bool


OutputSchema = NewType("OutputSchema", dict)


@dataclass(frozen=True)
class ModelConfig:
    model: str
    effort: str
    max_output_tokens: int
    schema: OutputSchema
    system_instructions: str
