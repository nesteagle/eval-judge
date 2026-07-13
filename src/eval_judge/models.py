from dataclasses import dataclass
from typing import NewType

_CONTEXT_TAG = "preceding_context"
_MESSAGE_TAG = "message_to_evaluate"


@dataclass(frozen=True)
class ScoreAnchor:
    score: int
    description: str


@dataclass(frozen=True)
class EvalDimension:
    key: str
    title: str
    definition: str
    penalize: list[str] = None
    reward: list[str] = None
    anchors: list[ScoreAnchor] = None
    caveat: str | None = None

    def __post_init__(self):
        scores = sorted(a.score for a in self.anchors)
        if scores != [1, 2, 3, 4, 5]:
            raise ValueError(
                f"{self.key} must have exactly one anchor per score 1-5, got {scores}"
            )


@dataclass(frozen=True)
class ContextMessage:
    """One of either raw message content or model's reasoning output, paired with optional context."""

    message_id: str
    content: str
    reasoning: bool = False
    context: str | None = None

    def formatted(self) -> str:
        """The fully wrapped string to send to the model, always use this over .content"""
        parts = []
        if self.context is not None:
            parts.append(f"<{_CONTEXT_TAG}>\n{self.context}\n</{_CONTEXT_TAG}>")
        parts.append(f"<{_MESSAGE_TAG}>\n{self.content}\n</{_MESSAGE_TAG}>")
        return "\n\n".join(parts)


OutputSchema = NewType("OutputSchema", dict)


@dataclass(frozen=True)
class JudgeConfig:
    model: str
    effort: str
    max_output_tokens: int
    dimensions = list[EvalDimension]
