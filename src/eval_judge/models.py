import dataclasses
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import NewType

logger = logging.getLogger("eval_judge")

_CONTEXT_TAG = "preceding_context"
_MESSAGE_TAG = "message_to_evaluate"

# Keys become JSON-schema property names (`key` and `key_analysis`), so keep them simple.
_KEY_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

FACET_CONTENT = "content"
FACET_REASONING = "reasoning"
FACETS = (FACET_CONTENT, FACET_REASONING)


@dataclass(frozen=True)
class ScoreAnchor:
    score: int
    description: str


@dataclass(frozen=True)
class EvalDimension:
    key: str
    title: str
    definition: str
    penalize: list[str] = field(default_factory=list)
    reward: list[str] = field(default_factory=list)
    anchors: list[ScoreAnchor] = field(default_factory=list)
    caveat: str | None = None

    def __post_init__(self):
        if not isinstance(self.key, str) or not _KEY_PATTERN.match(self.key):
            raise ValueError(
                f"dimension key {self.key!r} must match {_KEY_PATTERN.pattern}"
            )
        if not self.anchors:
            raise ValueError(f"{self.key}: anchors required (one per score 1-5)")
        scores = sorted(a.score for a in self.anchors)
        if scores != [1, 2, 3, 4, 5]:
            raise ValueError(
                f"{self.key} must have exactly one anchor per score 1-5, got {scores}"
            )


@dataclass(frozen=True)
class ContextMessage:
    """Either an agent's message content or its reasoning output, with optional context."""

    message_id: str
    content: str
    reasoning: bool = False
    context: str | None = None

    @property
    def facet(self) -> str:
        """Which part of the agent output this message is: "reasoning" or "content"."""
        return FACET_REASONING if self.reasoning else FACET_CONTENT

    def formatted(self) -> str:
        """The fully wrapped string sent to the model. Always use this over `.content`."""
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
    dimensions: list[EvalDimension]

    def __post_init__(self):
        if not self.dimensions:
            raise ValueError("JudgeConfig requires at least one dimension")
        keys = [d.key for d in self.dimensions]
        duplicates = sorted({k for k in keys if keys.count(k) > 1})
        if duplicates:
            raise ValueError(f"duplicate dimension keys: {duplicates}")
        if not isinstance(self.max_output_tokens, int) or self.max_output_tokens <= 0:
            raise ValueError(
                f"max_output_tokens must be a positive int, got {self.max_output_tokens!r}"
            )

    @property
    def dimension_keys(self) -> list[str]:
        return [d.key for d in self.dimensions]

    def system_prompt(self) -> str:
        """The exact system prompt (instructions) sent to the judge model."""
        from eval_judge.rubric import build_system_prompt

        return build_system_prompt(self.dimensions)

    def output_schema(self) -> OutputSchema:
        """The JSON schema the judge output must follow."""
        from eval_judge.rubric import build_output_schema

        return build_output_schema(self.dimensions)

    def rubric_hash(self) -> str:
        """sha256 of system prompt and output schema. Changes whenever the rubric does."""
        h = hashlib.sha256()
        h.update(self.system_prompt().encode("utf-8"))
        h.update(b"\0")
        h.update(json.dumps(self.output_schema(), sort_keys=True).encode("utf-8"))
        return h.hexdigest()

    def to_dict(self) -> dict:
        """JSON-serialisable snapshot of the config (see `from_dict`)."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "JudgeConfig":
        """Rebuilds a config from `to_dict` output.

        Unknown fields (for example from a manifest written by a newer eval-judge) are
        ignored with a warning rather than failing.
        """
        dimensions = [
            EvalDimension(
                **{
                    **_known_fields(EvalDimension, d),
                    "anchors": [
                        ScoreAnchor(**_known_fields(ScoreAnchor, a)) for a in d.get("anchors", [])
                    ],
                }
            )
            for d in data["dimensions"]
        ]
        return cls(**{**_known_fields(cls, data), "dimensions": dimensions})


def _known_fields(cls, data: dict) -> dict:
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        logger.warning("ignoring unknown %s fields: %s", cls.__name__, unknown)
    return {k: v for k, v in data.items() if k in known}
