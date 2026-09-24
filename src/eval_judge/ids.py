"""Structured batch custom_ids: ``{message_id}__{run}__{facet}``.

The run and facet are always the last two ``__``-separated parts, so decoding splits
from the right and works even when ``message_id`` itself contains ``-`` or ``__``.
"""

from typing import NamedTuple

from eval_judge.models import FACETS

SEPARATOR = "__"


class CustomId(NamedTuple):
    message_id: str
    run: int
    facet: str


def encode_custom_id(message_id: str, run: int, facet: str) -> str:
    if not message_id:
        raise ValueError("message_id must be a non-empty string")
    if facet not in FACETS:
        raise ValueError(f"facet must be one of {FACETS}, got {facet!r}")
    if not isinstance(run, int) or run < 0:
        raise ValueError(f"run must be a non-negative int, got {run!r}")
    return f"{message_id}{SEPARATOR}{run}{SEPARATOR}{facet}"


def decode_custom_id(custom_id: str) -> CustomId:
    parts = custom_id.rsplit(SEPARATOR, 2) if isinstance(custom_id, str) else []
    if len(parts) != 3 or not parts[0] or not parts[1].isdigit() or parts[2] not in FACETS:
        raise ValueError(
            f"custom_id {custom_id!r} is not of the form "
            f"'<message_id>{SEPARATOR}<run>{SEPARATOR}<{'|'.join(FACETS)}>'"
        )
    return CustomId(message_id=parts[0], run=int(parts[1]), facet=parts[2])
