"""Backwards-compatible import path; see `eval_judge.results`."""

from eval_judge.results import (  # noqa: F401
    ParsedRecord,
    Status,
    load_and_parse_results,
    parse_record,
)

# kept for now, but will be deprecated later.
_parse_record = parse_record
