"""Record-by-record parsing of batch output into `ParsedRecord`s with a status each.

A bad record never aborts the parse: failed, refused, truncated and malformed records
are returned alongside the good ones so they can be counted, retried or reported.
"""

import json
import logging
from collections import Counter
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from eval_judge.ids import decode_custom_id

if TYPE_CHECKING:
    from eval_judge.models import JudgeConfig
    from eval_judge.providers.base import Provider

logger = logging.getLogger("eval_judge")

ERRORS_SUFFIX = ".errors.jsonl"


class Status(StrEnum):
    OK = "ok"
    FAILED = "failed"  # request errored or expired; no model output
    REFUSED = "refused"  # model refused to answer
    TRUNCATED = "truncated"  # stopped at max_output_tokens
    INVALID = "invalid"  # output missing, not JSON, or not matching the rubric


@dataclass(frozen=True)
class ParsedRecord:
    custom_id: str | None
    status: Status
    message_id: str | None = None
    run: int | None = None
    facet: str | None = None
    scores: dict[str, int] | None = None
    reasoning_audit: dict[str, str] | None = None
    error: str | None = None
    usage: dict[str, int] | None = None  # input_tokens, output_tokens, reasoning_tokens
    attempt: int = 0  # 0 for the original request, n for the nth retry

    @property
    def ok(self) -> bool:
        return self.status is Status.OK


def make_record(custom_id: str | None, status: Status, **kwargs) -> ParsedRecord:
    """Builds a ParsedRecord, filling message_id/run/facet when the custom_id decodes."""
    try:
        cid = decode_custom_id(custom_id)
        ids = {"message_id": cid.message_id, "run": cid.run, "facet": cid.facet}
    except ValueError:
        ids = {}
    return ParsedRecord(custom_id=custom_id, status=status, **ids, **kwargs)


def record_from_output(custom_id: str | None, text: str) -> ParsedRecord:
    """Parses the judge's JSON text into an OK record, or INVALID if it can't be read."""
    try:
        result = json.loads(text)
    except (TypeError, ValueError) as e:
        return make_record(custom_id, Status.INVALID, error=f"output is not valid JSON: {e}")
    if not isinstance(result, dict):
        return make_record(custom_id, Status.INVALID, error="output is not a JSON object")
    scores = result.get("final_scores")
    audit = result.get("reasoning_audit")
    if not isinstance(scores, dict):
        return make_record(custom_id, Status.INVALID, error="output has no 'final_scores' object")
    return make_record(
        custom_id,
        Status.OK,
        scores=scores,
        reasoning_audit=audit if isinstance(audit, dict) else None,
    )


def validate_record(record: ParsedRecord, config: "JudgeConfig") -> ParsedRecord:
    """Checks an OK record against the rubric; returns it unchanged or as INVALID."""
    if record.status is not Status.OK:
        return record
    problems = []
    keys = config.dimension_keys
    scores = record.scores or {}
    missing = [k for k in keys if k not in scores]
    extra = [k for k in scores if k not in keys]
    if missing:
        problems.append(f"missing scores {missing}")
    if extra:
        problems.append(f"unexpected scores {extra}")
    bad = {
        k: v
        for k, v in scores.items()
        if k in keys and (type(v) is not int or not 1 <= v <= 5)
    }
    if bad:
        problems.append(f"scores not integers 1-5: {bad}")
    audit = record.reasoning_audit or {}
    missing_audit = [k for k in keys if f"{k}_analysis" not in audit]
    if missing_audit:
        problems.append(f"missing analyses for {missing_audit}")
    if not problems:
        return record
    return replace(record, status=Status.INVALID, error="; ".join(problems))


def _default_provider() -> "Provider":
    from eval_judge.providers.openai import OpenAIProvider

    return OpenAIProvider()


def parse_record(
    record: dict,
    config: "JudgeConfig | None" = None,
    provider: "Provider | None" = None,
) -> ParsedRecord:
    """Parses one raw batch output record. Never raises for a malformed record."""
    provider = provider or _default_provider()
    try:
        parsed = provider.parse_record(record)
    except Exception as e:  # a provider bug must not lose the rest of the batch
        custom_id = record.get("custom_id") if isinstance(record, dict) else None
        parsed = make_record(
            custom_id, Status.INVALID, error=f"unreadable record: {type(e).__name__}: {e}"
        )
    return validate_record(parsed, config) if config is not None else parsed


def _parse_file(path: Path, config, provider) -> list[ParsedRecord]:
    records = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except ValueError as e:
                records.append(
                    ParsedRecord(
                        custom_id=None,
                        status=Status.INVALID,
                        error=f"{path.name}:{lineno}: line is not valid JSON: {e}",
                    )
                )
                continue
            records.append(parse_record(raw, config=config, provider=provider))
    return records


def _request_ids(path: Path) -> list[str]:
    """The custom_ids of a requests JSONL file, in file order."""
    ids = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                custom_id = json.loads(line).get("custom_id")
                if custom_id is not None:
                    ids.append(custom_id)
    return ids


def load_and_parse_results(
    input_path: str | Path | None,
    *,
    config: "JudgeConfig | None" = None,
    error_path: str | Path | None = None,
    provider: "Provider | None" = None,
    requests_path: str | Path | None = None,
) -> list[ParsedRecord]:
    """Loads raw batch output, and its error file; returns one ParsedRecord per request.

    If `error_path` is not given, `<input_path>.errors.jsonl` is used when it exists.
    With `config`, OK records are also checked against the rubric (all keys present,
    integer scores 1-5) and become INVALID if they don't match. With `requests_path`,
    every request that has no line in either file is returned as FAILED.
    """
    provider = provider or _default_provider()
    paths = []
    if input_path is not None:
        paths.append(Path(input_path))
        if error_path is None:
            candidate = Path(str(input_path) + ERRORS_SUFFIX)
            if candidate.exists():
                error_path = candidate
    if error_path is not None:
        paths.append(Path(error_path))

    records = []
    for path in paths:
        records.extend(_parse_file(path, config, provider))

    if requests_path is not None:
        seen = {r.custom_id for r in records}
        records.extend(
            make_record(custom_id, Status.FAILED, error="no output or error line for this request")
            for custom_id in _request_ids(Path(requests_path))
            if custom_id not in seen
        )

    counts = Counter(r.status.value for r in records)
    logger.info(
        "Parsed %d records: %s",
        len(records),
        ", ".join(f"{s}={n}" for s, n in sorted(counts.items())) or "none",
    )
    return records


@dataclass
class RunGroup:
    """All runs (passes) of one message facet."""

    message_id: str
    facet: str
    records: list[ParsedRecord] = field(default_factory=list)

    @property
    def n_total(self) -> int:
        return len(self.records)

    @property
    def n_ok(self) -> int:
        return sum(r.ok for r in self.records)

    @property
    def statuses(self) -> dict[str, int]:
        return dict(Counter(r.status.value for r in self.records))

    @property
    def scores_by_dim(self) -> dict[str, list[int]]:
        """Scores from OK runs only, per dimension, in run order."""
        out: dict[str, list[int]] = {}
        for r in sorted(self.records, key=lambda r: r.run or 0):
            if r.ok and r.scores:
                for k, v in r.scores.items():
                    out.setdefault(k, []).append(v)
        return out


def group_runs(records: list[ParsedRecord]) -> dict[tuple[str, str], RunGroup]:
    """Groups records by (message_id, facet), keeping every run and its status.

    Records whose custom_id can't be decoded are skipped here; they are still in the
    list returned by `load_and_parse_results`.
    """
    groups: dict[tuple[str, str], RunGroup] = {}
    for r in records:
        if r.message_id is None or r.facet is None:
            continue
        key = (r.message_id, r.facet)
        groups.setdefault(key, RunGroup(message_id=r.message_id, facet=r.facet)).records.append(r)
    return groups
