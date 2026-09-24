"""eval-judge: LLM-as-a-judge scoring for AI systems."""

import logging
import sys
from importlib.metadata import PackageNotFoundError, version

from eval_judge.batch import build_requests_batch, dry_run
from eval_judge.client import (
    cancel_batch,
    check_batch,
    poll_batch,
    retrieve_batch,
    submit_batch,
)
from eval_judge.ids import CustomId, decode_custom_id, encode_custom_id
from eval_judge.judge import Judge
from eval_judge.manifest import RunManifest
from eval_judge.models import ContextMessage, EvalDimension, JudgeConfig, ScoreAnchor
from eval_judge.providers import (
    BatchState,
    DryRunReport,
    OpenAIProvider,
    Provider,
    RetrievedBatch,
)
from eval_judge.results import (
    ParsedRecord,
    RunGroup,
    Status,
    group_runs,
    load_and_parse_results,
    parse_record,
)
from eval_judge.run import Run, RunResults
from eval_judge.rubric import build_output_schema, build_system_prompt

try:
    __version__ = version("eval-judge")
except PackageNotFoundError:
    __version__ = "unknown"


class _FallbackHandler(logging.StreamHandler):
    """Shows progress on stderr only while the application hasn't configured logging."""

    def emit(self, record):
        if not logging.getLogger().handlers:
            super().emit(record)


_AUTO_LEVEL = logging.INFO - 1


class _RespectAppLevel(logging.Filter):
    """While default level is in place, follows the root level once the app configures logging."""

    def filter(self, record):
        if _logger.level != _AUTO_LEVEL:  # application chose level
            return True
        root = logging.getLogger()
        return not root.handlers or record.levelno >= root.getEffectiveLevel()


_logger = logging.getLogger("eval_judge")
if not _logger.handlers:
    _handler = _FallbackHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(_handler)
    _logger.addFilter(_RespectAppLevel())
    if _logger.level == logging.NOTSET:
        _logger.setLevel(_AUTO_LEVEL)

__all__ = [
    "__version__",
    "Judge",
    "Run",
    "RunResults",
    "JudgeConfig",
    "EvalDimension",
    "ScoreAnchor",
    "ContextMessage",
    "build_system_prompt",
    "build_output_schema",
    "build_requests_batch",
    "dry_run",
    "submit_batch",
    "check_batch",
    "poll_batch",
    "retrieve_batch",
    "cancel_batch",
    "load_and_parse_results",
    "parse_record",
    "group_runs",
    "ParsedRecord",
    "RunGroup",
    "Status",
    "RunManifest",
    "CustomId",
    "encode_custom_id",
    "decode_custom_id",
    "Provider",
    "OpenAIProvider",
    "BatchState",
    "RetrievedBatch",
    "DryRunReport",
]
