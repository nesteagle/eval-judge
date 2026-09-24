from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from eval_judge.models import JudgeConfig, OutputSchema
from eval_judge.results import ParsedRecord


@dataclass(frozen=True)
class BatchState:
    batch_id: str
    status: str
    terminal: bool
    completed: int = 0
    failed: int = 0
    total: int = 0
    output_file_id: str | None = None
    error_file_id: str | None = None


@dataclass(frozen=True)
class RetrievedBatch:
    batch_id: str
    status: str
    output_path: Path | None
    error_path: Path | None


@dataclass(frozen=True)
class DryRunReport:
    request_count: int
    file_bytes: int
    est_input_tokens: int
    over_limits: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.over_limits


class Provider(ABC):
    """One batch API: request format, job lifecycle and output parsing.

    System prompt and schema are handed to
    `build_body` for prompt consistency. Providers may adapt
    the schema to what their structured output feature accepts.
    """

    name: str
    max_requests: int
    max_file_bytes: int

    @abstractmethod
    def build_body(
        self,
        config: JudgeConfig,
        system_prompt: str,
        schema: OutputSchema,
        user_content: str,
    ) -> dict: ...

    @abstractmethod
    def build_request_line(self, custom_id: str, body: dict) -> dict: ...

    def validate_custom_id(self, custom_id: str) -> None:
        """Raise ValueError if the provider doesn't accept custom_id."""

    @abstractmethod
    def submit(self, jsonl_path: str | Path) -> str:
        """Uploads the JSONL file and starts a batch, returns batch id."""

    @abstractmethod
    def check(self, batch_id: str) -> BatchState:
        """Returns current batch state without blocking."""

    def cancel(self, batch_id: str) -> BatchState:
        """Asks provider to stop the batch; returns state afterwards."""
        raise NotImplementedError(f"{self.name} provider does not support cancelling")

    def set_max_output_tokens(self, line: dict, n: int) -> dict:
        """Returns a copy of a request line with a different output-token limit. Used for retries."""
        raise NotImplementedError(f"{self.name} provider does not support changing max_output_tokens")

    @abstractmethod
    def download(self, batch_id: str, output_path: str | Path) -> RetrievedBatch:
        """Downloads the output, and error, files of a batch in a terminal state."""

    @abstractmethod
    def parse_record(self, record: dict) -> ParsedRecord:
        """Parses one output or error-file record. Should not raise for bad records."""
