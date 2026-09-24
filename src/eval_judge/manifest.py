"""Run manifest: a JSON file next to the requests JSONL that records what was submitted.

It is written when requests are built and updated on submit and retrieve, so a batch
id survives an interrupted process and results stay traceable to their rubric.

File paths inside the manifest are stored relative to the manifest's directory, so a run
directory can be opened from any working directory (or moved) and still resolve.
"""

import dataclasses
import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from eval_judge.models import JudgeConfig

logger = logging.getLogger("eval_judge")

MANIFEST_SUFFIX = ".manifest.json"


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("eval-judge")
    except PackageNotFoundError:
        return "unknown"


def manifest_path_for(jsonl_path: str | Path) -> Path:
    return Path(str(jsonl_path) + MANIFEST_SUFFIX)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class RunManifest:
    provider: str
    config: dict
    rubric_hash: str
    passes: int
    request_count: int
    message_count: int
    input_path: str
    created_at: str
    eval_judge_version: str
    batch_id: str | None = None
    submitted_at: str | None = None
    status: str | None = None
    output_path: str | None = None
    error_path: str | None = None
    retrieved_at: str | None = None
    attempt: int = 0
    parent: str | None = None
    retries: list[str] = dataclasses.field(default_factory=list)
    path: Path | None = dataclasses.field(default=None, compare=False, repr=False)

    @classmethod
    def create(
        cls,
        *,
        provider: str,
        config: JudgeConfig,
        passes: int,
        request_count: int,
        message_count: int,
        input_path: str | Path,
        attempt: int = 0,
        parent: str | None = None,
    ) -> "RunManifest":
        path = manifest_path_for(input_path)
        return cls(
            provider=provider,
            config=config.to_dict(),
            rubric_hash=config.rubric_hash(),
            passes=passes,
            request_count=request_count,
            message_count=message_count,
            input_path=Path(input_path).name,
            created_at=utc_now(),
            eval_judge_version=_version(),
            attempt=attempt,
            parent=parent,
            path=path,
        )

    @property
    def dir(self) -> Path:
        if self.path is None:
            raise ValueError("manifest has no path")
        return self.path.parent

    def resolve(self, p: str | Path | None) -> Path | None:
        """Resolves a path stored in this manifest against the manifest's directory.

        Manifests written before paths became manifest-relative stored them relative to the
        working directory at the time; those are still found when they exist.
        """
        if p is None:
            return None
        p = Path(p)
        if p.is_absolute() or self.path is None:
            return p
        candidate = self.dir / p
        if candidate.exists() or not p.exists():
            return candidate
        return p

    def relative(self, p: str | Path | None) -> str | None:
        """The form a path is stored in: relative to the manifest directory when possible."""
        if p is None:
            return None
        try:
            return os.path.relpath(Path(p).resolve(), self.dir.resolve())
        except ValueError:  # different drive on Windows
            return str(Path(p).resolve())

    @property
    def requests_path(self) -> Path:
        return self.resolve(self.input_path)

    @property
    def retrieved(self) -> bool:
        """True when results were downloaded and every recorded file is still on disk."""
        files = [self.resolve(p) for p in (self.output_path, self.error_path) if p]
        return bool(self.retrieved_at) and all(f.exists() for f in files)

    @property
    def has_live_batch(self) -> bool:
        """True once submitted, unless the batch failed as a whole (nothing ran or was billed)."""
        return bool(self.batch_id) and self.status != "failed"

    @property
    def judge_config(self) -> JudgeConfig:
        """The rubric/config snapshot this run was built with."""
        return JudgeConfig.from_dict(self.config)

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path is not None else self.path
        if path is None:
            raise ValueError("no manifest path given")
        data = {k: v for k, v in dataclasses.asdict(self).items() if k != "path"}
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
        self.path = path
        return path

    @classmethod
    def load(cls, path: str | Path) -> "RunManifest":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        known = {f.name for f in dataclasses.fields(cls)} - {"path"}
        return cls(**{k: v for k, v in data.items() if k in known}, path=path)

    @classmethod
    def ensure_unsubmitted(cls, jsonl_path: str | Path) -> None:
        """Raises FileExistsError if these requests were already submitted as a batch.

        A batch that failed as a whole may be rebuilt over, since nothing in it ran.
        """
        existing = cls.for_requests(jsonl_path)
        if existing is not None and existing.has_live_batch:
            raise FileExistsError(
                f"{jsonl_path} was already submitted as batch {existing.batch_id} "
                f"(status: {existing.status}); open it with Judge.open(...) or "
                f"Run.load({str(existing.dir)!r}), or choose another path. "
                "If it has failed since, check() it first"
            )
        if existing is not None and existing.batch_id:
            logger.warning("replacing failed batch %s in %s", existing.batch_id, existing.path)

    @classmethod
    def for_requests(cls, jsonl_path: str | Path) -> "RunManifest | None":
        path = manifest_path_for(jsonl_path)
        return cls.load(path) if path.exists() else None
