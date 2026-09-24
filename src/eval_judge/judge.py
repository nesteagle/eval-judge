from datetime import UTC, datetime
from pathlib import Path

from eval_judge.batch import build_requests_batch
from eval_judge.manifest import MANIFEST_SUFFIX, manifest_path_for
from eval_judge.models import ContextMessage, JudgeConfig
from eval_judge.providers.base import Provider
from eval_judge.providers.openai import OpenAIProvider
from eval_judge.run import REQUESTS_NAME, Run, RunResults


class Judge:
    """Batch judging with one config: submit runs, then open, retry and read them later.

    Every run is a directory under `work_dir` (optionally named, default by UTC timestamp) holding
    the requests, a manifest and the downloaded output, so nothing is lost if the process
    stops while the provider is still working::

        run = judge.submit(messages, name="pilot")   # returns immediately
        results = judge.open("pilot").results()      # later, from any process
    """

    def __init__(
        self,
        config: JudgeConfig,
        provider: Provider | None = None,
        work_dir: str | Path = "eval-judge-runs",
    ):
        self.config = config
        self.provider = provider or OpenAIProvider()
        self.work_dir = Path(work_dir)

    def submit(
        self, messages: list[ContextMessage], passes: int = 1, name: str | None = None
    ) -> Run:
        """Builds and submits a run without waiting for it; returns its `Run` handle.

        The run goes to `<work_dir>/<name>/`, or a UTC timestamp when no name is given.
        A name whose run was already submitted is refused; open it with `open(name)`.
        """
        return self._submit_into(self._run_dir(name), messages, passes)

    def evaluate(
        self,
        messages: list[ContextMessage],
        passes: int = 1,
        name: str | None = None,
        poll_interval: float = 60,
        timeout: float | None = None,
    ) -> RunResults:
        """Submits a run and blocks until its results are in (batches can take hours)."""
        run = self.submit(messages, passes=passes, name=name)
        return run.results(poll_interval=poll_interval, timeout=timeout)

    def open(self, run: str | Path) -> Run:
        """Opens a run by name (under `work_dir`), run directory or manifest path.

        Refuses runs built with a different rubric than this Judge's config.
        """
        path = Path(run)
        if not path.exists() and (self.work_dir / path).exists():
            path = self.work_dir / path
        opened = Run.load(path, provider=self.provider)
        if opened.manifest.rubric_hash != self.config.rubric_hash():
            raise ValueError(
                f"run {path} was built with a different rubric "
                f"({opened.manifest.rubric_hash[:12]} != {self.config.rubric_hash()[:12]})"
            )
        return opened

    def runs(self) -> list[Run]:
        """Every run under `work_dir`, oldest first (whatever rubric it was built with)."""
        manifests = self.work_dir.glob(f"*/{REQUESTS_NAME}{MANIFEST_SUFFIX}")
        runs = [Run.load(m, provider=self.provider) for m in manifests]
        return sorted(runs, key=lambda r: r.manifest.created_at)

    # kept for now, but will be deprecated later. Prefer submit/open/evaluate.

    def evaluate_batch(
        self,
        messages: list[ContextMessage],
        passes: int = 1,
        run_dir: str | Path | None = None,
        poll_interval: float = 60,
    ) -> RunResults:
        """Like `evaluate`, with an explicit run directory instead of a name."""
        run_dir = Path(run_dir) if run_dir is not None else self._run_dir(None)
        return self._submit_into(run_dir, messages, passes).results(poll_interval=poll_interval)

    def resume(self, manifest_path: str | Path, poll_interval: float = 60) -> RunResults:
        """Waits for (if needed) and returns the results of an earlier run."""
        return self.open(manifest_path).results(poll_interval=poll_interval)

    def _run_dir(self, name: str | None) -> Path:
        return self.work_dir / (name or datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))

    def _submit_into(self, run_dir: Path, messages: list[ContextMessage], passes: int) -> Run:
        requests_path = run_dir / REQUESTS_NAME
        build_requests_batch(
            messages=messages,
            output_path=requests_path,
            config=self.config,
            passes=passes,
            provider=self.provider,
        )
        run = Run.load(manifest_path_for(requests_path), provider=self.provider)
        run.submit()
        return run
