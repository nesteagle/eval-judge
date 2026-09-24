"""A handle on one batch run: submit, check, wait, fetch results, retry and cancel.

A run is a directory holding `requests.jsonl`, its manifest and the downloaded output.
Retries of failed records are child runs in `retry-<n>/` subdirectories; they reuse the
original custom_ids, and `Run.results()` merges them so the latest attempt of each
request wins.
"""

import dataclasses
import json
import logging
import os
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from eval_judge.client import (
    _provider_for,
    cancel_batch,
    check_batch,
    poll_batch,
    retrieve_batch,
    submit_batch,
)
from eval_judge.ids import decode_custom_id
from eval_judge.manifest import MANIFEST_SUFFIX, RunManifest, manifest_path_for
from eval_judge.models import JudgeConfig
from eval_judge.providers.base import BatchState, Provider
from eval_judge.results import (
    ParsedRecord,
    RunGroup,
    Status,
    group_runs,
    load_and_parse_results,
)

logger = logging.getLogger("eval_judge")

REQUESTS_NAME = "requests.jsonl"
RETRY_PREFIX = "retry-"
RETRYABLE = (Status.FAILED, Status.TRUNCATED, Status.INVALID, Status.REFUSED)


class RunResults(list[ParsedRecord]):
    """The parsed records of a run: a plain list, plus summary helpers."""

    @property
    def counts(self) -> dict[str, int]:
        """Number of records per status, e.g. {"ok": 290, "failed": 6}."""
        return dict(Counter(r.status.value for r in self))

    @property
    def not_ok(self) -> list[ParsedRecord]:
        return [r for r in self if not r.ok]

    def groups(self) -> dict[tuple[str, str], RunGroup]:
        """All passes per (message_id, facet); see `group_runs`."""
        return group_runs(self)

    def usage(self) -> dict[str, int]:
        """Token usage summed over every record that reports it."""
        total: Counter[str] = Counter()
        for r in self:
            total.update(r.usage or {})
        return dict(total)

    def summary(self) -> str:
        counts = ", ".join(f"{s}={n}" for s, n in sorted(self.counts.items())) or "none"
        retried = sum(r.attempt > 0 for r in self)
        return f"{len(self)} records: {counts}" + (f" ({retried} from retries)" if retried else "")


class Run:
    """One run on disk: its manifest plus the provider that executes it.

    Open an existing run with `Run.load(run_dir_or_manifest)` (or `Judge.open`). Methods
    that talk to the provider keep the manifest up to date, and `results()` reads the
    downloaded files once they exist, so re-opening a finished run needs no network.
    """

    def __init__(self, manifest: RunManifest, provider: Provider | None = None):
        if manifest.path is None:
            raise ValueError("a Run needs a saved manifest")
        if provider is not None and provider.name != manifest.provider:
            raise ValueError(
                f"run {manifest.dir} was built for provider {manifest.provider!r}, "
                f"not {provider.name!r}"
            )
        manifest.path = manifest.path.absolute()  # the handle keeps working after a chdir
        self.manifest = manifest
        self.provider = _provider_for(manifest, provider)

    @classmethod
    def load(cls, path: str | Path, provider: Provider | None = None) -> "Run":
        """Opens a run from its directory or its manifest path."""
        path = Path(path)
        if path.is_dir():
            default = manifest_path_for(path / REQUESTS_NAME)
            candidates = [default] if default.exists() else sorted(path.glob("*" + MANIFEST_SUFFIX))
            if len(candidates) != 1:
                raise FileNotFoundError(
                    f"expected one *{MANIFEST_SUFFIX} in {path}, found {len(candidates)}"
                )
            path = candidates[0]
        return cls(RunManifest.load(path), provider)

    def __repr__(self) -> str:
        return f"Run({str(self.dir)!r}, status={self.status!r}, batch_id={self.batch_id!r})"

    # properties (no network)

    @property
    def dir(self) -> Path:
        return self.manifest.dir

    @property
    def name(self) -> str:
        return self.dir.name

    @property
    def batch_id(self) -> str | None:
        return self.manifest.batch_id

    @property
    def status(self) -> str | None:
        """The last status recorded in the manifest; use `check()` for a live one."""
        return self.manifest.status

    @property
    def config(self) -> JudgeConfig:
        """The config snapshot this run was built with."""
        return self.manifest.judge_config

    @property
    def attempt(self) -> int:
        return self.manifest.attempt

    @property
    def root(self) -> "Run":
        """The original run (itself, unless this is a retry)."""
        if not self.manifest.parent:
            return self
        parent = os.path.normpath(self.dir / self.manifest.parent)
        return Run.load(parent, provider=self.provider).root

    @property
    def retries(self) -> list["Run"]:
        """Retry runs of this run, oldest first."""
        self._reload()
        return [Run.load(self.dir / name, provider=self.provider) for name in self.manifest.retries]

    def _reload(self) -> None:
        """Re-reads the manifest, since another Run or process may have added retries."""
        self.manifest = RunManifest.load(self.manifest.path)

    # lifecycle

    def submit(self, *, force: bool = False) -> str:
        """Submits the run's requests, returns batch id. Refuses a second submission."""
        batch_id = submit_batch(self.manifest.requests_path, provider=self.provider, force=force)
        self._reload()
        return batch_id

    def check(self) -> BatchState:
        """The batch's live state, without blocking."""
        return check_batch(self.manifest, provider=self.provider)

    def wait(self, poll_interval: float = 60, timeout: float | None = None) -> BatchState:
        """Blocks until the batch reaches a terminal state."""
        return poll_batch(self.manifest, interval=poll_interval, timeout=timeout, provider=self.provider)

    def cancel(self) -> BatchState:
        """Cancels the batch. Finished requests can still be fetched with `results()`."""
        return cancel_batch(self.manifest, provider=self.provider)

    # results

    def results(
        self,
        *,
        wait: bool = True,
        poll_interval: float = 60,
        timeout: float | None = None,
        include_retries: bool = True,
    ) -> RunResults:
        """Parsed records for every request, one per custom_id.

        Downloads the output the first time and reads the local files afterwards. With
        `include_retries`, retry runs are merged in and the latest attempt of each request
        wins (`record.attempt` says which). With `wait=False`, raises if this run hasn't
        finished, and skips retries that haven't. `timeout` applies to each wait.
        """
        self._reload()
        own = self._own_results(wait, poll_interval, timeout)
        if own is None:
            raise RuntimeError(
                f"run {self.name} is still {self.status}; call wait() or results(wait=True)"
            )
        if not include_retries or not self.manifest.retries:
            return RunResults(own)
        return RunResults(self._merge(own, self._retry_results(wait, poll_interval, timeout)))

    def _own_results(
        self, wait: bool, poll_interval: float, timeout: float | None
    ) -> list[ParsedRecord] | None:
        """This run's records, downloading them first if needed; None if not finished."""
        m = self.manifest
        if not m.batch_id:
            raise ValueError(f"run {self.name} was never submitted; call submit()")
        if not m.retrieved:
            if wait:
                self.wait(poll_interval, timeout)
            elif not self.check().terminal:
                return None
            try:
                retrieve_batch(m, wait=False, provider=self.provider)
            except RuntimeError as e:
                if m.status != "failed":
                    raise
                raise RuntimeError(
                    f"{e}. Nothing ran; fix the cause and call run.submit() to submit it again"
                ) from e
        records = load_and_parse_results(
            m.resolve(m.output_path),
            config=self.config,
            error_path=m.resolve(m.error_path),
            provider=self.provider,
            requests_path=m.requests_path,
        )
        if m.attempt:
            records = [dataclasses.replace(r, attempt=m.attempt) for r in records]
        return records

    def _retry_results(
        self, wait: bool, poll_interval: float, timeout: float | None, strict: bool = False
    ) -> list[list[ParsedRecord]]:
        out = []
        for child in sorted(self.retries, key=lambda r: r.attempt):
            if not child.batch_id:
                problem = f"{child.name} was built but never submitted; submit it with run.retries[-1].submit()"
            else:
                records = child._own_results(wait, poll_interval, timeout)
                if records is not None:
                    out.append(records)
                    continue
                problem = f"{child.name} is still {child.status}"
            if strict:
                raise RuntimeError(problem)
            logger.warning("%s; its records are not included yet", problem)
        return out

    @staticmethod
    def _merge(
        original: list[ParsedRecord], attempts: list[list[ParsedRecord]]
    ) -> list[ParsedRecord]:
        merged: dict[str, ParsedRecord] = {}
        unkeyed = []
        for records in [original, *attempts]:
            for r in records:
                if r.custom_id is None:
                    unkeyed.append(r)
                else:
                    merged[r.custom_id] = r  # later attempts replace, keeping the original order
        return [*merged.values(), *unkeyed]

    # retry

    def retry(
        self,
        statuses: Iterable[Status | str] = RETRYABLE,
        *,
        max_output_tokens: int | None = None,
        submit: bool = True,
    ) -> "Run":
        """Builds, and by default submits, a retry run for records that aren't OK.

        The retry reuses the original request lines and custom_ids, taken from the
        original `requests.jsonl`, so no messages are needed. It lives in
        `<run>/retry-<n>/` and is merged into `results()` of the original run. Pass
        `max_output_tokens` to raise the limit, which truncated records usually need; it
        carries forward to later retries unless changed again. Every earlier attempt must
        have finished first.
        """
        root = self.root
        if root is not self:
            return root.retry(statuses, max_output_tokens=max_output_tokens, submit=submit)

        self._reload()
        limit = max_output_tokens
        if limit is None:  # carry forward the latest attempt's limit
            previous = self.retries
            limit = (previous[-1] if previous else self).config.max_output_tokens
        
        # invalid limit is rejected before anything is written
        config = dataclasses.replace(self.config, max_output_tokens=limit)
        wanted = {Status(s) for s in statuses}
        own = self._own_results(wait=False, poll_interval=0, timeout=None)
        if own is None:
            raise RuntimeError(f"run {self.name} is still {self.status}; wait for it before retrying")
        current = self._merge(own, self._retry_results(False, 0, None, strict=True))
        targets = {r.custom_id: r.status for r in current if r.custom_id and r.status in wanted}
        if not targets:
            raise ValueError(f"nothing to retry: no records with status in {sorted(wanted)}")
        n_truncated = sum(s is Status.TRUNCATED for s in targets.values())
        if n_truncated and max_output_tokens is None:
            logger.warning(
                "retrying %d truncated records with the same max_output_tokens (%d); "
                "pass max_output_tokens= to raise it",
                n_truncated,
                limit,
            )

        lines = []
        with self.manifest.requests_path.open(encoding="utf-8") as f:
            for raw in f:
                if not raw.strip():
                    continue
                line = json.loads(raw)
                if line.get("custom_id") in targets:
                    if limit != self.config.max_output_tokens:
                        line = self.provider.set_max_output_tokens(line, limit)
                    lines.append(json.dumps(line) + "\n")

        attempt = len(self.manifest.retries) + 1
        name = f"{RETRY_PREFIX}{attempt}"
        child_path = self.dir / name / REQUESTS_NAME
        RunManifest.ensure_unsubmitted(child_path)
        child_path.parent.mkdir(exist_ok=True)
        child_path.write_text("".join(lines), encoding="utf-8")

        RunManifest.create(
            provider=self.provider.name,
            config=config,
            passes=self.manifest.passes,
            request_count=len(lines),
            message_count=len({_message_id(cid) for cid in targets}),
            input_path=child_path,
            attempt=attempt,
            parent="..",
        ).save()
        self.manifest.retries.append(name)
        self.manifest.save()
        logger.info("Built %s with %d requests -> %s", name, len(lines), child_path)

        child = Run.load(child_path.parent, provider=self.provider)
        if submit:
            child.submit()
        return child


def _message_id(custom_id: str) -> str:
    try:
        return decode_custom_id(custom_id).message_id
    except ValueError:
        return custom_id
