"""Batch lifecycle: submit, check, poll and retrieve.

Every function accepts either a batch id or the path of a run manifest
(`<requests.jsonl>.manifest.json`). When a manifest is available it is updated with the
batch id, status and downloaded file paths, so an interrupted run can be resumed.
"""

import logging
import time
from pathlib import Path

from eval_judge.batch import dry_run
from eval_judge.manifest import MANIFEST_SUFFIX, RunManifest, utc_now
from eval_judge.providers.base import BatchState, Provider, RetrievedBatch
from eval_judge.providers.openai import OpenAIProvider

logger = logging.getLogger("eval_judge")

_PROVIDERS: dict[str, type[Provider]] = {"openai": OpenAIProvider}


def _provider_for(manifest: RunManifest | None, provider: Provider | None) -> Provider:
    if provider is not None:
        return provider
    name = manifest.provider if manifest else "openai"
    if name not in _PROVIDERS:
        raise ValueError(f"unknown provider {name!r}; pass provider= explicitly")
    return _PROVIDERS[name]()


def _resolve(batch: str | Path | RunManifest) -> tuple[str, RunManifest | None]:
    """Accepts a batch id, a manifest path, or a RunManifest; returns (batch_id, manifest)."""
    if isinstance(batch, RunManifest):
        manifest = batch
    elif str(batch).endswith(MANIFEST_SUFFIX) or Path(str(batch)).is_file():
        manifest = RunManifest.load(batch)
    else:
        return str(batch), None
    if not manifest.batch_id:
        raise ValueError(f"manifest {manifest.path} has no batch_id; it was never submitted")
    return manifest.batch_id, manifest


def default_output_path(manifest: RunManifest) -> Path:
    """Where a run's output is saved: `<requests stem>.output.jsonl` next to the requests."""
    return manifest.requests_path.with_suffix(".output.jsonl")


def submit_batch(
    jsonl_path: str | Path,
    provider: Provider | None = None,
    *,
    force: bool = False,
    resubmit: bool = False,
) -> str:
    """Uploads and submits a requests JSONL file as a batch; returns the batch id.

    The file is checked against the provider's limits first (skip with `force=True`).
    The batch id is saved to the run manifest before this returns. A file whose manifest
    already has a batch id is refused, since submitting again pays again; pass
    `resubmit=True` if that is really intended (the old batch id is then replaced). A
    batch recorded as `failed` (rejected as a whole, so nothing ran) may be resubmitted.
    """
    manifest = RunManifest.for_requests(jsonl_path)
    if manifest is not None and manifest.has_live_batch and not resubmit:
        raise ValueError(
            f"{jsonl_path} was already submitted as batch {manifest.batch_id} (status: "
            f"{manifest.status}); use check_batch/retrieve_batch, or pass resubmit=True to "
            "pay for a second batch. If it has failed since, check_batch it first"
        )
    if manifest is not None and manifest.batch_id and not manifest.has_live_batch:
        logger.warning("replacing failed batch %s", manifest.batch_id)
    provider = _provider_for(manifest, provider)
    report = dry_run(jsonl_path, provider=provider)
    if report.over_limits and not force:
        raise ValueError("batch file exceeds provider limits: " + "; ".join(report.over_limits))

    batch_id = provider.submit(jsonl_path)
    if manifest is None:
        logger.warning("no manifest found for %s; batch id is only logged", jsonl_path)
    else:
        manifest.batch_id = batch_id
        manifest.submitted_at = utc_now()
        manifest.status = "submitted"
        manifest.save()
    logger.info(
        "[batch] Submitted. id=%s%s",
        batch_id,
        f" manifest={manifest.path}" if manifest else "",
    )
    return batch_id


def check_batch(batch: str | Path | RunManifest, provider: Provider | None = None) -> BatchState:
    """Returns the batch's current state without blocking."""
    batch_id, manifest = _resolve(batch)
    state = _provider_for(manifest, provider).check(batch_id)
    if manifest is not None and manifest.status != state.status:
        manifest.status = state.status
        manifest.save()
    return state


def poll_batch(
    batch: str | Path | RunManifest,
    interval: float = 60,
    timeout: float | None = None,
    provider: Provider | None = None,
) -> BatchState:
    """Blocks until the batch reaches a terminal state (completed/failed/expired/cancelled)."""
    batch_id, manifest = _resolve(batch)
    provider = _provider_for(manifest, provider)
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        state = check_batch(manifest or batch_id, provider=provider)
        logger.info(
            "[batch] status=%s completed=%d/%d failed=%d",
            state.status,
            state.completed,
            state.total,
            state.failed,
        )
        if state.terminal:
            return state
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"batch {batch_id} still {state.status} after {timeout}s")
        time.sleep(interval)


def retrieve_batch(
    batch: str | Path | RunManifest,
    output_path: str | Path | None = None,
    *,
    wait: bool = True,
    interval: float = 60,
    provider: Provider | None = None,
) -> RetrievedBatch:
    """Downloads batch results, polling first if `wait` (default).

    Output goes to `output_path` and failed requests to `<output_path>.errors.jsonl`.
    Expired or cancelled batches return their partial output. With a manifest,
    `output_path` defaults to `<requests>.output.jsonl` next to the requests file.
    """
    batch_id, manifest = _resolve(batch)
    provider = _provider_for(manifest, provider)
    if output_path is None:
        if manifest is None:
            raise ValueError("output_path is required when retrieving by batch id")
        output_path = default_output_path(manifest)

    if wait:
        poll_batch(manifest or batch_id, interval=interval, provider=provider)
    retrieved = provider.download(batch_id, output_path)

    if manifest is not None:
        manifest.status = retrieved.status
        manifest.output_path = manifest.relative(retrieved.output_path)
        manifest.error_path = manifest.relative(retrieved.error_path)
        manifest.retrieved_at = utc_now()
        manifest.save()
    for p in (retrieved.output_path, retrieved.error_path):
        if p is not None:
            logger.info("Saved batch %s -> %s", "errors" if p == retrieved.error_path else "output", p)
    return retrieved


def cancel_batch(batch: str | Path | RunManifest, provider: Provider | None = None) -> BatchState:
    """Asks the provider to cancel the batch. Output finished so far can still be retrieved."""
    batch_id, manifest = _resolve(batch)
    state = _provider_for(manifest, provider).cancel(batch_id)
    if manifest is not None:
        manifest.status = state.status
        manifest.save()
    logger.info("[batch] Cancel requested. id=%s status=%s", batch_id, state.status)
    return state
