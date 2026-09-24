import json
import logging
from pathlib import Path

from eval_judge.ids import encode_custom_id
from eval_judge.manifest import RunManifest
from eval_judge.models import ContextMessage, JudgeConfig
from eval_judge.providers.base import DryRunReport, Provider
from eval_judge.providers.openai import OpenAIProvider

logger = logging.getLogger("eval_judge")


def build_requests_batch(
    messages: list[ContextMessage],
    output_path: str | Path,
    config: JudgeConfig,
    passes: int = 1,
    provider: Provider | None = None,
) -> str:
    """Converts messages into a batch-ready JSONL file and writes its run manifest.

    Each request's custom_id is `{message_id}__{run}__{facet}` (see `decode_custom_id`).
    The manifest is written to `<output_path>.manifest.json`. Refuses (FileExistsError)
    to overwrite requests whose batch is live; a failed batch may be rebuilt.
    """
    if passes < 1:
        raise ValueError(f"passes must be >= 1, got {passes}")
    RunManifest.ensure_unsubmitted(output_path)
    provider = provider or OpenAIProvider()
    system_prompt = config.system_prompt()
    schema = config.output_schema()

    lines = []
    seen = set()
    for message in messages:
        for run in range(passes):
            custom_id = encode_custom_id(message.message_id, run, message.facet)
            if custom_id in seen:
                raise ValueError(
                    f"duplicate request {custom_id!r}: message_id {message.message_id!r} "
                    f"appears more than once with facet {message.facet!r}"
                )
            seen.add(custom_id)
            provider.validate_custom_id(custom_id)
            body = provider.build_body(config, system_prompt, schema, message.formatted())
            lines.append(json.dumps(provider.build_request_line(custom_id, body)) + "\n")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    RunManifest.create(
        provider=provider.name,
        config=config,
        passes=passes,
        request_count=len(lines),
        message_count=len(messages),
        input_path=output_path,
    ).save()

    logger.info(
        "Wrote %d requests (%d messages, %d passes) -> %s",
        len(lines),
        len(messages),
        passes,
        output_path,
    )
    report = dry_run(output_path, provider=provider)
    for problem in report.over_limits:
        logger.warning("%s", problem)
    return str(output_path)


def dry_run(jsonl_path: str | Path, provider: Provider | None = None) -> DryRunReport:
    """Checks a requests file against the provider's batch limits without submitting.

    The token estimate is a rough chars/4 over each request body's prompt text.
    """
    provider = provider or OpenAIProvider()
    path = Path(jsonl_path)
    count = 0
    chars = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            count += 1
            body = json.loads(line).get("body", {})
            chars += len(str(body.get("instructions", ""))) + len(
                json.dumps(body.get("input", ""))
            )
    size = path.stat().st_size
    over = []
    if count > provider.max_requests:
        over.append(f"{count} requests exceeds {provider.name} limit of {provider.max_requests}")
    if size > provider.max_file_bytes:
        over.append(
            f"file is {size / 2**20:.1f} MiB, exceeds {provider.name} limit of "
            f"{provider.max_file_bytes / 2**20:.0f} MiB"
        )
    return DryRunReport(
        request_count=count, file_bytes=size, est_input_tokens=chars // 4, over_limits=over
    )


# kept for now, but will be deprecated later.
def _build_request_body(user_content: str, config: JudgeConfig) -> dict:
    """Backwards-compatible alias (0.1.x): the OpenAI request body for one message."""
    return OpenAIProvider().build_body(
        config, config.system_prompt(), config.output_schema(), user_content
    )
