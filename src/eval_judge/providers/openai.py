import logging
from dataclasses import replace
from pathlib import Path

from openai import OpenAI, OpenAIError

from eval_judge.models import JudgeConfig, OutputSchema
from eval_judge.providers.base import BatchState, Provider, RetrievedBatch
from eval_judge.results import (
    ERRORS_SUFFIX,
    ParsedRecord,
    Status,
    make_record,
    record_from_output,
)

logger = logging.getLogger("eval_judge")

ENDPOINT = "/v1/responses"
TERMINAL_STATUSES = frozenset({"completed", "failed", "expired", "cancelled"})


def _error_text(error) -> str:
    if isinstance(error, dict):
        code = error.get("code") or error.get("type")
        message = error.get("message")
        return f"{code}: {message}" if code else str(message)
    return str(error)


def _usage(body: dict) -> dict[str, int] | None:
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return None
    out = {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "reasoning_tokens": (usage.get("output_tokens_details") or {}).get("reasoning_tokens"),
    }
    return {k: v for k, v in out.items() if isinstance(v, int)} or None


class OpenAIProvider(Provider):
    """OpenAI Batch API against the Responses endpoint."""

    name = "openai"
    max_requests = 50_000
    max_file_bytes = 200 * 1024 * 1024

    def __init__(self, client: OpenAI | None = None, *, api_key: str | None = None):
        if client is not None and api_key is not None:
            raise ValueError("pass one of client or api_key")
        self._client = client
        self._api_key = api_key

    @property
    def client(self) -> OpenAI:
        """Created on first use; falls back to the OPENAI_API_KEY environment variable."""
        if self._client is None:
            try:
                self._client = OpenAI(api_key=self._api_key)
            except OpenAIError as e:
                raise OpenAIError(
                    "No OpenAI API key found. Set the OPENAI_API_KEY environment variable "
                    "or pass OpenAIProvider(api_key=...) since .env not auto-loaded"
                ) from e
        return self._client

    # requests

    def build_body(
        self,
        config: JudgeConfig,
        system_prompt: str,
        schema: OutputSchema,
        user_content: str,
    ) -> dict:
        return {
            "model": config.model,
            "store": False,
            "reasoning": {"effort": config.effort},
            "instructions": system_prompt,
            "input": [{"role": "user", "content": user_content}],
            "max_output_tokens": config.max_output_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "strict": True,
                    "name": "alignment_eval",
                    "schema": schema,
                }
            },
        }

    def build_request_line(self, custom_id: str, body: dict) -> dict:
        return {"custom_id": custom_id, "method": "POST", "url": ENDPOINT, "body": body}

    def set_max_output_tokens(self, line: dict, n: int) -> dict:
        return {**line, "body": {**line["body"], "max_output_tokens": n}}

    # batch

    def submit(self, jsonl_path: str | Path) -> str:
        """Uploads and submits a JSONL file as a batch to the OpenAI Responses API; returns the batch id."""
        with open(jsonl_path, "rb") as f:
            uploaded = self.client.files.create(file=f, purpose="batch")
        batch = self.client.batches.create(
            input_file_id=uploaded.id, endpoint=ENDPOINT, completion_window="24h"
        )
        return batch.id

    def check(self, batch_id: str) -> BatchState:
        batch = self.client.batches.retrieve(batch_id)
        counts = batch.request_counts
        return BatchState(
            batch_id=batch_id,
            status=batch.status,
            terminal=batch.status in TERMINAL_STATUSES,
            completed=counts.completed if counts else 0,
            failed=counts.failed if counts else 0,
            total=counts.total if counts else 0,
            output_file_id=batch.output_file_id,
            error_file_id=batch.error_file_id,
        )

    def cancel(self, batch_id: str) -> BatchState:
        self.client.batches.cancel(batch_id)
        return self.check(batch_id)

    def download(self, batch_id: str, output_path: str | Path) -> RetrievedBatch:
        """Downloads output and error files; works for partial (expired/cancelled) batches."""
        batch = self.client.batches.retrieve(batch_id)
        if batch.status not in TERMINAL_STATUSES:
            raise RuntimeError(f"batch {batch_id} is not finished (status={batch.status})")
        if batch.status == "failed":
            errors = getattr(getattr(batch, "errors", None), "data", None) or []
            details = "; ".join(
                f"line {e.line}: {e.code}: {e.message}" if getattr(e, "line", None) else f"{e.code}: {e.message}"
                for e in errors
            )
            raise RuntimeError(f"batch {batch_id} failed: {details or 'no details'}")

        output_path = Path(output_path)
        saved_output = saved_errors = None
        if batch.output_file_id:
            output_path.write_text(
                self.client.files.content(batch.output_file_id).text, encoding="utf-8"
            )
            saved_output = output_path
        if batch.error_file_id:
            error_path = Path(str(output_path) + ERRORS_SUFFIX)
            error_path.write_text(
                self.client.files.content(batch.error_file_id).text, encoding="utf-8"
            )
            saved_errors = error_path
        if saved_output is None and saved_errors is None:
            logger.warning("batch %s (%s) has no output or error file", batch_id, batch.status)
        return RetrievedBatch(
            batch_id=batch_id,
            status=batch.status,
            output_path=saved_output,
            error_path=saved_errors,
        )

    # output

    def parse_record(self, record: dict) -> ParsedRecord:
        custom_id = record.get("custom_id")
        if record.get("error"):
            return make_record(custom_id, Status.FAILED, error=_error_text(record["error"]))

        response = record.get("response")
        if not isinstance(response, dict):
            return make_record(custom_id, Status.FAILED, error="record has no response")

        body = response.get("body")
        body = body if isinstance(body, dict) else {}
        status_code = response.get("status_code")
        if status_code != 200:
            detail = _error_text(body["error"]) if body.get("error") else "no error details"
            return make_record(custom_id, Status.FAILED, error=f"HTTP {status_code}: {detail}")

        record = self._parse_body(custom_id, body)
        usage = _usage(body)
        return replace(record, usage=usage) if usage else record

    def _parse_body(self, custom_id: str | None, body: dict) -> ParsedRecord:
        body_status = body.get("status")
        if body_status == "incomplete":
            reason = (body.get("incomplete_details") or {}).get("reason")
            if reason == "max_output_tokens":
                return make_record(custom_id, Status.TRUNCATED, error="stopped at max_output_tokens")
            return make_record(custom_id, Status.FAILED, error=f"incomplete: {reason}")
        if body_status != "completed":
            detail = f": {_error_text(body['error'])}" if body.get("error") else ""
            return make_record(custom_id, Status.FAILED, error=f"response status {body_status!r}{detail}")

        output_items = body.get("output") or []
        message = next(
            (i for i in output_items if isinstance(i, dict) and i.get("type") == "message"),
            None,
        )
        if message is None:
            return make_record(custom_id, Status.INVALID, error="no message output item found")

        content = [c for c in message.get("content") or [] if isinstance(c, dict)]
        refusal = next((c for c in content if c.get("type") == "refusal"), None)
        if refusal is not None:
            return make_record(custom_id, Status.REFUSED, error=refusal.get("refusal"))
        text = next((c.get("text") for c in content if c.get("type") == "output_text"), None)
        if text is None:
            return make_record(custom_id, Status.INVALID, error="message has no output_text")
        return record_from_output(custom_id, text)
