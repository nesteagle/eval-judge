import time
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    load_dotenv()
    return OpenAI()


def submit_batch(jsonl_path: str) -> str:
    """
    Uploads and submits a JSONL file as batch to OpenAI Repsonses API; returns the batch_id.
    """
    client = _get_client()
    with open(jsonl_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=uploaded.id, endpoint="/v1/responses", completion_window="24h"
    )
    print(f"[batch] Submitted. id={batch.id}")
    return batch.id


def _poll_batch(batch_id: str, interval: int = 60) -> str:
    """
    Block until the batch completes (or fails).
    """
    client = _get_client()
    while True:
        batch = client.batches.retrieve(batch_id)
        print(
            f"[batch] status={batch.status} "
            f"completed={batch.request_counts.completed}/"
            f"{batch.request_counts.total}"
        )
        if batch.status == "completed":
            return batch.output_file_id
        if batch.status in ("failed", "expired", "cancelled"):
            raise RuntimeError(f"Batch ended with status: {batch.status}")
        time.sleep(interval)


def retrieve_batch(batch_id: str, output_path: str) -> str:
    """
    Retrieves completed batch output, polling if needed, and saves JSONL raw results to output_path.
    """
    client = _get_client()
    output_file_id = _poll_batch(batch_id)
    raw = client.files.content(output_file_id).text

    Path(output_path).write_text(raw, encoding="utf-8")
    print(f"Saved batch output -> {output_path}")

    return raw
