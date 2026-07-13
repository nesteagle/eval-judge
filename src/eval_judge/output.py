import json
from pathlib import Path


def _parse_results(records: list[dict]) -> list[dict]:
    """Parses a list of raw batch record dicts."""
    return [_parse_record(r) for r in records]


def load_and_parse_results(input_path: str | Path) -> list[dict]:
    """Loads JSONL OpenAI raw batch output and parses."""
    raw = Path(input_path).read_text(encoding="utf-8")
    records = [json.loads(line) for line in raw.strip().split("\n") if line.strip()]

    return _parse_results(records)


def _parse_record(record: dict) -> dict:
    custom_id = record.get("custom_id")
    try:
        output_items = record["response"]["body"]["output"]
        message_item = next(
            (item for item in output_items if item.get("type") == "message"),
            None,
        )
        if message_item is None:
            raise ValueError("no message output item found")
        result = json.loads(message_item["content"][0]["text"])
    except (KeyError, IndexError, ValueError) as e:
        raise ValueError(
            f"failed to parse batch record custom_id={custom_id!r}: {e}"
        ) from e
    result["custom_id"] = custom_id
    return result
