import json
from pathlib import Path


def parse_results(input_path: str) -> list[dict]:
    """Parses JSON results for OpenAI raw batch output."""
    raw = Path(input_path).read_text(encoding="utf-8")
    records = json.loads(raw)
    results = [_parse_record(r) for r in records]
    return results


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


def save_json(data: list, path: str) -> None:
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Saved {len(data)} objects -> {path}")

