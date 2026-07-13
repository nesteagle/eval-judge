import json

from eval_judge.models import ContextMessage, JudgeConfig
from eval_judge.rubric import _build_output_schema, _build_system_prompt


def _build_request_body(user_content: str, config: JudgeConfig) -> dict:
    """Builds OpenAI API request body contents."""
    return {
        "model": config.model,
        "store": False,
        "reasoning": {"effort": config.effort},
        "instructions": _build_system_prompt(config.dimensions),
        "input": [{"role": "user", "content": user_content}],
        "max_output_tokens": config.max_output_tokens,
        "text": {
            "format": {
                "type": "json_schema",
                "strict": True,
                "name": "alignment_eval",
                "schema": _build_output_schema(config.dimensions),
            }
        },
    }


def build_requests_batch(
    messages: list[ContextMessage],
    output_path: str,
    config: JudgeConfig,
    passes: int = 1,
) -> str:
    """Converts assembled prompt records into a batch-ready JSONL file."""
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for message in messages:
            for run in range(passes):
                suffix = "reasoning" if message.reasoning else "content"
                run_suffix = f"-{run}" if passes > 1 else ""

                body = _build_request_body(
                    user_content=message.formatted(), config=config
                )

                payload = {
                    "custom_id": f"{message.message_id}{run_suffix}-{suffix}",
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": body,
                }
                f.write(json.dumps(payload) + "\n")
                count += 1

    print(
        f"Wrote {count} requests ({len(messages)} messages, {passes} passes) -> {output_path}"
    )
    return output_path
