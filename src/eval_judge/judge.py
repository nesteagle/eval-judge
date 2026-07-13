from eval_judge.batch import build_requests_batch
from eval_judge.client import retrieve_batch, submit_batch
from eval_judge.models import ContextMessage, JudgeConfig
from eval_judge.output import load_and_parse_results


class Judge:
    def __init__(self, config: JudgeConfig):
        self.config = config

    def evaluate_batch(
        self,
        messages: list[ContextMessage],
        output_path: str | None = None,
    ) -> list[dict]:
        jsonl_path = build_requests_batch(
            messages=messages, config=self.config, output_path=None
        )
        batch_id = submit_batch(jsonl_path=jsonl_path)
        raw_output_path = retrieve_batch(batch_id=batch_id, output_path=output_path)
        return load_and_parse_results(input_path=raw_output_path)
