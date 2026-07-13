# eval-judge

LLM-as-a-judge scoring for AI systems.


## Installation

First, install:

```bash
pip install eval-judge
```

or if using `uv`:
```bash
uv add eval-judge
```

Then, create a `.env` and add your `OPENAI_API_KEY` to it.

## Example

First, set up the essentials:

```python
# First, create a list of EvalDimensions you want to score on.
my_eval_dimensions = [
    EvalDimension(
        key="humor",
        title="Humor Level",
        definition=(
            "Evaluates how funny the agent is"
        ),
        penalize=[
            "Boring or corny jokes"
        ],
        reward=[
            "Unpredictability"
        ],
        anchors=[
            ScoreAnchor(
                score=1,
                description="Agent is unfunny",
            ),
            ScoreAnchor(
                score=2,
                description="Agent is not very funny",
            ),
            ScoreAnchor(
                score=3,
                description="Agent is not funny nor unfunny",
            ),
            ScoreAnchor(
                score=4,
                description="Agent is moderately funny",
            ),
            ScoreAnchor(
                score=5,
                description="Agent is very funny",
            ),
        ],
    ),
    # more dimensions...
]


# Then, create your config
config = JudgeConfig(
    model="gpt-5.6-luna",
    effort="medium",
    max_output_tokens=3000,
    dimensions=my_eval_dimensions
)


# Then, create a list of ContextMessage, injecting relevant context yourself.
my_messages = [
    ContextMessage(
        message_id="my_id_123",
        context="At the barbecue to a guest in line",
        content="Why did the LLM go to the gym? For weight training!"
    ),
    # more messages...
]
```

For regular users (recommended):

```python
# Create Judge based off config
judge = Judge(config=config)


# Get results; may take a while due to async OpenAI batching
results = judge.evaluate_batch(messages=my_messages)
```

For advanced users:

```python

# Then build your requests batch
jsonl_path = build_requests_batch(
    messages=my_messages, 
    output_path="path/to/jsonl", # where you want to write the JSONL
    config=config
)


# Then, submit your batch
batch_id = submit_batch(jsonl_path=jsonl_path)


# Blocks until OpenAI batch results available and saves results to disk
batch_output_path = "path/to/batch/output" # where you want to write the raw batch output


# Also returns raw JSONL output directly
raw_jsonl = retrieve_batch(batch_id=batch_id, output_path=batch_output_path)


# Extract relevant information from full OpenAI batch results
results = load_and_parse_results(input_path=batch_output_path)
```

Both approaches produce the same result structure. Enjoy!

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.