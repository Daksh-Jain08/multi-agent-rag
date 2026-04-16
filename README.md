# Deductive Adjudication in Multi-Agent RAG

Research-first, minimal Python project for comparing multi-agent RAG architectures on conflict and refusal datasets.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
cp .env.example .env
```

Set `GOOGLE_API_KEY` in `.env`.

## Run Single Query

```bash
python main.py --architecture sequential --query "What do the sources conclude?" --top_k 5
```

## Run Batch

```bash
python main.py --architecture parallel_summarizer --dataset_path data/conflicts_normalized.jsonl --top_k 5
```

## Run Evaluation

```bash
python run_eval.py --datasets conflicts_normalized.jsonl refusals_normalized.jsonl --output outputs/eval/summary.json --report outputs/eval/summary.md
```

## API Rate Limiting And Logs

- Configure in `.env`:
	- `LLM_REQUESTS_PER_MINUTE`
	- `LLM_MAX_CONCURRENT_CALLS`
	- `LLM_MIN_INTERVAL_SECONDS`
	- `LLM_LOG_PATH`
- LLM API call logs are written to `outputs/logs/llm_calls.log` by default.

## Architectures

- `single_agent`
- `sequential`
- `debate`
- `parallel`
- `parallel_summarizer`
