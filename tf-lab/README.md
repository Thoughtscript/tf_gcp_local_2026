# tf-lab

ADK agent that runs TensorFlow experiments (linear regression, logistic regression, LSTM forecasting) and lets you tune input-pipeline, batching, distribution and memory settings.

## Docker (CPU; Docker on macOS has no GPU access)

```bash
docker compose --profile train build
docker compose run --rm train --demo forecast --set data.num_shards=16     # 2 virtual replicas by default
docker compose --profile multiworker up --abort-on-container-exit         # 2 workers via TF_CONFIG
docker compose --profile agent up                                          # ADK API on :8080 (startup needs no creds; chat needs GEMINI_API_KEY in .env or gcloud ADC)
```
Results land in `./runs`. Training uses legacy `tf_keras` (`TF_USE_LEGACY_KERAS=1`) because Keras 3 doesn't support multi-worker strategies.

## TensorFlow experiments

```bash
.venv/bin/python -m app.ml.train --demo linear
.venv/bin/python -m app.ml.train --demo forecast --set distribution.virtual_devices=2 --set data.num_shards=16
agents-cli run "Train logistic with 16 shards, GZIP and per-replica batch 128, then compare to 4 shards"
```

Defaults live in [app/ml/train.yaml](app/ml/train.yaml); override any key with `--set a.b=value` (or the agent's `run_experiment` tool, which only accepts known keys).

| Area | Keys |
|---|---|
| Batch size | `batch.per_replica_batch_size` (global = per-replica x replicas), `batch.drop_remainder` |
| Distributed GPU | `distribution.strategy` = `auto`/`none`/`mirrored`/`multiworker` (multiworker reads `TF_CONFIG`), `distribution.all_reduce` = `nccl`/`hierarchical_copy`/`reduce_to_one`, `distribution.gpus`, `distribution.virtual_devices` (simulate N replicas on CPU) |
| Sharded TFRecords | `data.num_shards`, `data.compression` (`""`/`GZIP`/`ZLIB`), `data.cycle_length` (parallel shard reads), `data.shuffle_buffer` |
| Memory | `memory.gpu_memory_growth`, `memory.gpu_memory_limit_mb` (hard cap, wins over growth), `memory.prefetch`, `memory.private_threadpool`, `memory.pin` (single-GPU `prefetch_to_device`) |

Outputs go to `runs/<demo>/` (`model.keras`, `metrics.json`). GPU paths need GPUs; CI uses virtual CPU replicas.

Agent generated with `agents-cli` version `1.11.0`

## Project Structure

```
tf-lab/
├── app/         # Core agent code
│   ├── agent.py               # Main agent logic
│   ├── fast_api_app.py        # FastAPI Backend server
│   └── app_utils/             # App utilities and helpers
├── tests/                     # Unit, integration, and load tests
├── GEMINI.md                  # AI-assisted development guide
└── pyproject.toml             # Project dependencies
```

> 💡 **Tip:** Use [Antigravity CLI](https://antigravity.google/) for AI-assisted development - project context is pre-configured in `GEMINI.md`.

## Requirements

Before you begin, ensure you have:
- **uv**: Python package manager (used for all dependency management in this project) - [Install](https://docs.astral.sh/uv/getting-started/installation/) ([add packages](https://docs.astral.sh/uv/concepts/dependencies/) with `uv add <package>`)
- **agents-cli**: Agents CLI - Install with `uv tool install google-agents-cli`
- **Google Cloud SDK**: For GCP services - [Install](https://cloud.google.com/sdk/docs/install)


## Quick Start

Install `agents-cli` and its skills if not already installed:

```bash
uvx google-agents-cli setup
```

Install required packages:

```bash
agents-cli install
```

Test the agent with a local web server:

```bash
agents-cli playground
```

You can also use features from the [ADK](https://adk.dev/) CLI with `uv run adk`.

## Commands

| Command              | Description                                                                                 |
| -------------------- | ------------------------------------------------------------------------------------------- |
| `agents-cli install` | Install dependencies using uv                                                         |
| `agents-cli playground` | Launch local development environment                                                  |
| `agents-cli lint`    | Run code quality checks                                                               |
| `agents-cli eval`    | Evaluate agent behavior (generate, grade, analyze, and more — see `agents-cli eval --help`) |
| `uv run pytest tests/unit tests/integration` | Run unit and integration tests                                                        || [A2A Inspector](https://github.com/a2aproject/a2a-inspector) | Launch A2A Protocol Inspector                                                        |

## 🛠️ Project Management

| Command | What It Does |
|---------|--------------|
| `agents-cli scaffold enhance` | Add CI/CD pipelines and Terraform infrastructure |
| `agents-cli infra cicd` | One-command setup of entire CI/CD pipeline + infrastructure |
| `agents-cli scaffold upgrade` | Auto-upgrade to latest version while preserving customizations |

---

## Development

Edit your agent logic in `app/agent.py` and test with `agents-cli playground` - it auto-reloads on save.

## Deployment

```bash
gcloud config set project <your-project-id>
agents-cli deploy
```

To add CI/CD and Terraform, run `agents-cli scaffold enhance`.
To set up your production infrastructure, run `agents-cli infra cicd`.

## Observability

Built-in telemetry exports to Cloud Trace, BigQuery, and Cloud Logging.

## A2A Inspector

This agent supports the [A2A Protocol](https://a2a-protocol.org/). Use the [A2A Inspector](https://github.com/a2aproject/a2a-inspector) to test interoperability.
See the [A2A Inspector docs](https://github.com/a2aproject/a2a-inspector) for details.
