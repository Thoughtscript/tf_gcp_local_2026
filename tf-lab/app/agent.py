# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import subprocess
import sys
from pathlib import Path

from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini
from google.genai import types

from app.ml import config as ml_config

MODEL = "gemini-3.8-flash"
ROOT = Path(__file__).resolve().parents[1]
DEMOS = ("linear", "logistic", "forecast")
ALLOWED = ml_config.flat_keys(ml_config.load())
LOCKED = {"output_dir", "data.dir"}  # keep runs inside the project


def list_settings() -> dict:
    """Returns every tunable training setting with its default value, as flat dotted keys.

    Covers batch size, distribution strategy / all-reduce, TFRecord sharding and
    compression, and memory options (GPU memory growth or limit, prefetch, threadpool).
    """
    cfg = ml_config.load()
    flat = {}
    for key in sorted(ALLOWED - LOCKED):
        node = cfg
        for part in key.split("."):
            node = node[part]
        flat[key] = node
    return flat


def run_experiment(demo: str, overrides: list[str] | None = None) -> dict:
    """Trains a TensorFlow demo model locally and returns its final metrics.

    Args:
        demo: One of "linear" (linear regression), "logistic" (binary classifier)
            or "forecast" (LSTM time-series forecaster).
        overrides: Optional settings as "dotted.key=value" strings, e.g.
            ["batch.per_replica_batch_size=128", "data.num_shards=16",
            "data.compression=GZIP", "distribution.virtual_devices=2",
            "memory.private_threadpool=4"]. Use list_settings for valid keys.

    Returns:
        The metrics dict (replicas, global_batch, shards, losses...) or an error.
    """
    if demo not in DEMOS:
        return {"error": f"demo must be one of {DEMOS}"}
    try:
        ml_config.parse_overrides(overrides or [], ALLOWED - LOCKED)
    except ValueError as e:
        return {"error": str(e)}
    args = [sys.executable, "-m", "app.ml.train", "--demo", demo]
    for o in overrides or []:
        args += ["--set", o]
    proc = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        return {"error": proc.stderr[-1500:]}
    return json.loads((ROOT / "runs" / demo / "metrics.json").read_text())


def run_tests(test_filter: str | None = None) -> dict:
    """Runs the project's unit tests (pytest, tests/unit) and returns a summary.

    Args:
        test_filter: Optional pytest -k expression, e.g. "forecast" or "shard".

    Returns:
        Pass/fail status plus the tail of the pytest output.
    """
    if not (ROOT / "tests" / "unit").is_dir():
        return {"error": "tests/unit is not present in this environment"}
    args = [
        sys.executable,
        "-m",
        "pytest",
        "tests/unit",
        "-q",
        "--no-header",
        "-p",
        "no:cacheprovider",
    ]
    if test_filter:
        args += ["-k", test_filter]
    proc = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=1800)
    return {
        "passed": proc.returncode == 0,
        "output": (proc.stdout + proc.stderr)[-2000:],
    }


root_agent = Agent(
    # Keep in sync with create_params.root_agent_name in agents-cli-manifest.yaml:
    # telemetry reports this name as gen_ai.agent.name, and agents-cli reports the
    # manifest's value. Renaming the agent only here makes the two disagree, and
    # anything selecting traces by name stops finding this agent's.
    name="tf_lab",
    model=Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=(
        "You run TensorFlow experiments (linear regression, logistic regression, "
        "forecasting) for the user. Use list_settings to see tunable options and "
        "run_experiment to train, and run_tests to run the unit tests. When comparing configurations, change one setting at "
        "a time and summarize global batch size, replicas, shard count and final metrics. "
        "Mirrored/multi-worker GPU strategies need GPUs (or distribution.virtual_devices=2 "
        "to simulate replicas on CPU)."
    ),
    tools=[list_settings, run_experiment, run_tests],
)

app = App(
    root_agent=root_agent,
    name="app",
)
