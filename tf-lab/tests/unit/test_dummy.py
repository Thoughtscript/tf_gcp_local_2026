import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.agent import list_settings, run_experiment
from app.ml import data as D

ROOT = Path(__file__).resolve().parents[2]


def test_override_validation():
    assert "error" in run_experiment("linear", ["nope.key=1"])
    assert "error" in run_experiment("linear", ["output_dir=/tmp/x"])
    assert "error" in run_experiment("bogus")
    assert "batch.per_replica_batch_size" in list_settings()


def test_shard_roundtrip(tmp_path):
    x, y = D.make_arrays("linear", 100, 1)
    files = D.write_shards(x, y, str(tmp_path), "t", 4, "GZIP")
    assert len(files) == 4 and files[0].endswith("00000-of-00004.tfrecord")
    ds = D.build_dataset(
        files,
        x.shape[1:],
        1,
        10,
        compression="GZIP",
        cycle_length=2,
        shuffle_buffer=10,
        drop_remainder=False,
        prefetch=2,
        training=False,
    )
    assert sum(len(b[0]) for b in ds) == 100


@pytest.mark.parametrize("demo", ["linear", "logistic", "forecast"])
def test_demos_with_virtual_replicas(demo, tmp_path):
    # subprocess: logical devices must be configured before TF initializes
    sets = [
        "epochs=2",
        "num_samples=2000",
        "distribution.virtual_devices=2",
        "data.num_shards=4",
        "memory.private_threadpool=2",
        "memory.prefetch=2",
        f"output_dir={tmp_path / 'runs'}",
        f"data.dir={tmp_path / 'data'}",
    ]
    args = [sys.executable, "-m", "app.ml.train", "--demo", demo]
    for s in sets:
        args += ["--set", s]
    out = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    r = json.loads((tmp_path / "runs" / demo / "metrics.json").read_text())
    assert r["replicas"] == 2 and r["global_batch"] == 128 and r["shards"] == 4
