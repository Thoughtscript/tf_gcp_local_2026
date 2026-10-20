"""python -m app.ml.train --demo linear [--config f.yaml] [--set batch.per_replica_batch_size=128 ...]"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# Keras 3 cannot drive MultiWorkerMirroredStrategy; legacy tf_keras can.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf

from . import config as C
from . import data as D
from . import models
from . import strategy as S


def run(cfg: dict) -> dict:
    tf.keras.utils.set_random_seed(cfg["seed"])
    demo, d, f, mem = cfg["demo"], cfg["data"], cfg["forecast"], cfg["memory"]
    gpus = S.configure_devices(cfg)  # must precede any op that initializes devices
    strat = S.make_strategy(cfg, gpus)
    replicas = strat.num_replicas_in_sync
    global_batch = cfg["batch"]["per_replica_batch_size"] * replicas

    x, y = D.make_arrays(
        demo, cfg["num_samples"], cfg["seed"], f["window"], f["horizon"]
    )
    x_shape, y_dim = tuple(x.shape[1:]), y.shape[1]
    n_train = int(len(x) * 0.9)
    shard_dir = str(Path(d["dir"]) / demo)
    files = {
        "train": D.write_shards(
            x[:n_train],
            y[:n_train],
            shard_dir,
            f"{demo}-train",
            d["num_shards"],
            d["compression"],
        ),
        "val": D.write_shards(
            x[n_train:],
            y[n_train:],
            shard_dir,
            f"{demo}-val",
            max(1, d["num_shards"] // 4),
            d["compression"],
        ),
    }
    common = {
        "compression": d["compression"],
        "cycle_length": d["cycle_length"],
        "shuffle_buffer": d["shuffle_buffer"],
        "drop_remainder": cfg["batch"]["drop_remainder"],
        "prefetch": mem["prefetch"],
        "private_threadpool": mem["private_threadpool"],
    }
    train_ds = D.build_dataset(
        files["train"], x_shape, y_dim, global_batch, training=True, **common
    )
    val_ds = D.build_dataset(
        files["val"], x_shape, y_dim, global_batch, training=False, **common
    )
    # Under a strategy Keras already stages per-replica batches on their devices.
    if mem["pin"] and gpus and replicas == 1:
        train_ds = train_ds.apply(
            tf.data.experimental.prefetch_to_device("/GPU:0", buffer_size=2)
        )

    with strat.scope():
        model = models.build(demo, x_shape, cfg["learning_rate"])
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["epochs"],
        steps_per_epoch=max(1, n_train // global_batch),
        verbose=2,
    )
    result = {
        "demo": demo,
        "replicas": replicas,
        "global_batch": global_batch,
        "shards": len(files["train"]),
        "gpus": len(gpus),
        **{k: float(v[-1]) for k, v in hist.history.items()},
    }
    out = Path(cfg["output_dir"]) / demo
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "model.keras")
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(C.DEFAULT_CONFIG))
    ap.add_argument("--demo", choices=["linear", "logistic", "forecast"])
    ap.add_argument(
        "--set", action="append", default=[], help="e.g. data.num_shards=16"
    )
    a = ap.parse_args()
    over = C.parse_overrides(a.set)
    if a.demo:
        over["demo"] = a.demo
    print(json.dumps(run(C.load(a.config, over)), indent=2))


if __name__ == "__main__":
    main()
