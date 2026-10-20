"""Synthetic datasets, sharded TFRecord IO and the tf.data input pipeline."""

from pathlib import Path

import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


def make_arrays(demo: str, n: int, seed: int, window: int = 24, horizon: int = 1):
    rng = np.random.default_rng(seed)
    if demo == "linear":
        x = rng.normal(size=(n, 4)).astype("float32")
        w = np.array([2.0, -1.0, 0.5, 3.0], "float32")
        y = (x @ w + 1.5 + rng.normal(scale=0.1, size=n)).astype("float32")[:, None]
    elif demo == "logistic":
        x = rng.normal(size=(n, 6)).astype("float32")
        w = rng.normal(size=6).astype("float32")
        p = 1 / (1 + np.exp(-(x @ w)))
        y = (rng.random(n) < p).astype("float32")[:, None]
    elif demo == "forecast":
        t = np.arange(n + window + horizon, dtype="float32")
        s = (
            np.sin(2 * np.pi * t / 24)
            + 0.3 * np.sin(2 * np.pi * t / 168)
            + 0.001 * t
            + rng.normal(scale=0.05, size=t.shape)
        ).astype("float32")
        idx = np.arange(n)[:, None] + np.arange(window)[None, :]
        x = s[idx][..., None]  # (n, window, 1)
        y = s[np.arange(n) + window + horizon - 1][:, None]
    else:
        raise ValueError(f"unknown demo {demo!r}")
    return x, y


def _example(x: np.ndarray, y: np.ndarray) -> tf.train.Example:
    feat = {
        "x": tf.train.Feature(float_list=tf.train.FloatList(value=x.ravel())),
        "y": tf.train.Feature(float_list=tf.train.FloatList(value=y.ravel())),
    }
    return tf.train.Example(features=tf.train.Features(feature=feat))


def write_shards(
    x, y, out_dir: str, name: str, num_shards: int, compression: str = ""
) -> list[str]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    opts = tf.io.TFRecordOptions(compression_type=compression)
    paths = [
        str(Path(out_dir) / f"{name}-{i:05d}-of-{num_shards:05d}.tfrecord")
        for i in range(num_shards)
    ]
    writers = [tf.io.TFRecordWriter(p, opts) for p in paths]
    try:
        for i in range(len(x)):  # round-robin keeps shards balanced
            writers[i % num_shards].write(_example(x[i], y[i]).SerializeToString())
    finally:
        for w in writers:
            w.close()
    return paths


def build_dataset(
    files: list[str],
    x_shape,
    y_dim: int,
    global_batch: int,
    *,
    compression: str,
    cycle_length: int,
    shuffle_buffer: int,
    drop_remainder: bool,
    prefetch,
    private_threadpool: int = 0,
    training: bool = True,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(sorted(files))
    if training:
        ds = ds.shuffle(len(files))
    ds = ds.interleave(
        lambda f: tf.data.TFRecordDataset(f, compression_type=compression or None),
        cycle_length=max(1, min(cycle_length, len(files))),
        num_parallel_calls=AUTOTUNE,
        deterministic=not training,
    )
    spec = {
        "x": tf.io.FixedLenFeature(int(np.prod(x_shape)), tf.float32),
        "y": tf.io.FixedLenFeature(y_dim, tf.float32),
    }

    def parse(batch):  # parse after batching: fewer, larger ops
        p = tf.io.parse_example(batch, spec)
        return tf.reshape(p["x"], [-1, *x_shape]), p["y"]

    if training:
        ds = ds.shuffle(shuffle_buffer).repeat()
    ds = ds.batch(
        global_batch,
        drop_remainder=drop_remainder or training,
        num_parallel_calls=AUTOTUNE,
    ).map(parse, num_parallel_calls=AUTOTUNE)
    opts = tf.data.Options()
    if private_threadpool:
        opts.threading.private_threadpool_size = private_threadpool
    ds = ds.with_options(opts)
    return ds.prefetch(AUTOTUNE if prefetch == "auto" else int(prefetch))
