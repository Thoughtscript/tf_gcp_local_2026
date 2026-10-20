"""Device setup: memory options, (virtual) GPUs and tf.distribute strategy selection."""

import tensorflow as tf


def configure_devices(cfg: dict) -> list:
    mem, dist = cfg["memory"], cfg["distribution"]
    gpus = tf.config.list_physical_devices("GPU")
    if dist.get("gpus") is not None:
        gpus = [gpus[i] for i in dist["gpus"]]
        tf.config.set_visible_devices(gpus, "GPU")
    for g in gpus:
        if mem.get("gpu_memory_limit_mb"):
            tf.config.set_logical_device_configuration(
                g,
                [
                    tf.config.LogicalDeviceConfiguration(
                        memory_limit=mem["gpu_memory_limit_mb"]
                    )
                ],
            )
        elif mem.get("gpu_memory_growth"):
            tf.config.experimental.set_memory_growth(g, True)
    n = dist.get("virtual_devices", 0)
    if n and not gpus:
        cpu = tf.config.list_physical_devices("CPU")[0]
        tf.config.set_logical_device_configuration(
            cpu, [tf.config.LogicalDeviceConfiguration() for _ in range(n)]
        )
    return gpus


def make_strategy(cfg: dict, gpus: list) -> tf.distribute.Strategy:
    dist = cfg["distribution"]
    kind = dist["strategy"]
    virtual = dist.get("virtual_devices", 0)
    if kind == "auto":
        kind = "mirrored" if len(gpus) > 1 or virtual > 1 else "none"
    if kind == "none":
        return tf.distribute.get_strategy()
    if kind == "multiworker":  # one process per worker, cluster described by TF_CONFIG
        return tf.distribute.MultiWorkerMirroredStrategy()
    if kind == "mirrored":
        if gpus:
            ops = {
                "nccl": tf.distribute.NcclAllReduce,
                "hierarchical_copy": tf.distribute.HierarchicalCopyAllReduce,
                "reduce_to_one": tf.distribute.ReductionToOneDevice,
            }[dist["all_reduce"]]()
            return tf.distribute.MirroredStrategy(
                [f"GPU:{i}" for i in range(len(gpus))], cross_device_ops=ops
            )
        if virtual > 1:
            return tf.distribute.MirroredStrategy([f"CPU:{i}" for i in range(virtual)])
        return tf.distribute.MirroredStrategy()
    raise ValueError(f"unknown strategy {kind!r}")
