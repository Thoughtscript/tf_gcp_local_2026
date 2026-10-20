import copy
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).with_name("train.yaml")


def deep_update(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load(path: str | Path = DEFAULT_CONFIG, overrides: dict | None = None) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    return deep_update(cfg, overrides or {})


def flat_keys(d: dict, prefix: str = "") -> set[str]:
    keys = set()
    for k, v in d.items():
        full = f"{prefix}{k}"
        if isinstance(v, dict):
            keys |= flat_keys(v, full + ".")
        else:
            keys.add(full)
    return keys


def parse_overrides(items: list[str], allowed: set[str] | None = None) -> dict:
    """`a.b.c=value` pairs with YAML-parsed values; unknown keys raise if `allowed` is given."""
    out: dict = {}
    for item in items or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override {item!r} must look like key=value")
        if allowed is not None and key not in allowed:
            raise ValueError(f"unknown setting {key!r}")
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return out
