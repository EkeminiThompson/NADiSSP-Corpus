"""Configuration loading.

configs/nadip.yaml holds every document-specified value. A second YAML (e.g. configs/smoke.yaml)
is deep-merged on top, and dotted CLI overrides ("training.stage3_joint.epochs=5") on top of that.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "nadip.yaml"


def _deep_merge(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _coerce(value: str) -> Any:
    return yaml.safe_load(value)


def _fix_floats(node: Any) -> Any:
    """PyYAML reads `8.4e6` (no exponent sign) as a string; turn such strings back into numbers."""
    if isinstance(node, dict):
        return {k: _fix_floats(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_fix_floats(v) for v in node]
    if isinstance(node, str):
        try:
            return float(node)
        except ValueError:
            return node
    return node


def apply_overrides(cfg: Dict[str, Any], overrides: Optional[Iterable[str]]) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like a.b.c=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(raw)
    return cfg


def load_config(path: Optional[str | Path] = None, overrides: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    with open(DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    if path and Path(path).resolve() != DEFAULT_CONFIG.resolve():
        with open(path) as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
    cfg = _fix_floats(apply_overrides(cfg, overrides))
    cfg["_root"] = str(ROOT)
    return cfg


def resolve_path(cfg: Dict[str, Any], key: str) -> Path:
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else Path(cfg["_root"]) / p


def add_common_args(parser) -> None:
    parser.add_argument("--config", default=None, help="extra YAML merged over configs/nadip.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        help="dotted override, e.g. --set training.stage3_joint.epochs=5")
