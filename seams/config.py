"""Repo-root-relative config loader."""
from __future__ import annotations

import os
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load(path: str | None = None) -> dict:
    path = path or os.path.join(ROOT, "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)

def rel(p: str) -> str:
    """Resolve a config path relative to the repo root."""
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(ROOT, p))
