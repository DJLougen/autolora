"""Seam: harvest the agent's OWN traces from Hermes + score them for curation.

`hermes sessions export` dumps every past session (one JSON per line) from the
SQLite session store. We score each session's quality from real signals already
in the trace (tool success, clean termination, productivity), and label a
calibration subset so the Youden cutoff has something to fit. No LLM judge, no
synthetic data — these are the agent's real runs.
"""
from __future__ import annotations

import json
import os
import subprocess

from self_improve_loop import Trace
from seams import config

ERR_END = {"error", "aborted", "cancelled", "max_iterations", "interrupted"}
ERR_FINISH = {"error", "length", "content_filter"}


def export(out_path: str | None = None) -> str:
    """Export the agent's sessions to a JSONL folder via the Hermes CLI."""
    cfg = config.load()
    out_path = out_path or config.rel(cfg["data"]["traces_raw"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    subprocess.run(["hermes", "sessions", "export", out_path],
                   check=True, capture_output=True, text=True)
    return out_path


def _read(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score(s: dict) -> float:
    """Heuristic trace quality in [0,1] from real session signals."""
    msgs = s.get("messages", [])
    issued = sum(len(m.get("tool_calls") or []) for m in msgs
                 if m.get("role") == "assistant")
    results = sum(1 for m in msgs if m.get("role") == "tool")
    success = (results / issued) if issued else 0.0
    clean = 0.0 if (s.get("end_reason") in ERR_END) else 1.0
    errs = sum(1 for m in msgs if m.get("finish_reason") in ERR_FINISH)
    err_pen = min(1.0, errs / max(1, len(msgs)))
    tc = s.get("tool_call_count", 0) or 0
    productive = min(1.0, tc / 10.0)
    out_ok = 1.0 if (s.get("output_tokens", 0) or 0) > 0 else 0.0
    q = (0.40 * min(1.0, success) + 0.25 * clean +
         0.20 * productive + 0.15 * out_ok) - 0.15 * err_pen
    return round(max(0.0, min(1.0, q)), 4)


def label(s: dict) -> int | None:
    """Binary good/bad for the calibration subset; None = unclear (excluded)."""
    msgs = s.get("messages", [])
    issued = sum(len(m.get("tool_calls") or []) for m in msgs
                 if m.get("role") == "assistant")
    results = sum(1 for m in msgs if m.get("role") == "tool")
    success = (results / issued) if issued else 0.0
    clean = s.get("end_reason") not in ERR_END
    tc = s.get("tool_call_count", 0) or 0
    if clean and tc >= 1 and success >= 0.9:
        return 1
    if (not clean) or tc == 0 or success < 0.5:
        return 0
    return None


def _sessions() -> list[dict]:
    cfg = config.load()
    path = config.rel(cfg["data"]["traces_raw"])
    if not os.path.exists(path):
        export(path)
    return _read(path)


def harvest_traces() -> list[Trace]:
    return [Trace(id=s["id"], quality=score(s)) for s in _sessions()]


def label_calibration() -> list[Trace]:
    out = []
    for s in _sessions():
        lab = label(s)
        if lab is not None:
            out.append(Trace(id=s["id"], quality=score(s), label=lab))
    return out


def load_records() -> dict[str, dict]:
    """id -> full session record (messages + meta), for the trainer."""
    return {s["id"]: s for s in _sessions()}


def summarize() -> dict:
    """Summary statistics over the harvested + curated traces (for the user)."""
    from self_improve_loop import youden_threshold, curate
    sess = _sessions()
    pool = [Trace(s["id"], score(s)) for s in sess]
    cal = label_calibration()
    cut = youden_threshold(cal)
    kept = curate(pool, cal)
    qs = sorted(t.quality for t in pool)
    pct = lambda p: qs[min(len(qs) - 1, int(p * len(qs)))] if qs else 0.0
    return {
        "sessions": len(sess),
        "messages": sum(s.get("message_count", 0) or 0 for s in sess),
        "tool_calls": sum(s.get("tool_call_count", 0) or 0 for s in sess),
        "calibration": {"labeled": len(cal),
                        "good": sum(1 for t in cal if t.label == 1),
                        "bad": sum(1 for t in cal if t.label == 0)},
        "quality": {"min": qs[0] if qs else 0.0, "p50": pct(0.5),
                    "p90": pct(0.9), "max": qs[-1] if qs else 0.0},
        "cutoff": cut, "kept": len(kept), "of": len(pool),
        "kept_frac": round(len(kept) / max(1, len(pool)), 3),
    }
