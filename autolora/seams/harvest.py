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

# basic task taxonomy — keyword vote over the user turns
TASK_CATEGORIES = {
    "coding":   ["code", "bug", "function", "refactor", "implement", "git",
                 "repo", "compile", "traceback", "python", "rust", "class ", "def "],
    "ml":       ["train", "fine-tune", "finetune", "lora", "model", "dataset",
                 "eval", "benchmark", "gpu", "huggingface", "quantiz", "checkpoint"],
    "research": ["search", "find", "look up", "research", "paper", "arxiv",
                 "latest", "news", "compare", "what is", "how does"],
    "writing":  ["write", "draft", "email", "summarize", "summary", "blog",
                 "essay", "rewrite", "translate"],
    "devops":   ["deploy", "server", "docker", "install", "setup", "configure",
                 "runpod", "stripe", "endpoint", "api key", "cron", "webhook"],
    "data":     ["csv", "sql", "query", "plot", "analyze", "dataframe", "table"],
}


def categorize(s: dict) -> str:
    """Bucket a trace by a keyword vote over its user turns."""
    if s.get("_demo_domain"):
        return s["_demo_domain"]
    text = " ".join((m.get("content") or "") for m in s.get("messages", [])
                    if m.get("role") == "user").lower()
    best, best_n = "other", 0
    for cat, kws in TASK_CATEGORIES.items():
        n = sum(text.count(k) for k in kws)
        if n > best_n:
            best, best_n = cat, n
    return best


def model_of(s: dict) -> str:
    """The model that produced this trace."""
    return s.get("model") or "unknown"


def render_trace(messages: list[dict]) -> str:
    """Faithful raw transcript (keeps tool calls + results), for SFT text."""
    parts = []
    for m in messages:
        role = m.get("role", "?")
        content = (m.get("content") or "").strip()
        if role == "tool":
            parts.append(f"<|tool:{m.get('tool_name') or 'tool'}|>\n{content}")
        elif role == "assistant":
            seg = f"<|assistant|>\n{content}"
            if m.get("tool_calls"):
                seg += "\n[tool_calls] " + json.dumps(m["tool_calls"], ensure_ascii=False)
            parts.append(seg)
        elif role in ("user", "system"):
            parts.append(f"<|{role}|>\n{content}")
    return "\n".join(p for p in parts if p.strip())


def export(out_path: str | None = None) -> str:
    """Export the agent's sessions to a JSONL folder via the Hermes CLI."""
    cfg = config.load()
    out_path = out_path or config.rel(cfg["data"].get("hermes_dump",
                "./data/traces_raw/sessions.jsonl"))
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
    """Binary good/bad for calibration. Prefer an explicit variant tag; else heuristic."""
    v = (s.get("_demo_variant") or "").lower()
    if v:
        return 1 if v == "good" else (0 if v == "bad" else None)
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
    excl = set((cfg.get("data") or {}).get("exclude_models", []))
    rows = _read(path)
    return [r for r in rows if (r.get("model") or "") not in excl]


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


def _enriched(sess: list[dict]) -> list[dict]:
    return [{"id": s["id"], "quality": score(s), "label": label(s),
             "model": model_of(s), "category": categorize(s)} for s in sess]


def _cat_cutoff(rows: list[dict], category: str, global_cut: float) -> float:
    """Per-category Youden cutoff when the category has both labels, else global."""
    from self_improve_loop import youden_threshold
    cal = [Trace(r["id"], r["quality"], r["label"]) for r in rows
           if r["category"] == category and r["label"] is not None]
    if any(t.label == 1 for t in cal) and any(t.label == 0 for t in cal):
        return youden_threshold(cal)
    return global_cut


def summarize() -> dict:
    """Summary statistics over the harvested + curated traces (for the user)."""
    from self_improve_loop import youden_threshold, curate
    sess = _sessions()
    rows = _enriched(sess)
    pool = [Trace(r["id"], r["quality"]) for r in rows]
    cal = [Trace(r["id"], r["quality"], r["label"]) for r in rows
           if r["label"] is not None]
    cut = youden_threshold(cal)
    kept = curate(pool, cal)
    qs = sorted(r["quality"] for r in rows)
    pct = lambda p: qs[min(len(qs) - 1, int(p * len(qs)))] if qs else 0.0

    def agg(key: str) -> dict:
        out: dict = {}
        for r in rows:
            d = out.setdefault(r[key], {"n": 0, "kept": 0, "_q": 0.0})
            d["n"] += 1
            d["_q"] += r["quality"]
            if r["quality"] >= cut:
                d["kept"] += 1
        for d in out.values():
            d["mean_q"] = round(d.pop("_q") / d["n"], 3)
            d["kept_frac"] = round(d["kept"] / d["n"], 3)
        return dict(sorted(out.items(), key=lambda kv: -kv[1]["n"]))

    by_cat = agg("category")
    for c in by_cat:
        by_cat[c]["cutoff"] = round(_cat_cutoff(rows, c, cut), 3)

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
        "by_model": agg("model"),
        "by_category": by_cat,
    }
