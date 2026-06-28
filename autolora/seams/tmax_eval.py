"""Seam: the agentic gate — evaluate a candidate on tmax's Terminal-Bench via
Harbor (github.com/hamishivi/tmax).

autolora improves a tool-using agent, so the most meaningful promotion gate is a
real terminal-agent benchmark, not just gsm8k. This runs tmax's published Harbor
dataset against the *served* candidate using tmax's Vanillux2Agent (falling back
to Harbor's built-in terminus-2), then returns the pass rate.

Prerequisites (else it skips cleanly, never faked):
  - `harbor` on PATH, or `uv` (we call `uv run harbor`)
  - a sandbox: Docker locally, or DAYTONA_API_KEY for the Daytona cloud sandbox
  - the candidate served at serving.endpoint (vLLM with the LoRA loaded)

terminal_bench(adapter, served_base) -> pass_rate in [0,1] | None (skipped)
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import time
import urllib.request

from seams import config


def _harbor_cmd() -> list[str] | None:
    if shutil.which("harbor"):
        return ["harbor"]
    if shutil.which("uv"):
        return ["uv", "run", "harbor"]
    return None


def _endpoint_up(base: str) -> bool:
    url = base.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=4) as r:
            return r.status == 200
    except Exception:
        return False


def _vanillux_available() -> bool:
    try:
        import Vanillux2Agent  # noqa: F401
        return True
    except Exception:
        return False


def _parse_rewards(job_dir: str) -> float | None:
    rewards = []
    for p in glob.glob(os.path.join(job_dir, "**", "reward.txt"), recursive=True):
        try:
            rewards.append(float(open(p).read().strip()))
        except Exception:
            continue
    if not rewards:
        return None
    return round(sum(1 for r in rewards if r > 0) / len(rewards), 4)


def terminal_bench(adapter: str | None, served_base: str | None = None) -> float | None:
    cfg = config.load()
    tc = cfg.get("tmax", {})
    if not tc.get("enabled", False):
        print("[tmax] disabled in config -> skip")
        return None

    base = served_base or cfg["serving"]["endpoint"]
    harbor = _harbor_cmd()
    if harbor is None:
        print("[tmax] harbor/uv not found -> skip "
              "(install tmax: clone github.com/hamishivi/tmax && uv sync)")
        return None
    if tc.get("env") == "daytona" and not os.getenv("DAYTONA_API_KEY"):
        print("[tmax] env=daytona but DAYTONA_API_KEY unset -> skip "
              "(set it, or use env: docker)")
        return None
    if not _endpoint_up(base):
        print(f"[tmax] candidate not served at {base} -> skip "
              "(serve the adapter with vLLM first)")
        return None

    agent = (["--agent-import-path", tc["agent"]] if _vanillux_available()
             else ["--agent", "terminus-2"])
    job = f"autolora-tmax-{int(time.time())}"
    cmd = harbor + [
        "run", "-d", tc["dataset"], *agent,
        "--model", f"openai/{tc['served_name']}",
        "--agent-kwarg", f"api_base={base}",
        "--env", tc.get("env", "docker"),
        "-l", str(tc.get("limit", 10)), "-k", str(tc.get("k", 1)),
        "--job-name", job,
    ]
    print(f"[tmax] terminal-bench: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, cwd=config.ROOT)
    except Exception as e:
        print(f"[tmax] harbor run failed: {repr(e)[:120]} -> skip")
        return None

    score = _parse_rewards(os.path.join(config.ROOT, "jobs", job))
    print(f"[tmax] terminal-bench pass rate ({adapter or 'base'}): {score}")
    return score
