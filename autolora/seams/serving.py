"""Seam: activate the new model. Pulls the adapter (from HF or local), writes a
HANDOFF.md that briefs the freshly-booted model on what just happened, and emits
the command that kills the old server and starts the new one with that context.

By default the kill/restart command is STAGED (printed + written), not executed —
so a demo doesn't tear down the live session. Set HERMES_LIVE_SWAP=1 to run it.
"""
from __future__ import annotations

import json
import os
import subprocess
import time

from seams import config

HANDOFF = "serve/HANDOFF.md"


def pull_adapter(adapter: str) -> str:
    """Ensure the adapter is on this device (download from HF if it's a repo id)."""
    if os.path.isdir(config.rel(adapter)):
        return config.rel(adapter)
    from huggingface_hub import snapshot_download
    local = snapshot_download(repo_id=adapter)
    print(f"[serve] pulled adapter from HF: {adapter} -> {local}")
    return local


def write_handoff(ctx: dict) -> str:
    cfg = config.load()
    path = config.rel(HANDOFF)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    md = f"""# Handoff — you were just hot-swapped

You are the **new** model for this Hermes session. A self-improvement cycle just
fine-tuned a LoRA on the agent's own traces and promoted it. Continue the user's
work from here.

## What just happened
- base model: `{cfg['base_model']}`
- new adapter: `{ctx.get('adapter')}`
- traces kept / total: {ctx.get('kept')} / {ctx.get('of')}  (Youden cutoff {ctx.get('cutoff')})
- benchmarks (incumbent -> candidate):
"""
    inc, cand = ctx.get("incumbent", {}), ctx.get("candidate", {})
    for b in cand:
        md += f"  - {b}: {inc.get(b)} -> {cand.get(b)}\n"
    md += f"- decision: **{'SWAP' if ctx.get('swap') else 'HOLD'}** — {ctx.get('reason')}\n"
    md += f"- swapped at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[serve] wrote handoff context -> {path}")
    return path


def restart_command(model_ref: str, handoff: str) -> str:
    cfg = config.load()
    port = cfg["serving"]["endpoint"].rsplit(":", 1)[-1].split("/")[0]
    return (
        f"# kill old server on :{port}, start new one, reboot agent with context\n"
        f"pkill -f 'vllm serve' || true\n"
        f"vllm serve {cfg['base_model']} --enable-lora "
        f"--lora-modules cand={model_ref} --port {port} &\n"
        f"hermes model --provider openai --endpoint {cfg['serving']['endpoint']}\n"
        f"hermes -z \"$(cat {handoff})\"\n"
    )


def swap(adapter: str, ctx: dict | None = None) -> dict:
    cfg = config.load()
    ctx = dict(ctx or {}); ctx["adapter"] = adapter
    local = pull_adapter(adapter)
    handoff = write_handoff(ctx)
    # publish which adapter is live
    ptr = config.rel(cfg["serving"]["active_pointer"])
    os.makedirs(os.path.dirname(ptr), exist_ok=True)
    with open(ptr, "w") as f:
        json.dump({"adapter": adapter, "local": local, "ts": time.time()}, f)
    cmd = restart_command(local, handoff)
    if os.getenv("HERMES_LIVE_SWAP") == "1":
        print("[serve] HERMES_LIVE_SWAP=1 -> executing kill/restart")
        subprocess.Popen(cmd, shell=True)
        return {"status": "restarting", "adapter": adapter, "handoff": handoff}
    print("[serve] staged swap (set HERMES_LIVE_SWAP=1 to run). Command:\n" + cmd)
    return {"status": "staged", "adapter": adapter, "handoff": handoff,
            "command": cmd}
