---
name: autolora
description: Export the agent's own Hermes traces, score and curate them by an
  AUC/Youden cutoff, fine-tune a LoRA candidate on RunPod with Unsloth (paid via
  Stripe, under a cap), push it to the HF Hub, eval it on the user's benchmarks,
  and hot-swap it in if it clears the user's thresholds — restarting onto the new
  server with a handoff note. Logs every swap.
---

When the user asks to "run an improvement cycle" (or on the scheduled cron):

1. **Harvest.** `python scripts/harvest_report.py` — exports the agent's own
   sessions (`hermes sessions export`), scores each trace, fits the Youden cutoff
   on the labeled calibration subset, and prints summary statistics.
2. **Inform + ask.** Show the user: sessions/messages/tool-calls, the quality
   distribution, the AUC/Youden cutoff, and the kept fraction. Then ask the open
   questions (benchmarks to gate on, promotion thresholds, RunPod spend cap, HF
   push target, GGUF yes/no). Defaults live in `config.yaml`.
3. **Pay + train.** `python scripts/run_cycle.py` runs the rest: charge the GPU
   compute via Stripe **under the cap** (surface the charge), launch a RunPod pod
   running Unsloth (`runpod/train_unsloth.py`) on the curated traces, and push the
   adapter to the HF Hub.
4. **Eval.** Pull the adapter back and eval candidate vs incumbent on the user's
   benchmarks (defaults: gsm8k + arc).
5. **Decide (SwitchPolicy).**
   - pass -> pull adapter, (optional GGUF), write `serve/HANDOFF.md`, then run the
     swap command that kills the old server, starts the new one, and reboots the
     agent with the handoff as context. Append the event to `swaps.jsonl`.
   - fail -> keep the incumbent, append the held decision + reason.
6. **Report.** kept/total, the cutoff, candidate vs incumbent scores, the decision
   and reason, and the Stripe charge.

Guardrails:
- Never swap without an eval result and a passing SwitchPolicy.
- Never spend above `payment.spend_cap_usd`. Surface every charge.
- The kill/restart is staged by default; only fires with `HERMES_LIVE_SWAP=1`.
- `.gitignore` covers `.env`, vault, `swaps.jsonl`, and `adapters/`. Never read
  credentials into context.

Config knobs (the only things the user touches):
- `curation.scorer` — default `youden`, swappable.
- `benchmarks` — the SwitchPolicy: per-benchmark `min_delta` / `max_regression` /
  `guard`, and the benchmark set to run.
