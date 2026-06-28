# GOAL — Self-Improving Hermes Loop

A Hermes agent that **improves a version of itself, end to end**:

```
export own traces (hermes sessions export)
  -> score + AUC/Youden curate -> summary stats + questions to user
  -> [Stripe pay, under cap] -> RunPod + Unsloth LoRA train on kept traces
  -> push adapter to HF -> pull back to device
  -> eval candidate vs incumbent on user benchmarks (default gsm8k + arc)
  -> SwitchPolicy (user thresholds)
       pass -> (optional GGUF) -> kill old server, start new one, reboot agent
               with serve/HANDOFF.md as context; append swaps.jsonl
       fail -> hold; append swaps.jsonl
```

Base model: `Qwen/Qwen3.5-4B`. Built around the provided `self_improve_loop.py`
(curation cutoff + SwitchPolicy + run_cycle).

## Status — demo proves every step (verified)
`python scripts/harvest_report.py` and `python scripts/run_cycle.py` run on this
box. One demo cycle showed: 22 sessions harvested, Youden cutoff 0.820, kept
16/22; incumbent eval; Stripe dry-run (capped); RunPod job spec + a real local
LoRA fallback (loss 1.87->1.65); candidate eval; honest SwitchPolicy decision;
`swaps.jsonl` audit line; `serve/HANDOFF.md` + staged kill/restart command.

This is a **demo of the mechanism**, not a quality run (tiny train + tiny eval).

## To go live (creds set up later — "that crap after")
- `RUNPOD_API_KEY` -> real RunPod+Unsloth training instead of local fallback.
- `STRIPE_API_KEY` -> real capped charge instead of dry-run (`pip install stripe`).
- `hf.adapter_repo` -> your HF namespace; `HF_TOKEN` already present.
- `HERMES_LIVE_SWAP=1` -> actually run the kill/restart on a pass.

## User knobs (config.yaml)
- `curation.scorer` (default `youden`, swappable)
- `benchmarks` — the SwitchPolicy: per-bench `min_delta`/`max_regression`/`guard`
  and which benchmarks to run.
- `payment.spend_cap_usd`, `hf.gguf`, `train.gpu`/`engine`.

## Files
- `self_improve_loop.py` — provided controller (unchanged)
- `seams/{harvest,train,evaluate,serving,payment,config}.py` — the seams
- `runpod/train_unsloth.py` — remote Unsloth trainer (runs on the pod)
- `scripts/harvest_report.py` — steps 1-3 (export+score+summary+questions)
- `scripts/run_cycle.py` — wires the whole cycle (+ `DEMO_SWAP=1` to show handoff)
- `skill/SKILL.md` — the Hermes skill
