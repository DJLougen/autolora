# AGENTS.md — autolora

Context for agents/contributors working in this repo. Read before editing.

## What this is
A **Hermes skill** that fine-tunes a new version of the agent on its **own
exported traces** and hot-swaps it in. The loop:

```
hermes sessions export            # raw traces (full trajectories)
  -> score + AUC/Youden curate    # keep good traces; per-model + per-category stats
  -> summary stats + questions     # inform the user
  -> [Stripe pay, under cap]       # pay for GPU
  -> RunPod + Unsloth LoRA train   # on the kept RAW traces
  -> push adapter to HF -> pull
  -> eval candidate vs incumbent   # gsm8k + arc (user benchmarks)
  -> SwitchPolicy (user thresholds)
       pass -> kill old server, start new, reboot agent w/ serve/HANDOFF.md
       fail -> hold
  -> swaps.jsonl                    # audit line either way
```

## Layout
```
autolora/                       # the installable skill dir (SKILL.md at its root)
  SKILL.md                      # skill manifest (frontmatter: name/description/...)
  self_improve_loop.py          # PROVIDED controller — DO NOT rewrite
  config.yaml                   # all knobs
  seams/                        # harvest · train · evaluate · serving · payment · config
  scripts/                      # harvest_report.py · run_cycle.py
  runpod/train_unsloth.py       # remote Unsloth trainer (runs ON the pod)
docs/                           # GOAL.md, hackathon_build_plan.md
README.md  LICENSE  AGENTS.md
```
Work happens inside `autolora/`. Run scripts from that dir.

## Run / verify
```bash
cd autolora
python scripts/harvest_report.py            # steps 1-3, instant, real
EVAL_LIMIT=3 LOCAL_MAX_STEPS=8 DEMO_SWAP=1 python scripts/run_cycle.py
```
Env overrides (demo/speed): `EVAL_LIMIT`, `LOCAL_MAX_STEPS`, `DEMO_SWAP=1`
(force the swap branch to exercise handoff), `HERMES_LIVE_SWAP=1` (actually run
the kill/restart — off by default so demos don't kill the session).

## Seam contract (each is a `run_cycle` argument)
- `harvest.harvest_traces() -> list[Trace]` — Trace(id, quality)
- `harvest.label_calibration() -> list[Trace]` — labeled subset for Youden
- `train.train_lora(kept) -> adapter_ref` — RunPod+Unsloth (key) or local fallback
- `evaluate.eval_on_benchmarks(adapter|None) -> {bench: score}`
- `serving.swap(adapter, ctx) -> dict` — handoff md + staged kill/restart
- curation/`SwitchPolicy`/`run_cycle` live in `self_improve_loop.py`

## Conventions (HARD)
- **Never rewrite `self_improve_loop.py`** — it's the provided controller. Use its
  `Trace`, `youden_threshold`, `curate`, `BenchRule`, `SwitchPolicy`, `run_cycle`.
- **Keep traces RAW.** The dataset preserves the full message trajectory (tool
  calls + results) plus `model` (provenance) and `category` per row. Don't trim.
- Quality scoring + labels are heuristic over real session signals
  (`seams/harvest.py`: `score`, `label`). Task buckets: `categorize`.
- **Secrets**: read `RUNPOD_API_KEY` / `STRIPE_API_KEY` / `HF_TOKEN` from env/.env
  at runtime. Never print or commit them. Missing key -> clearly-labeled dry-run,
  never a fake success.
- The agent's exported sessions are **private** — `.gitignore` covers
  `**/data/`, `**/swaps.jsonl`, `**/adapters/`, `**/serve/`, `.env`. Don't commit them.
- The two user knobs: `curation.scorer` and `benchmarks` (the SwitchPolicy) in
  `config.yaml`. Everything else is plumbing.

## Gotchas (learned the hard way)
- **Hermes skills dir is platform-specific**: `~/.hermes/skills` on Linux/macOS,
  `%LOCALAPPDATA%\hermes\skills` on Windows. Install the skill there (mlops/autolora).
  Find it: `python -c "from tools.skills_tool import SKILLS_DIR; print(SKILLS_DIR)"`.
- **Hub install is blocked by design**: the scanner flags credential reads +
  subprocess as "dangerous"; `--force` won't override. Install as a **local**
  skill (copy into the skills dir) — you trust your own repo.
- **Base model** `Qwen/Qwen3.5-4B` is multimodal; loads as `Qwen3_5ForCausalLM`
  via `AutoModelForCausalLM`. Hybrid attention -> LoRA targets include
  `in_proj_*` / `out_proj` plus the usual `q/k/v/o/gate/up/down_proj`.
- **Run torch as scripts**, not in an IPython/eval kernel (CUDA init can hang there).
- **datasets**: `mmlu` / `hellaswag` cached metadata is incompatible with the local
  `datasets` build — use `arc` (ARC-Easy) as the guard; `gsm8k` works.
- `trl` is too old for `transformers` 5.x locally -> the local fallback trainer
  hand-rolls peft + `Trainer`. The RunPod path uses Unsloth.

## After changing skill code
Re-sync the installed copy so Hermes picks it up:
```bash
cp -r autolora "$(python -c 'from tools.skills_tool import SKILLS_DIR;print(SKILLS_DIR)')/mlops/autolora"
```
