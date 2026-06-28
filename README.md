# autolora

**A Hermes skill that fine-tunes a new version of the agent on its own traces and hot-swaps it in — automatically.**

`autolora` closes the loop: the agent exports its own run history, keeps the good
traces (AUC/Youden curation), trains a LoRA candidate on **RunPod + Unsloth**
(paid for via **Stripe**, under a cap), evaluates it against the live model on
benchmarks *you* choose, and promotes it **only if it clears your thresholds** —
then restarts onto the new server, leaving the fresh model a note about what just
happened. Every swap is logged. Nothing silent.

```
hermes sessions export            # 1. export the agent's own traces
  -> score + AUC/Youden curate    # 2. keep the good ones (cutoff + kept fraction)
  -> summary stats + questions     # 3. inform the user, confirm knobs
  -> [Stripe pay, under cap]       # 4. pay for GPU compute
  -> RunPod + Unsloth LoRA train   # 5. fine-tune a candidate on the kept traces
  -> push adapter to HF -> pull    # 6. ship it to the Hub, pull it back
  -> eval candidate vs incumbent   # 7. gsm8k + arc (your benchmarks)
  -> SwitchPolicy (your thresholds)# 8. promote only if it clears the bar
       pass -> (optional GGUF) -> kill old server, start new one,
               reboot agent with serve/HANDOFF.md as context
       fail -> hold
  -> swaps.jsonl                    # audit line either way
```

Base model: `Qwen/Qwen3.5-4B`. Built around `self_improve_loop.py` (the curation
cutoff + `SwitchPolicy` + `run_cycle` controller).

## Install (as a Hermes skill)

```bash
# direct (owner/repo/skill-dir) — bundles SKILL.md + all code:
hermes skills install DJLougen/autolora/autolora --yes
hermes skills list            # shows: autolora

# or add the repo as a tap, then install by name:
hermes skills tap add DJLougen/autolora
hermes skills install autolora --yes
```

Then just ask Hermes to **"run an improvement cycle."**

## Run directly

```bash
git clone https://github.com/DJLougen/autolora && cd autolora/autolora
python scripts/harvest_report.py   # steps 1-3: export + score + summary + questions
python scripts/run_cycle.py        # the full cycle
```

The cycle runs end-to-end even without paid credentials: missing keys make the
paid steps **dry-run** (clearly labeled, never faked), with a small **local LoRA
fallback** so the rest of the pipeline has a real artifact to move. The kill /
restart swap is **staged** (printed + written) by default so a demo never tears
down the live session.

## Go live

| Want | Set |
|---|---|
| Real RunPod + Unsloth training | `RUNPOD_API_KEY` |
| Real capped Stripe charge | `STRIPE_API_KEY` (+ `pip install stripe`) |
| HF push / pull-back | `hf.adapter_repo` in `config.yaml` (`HF_TOKEN` for push) |
| GGUF export after training | `hf.gguf: true` |
| Actually fire the kill/restart | `HERMES_LIVE_SWAP=1` |

## The two knobs that matter (`config.yaml`)

```yaml
curation:
  scorer: youden          # AUC/Youden cutoff, swappable
benchmarks:               # the SwitchPolicy — your promotion bar
  gsm8k: {min_delta: 0.01}                    # must gain >= 1 pt
  arc:   {max_regression: 0.01, guard: true}  # may not drop > 1 pt
```

## Layout

```
SKILL.md                     the Hermes skill manifest
self_improve_loop.py         controller: Youden curation + SwitchPolicy + run_cycle
seams/                       harvest · train · evaluate · serving · payment · config
runpod/train_unsloth.py      remote Unsloth trainer (runs on the pod)
scripts/                     harvest_report.py · run_cycle.py
docs/                        design notes (GOAL.md, build plan)
```

## Safety

The agent's exported sessions are **private** and never committed
(`.gitignore` covers `data/`, `.env`, vaults, `swaps.jsonl`, `adapters/`,
`serve/`). The spend cap is always enforced. Credentials are read from the
environment at run time, never into agent context.

## License

Apache-2.0.
