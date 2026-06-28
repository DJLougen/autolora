# Hermes Accelerated Business Hackathon — Build Plan

**Objective.** Ship a filmable MVP of a Hermes agent that fine-tunes a version of
itself on its own exported traces and hot-swaps the result into the running
session, gated by AUC trace curation and user-set benchmark thresholds.

**Deadline.** EOD Tue 2026-06-30. **Judged on:** usefulness, viability,
presentation. Optimize for a clean 60-90s demo over feature count. Target is a
top-3 place (every tier ships a DGX Spark), which means low-variance presentation
beats an ambitious half-built system.

---

## Win condition (what the video must prove)

One unattended cycle where Hermes:
1. harvests its own traces,
2. curates them (cutoff shown on screen),
3. LoRA-fine-tunes a candidate locally,
4. evals candidate vs the live model on user-named benchmarks,
5. swaps the adapter in live when thresholds are met (model visibly changes
   mid-conversation),
6. logs the swap,
7. performs one real Stripe spend the cycle needs, under a cap.

Nothing silent, nothing faked beyond optionally pre-baked compute.

---

## Non-goals (do not build)

- **No MSPRT / sequential-probability gate.** The switch is fixed user
  thresholds. A parked `msprt_promotion_gate.py` exists; ignore it for this.
- **No silent self-replacement.** Every swap is governed by user-authored
  thresholds and writes an audit line.
- **No cloud training.** Fine-tune runs locally on the Spark. Keep the base model
  small and use LoRA so a cycle is minutes.
- **No new eval/trainer/harness.** Wire the existing ones (caliper, the LoRA
  trainer, the autoresearch harness) as seams.
- **No scope creep.** The loop is the spine. Everything else is one task each.

---

## Architecture

Loop (one cycle):

```
harvest_traces
  -> curate(scorer = AUC/Youden, swappable)
  -> train_lora(kept)
  -> eval_on_benchmarks(caliper)
  -> SwitchPolicy.evaluate(user thresholds)
       pass -> swap_adapter + append swaps.jsonl
       fail -> append swaps.jsonl (held + reason)
```

Configurable (the only two things a user touches):
1. **curation scorer** — defaults to the Youden's-J AUC cutoff, replaceable.
2. **SwitchPolicy** — per-benchmark `min_delta` and no-regression guards the user
   sets.

Seams (existing code the loop calls):
- `harvest_traces()` -> `list[Trace]` with quality scores, plus a labeled
  calibration subset, from the autoresearch harness.
- `train_lora(kept)` -> adapter dir (axolotl/trl).
- `eval_on_benchmarks(adapter)` -> `{bench: score}` (caliper).
- `swap_adapter(adapter)` -> activate the new LoRA in the local serving layer.

Platform:
- Hermes runs the cycle as a self-authored skill, on cron or on command.
- Base model served locally on the Spark via an adapter-swappable backend
  (vLLM `--enable-lora` or equivalent), exposed as an OpenAI-compatible endpoint
  Hermes points at. `swap_adapter()` activates the new adapter so the next turn
  uses it.
- One real Stripe money action via the payments skills, under a spend cap.
- Optional: run Hermes inside NemoClaw / OpenShell for the safety story.

---

## Provided: `self_improve_loop.py` (use as-is, build around it)

Already implemented and tested. Do not reimplement:

- `Trace(id, quality, label=None)`
- `youden_threshold(calibration) -> cutoff`  (default curator, swappable)
- `curate(traces, calibration, select_threshold=youden_threshold) -> kept`
- `BenchRule(min_delta=0.0, max_regression=0.0, guard=False)`
- `SwitchPolicy(rules: dict[str, BenchRule]).evaluate(incumbent, candidate) -> (swap, reason)`
- `run_cycle(harvest_traces, label_calibration, train_lora, eval_on_benchmarks, swap_adapter, policy, incumbent_scores, log_path) -> live_scores`

Fill the seam callables and wire `run_cycle`.

---

## Repo layout

```
.
├── self_improve_loop.py        # provided controller, do not rewrite
├── config.yaml                 # base model, scorer, SwitchPolicy rules, spend cap
├── seams/
│   ├── harvest.py              # autoresearch export + calibration labels
│   ├── train.py                # LoRA trainer wrapper
│   ├── evaluate.py             # caliper wrapper
│   └── serving.py              # swap_adapter against the local endpoint
├── serve/                      # local adapter-swappable model server
├── skill/SKILL.md              # the Hermes skill
├── scripts/run_cycle.py        # entrypoint that assembles seams + run_cycle
├── swaps.jsonl                 # audit log (gitignored)
└── .gitignore                  # MUST cover Stripe vault, .env, swaps.jsonl, adapters
```

---

## Tasks (in order, each with an acceptance gate)

- [ ] **T0 — Repo + config.** Scaffold the tree, drop in `self_improve_loop.py`,
  write `config.yaml` (base model id, scorer choice, SwitchPolicy rules, spend
  cap). *Done when:* `python self_improve_loop.py` prints the curation cutoff plus
  a swap and a hold.

- [ ] **T1 — Local adapter-swappable serving.** Serve the small base model with
  LoRA hot-load enabled; expose an OpenAI-compatible endpoint; point Hermes at it
  (`hermes model`, custom endpoint). *Done when:* Hermes chats through the local
  endpoint, and activating a LoRA adapter via an API call changes the next reply.

- [ ] **T2 — Seam: harvest + calibration.** Wire the autoresearch export into
  `harvest_traces()`; produce a labeled calibration subset. *Done when:* it
  returns traces with quality scores and `curate()` keeps a sane fraction.

- [ ] **T3 — Seam: train_lora.** Wire the LoRA trainer on the kept traces.
  *Done when:* it produces an adapter dir from kept traces in minutes on the
  Spark. **Pre-bake one adapter now** as the recording fallback.

- [ ] **T4 — Seam: eval.** Wire caliper on the user's benchmarks. *Done when:* it
  returns `{bench: score}` for both incumbent and candidate.

- [ ] **T5 — Wire run_cycle.** Assemble the seams in `scripts/run_cycle.py`.
  *Done when:* one unattended cycle runs, decides via `SwitchPolicy`, writes
  `swaps.jsonl`, and on a pass calls `swap_adapter` so the live model changes.

- [ ] **T6 — Hermes skill.** Author `skill/SKILL.md` (skeleton below); install with
  `hermes skills install ./skill`. *Done when:* `hermes skills list` shows it and
  asking Hermes to "run an improvement cycle" executes `run_cycle`.

- [ ] **T7 — One real Stripe spend.** Install a payments skill
  (`hermes skills install official/payments/stripe-projects`). Have the cycle
  provision/pay for one small service it genuinely needs, under the spend cap.
  Recommended: a logging/observability store for swap + eval records (e.g.
  ClickHouse via Stripe Projects; run `stripe projects catalog` first to confirm
  availability). Fallback: pay a per-call eval/grader API via the MPP skill.
  *Done when:* a real charge appears in the Stripe dashboard and spend is capped.
  **Before any run: confirm `.gitignore` covers the Stripe vault and `.env`.**

- [ ] **T8 (optional) — NemoClaw sandbox.** Run Hermes inside OpenShell per
  NemoClaw docs for the safety story. *Done when:* the cycle runs sandboxed.
  Skip if short on time.

- [ ] **T9 — Record.** Capture the win-condition sequence (checklist below).
  Pre-bake compute if live timing is tight; the judged artifact is the loop and
  the swap, not wall-clock training.

- [ ] **T10 — Submit.** Tweet (lead with the hook) tagging @NousResearch plus a
  short write-up; drop the link in the Nous Discord submissions channel; fill the
  submission form.

---

## `skill/SKILL.md` skeleton

```markdown
---
name: self-improve-cycle
description: Fine-tune a candidate on the agent's own exported traces, eval it on
  the user's benchmarks, and hot-swap it in if it clears the user's thresholds.
  Trains locally. Logs every swap. Spends only under the configured cap.
---

On request ("run an improvement cycle") or the scheduled cron:

1. Harvest traces from the autoresearch harness and a labeled calibration subset.
2. Curate: keep traces above the AUC/Youden cutoff (config: scorer). Report the
   cutoff and the kept fraction.
3. Provision/confirm the run's logging store via the Stripe Projects skill, under
   the spend cap in config. Never exceed the cap. Surface the charge.
4. Fine-tune a LoRA candidate on the kept traces (local, on the Spark).
5. Eval candidate and incumbent on the user's benchmarks via caliper.
6. Apply the SwitchPolicy (config: per-benchmark thresholds and guards):
   - pass -> activate the new adapter in the serving layer, tell the user the
     model changed, and append the event to swaps.jsonl.
   - fail -> keep the incumbent and append the held decision plus reason.
7. Report kept/total, the cutoff, candidate vs incumbent scores, the decision and
   reason, and the Stripe charge.

Guardrails:
- Never swap without an eval result and a passing SwitchPolicy.
- Never spend above the configured cap. Surface every charge.
- Ensure .gitignore covers the Stripe vault and .env before running. Never read
  credentials into context.
```

---

## Demo checklist (must be on screen)

- [ ] the agent exporting its own traces
- [ ] the curation cutoff (ROC / Youden) and the kept fraction
- [ ] the fine-tune starting (or pre-baked, stated honestly)
- [ ] candidate vs incumbent scores against the user-set thresholds
- [ ] the swap firing; the model/version changing mid-conversation
- [ ] one real Stripe charge in the dashboard
- [ ] the `swaps.jsonl` audit line

---

## Write-up hook (tweet's first line)

> Hermes fine-tunes a version of itself on its own traces and hot-swaps it in
> live, with AUC trace curation and user-set benchmark thresholds keeping it
> honest. Trains locally on a DGX Spark; pays for what it needs through Stripe,
> under a cap.

---

## Risks / fallbacks

- Live fine-tune too slow to film -> pre-bake one adapter; show the trigger and
  the swap.
- Stripe rail will not authorize -> switch rail (Projects, MPP, or Link); confirm
  before recording.
- Adapter hot-load flaky -> fall back to swapping the served model/endpoint Hermes
  points at; the on-camera "model changed" still lands.
- Credential leak -> verify `.gitignore` covers the vault and `.env`; never cat
  credentials into agent context.
```

