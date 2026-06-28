"""
Self-improving model-lab loop for a Hermes agent.

The agent harvests its own tool-use traces from the autoresearch harness,
curates them, fine-tunes a LoRA candidate on the kept set, evals the candidate
on the benchmarks the user cares about, and swaps the running adapter in if the
candidate clears the user's thresholds. The swap is logged, never silent.

Two things are configurable:
  1. the trace curator  (defaults to an AUC / Youden's J quality filter)
  2. the switch policy   (user sets per-benchmark thresholds)

Everything else (the harness, the trainer, the eval harness) is a seam you wire
to your own code. Pure stdlib. Run `python self_improve_loop.py` for a demo of
the curation cutoff and a switch decision on toy benchmark scores.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable, Optional


# --- 1. curation: default AUC / Youden's J cutoff, swappable ---------------- #

@dataclass
class Trace:
    id: str
    quality: float                 # your scalar quality score for this trace
    label: Optional[int] = None    # 1 good / 0 bad, on calibration traces only


def youden_threshold(calibration: list[Trace]) -> float:
    """Quality cutoff that maximizes TPR - FPR on labeled traces.

    The Ornstein v1 public filter: sweep candidate cutoffs, keep the one with
    the best Youden's J. Replace this whole function to customize curation.
    """
    labeled = [t for t in calibration if t.label is not None]
    pos = sum(t.label for t in labeled)
    neg = len(labeled) - pos
    if pos == 0 or neg == 0:
        return float("-inf")       # no usable labels: keep everything
    best_j, best_cut = -1.0, float("-inf")
    for cut in sorted({t.quality for t in labeled}):
        tp = sum(1 for t in labeled if t.quality >= cut and t.label == 1)
        fp = sum(1 for t in labeled if t.quality >= cut and t.label == 0)
        j = tp / pos - fp / neg
        if j > best_j:
            best_j, best_cut = j, cut
    return best_cut


def curate(traces: list[Trace], calibration: list[Trace],
           select_threshold: Callable[[list[Trace]], float] = youden_threshold
           ) -> list[Trace]:
    """Keep traces at or above the chosen quality cutoff."""
    cut = select_threshold(calibration)
    return [t for t in traces if t.quality >= cut]


# --- 2. switch policy: user-set per-benchmark thresholds -------------------- #

@dataclass
class BenchRule:
    min_delta: float = 0.0         # candidate must beat incumbent by this much
    max_regression: float = 0.0    # allowed drop on a guard metric (>= 0)
    guard: bool = False            # True = guard only, no improvement required


@dataclass
class SwitchPolicy:
    rules: dict[str, BenchRule]

    def evaluate(self, incumbent: dict[str, float],
                 candidate: dict[str, float]) -> tuple[bool, str]:
        """Return (swap?, reason). Swap only if every rule passes."""
        improved_any = False
        for bench, rule in self.rules.items():
            if bench not in candidate or bench not in incumbent:
                return False, f"missing scores for {bench}"
            delta = candidate[bench] - incumbent[bench]
            if rule.guard:
                if delta < -rule.max_regression:
                    return False, f"regressed on guard {bench}: {delta:+.4f}"
            else:
                if delta < rule.min_delta:
                    return False, (f"{bench} gain {delta:+.4f} below "
                                   f"threshold {rule.min_delta:+.4f}")
                improved_any = True
        if not improved_any:
            return False, "no target benchmark improved"
        gains = ", ".join(f"{b} {candidate[b] - incumbent[b]:+.4f}"
                          for b, r in self.rules.items() if not r.guard)
        return True, f"clears all thresholds ({gains})"


# --- 3. one cycle: heavy steps are seams to your own code ------------------- #

def run_cycle(harvest_traces, label_calibration, train_lora,
              eval_on_benchmarks, swap_adapter,
              policy: SwitchPolicy, incumbent_scores: dict,
              log_path: str = "swaps.jsonl") -> dict:
    """One self-improvement cycle. Each verb-named argument is your code:

        harvest_traces()         -> list[Trace]       (autoresearch harness)
        label_calibration()      -> list[Trace]       (labeled subset for Youden)
        train_lora(kept)         -> adapter_path
        eval_on_benchmarks(path) -> dict[str, float]  (caliper)
        swap_adapter(path)       -> None              (hot-swap in Hermes)

    Returns the score dict of whichever model is now live.
    """
    traces = harvest_traces()
    kept = curate(traces, label_calibration())
    adapter = train_lora(kept)
    candidate_scores = eval_on_benchmarks(adapter)

    swap, reason = policy.evaluate(incumbent_scores, candidate_scores)
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "ts": time.time(), "adapter": adapter,
            "kept": len(kept), "of": len(traces),
            "incumbent": incumbent_scores, "candidate": candidate_scores,
            "swap": swap, "reason": reason,
        }) + "\n")

    if swap:
        swap_adapter(adapter)
        return candidate_scores
    return incumbent_scores


# --- demo: curation cutoff + a swap and a hold, no external seams ----------- #

if __name__ == "__main__":
    import random
    random.seed(1)

    calib = [Trace(f"c{i}",
                   quality=random.gauss(0.70 if i % 2 else 0.40, 0.10),
                   label=1 if i % 2 else 0) for i in range(20)]
    pool = [Trace(f"t{i}", quality=random.gauss(0.60, 0.15)) for i in range(100)]

    cut = youden_threshold(calib)
    kept = curate(pool, calib)
    print(f"curation: Youden cutoff {cut:.3f}, "
          f"kept {len(kept)}/{len(pool)} traces\n")

    # user-set policy: improve gsm8k by >= 1.5 pts, do not drop mmlu by > 0.5 pt
    policy = SwitchPolicy({
        "gsm8k": BenchRule(min_delta=0.015),
        "mmlu":  BenchRule(max_regression=0.005, guard=True),
    })
    incumbent = {"gsm8k": 0.620, "mmlu": 0.710}
    candidates = {
        "candidate A": {"gsm8k": 0.645, "mmlu": 0.709},  # +2.5 gsm8k, -0.1 mmlu
        "candidate B": {"gsm8k": 0.660, "mmlu": 0.700},  # +4.0 gsm8k, -1.0 mmlu
    }
    for name, cand in candidates.items():
        swap, reason = policy.evaluate(incumbent, cand)
        print(f"{name}: {'SWAP' if swap else 'HOLD'}: {reason}")
