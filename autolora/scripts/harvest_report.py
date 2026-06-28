"""Step 1-3: export the agent's own traces, score them, curate by the Youden
cutoff, and print summary statistics + the questions the user should answer
before the (paid) training step.

    python scripts/harvest_report.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seams import config, harvest


def main():
    cfg = config.load()
    print("== harvest: exporting the agent's own traces ==")
    path = harvest.export()
    print(f"exported -> {path}\n")

    s = harvest.summarize()
    print("== summary statistics ==")
    print(f"sessions harvested : {s['sessions']}")
    print(f"messages           : {s['messages']}")
    print(f"tool calls         : {s['tool_calls']}")
    print(f"calibration labels : {s['calibration']['labeled']} "
          f"({s['calibration']['good']} good / {s['calibration']['bad']} bad)")
    q = s["quality"]
    print(f"quality dist       : min {q['min']}  p50 {q['p50']}  "
          f"p90 {q['p90']}  max {q['max']}")
    print(f"AUC/Youden cutoff  : {s['cutoff']:.3f}")
    print(f"kept after curation: {s['kept']}/{s['of']}  ({s['kept_frac']*100:.1f}%)\n")

    print("== questions for the user (defaults from config.yaml) ==")
    print(f"1. Benchmarks to gate on? default: {list(cfg['benchmarks'])}")
    print(f"2. Promotion thresholds?  default: {cfg['benchmarks']}")
    print(f"3. RunPod spend cap?      default: ${cfg['payment']['spend_cap_usd']:.2f}")
    print(f"4. Push adapter to?       default: {cfg['hf']['adapter_repo']}")
    print(f"5. Convert to GGUF?       default: {cfg['hf']['gguf']} "
          f"({cfg['hf']['gguf_quant']})")
    print("\nNext: python scripts/run_cycle.py")


if __name__ == "__main__":
    main()
