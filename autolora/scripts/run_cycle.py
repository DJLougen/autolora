"""One unattended self-improvement cycle, wiring the provided controller:

  export traces -> score/curate (Youden) -> summary -> [pay] -> RunPod+Unsloth
  train -> push HF -> eval vs incumbent -> SwitchPolicy -> swap (new server +
  handoff md) + swaps.jsonl.

  python scripts/run_cycle.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from self_improve_loop import run_cycle, SwitchPolicy, BenchRule
from seams import config, harvest, train, evaluate, serving


def banner(t):
    print("\n" + "=" * 64 + f"\n{t}\n" + "=" * 64)


def main():
    cfg = config.load()
    log_path = config.rel(cfg["log_path"])

    banner("STEP 1-3  harvest + score + summary")
    harvest.export()
    s = harvest.summarize()
    print(f"sessions {s['sessions']} | messages {s['messages']} | "
          f"tool_calls {s['tool_calls']}")
    print(f"Youden cutoff {s['cutoff']:.3f} | kept {s['kept']}/{s['of']} "
          f"({s['kept_frac']*100:.1f}%)")

    banner("STEP 4  measure incumbent (base) on benchmarks")
    incumbent = evaluate.eval_on_benchmarks(None)

    # gate only on benchmarks that actually produced a score (e.g. terminal_bench
    # is dropped when its tmax/Harbor infra is absent)
    rules = {b: BenchRule(**v) for b, v in cfg["benchmarks"].items()
             if b in incumbent}
    dropped = [b for b in cfg["benchmarks"] if b not in incumbent]
    if dropped:
        print(f"[policy] no score for {dropped} this run -> dropped from gate")
    policy = SwitchPolicy(rules)
    print(f"SwitchPolicy: {{{', '.join(rules)}}}")

    shared = {}

    def eval_wrap(adapter):
        sc = evaluate.eval_on_benchmarks(adapter)
        shared["candidate"] = sc
        return sc

    def swap_wrap(adapter):
        with open(log_path) as f:
            last = json.loads(f.readlines()[-1])
        ctx = {"adapter": adapter, "cutoff": round(s["cutoff"], 3),
               "kept": s["kept"], "of": s["of"], "incumbent": last["incumbent"],
               "candidate": last["candidate"], "swap": last["swap"],
               "reason": last["reason"]}
        return serving.swap(adapter, ctx)

    banner("STEP 5-8  pay -> train(RunPod+Unsloth) -> push HF -> eval -> decide")
    live = run_cycle(harvest.harvest_traces, harvest.label_calibration,
                     train.train_lora, eval_wrap, swap_wrap, policy,
                     incumbent, log_path)

    with open(log_path) as f:
        decision = json.loads(f.readlines()[-1])
    banner("RESULT")
    print(f"incumbent : {decision['incumbent']}")
    print(f"candidate : {decision['candidate']}")
    print(f"decision  : {'SWAP' if decision['swap'] else 'HOLD'} — {decision['reason']}")
    if os.getenv("DEMO_SWAP") == "1" and not decision["swap"]:
        banner("DEMO  forcing swap to exercise handoff + restart (honest decision was HOLD)")
        ctx = {"adapter": decision["adapter"], "cutoff": round(s["cutoff"], 3),
               "kept": s["kept"], "of": s["of"], "incumbent": decision["incumbent"],
               "candidate": decision["candidate"], "swap": decision["swap"],
               "reason": "DEMO forced — " + decision["reason"]}
        serving.swap(decision["adapter"], ctx)
    if cfg["hf"]["gguf"] and decision["swap"]:
        print(f"[gguf] convert command: python llama.cpp/convert_hf_to_gguf.py "
              f"{decision['adapter']} --outtype {cfg['hf']['gguf_quant']}")
    print(f"audit line appended -> {log_path}")
    print(f"live model scores: {live}")


if __name__ == "__main__":
    main()
