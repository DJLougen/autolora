"""Seam: train a LoRA candidate on the kept traces.

Primary path: RunPod GPU pod running Unsloth, pushing the adapter to the HF Hub
(paid for via the Stripe seam). Fallback for the keyless demo: a tiny local LoRA
so the rest of the pipeline (HF push, eval, swap) has a real artifact to move.

train_lora(kept) -> adapter reference (local dir or HF repo id).
"""
from __future__ import annotations

import json
import os
import time

from self_improve_loop import Trace
from seams import config, harvest, payment

def build_dataset(kept: list[Trace]) -> tuple[str, int]:
    """Turn kept sessions into a RAW SFT dataset (one JSON per line).

    Each row keeps the full raw message trajectory (tool calls + results), the
    model that produced it, the task category, and the quality score — plus a
    faithful rendered transcript for training.
    """
    cfg = config.load()
    recs = harvest.load_records()
    out = config.rel(cfg["data"]["dataset"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for t in kept:
            s = recs.get(t.id)
            if not s:
                continue
            raw = s.get("messages", [])
            text = harvest.render_trace(raw)
            if not text.strip() or len(raw) < 2:
                continue
            f.write(json.dumps({
                "id": t.id,
                "model": harvest.model_of(s),       # provenance: who made this trace
                "category": harvest.categorize(s),  # task bucket
                "quality": t.quality,
                "messages": raw,                    # the RAW trajectory, untrimmed
                "text": text,                       # faithful transcript for SFT
            }, ensure_ascii=False) + "\n")
            n += 1
    return out, n


def _runpod_spec(cfg: dict, dataset: str) -> dict:
    tc = cfg["train"]
    return {
        "image": tc["image"], "gpu": tc["gpu"], "engine": tc["engine"],
        "base_model": cfg["base_model"], "dataset": os.path.basename(dataset),
        "hf_repo": cfg["hf"]["adapter_repo"], "epochs": tc["epochs"],
        "lora_r": tc["lora_r"], "entry": "runpod/train_unsloth.py",
    }


def _train_runpod(cfg: dict, dataset: str) -> str:
    import runpod  # noqa
    runpod.api_key = os.environ["RUNPOD_API_KEY"]
    spec = _runpod_spec(cfg, dataset)
    print(f"[train] launching RunPod pod ({spec['gpu']}, {spec['engine']}) ...")
    # Upload dataset to HF as a private dataset so the pod can pull it, then run
    # runpod/train_unsloth.py which trains and pushes the adapter to hf.adapter_repo.
    cmd = (f"python train_unsloth.py --base {spec['base_model']} "
           f"--dataset {spec['hf_repo']}-data --push {spec['hf_repo']} "
           f"--epochs {spec['epochs']} --lora_r {spec['lora_r']}")
    pod = runpod.create_pod(name=f"hermes-lora-{int(time.time())}",
                            image_name=spec["image"], gpu_type_id=spec["gpu"],
                            docker_args=cmd, gpu_count=1)
    print(f"[train] pod {pod.get('id')} started; trains + pushes -> {spec['hf_repo']}")
    return cfg["hf"]["adapter_repo"]


def _train_local(cfg: dict, dataset: str, max_steps: int = 10) -> str:
    """Tiny local LoRA so the demo has a real adapter without RunPod."""
    max_steps = int(os.getenv("LOCAL_MAX_STEPS", max_steps))
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, Trainer, TrainingArguments)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    tc = cfg["train"]
    rows = [json.loads(l) for l in open(dataset, encoding="utf-8") if l.strip()]
    rows = rows[: tc["max_train_samples"]]
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], quantization_config=bnb, dtype=torch.bfloat16,
        device_map={"": 0})
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
               "down_proj", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a",
               "out_proj"]
    model = get_peft_model(model, LoraConfig(
        r=tc["lora_r"], lora_alpha=tc["lora_alpha"], lora_dropout=tc["lora_dropout"],
        bias="none", task_type="CAUSAL_LM", target_modules=targets))

    def tok_fn(ex):
        # train on the RAW rendered trajectory (tool calls + results preserved)
        text = ex.get("text") or harvest.render_trace(ex.get("messages", []))
        e = tok(text, truncation=True, max_length=tc["max_seq_len"])
        e["labels"] = e["input_ids"].copy()
        return e
    data = [tok_fn(r) for r in rows]

    def collate(b):
        m = max(len(x["input_ids"]) for x in b)
        pad = tok.pad_token_id
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for x in b:
            k = m - len(x["input_ids"])
            out["input_ids"].append(x["input_ids"] + [pad] * k)
            out["attention_mask"].append([1] * len(x["input_ids"]) + [0] * k)
            out["labels"].append(x["labels"] + [-100] * k)
        return {k: torch.tensor(v) for k, v in out.items()}

    out_dir = config.rel(os.path.join("adapters", f"cand-{int(time.time())}"))
    args = TrainingArguments(output_dir=out_dir, per_device_train_batch_size=1,
                             gradient_accumulation_steps=4, max_steps=max_steps,
                             learning_rate=float(tc["lr"]), bf16=True,
                             logging_steps=2, save_strategy="no", report_to=[],
                             gradient_checkpointing=True,
                             gradient_checkpointing_kwargs={"use_reentrant": False},
                             optim="paged_adamw_8bit")
    Trainer(model=model, args=args, train_dataset=data, data_collator=collate).train()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"[train] local LoRA saved -> {out_dir}")
    return out_dir


def train_lora(kept: list[Trace]) -> str:
    cfg = config.load()
    dataset, n = build_dataset(kept)
    print(f"[train] built SFT dataset: {n} examples -> {dataset}")
    if n == 0:
        raise ValueError("no usable kept traces for training")

    # the spend the cycle needs: GPU compute, under the cap
    payment.ensure(cfg["payment"]["description"])

    if cfg["train"]["provider"] == "runpod" and os.getenv("RUNPOD_API_KEY"):
        return _train_runpod(cfg, dataset)

    print("[train] RUNPOD_API_KEY not set -> demo. RunPod job spec that WOULD run:")
    print("        " + json.dumps(_runpod_spec(cfg, dataset)))
    print("[train] falling back to a tiny LOCAL LoRA so the demo has a real adapter.")
    return _train_local(cfg, dataset)


if __name__ == "__main__":
    from self_improve_loop import curate
    kept = curate(harvest.harvest_traces(), harvest.label_calibration())
    print("kept", len(kept))
    print(train_lora(kept))
