"""Seam: evaluate a model (incumbent base, or base+adapter candidate) on the
user's benchmarks. Defaults: gsm8k (exact-match) and mmlu (multiple-choice).
Real, just small — `eval.limit` keeps a demo cycle quick.

eval_on_benchmarks(adapter) -> {bench: score}
"""
from __future__ import annotations

import os
import re

from seams import config

_CACHE = {}


def _load(adapter: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    cfg = config.load()
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], quantization_config=bnb, dtype=torch.bfloat16,
        device_map={"": 0})
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model


def _gen(tok, model, prompt: str, max_new: int) -> str:
    import torch
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


def _gsm8k(tok, model, limit: int) -> float:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=f"test[:{limit}]")
    ok = 0
    for ex in ds:
        gold = ex["answer"].split("####")[-1].strip().replace(",", "")
        out = _gen(tok, model, ex["question"] +
                   "\nGive only the final numeric answer.", 256)
        nums = re.findall(r"-?\d+\.?\d*", out.replace(",", ""))
        if nums and nums[-1] == gold:
            ok += 1
    return round(ok / len(ds), 4)


def _arc(tok, model, limit: int) -> float:
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split=f"test[:{limit}]")
    ok = 0
    for ex in ds:
        labels = [str(l).upper() for l in ex["choices"]["label"]]
        texts = ex["choices"]["text"]
        opts = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
        out = _gen(tok, model, f"{ex['question']}\n{opts}\nAnswer with one option.", 8)
        pred = next((c for c in out.upper() if c in labels), None)
        if pred is not None and pred == str(ex["answerKey"]).upper():
            ok += 1
    return round(ok / len(ds), 4)

def eval_on_benchmarks(adapter: str | None) -> dict:
    cfg = config.load()
    limit = int(os.getenv("EVAL_LIMIT", cfg["eval"]["limit"]))
    benches = list(cfg["benchmarks"].keys())
    tag = adapter or "base"
    print(f"[eval] {tag}: {benches} (limit {limit}/bench)")
    tok, model = _load(adapter)
    fns = {"gsm8k": _gsm8k, "arc": _arc}
    scores = {}
    for b in benches:
        if b not in fns:
            print(f"[eval] no runner for '{b}', skipping"); continue
        try:
            scores[b] = fns[b](tok, model, limit)
        except Exception as e:  # keep the cycle alive if one bench can't load
            print(f"[eval] '{b}' failed: {repr(e)[:100]}; skipping")
    del model
    import torch, gc
    gc.collect(); torch.cuda.empty_cache()
    print(f"[eval] {tag} scores: {scores}")
    return scores
