"""Runs ON the RunPod pod: fast LoRA with Unsloth on the agent's curated traces,
then pushes the adapter to the HF Hub so the device can pull it back.

The pod image installs: unsloth, trl, peft, datasets, huggingface_hub.
Env: HF_TOKEN (push), dataset pulled from the HF hub.

  python train_unsloth.py --base Qwen/Qwen3.5-4B \
      --dataset <user>/hermes-self-improve-lora-data \
      --push <user>/hermes-self-improve-lora --epochs 2 --lora_r 16
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--dataset", required=True)   # HF dataset repo (messages JSONL)
    ap.add_argument("--push", required=True)       # HF adapter repo target
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--max_seq_len", type=int, default=2048)
    a = ap.parse_args()

    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig

    model, tok = FastLanguageModel.from_pretrained(
        a.base, max_seq_length=a.max_seq_len, load_in_4bit=True, dtype=None)
    model = FastLanguageModel.get_peft_model(
        model, r=a.lora_r, lora_alpha=a.lora_r * 2, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])

    ds = load_dataset(a.dataset, split="train")

    def fmt(ex):
        return {"text": tok.apply_chat_template(ex["messages"], tokenize=False)}
    ds = ds.map(fmt)

    trainer = SFTTrainer(
        model=model, tokenizer=tok, train_dataset=ds, dataset_text_field="text",
        args=SFTConfig(per_device_train_batch_size=2, gradient_accumulation_steps=4,
                       num_train_epochs=a.epochs, learning_rate=2e-4, bf16=True,
                       logging_steps=5, output_dir="out", report_to="none"))
    trainer.train()

    model.push_to_hub(a.push, token=True, private=True)
    tok.push_to_hub(a.push, token=True, private=True)
    print(f"pushed adapter -> {a.push}")


if __name__ == "__main__":
    main()
