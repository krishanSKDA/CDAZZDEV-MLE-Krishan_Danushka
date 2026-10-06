import gc
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

IGNORE_INDEX = -100
SEQ_LEN_MULTIPLE = 64


@dataclass
class TrainConfig:
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 1024
    weight_decay: float = 0.0
    max_grad_norm: float = 0.3
    optim: str = "paged_adamw_8bit"
    seed: int = 42
    logging_steps: int = 5


JUSTIFICATIONS = {
    "base_model": "Qwen2.5-3B-Instruct: strong instruction following and JSON output for its size, fits a T4 in 4-bit "
                  "with room for activations, and is a different family from the Llama teachers (no teacher=student leakage).",
    "lora_r": "16: the task is a narrow skill (schema + 12-way category and severity judgement), not new knowledge. r=16 on all "
              "linear layers gives ~30M trainable params (~1% of 3B); r=64 would quadruple that on only ~250 examples, "
              "raising over-fitting risk, while r=8 leaves less headroom for the multi-label decisions.",
    "lora_alpha": "32 (= 2r): effective update scale alpha/r = 2, the common QLoRA setting; keeps the LR meaningful if r changes.",
    "lora_dropout": "0.05: light regularisation for a small dataset without slowing convergence over only 3 epochs.",
    "target_modules": "All attention and MLP projections: the QLoRA paper shows adapting every linear layer is needed to match "
                      "full fine-tuning; attention-only adapters learn output format more slowly.",
    "learning_rate": "2e-4: QLoRA paper value for models <= 7B; LoRA weights start at zero so a higher LR than full FT is safe.",
    "lr_scheduler_type": "cosine: decays smoothly to ~0 by the last epoch, which stabilises late-epoch validation loss.",
    "warmup_ratio": "0.05: a few warm-up steps stop the first large updates from damaging the paged 8-bit optimiser state.",
    "num_train_epochs": "3: ~240 examples need multiple passes; validation loss is checked every epoch and the best "
                        "checkpoint is kept, so a late over-fit cannot leak into the saved model.",
    "per_device_train_batch_size": "2: largest that fits a T4 (15 GB) at this sequence length with gradient checkpointing.",
    "gradient_accumulation_steps": "4: effective batch 8 -> ~30 optimiser steps per epoch; larger batches leave too few updates.",
    "max_seq_length": "Set from the data: the longest tokenised example rounded up to a multiple of 64, so nothing is truncated "
                      "(truncation would cut the JSON answer the loss is computed on).",
    "weight_decay": "0.0: LoRA's low rank already constrains capacity; decay on zero-initialised adapters mostly slows learning.",
    "max_grad_norm": "0.3: QLoRA paper value; clips rare spikes from long examples under fp16.",
    "optim": "paged_adamw_8bit: 8-bit optimiser states with paging avoid OOM spikes on the 15 GB T4.",
    "seed": "42: fixed for reproducible shuffling and LoRA initialisation.",
    "logging_steps": "5: enough points to see the train-loss curve within each ~30-step epoch.",
    "precision": "fp16 compute (T4 has no bf16); 4-bit NF4 weights with double quantisation (saves ~0.4 bits/param).",
    "loss_masking": "Loss only on assistant tokens: system prompt and passage are labelled -100 so the model learns to extract, "
                    "not to reproduce inputs.",
}


def hyperparameter_table(cfg: TrainConfig) -> pd.DataFrame:
    values = asdict(cfg)
    rows = [{"parameter": k, "value": values.get(k, "-"), "justification": v} for k, v in JUSTIFICATIONS.items()]
    return pd.DataFrame(rows)


def tokenize_example(tokenizer, messages: list[dict], max_len: int | None = None) -> dict:
    prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    if not full_text.startswith(prompt_text):
        raise ValueError("chat template prompt is not a prefix of the full conversation")
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("prompt tokens differ at the assistant boundary")
    labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids):]
    if max_len is not None and len(full_ids) > max_len:
        raise ValueError(f"example of {len(full_ids)} tokens exceeds max_len {max_len}")
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


def token_length_stats(tokenizer, records: list[dict]) -> dict:
    lengths = np.array([len(tokenize_example(tokenizer, r["messages"])["input_ids"]) for r in records])
    return {"min": int(lengths.min()), "p50": int(np.percentile(lengths, 50)), "p95": int(np.percentile(lengths, 95)),
            "max": int(lengths.max()), "suggested_max_seq_length": int(math.ceil(lengths.max() / SEQ_LEN_MULTIPLE) * SEQ_LEN_MULTIPLE)}


class PadCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        import torch

        width = max(len(b["input_ids"]) for b in batch)

        def pad(key, value):
            return torch.tensor([b[key] + [value] * (width - len(b[key])) for b in batch])

        return {"input_ids": pad("input_ids", self.pad_token_id), "attention_mask": pad("attention_mask", 0),
                "labels": pad("labels", IGNORE_INDEX)}


class ListDataset:
    def __init__(self, items: list[dict]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def load_tokenizer(cfg: TrainConfig):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_qlora_model(cfg: TrainConfig):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, quantization_config=bnb, device_map={"": 0},
                                                 dtype=torch.float16, attn_implementation="sdpa")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                      target_modules=cfg.target_modules, bias="none", task_type="CAUSAL_LM")
    return get_peft_model(model, lora)


def build_trainer(cfg: TrainConfig, model, tokenizer, train_records: list[dict], val_records: list[dict],
                  output_dir: str, report_to: str = "none"):
    from transformers import Trainer, TrainingArguments

    train_ds = ListDataset([tokenize_example(tokenizer, r["messages"], cfg.max_seq_length) for r in train_records])
    val_ds = ListDataset([tokenize_example(tokenizer, r["messages"], cfg.max_seq_length) for r in val_records])
    # integer warmup steps: transformers v5 dropped `warmup_ratio`, and an int means the same thing in v4 and v5
    steps_per_epoch = math.ceil(len(train_ds) / (cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps))
    warmup_steps = max(1, math.ceil(cfg.warmup_ratio * steps_per_epoch * cfg.num_train_epochs))
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_steps=warmup_steps,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        optim=cfg.optim,
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=cfg.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        report_to=report_to,
        seed=cfg.seed,
    )
    return Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                   data_collator=PadCollator(tokenizer.pad_token_id))


def epoch_loss_table(log_history: list[dict]) -> pd.DataFrame:
    """Mean training loss per epoch alongside the end-of-epoch validation loss."""
    train = pd.DataFrame([h for h in log_history if "loss" in h and "eval_loss" not in h])
    evals = pd.DataFrame([h for h in log_history if "eval_loss" in h])
    if train.empty or evals.empty:
        return pd.DataFrame()
    train["epoch_idx"] = np.ceil(train["epoch"].clip(lower=1e-9)).astype(int)
    evals["epoch_idx"] = evals["epoch"].round().astype(int)
    table = train.groupby("epoch_idx")["loss"].mean().rename("train_loss").to_frame().join(
        evals.set_index("epoch_idx")["eval_loss"].rename("val_loss"))
    table["val_loss_delta"] = table["val_loss"].diff()
    return table.reset_index().rename(columns={"epoch_idx": "epoch"})


def free_gpu() -> None:
    """Call after `del model, trainer` in the caller so the references are actually released."""
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def merge_and_save(cfg: TrainConfig, adapter_dir: str, merged_dir: str, tokenizer) -> Path:
    """Reload the base in fp16 (not 4-bit) so merged weights are full precision, then fold the adapters in."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(cfg.base_model, dtype=torch.float16, device_map={"": 0})
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    return Path(merged_dir)


def push_to_hub(merged_dir: str, repo_id: str, token: str, model_card: str, private: bool = False) -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, private=private, exist_ok=True)
    (Path(merged_dir) / "README.md").write_text(model_card, encoding="utf-8")
    api.upload_folder(folder_path=merged_dir, repo_id=repo_id, commit_message="Upload merged QLoRA fine-tuned model")
    return f"https://huggingface.co/{repo_id}"
