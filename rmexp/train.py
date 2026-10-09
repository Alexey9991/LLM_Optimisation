"""LoRA training (r8 on the MLP projections, one epoch) with HF Trainer. The same function trains the SFT control
on the full model and heals the pruned model; only the base model path differs."""
import gc
import json
from collections import Counter
from pathlib import Path
from typing import Optional

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer,
                          TrainingArguments)

from .config import COMPUTE_DTYPE
from .data import build_training_texts, load_gsm8k
from .evaluate import is_quantized, quantization_config

IGNORE_INDEX = -100


class CausalDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return {"input_ids": self.input_ids[index],
                "attention_mask": self.attention_mask[index],
                "labels": self.labels[index]}


def _gsm8k_dataset(tokenizer, max_length: int, limit=None):
    dataset = load_gsm8k("train", limit)
    texts = build_training_texts(dataset, tokenizer.eos_token)
    encodings = tokenizer(texts, truncation=True, max_length=max_length, padding=False)
    return CausalDataset(encodings["input_ids"], encodings["attention_mask"],
                         [list(ids) for ids in encodings["input_ids"]])


class LoggedTrainer(Trainer):
    """Trainer that leaves log_history.json and train_meta.json next to the adapter."""

    def train(self, *args, **kwargs):
        out = super().train(*args, **kwargs)
        d = Path(self.args.output_dir).parent
        d.mkdir(parents=True, exist_ok=True)
        json.dump(self.state.log_history, open(d / "log_history.json", "w"), indent=1)
        json.dump({"train_loss": out.training_loss, "steps": self.state.global_step,
                   "runtime_min": round(out.metrics.get("train_runtime", 0) / 60, 1),
                   "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]},
                  open(d / "train_meta.json", "w"), indent=2)
        print("training log ->", d / "log_history.json")
        return out


def train_lora(config, model_path: str, adapter_dir: Path, epochs: Optional[float] = None,
               lm_head_on_second_gpu: Optional[bool] = None) -> Path:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train_dataset = _gsm8k_dataset(tokenizer, config.train_max_length, config.train_limit)

    if lm_head_on_second_gpu is None:
        lm_head_on_second_gpu = config.train_lm_head_on_second_gpu
    device_map = ({"model": 0, "lm_head": 1} if lm_head_on_second_gpu and torch.cuda.device_count() > 1
                  else {"": 0})
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=COMPUTE_DTYPE, device_map=device_map,
        quantization_config=quantization_config(config.precision))
    dmap = getattr(model, "hf_device_map", None) or {"": 0}      # not set when everything is on one device
    print("device map:", dict(Counter(str(d) for d in dmap.values())))
    model.config.use_cache = False
    if is_quantized(config.precision):
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.train_gradient_checkpointing)

    model = get_peft_model(model, LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.lora_target_modules,
    ))
    model.print_trainable_parameters()

    adapter_dir = Path(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Two GPUs are visible (prune uses both) but training runs on one. Without these flags the Trainer wraps
    # the 4-bit model in nn.DataParallel -> illegal memory access.
    model.is_parallelizable = True
    model.model_parallel = True
    targs = TrainingArguments(
        output_dir=str(adapter_dir / "checkpoints"),
        per_device_train_batch_size=config.train_batch_size,
        gradient_accumulation_steps=config.train_grad_accum,
        num_train_epochs=epochs if epochs is not None else config.train_epochs,
        learning_rate=config.train_learning_rate,
        warmup_steps=config.train_warmup_steps,
        lr_scheduler_type="linear",
        weight_decay=0.01,
        optim=config.train_optim,
        gradient_checkpointing=config.train_gradient_checkpointing,
        bf16=COMPUTE_DTYPE == torch.bfloat16,
        fp16=COMPUTE_DTYPE == torch.float16,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=config.seed,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    _ = targs.device
    targs._n_gpu = 1                       # one card, no DataParallel
    assert targs.n_gpu == 1

    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)
    trainer = LoggedTrainer(
        model=model,
        args=targs,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True,
                                             label_pad_token_id=IGNORE_INDEX),
    )

    trainer.train()
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print("peak GiB per card:", {i: round(torch.cuda.max_memory_allocated(i) / 2**30, 2)
                                 for i in range(torch.cuda.device_count())})

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return adapter_dir
