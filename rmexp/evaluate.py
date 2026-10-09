"""GSM8K evaluation with HF generate."""
import gc
import json
import time
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import COMPUTE_DTYPE
from .data import build_prompts, extract_final_answer, load_gsm8k, numbers_match


def quantization_config(precision: str):
    if precision == "bf16":
        return None
    if precision == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    if precision == "nf4":
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                  bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=COMPUTE_DTYPE)
    raise ValueError(precision)


def is_quantized(precision: str) -> bool:
    return precision != "bf16"


def load_for_generation(model_path: str, adapter_path: Optional[str] = None, precision: str = "nf4"):
    tokenizer = AutoTokenizer.from_pretrained(adapter_path or model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=COMPUTE_DTYPE, device_map={"": 0},
        quantization_config=quantization_config(precision))
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)
        if not is_quantized(precision):
            model = model.merge_and_unload()

    model.eval()
    model.config.use_cache = True
    model.generation_config.do_sample = False
    for attribute in ("temperature", "top_p", "top_k", "typical_p"):
        if hasattr(model.generation_config, attribute):
            setattr(model.generation_config, attribute, None)
    return model, tokenizer


def evaluate_gsm8k(model, tokenizer, batch_size: int, max_new_tokens: int,
                   limit: Optional[int] = None, dataset_name: str = "gsm8k") -> dict:
    if dataset_name == "gsm8k":
        dataset = load_gsm8k("test", limit)
        prompts = build_prompts(dataset)
        gold = [extract_final_answer(answer) for answer in dataset["answer"]]
    else:
        raise ValueError(f"unknown eval_dataset: {dataset_name!r}")

    prompt_lengths = [len(tokenizer(p)["input_ids"]) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: prompt_lengths[i])

    predictions = [None] * len(prompts)
    generations = [None] * len(prompts)
    started = time.perf_counter()

    for start in tqdm(range(0, len(order), batch_size), desc=f"Evaluating {dataset_name}"):
        indices = order[start:start + batch_size]
        encoded = tokenizer([prompts[i] for i in indices], return_tensors="pt", padding=True)
        encoded = {k: v.to(model.device) for k, v in encoded.items()}

        with torch.inference_mode():
            outputs = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)

        completions = tokenizer.batch_decode(outputs[:, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
        for index, completion in zip(indices, completions):
            generations[index] = completion
            predictions[index] = extract_final_answer(completion)

    correct = sum(numbers_match(p, g) for p, g in zip(predictions, gold))
    elapsed = time.perf_counter() - started

    return {
        "total": len(prompts),
        "correct": correct,
        "exact_match": correct / len(prompts),
        "seconds": round(elapsed, 1),
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "predictions": [
            {"id": i, "predicted": predictions[i], "gold": gold[i],
             "correct": numbers_match(predictions[i], gold[i]), "generation": generations[i]}
            for i in range(len(prompts))
        ],
    }


def result_path(results_dir, stage: str, precision: str, suffix_tag: str = "") -> Path:
    return Path(results_dir) / f"{stage}{suffix_tag}_{precision}.json"


def run_evaluation(config, stage: str, model_path: str, adapter_path: Optional[str] = None,
                   suffix_tag: str = "", results_dir=None) -> dict:
    results_dir = Path(results_dir or config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_for_generation(model_path, adapter_path, config.precision)

    result = evaluate_gsm8k(model, tokenizer, config.eval_batch_size, config.eval_max_new_tokens,
                            config.eval_limit, config.eval_dataset)
    result["stage"] = stage
    result["model_path"] = str(model_path)
    result["adapter_path"] = str(adapter_path) if adapter_path else None
    result["layers"] = model.config.num_hidden_layers
    result["precision"] = config.precision
    result["eval_dataset"] = config.eval_dataset
    result["hw"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    result["extra"] = {"mode": config.precision, "limit": config.eval_limit, "session": "hf_generate",
                       "batch_size": config.eval_batch_size, "max_new_tokens": config.eval_max_new_tokens}

    output_path = result_path(results_dir, stage, config.precision, suffix_tag)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"{stage}: EM {result['exact_match'] * 100:.2f}% ({result['correct']}/{result['total']}) "
          f"on {result['layers']} layers in {result['seconds'] / 60:.1f} min -> {output_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result
