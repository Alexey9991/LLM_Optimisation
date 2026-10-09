"""Transformer Layer Analysis and Optimization Toolkit (v2).

This module provides utilities for analyzing, transforming, and evaluating transformer
model layers. Includes functionality for distance metrics, model truncation, activation
analysis, and evaluation.

v2 changes (backward compatible, same public interface):
- truncate_model: also trims config.layer_types (Qwen3 etc.) so save_pretrained does not
  need the notebook-side monkeypatch, and re-indexes self_attn.layer_idx so the in-memory
  truncated model stays usable with KV cache.
- angular_distance: eps-guarded normalization (no NaN on zero vectors).
- NEW compute_block_distances_all_tokens: masked all-token block distance (much lower
  variance than the classic last-token-only metric). The classic functions are unchanged.
- select_non_overlapping_blocks: same selection convention as v1 —
  distances index i  <->  block (start, end) = (i+1, i+1+layers_to_skip),
  layers removed = range(start, end) (0-indexed), transform lives in layer start-1.
"""

import json
import os
import random
from typing import Dict, List, Optional, Tuple

import datasets
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

# Optional heavy deps: keep the module importable in envs without them.
try:
    from lm_eval import evaluator
except Exception:  # pragma: no cover
    evaluator = None
try:
    from torchmin import minimize
except Exception:  # pragma: no cover
    minimize = None
try:
    from termcolor import colored
except Exception:  # pragma: no cover
    def colored(s, *a, **k):
        return s


# --------------------------
# Seeding Function
# --------------------------
def seed_all(seed: int = 42):
    """Seed all major sources of randomness for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Seeded with seed: {seed}")


# --------------------------
# Distance Metric Functions
# --------------------------
def angular_distance(x_l: torch.Tensor, x_l_plus_n: torch.Tensor,
                     eps: float = 1e-8) -> torch.Tensor:
    """Compute angular distance between normalized layer outputs. Shape-preserving
    over all but the last dim: returns acos(cos_sim)/pi in [0, 1]."""
    x_l_norm = x_l / (torch.norm(x_l, dim=-1, keepdim=True) + eps)
    x_l_plus_n_norm = x_l_plus_n / (torch.norm(x_l_plus_n, dim=-1, keepdim=True) + eps)
    cosine_sim = (x_l_norm * x_l_plus_n_norm).sum(-1).clamp(min=-1, max=1)
    return torch.acos(cosine_sim) / torch.pi


# --------------------------
# Model Manipulation
# --------------------------
def get_decoder_layers(model: nn.Module):
    """Return (container_module, attribute_name) of the decoder layer list for
    standard (model.model.layers) and falcon (model.transformer.h) architectures."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model, "layers"
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer, "h"
    raise AttributeError("Could not locate decoder layers on this model.")


def truncate_model(model: nn.Module, start_layer: int, end_layer: int) -> nn.Module:
    """Truncate model by removing layers [start_layer, end_layer) (0-indexed).

    Also keeps config.layer_types (if present) and per-layer layer_idx attributes
    consistent, so the truncated model can be saved and used directly.
    """
    n_removed = end_layer - start_layer
    container, attr = get_decoder_layers(model)
    old_layers = getattr(container, attr)

    kept = [layer for idx, layer in enumerate(old_layers)
            if idx < start_layer or idx >= end_layer]
    setattr(container, attr, nn.ModuleList(kept))

    model.config.num_hidden_layers -= n_removed

    # Qwen3/gemma-style configs carry a per-layer list; trim it to match.
    lt = getattr(model.config, "layer_types", None)
    if lt is not None and len(lt) != model.config.num_hidden_layers:
        model.config.layer_types = [t for idx, t in enumerate(lt)
                                    if idx < start_layer or idx >= end_layer]

    # Re-index layer_idx so KV-cache indexing stays valid for in-memory use.
    for new_idx, layer in enumerate(getattr(container, attr)):
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = new_idx
        if hasattr(layer, "layer_idx"):
            layer.layer_idx = new_idx

    return model


# --------------------------
# Data Loading Utilities
# --------------------------
def _load_custom_json_dataset(file_path, text_column):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return datasets.Dataset.from_list(data)


def get_calib_dataloader(
    dataset: str,
    dataset_subset: str,
    dataset_column: str,
    dataset_size: Optional[int],
    batch_size: int,
    tokenizer: PreTrainedTokenizerBase
) -> DataLoader:
    """Load and prepare calibration dataset."""
    # dataset may be a path to a JSON file
    if os.path.isfile(dataset) and dataset.endswith('.json'):
        with open(dataset, 'r') as f:
            data = json.load(f)
        data = datasets.Dataset.from_list(data)
        if dataset_size:
            data = data.select(range(dataset_size))
        return DataLoader(data[dataset_column], batch_size=batch_size,
                          shuffle=False, drop_last=True)

    dataset_handlers = {
        'HuggingFaceFW/fineweb': lambda: datasets.load_dataset(dataset, name='sample-10BT', split=dataset_subset),
        'allenai/c4': lambda: datasets.load_dataset(dataset, 'en', split=dataset_subset),
        'arcee-ai/sec-data-mini': lambda: datasets.load_dataset(dataset, split=dataset_subset),
        'wikitext': lambda: datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split=dataset_subset),
        'openai/gsm8k': lambda: _load_gsm8k_dataset(dataset_subset),
        'gsm8k': lambda: _load_gsm8k_dataset(dataset_subset),
        'Open-Orca/SlimOrca': lambda: _load_orca_dataset(dataset_size, tokenizer),
        'fineweb_and_orca': lambda: _load_mixed_dataset(dataset_size, dataset_subset, tokenizer),
        'gsm8k_train.json': lambda: _load_custom_json_dataset(dataset, dataset_column),
    }

    if dataset not in dataset_handlers:
        raise ValueError(f"Dataset {dataset} not implemented")

    data = dataset_handlers[dataset]()
    if dataset_size:
        data = data.select(range(dataset_size))

    return DataLoader(data[dataset_column], batch_size=batch_size,
                      shuffle=False, drop_last=True)


def _load_gsm8k_dataset(subset: str) -> datasets.Dataset:
    """GSM8K formatted as 'Question: ...\\nAnswer: ...' text — the same format the
    Qwen3-8B notebook feeds through its patched dataloader."""
    split = subset if subset in ("train", "test") else "train"
    dd = datasets.load_dataset("openai/gsm8k", "main", split=split)
    return dd.map(lambda ex: {"text": f"Question: {ex['question']}\nAnswer: {ex['answer']}"})


def _load_orca_dataset(size: int, tokenizer: PreTrainedTokenizerBase) -> datasets.Dataset:
    """Helper to load and format Orca dataset."""
    dd = datasets.load_dataset("Open-Orca/SlimOrca", split="train").select(range(size))
    processed = []

    for item in dd:
        idx = 0 if item['conversations'][0]["from"] == "human" else 1
        dialog = [
            {"role": "user", "content": item['conversations'][idx]['value']},
            {"role": "assistant", "content": item['conversations'][idx + 1]['value']},
        ]
        text = tokenizer.apply_chat_template(dialog, tokenize=False)
        processed.append({"text": text})

    return datasets.Dataset.from_list(processed)


def _load_mixed_dataset(size: int, subset: str, tokenizer: PreTrainedTokenizerBase) -> datasets.Dataset:
    """Helper to load mixed Orca and FineWeb dataset."""
    orca_data = _load_orca_dataset(size // 2, tokenizer)
    fineweb = datasets.load_dataset(
        "HuggingFaceFW/fineweb",
        name='sample-10BT',
        split=subset
    ).select(range(size - size // 2))
    return datasets.concatenate_datasets([orca_data, fineweb])


# --------------------------
# Activation Analysis
# --------------------------
def get_last_non_padded_tokens(
    hidden_states: List[torch.Tensor],
    attention_mask: torch.Tensor
) -> List[torch.Tensor]:
    """Extract last non-padded tokens from each layer's hidden states."""
    return [
        torch.stack([
            layer[batch, mask.nonzero()[-1], :]
            for batch, mask in enumerate(attention_mask)
        ])
        for layer in hidden_states
    ]


# --------------------------
# Block Analysis Utilities
# --------------------------
def compute_block_distances(
    hidden_states: List[torch.Tensor],
    layers_to_skip: int,
    distance_fn=angular_distance
) -> List[float]:
    """Classic metric: distances between layer blocks on (already-extracted) tokens."""
    return [
        distance_fn(hidden_states[l], hidden_states[l + layers_to_skip]).mean().item()
        for l in range(len(hidden_states) - layers_to_skip)
    ]


def compute_block_distances_all_tokens(
    hidden_states: List[torch.Tensor],
    attention_mask: torch.Tensor,
    layers_to_skip: int,
    distance_fn=angular_distance,
) -> List[float]:
    """All-token variant: mean distance over every NON-PADDED token (not just the last
    one). hidden_states: list of [B, L, H] per layer; attention_mask: [B, L].
    Much lower estimator variance than the last-token metric at identical cost."""
    mask = attention_mask.bool()
    n_valid = mask.sum().clamp(min=1)
    out = []
    for l in range(len(hidden_states) - layers_to_skip):
        d = distance_fn(hidden_states[l], hidden_states[l + layers_to_skip])  # [B, L]
        out.append((d * mask).sum().item() / n_valid.item())
    return out


# --------------------------
# Optimization Methods (kept from v1 for package compatibility)
# --------------------------
class LowerTriangularLinear(nn.Module):
    """Linear layer with lower triangular weight matrix."""
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        identity = torch.eye(min(input_size, output_size))
        self.weight = nn.Parameter(torch.randn(output_size, input_size))
        self.weight.data[:identity.size(0), :identity.size(1)] = identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ torch.tril(self.weight).t()


def optimizing_method(a1, a2, a3=None, solver: str = "cg") -> torch.Tensor:
    """Optimize transformation matrix using torchmin solver (unchanged from v1)."""
    if minimize is None:
        raise ImportError("torchmin is required for optimizing_method")

    class ActivationDataset(Dataset):
        def __init__(self, a1, a2, a3):
            self.a1, self.a2, self.a3 = a1, a2, a3
        def __len__(self):
            return len(self.a1)
        def __getitem__(self, idx):
            attn = [-1]
            if self.a3 is not None:
                attn = self.a3[idx]
            return self.a1[idx], self.a2[idx], attn

    def cosine_loss(XA, Y):
        XA_norm = XA / XA.norm(dim=1, keepdim=True)
        Y_norm = Y / Y.norm(dim=1, keepdim=True)
        return 1 - (XA_norm * Y_norm).sum(dim=1).mean()

    dataset = ActivationDataset(a1, a2, a3)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    A = torch.eye(a1.shape[1], requires_grad=True, device="cuda")

    with tqdm(range(10), desc="Estimating Transformation") as pbar:
        for _ in pbar:
            for X, Y, Z in loader:
                def objective(A_tensor):
                    XA = X.float().to("cuda") @ A_tensor
                    if len(Z) != 1:
                        XA += Z.float().to("cuda")
                    return cosine_loss(XA, Y.float().to("cuda"))

                result = minimize(objective, A, method=solver, max_iter=1)
                pbar.set_postfix({'Loss': colored(f'{result.fun:.4f}', 'green')})
                A = result.x

    return A.to(torch.float64)


def adam_method(a1, a2, a3=None, loss: str = "cosine", diag: bool = False,
                two_vectors: bool = False, thri: bool = False) -> torch.Tensor:
    """Optimize transformation using Adam optimizer (unchanged from v1)."""
    class ActivationDataset(Dataset):
        def __init__(self, a1, a2, a3):
            self.a1, self.a2, self.a3 = a1, a2, a3
        def __len__(self):
            return len(self.a1)
        def __getitem__(self, idx):
            attn = [-1]
            if self.a3 is not None:
                attn = self.a3[idx]
            return self.a1[idx], self.a2[idx], attn

    if diag:
        transform = torch.ones(a1.shape[1], requires_grad=True, device="cuda")
        optimizer = torch.optim.Adam([transform], lr=1e-4)
    elif two_vectors:
        t1 = torch.ones((a1.shape[1], 1), requires_grad=True, device="cuda")
        t2 = torch.ones((a1.shape[1], 1), requires_grad=True, device="cuda")
        optimizer = torch.optim.Adam([t1, t2], lr=1e-4)
    else:
        model = LowerTriangularLinear(a1.shape[1], a1.shape[1]).to("cuda") if thri \
            else nn.Linear(a1.shape[1], a1.shape[1], bias=False).to("cuda")
        if not thri:
            model.weight.data.copy_(torch.eye(a1.shape[1]))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    def cosine_loss(XA, Y):
        XA_norm = XA / XA.norm(dim=1, keepdim=True)
        Y_norm = Y / Y.norm(dim=1, keepdim=True)
        return 1 - (XA_norm * Y_norm).sum(dim=1).mean()

    loss_fn = {
        "cosine": cosine_loss,
        "mse": nn.MSELoss(reduction='mean'),
        "elasticnet": lambda XA, Y: nn.MSELoss(reduction='mean')(XA, Y) +
                                    0.09 * torch.norm(XA, p=1) +
                                    0.045 * torch.norm(XA, p=2) ** 2
    }[loss]

    dataset = ActivationDataset(a1, a2, a3)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)

    with tqdm(range(10), desc="Optimizing Transformation") as pbar:
        for _ in pbar:
            for X, Y, Z in loader:
                optimizer.zero_grad()

                if diag:
                    XA = X.float().to("cuda") @ torch.diag(transform)
                elif two_vectors:
                    XA = X.float().to("cuda") @ (t1 @ t2.T)
                else:
                    XA = model(X.float().to("cuda"))
                if len(Z) != 1:
                    XA += Z.float().to("cuda")
                loss_val = loss_fn(XA, Y.float().to("cuda"))
                loss_val.backward()
                optimizer.step()

                pbar.set_postfix({f'{loss} Loss': colored(f'{loss_val.item():.4f}', 'green')})

    if diag:
        return torch.diag(transform).to(torch.float64)
    elif two_vectors:
        return (t1 @ t2.T).to(torch.float64)
    return model.triangular_weight.T.to(torch.float64) if thri else model.weight.T.to(torch.float64)


# --------------------------
# Evaluation Utilities (unchanged from v1)
# --------------------------
def eval_model(model_path: str) -> Dict:
    """Evaluate model on standard benchmark tasks."""
    if evaluator is None:
        raise ImportError("lm_eval is required for eval_model")
    result_path = f'benchmark_results/{os.path.basename(model_path)}.json'
    if os.path.exists(result_path):
        with open(result_path) as f:
            return json.load(f)

    wino_res = evaluator.simple_evaluate(
        model='hf',
        tasks=['winogrande'],
        model_args=f"pretrained={model_path},dtype=bfloat16,device=auto,parallelize=False,device_map=auto",
        num_fewshot=5
    )['results']

    other_res = evaluator.simple_evaluate(
        model='hf',
        tasks=['boolq', 'race', 'openbookqa', 'piqa', 'sciq', 'lambada_openai'],
        model_args=f"pretrained={model_path},dtype=bfloat16,device=auto,parallelize=False,device_map=auto",
        num_fewshot=0
    )['results']

    results = {**other_res, **wino_res}
    os.makedirs('benchmark_results', exist_ok=True)

    with open(result_path, 'w') as f:
        json.dump(results, f, indent=4)

    return results


def eval_model_specific(model_path: str, tasks: Dict) -> Dict:
    """Evaluate model on specified tasks with custom fewshot settings."""
    if evaluator is None:
        raise ImportError("lm_eval is required for eval_model_specific")
    result_path = f'benchmark_results/{os.path.basename(model_path)}.json'
    if os.path.exists(result_path):
        with open(result_path) as f:
            return json.load(f)

    results = {}
    for task, config in tasks.items():
        res = evaluator.simple_evaluate(
            model='hf',
            tasks=[task],
            model_args=f"pretrained={model_path},dtype=bfloat16,device=auto",
            num_fewshot=config["fewshots"]
        )['results']
        results.update(res)

    os.makedirs('benchmark_results', exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=4)

    return results


# --------------------------
# Block Selection Utilities
# --------------------------
def select_non_overlapping_blocks(
    average_distances: List[float],
    layers_to_skip: int,
    num_blocks: int = 4,
    merge_consecutive: bool = False
) -> List[Tuple[int, int]]:
    """Select non-overlapping layer blocks with the smallest distances.

    Convention (identical to v1, relied on by lstsq AND the notebook restore cells):
    distances index i  <->  block (i+1, i+1+layers_to_skip); the pruned model removes
    layers range(start, end) and the transform is fused into layer start-1.
    """
    blocks = [
        (i + 1, i + layers_to_skip + 1, avg)
        for i, avg in enumerate(average_distances)
    ]

    selected = []
    used_layers = set()

    for block in sorted(blocks, key=lambda x: x[2]):
        start, end, _ = block
        block_layers = set(range(start, end))

        if not block_layers & used_layers:
            selected.append(block)
            used_layers.update(block_layers)
            if len(selected) >= num_blocks:
                break

    if merge_consecutive and selected:
        selected.sort()
        merged = []
        current_start, current_end, _ = selected[0]

        for start, end, _ in selected[1:]:
            if start == current_end:
                current_end = end
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start, end

        merged.append((current_start, current_end))
        print(f"List of layers to prune {merged}")
        return merged
    selected = [(start, end) for start, end, _ in selected]
    print(f"List of layers to prune {selected}")
    return selected
